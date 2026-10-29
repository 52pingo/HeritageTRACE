"""Zhihu collector with two backends: signed API (default) and Playwright.

Zhihu closed guest search and signs its API with a JS-generated `x-zse-96`
header. By default this collector computes that signature in pure Python
(see ``zhihu_sign``) and calls the API directly -- no login needed; only a
`d_c0` visitor cookie is required, which is captured automatically on first
visit (or set via ``ZHIHU_DC0``).

A Playwright backend remains available for sites/flows the API cannot cover,
or as a fallback when the signature is rejected:

- ``ZHIHU_MODE=api``        signed API only (no browser).
- ``ZHIHU_MODE=playwright`` persistent browser profile (needs ``ZHIHU_PROFILE_DIR``).
- ``ZHIHU_MODE=auto``       try the API, fall back to Playwright (default).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any
from urllib.parse import quote, urlencode

import zhihu_sign
from exceptions import AuthenticationRequired, CollectorError
from models import Comment, ContentItem, DeepContent, Engagement
from utils import clean_html, iso_from_timestamp

from .base import BaseCollector, ProgressCallback

API_BASE = "https://www.zhihu.com"


def _extract_count(text: str, label: str) -> int:
    if not text:
        return 0
    match = re.search(r"([\d.]+[万kK]?)\s*" + re.escape(label), text)
    if not match:
        return 0
    raw = match.group(1)
    multiplier = 1
    if raw.endswith("万"):
        multiplier, raw = 10_000, raw[:-1]
    elif raw.lower().endswith("k"):
        multiplier, raw = 1_000, raw[:-1]
    try:
        return int(float(raw) * multiplier)
    except ValueError:
        return 0


class ZhihuCollector(BaseCollector):
    name = "zhihu"

    def __init__(self, settings: Any) -> None:
        super().__init__(settings)
        self._playwright: Any = None
        self._context: Any = None
        self._dc0: str | None = None

    # ------------------------------------------------------------------ #
    # Signed API mode
    # ------------------------------------------------------------------ #
    def _ensure_dc0(self) -> str:
        if self.settings.zhihu_dc0:
            return self.settings.zhihu_dc0
        # ZHIHU_COOKIE 里通常已含 d_c0,可直接从中提取,无需单独配置 ZHIHU_DC0
        if self.settings.zhihu_cookie:
            match = re.search(r"d_c0=([^;]+)", self.settings.zhihu_cookie)
            if match:
                return match.group(1).strip()
        if self._dc0:
            return self._dc0
        # A first visit to the homepage issues d_c0 to any visitor.
        self.request("GET", API_BASE + "/")
        value = self.client.cookies.get("d_c0")
        if not value:
            raise CollectorError(
                "知乎已不再向未登录访客自动下发 d_c0。请任选其一："
                "1) 浏览器登录 www.zhihu.com 后，F12 → Application → Cookies → www.zhihu.com，"
                "复制 d_c0 的值填入 .env 的 ZHIHU_DC0；"
                "2) 或配置 ZHIHU_PROFILE_DIR + ZHIHU_MODE=playwright，使用浏览器登录态采集"
            )
        self._dc0 = value
        self.log.info("已自动获取 d_c0 Cookie")
        return value

    def _cookie_header(self, d_c0: str) -> str:
        """构造 Cookie 头。

        知乎已要求登录态才返回搜索结果:只带访客 d_c0 时搜索接口返回
        401 ZERR_NOT_LOGIN(2026-10-01 实测)。配置了 ZHIHU_COOKIE 时优先使用
        完整 Cookie(含 z_c0),并确保其中含 d_c0(签名依赖它)。
        """
        cookie = self.settings.zhihu_cookie
        if not cookie:
            return f"d_c0={d_c0}"
        if "d_c0=" not in cookie:
            cookie = f"{cookie.rstrip('; ')}; d_c0={d_c0}"
        return cookie

    def _get_signed_json(self, path: str, d_c0: str, referer: str | None = None) -> dict[str, Any]:
        headers = {
            "User-Agent": self.random.choice(self._user_agents),
            "Referer": referer or "https://www.zhihu.com/",
            "Origin": "https://www.zhihu.com",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "x-requested-with": "fetch",
            "x-zse-93": zhihu_sign.ZSE93,
            "x-zse-96": zhihu_sign.compute_x_zse_96(path, d_c0),
            "x-api-version": "3.0.91",
            "Cookie": self._cookie_header(d_c0),
        }
        response = self.request("GET", API_BASE + path, headers=headers, allow_status=(401, 403))
        body = response.text
        content_type = response.headers.get("Content-Type", "").lower()
        try:
            value = response.json()
        except ValueError:
            value = None
        if isinstance(value, dict):
            error = value.get("error")
            if isinstance(error, dict) and (error.get("code") or error.get("message")):
                code = str(error.get("code") or "")
                message = str(error.get("message") or "")
                if code == "10003" or "请求参数异常" in message:
                    # 这是签名被拒，不是登录态问题；报成「登录墙」会把人引向错误的排查方向。
                    saved = self._debug_challenge(str(response.url), body, "x-zse-96 签名被拒 (10003)")
                    raise CollectorError(
                        "知乎拒绝了请求签名（x-zse-96，10003「请求参数异常，请升级客户端后重试」）。"
                        "这不是 Cookie 失效：请更新 zhihu_sign.py 中的 ZSE93/KEY/IV/ENCRYPT_CONF 常量，"
                        "或改用 ZHIHU_MODE=playwright 走浏览器签名。"
                        f"诊断：{saved}"
                    )
                if code in {"101", "100", "-2001"} or response.status_code in {401, 403}:
                    self._save_auth_screenshot_stub(referer or path)
                    raise AuthenticationRequired(
                        f"知乎 API 登录态无效（{code} {message}）；请更新 ZHIHU_COOKIE/ZHIHU_DC0"
                    )
                raise CollectorError(f"知乎 API 错误 {code}: {message}")
        if "html" in content_type or "signin" in body[:2000].lower() or "登录" in body[:2000]:
            self._save_auth_screenshot_stub(referer or path)
            raise AuthenticationRequired("知乎 API 返回登录墙；请确认 d_c0 有效，或配置 ZHIHU_PROFILE_DIR 改用 Playwright 模式")
        if value is None:
            raise CollectorError(
                f"知乎 API 返回非 JSON（签名可能已失效或需更新 ZHIHU_DC0），状态 {response.status_code}"
            )
        if not isinstance(value, dict):
            raise CollectorError("知乎 API 返回格式异常")
        return value

    def _save_auth_screenshot_stub(self, source: str) -> None:
        try:
            self.settings.debug_root.mkdir(parents=True, exist_ok=True)
            path = self.settings.debug_root / f"zhihu_auth_{int(time.time())}.json"
            path.write_text(
                json.dumps({"platform": "zhihu", "source": source, "reason": "API 登录墙"}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self.log.warning("知乎 API 触发登录墙，诊断已保存至 %s", path)
        except Exception:
            pass

    def _convert_api_result(self, result: Any, keyword: str) -> ContentItem | None:
        obj = result.get("object") if isinstance(result, dict) else None
        obj = obj if isinstance(obj, dict) else result
        if not isinstance(obj, dict) or not obj.get("id"):
            return None
        obj_url = obj.get("url") or ""
        if not obj_url:
            return None
        obj_type = obj.get("type") or "post"
        if obj_type not in {"answer", "question", "article", "pin", "zvideo"}:
            obj_type = "post"
        question = obj.get("question")
        title = (question or {}).get("title") if isinstance(question, dict) else None
        title = title or obj.get("title") or obj.get("headline") or ""
        excerpt = obj.get("excerpt") or clean_html(obj.get("content"))
        author = (obj.get("author") or {}).get("name") if isinstance(obj.get("author"), dict) else ""
        created = obj.get("created_time")
        return ContentItem(
            id=str(obj["id"]),
            platform="zhihu",
            keyword=keyword,
            content_type=obj_type,
            title=str(title)[:500],
            summary=str(excerpt)[:1000],
            author=str(author or ""),
            published_at=iso_from_timestamp(created) if created else "",
            engagement=Engagement(likes=int(obj.get("voteup_count") or 0), comments=int(obj.get("comment_count") or 0)),
            url=str(obj_url),
            extra={"zhihu_type": obj_type, "api": True},
        )

    def _search_api(self, keyword: str) -> list[ContentItem]:
        d_c0 = self._ensure_dc0()
        limit = min(self.config.page_size, 20)
        items: list[ContentItem] = []
        seen_ids: set[str] = set()
        page = 0
        while page < self.config.pages:
            params = {"t": "general", "q": keyword, "correction": "1", "offset": str(page * limit), "limit": str(limit)}
            path = "/api/v4/search_v3?" + urlencode(params)
            referer = f"https://www.zhihu.com/search?type=content&q={quote(keyword)}"
            try:
                data = self._get_signed_json(path, d_c0, referer=referer)
            except Exception as exc:
                # A 403 on page N must not discard pages 1..N-1.
                if items:
                    self.note_partial("search", exc)
                    break
                raise
            rows = data.get("data") or []
            if not rows:
                break
            new_count = 0
            for result in rows:
                item = self._convert_api_result(result, keyword)
                if item and item.id not in seen_ids:
                    seen_ids.add(item.id)
                    items.append(item)
                    new_count += 1
            if new_count == 0:
                break
            page += 1
        return items

    def _api_answer_content(self, aid: str, d_c0: str) -> str:
        path = f"/api/v4/answers/{aid}?include=content,excerpt,comment_count,voteup_count"
        data = self._get_signed_json(path, d_c0)
        return clean_html(data.get("content") or data.get("excerpt"))

    def _api_question_first_answer(self, qid: str, d_c0: str) -> tuple[str | None, str]:
        path = (
            f"/api/v4/questions/{qid}/answers"
            "?include=data[*].id,data[*].content,data[*].excerpt,data[*].author.name,data[*].voteup_count,data[*].comment_count"
            "&limit=1&offset=0&platform=desktop&sort_by=default"
        )
        data = self._get_signed_json(path, d_c0)
        rows = data.get("data") or []
        for row in rows:
            content = clean_html(row.get("content") or row.get("excerpt"))
            if content:
                return str(row.get("id")) if row.get("id") else None, content
        return None, ""

    def _api_comments(self, aid: str, d_c0: str) -> list[Comment]:
        path = f"/api/v4/comments/{aid}/root_comment?limit=20&offset=0&order_by=score"
        data = self._get_signed_json(path, d_c0)
        comments: list[Comment] = []
        for row in data.get("data") or []:
            text = clean_html(row.get("content"))
            if text:
                comments.append(Comment(text=text, likes=int(row.get("vote_count") or 0)))
        return comments

    def _api_article_content(self, article_id: str, d_c0: str) -> str:
        path = f"/api/v4/articles/{article_id}?include=content"
        data = self._get_signed_json(path, d_c0)
        return clean_html(data.get("content"))

    def _deep_api(self, item: ContentItem) -> DeepContent | None:
        url = item.url or ""
        d_c0 = self._ensure_dc0()
        answer_match = re.search(r"/answer/(\d+)", url)
        if answer_match:
            aid = answer_match.group(1)
            content = self._api_answer_content(aid, d_c0) or item.summary
            return DeepContent(base=item, full_content=content, comments=self._api_comments(aid, d_c0))
        question_match = re.search(r"/question/(\d+)", url)
        if question_match:
            aid, content = self._api_question_first_answer(question_match.group(1), d_c0)
            comments = self._api_comments(aid, d_c0) if aid else []
            return DeepContent(base=item, full_content=content or item.summary, comments=comments)
        article_match = re.search(r"/(?:p|article)/(\d+)", url)
        if article_match:
            content = self._api_article_content(article_match.group(1), d_c0)
            if content:
                return DeepContent(base=item, full_content=content, comments=[])
        return None

    # ------------------------------------------------------------------ #
    # Playwright backend
    # ------------------------------------------------------------------ #
    def _ensure_context(self) -> Any:
        if self._context is not None:
            return self._context
        if not self.settings.zhihu_profile_dir:
            raise AuthenticationRequired("知乎 Playwright 模式需要 ZHIHU_PROFILE_DIR（持久化浏览器配置目录）")
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise CollectorError(
                "知乎 Playwright 模式需要 playwright，请执行 pip install -r requirements-optional.txt && playwright install chromium"
            ) from exc
        self._playwright = sync_playwright().start()
        self._context = self._playwright.chromium.launch_persistent_context(
            user_data_dir=str(self.settings.zhihu_profile_dir),
            headless=self.settings.zhihu_headless,
            user_agent=self.random.choice(self._user_agents),
            locale="zh-CN",
            viewport={"width": 1280, "height": 800},
        )
        return self._context

    def _save_auth_screenshot(self, page: Any) -> None:
        try:
            self.settings.debug_root.mkdir(parents=True, exist_ok=True)
            path = self.settings.debug_root / f"{self.name}_auth_{int(time.time())}.png"
            page.screenshot(path=str(path))
            self.log.warning("知乎登录态缺失，诊断截图已保存至 %s", path)
        except Exception:
            pass

    def _ensure_logged_in(self, page: Any) -> None:
        lowered = (page.url or "").lower()
        if "signin" in lowered or "login" in lowered:
            self._save_auth_screenshot(page)
            raise AuthenticationRequired("知乎需要登录态；请先设置 ZHIHU_PROFILE_DIR，并用 ZHIHU_HEADLESS=false 跑一次，在打开的浏览器中手动登录")
        try:
            page.wait_for_selector(".LoginModal, .Modal-enter-active", timeout=2500)
            self._save_auth_screenshot(page)
            raise AuthenticationRequired("知乎需要登录态；请先设置 ZHIHU_PROFILE_DIR，并用 ZHIHU_HEADLESS=false 跑一次，在打开的浏览器中手动登录")
        except AuthenticationRequired:
            raise
        except Exception:
            pass

    @staticmethod
    def _convert_card(card: Any, keyword: str) -> ContentItem | None:
        link = card.query_selector(".ContentItem-title a")
        if not link:
            return None
        href = link.get_attribute("href") or ""
        if not href:
            return None
        if href.startswith("//"):
            href = "https:" + href
        elif not href.startswith("http"):
            href = "https://www.zhihu.com" + href
        title = link.inner_text().strip()
        excerpt = card.query_selector(".RichText")
        summary = excerpt.inner_text().strip() if excerpt else ""
        author = card.query_selector(".AuthorInfo-name, .UserLink-link")
        author_text = author.inner_text().strip() if author else ""
        meta = card.query_selector(".ContentItem-actions, .ContentItem-meta")
        meta_text = meta.inner_text() if meta else ""
        votes = _extract_count(meta_text, "赞同")
        comment_count = _extract_count(meta_text, "评论")
        matched = re.search(r"/(question|answer|article|pin|people|zvideo|p)/([^/?#]+)", href)
        item_id = matched.group(2) if matched else hashlib.sha256(href.encode("utf-8")).hexdigest()[:12]
        if "question" in href:
            content_type = "question"
        elif "pin" in href:
            content_type = "pin"
        elif any(marker in href for marker in ("article", "/p/", "zhuanlan")):
            content_type = "article"
        else:
            content_type = "post"
        return ContentItem(
            id=item_id,
            platform="zhihu",
            keyword=keyword,
            content_type=content_type,
            title=title,
            summary=summary[:1000],
            author=author_text,
            engagement=Engagement(likes=votes, comments=comment_count),
            url=href,
            extra={"zhihu_type": content_type},
        )

    def _search_playwright(self, keyword: str, progress: ProgressCallback | None = None) -> list[ContentItem]:
        page = None
        try:
            context = self._ensure_context()
            page = context.new_page()
            self._rate_limit()
            page.goto(
                f"https://www.zhihu.com/search?type=content&q={quote(keyword)}",
                wait_until="domcontentloaded",
                timeout=int(self.config.timeout * 1000),
            )
            self._ensure_logged_in(page)
            try:
                page.wait_for_selector(".SearchResult-Card", timeout=int(self.config.timeout * 1000))
            except Exception:
                self._ensure_logged_in(page)
                return []
            items: list[ContentItem] = []
            seen_ids: set[str] = set()
            for _ in range(max(1, self.config.pages)):
                for card in page.query_selector_all(".SearchResult-Card"):
                    item = self._convert_card(card, keyword)
                    if item and item.id not in seen_ids:
                        seen_ids.add(item.id)
                        items.append(item)
                if len(items) >= self.config.page_size:
                    break
                page.mouse.wheel(0, 4000)
                page.wait_for_timeout(1200)
            return items
        except (AuthenticationRequired, CollectorError):
            raise
        except Exception as exc:
            self.log.exception("知乎 Playwright 搜索失败 %s", keyword)
            raise CollectorError(f"知乎搜索失败: {exc}") from exc
        finally:
            if page is not None:
                try:
                    page.close()
                except Exception:
                    pass

    def _initial_data_answer(self, page: Any) -> str:
        initial = page.query_selector("#js-initialData")
        if not initial:
            return ""
        try:
            text = initial.text_content() or ""
            data = json.loads(text)
            answers = data.get("initialState", {}).get("entities", {}).get("answers", {})
            for answer in answers.values():
                content = clean_html(str(answer.get("content", "")))
                if content:
                    return content
        except Exception:
            pass
        return ""

    def _deep_playwright(self, item: ContentItem) -> DeepContent:
        page = None
        try:
            context = self._ensure_context()
            page = context.new_page()
            self._rate_limit()
            page.goto(
                item.url,
                wait_until="domcontentloaded",
                timeout=int(self.config.timeout * 1000),
            )
            self._ensure_logged_in(page)
            try:
                page.wait_for_selector(".AnswerItem", timeout=int(self.config.timeout * 1000))
            except Exception:
                page.wait_for_timeout(3000)
            full_content = ""
            comments: list[Comment] = []
            answer = page.query_selector(".AnswerItem")
            if answer:
                content_el = answer.query_selector(".RichText")
                if content_el:
                    full_content = content_el.inner_text().strip()
                for comment_el in answer.query_selector_all(".CommentItem")[:100]:
                    text_el = comment_el.query_selector(".CommentItem-content, .RichText")
                    text = text_el.inner_text().strip() if text_el else ""
                    if text:
                        comments.append(Comment(text=text))
            if not full_content:
                full_content = self._initial_data_answer(page) or item.summary
            return DeepContent(base=item, full_content=full_content, comments=comments)
        except (AuthenticationRequired, CollectorError):
            raise
        except Exception as exc:
            self.log.exception("知乎深挖失败 %s", item.id)
            return DeepContent(base=item, full_content=item.summary, errors=[f"deep: {exc}"])
        finally:
            if page is not None:
                try:
                    page.close()
                except Exception:
                    pass

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def search(self, keyword: str, progress: ProgressCallback | None = None) -> list[ContentItem]:
        mode = self.settings.zhihu_mode
        if mode in ("api", "auto"):
            try:
                items = self._search_api(keyword)
                if items or not (mode == "auto" and self.settings.zhihu_profile_dir):
                    if progress:
                        progress(self.name, 1, 1, f"{keyword}: {len(items)} 条")
                    return items
                self.log.warning("知乎签名 API 返回 0 条，按 auto 模式回退 Playwright")
            except (AuthenticationRequired, CollectorError) as exc:
                if not (mode == "auto" and self.settings.zhihu_profile_dir):
                    raise
                self.log.warning("知乎 API 签名模式失败（%s），回退 Playwright: %s", type(exc).__name__, exc)
        return self._search_playwright(keyword, progress)

    def deep_dive(self, item: ContentItem) -> DeepContent:
        mode = self.settings.zhihu_mode
        if mode in ("api", "auto"):
            try:
                deep = self._deep_api(item)
            except (AuthenticationRequired, CollectorError):
                if mode == "auto" and self.settings.zhihu_profile_dir:
                    return self._deep_playwright(item)
                raise
            if deep is not None:
                return deep
            if mode == "auto" and self.settings.zhihu_profile_dir:
                return self._deep_playwright(item)
            return DeepContent(base=item, full_content=item.summary, comments=[])
        return self._deep_playwright(item)

    def close(self) -> None:
        for close_target in (self._context, self._playwright):
            if close_target is None:
                continue
            try:
                close_target.close()
            except Exception:
                pass
        self._context = None
        self._playwright = None
        super().close()
