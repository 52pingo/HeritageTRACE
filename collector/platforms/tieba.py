"""Baidu Tieba collector.

The PC search page (``/f/search/res``) and the PC thread page (``/p/<tid>``)
are JavaScript SPA shells now — their HTML no longer contains any results, which
is why HTML scraping returned nothing at all. The mobile AJAX endpoints still
serve JSON/HTML fragments and are what this collector uses:

* search  ``/mo/q/search/thread?word=<kw>&pn=<page>&rn=<size>&is_ajax=1``
* floors  ``/mo/q/m?kz=<tid>&pn=<page>&is_ajax=1``  (JSON wrapping an HTML fragment)

A user Cookie (``TIEBA_COOKIE``) raises the rate limit and reduces the chance of
the "百度安全验证" interstitial.
"""

from __future__ import annotations

import html as html_module
import json
import re
from typing import Any, Iterator

from exceptions import ChallengeDetected, CollectorError
from models import Comment, ContentItem, DeepContent, Engagement
from utils import clean_html, extract_html_container, iso_from_timestamp, safe_int

from .base import BaseCollector, ProgressCallback

# Baidu shows a bot-check page instead of search results under aggressive polling.
VERIFY_MARKERS = ("百度安全验证", "网络不给力", "请打开百度安全验证", "安全验证")

# Each main floor starts with <li tid=...>; nested sub-replies (<li pid=...>)
# stay inside their parent floor's slice.
FLOOR_RE = re.compile(r"<li\s+tid=.*?(?=<li\s+tid=|\Z)", re.S)
DATA_INFO_RE = re.compile(r"data-info='([^']*)'")
MAIN_CONTENT_MARKER = 'class="content"'
FLOOR_TIME_RE = re.compile(r'<span class="list_item_time[^"]*">(.*?)</span>', re.S)
SUBFLOOR_RE = re.compile(r'<span class="floor_content">(.*?)</span>', re.S)
FLOOR_REPLY_RE = re.compile(r'btn_reply"><span class="btn_icon">(\d*)</span>', re.S)
# 移动版在正文里插入了图片引导横幅，不是用户内容。
BANNER_NOISE = re.compile(r"看高清大图[,，]?打开贴吧\s*App?", re.I)


class TiebaCollector(BaseCollector):
    name = "tieba"
    search_url = "https://tieba.baidu.com/mo/q/search/thread"
    floor_url = "https://tieba.baidu.com/mo/q/m"

    def _credential_headers(self) -> dict[str, str]:
        headers = {"Referer": "https://tieba.baidu.com/"}
        if self.settings.tieba_cookie:
            headers["Cookie"] = self.settings.tieba_cookie
        return headers

    @staticmethod
    def _is_verification_page(text: str, status_code: int) -> bool:
        if status_code in {401, 403}:
            return True
        return any(marker in text for marker in VERIFY_MARKERS)

    def _request_json(self, url: str, params: dict[str, Any], stage: str) -> dict[str, Any]:
        response = self.request(
            "GET", url, params=params, check_challenge=False, allow_status=(401, 403)
        )
        text = response.text
        if self._is_verification_page(text, response.status_code):
            saved = self._debug_challenge(
                str(response.url), text, f"tieba verification (HTTP {response.status_code})"
            )
            raise ChallengeDetected(f"贴吧触发安全验证，诊断已保存至 {saved}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise CollectorError(f"贴吧 {stage} 返回了非 JSON 内容（接口可能已变更）") from exc
        if not isinstance(payload, dict) or payload.get("no") not in (0, None):
            raise CollectorError(f"贴吧 {stage} 返回错误: {payload.get('error') or payload}")
        return payload

    @staticmethod
    def _user_name(row: dict[str, Any]) -> str:
        user = row.get("user") or {}
        return str(user.get("user_name") or user.get("name_show") or user.get("name") or "")

    @staticmethod
    def _images(row: dict[str, Any]) -> list[str]:
        urls: list[str] = []
        for media in row.get("media") or []:
            if not isinstance(media, dict):
                continue
            for key in ("water_pic", "big_pic", "src_pic", "origin_pic"):
                if media.get(key):
                    urls.append(str(media[key]))
                    break
        return urls

    def _convert(self, row: dict[str, Any], keyword: str) -> ContentItem | None:
        tid = str(row.get("tid") or "")
        if not tid:
            return None
        title = clean_html(str(row.get("title") or ""))
        content = clean_html(str(row.get("content") or ""))
        if not title and not content:
            return None
        forum = str(row.get("forum_name") or (row.get("forum_info") or {}).get("forum_name") or "")
        comments = [
            clean_html(str(item.get("content") or ""))
            for item in (row.get("comment_list") or [])
            if isinstance(item, dict) and item.get("content")
        ]
        return ContentItem(
            id=tid,
            platform=self.name,
            keyword=keyword,
            content_type="thread",
            title=title or content[:80],
            summary=content[:1000],
            author=self._user_name(row),
            published_at=iso_from_timestamp(row.get("time") or row.get("create_time")),
            engagement=Engagement(
                likes=safe_int(row.get("like_num")),
                comments=safe_int(row.get("post_num")),
                shares=safe_int(row.get("share_num")),
                views=safe_int(row.get("play_count")),
            ),
            url=str(row.get("pb_url") or f"https://tieba.baidu.com/p/{tid}"),
            extra={
                "tid": tid,
                "forum": forum,
                "forum_id": row.get("forum_id"),
                "reply_num": safe_int(row.get("post_num")),
                "tags": [forum] if forum else [],
                "picture_urls": self._images(row),
                "preview_comments": comments[:5],
                "source": "tieba-mobile-search",
            },
        )

    def search(self, keyword: str, progress: ProgressCallback | None = None) -> list[ContentItem]:
        items: list[ContentItem] = []
        for page in range(self.config.pages):
            try:
                payload = self._request_json(
                    self.search_url,
                    {"word": keyword, "pn": page, "rn": self.config.page_size, "is_ajax": 1},
                    "搜索",
                )
            except ChallengeDetected:
                raise
            except Exception as exc:
                # A failure on page N must not discard pages 1..N-1.
                if items:
                    self.note_partial("search", exc)
                    break
                raise
            data = payload.get("data") or {}
            rows = data.get("post_list") or []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                item = self._convert(row, keyword)
                if item:
                    items.append(item)
            if progress:
                progress(self.name, page + 1, self.config.pages, f"{keyword}: {len(items)} 条")
            if not rows or not data.get("has_more"):
                break
        deduped = list({item.id: item for item in items}.values())
        if not deduped:
            self.note_partial(
                "search",
                f"贴吧搜索「{keyword}」没有返回任何帖子（该关键词可能确实没有内容）",
                blocking=False,
            )
        return deduped

    def _floors(self, tid: str) -> list[dict[str, Any]]:
        """Return the mobile floor list for one thread (main floors + 楼中楼)."""
        payload = self._request_json(self.floor_url, {"kz": tid, "pn": 0, "is_ajax": 1}, "楼层")
        fragment = str((payload.get("data") or {}).get("html") or "")
        floors: list[dict[str, Any]] = []
        for block in FLOOR_RE.findall(fragment):
            info: dict[str, Any] = {}
            info_match = DATA_INFO_RE.search(block)
            if info_match:
                try:
                    info = json.loads(html_module.unescape(info_match.group(1)))
                except ValueError:
                    info = {}
            # The post body is a nested <div class="content">, so it needs
            # balanced extraction rather than a non-greedy tag match.
            content = clean_html(extract_html_container(block, MAIN_CONTENT_MARKER))
            content = BANNER_NOISE.sub("", content).strip()
            if not content:
                continue
            time_match = FLOOR_TIME_RE.search(block)
            reply_match = FLOOR_REPLY_RE.search(block)
            sub_replies = [
                {
                    "author": "",
                    "content": clean_html(raw),
                }
                for raw in SUBFLOOR_RE.findall(block)
                if clean_html(raw)
            ]
            floors.append(
                {
                    "floor": str(info.get("floor_num") or ""),
                    "author": str(info.get("un") or ""),
                    "content": content,
                    "time": clean_html(time_match.group(1)) if time_match else "",
                    "replies": reply_match.group(1) if reply_match else "",
                    "sub_replies": sub_replies,
                }
            )
        return floors

    def deep_dive(self, item: ContentItem) -> DeepContent:
        errors: list[str] = []
        tid = str(item.extra.get("tid") or item.id)
        main_parts = [item.title, item.summary]
        comments: list[Comment] = []
        try:
            floors = self._floors(tid)
            if floors:
                # Floor 1 is the original post; the rest are the replies.
                main_parts = [floors[0]["content"] or item.title]
                for floor in floors[1:]:
                    comments.append(
                        Comment(
                            text=floor["content"],
                            likes=safe_int(floor["replies"]),
                            author=floor["author"],
                            published_at=floor["time"],
                        )
                    )
                    for sub in floor.get("sub_replies") or []:
                        comments.append(
                            Comment(text=sub["content"], likes=0, author=sub["author"], published_at=floor["time"])
                        )
                for sub in floors[0].get("sub_replies") or []:
                    comments.append(Comment(text=sub["content"], likes=0, author=sub["author"], published_at=""))
            else:
                errors.append("楼层解析为空（移动版页面结构可能已变更）")
        except ChallengeDetected:
            raise
        except Exception as exc:
            errors.append(f"floors: {exc}")

        full_content = "\n".join(part for part in main_parts if part).strip() or item.summary
        # The search response already carries a comment preview; use it when the
        # floor crawl produced nothing so L2 is not silently empty.
        if not comments:
            for text in item.extra.get("preview_comments") or []:
                comments.append(Comment(text=str(text), likes=0, author="", published_at=""))
        return DeepContent(base=item, full_content=full_content[:20_000], comments=comments, errors=errors)
