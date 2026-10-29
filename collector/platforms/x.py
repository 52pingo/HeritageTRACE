"""X (Twitter) collector.

Two back ends:

* ``X_BEARER_TOKEN`` configured → the official X API v2 paid endpoint.
* No token → keyword search through **Nitter**, the open-source alternative
  frontend, whose ``/search/rss`` feed is a plain Atom/RSS document that needs
  no account. This yields text, author and timestamp but no like/repost counts,
  and depends on volunteer-run instances staying up, so several are tried.
"""

from __future__ import annotations

from typing import Any
import re

from exceptions import AuthenticationRequired, CollectorError
from media_tools import ocr_image_url
from models import Comment, ContentItem, DeepContent, Engagement
from utils import clean_html, parse_iso, safe_int

from .base import BaseCollector, ProgressCallback

def _mentions(item: ContentItem, keyword: str) -> bool:
    """Does the item's own text actually contain the query?

    Nitter instances are supposed to answer a search with matching tweets, but
    some fall back to a generic timeline when their search backend fails. This
    guard keeps that unrelated content out of the dataset.
    """
    haystack = f"{item.title} {item.summary}".lower()
    tokens = [t for t in re.split(r"[\s#]+", keyword.lower()) if len(t) >= 2]
    if not tokens:
        return bool(haystack.strip())
    return any(token in haystack for token in tokens)


def _rfc822_to_iso(value: str) -> str:
    """RFC 822 (RSS pubDate) -> ISO 8601 for the normalised model."""
    from email.utils import parsedate_to_datetime

    try:
        return parsedate_to_datetime(value.strip()).isoformat()
    except (TypeError, ValueError, OverflowError):
        return ""


# 公共 Nitter 实例。志愿者维护，随时可能失效，按顺序尝试。
DEFAULT_NITTER_INSTANCES = (
    "https://twiiit.com",
    "https://nitter.tiekoetter.com",
    "https://nitter.pp.ua",
)


class XCollector(BaseCollector):
    name = "x"
    foreign = True
    endpoint = "https://api.x.com/2/tweets/search/recent"

    def __init__(self, settings: Any) -> None:
        super().__init__(settings)
        self.has_token = bool(settings.x_bearer_token)
        self.instances = settings.x_nitter_instances or list(DEFAULT_NITTER_INSTANCES)
        if not self.has_token:
            self.log.warning(
                "X 未配置 X_BEARER_TOKEN，改用 Nitter 公开 RSS 搜索："
                "可拿到正文/作者/时间，但没有点赞与转推数，且依赖公共实例可用性"
            )

    def _credential_headers(self) -> dict[str, str]:
        if not self.settings.x_bearer_token:
            return {}
        return {"Authorization": f"Bearer {self.settings.x_bearer_token}"}

    @staticmethod
    def _includes(payload: dict[str, Any]) -> tuple[dict[str, dict], dict[str, dict]]:
        includes = payload.get("includes") or {}
        users = {str(row.get("id")): row for row in includes.get("users") or []}
        media = {str(row.get("media_key")): row for row in includes.get("media") or []}
        return users, media

    def _params(self, query: str) -> dict[str, Any]:
        return {
            "query": query,
            "max_results": max(10, min(self.config.page_size, 100)),
            "tweet.fields": "id,text,author_id,created_at,public_metrics,attachments,conversation_id,lang,note_tweet",
            "user.fields": "id,name,username,public_metrics",
            "expansions": "author_id,attachments.media_keys",
            "media.fields": "media_key,type,url,preview_image_url",
        }

    def _convert(self, keyword: str, raw: dict[str, Any], users: dict[str, dict], media: dict[str, dict]) -> ContentItem:
        metrics = raw.get("public_metrics") or {}
        user = users.get(str(raw.get("author_id")), {})
        username = str(user.get("username", ""))
        note = (raw.get("note_tweet") or {}).get("text")
        text = str(note or raw.get("text") or "")
        media_rows = [media[key] for key in (raw.get("attachments") or {}).get("media_keys", []) if key in media]
        return ContentItem(
            id=str(raw.get("id", "")),
            platform=self.name,
            keyword=keyword,
            content_type="tweet",
            title="",
            summary=text[:280],
            author=username or str(user.get("name", "")),
            published_at=str(raw.get("created_at", "")),
            engagement=Engagement(
                likes=safe_int(metrics.get("like_count")),
                comments=safe_int(metrics.get("reply_count")),
                shares=safe_int(metrics.get("retweet_count")) + safe_int(metrics.get("quote_count")),
                views=safe_int(metrics.get("impression_count")),
            ),
            url=f"https://x.com/{username or 'i'}/status/{raw.get('id')}",
            extra={
                "author_id": raw.get("author_id"),
                "author_followers": safe_int((user.get("public_metrics") or {}).get("followers_count")),
                "conversation_id": raw.get("conversation_id") or raw.get("id"),
                "reply_count": safe_int(metrics.get("reply_count")),
                "quote_count": safe_int(metrics.get("quote_count")),
                "media": media_rows,
                "full_text": text,
                "lang": raw.get("lang"),
                "hashtags": re.findall(r"#([^\s#]+)", text),
            },
        )

    def _search_nitter(self, keyword: str, progress: ProgressCallback | None = None) -> list[ContentItem]:
        """Keyword search via public Nitter instances (no account needed)."""
        last_error = ""
        reachable = 0
        unrelated = 0
        for instance in self.instances:
            try:
                response = self.request(
                    "GET",
                    f"{instance.rstrip('/')}/search/rss",
                    params={"f": "tweets", "q": keyword},
                    check_challenge=False,
                    allow_status=(403, 404, 429, 503),
                )
            except Exception as exc:
                last_error = str(exc)
                continue
            if response.status_code != 200:
                last_error = f"{instance} 返回 HTTP {response.status_code}"
                continue
            reachable += 1
            parsed = self._parse_nitter(response.text, keyword)
            # Some instances answer a failed search with their own generic
            # timeline instead of an empty feed. Keeping those would silently
            # mix unrelated posts into the dataset, so drop non-matching rows
            # and treat an all-unrelated response as "this instance is unusable".
            items = [item for item in parsed if _mentions(item, keyword)]
            if parsed and not items:
                unrelated += 1
                last_error = f"{instance} 返回了与关键词无关的内容"
                continue
            if items:
                if progress:
                    progress(self.name, 1, 1, f"{keyword}: {len(items)} 条")
                self.note_partial(
                    "search",
                    f"X 使用 {instance} 的 Nitter RSS（未配置 X_BEARER_TOKEN）："
                    "正文/作者/时间可用，没有点赞与转推数",
                    blocking=False,
                )
                return items
        if reachable:
            # An instance answered 200 with an empty feed: the keyword simply has
            # no matches. That is not "all instances are down".
            message = (
                f"X 的 Nitter 实例可访问，但「{keyword}」没有匹配推文"
                "（X 上中文政策词覆盖较少，可换更宽的词或英文词）"
            )
            if unrelated:
                message += f"；另有 {unrelated} 个实例在搜索失败时返回了无关内容，已丢弃"
            self.note_partial("search", message, blocking=False)
            return []
        raise CollectorError(
            f"所有 Nitter 实例都不可用（最后错误：{last_error}）。公共实例经常失效，"
            "可设置 X_NITTER_INSTANCES 指向可用实例，或配置 X_BEARER_TOKEN 使用官方 API"
        )

    @staticmethod
    def _parse_nitter(text: str, keyword: str) -> list[ContentItem]:
        items: list[ContentItem] = []
        for block in re.findall(r"<item>(.*?)</item>", text, re.S):
            title_match = re.search(r"<title>(.*?)</title>", block, re.S)
            link_match = re.search(r"<link>(.*?)</link>", block, re.S)
            date_match = re.search(r"<pubDate>(.*?)</pubDate>", block, re.S)
            creator_match = re.search(r"<dc:creator>(.*?)</dc:creator>", block, re.S)
            desc_match = re.search(r"<description><!\[CDATA\[(.*?)\]\]></description>", block, re.S)
            link = (link_match.group(1).strip() if link_match else "")
            if not link:
                continue
            author = (creator_match.group(1).strip().lstrip("@") if creator_match else "")
            text_value = clean_html(title_match.group(1)) if title_match else ""
            body = clean_html(desc_match.group(1)) if desc_match else text_value
            # Nitter 的链接指向实例自身，换成 x.com 的规范地址并取出推文 ID。
            status_match = re.search(r"/status/(\d+)", link)
            status_id = status_match.group(1) if status_match else link
            # parse_iso returns a datetime, and the model stores ISO strings.
            published = parse_iso(_rfc822_to_iso(date_match.group(1))) if date_match else None
            items.append(
                ContentItem(
                    id=status_id,
                    platform="x",
                    keyword=keyword,
                    content_type="post",
                    title=text_value[:100],
                    summary=body[:1000],
                    author=author,
                    published_at=published.isoformat() if published else "",
                    # Nitter 的 RSS 不提供互动数据。
                    engagement=Engagement(),
                    url=f"https://x.com/{author or 'i'}/status/{status_id}" if status_id.isdigit() else link,
                    extra={
                        "source": "nitter-rss",
                        "engagement_unavailable": True,
                        "hashtags": re.findall(r"#([^\s#]+)", f"{text_value} {body}"),
                    },
                )
            )
        return list({item.id: item for item in items}.values())

    def search(self, keyword: str, progress: ProgressCallback | None = None) -> list[ContentItem]:
        if not self.has_token:
            return self._search_nitter(keyword, progress)
        items: list[ContentItem] = []
        next_token: str | None = None
        for page in range(1, self.config.pages + 1):
            params = self._params(keyword)
            if next_token:
                params["next_token"] = next_token
            try:
                payload = self.get_json(self.endpoint, params=params)
            except Exception as exc:
                if items:
                    self.note_partial("search", exc)
                    break
                raise
            if payload.get("errors"):
                message = f"X API 错误: {payload['errors']}"
                if not payload.get("data"):
                    if items:
                        self.note_partial("search", message)
                        break
                    raise CollectorError(message)
                # Errors alongside partial data: keep the data but record the gap.
                self.note_partial("search", message)
            users, media = self._includes(payload)
            items.extend(self._convert(keyword, row, users, media) for row in payload.get("data") or [])
            if progress:
                progress(self.name, page, self.config.pages, f"{keyword}: {len(items)} 条")
            next_token = (payload.get("meta") or {}).get("next_token")
            if not next_token:
                break
        if self.settings.x_search_mode == "top":
            items.sort(key=lambda row: row.engagement_score, reverse=True)
        return items

    def _thread_and_replies(self, item: ContentItem, reply_limit: int = 100) -> tuple[str, list[Comment]]:
        query = f"conversation_id:{item.extra.get('conversation_id', item.id)}"
        thread: list[dict[str, Any]] = []
        comments: list[Comment] = []
        next_token: str | None = None
        for _ in range(self.config.pages):
            params = self._params(query)
            params["max_results"] = 100
            if next_token:
                params["next_token"] = next_token
            payload = self.get_json(self.endpoint, params=params)
            users, _ = self._includes(payload)
            for raw in payload.get("data") or []:
                user = users.get(str(raw.get("author_id")), {})
                text = str((raw.get("note_tweet") or {}).get("text") or raw.get("text") or "")
                if str(raw.get("author_id")) == str(item.extra.get("author_id")):
                    thread.append(raw)
                elif str(raw.get("id")) != item.id:
                    metrics = raw.get("public_metrics") or {}
                    comments.append(
                        Comment(
                            text=text,
                            likes=safe_int(metrics.get("like_count")),
                            author=str(user.get("username") or user.get("name") or ""),
                            published_at=str(raw.get("created_at", "")),
                        )
                    )
            next_token = (payload.get("meta") or {}).get("next_token")
            if not next_token or len(comments) >= reply_limit:
                break
        thread.sort(key=lambda row: str(row.get("created_at", "")))
        comments.sort(key=lambda row: row.likes, reverse=True)
        full = "\n".join(str((row.get("note_tweet") or {}).get("text") or row.get("text") or "") for row in thread)
        return full or str(item.extra.get("full_text") or item.summary), comments[:reply_limit]

    def deep_dive(self, item: ContentItem) -> DeepContent:
        errors: list[str] = []
        try:
            full_content, comments = self._thread_and_replies(item)
        except Exception as exc:
            errors.append(f"thread/replies: {exc}")
            full_content, comments = str(item.extra.get("full_text") or item.summary), []
        media_parts = []
        for media in item.extra.get("media", []):
            url = str(media.get("url") or media.get("preview_image_url") or "")
            media_parts.append(ocr_image_url(self.client, url, self.settings))
        return DeepContent(
            base=item,
            full_content=full_content,
            comments=comments,
            media_text="\n".join(filter(None, media_parts)),
            errors=errors,
        )
