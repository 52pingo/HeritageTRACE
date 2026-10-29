"""Weibo public mobile-search collector. A user-exported Cookie is required."""

from __future__ import annotations

from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any, Iterator

from exceptions import AuthenticationRequired, CollectorError
from media_tools import ocr_image_url
from models import Comment, ContentItem, DeepContent, Engagement
from utils import clean_html, iso_from_timestamp, safe_int

from .base import BaseCollector, ProgressCallback


class WeiboCollector(BaseCollector):
    name = "weibo"

    def __init__(self, settings: Any) -> None:
        super().__init__(settings)
        if not settings.weibo_cookie:
            raise AuthenticationRequired("微博采集需要 WEIBO_COOKIE；请从本人已登录浏览器导出公开网页 Cookie")

    def _credential_headers(self) -> dict[str, str]:
        return {"Cookie": self.settings.weibo_cookie, "Referer": "https://m.weibo.cn/"}

    @staticmethod
    def _walk_cards(cards: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        for card in cards:
            if card.get("mblog"):
                yield card["mblog"]
            groups = card.get("card_group") or []
            if groups:
                yield from WeiboCollector._walk_cards(groups)

    @staticmethod
    def _published(value: str) -> str:
        if not value:
            return ""
        try:
            return parsedate_to_datetime(value).isoformat()
        except (TypeError, ValueError, OverflowError):
            try:
                return datetime.fromisoformat(value).isoformat()
            except ValueError:
                return value

    def _container(self, keyword: str, mode: str) -> tuple[str, str]:
        # These are the public mobile search tabs; platform changes are surfaced as recoverable errors.
        if mode == "realtime":
            return f"100103type=61&q={keyword}&t=0", "searchwb"
        if mode == "hot":
            return f"100103type=60&q={keyword}&t=0", "searchwb"
        return f"100103type=1&q={keyword}", "searchall"

    def search(self, keyword: str, progress: ProgressCallback | None = None) -> list[ContentItem]:
        dedup: dict[str, ContentItem] = {}
        modes = [mode for mode in self.settings.weibo_search_modes if mode in {"comprehensive", "realtime", "hot"}]
        modes = modes or ["comprehensive"]
        total_pages = self.config.pages * len(modes)
        completed = 0
        abort = False
        for mode in modes:
            if abort:
                break
            container_id, page_type = self._container(keyword, mode)
            for page in range(1, self.config.pages + 1):
                try:
                    payload = self.get_json(
                        "https://m.weibo.cn/api/container/getIndex",
                        params={"containerid": container_id, "page_type": page_type, "page": page},
                    )
                except Exception as exc:
                    # Keep what earlier modes/pages already produced.
                    if dedup:
                        self.note_partial("search", exc)
                        abort = True
                        break
                    raise
                if payload.get("ok") != 1:
                    message = str(payload.get("msg") or payload)
                    # "这里还没有内容" is a legitimate empty result, not a failure.
                    if self._is_empty_response(message):
                        break
                    if page == 1 and not dedup:
                        raise CollectorError(f"微博 {mode} 搜索返回异常: {message}")
                    self.note_partial("search", f"微博 {mode} 搜索返回异常: {message}")
                    abort = True
                    break
                data = payload.get("data") or {}
                cards = data.get("cards") if isinstance(data, dict) else None
                rows = list(self._walk_cards(cards or []))
                if not rows:
                    break
                for row in rows:
                    item_id = str(row.get("idstr") or row.get("id") or "")
                    if not item_id:
                        continue
                    user = row.get("user") or {}
                    text = clean_html(row.get("text"))
                    topics = [part.strip("#") for part in text.split() if part.startswith("#") and part.endswith("#")]
                    # 微博的 pics 有时是字典列表，有时直接是 URL 字符串列表
                    pics = [
                        pic if isinstance(pic, str)
                        else ((pic.get("large") or {}).get("url") or pic.get("url"))
                        for pic in (row.get("pics") or [])
                    ]
                    item = ContentItem(
                        id=item_id,
                        platform=self.name,
                        keyword=keyword,
                        content_type="post",
                        title=text[:80],
                        summary=text[:200],
                        author=str(user.get("screen_name", "")),
                        published_at=self._published(str(row.get("created_at", ""))),
                        engagement=Engagement(
                            likes=safe_int(row.get("attitudes_count")),
                            comments=safe_int(row.get("comments_count")),
                            shares=safe_int(row.get("reposts_count")),
                            views=safe_int(row.get("reads_count")),
                        ),
                        url=f"https://m.weibo.cn/status/{item_id}",
                        extra={
                            "mid": str(row.get("mid") or item_id),
                            "search_modes": [mode],
                            "topics": topics,
                            "picture_urls": [url for url in pics if url],
                            "author_id": user.get("id"),
                            "author_followers": safe_int(user.get("followers_count")),
                            "is_long_text": bool(row.get("isLongText")),
                        },
                    )
                    if item_id in dedup:
                        previous = dedup[item_id].extra.setdefault("search_modes", [])
                        if mode not in previous:
                            previous.append(mode)
                    else:
                        dedup[item_id] = item
                completed += 1
                if progress:
                    progress(self.name, completed, total_pages, f"{mode}/{keyword}: {len(dedup)} 条")
        return list(dedup.values())

    # m.weibo.cn 用 ok=0 同时表达「没有内容」和「登录态失效」，靠文案区分。
    EMPTY_MARKERS = ("这里还没有内容", "还没有内容", "未找到", "没有找到")

    @classmethod
    def _is_empty_response(cls, message: str) -> bool:
        return any(marker in message for marker in cls.EMPTY_MARKERS)

    def _full_text(self, item: ContentItem) -> str:
        if not item.extra.get("is_long_text"):
            return item.summary
        payload = self.get_json("https://m.weibo.cn/statuses/extend", params={"id": item.id})
        return clean_html((payload.get("data") or {}).get("longTextContent")) or item.summary

    def _comments(self, item: ContentItem, limit: int = 500) -> tuple[list[Comment], list[str]]:
        comments: list[Comment] = []
        image_urls: list[str] = []
        max_id, max_id_type = 0, 0
        while len(comments) < limit:
            payload = self.get_json(
                "https://m.weibo.cn/comments/hotflow",
                params={"id": item.id, "mid": item.extra.get("mid", item.id), "max_id": max_id, "max_id_type": max_id_type},
            )
            data = payload.get("data") or {}
            rows = data.get("data") or []
            for row in rows:
                user = row.get("user") or {}
                picture = row.get("pic") or {}
                picture_url = (picture.get("large") or {}).get("url") or picture.get("url")
                if picture_url:
                    image_urls.append(str(picture_url))
                comments.append(
                    Comment(
                        text=clean_html(row.get("text")),
                        likes=safe_int(row.get("like_count")),
                        author=str(user.get("screen_name", "")),
                        published_at=self._published(str(row.get("created_at", ""))),
                    )
                )
            next_id = safe_int(data.get("max_id"))
            if not rows or not next_id or next_id == max_id:
                break
            max_id = next_id
            max_id_type = safe_int(data.get("max_id_type"))
        comments.sort(key=lambda row: row.likes, reverse=True)
        return comments[:limit], list(dict.fromkeys(image_urls))

    def deep_dive(self, item: ContentItem) -> DeepContent:
        errors: list[str] = []
        try:
            full_content = self._full_text(item)
        except Exception as exc:
            errors.append(f"full_text: {exc}")
            full_content = item.summary
        try:
            comments, comment_images = self._comments(item)
        except Exception as exc:
            errors.append(f"comments: {exc}")
            comments, comment_images = [], []
        media_urls = list(item.extra.get("picture_urls", [])) + comment_images
        media_parts = [ocr_image_url(self.client, url, self.settings) for url in media_urls]
        return DeepContent(
            base=item,
            full_content=full_content,
            comments=comments,
            media_text="\n".join(filter(None, media_parts)),
            errors=errors,
        )
