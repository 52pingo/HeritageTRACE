"""Juejin (稀土掘金) public search/content APIs. No account required.

Search uses the public ``search_api`` endpoint; full text is extracted from the
server-rendered article page (``itemprop="articleBody"``), and comments come
from the public comment list endpoint. All endpoints are plain HTTP JSON/HTML
with no anti-bot signatures, so the collector stays conservative and simply
surfaces errors when the platform changes.
"""

from __future__ import annotations

from typing import Any

from exceptions import CollectorError
from models import Comment, ContentItem, DeepContent, Engagement
from utils import clean_html, extract_html_container, iso_from_timestamp, safe_int

from .base import BaseCollector, ProgressCallback


class JuejinCollector(BaseCollector):
    name = "juejin"
    search_url = "https://api.juejin.cn/search_api/v1/search"
    comment_url = "https://api.juejin.cn/interact_api/v1/comment/list"

    def __init__(self, settings: Any) -> None:
        super().__init__(settings)
        self.client.headers.update({"Referer": "https://juejin.cn/", "Origin": "https://juejin.cn"})

    @staticmethod
    def _convert(item: dict[str, Any], keyword: str) -> ContentItem | None:
        model = item.get("result_model") or {}
        info = model.get("article_info") or {}
        article_id = str(info.get("article_id") or "")
        if not article_id:
            return None
        author = model.get("author_user_info") or {}
        tags = [str((tag or {}).get("tag_name")) for tag in model.get("tags") or [] if isinstance(tag, dict)]
        title = clean_html(str(info.get("title") or ""))
        summary = clean_html(str(info.get("brief_content") or ""))[:1000]
        return ContentItem(
            id=article_id,
            platform="juejin",
            keyword=keyword,
            content_type="article",
            title=title,
            summary=summary,
            author=str(author.get("user_name") or ""),
            published_at=iso_from_timestamp(info.get("ctime")),
            engagement=Engagement(
                likes=safe_int(info.get("digg_count")),
                comments=safe_int(info.get("comment_count")),
                shares=0,
                views=safe_int(info.get("view_count")),
            ),
            url=f"https://juejin.cn/post/{article_id}",
            extra={
                "cover_image": str(info.get("cover_image") or ""),
                "tags": tags,
                "category": str((model.get("category") or {}).get("category_name") or ""),
                "author_followers": safe_int(author.get("follower_count")),
            },
        )

    def search(self, keyword: str, progress: ProgressCallback | None = None) -> list[ContentItem]:
        items: list[ContentItem] = []
        cursor = "0"
        for page in range(1, self.config.pages + 1):
            response = self.request(
                "POST",
                self.search_url,
                json={
                    "key_word": keyword,
                    "id_type": 0,
                    "cursor": cursor,
                    "limit": min(self.config.page_size, 20),
                    "search_type": 0,
                },
            )
            data = response.json()
            if data.get("err_no") != 0:
                raise CollectorError(f"掘金搜索 API 错误: {data.get('err_msg') or data}")
            rows = data.get("data") or []
            for row in rows:
                item = self._convert(row, keyword)
                if item:
                    items.append(item)
            if progress:
                progress("juejin", page, self.config.pages, f"{keyword}: {len(items)} 条")
            if not data.get("has_more") or not rows:
                break
            cursor = str(data.get("cursor") or "")
            if not cursor:
                break
        return list({item.id: item for item in items}.values())

    def _full_text(self, item: ContentItem) -> str:
        # Article pages embed words like "captcha" inside JS bundles, which would
        # false-positive the challenge heuristic; extraction below validates itself.
        response = self.request("GET", item.url, check_challenge=False)
        body = extract_html_container(response.text, 'itemprop="articleBody"')
        return clean_html(body)[:20_000]

    def _comments(self, item: ContentItem, limit: int = 100) -> list[Comment]:
        comments: list[Comment] = []
        cursor = "0"
        while len(comments) < limit:
            response = self.request(
                "POST",
                self.comment_url,
                json={"item_id": item.id, "item_type": 2, "cursor": cursor, "limit": min(20, limit - len(comments))},
            )
            data = response.json()
            if data.get("err_no") != 0:
                break
            rows = data.get("data") or []
            for row in rows:
                info = row.get("comment_info") or {}
                user = row.get("user_info") or {}
                text = clean_html(str(info.get("comment_content") or ""))
                if text:
                    comments.append(
                        Comment(
                            text=text,
                            likes=safe_int(info.get("digg_count")),
                            author=str(user.get("user_name") or ""),
                            published_at=iso_from_timestamp(info.get("ctime")),
                        )
                    )
            if not data.get("has_more") or not rows:
                break
            cursor = str(data.get("cursor") or "")
            if not cursor:
                break
        comments.sort(key=lambda value: value.likes, reverse=True)
        return comments[:limit]

    def deep_dive(self, item: ContentItem) -> DeepContent:
        errors: list[str] = []
        try:
            full_content = self._full_text(item) or item.summary
        except Exception as exc:
            errors.append(f"full_text: {exc}")
            full_content = item.summary
        try:
            comments = self._comments(item, 100)
        except Exception as exc:
            errors.append(f"comments: {exc}")
            comments = []
        return DeepContent(base=item, full_content=full_content, comments=comments, errors=errors)
