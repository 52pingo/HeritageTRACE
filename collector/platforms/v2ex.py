"""V2EX collector: search via the public sov2ex API, details via v2ex JSON API.

sov2ex (sov2ex.com) is a community-run search index over public V2EX topics.
The official v2ex.com JSON API provides topic bodies and replies without
authentication. No cookies, no signatures.
"""

from __future__ import annotations

from typing import Any

from exceptions import CollectorError
from models import Comment, ContentItem, DeepContent, Engagement
from utils import clean_html, safe_int

from .base import BaseCollector, ProgressCallback


class V2EXCollector(BaseCollector):
    name = "v2ex"
    foreign = True
    search_url = "https://www.sov2ex.com/api/search"
    api_root = "https://www.v2ex.com/api"

    @staticmethod
    def _convert(source: dict[str, Any], keyword: str) -> ContentItem | None:
        topic_id = source.get("id")
        if topic_id is None:
            return None
        member = source.get("member") or ""
        if isinstance(member, dict):
            member = member.get("username") or ""
        return ContentItem(
            id=str(topic_id),
            platform="v2ex",
            keyword=keyword,
            content_type="post",
            title=clean_html(str(source.get("title") or "")),
            summary=clean_html(str(source.get("content") or ""))[:1000],
            author=str(member),
            published_at=str(source.get("created") or ""),
            engagement=Engagement(likes=0, comments=safe_int(source.get("replies")), shares=0, views=0),
            url=f"https://www.v2ex.com/t/{topic_id}",
            extra={"node_id": source.get("node")},
        )

    def search(self, keyword: str, progress: ProgressCallback | None = None) -> list[ContentItem]:
        items: list[ContentItem] = []
        page_size = min(self.config.page_size, 50)
        for page in range(1, self.config.pages + 1):
            payload = self.get_json(
                self.search_url,
                params={"q": keyword, "size": page_size, "from": (page - 1) * page_size},
            )
            hits = payload.get("hits") or []
            for hit in hits:
                item = self._convert(hit.get("_source") or {}, keyword)
                if item:
                    items.append(item)
            if progress:
                progress("v2ex", page, self.config.pages, f"{keyword}: {len(items)} 条")
            if not hits:
                break
        return list({item.id: item for item in items}.values())

    def deep_dive(self, item: ContentItem) -> DeepContent:
        errors: list[str] = []
        full_content = item.summary
        comments: list[Comment] = []
        try:
            # topics/show.json 返回 JSON 数组，不能用 get_json（它要求对象）
            response = self.request("GET", f"{self.api_root}/topics/show.json", params={"id": item.id})
            rows = response.json()
            rows = rows if isinstance(rows, list) else [rows]
            if rows and isinstance(rows[0], dict):
                topic = rows[0]
                content = clean_html(str(topic.get("content_rendered") or topic.get("content") or ""))
                if content:
                    full_content = content[:20_000]
                node = topic.get("node") or {}
                if isinstance(node, dict) and node.get("title"):
                    item.extra["node_title"] = node.get("title")
        except Exception as exc:
            errors.append(f"topic: {exc}")
        try:
            response = self.request("GET", f"{self.api_root}/replies/show.json", params={"topic_id": item.id})
            rows = response.json()
            rows = rows if isinstance(rows, list) else []
            for row in rows:
                member = (row.get("member") or {}) if isinstance(row, dict) else {}
                text = clean_html(str(row.get("content_rendered") or row.get("content") or ""))
                if text:
                    comments.append(
                        Comment(
                            text=text,
                            likes=0,
                            author=str(member.get("username") or ""),
                            published_at=str(row.get("created") or ""),
                        )
                    )
        except Exception as exc:
            errors.append(f"replies: {exc}")
        return DeepContent(base=item, full_content=full_content, comments=comments[:100], errors=errors)
