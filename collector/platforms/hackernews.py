"""Hacker News public Algolia + Firebase APIs."""

from __future__ import annotations

from typing import Any

from models import Comment, ContentItem, DeepContent, Engagement
from utils import clean_html, iso_from_timestamp, safe_int

from .base import BaseCollector, ProgressCallback


class HackerNewsCollector(BaseCollector):
    name = "hackernews"
    search_url = "https://hn.algolia.com/api/v1/search"
    item_url = "https://hacker-news.firebaseio.com/v0/item/{item_id}.json"

    def search(self, keyword: str, progress: ProgressCallback | None = None) -> list[ContentItem]:
        items: list[ContentItem] = []
        for page in range(self.config.pages):
            payload = self.get_json(
                self.search_url,
                params={
                    "query": keyword,
                    "tags": "(story,comment)",
                    "page": page,
                    "hitsPerPage": min(self.config.page_size, 100),
                },
            )
            hits = payload.get("hits") or []
            for row in hits:
                item_id = str(row.get("objectID", ""))
                title = clean_html(str(row.get("title") or row.get("story_title") or ""))
                body = clean_html(str(row.get("story_text") or row.get("comment_text") or ""))
                if not item_id or not (title or body):
                    continue
                tags = [str(tag) for tag in row.get("_tags") or [] if tag not in {"story", "comment"}]
                url = str(row.get("url") or row.get("story_url") or f"https://news.ycombinator.com/item?id={item_id}")
                items.append(
                    ContentItem(
                        id=item_id,
                        platform=self.name,
                        keyword=keyword,
                        content_type="post",
                        title=title,
                        summary=body[:1000],
                        author=str(row.get("author", "")),
                        published_at=str(row.get("created_at") or iso_from_timestamp(row.get("created_at_i"))),
                        engagement=Engagement(
                            likes=safe_int(row.get("points")),
                            comments=safe_int(row.get("num_comments")),
                            shares=0,
                            views=0,
                        ),
                        url=url,
                        extra={"tags": tags, "story_id": row.get("story_id"), "object_type": row.get("_tags", [])},
                    )
                )
            if progress:
                progress(self.name, page + 1, self.config.pages, f"{keyword}: {len(items)} 条")
            if not hits:
                break
        return list({(item.id, item.keyword): item for item in items}.values())

    def _item(self, item_id: int) -> dict[str, Any]:
        return self.get_json(self.item_url.format(item_id=item_id))

    def deep_dive(self, item: ContentItem) -> DeepContent:
        errors: list[str] = []
        comments: list[Comment] = []
        full_parts = [item.title, item.summary]
        queue = [safe_int(item.id)] if item.id.isdigit() else []
        seen: set[int] = set()
        try:
            while queue and len(comments) < 200:
                current = queue.pop(0)
                if not current or current in seen:
                    continue
                seen.add(current)
                row = self._item(current)
                if row.get("type") == "story":
                    full_parts.extend([clean_html(str(row.get("title", ""))), clean_html(str(row.get("text", "")))])
                elif row.get("type") == "comment":
                    comments.append(
                        Comment(
                            text=clean_html(str(row.get("text", ""))),
                            likes=0,
                            author=str(row.get("by", "")),
                            published_at=iso_from_timestamp(row.get("time")),
                        )
                    )
                queue.extend(safe_int(child) for child in row.get("kids") or [])
        except Exception as exc:
            errors.append(f"thread: {exc}")
        return DeepContent(
            base=item,
            full_content="\n\n".join(filter(None, full_parts)),
            comments=comments[:200],
            errors=errors,
        )

