"""Wikipedia open API collector (MediaWiki action API, keyless)."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

from exceptions import CollectorError
from models import ContentItem, DeepContent, Engagement
from utils import clean_html, parse_iso

from .base import BaseCollector, ProgressCallback


class WikipediaCollector(BaseCollector):
    name = "wikipedia"
    foreign = True
    api_url = "https://zh.wikipedia.org/w/api.php"

    def search(self, keyword: str, progress: ProgressCallback | None = None) -> list[ContentItem]:
        items: list[ContentItem] = []
        limit = min(self.config.page_size, 50)
        for page in range(1, self.config.pages + 1):
            payload = self.get_json(
                self.api_url,
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": keyword,
                    "srlimit": limit,
                    "sroffset": (page - 1) * limit,
                    "srprop": "snippet|timestamp",
                    "format": "json",
                },
            )
            if payload.get("error"):
                raise CollectorError(f"维基百科 API 错误: {payload['error'].get('info')}")
            rows = (payload.get("query") or {}).get("search") or []
            for row in rows:
                title = clean_html(str(row.get("title") or ""))
                page_id = row.get("pageid")
                if not title or page_id is None:
                    continue
                parsed = parse_iso(str(row.get("timestamp") or ""))
                items.append(
                    ContentItem(
                        id=str(page_id),
                        platform="wikipedia",
                        keyword=keyword,
                        content_type="wiki",
                        title=title,
                        summary=clean_html(str(row.get("snippet") or ""))[:1000],
                        author="",
                        published_at=parsed.isoformat() if parsed else "",
                        engagement=Engagement(),
                        url=f"https://zh.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}",
                        extra={"wordcount": row.get("wordcount"), "size": row.get("size")},
                    )
                )
            if progress:
                progress("wikipedia", page, self.config.pages, f"{keyword}: {len(items)} 条")
            if not payload.get("continue") or not rows:
                break
        return list({item.id: item for item in items}.values())

    def deep_dive(self, item: ContentItem) -> DeepContent:
        errors: list[str] = []
        full_content = item.summary
        try:
            payload = self.get_json(
                self.api_url,
                params={
                    "action": "query",
                    "prop": "extracts",
                    "explaintext": 1,
                    "exsectionformat": "plain",
                    "pageids": item.id,
                    "format": "json",
                },
            )
            pages = (payload.get("query") or {}).get("pages") or {}
            for page in pages.values():
                extract = str(page.get("extract") or "").strip()
                if extract:
                    full_content = extract[:20_000]
                    break
        except Exception as exc:
            errors.append(f"extract: {exc}")
        return DeepContent(base=item, full_content=full_content, comments=[], errors=errors)
