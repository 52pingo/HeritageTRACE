"""CSDN public blog search API (so.csdn.net). No account required.

Search hits the public JSON endpoint and keeps only blog results. Full text is
extracted from the article page's ``content_views`` container; comments are not
publicly listed without login, so deep dives carry the article body only.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from exceptions import CollectorError
from models import ContentItem, DeepContent, Engagement
from utils import clean_html, extract_html_container, safe_int

from .base import BaseCollector, ProgressCallback

CN_TZ = timezone(timedelta(hours=8))


class CSDNCollector(BaseCollector):
    name = "csdn"
    search_url = "https://so.csdn.net/api/v3/search"

    def __init__(self, settings: Any) -> None:
        super().__init__(settings)
        self.client.headers.update({"Referer": "https://so.csdn.net/"})

    @staticmethod
    def _published(value: Any) -> str:
        text = str(value or "").strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(text, fmt).replace(tzinfo=CN_TZ).isoformat()
            except ValueError:
                continue
        return text

    @staticmethod
    def _convert(row: dict[str, Any], keyword: str) -> ContentItem | None:
        url = str(row.get("url") or "")
        if not url:
            return None
        title = clean_html(str(row.get("title") or ""))
        description = clean_html(str(row.get("description") or row.get("digest") or ""))
        if not title and not description:
            return None
        tags = row.get("tags") or []
        if isinstance(tags, str):
            tags = [part.strip() for part in tags.split(",") if part.strip()]
        return ContentItem(
            id=str(row.get("id") or url),
            platform="csdn",
            keyword=keyword,
            content_type="article",
            title=title,
            summary=description[:1000],
            author=str(row.get("nickname") or row.get("user_name") or clean_html(str(row.get("author") or ""))),
            published_at=CSDNCollector._published(row.get("created_at")),
            engagement=Engagement(
                likes=safe_int(row.get("digg")),
                comments=safe_int(row.get("comment")),
                shares=0,
                views=safe_int(row.get("view")),
            ),
            url=url.split("?")[0] if "csdn.net" in url else url,
            extra={
                "tags": [str(tag) for tag in tags],
                "category": str(row.get("category_first") or ""),
                "search_result_type": str(row.get("type") or ""),
            },
        )

    def search(self, keyword: str, progress: ProgressCallback | None = None) -> list[ContentItem]:
        items: list[ContentItem] = []
        for page in range(1, self.config.pages + 1):
            payload = self.get_json(
                self.search_url,
                params={
                    "q": keyword,
                    "t": "blog",
                    "p": page,
                    "s": min(self.config.page_size, 100),
                    "tm": 0,
                    "lv": -1,
                    "ft": 0,
                    "l": "",
                },
            )
            if not isinstance(payload, dict) or "result_vos" not in payload:
                raise CollectorError(f"CSDN 搜索返回异常: {str(payload)[:200]}")
            rows = payload.get("result_vos") or []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if str(row.get("type") or "") not in {"", "blog"}:
                    continue
                item = self._convert(row, keyword)
                if item:
                    items.append(item)
            if progress:
                progress("csdn", page, self.config.pages, f"{keyword}: {len(items)} 条")
            total_page = safe_int(payload.get("total_page"))
            if not rows or (total_page and page >= total_page):
                break
        return list({item.id: item for item in items}.values())

    def deep_dive(self, item: ContentItem) -> DeepContent:
        errors: list[str] = []
        try:
            # Blog pages mention "captcha" inside inline scripts; the content
            # extraction below already validates that the article body exists.
            response = self.request("GET", item.url, check_challenge=False)
            body = extract_html_container(response.text, 'id="content_views"')
            full_content = clean_html(body)[:20_000] or item.summary
        except Exception as exc:
            errors.append(f"full_text: {exc}")
            full_content = item.summary
        return DeepContent(base=item, full_content=full_content, comments=[], errors=errors)
