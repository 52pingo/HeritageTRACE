"""Stack Exchange official API v2.3 collector (Stack Overflow etc.).

The API is free and keyless with a generous anonymous quota. Responses carry a
``backoff`` field which we honor as a courtesy; quota errors surface as
recoverable ``CollectorError`` so the pipeline continues with other platforms.
"""

from __future__ import annotations

import time
from typing import Any

from exceptions import CollectorError
from models import Comment, ContentItem, DeepContent, Engagement
from utils import clean_html, iso_from_timestamp, safe_int

from .base import BaseCollector, ProgressCallback


class StackExchangeCollector(BaseCollector):
    name = "stackexchange"
    foreign = True
    api_root = "https://api.stackexchange.com/2.3"

    def __init__(self, settings: Any) -> None:
        super().__init__(settings)
        self.site = (getattr(settings, "stackexchange_site", "") or "stackoverflow").strip()

    def _api(self, resource: str, **params: Any) -> dict[str, Any]:
        params.setdefault("site", self.site)
        payload = self.get_json(f"{self.api_root}/{resource}", params=params)
        backoff = safe_int(payload.get("backoff"))
        if backoff:
            self.log.info("Stack Exchange 要求 backoff=%ds", backoff)
            time.sleep(backoff)
        if payload.get("error_id"):
            raise CollectorError(f"Stack Exchange API 错误 {payload.get('error_id')}: {payload.get('error_message')}")
        return payload

    @staticmethod
    def _convert(row: dict[str, Any], keyword: str) -> ContentItem | None:
        question_id = safe_int(row.get("question_id"))
        if not question_id:
            return None
        owner = row.get("owner") or {}
        return ContentItem(
            id=str(question_id),
            platform="stackexchange",
            keyword=keyword,
            content_type="qa",
            title=clean_html(str(row.get("title") or "")),
            summary=clean_html(str(row.get("body") or ""))[:1000],
            author=str(owner.get("display_name") or ""),
            published_at=iso_from_timestamp(row.get("creation_date")),
            engagement=Engagement(
                likes=safe_int(row.get("score")),
                comments=safe_int(row.get("answer_count")),
                shares=0,
                views=safe_int(row.get("view_count")),
            ),
            url=str(row.get("link") or f"https://stackoverflow.com/questions/{question_id}"),
            extra={
                "tags": [str(tag) for tag in row.get("tags") or []],
                "is_answered": bool(row.get("is_answered")),
                "site": row.get("site") or "",
            },
        )

    def search(self, keyword: str, progress: ProgressCallback | None = None) -> list[ContentItem]:
        items: list[ContentItem] = []
        for page in range(1, self.config.pages + 1):
            payload = self._api(
                "search/advanced",
                q=keyword,
                order="desc",
                sort="relevance",
                page=page,
                pagesize=min(self.config.page_size, 100),
                filter="withbody",
            )
            rows = payload.get("items") or []
            for row in rows:
                item = self._convert(row, keyword)
                if item:
                    items.append(item)
            if progress:
                progress("stackexchange", page, self.config.pages, f"{keyword}: {len(items)} 条")
            if not payload.get("has_more") or not rows:
                break
        return list({item.id: item for item in items}.values())

    def deep_dive(self, item: ContentItem) -> DeepContent:
        errors: list[str] = []
        parts = [item.summary]
        try:
            payload = self._api(
                "questions/" + item.id + "/answers",
                order="desc",
                sort="votes",
                pagesize=20,
                filter="withbody",
            )
            for row in payload.get("items") or []:
                body = clean_html(str(row.get("body") or ""))
                if body:
                    parts.append(f"【回答 · {safe_int(row.get('score'))} 票】\n{body}")
        except Exception as exc:
            errors.append(f"answers: {exc}")
        comments: list[Comment] = []
        try:
            payload = self._api("questions/" + item.id + "/comments", order="desc", sort="votes", pagesize=50)
            for row in payload.get("items") or []:
                text = clean_html(str(row.get("body") or ""))
                if text:
                    owner = row.get("owner") or {}
                    comments.append(
                        Comment(
                            text=text,
                            likes=safe_int(row.get("score")),
                            author=str(owner.get("display_name") or ""),
                            published_at=iso_from_timestamp(row.get("creation_date")),
                        )
                    )
        except Exception as exc:
            errors.append(f"comments: {exc}")
        return DeepContent(
            base=item,
            full_content="\n\n".join(filter(None, parts))[:20_000],
            comments=comments[:100],
            errors=errors,
        )
