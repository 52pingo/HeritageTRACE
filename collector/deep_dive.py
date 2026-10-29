"""Orchestrates L2 extraction while isolating per-item failures."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Callable

from analyzer import sentiment_score, tokenize
from models import ContentItem
from platforms.base import BaseCollector
from storage import DatasetStorage


class DeepDiveRunner:
    def __init__(self, collectors: dict[str, BaseCollector], storage: DatasetStorage) -> None:
        self.collectors = collectors
        self.storage = storage
        self.log = logging.getLogger("deep_dive")

    def run(
        self,
        items: list[ContentItem],
        progress: Callable[[str, int, int, str], None] | None = None,
    ) -> tuple[dict[str, list[dict]], list[dict]]:
        selected = [item for item in items if item.deep_dive_flag]
        grouped: dict[str, list[dict]] = {platform: [] for platform in self.collectors}
        failures: list[dict] = []
        total = len(selected)
        for index, item in enumerate(selected, 1):
            if progress:
                progress(item.platform, index - 1, total, f"deep: 深挖 {item.id}")
            try:
                deep = self.collectors[item.platform].deep_dive(item)
                text = " ".join([deep.full_content, deep.media_text, *(comment.text for comment in deep.comments)])
                deep.sentiment_score = sentiment_score(text)
                deep.keywords_extracted = [token for token, _ in Counter(tokenize(text)).most_common(20)]
                deep.content_length = len(deep.full_content)
                grouped[item.platform].append(deep.to_dict())
            except Exception as exc:
                self.log.exception("深挖失败 %s:%s", item.platform, item.id)
                failures.append({"platform": item.platform, "id": item.id, "keyword": item.keyword, "error": str(exc)})
            if progress:
                progress(item.platform, index, total, f"deep: 完成 {item.id}")
        for platform, rows in grouped.items():
            self.storage.save_deep(platform, rows, replace=True)
        return grouped, failures
