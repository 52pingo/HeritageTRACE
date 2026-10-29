"""Baidu Hot Search (百度热搜) board collector.

The board endpoint is public JSON with no authentication. It is a ranked list
rather than a search engine, so ``search()`` keeps entries whose title or
description matches the keyword (use a broad keyword; set ``BAIDU_MATCH_ALL=true``
to archive the whole board). Extra boards beyond 实时 can be selected with
``BAIDU_TAB`` (realtime / novel / movie / teleplay / car / game).
"""

from __future__ import annotations

from typing import Any, Iterator

from exceptions import CollectorError
from models import ContentItem, DeepContent, Engagement
from utils import clean_html, safe_int

from .base import BaseCollector, ProgressCallback


class BaiduHotCollector(BaseCollector):
    name = "baidu"
    board_url = "https://top.baidu.com/api/board"

    def __init__(self, settings: Any) -> None:
        super().__init__(settings)
        self.tab = (getattr(settings, "baidu_tab", "") or "realtime").strip()
        self.match_all = bool(getattr(settings, "baidu_match_all", False))

    @staticmethod
    def _walk(value: Any) -> Iterator[dict[str, Any]]:
        if isinstance(value, dict):
            if "word" in value:
                yield value
            for child in value.values():
                yield from BaiduHotCollector._walk(child)
        elif isinstance(value, list):
            for child in value:
                yield from BaiduHotCollector._walk(child)

    def search(self, keyword: str, progress: ProgressCallback | None = None) -> list[ContentItem]:
        payload = self.get_json(self.board_url, params={"platform": "pc", "tab": self.tab})
        if not payload.get("success"):
            raise CollectorError(f"百度热搜返回异常: {str(payload)[:200]}")
        needle = keyword.lower()
        items: list[ContentItem] = []
        for entry in self._walk((payload.get("data") or {}).get("cards") or []):
            word = clean_html(str(entry.get("word") or ""))
            desc = clean_html(str(entry.get("desc") or ""))
            if not word:
                continue
            if not self.match_all and needle not in f"{word} {desc}".lower():
                continue
            rank = safe_int(entry.get("index"))
            items.append(
                ContentItem(
                    id=str(entry.get("query") or word)[:120],
                    platform="baidu",
                    keyword=keyword,
                    content_type="trend",
                    title=word,
                    summary=desc[:1000],
                    author="百度热搜",
                    published_at="",
                    engagement=Engagement(views=safe_int(entry.get("hotScore"))),
                    url=str(entry.get("rawUrl") or entry.get("url") or ""),
                    extra={
                        "rank": rank,
                        "tab": self.tab,
                        "hot_tag": clean_html(str(entry.get("hotTag") or "")),
                        "image": str(entry.get("img") or ""),
                    },
                )
            )
        if progress:
            progress("baidu", 1, 1, f"{keyword}: {len(items)} 条 (榜单共 {sum(1 for _ in self._walk((payload.get('data') or {}).get('cards') or []))} 条)")
        return list({item.url or item.id: item for item in items}.values())

    def deep_dive(self, item: ContentItem) -> DeepContent:
        # 榜单条目没有官方全文接口；摘要即完整可用文本。
        return DeepContent(base=item, full_content=item.summary or item.title, comments=[], errors=[])
