"""arXiv public Atom API collector for paper metadata and abstracts."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

from exceptions import CollectorError
from models import ContentItem, DeepContent, Engagement
from utils import clean_html

from .base import BaseCollector, ProgressCallback

ATOM = {"a": "http://www.w3.org/2005/Atom"}
OPENSEARCH = {"o": "http://a9.com/-/spec/opensearch/1.1/"}


class ArxivCollector(BaseCollector):
    name = "arxiv"
    foreign = True
    api_url = "http://export.arxiv.org/api/query"

    @staticmethod
    def _text(element: ET.Element | None, path: str) -> str:
        node = element.find(path, ATOM) if element is not None else None
        return " ".join((node.text or "").split()) if node is not None else ""

    def _convert(self, entry: ET.Element, keyword: str) -> ContentItem | None:
        abs_url = ""
        for link in entry.findall("a:link", ATOM):
            if link.get("rel") in {None, "alternate"} and link.get("href"):
                abs_url = str(link.get("href"))
                break
        arxiv_id = self._text(entry, "a:id").rsplit("/abs/", 1)[-1]
        if not arxiv_id:
            return None
        authors = [self._text(author, "a:name") for author in entry.findall("a:author", ATOM)]
        categories = [str(cat.get("term")) for cat in entry.findall("a:category") if cat.get("term")]
        title = self._text(entry, "a:title")
        summary = self._text(entry, "a:summary")
        return ContentItem(
            id=arxiv_id,
            platform="arxiv",
            keyword=keyword,
            content_type="paper",
            title=title,
            summary=summary[:1000],
            author=", ".join(authors[:5]),
            published_at=self._text(entry, "a:published"),
            engagement=Engagement(),
            url=abs_url or f"https://arxiv.org/abs/{arxiv_id}",
            extra={
                "authors": authors,
                "categories": categories,
                "updated": self._text(entry, "a:updated"),
            },
        )

    def search(self, keyword: str, progress: ProgressCallback | None = None) -> list[ContentItem]:
        items: list[ContentItem] = []
        start = 0
        page_size = min(self.config.page_size, 50)
        for page in range(1, self.config.pages + 1):
            response = self.request(
                "GET",
                self.api_url,
                params={
                    "search_query": f"all:{keyword}",
                    "start": start,
                    "max_results": page_size,
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                },
            )
            try:
                root = ET.fromstring(response.content)
            except ET.ParseError as exc:
                raise CollectorError(f"arXiv 返回了无法解析的 XML: {exc}") from exc
            entries = root.findall("a:entry", ATOM)
            for entry in entries:
                item = self._convert(entry, keyword)
                if item:
                    items.append(item)
            total_node = root.find("o:totalResults", OPENSEARCH)
            try:
                total = int(total_node.text) if total_node is not None and total_node.text else 0
            except ValueError:
                total = 0
            if progress:
                progress("arxiv", page, self.config.pages, f"{keyword}: {len(items)}/{total} 条")
            start += len(entries)
            if not entries or start >= total:
                break
        return list({item.id: item for item in items}.values())

    def deep_dive(self, item: ContentItem) -> DeepContent:
        # arXiv 正文是 PDF/源码，超出文本采集范围；摘要即完整可用文本。
        full_content = "\n\n".join(filter(None, [item.title, item.summary]))
        return DeepContent(base=item, full_content=full_content, comments=[], errors=[])
