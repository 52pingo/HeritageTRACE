"""Generic RSS/Atom feed collector configured via ``RSS_FEEDS_JSON``.

Feeds are standard syndication endpoints published by content providers for
exactly this kind of consumption. There is no search API, so ``search()``
fetches the configured feeds and keeps recent entries whose title or summary
matches the keyword (case-insensitive). Set ``RSS_MATCH_ALL=true`` to keep
every entry regardless of keyword.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from typing import Any

from exceptions import CollectorError
from models import ContentItem, DeepContent, Engagement
from utils import clean_html, parse_iso

from .base import BaseCollector, ProgressCallback


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _text(element: ET.Element | None) -> str:
    return clean_html(element.text or "") if element is not None else ""


def _find_text(item: ET.Element, names: tuple[str, ...]) -> str:
    for child in item:
        if _local(child.tag) in names:
            value = _text(child)
            if value:
                return value
    return ""


def _find_link(item: ET.Element) -> str:
    for child in item:
        if _local(child.tag) != "link":
            continue
        href = child.attrib.get("href", "")
        if href:
            return href.strip()
        if child.text and child.text.strip():
            return child.text.strip()
    return ""


def _entries(root: ET.Element) -> tuple[str, list[ET.Element]]:
    name = _local(root.tag)
    if name == "feed":  # Atom
        return _find_text(root, ("title",)), [child for child in root if _local(child.tag) == "entry"]
    if name == "rss":
        channel = next((child for child in root if _local(child.tag) == "channel"), root)
        return _find_text(channel, ("title",)), [child for child in channel if _local(child.tag) == "item"]
    if name == "rdf":  # RSS 1.0
        return _find_text(root, ("title",)), [child for child in root if _local(child.tag) == "item"]
    return "", []


def _published(item: ET.Element) -> str:
    raw = _find_text(item, ("pubdate", "published", "updated", "date"))
    if not raw:
        return ""
    try:
        return parsedate_to_datetime(raw).isoformat()
    except (TypeError, ValueError, OverflowError):
        parsed = parse_iso(raw)
        return parsed.isoformat() if parsed else raw


class RSSCollector(BaseCollector):
    name = "rss"

    def __init__(self, settings: Any) -> None:
        super().__init__(settings)
        if not settings.rss_feeds:
            raise CollectorError("RSS 采集需要 RSS_FEEDS_JSON，如 {\"少数派\": \"https://sspai.com/feed\"}")

    def _feed_items(self, feed_name: str, feed_url: str, keyword: str) -> list[ContentItem]:
        response = self.request("GET", feed_url)
        try:
            root = ET.fromstring(response.content)
        except ET.ParseError as exc:
            raise CollectorError(f"RSS 源解析失败 {feed_url}: {exc}") from exc
        feed_title, entries = _entries(root)
        needle = keyword.lower()
        items: list[ContentItem] = []
        for entry in entries:
            title = _find_text(entry, ("title",))
            summary = _find_text(entry, ("description", "summary", "content"))
            if not self.settings.rss_match_all and needle not in f"{title} {summary}".lower():
                continue
            link = _find_link(entry)
            guid = _find_text(entry, ("guid", "id")) or link
            if not guid:
                continue
            author = _find_text(entry, ("creator", "author", "name"))
            items.append(
                ContentItem(
                    id=guid,
                    platform="rss",
                    keyword=keyword,
                    content_type="article",
                    title=title,
                    summary=summary[:1000],
                    author=author,
                    published_at=_published(entry),
                    engagement=Engagement(),
                    url=link,
                    extra={"feed": feed_name, "feed_title": feed_title, "feed_url": feed_url},
                )
            )
        return items

    def search(self, keyword: str, progress: ProgressCallback | None = None) -> list[ContentItem]:
        items: list[ContentItem] = []
        feeds = list(self.settings.rss_feeds.items())
        for index, (feed_name, feed_url) in enumerate(feeds, start=1):
            try:
                items.extend(self._feed_items(feed_name, feed_url, keyword))
            except Exception as exc:
                self.log.warning("RSS 源 %s 获取失败: %s", feed_name, exc)
            if progress:
                progress("rss", index, len(feeds), f"{keyword}: {feed_name} 累计 {len(items)} 条")
        return list({item.id: item for item in items}.values())

    def deep_dive(self, item: ContentItem) -> DeepContent:
        # Feed entries already carry the full provider summary; per-site article
        # extraction would need site-specific parsers, so keep the summary.
        return DeepContent(base=item, full_content=item.summary or item.title, comments=[], errors=[])
