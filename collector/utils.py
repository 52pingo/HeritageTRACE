"""Small, dependency-light normalization helpers."""

from __future__ import annotations

import html
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse


TAG_RE = re.compile(r"<[^>]+>")


def clean_html(value: str | None) -> str:
    return re.sub(r"\s+", " ", html.unescape(TAG_RE.sub("", value or ""))).strip()


def safe_int(value: object) -> int:
    if value is None or value == "":
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip().replace(",", "")
    multipliers = {"万": 10_000, "亿": 100_000_000, "K": 1_000, "M": 1_000_000}
    for suffix, multiplier in multipliers.items():
        if text.upper().endswith(suffix.upper()) or text.endswith(suffix):
            try:
                return int(float(text[: -len(suffix)]) * multiplier)
            except ValueError:
                return 0
    match = re.search(r"-?[\d.]+", text)
    return int(float(match.group())) if match else 0


def iso_from_timestamp(value: int | float | str | None) -> str:
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return ""


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _strip_noncontent(html_text: str) -> str:
    """Drop <script>/<style> blocks so their text does not leak into extracted content."""
    return re.sub(r"<(script|style)\b[^>]*>.*?</\1>", " ", html_text, flags=re.I | re.S)


def extract_html_container(text: str, marker: str) -> str:
    """Extract the inner HTML of the first <div> whose opening tag contains ``marker``.

    Balances nested <div>…</div> pairs from the marker's own opening tag and
    removes embedded <script>/<style> blocks. Returns "" when the marker is missing.
    """
    start = text.find(marker)
    if start < 0:
        return ""
    open_end = text.find(">", start)
    if open_end < 0:
        return ""
    depth, pos = 1, open_end + 1
    opener = re.compile(r"<div\b", re.I)
    closer = re.compile(r"</div>", re.I)
    while depth > 0:
        next_open = opener.search(text, pos)
        next_close = closer.search(text, pos)
        if next_close is None:
            return ""
        if next_open is not None and next_open.start() < next_close.start():
            depth += 1
            pos = next_open.end()
        else:
            depth -= 1
            if depth == 0:
                return _strip_noncontent(text[open_end + 1 : next_close.start()])
            pos = next_close.end()
    return ""


def safe_slug(value: str, max_length: int = 80) -> str:
    value = re.sub(r"[<>:\"/\\|?*\x00-\x1f]+", "_", value.strip())
    value = re.sub(r"\s+", "_", value).strip("._")
    return (value or "dataset")[:max_length]


def valid_public_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def expand_search_queries(keyword: str, expansions: dict[str, list[str]] | None = None) -> list[str]:
    """Return deterministic query variants that include hashtag and configured synonyms."""
    keyword = keyword.strip()
    if not keyword:
        return []
    values = [keyword]
    compact = re.sub(r"\s+", "", keyword)
    if not compact.startswith("#"):
        values.append(f"#{compact}")
    values.extend((expansion.strip() for expansion in (expansions or {}).get(keyword, []) if expansion.strip()))
    return list(dict.fromkeys(values))


def ensure_within(path: Path, root: Path) -> Path:
    resolved, resolved_root = path.resolve(), root.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"Path escapes configured root: {resolved}")
    return resolved


def split_datetime_range(since: datetime, until: datetime, days: int) -> list[tuple[datetime, datetime]]:
    """Split a date range into chunks of at most `days` days."""
    if days <= 0:
        raise ValueError("days must be positive")
    if since > until:
        raise ValueError("since must not be later than until")
    chunks: list[tuple[datetime, datetime]] = []
    cursor = since
    while cursor < until:
        end = min(until, cursor + timedelta(days=days))
        chunks.append((cursor, end))
        cursor = end
    return chunks


def merge_items_by_url(items: list[Any]) -> list[Any]:
    """Deduplicate items by URL/id across keywords, preserving all matched queries."""
    seen: dict[str, Any] = {}
    for item in items:
        key = str(getattr(item, "url", "") or getattr(item, "id", ""))
        if not key:
            key = f"{getattr(item, 'platform', '')}:{getattr(item, 'id', '')}"
        if key in seen:
            existing = seen[key]
            existing_queries = set(existing.extra.get("matched_search_queries", []))
            existing_queries.update(item.extra.get("matched_search_queries", []))
            existing.extra["matched_search_queries"] = sorted(existing_queries)
            existing.extra.setdefault("merged_keywords", []).append(getattr(item, "keyword", ""))
            if getattr(item, "engagement_score", 0) > getattr(existing, "engagement_score", 0):
                # Keep the keyword that has richer engagement metadata but keep merged queries.
                merged_queries = existing.extra.get("matched_search_queries", [])
                merged_keywords = existing.extra.get("merged_keywords", [])
                seen[key] = item
                seen[key].extra["matched_search_queries"] = merged_queries
                seen[key].extra.setdefault("merged_keywords", []).extend(merged_keywords)
        else:
            item.extra.setdefault("merged_keywords", [])
            seen[key] = item
    return list(seen.values())
