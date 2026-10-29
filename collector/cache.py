"""Cross-run URL-hash cache used for incremental collection."""

from __future__ import annotations

import hashlib
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from models import ContentItem


class HarvestCache:
    def __init__(self, data_root: Path, ttl_days: int = 0) -> None:
        self.path = data_root / "cache" / "seen.db"
        # 0 表示已见 URL 永不过期（保持原有语义）。
        self.ttl_days = max(int(ttl_days or 0), 0)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS seen (
                    platform TEXT NOT NULL,
                    keyword TEXT NOT NULL,
                    url_hash TEXT NOT NULL,
                    item_id TEXT,
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    PRIMARY KEY(platform, keyword, url_hash)
                )
                """
            )

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path)
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _hash(item: ContentItem) -> str:
        identity = item.url or f"{item.platform}:{item.id}"
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def _cutoff(self) -> str | None:
        if not self.ttl_days:
            return None
        return (datetime.now(timezone.utc) - timedelta(days=self.ttl_days)).isoformat()

    def filter_new(self, items: list[ContentItem]) -> tuple[list[ContentItem], int]:
        if not items:
            return [], 0
        cutoff = self._cutoff()
        with self._connect() as connection:
            if cutoff:
                connection.execute("DELETE FROM seen WHERE last_seen_at < ?", (cutoff,))
                rows = connection.execute(
                    "SELECT platform, keyword, url_hash FROM seen WHERE last_seen_at >= ?", (cutoff,)
                ).fetchall()
            else:
                rows = connection.execute("SELECT platform, keyword, url_hash FROM seen").fetchall()
            keys = {(row[0], row[1], row[2]) for row in rows}
        fresh = [item for item in items if (item.platform, item.keyword, self._hash(item)) not in keys]
        return fresh, len(items) - len(fresh)

    def mark(self, items: list[ContentItem]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO seen(platform, keyword, url_hash, item_id, first_seen_at, last_seen_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(platform, keyword,url_hash) DO UPDATE SET
                    item_id=excluded.item_id, last_seen_at=excluded.last_seen_at
                """,
                [(item.platform, item.keyword, self._hash(item), item.id, now, now) for item in items],
            )
