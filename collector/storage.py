"""JSONL + SQLite dual persistence with idempotent upserts."""

from __future__ import annotations

import csv
import json
import logging
import sqlite3
from contextlib import contextmanager
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from config import Settings
from models import ContentItem
from utils import ensure_within, safe_slug

LOGGER = logging.getLogger("storage")

UPSERT_SQL = """
INSERT INTO items(id, keyword, platform, url, published_at, fetched_at,
                  deep_dive_flag, content_hash, payload)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(id, keyword) DO UPDATE SET
    url=excluded.url, published_at=excluded.published_at,
    fetched_at=excluded.fetched_at, deep_dive_flag=excluded.deep_dive_flag,
    content_hash=excluded.content_hash, payload=excluded.payload
"""


def _row_values(row: dict[str, Any]) -> tuple:
    return (
        str(row.get("id", "")),
        str(row.get("keyword", "")),
        str(row.get("platform", "")),
        str(row.get("url", "")),
        str(row.get("published_at", "")),
        str(row.get("fetched_at", row.get("deep_fetched_at", ""))),
        int(bool(row.get("deep_dive_flag"))),
        str(row.get("content_hash", "")),
        json.dumps(row, ensure_ascii=False, separators=(",", ":")),
    )


class DatasetStorage:
    def __init__(self, settings: Settings, keywords: list[str], run_id: str | None = None) -> None:
        stamp = safe_slug(run_id) if run_id else datetime.now().strftime("%Y%m%d_%H%M%S")
        name = safe_slug("_".join(keywords))
        self.root = ensure_within(settings.data_root / "datasets" / f"{name}_{stamp}", settings.data_root)
        self.raw_dir = self.root / "raw"
        self.deep_dir = self.root / "deep"
        self.report_dir = self.root / "report"
        for directory in (self.raw_dir, self.deep_dir, self.report_dir):
            directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    @contextmanager
    def _connect(path: Path):
        connection = sqlite3.connect(path)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS items (
                    id TEXT NOT NULL,
                    keyword TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    url TEXT,
                    published_at TEXT,
                    fetched_at TEXT,
                    deep_dive_flag INTEGER NOT NULL DEFAULT 0,
                    content_hash TEXT,
                    payload TEXT NOT NULL,
                    PRIMARY KEY (id, keyword)
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_items_url ON items(url)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_items_deep ON items(deep_dive_flag)")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _save(self, directory: Path, stem: str, rows: Iterable[dict[str, Any]], replace_jsonl: bool) -> int:
        materialized = list(rows)
        jsonl_path = directory / f"{stem}.jsonl"
        db_path = directory / f"{stem}.db"
        with self._connect(db_path) as connection:
            existing = connection.execute("SELECT COUNT(*) FROM items").fetchone()[0]
            if replace_jsonl and not materialized and existing:
                # Never let an empty result set destroy a previous run's records.
                LOGGER.warning(
                    "%s: 本次没有采集到内容，保留已有 %d 条记录（不覆盖）", stem, existing
                )
                return 0
            if replace_jsonl:
                connection.execute("DELETE FROM items")
            if materialized:
                connection.executemany(UPSERT_SQL, [_row_values(row) for row in materialized])
            payloads = [row[0] for row in connection.execute("SELECT payload FROM items ORDER BY rowid")]

        with jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
            for payload in payloads:
                handle.write(payload + "\n")
        return len(materialized)

    def save_raw(
        self,
        platform: str,
        items: Iterable[ContentItem],
        replace: bool = True,
        merge: bool = False,
    ) -> int:
        """Persist one platform's L1 rows.

        ``merge=True`` upserts the incoming rows on top of whatever this dataset
        directory already contains instead of replacing it, so re-running the
        same command accumulates data rather than discarding the earlier run.
        """
        return self._save(self.raw_dir, platform, (item.to_dict() for item in items), replace and not merge)

    def load_all_raw(self) -> dict[str, list[ContentItem]]:
        """Every platform already persisted in this dataset directory."""
        if not self.raw_dir.exists():
            return {}
        loaded: dict[str, list[ContentItem]] = {}
        for path in sorted(self.raw_dir.glob("*.db")):
            items = self.load_raw(path.stem)
            if items:
                loaded[path.stem] = items
        return loaded

    def deep_dive_keys(self) -> set[tuple[str, str]]:
        """(id, keyword) pairs that already have an L2 record in this directory."""
        if not self.deep_dir.exists():
            return set()
        keys: set[tuple[str, str]] = set()
        for path in self.deep_dir.glob("*_deep.db"):
            with self._connect(path) as connection:
                keys.update(connection.execute("SELECT id, keyword FROM items").fetchall())
        return keys

    def save_deep(self, platform: str, rows: Iterable[dict[str, Any]], replace: bool = True) -> int:
        # Hash-based dedup keeps the first record while retaining items with genuinely different content.
        unique: dict[str, dict[str, Any]] = {}
        for row in rows:
            key = str(row.get("content_hash") or f"{row.get('id')}:{row.get('keyword')}")
            unique.setdefault(key, row)
        return self._save(self.deep_dir, f"{platform}_deep", unique.values(), replace)

    def load_raw(self, platform: str) -> list[ContentItem]:
        path = self.raw_dir / f"{platform}.db"
        if not path.exists():
            return []
        with self._connect(path) as connection:
            rows = connection.execute("SELECT payload FROM items ORDER BY rowid").fetchall()
        return [ContentItem.from_dict(json.loads(row[0])) for row in rows]

    def write_json(self, relative_path: str, payload: Any) -> Path:
        path = ensure_within(self.root / relative_path, self.root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def export_csv(self, platform: str, deep: bool = False) -> Path:
        source = self.deep_dir / f"{platform}_deep.jsonl" if deep else self.raw_dir / f"{platform}.jsonl"
        target = source.with_suffix(".csv")
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            target.write_text("", encoding="utf-8")
            return target
        normalized = []
        for row in rows:
            normalized.append({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})
        fields = sorted({field for row in normalized for field in row})
        with target.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(normalized)
        return target

    def export_parquet(self, platform: str, deep: bool = False) -> Path:
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("Parquet 导出需要 pandas 和 pyarrow") from exc
        source = self.deep_dir / f"{platform}_deep.jsonl" if deep else self.raw_dir / f"{platform}.jsonl"
        target = source.with_suffix(".parquet")
        pd.read_json(source, lines=True).to_parquet(target, index=False)
        return target

    def export_huggingface(self, platform: str, deep: bool = False) -> Path:
        try:
            from datasets import Dataset
        except ImportError as exc:
            raise RuntimeError("HuggingFace 导出需要 datasets") from exc
        source = self.deep_dir / f"{platform}_deep.jsonl" if deep else self.raw_dir / f"{platform}.jsonl"
        target = source.with_suffix("")
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        Dataset.from_list(rows).save_to_disk(str(target) + "_hf")
        return Path(str(target) + "_hf")
