"""Build a unified, ML-ready dataset from one or more collector outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

from utils import clean_html


def _find_dataset_dirs(root: Path) -> list[Path]:
    """Return all dataset directories under root, newest first."""
    datasets_root = root / "datasets"
    if not datasets_root.exists():
        return []
    dirs = [path for path in datasets_root.iterdir() if path.is_dir()]
    return sorted(dirs, key=lambda p: p.stat().st_mtime, reverse=True)


def _text_parts(row: dict[str, Any]) -> list[str]:
    """Extract all textual content suitable for NLP models."""
    parts: list[str] = []
    for key in ("title", "summary", "full_content", "media_text"):
        value = row.get(key)
        if value:
            parts.append(clean_html(str(value)))
    for comment in row.get("comments") or []:
        text = comment.get("text") if isinstance(comment, dict) else getattr(comment, "text", None)
        if text:
            parts.append(clean_html(str(text)))
    for key in ("tags", "hashtags", "topics"):
        values = row.get(key)
        if isinstance(values, list):
            parts.extend(str(v) for v in values if v)
    return parts


def _engagement_features(row: dict[str, Any]) -> dict[str, float | int]:
    engagement = row.get("engagement") or {}
    if not isinstance(engagement, dict):
        engagement = {}
    score = row.get("engagement_score")
    if score is None and isinstance(engagement, dict):
        likes = engagement.get("likes", 0) or 0
        comments = engagement.get("comments", 0) or 0
        shares = engagement.get("shares", 0) or 0
        views = engagement.get("views", 0) or 0
        score = likes + comments * 2 + shares * 3 + views * 0.001
    return {
        "likes": engagement.get("likes", 0) or 0,
        "comments": engagement.get("comments", 0) or 0,
        "shares": engagement.get("shares", 0) or 0,
        "views": engagement.get("views", 0) or 0,
        "engagement_score": round(float(score or 0), 4),
    }


def _label_fields(row: dict[str, Any]) -> dict[str, Any]:
    sentiment = row.get("sentiment_score", 0)
    if not isinstance(sentiment, (int, float)):
        sentiment = 0
    return {
        "deep_dive_flag": int(bool(row.get("deep_dive_flag"))),
        "sentiment_score": round(float(sentiment), 4),
        "content_length": int(row.get("content_length") or len(row.get("full_content", "") or row.get("summary", ""))),
    }


def _build_sample(row: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize a single raw/deep record into an ML sample."""
    platform = str(row.get("platform", ""))
    item_id = str(row.get("id", ""))
    keyword = str(row.get("keyword", ""))
    if not platform or not item_id:
        return None

    text = "\n".join(_text_parts(row)).strip()
    if not text:
        return None

    engagement = _engagement_features(row)
    labels = _label_fields(row)
    url = str(row.get("url", ""))
    canonical = json.dumps({"platform": platform, "id": item_id, "keyword": keyword, "url": url}, ensure_ascii=False, sort_keys=True)
    sample = {
        "sample_id": hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16],
        "platform": platform,
        "content_type": str(row.get("content_type", "")),
        "keyword": keyword,
        "title": clean_html(str(row.get("title", ""))),
        "text": text,
        "url": url,
        "author": str(row.get("author", "")),
        "published_at": str(row.get("published_at", "")),
        "fetched_at": str(row.get("fetched_at", "")),
        **engagement,
        **labels,
        "deep_dive_reasons": row.get("extra", {}).get("deep_dive_reasons", []),
        "matched_search_queries": row.get("extra", {}).get("matched_search_queries", []),
        "source_file": "",
    }
    return sample


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def build_ml_dataset(
    dataset_dir: Path,
    *,
    deep_only: bool = False,
    dedup_text: bool = True,
    min_text_length: int = 10,
    embeddings: str | None = None,
    seed: int = 42,
    splits: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, Any]:
    """Build train/val/test splits and a unified ML dataset from a collector output directory."""
    if sum(splits) != 1.0:
        raise ValueError("splits must sum to 1.0")

    sources: list[tuple[str, list[dict[str, Any]]]] = []
    deep_dir = dataset_dir / "deep"
    raw_dir = dataset_dir / "raw"
    if deep_dir.exists():
        for path in sorted(deep_dir.glob("*_deep.jsonl")):
            sources.append((path.name, _read_jsonl_rows(path)))
    if not deep_only and raw_dir.exists():
        for path in sorted(raw_dir.glob("*.jsonl")):
            if path.name.endswith("_deep.jsonl"):
                continue
            sources.append((path.name, _read_jsonl_rows(path)))

    samples: list[dict[str, Any]] = []
    seen_text: set[str] = set()
    sample_index_by_id: dict[str, int] = {}
    for source_name, rows in sources:
        is_deep = "_deep.jsonl" in source_name
        for row in rows:
            sample = _build_sample(row)
            if sample is None:
                continue
            existing_index = sample_index_by_id.get(sample["sample_id"])
            if existing_index is not None:
                # Prefer deep record over raw record for the same content.
                if is_deep:
                    sample["source_file"] = source_name
                    samples[existing_index] = sample
                continue
            if len(sample["text"]) < min_text_length:
                continue
            if dedup_text:
                normalized = re.sub(r"\s+", "", sample["text"])
                if normalized in seen_text:
                    continue
                seen_text.add(normalized)
            sample["source_file"] = source_name
            sample_index_by_id[sample["sample_id"]] = len(samples)
            samples.append(sample)

    rng = random.Random(seed)
    rng.shuffle(samples)

    n = len(samples)
    train_end = int(n * splits[0])
    val_end = train_end + int(n * splits[1])
    for idx, sample in enumerate(samples):
        if idx < train_end:
            sample["split"] = "train"
        elif idx < val_end:
            sample["split"] = "val"
        else:
            sample["split"] = "test"

    embeddings_list: list[list[float]] | None = None
    if embeddings:
        embeddings_list = _compute_embeddings([s["text"] for s in samples], model_name=embeddings)
        if embeddings_list is not None:
            for sample, vector in zip(samples, embeddings_list):
                sample["embedding"] = vector

    out_dir = dataset_dir / "ml"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write JSONL
    jsonl_path = out_dir / "dataset.jsonl"
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # Write CSV
    csv_path = out_dir / "dataset.csv"
    _write_csv(csv_path, samples)

    # Write metadata
    metadata = {
        "dataset_dir": str(dataset_dir),
        "total_samples": len(samples),
        "split_counts": {"train": sum(1 for s in samples if s["split"] == "train"), "val": sum(1 for s in samples if s["split"] == "val"), "test": sum(1 for s in samples if s["split"] == "test")},
        "platform_counts": _count_by_key(samples, "platform"),
        "keyword_counts": _count_by_key(samples, "keyword"),
        "deep_only": deep_only,
        "dedup_text": dedup_text,
        "min_text_length": min_text_length,
        "embeddings": embeddings,
        "embedding_dim": len(embeddings_list[0]) if embeddings_list else None,
        "fields": list(samples[0].keys()) if samples else [],
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"samples": samples, "metadata": metadata, "out_dir": out_dir}


def _count_by_key(samples: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        value = str(sample.get(key, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return counts


def _write_csv(path: Path, samples: list[dict[str, Any]]) -> None:
    import csv

    if not samples:
        path.write_text("", encoding="utf-8-sig")
        return
    # Flatten list/dict fields for CSV.
    normalized = []
    for sample in samples:
        row: dict[str, Any] = {}
        for key, value in sample.items():
            if isinstance(value, (dict, list)):
                row[key] = json.dumps(value, ensure_ascii=False)
            else:
                row[key] = value
        normalized.append(row)
    fields = sorted({key for row in normalized for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(normalized)


def _compute_embeddings(texts: list[str], model_name: str) -> list[list[float]] | None:
    """Compute sentence embeddings if sentence-transformers is available."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("文本嵌入需要 sentence-transformers，请执行 pip install sentence-transformers") from exc

    model = SentenceTransformer(model_name)
    return [vector.tolist() for vector in model.encode(texts, show_progress_bar=True)]


def main() -> int:
    try:
        import sys

        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="从采集结果构建 ML-ready 数据集")
    parser.add_argument("dataset_dir", nargs="?", help="采集输出目录，默认使用最新的 data/datasets 子目录")
    parser.add_argument("--data-root", default="./data", help="项目 data 根目录（用于自动查找最新数据集）")
    parser.add_argument("--deep-only", action="store_true", help="只使用 deep 目录的数据")
    parser.add_argument("--no-dedup-text", action="store_true", help="不对 text 做归一化去重")
    parser.add_argument("--min-text-length", type=int, default=10, help="text 最小字符数")
    parser.add_argument("--embeddings", default=None, help="sentence-transformers 模型名，如 paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--seed", type=int, default=42, help="划分随机种子")
    parser.add_argument("--split", default="0.8,0.1,0.1", help="训练/验证/测试比例")
    args = parser.parse_args()

    dataset_dir: Path
    if args.dataset_dir:
        dataset_dir = Path(args.dataset_dir)
    else:
        candidates = _find_dataset_dirs(Path(args.data_root))
        if not candidates:
            print("未找到任何数据集目录", file=sys.stderr)
            return 1
        dataset_dir = candidates[0]
        print(f"使用最新数据集: {dataset_dir}")

    splits = tuple(float(part) for part in args.split.split(","))
    if len(splits) != 3:
        print("--split 必须是三个数字，如 0.8,0.1,0.1", file=sys.stderr)
        return 1

    result = build_ml_dataset(
        dataset_dir,
        deep_only=args.deep_only,
        dedup_text=not args.no_dedup_text,
        min_text_length=args.min_text_length,
        embeddings=args.embeddings,
        seed=args.seed,
        splits=splits,  # type: ignore[arg-type]
    )
    metadata = result["metadata"]
    print(f"已生成 ML 数据集: {result['out_dir']}")
    print(f"总样本: {metadata['total_samples']}, train={metadata['split_counts']['train']}, val={metadata['split_counts']['val']}, test={metadata['split_counts']['test']}")
    print(f"平台分布: {metadata['platform_counts']}")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main())
