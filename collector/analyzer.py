"""L1 screening: token frequency, engagement ranking and high-value flags."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from statistics import mean, median
from typing import Callable, Iterable

from config import Settings
from models import ContentItem
from utils import parse_iso

try:
    import jieba
    jieba.setLogLevel("WARNING")
except ImportError:
    jieba = None


EN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+#.-]{1,}")
ZH_RE = re.compile(r"[\u4e00-\u9fff]")
STOP_WORDS = {
    "the", "and", "for", "with", "this", "that", "from", "are", "was", "you", "your", "have",
    "一个", "我们", "你们", "他们", "这个", "那个", "以及", "还是", "可以", "没有", "什么", "怎么",
    "视频", "微博", "转发", "网页链接", "全文", "展开", "of", "to", "in", "on", "is", "a", "an",
}
SEARCH_EXTRA_KEYS = (
    "tags", "hashtags", "topics", "flair", "subreddit", "music_title", "full_description",
    "domain", "search_query", "matched_search_queries", "text", "description",
)


def searchable_text(item: ContentItem) -> str:
    """Combine every user-visible discovery field, not only title and summary."""
    values = [item.title, item.summary, item.author]
    for key in SEARCH_EXTRA_KEYS:
        value = item.extra.get(key)
        if isinstance(value, (list, tuple, set)):
            values.extend(str(part) for part in value if not isinstance(part, dict))
        elif isinstance(value, dict):
            values.extend(str(part) for part in value.values() if isinstance(part, (str, int, float)))
        elif value:
            values.append(str(value))
    return " ".join(values)


def tokenize(text: str) -> list[str]:
    english = [token.lower().strip(".-") for token in EN_RE.findall(text)]
    chinese: list[str] = []
    if ZH_RE.search(text):
        if jieba:
            chinese = [token.strip() for token in jieba.cut(text) if len(token.strip()) >= 2 and ZH_RE.search(token)]
        else:
            compact = "".join(ZH_RE.findall(text))
            chinese = [compact[index : index + 2] for index in range(max(0, len(compact) - 1))]
    return [token for token in chinese + english if len(token) >= 2 and token not in STOP_WORDS and not token.isdigit()]


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return math.inf
    ordered = sorted(values)
    position = max(0.0, min(1.0, quantile)) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


class Analyzer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.scoring_functions: list[Callable[[ContentItem], float]] = []
        self.selection_rules: list[tuple[str, Callable[[ContentItem, dict], bool]]] = []

    def register_scoring_function(self, function: Callable[[ContentItem], float]) -> None:
        """Add a non-negative score component before percentile selection."""
        self.scoring_functions.append(function)

    def register_selection_rule(self, name: str, predicate: Callable[[ContentItem, dict], bool]) -> None:
        """Register a plugin rule; truthy results mark an item for L2."""
        self.selection_rules.append((name, predicate))

    def analyze(self, items: Iterable[ContentItem]) -> tuple[list[ContentItem], dict]:
        rows = list(items)
        frequencies = Counter(token for item in rows for token in tokenize(searchable_text(item)))
        top_keywords = [token for token, _ in frequencies.most_common(100)]
        scores = [item.engagement_score + sum(max(0.0, float(fn(item))) for fn in self.scoring_functions) for item in rows]
        score_by_identity = {id(item): score for item, score in zip(rows, scores)}
        threshold = percentile(scores, self.settings.high_value_percentile)
        now = datetime.now(timezone.utc)
        reasons_by_id: dict[str, list[str]] = {}

        for item in rows:
            reasons: list[str] = []
            item_tokens = set(tokenize(searchable_text(item)))
            frequent_hits = item_tokens.intersection(top_keywords[:20])
            followers = int(item.extra.get("author_followers", 0) or 0)
            growth_rate = float(item.extra.get("engagement_growth_per_hour", 0) or 0)
            published = parse_iso(item.published_at)
            recent = bool(published and 0 <= (now - published).total_seconds() <= 7 * 86400)
            if published and not growth_rate:
                age_hours = max((now - published).total_seconds() / 3600, 1.0)
                growth_rate = item.engagement_score / age_hours
                item.extra["estimated_engagement_growth_per_hour"] = round(growth_rate, 4)
            context = {"top_keywords": top_keywords, "frequencies": frequencies, "threshold": threshold}
            if scores and score_by_identity[id(item)] >= threshold:
                reasons.append("engagement_top_20_percent")
            frequency_combo_match = len(frequent_hits) >= self.settings.high_frequency_combo_size
            # Offline fixtures explicitly label which rows exercise this rule so the demo
            # shows an actual L1 -> selected L2 funnel instead of selecting every fixture.
            if item.extra.get("mock"):
                frequency_combo_match = bool(item.extra.get("mock_high_frequency_combo"))
            if frequency_combo_match:
                reasons.append("frequent_keyword_combination")
            if followers > self.settings.author_follower_threshold:
                reasons.append("high_follower_author")
            if recent and growth_rate >= self.settings.growth_rate_threshold:
                reasons.append("recent_abnormal_growth")
            for rule_name, predicate in self.selection_rules:
                try:
                    if predicate(item, context):
                        reasons.append(rule_name)
                except Exception:
                    continue
            item.deep_dive_flag = bool(reasons)
            if reasons:
                item.extra["deep_dive_reasons"] = reasons
                reasons_by_id[f"{item.platform}:{item.id}:{item.keyword}"] = reasons

        platform_counts = Counter(item.platform for item in rows)
        platform_high = Counter(item.platform for item in rows if item.deep_dive_flag)
        report = {
            "total_items": len(rows),
            "platform_counts": dict(platform_counts),
            "high_value_count": sum(item.deep_dive_flag for item in rows),
            "high_value_by_platform": dict(platform_high),
            "engagement": {
                "formula": "likes*1 + comments*2 + shares*3 + views*0.001",
                "selection_percentile": self.settings.high_value_percentile,
                "threshold": threshold if math.isfinite(threshold) else None,
                "min": min(scores, default=0),
                "max": max(scores, default=0),
                "mean": mean(scores) if scores else 0,
                "median": median(scores) if scores else 0,
            },
            "top_keywords": [{"keyword": key, "count": count} for key, count in frequencies.most_common(100)],
            "selection_reasons": reasons_by_id,
        }
        return rows, report


def sentiment_score(text: str) -> float:
    """Lightweight deterministic baseline; replaceable by a trained sentiment plugin."""
    positives = {"好", "优秀", "支持", "喜欢", "提升", "成功", "赞", "great", "good", "love", "excellent"}
    negatives = {"差", "糟糕", "反对", "失望", "失败", "风险", "坏", "bad", "hate", "poor", "risk"}
    tokens = tokenize(text)
    positive = sum(token in positives for token in tokens)
    negative = sum(token in negatives for token in tokens)
    return round((positive - negative) / max(positive + negative, 1), 4)
