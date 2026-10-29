"""Unified data models shared by collectors, storage and analysis."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class Engagement:
    likes: int = 0
    comments: int = 0
    shares: int = 0
    views: int = 0

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "Engagement":
        value = value or {}
        return cls(**{key: int(value.get(key, 0) or 0) for key in cls.__annotations__})


@dataclass(slots=True)
class ContentItem:
    id: str
    platform: str
    keyword: str
    content_type: str
    title: str = ""
    summary: str = ""
    author: str = ""
    published_at: str = ""
    engagement: Engagement = field(default_factory=Engagement)
    url: str = ""
    fetched_at: str = field(default_factory=utc_now)
    deep_dive_flag: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ContentItem":
        known = {name for name in cls.__annotations__}
        payload = {key: val for key, val in value.items() if key in known}
        payload["engagement"] = Engagement.from_dict(payload.get("engagement"))
        return cls(**payload)

    @property
    def engagement_score(self) -> float:
        e = self.engagement
        return e.likes + e.comments * 2 + e.shares * 3 + e.views * 0.001


@dataclass(slots=True)
class Comment:
    text: str
    likes: int = 0
    author: str = ""
    published_at: str = ""


@dataclass(slots=True)
class DeepContent:
    base: ContentItem
    full_content: str = ""
    comments: list[Comment] = field(default_factory=list)
    media_text: str = ""
    sentiment_score: float = 0.0
    keywords_extracted: list[str] = field(default_factory=list)
    content_length: int = 0
    deep_fetched_at: str = field(default_factory=utc_now)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = self.base.to_dict()
        payload["deep_dive_flag"] = True
        payload.update(
            {
                "full_content": self.full_content,
                "comments": [asdict(comment) for comment in self.comments],
                "media_text": self.media_text,
                "sentiment_score": self.sentiment_score,
                "keywords_extracted": self.keywords_extracted,
                "content_length": self.content_length or len(self.full_content),
                "deep_fetched_at": self.deep_fetched_at,
                "errors": self.errors,
            }
        )
        canonical = json.dumps(
            {
                "full_content": payload["full_content"],
                "comments": payload["comments"],
                "media_text": payload["media_text"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        payload["content_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return payload

