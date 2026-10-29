"""Deterministic offline collector used for demos and tests."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

from models import Comment, ContentItem, DeepContent, Engagement

from .base import BaseCollector, ProgressCallback


class MockCollector(BaseCollector):
    def search(self, keyword: str, progress: ProgressCallback | None = None) -> list[ContentItem]:
        nouns = ["趋势观察", "产业报告", "用户讨论", "技术解析", "风险与机会", "一周复盘"]
        now = datetime.now(timezone.utc)
        items: list[ContentItem] = []
        count = min(self.config.page_size, 12)
        for index in range(count):
            digest = hashlib.sha1(f"{self.name}:{keyword}:{index}".encode()).hexdigest()[:12]
            base = (index + 1) * (len(keyword) + len(self.name) + 3)
            content_type = "video" if self.name in {"bilibili", "youtube"} else "tweet" if self.name == "x" else "post"
            items.append(
                ContentItem(
                    id=f"mock-{digest}",
                    platform=self.name,
                    keyword=keyword,
                    content_type=content_type,
                    title=f"{keyword}：{nouns[index % len(nouns)]} #{index + 1}",
                    summary=f"这是一条用于演示完整采集、初筛和深度分析流程的 {self.name} mock 数据，讨论 {keyword} 的影响。",
                    author=f"mock_author_{index % 4}",
                    published_at=(now - timedelta(hours=index * 7)).isoformat(),
                    engagement=Engagement(
                        likes=base * 13,
                        comments=base * 3,
                        shares=base * 2,
                        views=base * 1200 if content_type == "video" else base * 80,
                    ),
                    url=f"https://example.invalid/{self.name}/{digest}",
                    extra={
                        "author_followers": 150_000 if index == 1 else base * 200,
                        "engagement_growth_per_hour": 140 if index == 2 else base / 2,
                        "mock": True,
                        "mock_high_frequency_combo": index == 3,
                    },
                )
            )
            if progress:
                progress(self.name, index + 1, count, f"mock {keyword}")
        return items

    def deep_dive(self, item: ContentItem) -> DeepContent:
        content = (
            f"{item.title}。这是 {item.platform} 上围绕“{item.keyword}”的模拟完整内容。"
            "样本包含背景、观点、支持意见、潜在风险与后续趋势，适合验证机器学习数据管线。"
        )
        return DeepContent(
            base=item,
            full_content=content,
            comments=[
                Comment(text=f"我支持继续关注{item.keyword}，信息总结得很好。", likes=28, author="mock_reader_a"),
                Comment(text="需要警惕样本偏差和隐私风险。", likes=17, author="mock_reader_b"),
                Comment(text="期待后续数据和更完整的对比。", likes=9, author="mock_reader_c"),
            ],
            media_text=f"OCR mock: {item.keyword} / public research sample",
        )
