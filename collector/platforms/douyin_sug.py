"""Douyin search-suggestion collector (keyless).

Douyin's *content* search needs a forged ``a_bogus`` signature, but its
search-suggestion endpoint answers anonymous callers. The suggestions are what
real users actually type into Douyin, which makes this a good source for
research on how a policy is phrased and asked about by the public — e.g.
「低保一人每月多少钱」「危房改造归哪个部门管」— rather than for media content.

Each row is a search phrase, not a post: it has no author, no timestamp and no
engagement. Treat it as a terminology / search-intent corpus, not as content.
"""

from __future__ import annotations

from typing import Any

from models import ContentItem, DeepContent, Engagement
from utils import clean_html

from .base import BaseCollector, ProgressCallback

SUG_URL = "https://www.douyin.com/aweme/v1/web/search/sug/"
SUG_PARAMS = {"device_platform": "webapp", "aid": 6383, "channel": "channel_pc_web"}


class DouyinSugCollector(BaseCollector):
    name = "douyin_sug"

    def _credential_headers(self) -> dict[str, str]:
        return {"Referer": "https://www.douyin.com/", "Accept": "application/json, text/plain, */*"}

    @staticmethod
    def _word(row: dict[str, Any]) -> str:
        record = row.get("word_record") or {}
        for value in (row.get("content"), row.get("word"), record.get("words_content")):
            if value:
                return clean_html(str(value))
        return ""

    def search(self, keyword: str, progress: ProgressCallback | None = None) -> list[ContentItem]:
        payload = self.get_json(SUG_URL, params={**SUG_PARAMS, "keyword": keyword})
        rows = payload.get("sug_list") or []
        if not isinstance(rows, list):
            rows = []
        items: list[ContentItem] = []
        for rank, row in enumerate(rows, 1):
            if not isinstance(row, dict):
                continue
            word = self._word(row)
            if not word:
                continue
            record = row.get("word_record") or {}
            items.append(
                ContentItem(
                    id=f"douyin_sug_{keyword}_{rank}"[:120],
                    platform=self.name,
                    keyword=keyword,
                    content_type="suggestion",
                    title=word,
                    summary=word,
                    author="",
                    published_at="",
                    # 联想词没有互动数据，只有搜索位次。
                    engagement=Engagement(),
                    url=f"https://www.douyin.com/search/{word}",
                    extra={
                        "suggestion_for": keyword,
                        "suggestion_rank": rank,
                        "group_id": record.get("group_id"),
                        "words_source": record.get("words_source"),
                        "source": "douyin-search-suggestion",
                        "is_search_intent": True,
                    },
                )
            )
        if progress:
            progress(self.name, 1, 1, f"{keyword}: {len(items)} 条")
        if not items:
            # 冷门词没有联想词，不是采集失败。
            self.note_partial(
                "search",
                f"抖音搜索联想词为空：没有用户搜过「{keyword}」（这本身就是一条研究信息）",
                blocking=False,
            )
        return items

    def deep_dive(self, item: ContentItem) -> DeepContent:
        # 联想词没有正文可深挖。
        return DeepContent(
            base=item,
            full_content=f"抖音搜索联想词：{item.title}（针对「{item.extra.get('suggestion_for')}」）",
            comments=[],
        )


__all__ = ["DouyinSugCollector", "SUG_URL"]
