"""Douyin hot-search board collector (keyless).

``douyin`` search is unusable without forging its ``a_bogus`` request signature
(the API answers ``search_nil_type=verify_check`` even with a valid login
Cookie). This collector covers the platform through endpoint that anonymous
callers can still reach, the same shape as the 百度热搜 collector: it is a
billboard, not a search engine, so keywords only filter the entries.

``DOUYIN_HOT_MATCH_ALL=true`` archives the whole board regardless of keyword.
"""

from __future__ import annotations

from typing import Any

from exceptions import CollectorError
from models import ContentItem, DeepContent, Engagement
from utils import clean_html, iso_from_timestamp, safe_int

from .base import BaseCollector, ProgressCallback

# 首选：网页版热点榜，字段最全（视频数、讨论数、事件时间、封面），无需签名。
WEB_BOARD_URL = "https://www.douyin.com/aweme/v1/web/hot/search/list/"
WEB_BOARD_PARAMS = {"device_platform": "webapp", "aid": 6383, "channel": "channel_pc_web"}
# 备用：iesdouyin 榜单开放接口，字段较少但同样免签名。
BILLBOARD_URL = "https://www.iesdouyin.com/web/api/v2/hotsearch/billboard/word/"

# 兼容旧引用
BOARD_URL = BILLBOARD_URL


class DouyinHotCollector(BaseCollector):
    name = "douyin_hot"

    def __init__(self, settings: Any) -> None:
        super().__init__(settings)
        self._board_cache: list[dict[str, Any]] | None = None
        self._mismatch_reported = False

    def _credential_headers(self) -> dict[str, str]:
        # 网页版热点榜接口会检查来源；不带 Referer 会返回 HTML 而非 JSON。
        return {"Referer": "https://www.douyin.com/", "Accept": "application/json, text/plain, */*"}

    def _board(self) -> list[dict[str, Any]]:
        """Fetch the board once per run, preferring the richer web endpoint.

        The board is the same for every keyword, so caching it avoids one HTTP
        call per query variant (which adds up over a multi-keyword run).
        """
        if self._board_cache is not None:
            return self._board_cache
        self._board_cache = self._fetch_board()
        return self._board_cache

    def _fetch_board(self) -> list[dict[str, Any]]:
        try:
            payload = self.get_json(WEB_BOARD_URL, params=dict(WEB_BOARD_PARAMS))
            words = (payload.get("data") or {}).get("word_list") or []
            if isinstance(words, list) and words:
                return [row for row in words if isinstance(row, dict)]
        except Exception as exc:
            self.log.warning("抖音网页版热点榜不可用（%s），改用备用榜单接口", exc)
        payload = self.get_json(BILLBOARD_URL)
        words = payload.get("word_list") or []
        return [row for row in words if isinstance(row, dict)] if isinstance(words, list) else []

    def search(self, keyword: str, progress: ProgressCallback | None = None) -> list[ContentItem]:
        words = self._board()
        if not words:
            raise CollectorError("抖音热点榜没有返回任何条目（接口可能已变更）")
        match_all = bool(self.settings.douyin_hot_match_all)
        items: list[ContentItem] = []
        for rank, row in enumerate(words, 1):
            word = clean_html(str(row.get("word") or ""))
            if not word:
                continue
            if not match_all and keyword.strip() and keyword.strip() not in word:
                continue
            hot = safe_int(row.get("hot_value"))
            cover = row.get("word_cover") or {}
            cover_url = ""
            if isinstance(cover, dict):
                urls = cover.get("url_list") or []
                cover_url = str(urls[0]) if urls else str(cover.get("uri") or "")
            items.append(
                ContentItem(
                    id=f"douyin_hot_{row.get('sentence_id') or row.get('group_id') or rank}_{word}"[:120],
                    platform=self.name,
                    keyword=keyword,
                    content_type="hotword",
                    title=word,
                    summary=word,
                    author="",
                    # 榜单项没有发布时间，用榜单统计时间保持可排序。
                    published_at=iso_from_timestamp(row.get("event_time")),
                    # 榜单没有点赞/评论，只有热度值；映射到 views 以参与互动初筛。
                    engagement=Engagement(
                        views=hot,
                        comments=safe_int(row.get("discuss_video_count")),
                    ),
                    url=f"https://www.douyin.com/search/{word}",
                    extra={
                        "rank": safe_int(row.get("position")) or rank,
                        "hot_value": hot,
                        "label": row.get("label"),
                        "video_count": safe_int(row.get("video_count")),
                        "discuss_video_count": safe_int(row.get("discuss_video_count")),
                        "sentence_id": row.get("sentence_id"),
                        "group_id": row.get("group_id"),
                        "cover_url": cover_url,
                        "source": "douyin-hot-board",
                    },
                )
            )
        if progress:
            progress(self.name, 1, 1, f"{keyword}: {len(items)} 条")
        if not items and not self._mismatch_reported:
            # 命中 0 条只是因为这个关键词没上榜单，不是采集失败。每次运行只报一次，
            # 否则多关键词 × 扩展词会把报告刷满重复条目。
            self._mismatch_reported = True
            self.note_partial(
                "search",
                f"抖音热点榜当前 {len(words)} 条热点中没有匹配「{keyword}」的条目"
                "（这是榜单不是搜索引擎，可设 DOUYIN_HOT_MATCH_ALL=true 全量归档）",
                blocking=False,
            )
        return items

    def deep_dive(self, item: ContentItem) -> DeepContent:
        # 榜单条目本身没有正文，能提供的只有词条与热度统计。
        rank = item.extra.get("rank")
        full = (
            f"抖音热榜第 {rank} 位：{item.title}"
            f"（热度 {item.extra.get('hot_value')}，相关视频 {item.extra.get('video_count')} 条，"
            f"参与讨论 {item.extra.get('discuss_video_count')} 条）"
        )
        return DeepContent(base=item, full_content=full, comments=[])


__all__ = ["DouyinHotCollector", "WEB_BOARD_URL", "BILLBOARD_URL", "BOARD_URL"]

