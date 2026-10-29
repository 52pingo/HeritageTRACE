"""Collector registry."""

from __future__ import annotations

from config import Settings

from .base import BaseCollector


def create_collectors(settings: Settings, names: list[str], mock: bool = False) -> dict[str, BaseCollector]:
    if mock:
        from .mock import MockCollector

        return {name: MockCollector(settings, platform_name=name) for name in names}

    from .bilibili import BilibiliCollector
    from .weibo import WeiboCollector
    from .x import XCollector
    from .youtube import YouTubeCollector
    from .douyin import DouyinCollector
    from .douyin_hot import DouyinHotCollector
    from .douyin_sug import DouyinSugCollector
    from .reddit import RedditCollector
    from .hackernews import HackerNewsCollector
    from .bluesky import BlueskyCollector
    from .tieba import TiebaCollector
    from .zhihu import ZhihuCollector
    from .juejin import JuejinCollector
    from .csdn import CSDNCollector
    from .rss import RSSCollector
    from .arxiv import ArxivCollector
    from .stackexchange import StackExchangeCollector
    from .wikipedia import WikipediaCollector
    from .github import GitHubCollector
    from .v2ex import V2EXCollector
    from .baidu import BaiduHotCollector
    from .douban import DoubanCollector

    classes = {
        "bilibili": BilibiliCollector,
        "weibo": WeiboCollector,
        "x": XCollector,
        "youtube": YouTubeCollector,
        "douyin": DouyinCollector,
        "douyin_hot": DouyinHotCollector,
        "douyin_sug": DouyinSugCollector,
        "reddit": RedditCollector,
        "hackernews": HackerNewsCollector,
        "bluesky": BlueskyCollector,
        "tieba": TiebaCollector,
        "zhihu": ZhihuCollector,
        "juejin": JuejinCollector,
        "csdn": CSDNCollector,
        "rss": RSSCollector,
        "arxiv": ArxivCollector,
        "stackexchange": StackExchangeCollector,
        "wikipedia": WikipediaCollector,
        "github": GitHubCollector,
        "v2ex": V2EXCollector,
        "baidu": BaiduHotCollector,
        "douban": DoubanCollector,
    }
    return {name: classes[name](settings) for name in names}


__all__ = ["BaseCollector", "create_collectors"]
