"""Douban search-page collector (books / movies / music subjects).

Douban has no public search API, so this parses the public search result page
(server-rendered HTML, no login required for basic queries). Subject metadata
(title, rating, abstract) comes from the search page; deep dives fetch the
subject page and extract the introduction blocks. Be polite: douban rate-limits
aggressively, keep delays at or above the defaults.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import unquote

from exceptions import CollectorError
from models import ContentItem, DeepContent, Engagement
from utils import clean_html, safe_int

from .base import BaseCollector, ProgressCallback

BLOCK_RE = re.compile(r"<div class=\"result\b", re.I)
# 出现这些标记说明是风控/验证页，而不是「这个词条没有结果」。
BLOCK_MARKERS = ("captcha", "安全验证", "访问过于频繁", "登录", "robot", "forbidden")
SID_RE = re.compile(r"sid: (\d+)")
KIND_RE = re.compile(r"<span>\[(.*?)\]</span>")
TITLE_RE = re.compile(r"<h3>\s*<span>.*?</span>.*?<a [^>]*>(.*?)</a>", re.S)
RATING_RE = re.compile(r"class=\"rating_nums\">\s*([\d.]+)")
LINK2_RE = re.compile(r"link2/\?url=([^&\"]+)")


class DoubanCollector(BaseCollector):
    name = "douban"
    search_url = "https://www.douban.com/search"

    def __init__(self, settings: Any) -> None:
        super().__init__(settings)
        self.cat = (getattr(settings, "douban_cat", "") or "").strip()

    def _convert(self, block: str, keyword: str) -> ContentItem | None:
        sid_match = SID_RE.search(block)
        if not sid_match:
            return None
        sid = sid_match.group(1)
        kind_match = KIND_RE.search(block)
        title_match = TITLE_RE.search(block)
        title = clean_html(title_match.group(1)) if title_match else ""
        rating_match = RATING_RE.search(block)
        rating = rating_match.group(1) if rating_match else ""
        link_match = LINK2_RE.search(block)
        url = ""
        if link_match:
            url = unquote(link_match.group(1))
        if not title and not url:
            return None
        if not url:
            url = f"https://www.douban.com/subject/{sid}/"
        # 摘要：评分之后的纯文本区域（作者/出版社/简介）
        summary = ""
        if rating_match:
            tail = clean_html(block[rating_match.end() : rating_match.end() + 2000])
            summary = " ".join(tail.split())[:400]
        return ContentItem(
            id=sid,
            platform="douban",
            keyword=keyword,
            content_type="subject",
            title=title,
            summary=summary[:1000],
            author="",
            published_at="",
            engagement=Engagement(likes=int(float(rating) * 10) if rating else 0),
            url=url,
            extra={
                "kind": clean_html(kind_match.group(1)) if kind_match else "",
                "rating": float(rating) if rating else 0.0,
            },
        )

    def search(self, keyword: str, progress: ProgressCallback | None = None) -> list[ContentItem]:
        params: dict[str, Any] = {"q": keyword}
        if self.cat:
            params["cat"] = self.cat
        response = self.request("GET", self.search_url, params=params)
        text = response.text
        if "captcha" in text[:10_000].lower() and BLOCK_RE.search(text) is None:
            saved = self._debug_challenge(str(response.url), text, "douban captcha")
            raise CollectorError(f"豆瓣触发验证码，诊断已保存至 {saved}")
        items: list[ContentItem] = []
        blocks = BLOCK_RE.split(text)[1:]
        for block in blocks:
            item = self._convert(block.split('<div class="result"', 1)[0], keyword)
            if item:
                items.append(item)
        if not items:
            # 豆瓣对「这个词条没有结果」和「被风控 / 页面结构变更」都返回一个没有
            # result 块的页面。这里不再一律抛错（否则连续几个冷门关键词就会被误判
            # 成平台故障），但也不静默吞掉：记录一条非阻塞的说明，交给报告呈现。
            suspicious = any(marker in text[:20_000].lower() for marker in BLOCK_MARKERS)
            self.note_partial(
                "search",
                "豆瓣搜索没有解析到结果"
                + ("（页面含风控/验证标记，更可能是被拦截）" if suspicious else "（该关键词可能确实没有条目）")
                + "；可稍后重试、降低频率，或更换 DOUBAN_CAT（1001 图书 / 1002 电影 / 1003 音乐）",
                blocking=suspicious,
            )
        if progress:
            progress("douban", 1, 1, f"{keyword}: {len(items)} 条")
        return list({item.id: item for item in items}.values())

    def deep_dive(self, item: ContentItem) -> DeepContent:
        errors: list[str] = []
        full_content = item.summary
        try:
            response = self.request("GET", item.url, check_challenge=False)
            text = response.text
            intros = re.findall(r'<div class="intro">(.*?)</div>', text, re.S)
            body = "\n".join(clean_html(part) for part in intros if clean_html(part))
            if body:
                full_content = body[:20_000]
        except CollectorError:
            raise
        except Exception as exc:
            errors.append(f"subject: {exc}")
        return DeepContent(base=item, full_content=full_content, comments=[], errors=errors)
