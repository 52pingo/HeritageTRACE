"""Reddit search adapter (OAuth).

Reddit no longer serves the unauthenticated ``*.json`` endpoints: ``www.reddit.com``
answers 403 and ``old.reddit.com`` redirects to the login page. Collection therefore
uses the official OAuth API, which needs a free "script" app
(``REDDIT_CLIENT_ID`` + ``REDDIT_CLIENT_SECRET``).
"""

from __future__ import annotations

import html as html_module
import re
import time
from typing import Any

from exceptions import AuthenticationRequired, CollectorError
from media_tools import ocr_image_url
from models import Comment, ContentItem, DeepContent, Engagement
from utils import clean_html, iso_from_timestamp, safe_int

from .base import BaseCollector, ProgressCallback

DEFAULT_UA = "windows:social-sentiment-research:v1.0 (by /u/researcher)"


def _text_of(block: str, tag: str) -> str:
    """Inner text of the first <tag>…</tag> in an Atom entry."""
    match = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", block, re.S)
    return match.group(1).strip() if match else ""


def _link_of(entry: str) -> str:
    """The <link href=…> Reddit puts on each entry."""
    match = re.search(r'<link[^>]*href="([^"]+)"', entry, re.S)
    return html_module.unescape(match.group(1)) if match else ""


def _contributor_category(entry: str) -> str:
    """The subreddit, carried on the entry's first <category term=…>."""
    match = re.search(r'<category[^>]*term="([^"]*)"', entry, re.S)
    return html_module.unescape(match.group(1)) if match else ""


class RedditCollector(BaseCollector):
    name = "reddit"
    foreign = True
    token_url = "https://www.reddit.com/api/v1/access_token"
    oauth_root = "https://oauth.reddit.com"

    def __init__(self, settings: Any) -> None:
        super().__init__(settings)
        self._token = ""
        self._token_expires_at = 0.0
        self.has_oauth = bool(settings.reddit_client_id and settings.reddit_client_secret)
        if not self.has_oauth:
            self.log.warning(
                "Reddit 未配置 OAuth 凭据，改用公开 RSS 搜索：可以拿到标题/正文/作者/时间，"
                "但**没有**点赞数与评论数，且请求频率受限。配置 REDDIT_CLIENT_ID/SECRET 可启用完整 API"
            )
        # Reddit's API requires one stable, descriptive User-Agent.
        self._user_agents = [self.user_agent]

    def _credential_headers(self) -> dict[str, str]:
        return {"Authorization": f"bearer {self._token}"} if self._token else {}

    @property
    def user_agent(self) -> str:
        return self.settings.reddit_user_agent or DEFAULT_UA

    def _login(self) -> None:
        """Fetch an application-only OAuth token (valid ~1h)."""
        if not self.has_oauth:
            raise AuthenticationRequired(
                "Reddit 需要 OAuth 凭据才能调用官方 API：请在 .env 配置 REDDIT_CLIENT_ID 与 "
                "REDDIT_CLIENT_SECRET（在 reddit.com/prefs/apps 创建 script 类型应用）"
            )
        if self._token and time.monotonic() < self._token_expires_at:
            return
        response = self.request(
            "POST",
            self.token_url,
            data={"grant_type": "client_credentials"},
            auth=(self.settings.reddit_client_id, self.settings.reddit_client_secret),
            headers={"User-Agent": self.user_agent},
        )
        payload = self._decode_json(response)
        token = str(payload.get("access_token") or "")
        if not token:
            raise AuthenticationRequired(f"Reddit OAuth 未返回 access_token: {payload}")
        self._token = token
        self._token_expires_at = time.monotonic() + max(int(payload.get("expires_in", 3600)) - 60, 60)
        # The Authorization header itself is re-applied per request from
        # _credential_headers(), so nothing else to sync here.

    def search(self, keyword: str, progress: ProgressCallback | None = None) -> list[ContentItem]:
        if not self.has_oauth:
            # www.reddit.com/*.json is IP-blocked and old.reddit.com redirects to
            # the login page, but the RSS feed is still served.
            return self._search_rss(keyword, progress)
        self._login()
        items: list[ContentItem] = []
        after: str | None = None
        for page in range(1, self.config.pages + 1):
            params: dict[str, Any] = {
                "q": keyword,
                "sort": "relevance",
                "t": "all",
                "limit": min(self.config.page_size, 100),
                "raw_json": 1,
            }
            if after:
                params["after"] = after
            try:
                payload = self.get_json(f"{self.oauth_root}/search.json", params=params)
            except Exception as exc:
                if items:
                    self.note_partial("search", exc)
                    break
                raise
            children = ((payload.get("data") or {}).get("children") or [])
            for child in children:
                row = child.get("data") or {}
                post_id = str(row.get("id", ""))
                if not post_id:
                    continue
                preview = row.get("preview") or {}
                images = [
                    str(source.get("url", "")).replace("&amp;", "&")
                    for image in preview.get("images") or []
                    for source in [image.get("source") or {}]
                    if source.get("url")
                ]
                items.append(
                    ContentItem(
                        id=post_id,
                        platform=self.name,
                        keyword=keyword,
                        content_type="post",
                        title=clean_html(str(row.get("title", ""))),
                        summary=clean_html(str(row.get("selftext", "")))[:1000],
                        author=str(row.get("author", "")),
                        published_at=iso_from_timestamp(row.get("created_utc")),
                        engagement=Engagement(
                            likes=safe_int(row.get("score")),
                            comments=safe_int(row.get("num_comments")),
                            shares=0,
                            views=0,
                        ),
                        url=f"https://www.reddit.com{row.get('permalink', '')}",
                        extra={
                            "subreddit": row.get("subreddit"),
                            "tags": [str(row.get("link_flair_text"))] if row.get("link_flair_text") else [],
                            "flair": row.get("link_flair_text"),
                            "picture_urls": images,
                            "domain": row.get("domain"),
                            "is_video": bool(row.get("is_video")),
                        },
                    )
                )
            if progress:
                progress(self.name, page, self.config.pages, f"{keyword}: {len(items)} 条")
            after = (payload.get("data") or {}).get("after")
            if not after or not children:
                break
        return list({(item.id, item.keyword): item for item in items}.values())

    def _search_rss(self, keyword: str, progress: ProgressCallback | None = None) -> list[ContentItem]:
        """Credential-free fallback via Reddit's Atom feed.

        Trade-off: the feed carries title, body, author, date and subreddit, but
        no score/comment counts, so these rows score 0 on engagement and get
        selected only by the keyword rules.
        """
        response = self.request(
            "GET",
            "https://www.reddit.com/search.rss",
            params={
                "q": keyword,
                "sort": "relevance",
                "t": "all",
                "limit": min(self.config.page_size * self.config.pages, 100),
            },
        )
        text = response.text
        if "<entry>" not in text:
            # HTTP 200 with an empty feed means the keyword genuinely matches
            # nothing — that is not a collection failure and must not be
            # reported as one. (Rate limiting surfaces as HTTP 429 instead.)
            self.note_partial(
                "search",
                f"Reddit RSS 中「{keyword}」没有匹配内容（Reddit 中文语料稀少，建议用更宽的英文/通用词）",
                blocking=False,
            )
            return []
        items: list[ContentItem] = []
        for entry in re.findall(r"<entry>(.*?)</entry>", text, re.S):
            title = clean_html(_text_of(entry, "title"))
            link = _link_of(entry)
            author = clean_html(_text_of(entry, "name")).lstrip("/u/")
            updated = _text_of(entry, "updated")
            category = _contributor_category(entry)
            if not link:
                continue
            post_id = link.rstrip("/").rsplit("/", 1)[-1].split("?")[0]
            body_match = re.search(r'<content type="html">(.*?)</content>', entry, re.S)
            body = clean_html(body_match.group(1)) if body_match else ""
            items.append(
                ContentItem(
                    id=post_id or link,
                    platform=self.name,
                    keyword=keyword,
                    content_type="post",
                    title=title or body[:80],
                    summary=body[:1000],
                    author=author,
                    published_at=updated,
                    # Reddit's feed exposes no engagement numbers at all.
                    engagement=Engagement(),
                    url=link,
                    extra={
                        "subreddit": category,
                        "tags": [category] if category else [],
                        "source": "reddit-rss",
                        "engagement_unavailable": True,
                    },
                )
            )
        if items:
            self.note_partial(
                "search",
                "Reddit 使用 RSS 回退（未配置 OAuth）：标题/正文/作者/时间可用，但没有点赞与评论数",
                blocking=False,
            )
        if progress:
            progress(self.name, 1, 1, f"{keyword}: {len(items)} 条")
        return list({(item.id, item.keyword): item for item in items}.values())

    @staticmethod
    def _walk_comments(value: Any) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for child in (value or {}).get("data", {}).get("children", []) if isinstance(value, dict) else []:
            data = child.get("data") or {}
            if child.get("kind") == "t1":
                rows.append(data)
                rows.extend(RedditCollector._walk_comments(data.get("replies")))
        return rows

    def deep_dive(self, item: ContentItem) -> DeepContent:
        errors: list[str] = []
        comments: list[Comment] = []
        try:
            self._login()
            post_id = item.id
            payload = self.get_json(
                f"{self.oauth_root}/comments/{post_id}.json",
                allow_list=True,
                params={"raw_json": 1, "limit": 500, "sort": "top"},
            )
            rows = self._walk_comments(payload[1] if isinstance(payload, list) and len(payload) > 1 else {})
            for row in rows[:500]:
                comments.append(
                    Comment(
                        text=clean_html(str(row.get("body", ""))),
                        likes=safe_int(row.get("score")),
                        author=str(row.get("author", "")),
                        published_at=iso_from_timestamp(row.get("created_utc")),
                    )
                )
        except Exception as exc:
            errors.append(f"comments: {exc}")
        comments.sort(key=lambda value: value.likes, reverse=True)
        media_text = "\n".join(
            filter(None, [ocr_image_url(self.client, url, self.settings) for url in item.extra.get("picture_urls", [])])
        )
        return DeepContent(base=item, full_content="\n\n".join(filter(None, [item.title, item.summary])), comments=comments, media_text=media_text, errors=errors)

