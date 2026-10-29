"""Bluesky search and thread APIs.

``app.bsky.feed.searchPosts`` is no longer served to anonymous callers on the
public AppView (it answers 403), so an account is required. Configure
``BLUESKY_HANDLE`` + ``BLUESKY_APP_PASSWORD`` (an App Password, not the account
password) and the collector opens a session at ``bsky.social``; without them it
falls back to the public AppView, where search will return nothing.
"""

from __future__ import annotations

from typing import Any

from exceptions import AuthenticationRequired
from media_tools import ocr_image_url
from models import Comment, ContentItem, DeepContent, Engagement
from utils import clean_html, safe_int

from .base import BaseCollector, ProgressCallback


class BlueskyCollector(BaseCollector):
    name = "bluesky"
    foreign = True
    public_root = "https://public.api.bsky.app/xrpc"
    auth_root = "https://bsky.social/xrpc"

    def __init__(self, settings: Any) -> None:
        super().__init__(settings)
        self._access_jwt = ""
        self._session_error = ""

    def _credential_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._access_jwt}"} if self._access_jwt else {}

    @property
    def api_root(self) -> str:
        return self.auth_root if self._access_jwt else self.public_root

    def _login(self) -> bool:
        """Open an authenticated session once; no-op when already logged in."""
        if self._access_jwt:
            return True
        handle = self.settings.bluesky_handle
        password = self.settings.bluesky_app_password
        if not (handle and password):
            return False
        try:
            payload = self.post_json(
                f"{self.auth_root}/com.atproto.server.createSession",
                {"identifier": handle, "password": password},
            )
        except Exception as exc:
            self._session_error = str(exc)
            self.log.warning("Bluesky 登录失败，将退回公开接口（搜索会返回空）: %s", exc)
            return False
        token = str(payload.get("accessJwt") or "")
        if not token:
            self._session_error = "createSession 未返回 accessJwt"
            return False
        self._access_jwt = token
        self.log.info("Bluesky 已建立认证会话: %s", payload.get("handle") or handle)
        return True

    def _require_session(self) -> None:
        if self._login():
            return
        if self.settings.bluesky_handle or self.settings.bluesky_app_password:
            raise AuthenticationRequired(
                f"Bluesky 登录失败（{self._session_error or '未知原因'}）；请检查 BLUESKY_HANDLE 与 "
                "BLUESKY_APP_PASSWORD 是否为 App Password"
            )
        raise AuthenticationRequired(
            "Bluesky 的搜索接口已要求登录：请在 .env 配置 BLUESKY_HANDLE 与 BLUESKY_APP_PASSWORD"
            "（在 bsky.app → Settings → App Passwords 生成）"
        )

    @staticmethod
    def _rkey(uri: str) -> str:
        return uri.rsplit("/", 1)[-1]

    @staticmethod
    def _hashtags(record: dict[str, Any]) -> list[str]:
        values: list[str] = []
        for facet in record.get("facets") or []:
            for feature in facet.get("features") or []:
                if feature.get("$type") == "app.bsky.richtext.facet#tag" and feature.get("tag"):
                    values.append(str(feature["tag"]))
        return list(dict.fromkeys(values))

    def search(self, keyword: str, progress: ProgressCallback | None = None) -> list[ContentItem]:
        self._require_session()
        items: list[ContentItem] = []
        cursor: str | None = None
        for page in range(1, self.config.pages + 1):
            params: dict[str, Any] = {"q": keyword, "limit": min(self.config.page_size, 100), "sort": "latest"}
            if cursor:
                params["cursor"] = cursor
            try:
                payload = self.get_json(f"{self.api_root}/app.bsky.feed.searchPosts", params=params)
            except Exception as exc:
                if items:
                    self.note_partial("search", exc)
                    break
                raise
            for row in payload.get("posts") or []:
                record = row.get("record") or {}
                author = row.get("author") or {}
                uri = str(row.get("uri", ""))
                post_id = uri.rsplit("/", 1)[-1]
                text = clean_html(str(record.get("text", "")))
                handle = str(author.get("handle") or author.get("displayName") or "")
                if not post_id:
                    continue
                items.append(
                    ContentItem(
                        id=post_id,
                        platform=self.name,
                        keyword=keyword,
                        content_type="post",
                        title=text[:100],
                        summary=text[:1000],
                        author=handle,
                        published_at=str(record.get("createdAt") or row.get("indexedAt", "")),
                        engagement=Engagement(
                            likes=safe_int(row.get("likeCount")),
                            comments=safe_int(row.get("replyCount")),
                            shares=safe_int(row.get("repostCount")) + safe_int(row.get("quoteCount")),
                            views=0,
                        ),
                        url=f"https://bsky.app/profile/{handle}/post/{self._rkey(uri)}",
                        extra={
                            "uri": uri,
                            "author_did": author.get("did"),
                            "hashtags": self._hashtags(record),
                            "image_urls": [str(image.get("thumb", "")) for image in (row.get("embed") or {}).get("images", [])],
                        },
                    )
                )
            if progress:
                progress(self.name, page, self.config.pages, f"{keyword}: {len(items)} 条")
            cursor = payload.get("cursor")
            if not cursor or not payload.get("posts"):
                break
        return list({(item.id, item.keyword): item for item in items}.values())

    def _thread_rows(self, value: dict[str, Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        post = value.get("post") or {}
        if post:
            rows.append(post)
        for reply in value.get("replies") or []:
            rows.extend(self._thread_rows(reply))
        return rows

    def deep_dive(self, item: ContentItem) -> DeepContent:
        errors: list[str] = []
        comments: list[Comment] = []
        full_parts = [item.summary]
        try:
            payload = self.get_json(f"{self.api_root}/app.bsky.feed.getPostThread", params={"uri": item.extra.get("uri", ""), "depth": 100})
            rows = self._thread_rows(payload.get("thread") or {})
            for row in rows:
                record = row.get("record") or {}
                text = clean_html(str(record.get("text", "")))
                if str(row.get("uri", "")).endswith(item.id):
                    full_parts.append(text)
                elif text:
                    comments.append(Comment(text=text, likes=safe_int(row.get("likeCount")), author=str((row.get("author") or {}).get("handle", "")), published_at=str(record.get("createdAt", ""))))
        except Exception as exc:
            errors.append(f"thread: {exc}")
        media_text = "\n".join(filter(None, [ocr_image_url(self.client, url, self.settings) for url in item.extra.get("image_urls", [])]))
        return DeepContent(base=item, full_content="\n\n".join(filter(None, full_parts)), comments=comments[:100], media_text=media_text, errors=errors)

