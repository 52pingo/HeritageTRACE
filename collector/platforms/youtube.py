"""YouTube Data API v3 collector with transcript-first deep extraction."""

from __future__ import annotations

from typing import Any

from exceptions import AuthenticationRequired, CollectorError
from media_tools import ocr_image_url, whisper_transcribe_url
from models import Comment, ContentItem, DeepContent, Engagement
from utils import clean_html, safe_int

from .base import BaseCollector, ProgressCallback


class YouTubeCollector(BaseCollector):
    name = "youtube"
    foreign = True
    api_root = "https://www.googleapis.com/youtube/v3"

    def __init__(self, settings: Any) -> None:
        super().__init__(settings)
        if not settings.youtube_api_key:
            raise AuthenticationRequired("YouTube 采集需要 YOUTUBE_API_KEY（Data API v3）")

    def _api(self, resource: str, **params: Any) -> dict[str, Any]:
        params["key"] = self.settings.youtube_api_key
        payload = self.get_json(f"{self.api_root}/{resource}", params=params)
        if payload.get("error"):
            raise CollectorError(f"YouTube API 错误: {payload['error']}")
        return payload

    def _details(self, ids: list[str]) -> dict[str, dict[str, Any]]:
        if not ids:
            return {}
        payload = self._api("videos", part="snippet,statistics,contentDetails", id=",".join(ids))
        return {str(row.get("id")): row for row in payload.get("items") or []}

    def _channel_followers(self, ids: list[str]) -> dict[str, int]:
        unique = list(dict.fromkeys(filter(None, ids)))
        if not unique:
            return {}
        payload = self._api("channels", part="statistics", id=",".join(unique[:50]))
        return {
            str(row.get("id")): safe_int((row.get("statistics") or {}).get("subscriberCount"))
            for row in payload.get("items") or []
        }

    def search(self, keyword: str, progress: ProgressCallback | None = None) -> list[ContentItem]:
        items: list[ContentItem] = []
        token: str | None = None
        for page in range(1, self.config.pages + 1):
            params: dict[str, Any] = {
                "part": "snippet",
                "q": keyword,
                "type": "video",
                "maxResults": max(1, min(self.config.page_size, 50)),
                "order": "relevance",
            }
            if token:
                params["pageToken"] = token
            try:
                payload = self._api("search", **params)
            except Exception as exc:
                if items:
                    self.note_partial("search", exc)
                    break
                raise
            search_rows = payload.get("items") or []
            ids = [str((row.get("id") or {}).get("videoId")) for row in search_rows if (row.get("id") or {}).get("videoId")]
            # Enrichment is optional: a failing statistics call must not discard
            # the pages already collected.
            try:
                details = self._details(ids)
            except Exception as exc:
                self.note_partial("details", exc)
                details = {}
            try:
                channel_followers = self._channel_followers(
                    [str((row.get("snippet") or {}).get("channelId", "")) for row in details.values()]
                )
            except Exception as exc:
                self.note_partial("channels", exc)
                channel_followers = {}
            for video_id in ids:
                row = details.get(video_id, {})
                snippet = row.get("snippet") or {}
                stats = row.get("statistics") or {}
                thumbnails = snippet.get("thumbnails") or {}
                thumb = (thumbnails.get("maxres") or thumbnails.get("high") or thumbnails.get("default") or {}).get("url", "")
                description = clean_html(str(snippet.get("description", "")))
                channel_id = str(snippet.get("channelId", ""))
                items.append(
                    ContentItem(
                        id=video_id,
                        platform=self.name,
                        keyword=keyword,
                        content_type="video",
                        title=clean_html(str(snippet.get("title", ""))),
                        summary=description[:500],
                        author=str(snippet.get("channelTitle", "")),
                        published_at=str(snippet.get("publishedAt", "")),
                        engagement=Engagement(
                            likes=safe_int(stats.get("likeCount")),
                            comments=safe_int(stats.get("commentCount")),
                            shares=0,
                            views=safe_int(stats.get("viewCount")),
                        ),
                        url=f"https://www.youtube.com/watch?v={video_id}",
                        extra={
                            "channel_id": channel_id,
                            "author_followers": channel_followers.get(channel_id, 0),
                            "full_description": description,
                            "thumbnail_url": thumb,
                            "duration": (row.get("contentDetails") or {}).get("duration"),
                            "tags": [str(tag) for tag in snippet.get("tags") or []],
                        },
                    )
                )
            if progress:
                progress(self.name, page, self.config.pages, f"{keyword}: {len(items)} 条")
            token = payload.get("nextPageToken")
            if not token:
                break
        return items

    @staticmethod
    def _transcript(video_id: str) -> str:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi

            try:  # youtube-transcript-api >= 1.2
                rows = YouTubeTranscriptApi().fetch(video_id, languages=["zh-Hans", "zh-Hant", "zh", "en"])
                return " ".join(str(getattr(row, "text", "")) for row in rows).strip()
            except AttributeError:  # compatibility with older releases
                rows = YouTubeTranscriptApi.get_transcript(video_id, languages=["zh-Hans", "zh-Hant", "zh", "en"])
                return " ".join(str(row.get("text", "")) for row in rows).strip()
        except Exception:
            return ""

    def _comments(self, video_id: str, limit: int = 200) -> list[Comment]:
        comments: list[Comment] = []
        token: str | None = None
        while len(comments) < limit:
            params: dict[str, Any] = {
                "part": "snippet,replies",
                "videoId": video_id,
                "maxResults": min(100, limit - len(comments)),
                "order": "relevance",
                "textFormat": "plainText",
            }
            if token:
                params["pageToken"] = token
            try:
                payload = self._api("commentThreads", **params)
            except CollectorError as exc:
                if "commentsDisabled" in str(exc):
                    return comments
                raise
            for thread in payload.get("items") or []:
                thread_snippet = thread.get("snippet") or {}
                top_level = thread_snippet.get("topLevelComment") or {}
                snippet = top_level.get("snippet") or {}
                comments.append(
                    Comment(
                        text=str(snippet.get("textOriginal") or snippet.get("textDisplay") or ""),
                        likes=safe_int(snippet.get("likeCount")),
                        author=str(snippet.get("authorDisplayName", "")),
                        published_at=str(snippet.get("publishedAt", "")),
                    )
                )
                embedded_replies = (thread.get("replies") or {}).get("comments") or []
                # commentThreads may embed only a subset. Fetch the complete reply set when needed.
                if safe_int(thread_snippet.get("totalReplyCount")) > len(embedded_replies):
                    reply_rows = self._all_replies(str(top_level.get("id", "")), limit - len(comments))
                else:
                    reply_rows = embedded_replies
                for reply in reply_rows:
                    reply_snippet = reply.get("snippet") or {}
                    comments.append(
                        Comment(
                            text=str(reply_snippet.get("textOriginal") or reply_snippet.get("textDisplay") or ""),
                            likes=safe_int(reply_snippet.get("likeCount")),
                            author=str(reply_snippet.get("authorDisplayName", "")),
                            published_at=str(reply_snippet.get("publishedAt", "")),
                        )
                    )
                    if len(comments) >= limit:
                        break
                if len(comments) >= limit:
                    break
            token = payload.get("nextPageToken")
            if not token:
                break
        comments.sort(key=lambda row: row.likes, reverse=True)
        return comments[:limit]

    def _all_replies(self, parent_id: str, limit: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        token: str | None = None
        while parent_id and len(rows) < limit:
            params: dict[str, Any] = {
                "part": "snippet",
                "parentId": parent_id,
                "maxResults": min(100, limit - len(rows)),
                "textFormat": "plainText",
            }
            if token:
                params["pageToken"] = token
            payload = self._api("comments", **params)
            rows.extend(payload.get("items") or [])
            token = payload.get("nextPageToken")
            if not token:
                break
        return rows[:limit]

    def deep_dive(self, item: ContentItem) -> DeepContent:
        errors: list[str] = []
        transcript = self._transcript(item.id)
        if not transcript:
            transcript = whisper_transcribe_url(item.url, self.settings)
        description = str(item.extra.get("full_description") or item.summary)
        full_content = "\n\n".join(filter(None, [description, transcript]))
        try:
            comments = self._comments(item.id)
        except Exception as exc:
            errors.append(f"comments: {exc}")
            comments = []
        media_text = ocr_image_url(self.client, str(item.extra.get("thumbnail_url", "")), self.settings)
        return DeepContent(base=item, full_content=full_content, comments=comments, media_text=media_text, errors=errors)
