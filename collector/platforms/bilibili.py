"""Bilibili public web-API collector with conservative request pacing."""

from __future__ import annotations

import hashlib
import time
from typing import Any
from urllib.parse import urlencode, urlparse

from exceptions import CollectorError
from media_tools import ocr_image_url, whisper_transcribe_url
from models import Comment, ContentItem, DeepContent, Engagement
from utils import clean_html, iso_from_timestamp, safe_int

from .base import BaseCollector, ProgressCallback


class BilibiliCollector(BaseCollector):
    name = "bilibili"
    search_url = "https://api.bilibili.com/x/web-interface/wbi/search/type"
    mixin_key_order = (
        46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35,
        27, 43, 5, 49, 33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13,
        37, 48, 7, 16, 24, 55, 40, 61, 26, 17, 0, 1, 60, 51, 30, 4,
        22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11, 36, 20, 34, 44, 52,
    )

    def __init__(self, settings: Any) -> None:
        super().__init__(settings)
        self._mixin_key = ""
        self._mixin_key_at = 0.0

    def _credential_headers(self) -> dict[str, str]:
        headers = {"Referer": "https://search.bilibili.com/"}
        if self.settings.bilibili_cookie:
            headers["Cookie"] = self.settings.bilibili_cookie
        return headers

    @staticmethod
    def _key_from_url(url: str) -> str:
        return urlparse(url).path.rsplit("/", 1)[-1].split(".", 1)[0]

    def _get_mixin_key(self) -> str:
        if self._mixin_key and time.monotonic() - self._mixin_key_at < 600:
            return self._mixin_key
        payload = self.get_json("https://api.bilibili.com/x/web-interface/nav")
        wbi = ((payload.get("data") or {}).get("wbi_img") or {})
        raw = self._key_from_url(str(wbi.get("img_url", ""))) + self._key_from_url(str(wbi.get("sub_url", "")))
        if len(raw) < 64:
            raise CollectorError("Bilibili 未返回有效 WBI 密钥")
        self._mixin_key = "".join(raw[index] for index in self.mixin_key_order)[:32]
        self._mixin_key_at = time.monotonic()
        return self._mixin_key

    def _sign(self, params: dict[str, Any]) -> dict[str, Any]:
        signed = {**params, "wts": int(time.time())}
        filtered = {
            key: "".join(char for char in str(value) if char not in "!'()*")
            for key, value in sorted(signed.items())
        }
        query = urlencode(filtered)
        filtered["w_rid"] = hashlib.md5((query + self._get_mixin_key()).encode()).hexdigest()
        return filtered

    def _detail(self, bvid: str) -> dict[str, Any]:
        payload = self.get_json("https://api.bilibili.com/x/web-interface/view", params={"bvid": bvid})
        return payload.get("data") or {} if payload.get("code") == 0 else {}

    def search(self, keyword: str, progress: ProgressCallback | None = None) -> list[ContentItem]:
        items: list[ContentItem] = []
        details_ok = True
        for page in range(1, self.config.pages + 1):
            try:
                payload = self.get_json(
                    self.search_url,
                    params=self._sign(
                        {"search_type": "video", "keyword": keyword, "page": page, "page_size": self.config.page_size}
                    ),
                )
            except Exception as exc:
                # A failure on page N must not discard pages 1..N-1.
                if items:
                    self.note_partial("search", exc)
                    break
                raise
            if payload.get("code") != 0:
                message = f"Bilibili API 错误 {payload.get('code')}: {payload.get('message')}"
                if items:
                    self.note_partial("search", message)
                    break
                raise CollectorError(message)
            data = payload.get("data") or {}
            if "v_voucher" in data and not data.get("result"):
                message = (
                    "Bilibili 搜索触发风控验证（v_voucher）；"
                    "请降低请求频率、稍后重试，或检查 BILIBILI_COOKIE 是否有效"
                )
                if items:
                    self.note_partial("search", message)
                    break
                raise CollectorError(message)
            results = data.get("result") or []
            if not results:
                break
            for raw in results:
                bvid = str(raw.get("bvid") or raw.get("id") or "")
                # Each detail call is an extra request; when they start failing
                # (risk control) fall back to the search payload instead of
                # hammering the API and losing the whole keyword.
                detail: dict[str, Any] = {}
                if bvid and self.settings.bilibili_fetch_detail and details_ok:
                    try:
                        detail = self._detail(bvid)
                    except Exception as exc:
                        details_ok = False
                        self.note_partial("detail", exc)
                stat = detail.get("stat") or {}
                owner = detail.get("owner") or {}
                aid = detail.get("aid") or raw.get("aid")
                cover_url = str(detail.get("pic") or raw.get("pic") or "")
                if cover_url.startswith("//"):
                    cover_url = "https:" + cover_url
                items.append(
                    ContentItem(
                        id=bvid or str(aid),
                        platform=self.name,
                        keyword=keyword,
                        content_type="video",
                        title=clean_html(detail.get("title") or raw.get("title")),
                        summary=clean_html(detail.get("desc") or raw.get("description"))[:1000],
                        author=str(owner.get("name") or raw.get("author") or ""),
                        published_at=iso_from_timestamp(detail.get("pubdate") or raw.get("pubdate")),
                        engagement=Engagement(
                            likes=safe_int(stat.get("like") or raw.get("like")),
                            comments=safe_int(stat.get("reply") or raw.get("review")),
                            shares=safe_int(stat.get("share")),
                            views=safe_int(stat.get("view") or raw.get("play")),
                        ),
                        url=f"https://www.bilibili.com/video/{bvid}" if bvid else str(raw.get("arcurl", "")),
                        extra={
                            "aid": aid,
                            "bvid": bvid,
                            "cid": (detail.get("pages") or [{}])[0].get("cid"),
                            "coins": safe_int(stat.get("coin")),
                            "favorites": safe_int(stat.get("favorite") or raw.get("favorites")),
                            "danmaku": safe_int(stat.get("danmaku") or raw.get("video_review")),
                            "cover_url": cover_url,
                            "author_id": owner.get("mid") or raw.get("mid"),
                            "tags": [str(tag.get("tag_name") or tag.get("name")) for tag in detail.get("tags") or [] if isinstance(tag, dict)],
                        },
                    )
                )
            if progress:
                progress(self.name, page, self.config.pages, f"{keyword}: {len(items)} 条")
        return items

    def _subtitles(self, item: ContentItem) -> str:
        cid, bvid = item.extra.get("cid"), item.extra.get("bvid")
        if not cid or not bvid:
            return ""
        payload = self.get_json("https://api.bilibili.com/x/player/v2", params={"bvid": bvid, "cid": cid})
        subtitles = ((payload.get("data") or {}).get("subtitle") or {}).get("subtitles") or []
        texts: list[str] = []
        for subtitle in subtitles[:1]:
            url = str(subtitle.get("subtitle_url") or "")
            if url.startswith("//"):
                url = "https:" + url
            if not url:
                continue
            data = self.get_json(url)
            texts.extend(str(row.get("content", "")) for row in data.get("body", []))
        return "\n".join(filter(None, texts))

    def _comments(self, aid: Any, limit: int = 100) -> list[Comment]:
        if not aid:
            return []
        comments: list[Comment] = []
        next_value = 0
        while len(comments) < limit:
            payload = self.get_json(
                "https://api.bilibili.com/x/v2/reply/main",
                params={"oid": aid, "type": 1, "mode": 3, "next": next_value, "ps": min(20, limit - len(comments))},
            )
            data = payload.get("data") or {}
            replies = data.get("replies") or []
            for row in replies:
                member = row.get("member") or {}
                comments.append(
                    Comment(
                        text=str((row.get("content") or {}).get("message", "")),
                        likes=safe_int(row.get("like")),
                        author=str(member.get("uname", "")),
                        published_at=iso_from_timestamp(row.get("ctime")),
                    )
                )
            cursor = data.get("cursor") or {}
            if not replies or cursor.get("is_end"):
                break
            next_value = safe_int(cursor.get("next"))
        return comments[:limit]

    def deep_dive(self, item: ContentItem) -> DeepContent:
        errors: list[str] = []
        bvid = str(item.extra.get("bvid") or "")
        if bvid and (not item.extra.get("cid") or not item.extra.get("aid")):
            try:
                detail = self._detail(bvid)
                item.extra.setdefault("aid", detail.get("aid"))
                item.extra.setdefault("cid", ((detail.get("pages") or [{}])[0] or {}).get("cid"))
            except Exception as exc:
                errors.append(f"detail: {exc}")
        try:
            full_content = self._subtitles(item)
        except Exception as exc:
            errors.append(f"subtitle: {exc}")
            full_content = ""
        if not full_content:
            full_content = whisper_transcribe_url(item.url, self.settings) or item.summary
        try:
            comments = self._comments(item.extra.get("aid"), 100)
        except Exception as exc:
            errors.append(f"comments: {exc}")
            comments = []
        media_text = ocr_image_url(self.client, str(item.extra.get("cover_url", "")), self.settings)
        return DeepContent(base=item, full_content=full_content, comments=comments, media_text=media_text, errors=errors)
