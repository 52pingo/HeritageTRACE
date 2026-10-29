"""Shared rate limiting, retry, caching and challenge diagnostics."""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

from config import Settings
from exceptions import ChallengeDetected, CollectorError
from models import ContentItem, DeepContent
from utils import safe_slug

try:
    import httpx
except ImportError:  # imported lazily so mock mode remains dependency-light
    httpx = None


ProgressCallback = Callable[[str, int, int | None, str], None]


class BaseCollector(ABC):
    name = "base"
    # Platforms unreachable directly from the domestic network; these use the
    # configured proxies when PROXIES_FOREIGN_ONLY=true (the default).
    foreign = False
    default_user_agents = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    )

    def __init__(self, settings: Settings, platform_name: str | None = None) -> None:
        self.settings = settings
        if platform_name:
            self.name = platform_name
        self.config = settings.platform[self.name]
        self.log = logging.getLogger(f"collector.{self.name}")
        self.random = random.Random(settings.random_seed + sum(map(ord, self.name)))
        self._user_agents = settings.user_agents or list(self.default_user_agents)
        self._last_request_at = 0.0
        self._client: Any = None
        # 非致命的分页/深挖失败：不丢弃已采到的结果，但要让调用方看到原因。
        self.partial_errors: list[dict[str, Any]] = []

    def _credential_headers(self) -> dict[str, str]:
        """Credential-bearing headers, re-applied on every client (re)creation.

        Subclasses must return their Cookie/Authorization here instead of
        mutating ``self.client.headers`` once in ``__init__``: ``close()``
        discards the client, and a header set only at construction time would
        silently disappear afterwards.
        """
        return {}

    def _client_headers(self) -> dict[str, str]:
        headers = {
            "User-Agent": self.random.choice(self._user_agents),
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }
        headers.update(self._credential_headers())
        return headers

    @property
    def client(self) -> Any:
        if self._client is None:
            if httpx is None:
                raise CollectorError("缺少 httpx，请执行 pip install -r requirements.txt")
            proxy = None
            if self.settings.proxies and (self.foreign or not self.settings.proxies_foreign_only):
                proxy = self.random.choice(self.settings.proxies)
            elif self.foreign and not self.settings.proxies:
                self.log.warning("%s 属于外网平台，HTTP_PROXIES 未配置，将直连访问", self.name)
            kwargs: dict[str, Any] = {
                "headers": self._client_headers(),
                "timeout": self.config.timeout,
                "follow_redirects": True,
            }
            if proxy:
                kwargs["proxy"] = proxy
            self._client = httpx.Client(**kwargs)
        return self._client

    def note_partial(self, stage: str, exc: object, *, blocking: bool = True) -> None:
        """Record a non-fatal failure so partial results can still be returned.

        ``blocking=True`` (the default) means the platform is refusing to serve
        us — a captcha, a login wall, an exhausted quota — and the caller may
        stop querying this platform entirely. Pass ``blocking=False`` when the
        result is merely empty or the parser found nothing: that must not count
        against the platform.
        """
        message = str(exc)
        self.partial_errors.append(
            {"platform": self.name, "stage": stage, "error": message, "blocking": blocking}
        )
        if blocking:
            self.log.warning("%s %s 部分失败，已保留已采到的结果：%s", self.name, stage, message)
        else:
            self.log.info("%s %s 未采到内容（非故障）：%s", self.name, stage, message)

    def drain_partial_errors(self) -> list[dict[str, Any]]:
        drained, self.partial_errors = self.partial_errors, []
        return drained

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "BaseCollector":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _rate_limit(self) -> None:
        desired = self.random.uniform(self.config.delay_min, self.config.delay_max)
        remaining = desired - (time.monotonic() - self._last_request_at)
        if remaining > 0:
            time.sleep(remaining)

    def _debug_challenge(self, url: str, body: str, reason: str) -> Path:
        self.settings.debug_root.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(url.encode()).hexdigest()[:10]
        stem = safe_slug(f"{self.name}_{int(time.time())}_{digest}")
        html_path = self.settings.debug_root / f"{stem}.html"
        html_path.write_text(body, encoding="utf-8", errors="replace")
        metadata = {"platform": self.name, "url": url, "reason": reason, "saved_at": time.time()}
        html_path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        if self.settings.playwright_challenge_screenshot:
            try:
                from playwright.sync_api import sync_playwright

                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch(headless=True)
                    page = browser.new_page()
                    page.goto(url, wait_until="domcontentloaded", timeout=int(self.config.timeout * 1000))
                    page.screenshot(path=str(html_path.with_suffix(".png")), full_page=True)
                    browser.close()
            except Exception as exc:  # diagnostics must never hide the original challenge
                self.log.warning("无法生成 Playwright 验证截图: %s", exc)
        return html_path

    @staticmethod
    def _http_error_reason(response: Any) -> str:
        """Best-effort extraction of the provider's own error reason.

        Without this, every 401/403 looks identical — YouTube's ``quotaExceeded``
        (daily quota gone) and Reddit's IP block would otherwise both be reported
        as a bare "请求被拒绝".
        """
        try:
            payload = response.json()
        except Exception:
            return ""
        candidates: list[str] = []
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                for item in error.get("errors") or []:
                    if isinstance(item, dict):
                        candidates.append(str(item.get("reason") or item.get("message") or ""))
                candidates.append(str(error.get("message") or ""))
                candidates.append(str(error.get("code") or ""))
            elif error:
                candidates.append(str(error))
            for key in ("message", "msg", "reason", "error_description", "status_msg"):
                if payload.get(key):
                    candidates.append(str(payload[key]))
        return "; ".join(dict.fromkeys(value for value in candidates if value))[:200]

    def request(
        self,
        method: str,
        url: str,
        *,
        check_challenge: bool = True,
        allow_status: tuple[int, ...] = (),
        **kwargs: Any,
    ) -> Any:
        """Perform a request with pacing and retries.

        ``allow_status`` opts specific status codes out of the generic
        authorization failure so the collector can read the provider's own error
        body (Zhihu returns a JSON ``{"error": {"code": 10003}}`` signature
        rejection as HTTP 403, which is not a login problem).
        """
        last_error: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            self._rate_limit()
            try:
                # Re-apply credentials on every attempt: a collector may acquire
                # its session (Bluesky) or refresh its OAuth token (Reddit) after
                # the client already exists.
                self.client.headers.update(self._credential_headers())
                self.client.headers["User-Agent"] = self.random.choice(self._user_agents)
                response = self.client.request(method, url, **kwargs)
                self._last_request_at = time.monotonic()
                body_start = response.text[:10_000].lower()
                content_type = response.headers.get("Content-Type", "").lower()
                challenge_markers = ("captcha", "geetest", "人机验证", "访问频次过高", "security check")
                if check_challenge and "html" in content_type and any(marker in body_start for marker in challenge_markers):
                    saved = self._debug_challenge(str(response.url), response.text, "challenge marker")
                    raise ChallengeDetected(f"{self.name} 触发登录/人机验证，诊断已保存至 {saved}")
                if response.status_code in {401, 403} and response.status_code not in allow_status:
                    reason = self._http_error_reason(response)
                    saved = self._debug_challenge(
                        str(response.url), response.text, f"HTTP {response.status_code}" + (f" ({reason})" if reason else "")
                    )
                    detail = f"，{reason}" if reason else ""
                    raise CollectorError(
                        f"{self.name} 请求被拒绝（HTTP {response.status_code}{detail}），诊断：{saved}"
                    )
                if response.status_code == 429 or response.status_code >= 500:
                    retry_after = min(float(response.headers.get("Retry-After", 0) or 0), 60.0)
                    raise httpx.HTTPStatusError(
                        f"retryable HTTP {response.status_code}; retry_after={retry_after}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                return response
            except ChallengeDetected:
                raise
            except Exception as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    break
                response = getattr(exc, "response", None)
                retry_after = 0.0
                if response is not None:
                    try:
                        retry_after = min(float(response.headers.get("Retry-After", 0) or 0), 60.0)
                    except ValueError:
                        pass
                wait = retry_after or min(2 ** (attempt - 1) + self.random.random(), 30.0)
                self.log.warning("请求失败，第 %d/%d 次，%.1fs 后重试: %s", attempt, self.config.max_retries, wait, exc)
                time.sleep(wait)
        raise CollectorError(f"{self.name} 请求失败: {last_error}") from last_error

    def get_json(self, url: str, *, allow_list: bool = False, **kwargs: Any) -> Any:
        """GET and decode JSON.

        ``allow_list=True`` accepts a top-level JSON array. Some endpoints
        (Reddit's ``/comments/<id>.json``) return ``[post, comments]``; without
        this the shared layer rejected the payload outright.
        """
        response = self.request("GET", url, **kwargs)
        return self._decode_json(response, allow_list=allow_list)

    def post_json(self, url: str, payload: dict[str, Any], **kwargs: Any) -> Any:
        """POST a JSON body and decode the JSON response."""
        kwargs.setdefault("json", payload)
        response = self.request("POST", url, **kwargs)
        return self._decode_json(response)

    def _decode_json(self, response: Any, *, allow_list: bool = False) -> Any:
        try:
            value = response.json()
        except ValueError as exc:
            saved = self._debug_challenge(str(response.url), response.text, "invalid JSON")
            raise CollectorError(f"{self.name} 返回了非 JSON 内容，已保存至 {saved}") from exc
        if not isinstance(value, dict) and not (allow_list and isinstance(value, list)):
            raise CollectorError(f"{self.name} 返回格式异常")
        return value

    @abstractmethod
    def search(self, keyword: str, progress: ProgressCallback | None = None) -> list[ContentItem]:
        raise NotImplementedError

    @abstractmethod
    def deep_dive(self, item: ContentItem) -> DeepContent:
        raise NotImplementedError
