"""Environment-based configuration. Secrets are never stored in generated datasets."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # mock mode works with only the standard library
    load_dotenv = None

LOGGER = logging.getLogger("config")


ROOT = Path(__file__).resolve().parent
if load_dotenv:
    load_dotenv(ROOT / ".env")


def _env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(slots=True)
class PlatformConfig:
    enabled: bool = True
    pages: int = 10
    page_size: int = 20
    delay_min: float = 1.0
    delay_max: float = 3.0
    max_retries: int = 3
    timeout: float = 20.0


@dataclass(slots=True)
class Settings:
    data_root: Path = field(default_factory=lambda: Path(os.getenv("DATA_ROOT", ROOT / "data")))
    debug_root: Path = field(default_factory=lambda: Path(os.getenv("DEBUG_ROOT", ROOT / "debug")))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    youtube_api_key: str = field(default_factory=lambda: os.getenv("YOUTUBE_API_KEY", ""))
    x_bearer_token: str = field(default_factory=lambda: os.getenv("X_BEARER_TOKEN", ""))
    bilibili_cookie: str = field(default_factory=lambda: os.getenv("BILIBILI_COOKIE", ""))
    bilibili_fetch_detail: bool = field(default_factory=lambda: _env_bool("BILIBILI_FETCH_DETAIL", True))
    weibo_cookie: str = field(default_factory=lambda: os.getenv("WEIBO_COOKIE", ""))
    douyin_cookie: str = field(default_factory=lambda: os.getenv("DOUYIN_COOKIE", ""))
    tieba_cookie: str = field(default_factory=lambda: os.getenv("TIEBA_COOKIE", ""))
    zhihu_profile_dir: str = field(default_factory=lambda: os.getenv("ZHIHU_PROFILE_DIR", "").strip())
    zhihu_headless: bool = field(default_factory=lambda: _env_bool("ZHIHU_HEADLESS", True))
    zhihu_dc0: str = field(default_factory=lambda: os.getenv("ZHIHU_DC0", "").strip())
    # 完整 Cookie 串(含 z_c0 登录态)。知乎已要求登录才返回搜索结果:
    # 仅有访客 d_c0 时搜索接口返回 401 ZERR_NOT_LOGIN(2026-10-01 实测)。
    zhihu_cookie: str = field(default_factory=lambda: os.getenv("ZHIHU_COOKIE", "").strip())
    zhihu_mode: str = field(default_factory=lambda: os.getenv("ZHIHU_MODE", "auto").strip().lower())
    # Reddit 已关闭未认证的 *.json 公开接口，必须走 OAuth 才能采集。
    reddit_client_id: str = field(default_factory=lambda: os.getenv("REDDIT_CLIENT_ID", "").strip())
    reddit_client_secret: str = field(default_factory=lambda: os.getenv("REDDIT_CLIENT_SECRET", "").strip())
    reddit_user_agent: str = field(default_factory=lambda: os.getenv("REDDIT_USER_AGENT", "").strip())
    # Bluesky 的 searchPosts 已要求登录，用账号 + App Password 建立会话。
    bluesky_handle: str = field(default_factory=lambda: os.getenv("BLUESKY_HANDLE", "").strip())
    bluesky_app_password: str = field(default_factory=lambda: os.getenv("BLUESKY_APP_PASSWORD", "").strip())
    proxies: list[str] = field(
        default_factory=lambda: [value.strip() for value in os.getenv("HTTP_PROXIES", "").split(",") if value.strip()]
    )
    user_agents: list[str] = field(
        default_factory=lambda: [value.strip() for value in os.getenv("USER_AGENTS", "").split("||") if value.strip()]
    )
    author_follower_threshold: int = field(default_factory=lambda: _env_int("AUTHOR_FOLLOWER_THRESHOLD", 100_000))
    high_value_percentile: float = field(default_factory=lambda: _env_float("HIGH_VALUE_PERCENTILE", 0.8))
    high_frequency_combo_size: int = field(default_factory=lambda: _env_int("HIGH_FREQUENCY_COMBO_SIZE", 2))
    growth_rate_threshold: float = field(default_factory=lambda: _env_float("GROWTH_RATE_THRESHOLD", 100.0))
    ocr_enabled: bool = field(default_factory=lambda: _env_bool("OCR_ENABLED", False))
    whisper_enabled: bool = field(default_factory=lambda: _env_bool("WHISPER_ENABLED", False))
    whisper_model: str = field(default_factory=lambda: os.getenv("WHISPER_MODEL", "tiny"))
    playwright_challenge_screenshot: bool = field(
        default_factory=lambda: _env_bool("PLAYWRIGHT_CHALLENGE_SCREENSHOT", False)
    )
    incremental: bool = field(default_factory=lambda: _env_bool("INCREMENTAL", True))
    # 已见 URL 的保留天数。0 表示永不过期（保持原有语义）。
    incremental_ttl_days: int = field(default_factory=lambda: _env_int("INCREMENTAL_TTL_DAYS", 0))
    # 同一平台连续失败多少次后跳过该平台其余查询（0 表示不熔断）。
    platform_failure_threshold: int = field(default_factory=lambda: _env_int("PLATFORM_FAILURE_THRESHOLD", 3))
    random_seed: int = field(default_factory=lambda: _env_int("RANDOM_SEED", 42))
    weibo_search_modes: list[str] = field(
        default_factory=lambda: [value.strip().lower() for value in os.getenv("WEIBO_SEARCH_MODES", "comprehensive,realtime,hot").split(",") if value.strip()]
    )
    x_search_mode: str = field(default_factory=lambda: os.getenv("X_SEARCH_MODE", "recent").strip().lower())
    # 未配置 X_BEARER_TOKEN 时用于回退搜索的公共 Nitter 实例（逗号分隔）。
    x_nitter_instances: list[str] = field(
        default_factory=lambda: [v.strip() for v in os.getenv("X_NITTER_INSTANCES", "").split(",") if v.strip()]
    )
    rss_feeds: dict[str, str] = field(default_factory=dict)
    rss_match_all: bool = field(default_factory=lambda: _env_bool("RSS_MATCH_ALL", False))
    stackexchange_site: str = field(default_factory=lambda: os.getenv("STACKEXCHANGE_SITE", "stackoverflow").strip())
    baidu_tab: str = field(default_factory=lambda: os.getenv("BAIDU_TAB", "realtime").strip())
    baidu_match_all: bool = field(default_factory=lambda: _env_bool("BAIDU_MATCH_ALL", False))
    douban_cat: str = field(default_factory=lambda: os.getenv("DOUBAN_CAT", "1001").strip())
    douyin_hot_match_all: bool = field(default_factory=lambda: _env_bool("DOUYIN_HOT_MATCH_ALL", False))
    proxies_foreign_only: bool = field(default_factory=lambda: _env_bool("PROXIES_FOREIGN_ONLY", True))
    keyword_expansions: dict[str, list[str]] = field(default_factory=dict)
    platform: dict[str, PlatformConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        platform_names = ("bilibili", "weibo", "x", "youtube", "douyin", "douyin_hot", "douyin_sug", "reddit", "hackernews", "bluesky", "tieba", "zhihu", "juejin", "csdn", "rss", "arxiv", "stackexchange", "wikipedia", "github", "v2ex", "baidu", "douban")
        defaults = {
            name: PlatformConfig(
                pages=_env_int(f"{name.upper()}_PAGES", 10),
                page_size=_env_int(f"{name.upper()}_PAGE_SIZE", 20),
                delay_min=_env_float(f"{name.upper()}_DELAY_MIN", 1.0),
                delay_max=_env_float(f"{name.upper()}_DELAY_MAX", 3.0),
                max_retries=_env_int(f"{name.upper()}_MAX_RETRIES", 3),
                timeout=_env_float(f"{name.upper()}_TIMEOUT", 20.0),
            )
            for name in platform_names
        }
        for name, default in defaults.items():
            self.platform.setdefault(name, default)
        if not self.keyword_expansions:
            raw_expansions = os.getenv("KEYWORD_EXPANSIONS_JSON", "")
            if raw_expansions:
                try:
                    parsed = json.loads(raw_expansions)
                    if isinstance(parsed, dict):
                        self.keyword_expansions = {
                            str(key): [str(value) for value in values]
                            for key, values in parsed.items()
                            if isinstance(values, list)
                        }
                except json.JSONDecodeError:
                    self.keyword_expansions = {}
        if not self.rss_feeds:
            raw_feeds = os.getenv("RSS_FEEDS_JSON", "")
            if raw_feeds:
                try:
                    parsed = json.loads(raw_feeds)
                    if isinstance(parsed, dict):
                        self.rss_feeds = {str(key): str(value) for key, value in parsed.items() if str(value).strip()}
                except json.JSONDecodeError:
                    self.rss_feeds = {}
        self.data_root = self.data_root.resolve()
        self.debug_root = self.debug_root.resolve()
        if self.zhihu_mode not in {"api", "playwright", "auto"}:
            self.zhihu_mode = "auto"

    def validate_platform(self, name: str, mock: bool = False) -> list[str]:
        if mock:
            return []
        missing: list[str] = []
        if name == "youtube" and not self.youtube_api_key:
            missing.append("YOUTUBE_API_KEY")
        elif name == "x" and not self.x_bearer_token:
            # 不致命：会退回公共 Nitter 实例的 RSS 搜索，仅缺少互动数据。
            LOGGER.warning("X 未配置 X_BEARER_TOKEN，将使用 Nitter 公开 RSS 搜索（无点赞/转推数）")
        elif name == "weibo" and not self.weibo_cookie:
            missing.append("WEIBO_COOKIE")
        elif name == "zhihu" and self.zhihu_mode == "playwright" and not self.zhihu_profile_dir:
            missing.append("ZHIHU_PROFILE_DIR")
        elif name == "rss" and not self.rss_feeds:
            missing.append("RSS_FEEDS_JSON")
        elif name == "reddit" and not (self.reddit_client_id and self.reddit_client_secret):
            # 不是致命缺失：会退回公开 RSS，仅缺少互动数据。
            LOGGER.warning(
                "Reddit 未配置 OAuth 凭据，将使用公开 RSS 搜索（无点赞/评论数，且频率受限）"
            )
        elif name == "bluesky" and not (self.bluesky_handle and self.bluesky_app_password):
            missing.append("BLUESKY_HANDLE/BLUESKY_APP_PASSWORD（Bluesky 搜索接口已要求登录）")
        elif name == "douyin" and not self.douyin_cookie:
            missing.append(
                "DOUYIN_COOKIE（抖音搜索需登录态；无 Cookie 时可改用免凭据的 douyin_hot 抖音热搜榜）"
            )
        return missing

    def snapshot(self) -> dict[str, Any]:
        value = asdict(self)
        value["data_root"] = str(self.data_root)
        value["debug_root"] = str(self.debug_root)
        for secret in (
            "youtube_api_key",
            "x_bearer_token",
            "bilibili_cookie",
            "weibo_cookie",
            "douyin_cookie",
            "tieba_cookie",
            "zhihu_dc0",
            "zhihu_cookie",
            "reddit_client_id",
            "reddit_client_secret",
            "bluesky_app_password",
        ):
            value[secret] = "***configured***" if value[secret] else ""
        value["proxies"] = [re.sub(r"//.*@", "//***@", proxy) for proxy in value["proxies"]]
        return value

    def write_snapshot(self, path: Path) -> None:
        path.write_text(json.dumps(self.snapshot(), ensure_ascii=False, indent=2), encoding="utf-8")


SUPPORTED_PLATFORMS = ("bilibili", "weibo", "x", "youtube", "douyin", "douyin_hot", "douyin_sug", "reddit", "hackernews", "bluesky", "tieba", "zhihu", "juejin", "csdn", "rss", "arxiv", "stackexchange", "wikipedia", "github", "v2ex", "baidu", "douban")
