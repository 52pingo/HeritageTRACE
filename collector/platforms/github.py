"""GitHub public Search/REST API collector (keyless, 10 req/min anonymous)."""

from __future__ import annotations

from typing import Any

from exceptions import CollectorError
from models import ContentItem, DeepContent, Engagement
from utils import clean_html, safe_int

from .base import BaseCollector, ProgressCallback


class GitHubCollector(BaseCollector):
    name = "github"
    foreign = True
    api_root = "https://api.github.com"

    def __init__(self, settings: Any) -> None:
        super().__init__(settings)
        self.client.headers["Accept"] = "application/vnd.github+json"

    @staticmethod
    def _convert(row: dict[str, Any], keyword: str) -> ContentItem | None:
        repo_id = row.get("id")
        full_name = str(row.get("full_name") or "")
        if repo_id is None or not full_name:
            return None
        owner = row.get("owner") or {}
        license_info = row.get("license") or {}
        return ContentItem(
            id=str(repo_id),
            platform="github",
            keyword=keyword,
            content_type="repo",
            title=full_name,
            summary=clean_html(str(row.get("description") or ""))[:1000],
            author=str(owner.get("login") or ""),
            published_at=str(row.get("created_at") or ""),
            engagement=Engagement(
                likes=safe_int(row.get("stargazers_count")),
                comments=safe_int(row.get("open_issues_count")),
                shares=safe_int(row.get("forks_count")),
                views=safe_int(row.get("watchers_count")),
            ),
            url=str(row.get("html_url") or f"https://github.com/{full_name}"),
            extra={
                "language": row.get("language") or "",
                "topics": [str(topic) for topic in row.get("topics") or []],
                "license": license_info.get("spdx_id") or "",
                "pushed_at": row.get("pushed_at") or "",
                "archived": bool(row.get("archived")),
            },
        )

    def search(self, keyword: str, progress: ProgressCallback | None = None) -> list[ContentItem]:
        items: list[ContentItem] = []
        for page in range(1, min(self.config.pages, 10) + 1):
            payload = self.get_json(
                f"{self.api_root}/search/repositories",
                params={"q": keyword, "sort": "stars", "order": "desc", "per_page": min(self.config.page_size, 100), "page": page},
            )
            if payload.get("message"):
                raise CollectorError(f"GitHub API 错误: {payload['message']}")
            rows = payload.get("items") or []
            for row in rows:
                item = self._convert(row, keyword)
                if item:
                    items.append(item)
            if progress:
                progress("github", page, self.config.pages, f"{keyword}: {len(items)} 条")
            if len(rows) < min(self.config.page_size, 100):
                break
        return list({item.id: item for item in items}.values())

    def deep_dive(self, item: ContentItem) -> DeepContent:
        errors: list[str] = []
        full_content = item.summary
        try:
            response = self.request(
                "GET",
                f"{self.api_root}/repos/{item.title}/readme",
                headers={"Accept": "application/vnd.github.raw"},
            )
            readme = response.text.strip()
            if readme:
                full_content = readme[:20_000]
        except Exception as exc:
            errors.append(f"readme: {exc}")
        return DeepContent(base=item, full_content=full_content, comments=[], errors=errors)
