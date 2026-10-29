"""Interactive and CLI entry point for collection, screening and L2 extraction."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from analyzer import Analyzer
from cache import HarvestCache
from config import SUPPORTED_PLATFORMS, Settings
from deep_dive import DeepDiveRunner
from exceptions import CollectorError
from ml_dataset import build_ml_dataset
from models import ContentItem
from platforms import create_collectors
from reporting import write_reports
from storage import DatasetStorage
from utils import expand_search_queries, merge_items_by_url, parse_iso, split_datetime_range


def configure_logging(settings: Settings) -> Path:
    settings.data_root.mkdir(parents=True, exist_ok=True)
    log_dir = settings.data_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    # Try to force UTF-8 for console output on Windows terminals.
    try:
        import io

        console_stream: Any = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", line_buffering=True)
    except Exception:
        console_stream = sys.stderr
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(console_stream)],
        force=True,
    )
    return log_path


class ProgressDisplay:
    def __init__(self) -> None:
        self.started: dict[str, float] = {}
        self.last_reported: dict[str, int] = {}
        self.last_line_length = 0

    @staticmethod
    def _duration(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s"
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"

    def __call__(self, platform: str, current: int, total: int | None, message: str) -> None:
        key = "deep" if message.startswith("deep:") else f"collect:{platform}:{message.split(':', 1)[0]}"
        started = self.started.setdefault(key, time.monotonic())
        elapsed = time.monotonic() - started
        eta = "?"
        percent = ""
        if total and current:
            eta = self._duration(max(0.0, elapsed / current * (total - current)))
            percent = f" {current / total:6.1%}"
        line = f"[{platform:9}] {current}/{total or '?'}{percent} | ETA {eta:>6} | {message}"
        if sys.stdout.isatty():
            print("\r" + line.ljust(self.last_line_length), end="", flush=True)
            self.last_line_length = max(self.last_line_length, len(line))
            if total and current >= total:
                print()
                self.last_line_length = 0
        elif (current == 1 and self.last_reported.get(key) != current) or (total and current >= total):
            print(line, flush=True)
        self.last_reported[key] = current


def parse_keywords(raw: str) -> list[str]:
    values = [part.strip() for part in raw.replace("，", ",").split(",") if part.strip()]
    return list(dict.fromkeys(values))


def parse_platforms(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(SUPPORTED_PLATFORMS)
    values = [part.strip().lower() for part in raw.replace("，", ",").split(",") if part.strip()]
    unknown = sorted(set(values).difference(SUPPORTED_PLATFORMS))
    if unknown:
        raise ValueError(f"不支持的平台: {', '.join(unknown)}")
    return list(dict.fromkeys(values))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="社交媒体公开舆情数据采集与深度分析")
    parser.add_argument("--keywords", help="关键词，逗号分隔")
    parser.add_argument(
        "--platforms",
        default=None,
        help=f"all 或逗号分隔的平台列表，如 {', '.join(SUPPORTED_PLATFORMS)}",
    )
    parser.add_argument("--mode", choices=("quick", "full"), default=None)
    parser.add_argument("--mock", action="store_true", help="使用确定性离线 mock 数据")
    parser.add_argument("--non-interactive", action="store_true", help="缺少参数时直接报错，不提示输入")
    parser.add_argument("--include-seen", action="store_true", help="不使用跨运行 URL 缓存过滤")
    parser.add_argument("--since", help="只保留该 ISO8601 时间之后发布的内容")
    parser.add_argument("--until", help="只保留该 ISO8601 时间之前发布的内容")
    parser.add_argument("--export", choices=("none", "csv", "parquet", "huggingface", "ml"), default="none")
    parser.add_argument("--run-id", help="自定义数据集时间戳/运行标识（便于测试）")
    parser.add_argument("--split-days", type=int, default=0, help="将 --since/--until 区间按 N 天拆分后分片采集")
    parser.add_argument("--dedup-by-url", action="store_true", help="跨关键词按 URL 全局去重，合并 matched_search_queries")
    return parser


def resolve_inputs(args: argparse.Namespace) -> tuple[list[str], list[str], str]:
    if not args.keywords and not args.non_interactive:
        args.keywords = input("请输入调研关键词（多个用逗号分隔）：").strip()
    if not args.platforms and not args.non_interactive:
        args.platforms = input(f"请选择平台（all 或 {', '.join(SUPPORTED_PLATFORMS)}）：").strip() or "all"
    if not args.mode and not args.non_interactive:
        args.mode = input("请选择模式（quick/full）：").strip().lower() or "quick"
    if not args.keywords:
        raise ValueError("至少需要一个关键词（--keywords）")
    if not args.platforms:
        raise ValueError("需要指定平台（--platforms）")
    mode = args.mode or "quick"
    if mode not in {"quick", "full"}:
        raise ValueError("模式必须为 quick 或 full")
    keywords, platforms = parse_keywords(args.keywords), parse_platforms(args.platforms)
    if not keywords or not platforms:
        raise ValueError("关键词和平台不能为空")
    return keywords, platforms, mode


def _classify_error(exc: Exception) -> str:
    """Classify an exception into a coarse error type for reporting."""
    chain = [exc]
    if exc.__cause__:
        chain.append(exc.__cause__)
    if exc.__context__ and exc.__context__ is not exc.__cause__:
        chain.append(exc.__context__)
    names = {type(item).__name__ for item in chain}
    if "ChallengeDetected" in names:
        return "challenge"
    if names & {"ConnectError", "NetworkError", "ReadTimeout", "ConnectTimeout", "TimeoutException", "RemoteProtocolError"}:
        return "network"
    if "HTTPStatusError" in names:
        return "http"
    if "AuthenticationRequired" in names:
        return "auth"
    if "CollectorError" in names:
        message = str(exc).lower()
        if "quotaexceeded" in message or "quota" in message or "配额" in message:
            return "quota"
        if any(marker in message for marker in ("拒绝", "403", "401", "banned", "unauthorized")):
            return "auth_or_access"
        if any(marker in message for marker in ("captcha", "验证", "challenge", "人机")):
            return "challenge"
        return "platform"
    return "unknown"


def _classify_message(message: str) -> str:
    """Classify a recorded message. Partial failures only keep the text."""
    return _classify_error(CollectorError(message))


# 只有「平台在明确拒绝我们」的错误才触发熔断：验证码/风控、登录态失效、配额耗尽。
# 解析失败、限流和网络抖动会自行恢复，把剩余关键词一起跳过反而是丢数据。
BLOCKING_ERROR_TYPES = {"challenge", "auth", "auth_or_access", "quota"}


def _collect_window(
    args: argparse.Namespace,
    settings: Settings,
    platforms: list[str],
    collectors: dict,
    cache: HarvestCache | None,
    keywords: list[str],
    cutoff: datetime | None,
    until: datetime | None,
    progress: ProgressDisplay,
    errors: list[dict],
    warnings: list[str],
    status: dict[str, dict] | None = None,
    storage=None,
) -> list[ContentItem]:
    """Collect one time window and return normalized items."""
    log = logging.getLogger("main")
    threshold = settings.platform_failure_threshold
    all_items: list[ContentItem] = []
    for platform in platforms:
        collected: list[ContentItem] = []
        collector = collectors.get(platform)
        if not collector:
            continue
        if status is None:
            status = {}
        platform_status = status.setdefault(
            platform,
            {"queries": 0, "collected": 0, "cache_filtered": 0, "partial_errors": 0, "skipped_queries": 0},
        )
        consecutive_failures = 0
        for keyword in keywords:
            if threshold > 0 and consecutive_failures >= threshold:
                # The platform is blocked (captcha / quota / login): further
                # queries only burn time and fill debug/ with identical dumps.
                platform_status["skipped_queries"] += 1
                continue
            queries = [keyword] if args.mock else expand_search_queries(keyword, settings.keyword_expansions)
            for query in queries:
                platform_status["queries"] += 1
                failed = False
                found: list[ContentItem] = []
                try:
                    found = collector.search(query, progress)
                    for item in found:
                        item.keyword = keyword
                        matched = item.extra.setdefault("matched_search_queries", [])
                        if query not in matched:
                            matched.append(query)
                    collected.extend(found)
                except Exception as exc:
                    error_type = _classify_error(exc)
                    # Only a platform that is actively refusing us counts toward
                    # the circuit breaker; a parse miss or rate limit must not
                    # cause the remaining keywords to be skipped.
                    failed = error_type in BLOCKING_ERROR_TYPES
                    errors.append(
                        {
                            "stage": "collect",
                            "platform": platform,
                            "keyword": keyword,
                            "query": query,
                            "error": str(exc),
                            "error_type": error_type,
                        }
                    )
                    log.exception("采集失败 %s/%s (query=%s)", platform, keyword, query)
                for note in collector.drain_partial_errors():
                    platform_status["partial_errors"] += 1
                    if note.get("blocking"):
                        failed = True
                    message = str(note.get("error", ""))
                    errors.append(
                        {
                            "platform": platform,
                            "stage": "collect_partial",
                            "collector_stage": note.get("stage", ""),
                            "keyword": keyword,
                            "query": query,
                            "error": message,
                            "blocking": bool(note.get("blocking")),
                            "error_type": _classify_message(message),
                        }
                    )
                if failed and not found:
                    consecutive_failures += 1
                else:
                    consecutive_failures = 0
            # 增量落盘：每跑完一个关键词就把该平台已采到的写一次。
            # 原先整窗采完才落盘，进程被杀（超时/风控/休眠）会丢掉全部成果；
            # 逐关键词落盘后，硬杀最多损失当前这一个关键词。
            if storage is not None and collected:
                try:
                    storage.save_raw(platform, collected, replace=True)
                except Exception:
                    log.debug("增量落盘失败（不影响采集）", exc_info=True)
        if platform_status["skipped_queries"]:
            message = (
                f"{platform} 连续失败 {threshold} 次后已跳过本次运行剩余的 "
                f"{platform_status['skipped_queries']} 个关键词（避免继续空转与重复写诊断文件）"
            )
            warnings.append(message)
            log.warning(message)
        # Deduplicate the same result inside this run but retain one row per keyword.
        collected = list({(item.id, item.keyword): item for item in collected}.values())
        before_filter = len(collected)
        if cutoff:
            collected = [
                item
                for item in collected
                if (parse_iso(item.published_at) or datetime.min.replace(tzinfo=timezone.utc)) >= cutoff
            ]
        if until:
            collected = [
                item
                for item in collected
                if (parse_iso(item.published_at) or datetime.max.replace(tzinfo=timezone.utc)) <= until
            ]
        if cache is not None:
            collected, skipped = cache.filter_new(collected)
            platform_status["cache_filtered"] += skipped
            if skipped:
                warnings.append(
                    f"{platform} 由增量缓存过滤 {skipped} 条已见 URL（这是增量去重而非采集失败；"
                    "如需重新采集请加 --include-seen）"
                )
                if not collected:
                    warnings.append(
                        f"{platform} 本次 0 条全部来自增量缓存过滤：数据其实采到了，只是此前已入库。"
                        "这些记录仍保留在数据集中，未丢失。"
                    )
        platform_status["collected"] += len(collected)
        all_items.extend(collected)
        log.info(
            "窗口采集完成 platform=%s items=%d (过滤前 %d)", platform, len(collected), before_filter
        )
        if cache is not None:
            cache.mark(collected)
    return all_items


def _safe_save(storage, platform: str, items: list) -> None:
    """结尾落盘的保护壳：不能把增量落盘的成果覆盖没了。

    背景：_collect_window 每跑完一个关键词就 save_raw(replace=True) 一次；
    但关键词循环结束后会过一遍 cache.filter_new，若这批 URL 此前都已入库,
    collected 会被清空，结尾再用 replace=True 写空列表就把成果抹掉了。

    2026-10-12 实测：bilibili 采到 2000 条、全部 HTTP 200，最终落盘 0 条，
    正是踩了这条。weibo/zhihu/tieba 侥幸保住数据，只是因为它们被超时杀在了
    这一步之前。

    规则：待写为空、而磁盘上该平台已有数据时，保留既有数据不覆盖。
    """
    if not items and storage is not None:
        try:
            on_disk = storage.load_raw(platform)
        except Exception:
            on_disk = []
        if on_disk:
            log.warning(
                "结尾落盘为空但磁盘已有 %d 条 %s（通常由增量缓存过滤所致），"
                "保留既有数据不覆盖",
                len(on_disk),
                platform,
            )
            return
    storage.save_raw(platform, items, replace=True)


def run(args: argparse.Namespace) -> Path:
    started = time.monotonic()
    keywords, platforms, mode = resolve_inputs(args)
    cutoff = parse_iso(args.since) if args.since else None
    until = parse_iso(args.until) if args.until else None
    if args.since and cutoff is None:
        raise ValueError("--since 必须是有效 ISO8601 时间")
    if args.until and until is None:
        raise ValueError("--until 必须是有效 ISO8601 时间")
    if cutoff and until and cutoff > until:
        raise ValueError("--since 不能晚于 --until")

    split_days = args.split_days
    if split_days:
        if not cutoff:
            raise ValueError("--split-days 需要配合 --since 使用")
        if split_days <= 0:
            raise ValueError("--split-days 必须大于 0")
        until = until or datetime.now(timezone.utc)

    settings = Settings()
    log_path = configure_logging(settings)
    log = logging.getLogger("main")
    storage = DatasetStorage(settings, keywords, args.run_id)
    settings.write_snapshot(storage.root / "config_snapshot.json")
    log.info(
        "开始运行 keywords=%s platforms=%s mode=%s mock=%s split_days=%s dedup_by_url=%s",
        keywords,
        platforms,
        mode,
        args.mock,
        split_days,
        args.dedup_by_url,
    )
    progress = ProgressDisplay()
    errors: list[dict] = []
    warnings: list[str] = []

    collectors = {}
    for platform in platforms:
        missing = settings.validate_platform(platform, args.mock)
        if missing:
            message = f"{platform} 已跳过：缺少 {', '.join(missing)}"
            warnings.append(message)
            log.warning(message)
            continue
        try:
            collectors.update(create_collectors(settings, [platform], mock=args.mock))
        except Exception as exc:
            errors.append({"stage": "initialize", "platform": platform, "error": str(exc), "error_type": _classify_error(exc)})
            log.exception("初始化 %s 失败", platform)

    cache = (
        None
        if args.mock or args.include_seen or not settings.incremental
        else HarvestCache(settings.data_root, ttl_days=settings.incremental_ttl_days)
    )

    # Re-running into the same dataset directory must accumulate, not restart:
    # load whatever this directory already holds so the union is analysed and
    # written back (an empty collection can no longer erase earlier results).
    carried: dict[str, list[ContentItem]] = storage.load_all_raw()
    carried_items = [item for items in carried.values() for item in items]
    if carried_items:
        log.info("数据集目录已存在，载入既有记录 %d 条参与合并", len(carried_items))
        warnings.append(f"复用已有数据集目录，已载入 {len(carried_items)} 条既有记录并合并新采集结果")

    status: dict[str, dict] = {}
    all_items: list[ContentItem] = []
    if split_days and cutoff and until:
        windows = split_datetime_range(cutoff, until, split_days)
        log.info("时间窗口拆分: %d 个窗口", len(windows))
        for window_idx, (window_since, window_until) in enumerate(windows, 1):
            log.info("开始第 %d/%d 个窗口: %s -> %s", window_idx, len(windows), window_since.isoformat(), window_until.isoformat())
            window_items = _collect_window(
                args, settings, platforms, collectors, cache, keywords, window_since, window_until,
                progress, errors, warnings, status, storage,
            )
            all_items.extend(window_items)
            for platform in platforms:
                platform_items = [item for item in window_items if item.platform == platform]
                storage.save_raw(platform, platform_items, replace=(window_idx == 1))
    else:
        all_items = _collect_window(
            args, settings, platforms, collectors, cache, keywords, cutoff, until, progress, errors, warnings, status, storage
        )

    if carried_items:
        # Rows from earlier runs into this same directory join the union first,
        # so every write below stays a plain replace of the complete dataset.
        fresh_keys = {(item.platform, item.id, item.keyword) for item in all_items}
        merged_carried = [item for item in carried_items if (item.platform, item.id, item.keyword) not in fresh_keys]
        all_items.extend(merged_carried)
        log.info("合并既有记录后总量: %d 条（其中本次新增 %d 条）", len(all_items), len(all_items) - len(merged_carried))
        for platform in platforms:
            _safe_save(storage, platform, [item for item in all_items if item.platform == platform])

    if not (split_days and cutoff and until):
        for platform in platforms:
            _safe_save(storage, platform, [item for item in all_items if item.platform == platform])

    if args.dedup_by_url:
        before = len(all_items)
        all_items = merge_items_by_url(all_items)
        log.info("全局 URL 去重: %d -> %d", before, len(all_items))
        warnings.append(f"全局 URL 去重: {before} -> {len(all_items)}")
        for platform in platforms:
            platform_items = [item for item in all_items if item.platform == platform]
            _safe_save(storage, platform, platform_items)

    analyzed, analysis = Analyzer(settings).analyze(all_items)
    by_platform: dict[str, list[ContentItem]] = defaultdict(list)
    for item in analyzed:
        by_platform[item.platform].append(item)
    for platform in platforms:
        _safe_save(storage, platform, by_platform.get(platform, []))

    deep_rows: dict[str, list[dict]] = {}
    deep_failures: list[dict] = []
    if mode == "full" and analyzed:
        already_deep = storage.deep_dive_keys()
        to_deepen = [
            item for item in analyzed
            if item.deep_dive_flag and (item.id, item.keyword) not in already_deep
        ]
        if len(to_deepen) != len([item for item in analyzed if item.deep_dive_flag]):
            log.info("跳过 %d 条已存在 L2 记录的内容", len([i for i in analyzed if i.deep_dive_flag]) - len(to_deepen))
        deep_rows, deep_failures = DeepDiveRunner(collectors, storage).run(to_deepen, progress)
        errors.extend({"stage": "deep_dive", **failure} for failure in deep_failures)
    for platform in platforms:
        if platform not in deep_rows:
            storage.save_deep(platform, [], replace=True)

    if args.export != "none":
        if args.export == "ml":
            try:
                ml_result = build_ml_dataset(storage.root, deep_only=(mode == "full"), dedup_text=True)
                log.info("ML 数据集已生成: %s", ml_result["out_dir"])
            except Exception as exc:
                errors.append({"stage": "export", "platform": "all", "error": str(exc), "error_type": _classify_error(exc)})
                log.exception("ML 数据集导出失败")
        else:
            for platform in platforms:
                try:
                    if args.export == "csv":
                        storage.export_csv(platform)
                        if mode == "full":
                            storage.export_csv(platform, deep=True)
                    elif args.export == "parquet":
                        storage.export_parquet(platform)
                        if mode == "full":
                            storage.export_parquet(platform, deep=True)
                    else:
                        storage.export_huggingface(platform)
                        if mode == "full":
                            storage.export_huggingface(platform, deep=True)
                except Exception as exc:
                    errors.append({"stage": "export", "platform": platform, "error": str(exc), "error_type": _classify_error(exc)})
                    log.exception("导出失败 %s", platform)

    deep_count = sum(len(rows) for rows in deep_rows.values())
    final_counts = Counter(item.platform for item in analyzed)
    platform_status = {
        platform: {
            "skipped_reason": next(
                (w for w in warnings if w.startswith(f"{platform} 已跳过")), ""
            ),
            **status.get(platform, {}),
            # 数据集内该平台的最终条数（含上一轮既有记录）。
            "items": final_counts.get(platform, 0),
        }
        for platform in platforms
    }
    empty_platforms = [
        platform
        for platform, row in platform_status.items()
        if row["items"] == 0 and not row["skipped_reason"]
    ]
    if empty_platforms:
        warnings.append(
            "以下平台本次没有采集到任何内容："
            + "、".join(empty_platforms)
            + "。请查看 report/overview.json 的 platform_status 与 errors 字段确认原因"
            "（增量缓存过滤 / 风控验证 / 缺少凭据 / 接口变更各不相同）。"
        )
    run_report = {
        "keywords": keywords,
        "platforms": platforms,
        "mode": mode,
        "mock": args.mock,
        "query_expansion_enabled": not args.mock,
        "configured_keyword_expansion_count": sum(len(values) for values in settings.keyword_expansions.values()),
        "split_days": split_days,
        "dedup_by_url": args.dedup_by_url,
        "total_items": len(analyzed),
        "carried_over_items": len(carried_items),
        # 本次新生成的 L2 条数，以及数据集目录里 L2 的总条数（含既往运行）。
        "deep_items": deep_count,
        "deep_items_total": len(storage.deep_dive_keys()),
        "platform_status": platform_status,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "warnings": warnings,
        "errors": errors,
        "log_file": str(log_path),
    }
    log.info("运行完成 L1=%d L2=%d errors=%d elapsed=%.3fs", len(analyzed), deep_count, len(errors), run_report["elapsed_seconds"])
    write_reports(storage, analyzed, analysis, run_report)
    for collector in collectors.values():
        collector.close()
    print(f"全部完成！L1={len(analyzed)}，L2={deep_count}，错误={len(errors)}")
    print(f"数据集已保存至：{storage.root}")
    return storage.root


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args)
        return 0
    except (KeyboardInterrupt, EOFError):
        print("\n已取消。", file=sys.stderr)
        return 130
    except Exception as exc:
        logging.getLogger("main").exception("运行失败")
        print(f"运行失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
