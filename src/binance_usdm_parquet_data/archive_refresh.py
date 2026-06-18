from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from binance_usdm_parquet_data.archive_download import (
    ArchiveDownloadFailure,
    ArchiveHttpClient,
    DailyArchiveRequest,
    DownloadedArchiveFile,
    MonthlyArchiveRequest,
    download_daily_archive_to_parquet,
    download_monthly_archive_to_parquet,
)
from binance_usdm_parquet_data.manifest import CollectorFailure, CollectorSource
from binance_usdm_parquet_data.premium_index import (
    PREMIUM_INDEX_KLINES_URL,
    PremiumIndexKlineClient,
    backfill_premium_index_klines,
)


@dataclass(frozen=True, slots=True)
class ArchiveRefreshConfig:
    root: Path
    interval: str
    archive_granularity: str


@dataclass(frozen=True, slots=True)
class ArchiveRefreshOutcome:
    source: CollectorSource | None
    failure: CollectorFailure | None
    kline_path: Path | None


@dataclass(frozen=True, slots=True)
class ArchiveRefreshTarget:
    dataset: str
    symbol: str
    target_day: date


def refresh_archive_dataset(
    config: ArchiveRefreshConfig,
    *,
    archive_client: ArchiveHttpClient,
    premium_client: PremiumIndexKlineClient | None,
    target: ArchiveRefreshTarget,
) -> ArchiveRefreshOutcome:
    result = _refresh_archive_item(
        config,
        archive_client,
        target,
    )
    if isinstance(result, DownloadedArchiveFile):
        return ArchiveRefreshOutcome(
            source=_archive_source(
                result,
                target.dataset,
                target.symbol,
                config.interval,
                target.target_day,
            ),
            failure=None,
            kline_path=result.output_path if target.dataset == "klines" else None,
        )
    try:
        fallback = _refresh_premium_fallback(
            config,
            premium_client,
            target,
        )
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        return ArchiveRefreshOutcome(
            source=None,
            failure=CollectorFailure(
                dataset=target.dataset,
                symbol=target.symbol,
                interval=config.interval,
                target_date=target.target_day.isoformat(),
                source_url=PREMIUM_INDEX_KLINES_URL,
                attempt_count=1,
                error_code="premium_fallback_exception",
                error_message=str(exc),
                retryable=True,
            ),
            kline_path=None,
        )
    if fallback is not None:
        return ArchiveRefreshOutcome(source=fallback, failure=None, kline_path=None)
    failure = (
        _archive_failure(result, target.dataset, target.symbol, config.interval, target.target_day)
        if isinstance(result, ArchiveDownloadFailure)
        else result
    )
    return ArchiveRefreshOutcome(source=None, failure=failure, kline_path=None)


def _refresh_archive_item(
    config: ArchiveRefreshConfig,
    archive_client: ArchiveHttpClient,
    target: ArchiveRefreshTarget,
) -> DownloadedArchiveFile | ArchiveDownloadFailure | CollectorFailure:
    if config.archive_granularity not in {"daily", "monthly"}:
        msg = f"unsupported archive granularity: {config.archive_granularity}"
        raise ValueError(msg)
    try:
        match config.archive_granularity:
            case "monthly":
                return download_monthly_archive_to_parquet(
                    archive_client,
                    MonthlyArchiveRequest(
                        dataset=target.dataset,
                        symbol=target.symbol,
                        interval=config.interval,
                        month=target.target_day.strftime("%Y-%m"),
                    ),
                    config.root,
                )
            case "daily":
                return download_daily_archive_to_parquet(
                    archive_client,
                    DailyArchiveRequest(
                        target.dataset,
                        target.symbol,
                        config.interval,
                        target.target_day,
                    ),
                    config.root,
                )
            case unreachable:
                raise AssertionError(unreachable)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        return CollectorFailure(
            dataset=target.dataset,
            symbol=target.symbol,
            interval=config.interval,
            target_date=target.target_day.isoformat(),
            source_url="",
            attempt_count=1,
            error_code="archive_exception",
            error_message=str(exc),
            retryable=True,
        )


def _refresh_premium_fallback(
    config: ArchiveRefreshConfig,
    premium_client: PremiumIndexKlineClient | None,
    target: ArchiveRefreshTarget,
) -> CollectorSource | None:
    if target.dataset != "premiumIndexKlines" or premium_client is None:
        return None
    backfill = backfill_premium_index_klines(
        root=config.root,
        symbol=target.symbol,
        day=target.target_day,
        interval=config.interval,
        client=premium_client,
    )
    return CollectorSource(
        dataset=target.dataset,
        symbol=target.symbol,
        interval=config.interval,
        target_date=target.target_day.isoformat(),
        source_url=backfill.source_url,
        output_path=str(backfill.output_path),
        checksum=None,
        row_count=backfill.row_count,
    )


def _archive_failure(
    failure: ArchiveDownloadFailure,
    dataset: str,
    symbol: str,
    interval: str,
    day: date,
) -> CollectorFailure:
    return CollectorFailure(
        dataset=dataset,
        symbol=symbol,
        interval=interval,
        target_date=day.isoformat(),
        source_url=failure.source_url,
        attempt_count=1,
        error_code=failure.error_code,
        error_message=failure.error_message,
        retryable=failure.retryable,
    )


def _archive_source(
    source: DownloadedArchiveFile,
    dataset: str,
    symbol: str,
    interval: str,
    day: date,
) -> CollectorSource:
    return CollectorSource(
        dataset=dataset,
        symbol=symbol,
        interval=interval,
        target_date=day.isoformat(),
        source_url=source.source_url,
        output_path=str(source.output_path),
        checksum=source.checksum,
        row_count=source.row_count,
    )
