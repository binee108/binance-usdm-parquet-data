from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from time import sleep
from typing import Final

import httpx2

from binance_usdm_parquet_data.archive_download import (
    ArchiveDownloadFailure,
    ArchiveHttpClient,
    DownloadedArchiveFile,
)
from binance_usdm_parquet_data.funding_archive import (
    FundingRateArchiveRequest,
    download_monthly_funding_rate_archive_to_parquet,
)
from binance_usdm_parquet_data.funding_rate import FundingRateClient
from binance_usdm_parquet_data.funding_rest import refresh_funding_rest_day
from binance_usdm_parquet_data.manifest import CollectorFailure, CollectorSource

DECEMBER: Final = 12
_FUNDING_ARCHIVE_ERROR_TYPES = (OSError, ValueError, TypeError, RuntimeError, httpx2.HTTPError)


@dataclass(frozen=True, slots=True)
class FundingRefreshRequest:
    root: Path
    symbol: str
    start_day: date
    end_day: date
    archive_granularity: str
    funding_rest_sleep_seconds: float


@dataclass(frozen=True, slots=True)
class FundingRefreshOutcome:
    success_count: int
    sources: tuple[CollectorSource, ...]
    failures: tuple[CollectorFailure, ...]
    kline_paths: tuple[Path, ...] = ()


def refresh_funding_rate(
    request: FundingRefreshRequest,
    *,
    archive_client: ArchiveHttpClient,
    funding_client: FundingRateClient,
    sleep_fn: Callable[[float], object] | None = None,
) -> FundingRefreshOutcome:
    sources: list[CollectorSource] = []
    failures: list[CollectorFailure] = []
    archive_months = _archive_months(request)
    archive_days = {day for month in archive_months for day in _days(month, _month_end(month))}
    for month in archive_months:
        result = _refresh_monthly_archive(request, archive_client, month)
        match result:
            case CollectorSource():
                sources.append(result)
            case CollectorFailure():
                fallback_days = tuple(
                    day for day in _days(month, _month_end(month)) if day in archive_days
                )
                rest_outcome = _refresh_rest_days(
                    request,
                    funding_client=funding_client,
                    days=fallback_days,
                    sleep_fn=sleep_fn,
                )
                sources.extend(rest_outcome.sources)
                failures.extend(rest_outcome.failures)
    rest_days = tuple(
        day for day in _days(request.start_day, request.end_day) if day not in archive_days
    )
    rest_outcome = _refresh_rest_days(
        request,
        funding_client=funding_client,
        days=rest_days,
        sleep_fn=sleep_fn,
    )
    sources.extend(rest_outcome.sources)
    failures.extend(rest_outcome.failures)
    return FundingRefreshOutcome(
        success_count=len(sources),
        sources=tuple(sources),
        failures=tuple(failures),
    )


def _archive_months(request: FundingRefreshRequest) -> tuple[date, ...]:
    match request.archive_granularity:
        case "daily":
            return ()
        case "monthly":
            return tuple(
                month
                for month in _month_starts(request.start_day, request.end_day)
                if request.start_day <= month and _month_end(month) <= request.end_day
            )
        case _:
            msg = f"unsupported archive granularity: {request.archive_granularity}"
            raise ValueError(msg)


def _refresh_monthly_archive(
    request: FundingRefreshRequest,
    archive_client: ArchiveHttpClient,
    month: date,
) -> CollectorSource | CollectorFailure | None:
    month_key = month.strftime("%Y-%m")
    try:
        result = download_monthly_funding_rate_archive_to_parquet(
            archive_client,
            FundingRateArchiveRequest(symbol=request.symbol, month=month_key),
            request.root,
        )
    except _FUNDING_ARCHIVE_ERROR_TYPES as exc:
        if _is_missing_funding_archive_error(exc):
            return None
        return _funding_archive_exception(request, month_key, exc)
    match result:
        case DownloadedArchiveFile():
            return _funding_archive_source(result, request.symbol, month_key)
        case ArchiveDownloadFailure():
            return _funding_archive_failure(result, request.symbol, month_key)


def _is_missing_funding_archive_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "404" in message and "not found" in message


def _refresh_rest_days(
    request: FundingRefreshRequest,
    *,
    funding_client: FundingRateClient,
    days: tuple[date, ...],
    sleep_fn: Callable[[float], object] | None,
) -> FundingRefreshOutcome:
    sources: list[CollectorSource] = []
    failures: list[CollectorFailure] = []
    rest_sleep = sleep if sleep_fn is None else sleep_fn
    for day in days:
        result = refresh_funding_rest_day(
            root=request.root,
            symbol=request.symbol,
            client=funding_client,
            day=day,
        )
        match result:
            case CollectorSource():
                sources.append(result)
            case CollectorFailure():
                failures.append(result)
        if request.funding_rest_sleep_seconds > 0:
            _ = rest_sleep(request.funding_rest_sleep_seconds)
    return FundingRefreshOutcome(
        success_count=len(sources),
        sources=tuple(sources),
        failures=tuple(failures),
    )


def _funding_archive_source(
    source: DownloadedArchiveFile,
    symbol: str,
    month_key: str,
) -> CollectorSource:
    return CollectorSource(
        dataset="fundingRate",
        symbol=symbol,
        interval=None,
        target_date=month_key,
        source_url=source.source_url,
        output_path=str(source.output_path),
        checksum=source.checksum,
        row_count=source.row_count,
    )


def _funding_archive_failure(
    failure: ArchiveDownloadFailure,
    symbol: str,
    month_key: str,
) -> CollectorFailure:
    return CollectorFailure(
        dataset="fundingRate",
        symbol=symbol,
        interval=None,
        target_date=month_key,
        source_url=failure.source_url,
        attempt_count=1,
        error_code=failure.error_code,
        error_message=failure.error_message,
        retryable=failure.retryable,
    )


def _funding_archive_exception(
    request: FundingRefreshRequest,
    month_key: str,
    exc: OSError | ValueError | TypeError | RuntimeError,
) -> CollectorFailure:
    return CollectorFailure(
        dataset="fundingRate",
        symbol=request.symbol,
        interval=None,
        target_date=month_key,
        source_url="",
        attempt_count=1,
        error_code="funding_archive_exception",
        error_message=str(exc),
        retryable=True,
    )


def _days(start: date, end: date) -> list[date]:
    day_count = (end - start).days
    return [start + timedelta(days=offset) for offset in range(day_count + 1)]


def _month_starts(start: date, end: date) -> list[date]:
    current = start.replace(day=1)
    stop = end.replace(day=1)
    months: list[date] = []
    while current <= stop:
        months.append(current)
        if current.month == DECEMBER:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)
    return months


def _month_end(month: date) -> date:
    if month.month == DECEMBER:
        return date(month.year + 1, 1, 1) - timedelta(days=1)
    return date(month.year, month.month + 1, 1) - timedelta(days=1)
