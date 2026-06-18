from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from binance_usdm_parquet_data.archive_download import ArchiveHttpClient
from binance_usdm_parquet_data.archive_refresh import (
    ArchiveRefreshConfig,
    ArchiveRefreshTarget,
    refresh_archive_dataset,
)
from binance_usdm_parquet_data.duckdb_optimize import optimize_klines
from binance_usdm_parquet_data.funding_rate import FundingRateClient
from binance_usdm_parquet_data.funding_refresh import (
    FundingRefreshOutcome,
    FundingRefreshRequest,
    refresh_funding_rate,
)
from binance_usdm_parquet_data.locks import shared_file_lock
from binance_usdm_parquet_data.manifest import (
    CollectorFailure,
    CollectorRun,
    CollectorSource,
    DatasetFreshness,
    ManifestStore,
)
from binance_usdm_parquet_data.premium_index import PremiumIndexKlineClient

DECEMBER = 12


@dataclass(frozen=True, slots=True)
class RefreshRequest:
    root: Path
    symbols: tuple[str, ...]
    start_day: date
    end_day: date
    datasets: tuple[str, ...] = ("klines", "premiumIndexKlines", "fundingRate")
    interval: str = "1m"
    optimize: bool = True
    archive_granularity: str = "daily"
    max_concurrent_downloads: int = 4
    http_timeout_seconds: float = 30.0
    funding_rest_sleep_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class RefreshResult:
    run_id: str
    status: str
    success_count: int
    failure_count: int


@dataclass(frozen=True, slots=True)
class RefreshClients:
    archive_client: ArchiveHttpClient
    funding_client: FundingRateClient
    premium_client: PremiumIndexKlineClient | None


@dataclass(frozen=True, slots=True)
class DatasetRefreshOutcome:
    success_count: int
    sources: tuple[CollectorSource, ...]
    failures: tuple[CollectorFailure, ...]
    kline_paths: tuple[Path, ...]


def refresh_market_data(
    request: RefreshRequest,
    *,
    archive_client: ArchiveHttpClient,
    funding_client: FundingRateClient,
    premium_client: PremiumIndexKlineClient | None = None,
) -> RefreshResult:
    _validate_request(request)
    refresh_marker = request.root / "manifests" / "binance" / "usdm" / "refresh.json"
    refresh_lock = refresh_marker.with_name(".refresh.lock")
    with shared_file_lock(refresh_marker, "refresh_market_data", lock_path=refresh_lock):
        return _refresh_market_data_unlocked(
            request,
            clients=RefreshClients(
                archive_client=archive_client,
                funding_client=funding_client,
                premium_client=premium_client,
            ),
        )


def _validate_request(request: RefreshRequest) -> None:
    if request.max_concurrent_downloads < 1:
        msg = "max_concurrent_downloads must be at least 1"
        raise ValueError(msg)
    if request.http_timeout_seconds <= 0:
        msg = "http_timeout_seconds must be positive"
        raise ValueError(msg)
    if request.funding_rest_sleep_seconds < 0:
        msg = "funding_rest_sleep_seconds must be non-negative"
        raise ValueError(msg)


def _refresh_market_data_unlocked(
    request: RefreshRequest,
    *,
    clients: RefreshClients,
) -> RefreshResult:
    run_id = str(uuid4())
    started_at = datetime.now(UTC)
    downloaded_klines: dict[str, list[Path]] = {symbol: [] for symbol in request.symbols}
    failures: list[CollectorFailure] = []
    sources: list[CollectorSource] = []
    success_count = 0
    for symbol in request.symbols:
        for dataset in request.datasets:
            outcome = _refresh_dataset_items(
                request,
                clients=clients,
                symbol=symbol,
                dataset=dataset,
            )
            success_count += outcome.success_count
            sources.extend(outcome.sources)
            failures.extend(outcome.failures)
            downloaded_klines[symbol].extend(outcome.kline_paths)
    if request.optimize:
        _optimize_downloaded_klines(request, downloaded_klines)
    status = "succeeded" if not failures else "failed"
    finished_at = datetime.now(UTC)
    ManifestStore(request.root).publish_status(
        last_run=CollectorRun(
            run_id=run_id,
            mode="manual",
            status=status,
            started_at=started_at.isoformat(),
            finished_at=finished_at.isoformat(),
            symbol_count=len(request.symbols),
            item_count=success_count + len(failures),
            success_count=success_count,
            failure_count=len(failures),
            last_error=None if not failures else failures[0].error_message,
        ),
        freshness=_freshness(request, success_count),
        failures=tuple(failures),
        sources=tuple(sources),
    )
    return RefreshResult(
        run_id=run_id,
        status=status,
        success_count=success_count,
        failure_count=len(failures),
    )


def _refresh_dataset_items(
    request: RefreshRequest,
    *,
    clients: RefreshClients,
    symbol: str,
    dataset: str,
) -> DatasetRefreshOutcome | FundingRefreshOutcome:
    if dataset == "fundingRate":
        return refresh_funding_rate(
            FundingRefreshRequest(
                root=request.root,
                symbol=symbol,
                start_day=request.start_day,
                end_day=request.end_day,
                archive_granularity=request.archive_granularity,
                funding_rest_sleep_seconds=request.funding_rest_sleep_seconds,
            ),
            archive_client=clients.archive_client,
            funding_client=clients.funding_client,
        )
    archive_sources: list[CollectorSource] = []
    failures: list[CollectorFailure] = []
    kline_paths: list[Path] = []
    config = ArchiveRefreshConfig(
        root=request.root,
        interval=request.interval,
        archive_granularity=request.archive_granularity,
    )
    for target_day in _archive_target_days(request):
        outcome = refresh_archive_dataset(
            config,
            archive_client=clients.archive_client,
            premium_client=clients.premium_client,
            target=ArchiveRefreshTarget(
                dataset=dataset,
                symbol=symbol,
                target_day=target_day,
            ),
        )
        if outcome.failure is not None:
            failures.append(outcome.failure)
        if outcome.source is not None:
            archive_sources.append(outcome.source)
        if outcome.kline_path is not None:
            kline_paths.append(outcome.kline_path)
    return DatasetRefreshOutcome(
        success_count=len(archive_sources),
        sources=tuple(archive_sources),
        failures=tuple(failures),
        kline_paths=tuple(kline_paths),
    )


def _optimize_downloaded_klines(
    request: RefreshRequest,
    downloaded_klines: dict[str, list[Path]],
) -> None:
    for symbol, paths in downloaded_klines.items():
        if not paths:
            continue
        _ = optimize_klines(
            raw_files=tuple(paths),
            output_root=request.root / "parbp_optimized" / "binance" / "futures",
            symbol=symbol,
            interval=request.interval,
        )


def _freshness(request: RefreshRequest, success_count: int) -> tuple[DatasetFreshness, ...]:
    if success_count == 0:
        return ()
    return tuple(
        DatasetFreshness(
            dataset=dataset,
            interval=None if dataset == "fundingRate" else request.interval,
            symbol_count=len(request.symbols),
            latest_complete_utc_day=request.end_day.isoformat(),
        )
        for dataset in request.datasets
    )


def _days(start: date, end: date) -> list[date]:
    day_count = (end - start).days
    return [start + timedelta(days=offset) for offset in range(day_count + 1)]


def _archive_target_days(request: RefreshRequest) -> tuple[date, ...]:
    match request.archive_granularity:
        case "daily":
            return tuple(_days(request.start_day, request.end_day))
        case "monthly":
            return tuple(_month_starts(request.start_day, request.end_day))
        case _:
            msg = f"unsupported archive granularity: {request.archive_granularity}"
            raise ValueError(msg)


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
