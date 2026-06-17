from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import sleep
from uuid import uuid4

import polars as pl

from binance_usdm_parquet_data.archive_download import (
    ArchiveDownloadFailure,
    ArchiveHttpClient,
    DailyArchiveRequest,
    DownloadedArchiveFile,
    MonthlyArchiveRequest,
    download_daily_archive_to_parquet,
    download_monthly_archive_to_parquet,
)
from binance_usdm_parquet_data.duckdb_optimize import optimize_klines
from binance_usdm_parquet_data.funding_rate import (
    FundingRateClient,
    FundingRateRecord,
    FundingRateRequest,
    collect_funding_rates,
)
from binance_usdm_parquet_data.locks import shared_file_lock
from binance_usdm_parquet_data.manifest import (
    CollectorFailure,
    CollectorRun,
    CollectorSource,
    DatasetFreshness,
    ManifestStore,
)
from binance_usdm_parquet_data.premium_index import (
    PREMIUM_INDEX_KLINES_URL,
    PremiumIndexKlineClient,
    backfill_premium_index_klines,
)
from binance_usdm_parquet_data.storage_keys import symbol_storage_key

FUNDING_RATE_SOURCE_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
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
class ArchiveRefreshOutcome:
    source: CollectorSource | None
    failure: CollectorFailure | None
    kline_path: Path | None


@dataclass(frozen=True, slots=True)
class ArchiveRefreshCommand:
    request: RefreshRequest
    archive_client: ArchiveHttpClient
    premium_client: PremiumIndexKlineClient | None
    dataset: str
    symbol: str
    target_day: date


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
    if request.max_concurrent_downloads < 1:
        msg = "max_concurrent_downloads must be at least 1"
        raise ValueError(msg)
    if request.http_timeout_seconds <= 0:
        msg = "http_timeout_seconds must be positive"
        raise ValueError(msg)
    if request.funding_rest_sleep_seconds < 0:
        msg = "funding_rest_sleep_seconds must be non-negative"
        raise ValueError(msg)
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
) -> DatasetRefreshOutcome:
    if dataset == "fundingRate":
        funding_sources = tuple(
            _refresh_funding_item(request, clients.funding_client, symbol, day)
            for day in _days(request.start_day, request.end_day)
        )
        return DatasetRefreshOutcome(
            success_count=len(funding_sources),
            sources=funding_sources,
            failures=(),
            kline_paths=(),
        )
    archive_sources: list[CollectorSource] = []
    failures: list[CollectorFailure] = []
    kline_paths: list[Path] = []
    for target_day in _archive_target_days(request):
        outcome = _handle_archive_refresh(
            ArchiveRefreshCommand(
                request=request,
                archive_client=clients.archive_client,
                premium_client=clients.premium_client,
                dataset=dataset,
                symbol=symbol,
                target_day=target_day,
            )
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


def _refresh_funding_item(
    request: RefreshRequest,
    funding_client: FundingRateClient,
    symbol: str,
    day: date,
) -> CollectorSource:
    source = _refresh_funding(request, funding_client, symbol, day)
    if request.funding_rest_sleep_seconds > 0:
        sleep(request.funding_rest_sleep_seconds)
    return source


def _handle_archive_refresh(command: ArchiveRefreshCommand) -> ArchiveRefreshOutcome:
    result = _refresh_archive_item(
        command.request,
        command.archive_client,
        command.dataset,
        command.symbol,
        command.target_day,
    )
    if isinstance(result, DownloadedArchiveFile):
        return ArchiveRefreshOutcome(
            source=_archive_source(
                result,
                command.dataset,
                command.symbol,
                command.request.interval,
                command.target_day,
            ),
            failure=None,
            kline_path=result.output_path if command.dataset == "klines" else None,
        )
    try:
        fallback = _refresh_premium_fallback(
            command.request,
            command.premium_client,
            command.dataset,
            command.symbol,
            command.target_day,
        )
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        return ArchiveRefreshOutcome(
            source=None,
            failure=CollectorFailure(
                dataset=command.dataset,
                symbol=command.symbol,
                interval=command.request.interval,
                target_date=command.target_day.isoformat(),
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
        _archive_failure(
            result,
            command.dataset,
            command.symbol,
            command.request.interval,
            command.target_day,
        )
        if isinstance(result, ArchiveDownloadFailure)
        else result
    )
    return ArchiveRefreshOutcome(source=None, failure=failure, kline_path=None)


def _refresh_premium_fallback(
    request: RefreshRequest,
    premium_client: PremiumIndexKlineClient | None,
    dataset: str,
    symbol: str,
    day: date,
) -> CollectorSource | None:
    if dataset != "premiumIndexKlines" or premium_client is None:
        return None
    backfill = backfill_premium_index_klines(
        root=request.root,
        symbol=symbol,
        day=day,
        interval=request.interval,
        client=premium_client,
    )
    return CollectorSource(
        dataset=dataset,
        symbol=symbol,
        interval=request.interval,
        target_date=day.isoformat(),
        source_url=backfill.source_url,
        output_path=str(backfill.output_path),
        checksum=None,
        row_count=backfill.row_count,
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


def _refresh_archive_item(
    request: RefreshRequest,
    archive_client: ArchiveHttpClient,
    dataset: str,
    symbol: str,
    day: date,
) -> DownloadedArchiveFile | ArchiveDownloadFailure | CollectorFailure:
    try:
        if request.archive_granularity == "monthly":
            return download_monthly_archive_to_parquet(
                archive_client,
                MonthlyArchiveRequest(
                    dataset=dataset,
                    symbol=symbol,
                    interval=request.interval,
                    month=day.strftime("%Y-%m"),
                ),
                request.root,
            )
        return download_daily_archive_to_parquet(
            archive_client,
            DailyArchiveRequest(dataset, symbol, request.interval, day),
            request.root,
        )
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        return CollectorFailure(
            dataset=dataset,
            symbol=symbol,
            interval=request.interval,
            target_date=day.isoformat(),
            source_url="",
            attempt_count=1,
            error_code="archive_exception",
            error_message=str(exc),
            retryable=True,
        )


def _refresh_funding(
    request: RefreshRequest,
    client: FundingRateClient,
    symbol: str,
    day: date,
) -> CollectorSource:
    records = collect_funding_rates(
        client,
        FundingRateRequest(
            symbol=symbol,
            start_time_ms=_day_start_ms(day),
            end_time_ms=_day_end_ms(day),
        ),
    )
    output_path = _write_funding_parquet(request.root, symbol, day, records)
    return CollectorSource(
        dataset="fundingRate",
        symbol=symbol,
        interval=None,
        target_date=day.isoformat(),
        source_url=FUNDING_RATE_SOURCE_URL,
        output_path=str(output_path),
        checksum=None,
        row_count=len(records),
    )


def _write_funding_parquet(
    root: Path,
    symbol: str,
    day: date,
    records: list[FundingRateRecord],
) -> Path:
    storage_key = symbol_storage_key(symbol)
    output = (
        root
        / "binance"
        / "futures"
        / "fundingRate"
        / storage_key
        / f"{storage_key}_fundingRate_{day.isoformat()}.parquet"
    )
    frame = pl.DataFrame(
        {
            "funding_time": [record.funding_time for record in records],
            "funding_rate": [record.funding_rate for record in records],
            "mark_price": [record.mark_price for record in records],
        }
    ).with_columns(
        pl.from_epoch(pl.col("funding_time").cast(pl.Int64), time_unit="ms")
        .dt.replace_time_zone("UTC")
        .cast(pl.Datetime(time_unit="ms", time_zone="UTC"))
        .alias("funding_time"),
        pl.col("funding_rate").cast(pl.Float64),
        pl.col("mark_price").cast(pl.Float64),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(suffix=".parquet", dir=output.parent, delete=False) as handle:
        temp_path = Path(handle.name)
    frame.write_parquet(temp_path)
    _ = temp_path.replace(output)
    return output


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


def _day_start_ms(day: date) -> int:
    return int(datetime.combine(day, time.min, tzinfo=UTC).timestamp() * 1000)


def _day_end_ms(day: date) -> int:
    return int(datetime.combine(day, time.max, tzinfo=UTC).timestamp() * 1000)
