from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

import polars as pl

from binance_usdm_parquet_data.archive_download import (
    ArchiveDownloadFailure,
    ArchiveHttpClient,
    DailyArchiveRequest,
    download_daily_archive_to_parquet,
)
from binance_usdm_parquet_data.duckdb_optimize import optimize_klines
from binance_usdm_parquet_data.funding_rate import (
    FundingRateClient,
    FundingRateRecord,
    FundingRateRequest,
    collect_funding_rates,
)
from binance_usdm_parquet_data.manifest import (
    CollectorFailure,
    CollectorRun,
    DatasetFreshness,
    ManifestStore,
)


@dataclass(frozen=True, slots=True)
class RefreshRequest:
    root: Path
    symbols: tuple[str, ...]
    start_day: date
    end_day: date
    datasets: tuple[str, ...] = ("klines", "premiumIndexKlines", "fundingRate")
    interval: str = "1m"
    optimize: bool = True


@dataclass(frozen=True, slots=True)
class RefreshResult:
    run_id: str
    status: str
    success_count: int
    failure_count: int


def refresh_market_data(
    request: RefreshRequest,
    *,
    archive_client: ArchiveHttpClient,
    funding_client: FundingRateClient,
) -> RefreshResult:
    run_id = str(uuid4())
    started_at = datetime.now(UTC)
    downloaded_klines: dict[str, list[Path]] = {symbol: [] for symbol in request.symbols}
    failures: list[CollectorFailure] = []
    success_count = 0
    for symbol in request.symbols:
        for day in _days(request.start_day, request.end_day):
            for dataset in request.datasets:
                if dataset == "fundingRate":
                    success_count += _refresh_funding(request, funding_client, symbol, day)
                else:
                    result = download_daily_archive_to_parquet(
                        archive_client,
                        DailyArchiveRequest(
                            dataset=dataset,
                            symbol=symbol,
                            interval=request.interval,
                            day=day,
                        ),
                        request.root,
                    )
                    if isinstance(result, ArchiveDownloadFailure):
                        failures.append(
                            _archive_failure(result, dataset, symbol, request.interval, day)
                        )
                    else:
                        success_count += 1
                        if dataset == "klines":
                            downloaded_klines[symbol].append(result.output_path)
    if request.optimize:
        for symbol, paths in downloaded_klines.items():
            if paths:
                _ = optimize_klines(
                    raw_files=tuple(paths),
                    output_root=request.root / "parbp_optimized" / "binance" / "futures",
                    symbol=symbol,
                    interval=request.interval,
                )
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
    )
    return RefreshResult(
        run_id=run_id,
        status=status,
        success_count=success_count,
        failure_count=len(failures),
    )


def _refresh_funding(
    request: RefreshRequest,
    client: FundingRateClient,
    symbol: str,
    day: date,
) -> int:
    records = collect_funding_rates(
        client,
        FundingRateRequest(
            symbol=symbol,
            start_time_ms=_day_start_ms(day),
            end_time_ms=_day_end_ms(day),
        ),
    )
    _ = _write_funding_parquet(request.root, symbol, day, records)
    return 1


def _write_funding_parquet(
    root: Path,
    symbol: str,
    day: date,
    records: list[FundingRateRecord],
) -> Path:
    output = (
        root
        / "binance"
        / "futures"
        / "fundingRate"
        / symbol
        / f"{symbol}_fundingRate_{day.isoformat()}.parquet"
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


def _day_start_ms(day: date) -> int:
    return int(datetime.combine(day, time.min, tzinfo=UTC).timestamp() * 1000)


def _day_end_ms(day: date) -> int:
    return int(datetime.combine(day, time.max, tzinfo=UTC).timestamp() * 1000)
