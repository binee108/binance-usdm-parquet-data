from __future__ import annotations

from datetime import UTC, date, datetime, time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Final

import httpx2
import polars as pl

from binance_usdm_parquet_data.funding_rate import (
    FundingRateClient,
    FundingRateRecord,
    FundingRateRequest,
    collect_funding_rates,
)
from binance_usdm_parquet_data.manifest import CollectorFailure, CollectorSource
from binance_usdm_parquet_data.storage_keys import symbol_storage_key

FUNDING_RATE_SOURCE_URL: Final = "https://fapi.binance.com/fapi/v1/fundingRate"
_FUNDING_REST_ERROR_TYPES = (OSError, ValueError, TypeError, RuntimeError, httpx2.HTTPError)


def refresh_funding_rest_day(
    *,
    root: Path,
    symbol: str,
    client: FundingRateClient,
    day: date,
) -> CollectorSource | CollectorFailure | None:
    try:
        records = collect_funding_rates(
            client,
            FundingRateRequest(
                symbol=symbol,
                start_time_ms=_day_start_ms(day),
                end_time_ms=_day_end_ms(day),
            ),
        )
        output_path = _write_funding_parquet(root, symbol, day, records)
    except _FUNDING_REST_ERROR_TYPES as exc:
        if _is_missing_funding_rest_error(exc):
            return None
        return CollectorFailure(
            dataset="fundingRate",
            symbol=symbol,
            interval=None,
            target_date=day.isoformat(),
            source_url=FUNDING_RATE_SOURCE_URL,
            attempt_count=1,
            error_code="funding_rest_exception",
            error_message=str(exc),
            retryable=True,
        )
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


def _is_missing_funding_rest_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        ("400" in message and "bad request" in message)
        or ("404" in message and "not found" in message)
        or "invalid symbol" in message
        or "-1121" in message
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


def _day_start_ms(day: date) -> int:
    return int(datetime.combine(day, time.min, tzinfo=UTC).timestamp() * 1000)


def _day_end_ms(day: date) -> int:
    return int(datetime.combine(day, time.max, tzinfo=UTC).timestamp() * 1000)
