from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Final, Protocol, override

import polars as pl

from binance_usdm_parquet_data.funding_rate import JsonValue
from binance_usdm_parquet_data.paths import require_interval, require_symbol

PREMIUM_INDEX_KLINES_URL: Final = "https://fapi.binance.com/fapi/v1/premiumIndexKlines"
MAX_PREMIUM_INDEX_KLINE_LIMIT: Final = 1500
MIN_PREMIUM_INDEX_ROW_LENGTH: Final = 9


class PremiumIndexKlineClient(Protocol):
    def get_json(self, url: str, params: dict[str, str]) -> JsonValue: ...


@dataclass(frozen=True, slots=True)
class PremiumIndexKlineBackfill:
    source_url: str
    output_path: Path
    row_count: int


@dataclass(frozen=True, slots=True)
class PremiumIndexKline:
    open_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    trade_count: int


@dataclass(frozen=True, slots=True)
class PremiumIndexKlinePayloadError(ValueError):
    field: str
    value: str

    @override
    def __str__(self) -> str:
        return f"invalid premium index kline {self.field}: {self.value}"


def backfill_premium_index_klines(
    *,
    root: Path,
    symbol: str,
    day: date,
    interval: str,
    client: PremiumIndexKlineClient,
) -> PremiumIndexKlineBackfill:
    checked_symbol = require_symbol(symbol)
    checked_interval = require_interval(interval)
    payload = client.get_json(
        PREMIUM_INDEX_KLINES_URL,
        {
            "symbol": checked_symbol,
            "interval": checked_interval,
            "startTime": str(_day_start_ms(day)),
            "endTime": str(_day_end_ms(day)),
            "limit": str(MAX_PREMIUM_INDEX_KLINE_LIMIT),
        },
    )
    records = _parse_payload(payload)
    output_path = _premium_index_kline_path(root, checked_symbol, checked_interval, day)
    _write_parquet(output_path, records)
    return PremiumIndexKlineBackfill(
        source_url=PREMIUM_INDEX_KLINES_URL,
        output_path=output_path,
        row_count=len(records),
    )


def _parse_payload(payload: JsonValue) -> tuple[PremiumIndexKline, ...]:
    if not isinstance(payload, list):
        field = "payload"
        raise PremiumIndexKlinePayloadError(field, repr(payload))
    return tuple(_parse_row(row) for row in payload)


def _parse_row(row: JsonValue) -> PremiumIndexKline:
    if not isinstance(row, list):
        field = "row"
        raise PremiumIndexKlinePayloadError(field, repr(row))
    if len(row) < MIN_PREMIUM_INDEX_ROW_LENGTH:
        field = "row"
        raise PremiumIndexKlinePayloadError(field, repr(row))
    return PremiumIndexKline(
        open_time_ms=_as_int(row[0], "open_time"),
        open=_as_float(row[1], "open"),
        high=_as_float(row[2], "high"),
        low=_as_float(row[3], "low"),
        close=_as_float(row[4], "close"),
        volume=_as_float(row[5], "volume"),
        trade_count=_as_int(row[8], "trade_count"),
    )


def _as_int(value: JsonValue, field: str) -> int:
    if isinstance(value, bool | list | dict) or value is None:
        raise PremiumIndexKlinePayloadError(field, repr(value))
    try:
        return int(value)
    except ValueError as exc:
        raise PremiumIndexKlinePayloadError(field, repr(value)) from exc


def _as_float(value: JsonValue, field: str) -> float:
    if isinstance(value, bool | list | dict) or value is None:
        raise PremiumIndexKlinePayloadError(field, repr(value))
    try:
        return float(value)
    except ValueError as exc:
        raise PremiumIndexKlinePayloadError(field, repr(value)) from exc


def _write_parquet(path: Path, records: tuple[PremiumIndexKline, ...]) -> None:
    frame = pl.DataFrame(
        {
            "open_time": [
                datetime.fromtimestamp(record.open_time_ms / 1000, tz=UTC)
                for record in records
            ],
            "open": [record.open for record in records],
            "high": [record.high for record in records],
            "low": [record.low for record in records],
            "close": [record.close for record in records],
            "volume": [record.volume for record in records],
            "trade_count": [record.trade_count for record in records],
        },
        schema={
            "open_time": pl.Datetime(time_unit="ms", time_zone="UTC"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
            "trade_count": pl.Int64,
        },
        strict=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(suffix=".parquet", dir=path.parent, delete=False) as handle:
        temp_path = Path(handle.name)
    frame.write_parquet(temp_path)
    _ = temp_path.replace(path)


def _premium_index_kline_path(root: Path, symbol: str, interval: str, day: date) -> Path:
    return (
        root
        / "binance"
        / "futures"
        / "premiumIndexKlines"
        / symbol
        / f"{symbol}_premiumIndexKlines_{interval}_{day.isoformat()}.parquet"
    )


def _day_start_ms(day: date) -> int:
    return int(datetime.combine(day, time.min, tzinfo=UTC).timestamp() * 1000)


def _day_end_ms(day: date) -> int:
    return int(datetime.combine(day, time.max, tzinfo=UTC).timestamp() * 1000)
