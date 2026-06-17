from __future__ import annotations

from pathlib import Path
from typing import Final, cast

import duckdb

from binance_usdm_parquet_data.paths import (
    funding_rate_files,
    kline_files_for_read,
    premium_index_kline_files,
)
from binance_usdm_parquet_data.quality import record_missing_klines
from binance_usdm_parquet_data.reader_sql import (
    FundingRow,
    GapRow,
    IntrabarRow,
    KlineRow,
    first_existing,
    funding_sql,
    intrabar_sql,
    missing_gap_sql,
    one_minute_sql,
    parquet_columns,
    resample_sql,
    timestamp_seconds,
)
from binance_usdm_parquet_data.records import (
    FundingRateEventRecord,
    IntrabarKlineRecord,
    KlineRecord,
    MissingKlineRange,
    QueryWindow,
)

ONE_MINUTE_SECONDS: Final = 60


def load_klines(
    *,
    root: Path | None,
    symbol: str,
    interval: str,
    start_ts: int | None = None,
    end_ts: int | None = None,
    record_gaps: bool = False,
) -> tuple[KlineRecord, ...]:
    files = kline_files_for_read(root, symbol, interval)
    if not files:
        return ()
    window = QueryWindow(start_ts, end_ts)
    if record_gaps:
        record_kline_gaps(
            root=root,
            symbol=symbol,
            interval=interval,
            window=window,
            expected_seconds=ONE_MINUTE_SECONDS,
        )
    sql, params = one_minute_sql(files, window)
    with duckdb.connect(":memory:") as connection:
        rows = cast("list[KlineRow]", connection.execute(sql, params).fetchall())
    return _kline_records(rows)


def load_resampled_klines(
    *,
    root: Path | None,
    symbol: str,
    timeframe_seconds: int,
    start_ts: int | None = None,
    end_ts: int | None = None,
    record_gaps: bool = False,
) -> tuple[KlineRecord, ...]:
    files = kline_files_for_read(root, symbol, "1m")
    if not files:
        return ()
    window = QueryWindow(start_ts, end_ts)
    if record_gaps:
        record_kline_gaps(
            root=root,
            symbol=symbol,
            interval="1m",
            window=window,
            expected_seconds=ONE_MINUTE_SECONDS,
        )
    sql, params = (
        one_minute_sql(files, window)
        if timeframe_seconds == ONE_MINUTE_SECONDS
        else resample_sql(files, timeframe_seconds, window)
    )
    with duckdb.connect(":memory:") as connection:
        rows = cast("list[KlineRow]", connection.execute(sql, params).fetchall())
    return _kline_records(rows)


def load_intrabar_buckets(
    *,
    root: Path | None,
    symbol: str,
    primary_timeframe_seconds: int,
    start_ts: int | None = None,
    end_ts: int | None = None,
    record_gaps: bool = False,
) -> dict[int, tuple[IntrabarKlineRecord, ...]]:
    files = kline_files_for_read(root, symbol, "1m")
    if not files:
        return {}
    window = QueryWindow(start_ts, end_ts)
    if record_gaps:
        record_kline_gaps(
            root=root,
            symbol=symbol,
            interval="1m",
            window=window,
            expected_seconds=ONE_MINUTE_SECONDS,
        )
    sql, params = intrabar_sql(files, primary_timeframe_seconds, window)
    grouped: dict[int, list[IntrabarKlineRecord]] = {}
    with duckdb.connect(":memory:") as connection:
        rows = cast("list[IntrabarRow]", connection.execute(sql, params).fetchall())
        for row in rows:
            bucket = int(row[0])
            grouped.setdefault(bucket, []).append(
                IntrabarKlineRecord(
                    timestamp=int(row[1]),
                    open=float(row[2]),
                    high=float(row[3]),
                    low=float(row[4]),
                    close=float(row[5]),
                )
            )
    return {bucket: tuple(rows) for bucket, rows in grouped.items()}


def load_funding_rates(
    *,
    root: Path | None,
    symbol: str,
    start_ts: int | None = None,
    end_ts: int | None = None,
) -> tuple[FundingRateEventRecord, ...]:
    files = funding_rate_files(root, symbol)
    if not files:
        return ()
    file_paths = [path.as_posix() for path in files]
    with duckdb.connect(":memory:") as connection:
        columns = parquet_columns(connection, file_paths)
        time_column = first_existing(columns, ("funding_time", "fundingTime", "timestamp", "time"))
        rate_column = first_existing(columns, ("funding_rate", "fundingRate"))
        if time_column is None or rate_column is None:
            return ()
        rows = cast(
            "list[FundingRow]",
            connection.execute(funding_sql(file_paths, time_column, rate_column)).fetchall(),
        )
    events = tuple(
        FundingRateEventRecord(timestamp=timestamp_seconds(row[0]), funding_rate=float(row[1]))
        for row in rows
        if row[0] is not None and row[1] is not None
    )
    return tuple(
        event
        for event in sorted(events, key=lambda item: item.timestamp)
        if (start_ts is None or event.timestamp >= start_ts)
        and (end_ts is None or event.timestamp <= end_ts)
    )


def load_premium_index_klines(
    root: Path | None,
    symbol: str,
    interval: str,
    start_ts: int | None = None,
    end_ts: int | None = None,
) -> tuple[KlineRecord, ...]:
    files = premium_index_kline_files(root, symbol, interval)
    if not files:
        return ()
    sql, params = one_minute_sql(files, QueryWindow(start_ts, end_ts))
    with duckdb.connect(":memory:") as connection:
        rows = cast("list[KlineRow]", connection.execute(sql, params).fetchall())
    return _kline_records(rows)


def record_kline_gaps(
    *,
    root: Path | None,
    symbol: str,
    interval: str,
    window: QueryWindow,
    expected_seconds: int,
) -> None:
    files = kline_files_for_read(root, symbol, interval)
    if not files:
        return
    sql, params = missing_gap_sql(files, window, expected_seconds)
    with duckdb.connect(":memory:") as connection:
        rows = cast("list[GapRow]", connection.execute(sql, params).fetchall())
    ranges = [
        MissingKlineRange(
            symbol=symbol,
            interval=interval,
            missing_start_ts=int(row[0]),
            missing_end_ts=int(row[1]),
            missing_count=int(row[2]),
            observed_before_ts=int(row[3]),
            observed_after_ts=int(row[4]),
        )
        for row in rows
    ]
    record_missing_klines(root, ranges)


def _kline_records(rows: list[KlineRow]) -> tuple[KlineRecord, ...]:
    return tuple(
        KlineRecord(
            timestamp=int(row[0]),
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
        )
        for row in rows
    )
