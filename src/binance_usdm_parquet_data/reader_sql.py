from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

if TYPE_CHECKING:
    import duckdb

from binance_usdm_parquet_data.records import QueryWindow

MILLISECONDS_EPOCH_THRESHOLD: Final = 10_000_000_000
type KlineRow = tuple[int, float, float, float, float, float]
type IntrabarRow = tuple[int, int, float, float, float, float]
type GapRow = tuple[int, int, int, int, int]
type FundingRow = tuple[datetime | float | None, float | None]


def resample_sql(
    files: list[Path],
    bucket_seconds: int,
    window: QueryWindow,
) -> tuple[str, list[int]]:
    clauses, params = time_clauses(window)
    sql = (
        "WITH source AS ("
        "SELECT CAST(epoch(open_time) AS BIGINT) AS open_epoch, "
        "try_cast(open AS DOUBLE) AS open, try_cast(high AS DOUBLE) AS high, "
        "try_cast(low AS DOUBLE) AS low, try_cast(close AS DOUBLE) AS close, "
        "try_cast(volume AS DOUBLE) AS volume "
        f"FROM {read_parquet_expr(files)} WHERE {' AND '.join(clauses)}"
        "), clean AS ("
        "SELECT * FROM source WHERE open IS NOT NULL AND high IS NOT NULL AND low IS NOT NULL "
        "AND close IS NOT NULL AND volume IS NOT NULL"
        "), bucketed AS ("
        "SELECT CAST(FLOOR(open_epoch / ?) * ? AS BIGINT) AS bucket_epoch, * FROM clean"
        ") SELECT bucket_epoch, arg_min(open, open_epoch), max(high), min(low), "
        "arg_max(close, open_epoch), sum(volume) FROM bucketed GROUP BY bucket_epoch "
        "ORDER BY bucket_epoch"
    )
    return sql, [*params, bucket_seconds, bucket_seconds]


def one_minute_sql(files: list[Path], window: QueryWindow) -> tuple[str, list[int]]:
    clauses, params = time_clauses(window)
    sql = (
        "SELECT CAST(epoch(open_time) AS BIGINT), try_cast(open AS DOUBLE), "
        "try_cast(high AS DOUBLE), try_cast(low AS DOUBLE), try_cast(close AS DOUBLE), "
        f"try_cast(volume AS DOUBLE) FROM {read_parquet_expr(files)} "
        f"WHERE {' AND '.join(clauses)} "
        "AND try_cast(open AS DOUBLE) IS NOT NULL AND try_cast(high AS DOUBLE) IS NOT NULL "
        "AND try_cast(low AS DOUBLE) IS NOT NULL AND try_cast(close AS DOUBLE) IS NOT NULL "
        "AND try_cast(volume AS DOUBLE) IS NOT NULL ORDER BY open_time ASC"
    )
    return sql, params


def intrabar_sql(
    files: list[Path],
    bucket_seconds: int,
    window: QueryWindow,
) -> tuple[str, list[int]]:
    clauses, params = time_clauses(window)
    sql = (
        "SELECT CAST(FLOOR(CAST(epoch(open_time) AS BIGINT) / ?) * ? AS BIGINT), "
        "CAST(epoch(open_time) AS BIGINT), try_cast(open AS DOUBLE), try_cast(high AS DOUBLE), "
        "try_cast(low AS DOUBLE), try_cast(close AS DOUBLE) "
        f"FROM {read_parquet_expr(files)} WHERE {' AND '.join(clauses)} "
        "AND try_cast(open AS DOUBLE) IS NOT NULL AND try_cast(high AS DOUBLE) IS NOT NULL "
        "AND try_cast(low AS DOUBLE) IS NOT NULL AND try_cast(close AS DOUBLE) IS NOT NULL "
        "ORDER BY open_time ASC"
    )
    return sql, [bucket_seconds, bucket_seconds, *params]


def missing_gap_sql(
    files: list[Path],
    window: QueryWindow,
    expected_seconds: int,
) -> tuple[str, list[int]]:
    clauses, params = time_clauses(window)
    sql = (
        "WITH source AS (SELECT DISTINCT CAST(epoch(open_time) AS BIGINT) AS open_epoch "
        f"FROM {read_parquet_expr(files)} WHERE {' AND '.join(clauses)}), "
        "ordered AS (SELECT open_epoch, lag(open_epoch) OVER (ORDER BY open_epoch) "
        "AS previous_epoch FROM source) SELECT previous_epoch + ?, open_epoch - ?, "
        "CAST(((open_epoch - previous_epoch) / ?) - 1 AS BIGINT), previous_epoch, open_epoch "
        "FROM ordered WHERE previous_epoch IS NOT NULL AND open_epoch - previous_epoch > ? "
        "ORDER BY previous_epoch"
    )
    return sql, [
        *params,
        expected_seconds,
        expected_seconds,
        expected_seconds,
        expected_seconds,
    ]


def parquet_columns(connection: duckdb.DuckDBPyConnection, file_paths: list[str]) -> set[str]:
    rows = cast(
        "list[tuple[str]]",
        connection.execute(
            "DESCRIBE SELECT * FROM read_parquet(?, union_by_name = true)",
            [file_paths],
        ).fetchall(),
    )
    return {str(row[0]) for row in rows}


def first_existing(names: set[str], candidates: tuple[str, ...]) -> str | None:
    return next((name for name in candidates if name in names), None)


def funding_sql(file_paths: list[str], time_column: str, rate_column: str) -> str:
    return (
        "SELECT coalesce(epoch_ms(try_cast("
        f"{time_column} AS TIMESTAMP)), try_cast({time_column} AS BIGINT)), "
        f"try_cast({rate_column} AS DOUBLE) FROM {read_parquet_expr_for_strings(file_paths)}"
    )


def timestamp_seconds(value: datetime | float) -> int:
    if isinstance(value, datetime):
        parsed = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return int(parsed.timestamp())
    numeric = int(value)
    return numeric // 1000 if numeric > MILLISECONDS_EPOCH_THRESHOLD else numeric


def time_clauses(window: QueryWindow) -> tuple[list[str], list[int]]:
    clauses = ["open_time IS NOT NULL"]
    params: list[int] = []
    if window.start_ts is not None:
        clauses.append("CAST(epoch(open_time) AS BIGINT) >= ?")
        params.append(window.start_ts)
    if window.end_ts is not None:
        clauses.append("CAST(epoch(open_time) AS BIGINT) <= ?")
        params.append(window.end_ts)
    return clauses, params


def read_parquet_expr(files: list[Path]) -> str:
    joined = ", ".join(_sql_string(path) for path in files)
    return f"read_parquet([{joined}], union_by_name = true)"


def read_parquet_expr_for_strings(file_paths: list[str]) -> str:
    joined = ", ".join("'" + path.replace("'", "''") + "'" for path in file_paths)
    return f"read_parquet([{joined}], union_by_name = true)"


def _sql_string(path: Path) -> str:
    return "'" + path.as_posix().replace("'", "''") + "'"
