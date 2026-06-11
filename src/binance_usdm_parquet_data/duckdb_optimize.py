from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import duckdb

from binance_usdm_parquet_data.storage_keys import symbol_storage_key


def optimize_klines(
    *,
    raw_files: tuple[Path, ...],
    output_root: Path,
    symbol: str,
    interval: str,
) -> Path:
    if not raw_files:
        msg = "raw_files must not be empty"
        raise ValueError(msg)
    output = (
        output_root
        / "klines"
        / f"symbol={symbol_storage_key(symbol)}"
        / f"interval={interval}"
        / "candles.parquet"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(suffix=".parquet", dir=output.parent, delete=False) as handle:
        temp_path = Path(handle.name)
    interval_ms = _interval_ms(interval)
    file_paths = [path.as_posix() for path in raw_files]
    with duckdb.connect(":memory:") as connection:
        query = _optimized_query()
        params = [file_paths, interval_ms, interval_ms]
        relation = connection.sql(query, params=params)
        relation.write_parquet(temp_path.as_posix(), compression="zstd")
    _ = temp_path.replace(output)
    return output


def _optimized_query() -> str:
    return (
        "WITH normalized AS ("
        "SELECT "
        "coalesce(epoch_ms(try_cast(open_time AS TIMESTAMP)), try_cast(open_time AS BIGINT)) "
        "AS open_time_ms, "
        "try_cast(open AS DOUBLE) AS open, "
        "try_cast(high AS DOUBLE) AS high, "
        "try_cast(low AS DOUBLE) AS low, "
        "try_cast(close AS DOUBLE) AS close, "
        "try_cast(volume AS DOUBLE) AS volume, "
        "try_cast(COLUMNS('^(trade_count|number_of_trades)$') AS BIGINT) AS trade_count "
        "FROM read_parquet(?)"
        "), bucketed AS ("
        "SELECT *, floor(open_time_ms / ?) * ? AS bucket_ms "
        "FROM normalized "
        "WHERE open_time_ms IS NOT NULL "
        "AND open IS NOT NULL "
        "AND high IS NOT NULL "
        "AND low IS NOT NULL "
        "AND close IS NOT NULL "
        "AND volume IS NOT NULL "
        "AND trade_count IS NOT NULL"
        ") "
        "SELECT "
        "to_timestamp(cast(bucket_ms AS DOUBLE) / 1000.0) AS open_time, "
        "arg_min(open, open_time_ms) AS open, "
        "max(high) AS high, "
        "min(low) AS low, "
        "arg_max(close, open_time_ms) AS close, "
        "sum(volume) AS volume, "
        "cast(sum(trade_count) AS BIGINT) AS trade_count "
        "FROM bucketed "
        "GROUP BY bucket_ms "
        "ORDER BY bucket_ms"
    )


def _interval_ms(interval: str) -> int:
    unit = interval[-1]
    amount = int(interval[:-1])
    match unit:
        case "m":
            return amount * 60_000
        case "h":
            return amount * 3_600_000
        case "d":
            return amount * 86_400_000
        case _:
            msg = f"unsupported interval: {interval}"
            raise ValueError(msg)
