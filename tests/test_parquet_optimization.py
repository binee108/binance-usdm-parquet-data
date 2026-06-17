from __future__ import annotations

import json
import time
from pathlib import Path

import polars as pl
import pytest

from binance_usdm_parquet_data.duckdb_optimize import optimize_klines
from binance_usdm_parquet_data.locks import default_lock_path, lock_metadata
from binance_usdm_parquet_data.parquet_storage import normalize_kline_frame


def test_normalize_kline_frame_casts_strings_to_deterministic_numeric_types() -> None:
    frame = pl.DataFrame(
        {
            "open_time": [1_700_000_000_000],
            "open": ["100.1"],
            "high": ["101.2"],
            "low": ["99.9"],
            "close": ["100.8"],
            "volume": ["12.34"],
            "trade_count": ["42"],
        }
    )

    normalized = normalize_kline_frame(frame)

    assert normalized.schema["open_time"] == pl.Datetime(time_unit="ms", time_zone="UTC")
    assert normalized.schema["open"] == pl.Float64
    assert normalized.schema["trade_count"] == pl.Int64
    assert normalized.item(0, "volume") == 12.34


def test_optimize_klines_writes_symbol_interval_layout_and_resamples(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.parquet"
    pl.DataFrame(
        {
            "open_time": [1_700_000_000_000, 1_700_000_060_000],
            "open": ["100.0", "101.0"],
            "high": ["102.0", "103.0"],
            "low": ["99.0", "100.5"],
            "close": ["101.0", "102.5"],
            "volume": ["10.0", "20.0"],
            "trade_count": ["2", "3"],
        }
    ).write_parquet(raw_path)

    output = optimize_klines(
        raw_files=(raw_path,),
        output_root=tmp_path / "optimized",
        symbol="BTCUSDT",
        interval="1h",
    )

    assert output == (
        tmp_path / "optimized" / "klines" / "symbol=BTCUSDT" / "interval=1h" / "candles.parquet"
    )
    result = pl.read_parquet(output)
    assert result.to_dicts() == [
        {
            "open_time": result.item(0, "open_time"),
            "open": 100.0,
            "high": 103.0,
            "low": 99.0,
            "close": 102.5,
            "volume": 30.0,
            "trade_count": 5,
        }
    ]


def test_optimize_klines_canonicalizes_symbol_layout(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.parquet"
    pl.DataFrame(
        {
            "open_time": [1_700_000_000_000],
            "open": ["100.0"],
            "high": ["102.0"],
            "low": ["99.0"],
            "close": ["101.0"],
            "volume": ["10.0"],
        }
    ).write_parquet(raw_path)

    output = optimize_klines(
        raw_files=(raw_path,),
        output_root=tmp_path / "optimized",
        symbol="btcusdt",
        interval="1m",
    )

    assert output == (
        tmp_path / "optimized" / "klines" / "symbol=BTCUSDT" / "interval=1m" / "candles.parquet"
    )


def test_optimize_klines_handles_mixed_trade_count_schemas(tmp_path: Path) -> None:
    first = tmp_path / "number_of_trades.parquet"
    second = tmp_path / "trade_count.parquet"
    pl.DataFrame(
        {
            "open_time": [1_700_000_000_000],
            "open": ["100.0"],
            "high": ["102.0"],
            "low": ["99.0"],
            "close": ["101.0"],
            "volume": ["10.0"],
            "number_of_trades": [2],
        }
    ).write_parquet(first)
    pl.DataFrame(
        {
            "open_time": [1_700_000_060_000],
            "open": ["101.0"],
            "high": ["103.0"],
            "low": ["100.5"],
            "close": ["102.5"],
            "volume": ["20.0"],
            "trade_count": [3],
        }
    ).write_parquet(second)

    output = optimize_klines(
        raw_files=(first, second),
        output_root=tmp_path / "optimized",
        symbol="BTCUSDT",
        interval="1h",
    )

    result = pl.read_parquet(output)
    assert result.item(0, "trade_count") == 5


def test_optimize_klines_claims_stale_output_lock(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.parquet"
    pl.DataFrame(
        {
            "open_time": [1_700_000_000_000],
            "open": ["100.0"],
            "high": ["102.0"],
            "low": ["99.0"],
            "close": ["101.0"],
            "volume": ["10.0"],
        }
    ).write_parquet(raw_path)
    output = (
        tmp_path
        / "optimized"
        / "klines"
        / "symbol=BTCUSDT"
        / "interval=1m"
        / "candles.parquet"
    )
    lock_path = default_lock_path(output)
    lock_path.parent.mkdir(parents=True)
    metadata = lock_metadata(output, "optimize_klines", "stale-owner")
    metadata["created_at_epoch"] = time.time() - 1_000
    _ = lock_path.write_text(json.dumps(metadata), encoding="utf-8")

    written = optimize_klines(
        raw_files=(raw_path,),
        output_root=tmp_path / "optimized",
        symbol="BTCUSDT",
        interval="1m",
    )

    assert written == output
    assert written.exists()
    assert not lock_path.exists()


def test_optimize_klines_rejects_traversal_interval_before_filesystem_write(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw.parquet"
    raw_path.touch()
    output_root = tmp_path / "optimized"

    with pytest.raises(ValueError, match="Invalid market data interval"):
        _ = optimize_klines(
            raw_files=(raw_path,),
            output_root=output_root,
            symbol="BTCUSDT",
            interval="x/../../../../outside/1m",
        )

    assert not output_root.exists()
    assert not (tmp_path / "outside").exists()
