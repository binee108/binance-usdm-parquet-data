from __future__ import annotations

from pathlib import Path

import polars as pl

from binance_usdm_parquet_data.duckdb_optimize import optimize_klines
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
