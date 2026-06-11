from __future__ import annotations

import polars as pl

KLINE_COLUMNS = ("open_time", "open", "high", "low", "close", "volume", "trade_count")


def normalize_kline_frame(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.select(
        pl.from_epoch(pl.col("open_time").cast(pl.Int64), time_unit="ms")
        .dt.replace_time_zone("UTC")
        .cast(pl.Datetime(time_unit="ms", time_zone="UTC"))
        .alias("open_time"),
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Float64),
        pl.col("trade_count").cast(pl.Int64),
    )
