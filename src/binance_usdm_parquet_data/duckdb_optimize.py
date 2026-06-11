from __future__ import annotations

from pathlib import Path

import polars as pl

from binance_usdm_parquet_data.parquet_storage import normalize_kline_frame


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
    frames = [normalize_kline_frame(pl.read_parquet(path)) for path in raw_files]
    merged = pl.concat(frames).sort("open_time")
    optimized = _resample(merged, interval)
    output = (
        output_root / "klines" / f"symbol={symbol}" / f"interval={interval}" / "candles.parquet"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    optimized.write_parquet(output)
    return output


def _resample(frame: pl.DataFrame, interval: str) -> pl.DataFrame:
    interval_ms = _interval_ms(interval)
    with_bucket = frame.with_columns(
        ((pl.col("open_time").dt.epoch("ms") // interval_ms) * interval_ms).alias("bucket_ms")
    )
    return (
        with_bucket.group_by("bucket_ms", maintain_order=True)
        .agg(
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum(),
            pl.col("trade_count").sum(),
        )
        .with_columns(
            pl.from_epoch(pl.col("bucket_ms"), time_unit="ms")
            .dt.replace_time_zone("UTC")
            .cast(pl.Datetime(time_unit="ms", time_zone="UTC"))
            .alias("open_time")
        )
        .select("open_time", "open", "high", "low", "close", "volume", "trade_count")
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
