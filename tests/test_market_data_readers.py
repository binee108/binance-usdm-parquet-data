from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from binance_usdm_parquet_data.paths import optimized_klines_file
from binance_usdm_parquet_data.quality import (
    MissingKlinesStaleLockError,
    append_missing_klines,
    missing_klines_path,
    read_missing_klines,
    record_missing_klines,
)
from binance_usdm_parquet_data.readers import (
    load_funding_rates,
    load_intrabar_buckets,
    load_premium_index_klines,
    load_resampled_klines,
)
from binance_usdm_parquet_data.records import MissingKlineRange
from binance_usdm_parquet_data.status_scan import scan_local_status


def _write_klines(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "open_time": [
                datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
                datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
                datetime(2024, 1, 1, 0, 15, tzinfo=UTC),
                datetime(2024, 1, 1, 1, 0, tzinfo=UTC),
            ],
            "open": ["100", "105", "111", "114"],
            "high": ["110", "112", "115", "120"],
            "low": ["99", "104", "109", "113"],
            "close": ["105", "111", "114", "118"],
            "volume": ["10", "20", "30", "40"],
            "trade_count": [1, 2, 3, 4],
        }
    ).write_parquet(path)


def _write_funding_rates(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "fundingTime": [1_704_067_200_000, 1_704_096_000_000],
            "fundingRate": ["0.0001", "-0.0002"],
        }
    ).write_parquet(path)


def _write_premium_index_klines(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "open_time": [
                datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
                datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
                datetime(2024, 1, 1, 0, 2, tzinfo=UTC),
            ],
            "open": ["1.00", "1.10", "1.20"],
            "high": ["1.20", "1.30", "1.40"],
            "low": ["0.90", "1.00", "1.10"],
            "close": ["1.10", "1.20", "1.30"],
            "volume": ["0", "0", "0"],
        }
    ).write_parquet(path)


def test_load_resampled_klines_reads_optimized_1m_source_and_records_gaps(tmp_path: Path) -> None:
    optimized = optimized_klines_file(tmp_path, "BTCUSDT", "1m")
    _write_klines(optimized)

    candles = load_resampled_klines(
        root=tmp_path,
        symbol="BTCUSDT",
        timeframe_seconds=900,
        start_ts=1_704_067_200,
        end_ts=1_704_153_599,
        record_gaps=True,
    )

    assert [(c.timestamp, c.open, c.high, c.low, c.close, c.volume) for c in candles] == [
        (1_704_067_200, 100.0, 112.0, 99.0, 111.0, 30.0),
        (1_704_068_100, 111.0, 115.0, 109.0, 114.0, 30.0),
        (1_704_070_800, 114.0, 120.0, 113.0, 118.0, 40.0),
    ]
    assert [item["missing_count"] for item in read_missing_klines(tmp_path)] == [13, 44]


def test_load_intrabar_buckets_groups_1m_rows_by_primary_timeframe(tmp_path: Path) -> None:
    optimized = optimized_klines_file(tmp_path, "BTCUSDT", "1m")
    _write_klines(optimized)

    buckets = load_intrabar_buckets(
        root=tmp_path,
        symbol="BTCUSDT",
        primary_timeframe_seconds=10_800,
        start_ts=1_704_067_260,
        end_ts=1_704_068_100,
    )

    assert [bar.timestamp for rows in buckets.values() for bar in rows] == [
        1_704_067_260,
        1_704_068_100,
    ]


def test_load_funding_rates_reads_legacy_rest_columns_and_filters_seconds(tmp_path: Path) -> None:
    _write_funding_rates(
        tmp_path
        / "binance"
        / "futures"
        / "fundingRate"
        / "BTCUSDT"
        / "BTCUSDT_fundingRate_2024-01-01.parquet"
    )

    events = load_funding_rates(
        root=tmp_path,
        symbol="BTCUSDT",
        start_ts=1_704_067_200,
        end_ts=1_704_067_200,
    )

    assert [(event.timestamp, event.funding_rate) for event in events] == [
        (1_704_067_200, 0.0001)
    ]


def test_load_premium_index_klines_reads_raw_files_and_filters_window(
    tmp_path: Path,
) -> None:
    _write_premium_index_klines(
        tmp_path
        / "binance"
        / "futures"
        / "premiumIndexKlines"
        / "BTCUSDT"
        / "BTCUSDT_premiumIndexKlines_1m_2024-01.parquet"
    )

    candles = load_premium_index_klines(
        tmp_path,
        "BTCUSDT",
        "1m",
        start_ts=1_704_067_260,
        end_ts=1_704_067_320,
    )

    assert [(candle.timestamp, candle.open, candle.close) for candle in candles] == [
        (1_704_067_260, 1.1, 1.2),
        (1_704_067_320, 1.2, 1.3),
    ]


def test_load_premium_index_klines_returns_empty_tuple_when_files_are_missing(
    tmp_path: Path,
) -> None:
    candles = load_premium_index_klines(tmp_path, "BTCUSDT", "1m")

    assert candles == ()


def test_scan_local_status_reports_existing_files_when_status_manifest_is_missing(
    tmp_path: Path,
) -> None:
    optimized = optimized_klines_file(tmp_path, "BTCUSDT", "1m")
    optimized.parent.mkdir(parents=True)
    _ = optimized.touch()
    _ = optimized.with_name("candles.manifest.json").write_text(
        json.dumps(
            {
                "source_files": 2,
                "last_source": str(
                    tmp_path
                    / "binance"
                    / "futures"
                    / "klines"
                    / "BTCUSDT"
                    / "BTCUSDT_klines_1m_2026-06.parquet"
                ),
            }
        ),
        encoding="utf-8",
    )

    summary = scan_local_status(tmp_path)

    assert summary.source_count == 2
    assert [item.dataset for item in summary.freshness] == ["klines"]
    assert summary.freshness[0].latest_complete_utc_day == "2026-06-30"


def test_append_missing_klines_public_name_records_missing_ranges(tmp_path: Path) -> None:
    append_missing_klines(
        tmp_path,
        [
            MissingKlineRange(
                symbol="BTCUSDT",
                interval="1m",
                missing_start_ts=1_700_000_000,
                missing_end_ts=1_700_000_060,
                missing_count=1,
                observed_before_ts=1_699_999_940,
                observed_after_ts=1_700_000_120,
            )
        ],
    )

    assert [item["missing_start_ts"] for item in read_missing_klines(tmp_path)] == [
        1_700_000_000
    ]


def test_missing_klines_stale_lock_rejects_different_target_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = missing_klines_path(tmp_path)
    lock_path = path.with_name(f".{path.name}.lock")
    lock_path.parent.mkdir(parents=True)
    _ = lock_path.write_text(
        json.dumps(
            {
                "pid": 999999,
                "hostname": "test",
                "created_at_epoch": time.time() - 1_000,
                "target_path": str((tmp_path / "other.jsonl").resolve()),
                "operation": "test",
                "owner_token": "other",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("binance_usdm_parquet_data.quality.LOCK_STALE_SECONDS", 0.01)

    with pytest.raises(MissingKlinesStaleLockError):
        record_missing_klines(
            tmp_path,
            [
                MissingKlineRange(
                    symbol="BTCUSDT",
                    interval="1m",
                    missing_start_ts=1_700_000_000,
                    missing_end_ts=1_700_000_060,
                    missing_count=1,
                    observed_before_ts=1_699_999_940,
                    observed_after_ts=1_700_000_120,
                )
            ],
        )

    assert lock_path.exists()
    assert not path.exists()
