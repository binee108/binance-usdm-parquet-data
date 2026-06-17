from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from binance_usdm_parquet_data.funding_rate import JsonValue
from binance_usdm_parquet_data.premium_index import backfill_premium_index_klines


class EmptyPremiumClient:
    def get_json(self, url: str, params: dict[str, str]) -> JsonValue:
        del url, params
        return []


class RowPremiumClient:
    def get_json(self, url: str, params: dict[str, str]) -> JsonValue:
        del url, params
        return [
            [
                1_781_481_600_000,
                "-0.00056945",
                "-0.00041607",
                "-0.00073523",
                "-0.00057101",
                "0",
                1_781_481_659_999,
                "0",
                12,
                "0",
                "0",
                "0",
            ]
        ]


def test_backfill_premium_index_klines_writes_empty_completion_file(
    tmp_path: Path,
) -> None:
    result = backfill_premium_index_klines(
        root=tmp_path,
        symbol="1000BTTCUSDT",
        day=date(2026, 6, 15),
        interval="1m",
        client=EmptyPremiumClient(),
    )

    assert result.row_count == 0
    assert result.output_path.exists()
    assert result.output_path.name == "1000BTTCUSDT_premiumIndexKlines_1m_2026-06-15.parquet"
    frame = pl.read_parquet(result.output_path)
    assert frame.height == 0
    assert frame.schema == {
        "open_time": pl.Datetime(time_unit="ms", time_zone="UTC"),
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Float64,
        "trade_count": pl.Int64,
    }


def test_backfill_premium_index_klines_writes_rest_rows(tmp_path: Path) -> None:
    result = backfill_premium_index_klines(
        root=tmp_path,
        symbol="BTCUSDT",
        day=date(2026, 6, 15),
        interval="1m",
        client=RowPremiumClient(),
    )

    assert result.row_count == 1
    assert pl.read_parquet(result.output_path).to_dict(as_series=False) == {
        "open_time": [datetime(2026, 6, 15, 0, 0, tzinfo=UTC)],
        "open": [-0.00056945],
        "high": [-0.00041607],
        "low": [-0.00073523],
        "close": [-0.00057101],
        "volume": [0.0],
        "trade_count": [12],
    }
