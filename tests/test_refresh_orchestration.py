from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import date
from pathlib import Path
from typing import cast

import polars as pl

from binance_usdm_parquet_data.archive_download import ArchiveHttpClient
from binance_usdm_parquet_data.funding_rate import FundingRateClient, FundingRateRecord
from binance_usdm_parquet_data.refresh import RefreshRequest, refresh_market_data


class FakeArchiveClient:
    def __init__(self, archive: bytes) -> None:
        self.archive: bytes = archive
        self.checksum: str = hashlib.sha256(archive).hexdigest()

    def get_bytes(self, url: str) -> bytes:
        assert "daily/klines/BTCUSDT/1m/" in url
        return self.archive

    def get_text(self, url: str) -> str:
        assert url.endswith(".CHECKSUM")
        return f"{self.checksum}  BTCUSDT-1m-2026-06-09.zip"


class FakeFundingClient:
    def get_funding_rates(
        self,
        *,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int,
    ) -> list[FundingRateRecord]:
        assert symbol == "BTCUSDT"
        assert start_time_ms == 1_780_963_200_000
        assert end_time_ms == 1_781_049_599_999
        assert limit == 1000
        return [FundingRateRecord(1_780_992_000_000, 0.0001, 105_000.0)]


def test_refresh_writes_raw_optimized_funding_and_manifest(tmp_path: Path) -> None:
    archive: ArchiveHttpClient = FakeArchiveClient(
        _zip_csv(
            "BTCUSDT-1m-2026-06-09.csv",
            "1780963200000,100,101,99,100.5,10,1780963259999,1005,3,5,500,0\n",
        )
    )
    funding: FundingRateClient = FakeFundingClient()

    result = refresh_market_data(
        RefreshRequest(
            root=tmp_path,
            symbols=("BTCUSDT",),
            start_day=date(2026, 6, 9),
            end_day=date(2026, 6, 9),
            datasets=("klines", "fundingRate"),
            interval="1m",
            optimize=True,
        ),
        archive_client=archive,
        funding_client=funding,
    )

    assert result.status == "succeeded"
    assert (tmp_path / "binance" / "futures" / "klines" / "BTCUSDT").exists()
    assert (
        tmp_path
        / "parbp_optimized"
        / "binance"
        / "futures"
        / "klines"
        / "symbol=BTCUSDT"
        / "interval=1m"
        / "candles.parquet"
    ).exists()
    funding_root = tmp_path / "binance" / "futures" / "fundingRate" / "BTCUSDT"
    funding_files = list(funding_root.glob("*.parquet"))
    assert len(funding_files) == 1
    funding_frame = pl.read_parquet(funding_files[0])
    assert funding_frame.item(0, "funding_rate") == 0.0001
    status = cast(
        dict[str, object],
        json.loads((tmp_path / "manifests" / "binance" / "usdm" / "status.json").read_text()),
    )
    assert status["failed_item_count"] == 0


def _zip_csv(name: str, body: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, body)
    return buffer.getvalue()
