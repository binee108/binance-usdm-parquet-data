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


class FailingArchiveClient:
    def get_bytes(self, url: str) -> bytes:
        raise OSError(url)

    def get_text(self, url: str) -> str:
        raise OSError(url)


class RecordingArchiveClient:
    def __init__(self, archive: bytes) -> None:
        self.archive: bytes = archive
        self.checksum: str = hashlib.sha256(archive).hexdigest()
        self.urls: list[str] = []

    def get_bytes(self, url: str) -> bytes:
        self.urls.append(url)
        return self.archive

    def get_text(self, url: str) -> str:
        self.urls.append(url)
        return f"{self.checksum}  BTCUSDT-1m-2026-06.zip"


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
    assert status["source_count"] == 2
    source_lines = (
        tmp_path / "manifests" / "binance" / "usdm" / "sources.jsonl"
    ).read_text().splitlines()
    assert len(source_lines) == 2


def test_refresh_records_archive_exception_as_item_failure(tmp_path: Path) -> None:
    result = refresh_market_data(
        RefreshRequest(
            root=tmp_path,
            symbols=("BTCUSDT",),
            start_day=date(2026, 6, 9),
            end_day=date(2026, 6, 9),
            datasets=("klines",),
            interval="1m",
            optimize=True,
        ),
        archive_client=FailingArchiveClient(),
        funding_client=FakeFundingClient(),
    )

    assert result.status == "failed"
    status = cast(
        dict[str, object],
        json.loads((tmp_path / "manifests" / "binance" / "usdm" / "status.json").read_text()),
    )
    failures = cast("list[dict[str, object]]", status["failures"])
    assert failures[0]["error_code"] == "archive_exception"
    assert failures[0]["retryable"] is True


def test_refresh_monthly_archive_downloads_each_month_once(tmp_path: Path) -> None:
    archive_client = RecordingArchiveClient(
        _zip_csv(
            "BTCUSDT-1m-2026-06.csv",
            "1780963200000,100,101,99,100.5,10,1780963259999,1005,3,5,500,0\n",
        )
    )

    result = refresh_market_data(
        RefreshRequest(
            root=tmp_path,
            symbols=("BTCUSDT",),
            start_day=date(2026, 6, 1),
            end_day=date(2026, 6, 30),
            datasets=("klines",),
            interval="1m",
            optimize=False,
            archive_granularity="monthly",
        ),
        archive_client=archive_client,
        funding_client=FakeFundingClient(),
    )

    assert result.status == "succeeded"
    assert result.success_count == 1
    zip_urls = [url for url in archive_client.urls if url.endswith(".zip")]
    monthly_url = (
        "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1m/"
        "BTCUSDT-1m-2026-06.zip"
    )
    assert zip_urls == [monthly_url]
    assert (
        tmp_path
        / "binance"
        / "futures"
        / "klines"
        / "BTCUSDT"
        / "BTCUSDT_klines_1m_2026-06.parquet"
    ).exists()


def _zip_csv(name: str, body: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, body)
    return buffer.getvalue()
