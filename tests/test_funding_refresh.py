from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, cast

import polars as pl

from binance_usdm_parquet_data.archive_download import ArchiveHttpClient
from binance_usdm_parquet_data.funding_rate import FundingRateClient, FundingRateRecord
from binance_usdm_parquet_data.refresh import RefreshRequest, refresh_market_data

if TYPE_CHECKING:
    import pytest


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


class MissingFundingArchiveClient:
    def get_bytes(self, url: str) -> bytes:
        del url
        message = "404 Not Found"
        raise OSError(message)

    def get_text(self, url: str) -> str:
        del url
        message = "404 Not Found"
        raise OSError(message)


class MissingFundingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int, int]] = []

    def get_funding_rates(
        self,
        *,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int,
    ) -> list[FundingRateRecord]:
        self.calls.append((symbol, start_time_ms, end_time_ms, limit))
        message = "400 Bad Request: Invalid symbol"
        raise OSError(message)


class FundingMonthlyArchiveClient:
    def __init__(self, archive: bytes) -> None:
        self.archive: bytes = archive
        self.checksum: str = hashlib.sha256(archive).hexdigest()
        self.urls: list[str] = []

    def get_bytes(self, url: str) -> bytes:
        self.urls.append(url)
        return self.archive

    def get_text(self, url: str) -> str:
        self.urls.append(url)
        return f"{self.checksum}  BTCUSDT-fundingRate-2026-05.zip"


class RecordingFundingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int, int]] = []

    def get_funding_rates(
        self,
        *,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int,
    ) -> list[FundingRateRecord]:
        self.calls.append((symbol, start_time_ms, end_time_ms, limit))
        return [FundingRateRecord(start_time_ms, 0.0001, 105_000.0)]


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
    funding_files = list(
        (tmp_path / "binance" / "futures" / "fundingRate" / "BTCUSDT").glob("*.parquet")
    )
    assert len(funding_files) == 1
    assert pl.read_parquet(funding_files[0]).item(0, "funding_rate") == 0.0001
    status = cast(
        dict[str, object],
        json.loads((tmp_path / "manifests" / "binance" / "usdm" / "status.json").read_text()),
    )
    assert status["failed_item_count"] == 0
    assert status["source_count"] == 2


def test_refresh_applies_funding_sleep_option(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sleep_calls: list[float] = []
    monkeypatch.setattr("binance_usdm_parquet_data.funding_refresh.sleep", sleep_calls.append)

    result = refresh_market_data(
        RefreshRequest(
            root=tmp_path,
            symbols=("BTCUSDT",),
            start_day=date(2026, 6, 9),
            end_day=date(2026, 6, 9),
            datasets=("fundingRate",),
            interval="1m",
            optimize=False,
            max_concurrent_downloads=7,
            http_timeout_seconds=12.5,
            funding_rest_sleep_seconds=0.25,
        ),
        archive_client=FailingArchiveClient(),
        funding_client=FakeFundingClient(),
    )

    assert result.status == "succeeded"
    assert sleep_calls == [0.25]


def test_refresh_skips_missing_funding_for_delisted_symbol(tmp_path: Path) -> None:
    funding_client = MissingFundingClient()

    result = refresh_market_data(
        RefreshRequest(
            root=tmp_path,
            symbols=("DELISTEDUSDT",),
            start_day=date(2026, 6, 9),
            end_day=date(2026, 6, 9),
            datasets=("fundingRate",),
            interval="1m",
            optimize=False,
        ),
        archive_client=MissingFundingArchiveClient(),
        funding_client=funding_client,
    )

    assert result.status == "succeeded"
    assert result.success_count == 0
    assert result.failure_count == 0
    assert len(funding_client.calls) == 1
    status = cast(
        dict[str, object],
        json.loads((tmp_path / "manifests" / "binance" / "usdm" / "status.json").read_text()),
    )
    assert status["failed_item_count"] == 0
    assert status["source_count"] == 0


def test_refresh_uses_monthly_funding_archive_for_complete_month(tmp_path: Path) -> None:
    csv_body = _csv_body(
        "calc_time,funding_interval_hours,last_funding_rate",
        "1777593600000,8,-0.00003746",
    )
    archive_client = FundingMonthlyArchiveClient(
        _zip_csv(
            "BTCUSDT-fundingRate-2026-05.csv",
            csv_body,
        )
    )
    funding_client = RecordingFundingClient()

    result = refresh_market_data(
        RefreshRequest(
            root=tmp_path,
            symbols=("BTCUSDT",),
            start_day=date(2026, 5, 1),
            end_day=date(2026, 5, 31),
            datasets=("fundingRate",),
            interval="1m",
            optimize=False,
            archive_granularity="monthly",
        ),
        archive_client=archive_client,
        funding_client=funding_client,
    )

    assert result.status == "succeeded"
    assert result.success_count == 1
    assert funding_client.calls == []
    zip_urls = [url for url in archive_client.urls if url.endswith(".zip")]
    expected_url = (
        "https://data.binance.vision/data/futures/um/monthly/fundingRate/BTCUSDT/"
        "BTCUSDT-fundingRate-2026-05.zip"
    )
    assert zip_urls == [expected_url]


def test_refresh_uses_rest_for_partial_monthly_funding_range(tmp_path: Path) -> None:
    csv_body = _csv_body(
        "calc_time,funding_interval_hours,last_funding_rate",
        "1777593600000,8,-0.00003746",
    )
    archive_client = FundingMonthlyArchiveClient(
        _zip_csv(
            "BTCUSDT-fundingRate-2026-05.csv",
            csv_body,
        )
    )
    funding_client = RecordingFundingClient()

    result = refresh_market_data(
        RefreshRequest(
            root=tmp_path,
            symbols=("BTCUSDT",),
            start_day=date(2026, 5, 30),
            end_day=date(2026, 6, 2),
            datasets=("fundingRate",),
            interval="1m",
            optimize=False,
            archive_granularity="monthly",
        ),
        archive_client=archive_client,
        funding_client=funding_client,
    )

    assert result.status == "succeeded"
    assert result.success_count == 4
    assert [url for url in archive_client.urls if url.endswith(".zip")] == []
    assert len(funding_client.calls) == 4


def _zip_csv(name: str, body: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, body)
    return buffer.getvalue()


def _csv_body(*rows: str) -> str:
    return "\n".join(rows) + "\n"
