from __future__ import annotations

import hashlib
import io
import zipfile
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from binance_usdm_parquet_data.archive_download import (
    ArchiveDownloadFailure,
    ArchiveHttpClient,
    DailyArchiveRequest,
    MonthlyArchiveRequest,
    download_daily_archive_to_parquet,
    download_monthly_archive_to_parquet,
)
from binance_usdm_parquet_data.paths import InvalidMarketDataKeyError


class FakeArchiveClient:
    def __init__(self, archive: bytes, checksum_text: str) -> None:
        self.archive: bytes = archive
        self.checksum_text: str = checksum_text
        self.requested_urls: list[str] = []

    def get_bytes(self, url: str) -> bytes:
        self.requested_urls.append(url)
        assert url.endswith(".zip")
        return self.archive

    def get_text(self, url: str) -> str:
        self.requested_urls.append(url)
        assert url.endswith(".zip.CHECKSUM")
        return self.checksum_text


def test_checksum_mismatch_records_failure_without_partial_publish(tmp_path: Path) -> None:
    client: ArchiveHttpClient = FakeArchiveClient(_zip_csv("bad.csv", "1,2,3\n"), "0" * 64)

    result = download_daily_archive_to_parquet(
        client,
        DailyArchiveRequest("klines", "BTCUSDT", "1m", date(2026, 6, 9)),
        tmp_path,
    )

    assert isinstance(result, ArchiveDownloadFailure)
    assert result.error_code == "checksum_mismatch"
    assert list(tmp_path.rglob("*.parquet")) == []


def test_daily_kline_zip_is_normalized_to_parquet(tmp_path: Path) -> None:
    body = "1700000000000,100.1,101.2,99.9,100.8,12.34,1700000059999,1234.5,42,6.0,600.0,0\n"
    archive = _zip_csv("BTCUSDT-1m-2026-06-09.csv", body)
    checksum = hashlib.sha256(archive).hexdigest()
    client: ArchiveHttpClient = FakeArchiveClient(archive, f"{checksum}  BTCUSDT-1m-2026-06-09.zip")

    result = download_daily_archive_to_parquet(
        client,
        DailyArchiveRequest("klines", "BTCUSDT", "1m", date(2026, 6, 9)),
        tmp_path,
    )

    assert not isinstance(result, ArchiveDownloadFailure)
    assert result.row_count == 1
    frame = pl.read_parquet(result.output_path)
    assert frame.schema["open"] == pl.Float64
    assert frame.item(0, "trade_count") == 42


def test_daily_kline_zip_with_header_is_normalized_to_parquet(tmp_path: Path) -> None:
    body = (
        "open_time,open,high,low,close,volume,close_time,quote_volume,count,"
        "taker_buy_volume,taker_buy_quote_volume,ignore\n"
        "1700000000000,100.1,101.2,99.9,100.8,12.34,1700000059999,"
        "1234.5,42,6.0,600.0,0\n"
    )
    archive = _zip_csv("BTCUSDT-1m-2026-06-09.csv", body)
    checksum = hashlib.sha256(archive).hexdigest()
    client: ArchiveHttpClient = FakeArchiveClient(archive, f"{checksum}  BTCUSDT-1m-2026-06-09.zip")

    result = download_daily_archive_to_parquet(
        client,
        DailyArchiveRequest("klines", "BTCUSDT", "1m", date(2026, 6, 9)),
        tmp_path,
    )

    assert not isinstance(result, ArchiveDownloadFailure)
    assert pl.read_parquet(result.output_path).item(0, "trade_count") == 42


def test_monthly_kline_zip_is_normalized_to_monthly_parquet(tmp_path: Path) -> None:
    body = "1700000000000,100.1,101.2,99.9,100.8,12.34,1700000059999,1234.5,42,6.0,600.0,0\n"
    archive = _zip_csv("BTCUSDT-1m-2026-06.csv", body)
    checksum = hashlib.sha256(archive).hexdigest()
    client: ArchiveHttpClient = FakeArchiveClient(archive, f"{checksum}  BTCUSDT-1m-2026-06.zip")

    result = download_monthly_archive_to_parquet(
        client,
        MonthlyArchiveRequest("klines", "BTCUSDT", "1m", "2026-06"),
        tmp_path,
    )

    assert not isinstance(result, ArchiveDownloadFailure)
    assert result.row_count == 1
    assert result.output_path.name == "BTCUSDT_klines_1m_2026-06.parquet"
    assert pl.read_parquet(result.output_path).item(0, "trade_count") == 42


def test_symbol_path_segment_cannot_escape_root(tmp_path: Path) -> None:
    body = "1700000000000,100.1,101.2,99.9,100.8,12.34,1700000059999,1234.5,42,6.0,600.0,0\n"
    archive = _zip_csv("ABSUSDT-1m-2026-06-09.csv", body)
    checksum = hashlib.sha256(archive).hexdigest()
    client: ArchiveHttpClient = FakeArchiveClient(archive, f"{checksum}  ABSUSDT-1m-2026-06-09.zip")

    result = download_daily_archive_to_parquet(
        client,
        DailyArchiveRequest("klines", "/ABSUSDT", "1m", date(2026, 6, 9)),
        tmp_path,
    )

    assert not isinstance(result, ArchiveDownloadFailure)
    assert result.output_path.is_relative_to(tmp_path)
    assert result.output_path.parent.name == "%2FABSUSDT"
    assert not Path("/ABSUSDT_klines_1m_2026-06-09.parquet").exists()


def test_daily_archive_rejects_traversal_interval_before_url_or_write(
    tmp_path: Path,
) -> None:
    body = "1700000000000,100.1,101.2,99.9,100.8,12.34,1700000059999,1234.5,42,6.0,600.0,0\n"
    archive = _zip_csv("BTCUSDT-1m-2026-06-09.csv", body)
    checksum = hashlib.sha256(archive).hexdigest()
    client = FakeArchiveClient(archive, f"{checksum}  BTCUSDT-1m-2026-06-09.zip")

    with pytest.raises(InvalidMarketDataKeyError) as exc_info:
        _ = download_daily_archive_to_parquet(
            client,
            DailyArchiveRequest("klines", "BTCUSDT", "../../escape", date(2026, 6, 9)),
            tmp_path,
        )

    assert exc_info.value.field == "interval"
    assert client.requested_urls == []
    assert list(tmp_path.rglob("*.parquet")) == []


def test_monthly_archive_rejects_traversal_interval_before_url_or_write(
    tmp_path: Path,
) -> None:
    body = "1700000000000,100.1,101.2,99.9,100.8,12.34,1700000059999,1234.5,42,6.0,600.0,0\n"
    archive = _zip_csv("BTCUSDT-1m-2026-06.csv", body)
    checksum = hashlib.sha256(archive).hexdigest()
    client = FakeArchiveClient(archive, f"{checksum}  BTCUSDT-1m-2026-06.zip")

    with pytest.raises(InvalidMarketDataKeyError) as exc_info:
        _ = download_monthly_archive_to_parquet(
            client,
            MonthlyArchiveRequest("klines", "BTCUSDT", "../../escape", "2026-06"),
            tmp_path,
        )

    assert exc_info.value.field == "interval"
    assert client.requested_urls == []
    assert list(tmp_path.rglob("*.parquet")) == []


def _zip_csv(name: str, body: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, body)
    return buffer.getvalue()
