from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import polars as pl

from binance_usdm_parquet_data.archive_download import ArchiveHttpClient, DownloadedArchiveFile
from binance_usdm_parquet_data.funding_archive import (
    FundingRateArchiveRequest,
    download_monthly_funding_rate_archive_to_parquet,
)
from binance_usdm_parquet_data.funding_rate import (
    BinanceFundingRateClient,
    FundingRateClient,
    FundingRateRecord,
    FundingRateRequest,
    JsonHttpClient,
    JsonValue,
    collect_funding_rates,
)


class FakeFundingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int, int]] = []

    def get_funding_rates(
        self,
        *,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int,
    ) -> list[FundingRateRecord]:
        del symbol
        self.calls.append((start_time_ms, end_time_ms, limit))
        if start_time_ms == 1000:
            return [
                FundingRateRecord(funding_time=1000, funding_rate=0.0001, mark_price=100.0),
                FundingRateRecord(funding_time=2000, funding_rate=0.0002, mark_price=101.0),
            ]
        if start_time_ms == 2001:
            return [
                FundingRateRecord(funding_time=2000, funding_rate=0.0002, mark_price=101.0),
                FundingRateRecord(funding_time=3000, funding_rate=0.0003, mark_price=102.0),
            ]
        return []


def test_funding_rate_pagination_advances_without_duplicates() -> None:
    client: FundingRateClient = FakeFundingClient()
    request = FundingRateRequest(symbol="BTCUSDT", start_time_ms=1000, end_time_ms=4000, limit=2)

    records = collect_funding_rates(client, request)

    assert [record.funding_time for record in records] == [1000, 2000, 3000]
    assert client.calls == [(1000, 4000, 2), (2001, 4000, 2), (3001, 4000, 2)]


class FakeJsonHttpClient:
    def get_json(self, url: str, params: dict[str, str]) -> JsonValue:
        assert url == "https://fapi.binance.com/fapi/v1/fundingRate"
        assert params == {
            "symbol": "BTCUSDT",
            "startTime": "1000",
            "endTime": "2000",
            "limit": "1000",
        }
        return [
            {"fundingTime": 1000, "fundingRate": "0.00010000", "markPrice": "100.5"},
            {"fundingTime": 2000, "fundingRate": "0.00020000", "markPrice": None},
        ]


def test_binance_rest_funding_client_parses_string_payload() -> None:
    client = BinanceFundingRateClient(http=FakeJsonHttpClient())
    json_client: JsonHttpClient = client.http

    records = client.get_funding_rates(
        symbol="BTCUSDT",
        start_time_ms=1000,
        end_time_ms=2000,
        limit=1000,
    )

    assert json_client is not None
    assert records == [
        FundingRateRecord(funding_time=1000, funding_rate=0.0001, mark_price=100.5),
        FundingRateRecord(funding_time=2000, funding_rate=0.0002, mark_price=None),
    ]


class FakeFundingArchiveClient:
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


def test_monthly_funding_archive_download_writes_normalized_parquet(tmp_path: Path) -> None:
    csv_body = _csv_body(
        "calc_time,funding_interval_hours,last_funding_rate",
        "1777593600000,8,-0.00003746",
        "1777622400005,8,-0.00001810",
    )
    archive_client: ArchiveHttpClient = FakeFundingArchiveClient(
        _zip_csv(
            "BTCUSDT-fundingRate-2026-05.csv",
            csv_body,
        )
    )

    result = download_monthly_funding_rate_archive_to_parquet(
        archive_client,
        FundingRateArchiveRequest(symbol="BTCUSDT", month="2026-05"),
        tmp_path,
    )

    assert isinstance(result, DownloadedArchiveFile)
    assert result.row_count == 2
    expected_url = (
        "https://data.binance.vision/data/futures/um/monthly/fundingRate/BTCUSDT/"
        "BTCUSDT-fundingRate-2026-05.zip"
    )
    assert result.source_url == expected_url
    frame = pl.read_parquet(result.output_path)
    assert frame["funding_rate"].to_list() == [-0.00003746, -0.00001810]
    assert frame["mark_price"].null_count() == 2


def _zip_csv(name: str, body: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, body)
    return buffer.getvalue()


def _csv_body(*rows: str) -> str:
    return "\n".join(rows) + "\n"
