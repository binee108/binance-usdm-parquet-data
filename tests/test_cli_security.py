from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from binance_usdm_parquet_data import cli
from binance_usdm_parquet_data.archive_download import ArchiveHttpClient
from binance_usdm_parquet_data.cli import app
from binance_usdm_parquet_data.funding_rate import FundingRateClient
from binance_usdm_parquet_data.refresh import RefreshRequest, RefreshResult

if TYPE_CHECKING:
    import pytest


class FakeArchiveClient:
    def get_bytes(self, url: str) -> bytes:
        return url.encode()

    def get_text(self, url: str) -> str:
        return url


class FakeFundingClient:
    pass


def fake_funding_client(http: ArchiveHttpClient) -> FakeFundingClient:
    _ = http
    return FakeFundingClient()


def test_refresh_command_rejects_traversal_interval_before_refresh(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_requests: list[RefreshRequest] = []
    monkeypatch.setattr(cli, "Httpx2SyncClient", FakeArchiveClient)
    monkeypatch.setattr(cli, "BinanceFundingRateClient", fake_funding_client)

    def fake_refresh_market_data(
        request: RefreshRequest,
        *,
        archive_client: ArchiveHttpClient,
        funding_client: FundingRateClient,
    ) -> RefreshResult:
        _ = (archive_client, funding_client)
        captured_requests.append(request)
        return RefreshResult(
            run_id="run-invalid",
            status="succeeded",
            success_count=0,
            failure_count=0,
        )

    monkeypatch.setattr(cli, "refresh_market_data", fake_refresh_market_data)

    result = CliRunner().invoke(
        app,
        [
            "refresh",
            "--root",
            str(tmp_path),
            "--symbols",
            "BTCUSDT",
            "--start-day",
            "2026-06-09",
            "--end-day",
            "2026-06-09",
            "--interval",
            "../../escape",
        ],
    )

    assert result.exit_code != 0
    assert captured_requests == []


def test_optimize_command_rejects_traversal_interval_before_raw_glob(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_calls: list[tuple[str, str]] = []

    def fake_raw_kline_files(root: Path, symbol: str, interval: str) -> tuple[Path, ...]:
        _ = root
        captured_calls.append((symbol, interval))
        return ()

    monkeypatch.setattr(cli, "_raw_kline_files", fake_raw_kline_files)

    result = CliRunner().invoke(
        app,
        [
            "optimize",
            "--root",
            str(tmp_path),
            "--symbols",
            "BTCUSDT",
            "--interval",
            "../../escape",
        ],
    )

    assert result.exit_code != 0
    assert captured_calls == []
