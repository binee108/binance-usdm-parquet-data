from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

from typer.testing import CliRunner

from binance_usdm_parquet_data import cli
from binance_usdm_parquet_data.archive_download import ArchiveHttpClient
from binance_usdm_parquet_data.cli import app
from binance_usdm_parquet_data.funding_rate import FundingRateClient, FundingRateRecord
from binance_usdm_parquet_data.manifest import CollectorRun, DatasetFreshness, ManifestStore
from binance_usdm_parquet_data.refresh import RefreshRequest, RefreshResult
from binance_usdm_parquet_data.symbol_universe import (
    ArchiveDataset,
    SymbolMetadata,
    SymbolUniverse,
)

if TYPE_CHECKING:
    import pytest


def test_status_command_reads_manifest(tmp_path: Path) -> None:
    ManifestStore(tmp_path).publish_status(
        last_run=CollectorRun(
            run_id="run-cli",
            mode="manual",
            status="succeeded",
            started_at="2026-06-12T00:00:00+00:00",
            finished_at="2026-06-12T00:00:01+00:00",
            symbol_count=1,
            item_count=1,
            success_count=1,
            failure_count=0,
            last_error=None,
        ),
        freshness=(
            DatasetFreshness(
                dataset="klines",
                interval="1m",
                symbol_count=1,
                latest_complete_utc_day="2026-06-11",
            ),
        ),
        failures=(),
    )

    result = CliRunner().invoke(app, ["status", "--root", str(tmp_path)])

    assert result.exit_code == 0
    payload = cast("dict[str, object]", json.loads(result.output))
    last_run = cast("dict[str, object]", payload["last_run"])
    freshness = cast("list[dict[str, object]]", payload["freshness"])
    assert last_run["id"] == "run-cli"
    assert freshness[0]["latest_complete_utc_day"] == "2026-06-11"


class FakeArchiveClient:
    def get_bytes(self, url: str) -> bytes:
        return url.encode()

    def get_text(self, url: str) -> str:
        return url


class FakeFundingClient:
    def get_funding_rates(
        self,
        *,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int,
    ) -> list[FundingRateRecord]:
        _ = (symbol, start_time_ms, end_time_ms, limit)
        return []


def fake_funding_client(http: ArchiveHttpClient) -> FakeFundingClient:
    _ = http
    return FakeFundingClient()


def test_discover_command_reports_archive_first_usdt_universe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    universe = SymbolUniverse(
        symbols=(
            SymbolMetadata(
                symbol="BTCUSDT",
                quote_asset="USDT",
                storage_key="BTCUSDT",
                datasets={
                    ArchiveDataset.KLINES: True,
                    ArchiveDataset.PREMIUM_INDEX_KLINES: True,
                },
            ),
            SymbolMetadata(
                symbol="DELISTEDUSDT",
                quote_asset="USDT",
                storage_key="DELISTEDUSDT",
                datasets={
                    ArchiveDataset.KLINES: True,
                    ArchiveDataset.PREMIUM_INDEX_KLINES: False,
                },
            ),
        )
    )
    monkeypatch.setattr(cli, "_discover_usdt_universe", lambda: universe)

    result = CliRunner().invoke(app, ["discover"])

    assert result.exit_code == 0
    payload = cast("dict[str, object]", json.loads(result.output))
    symbols = cast("list[dict[str, object]]", payload["symbols"])
    assert payload["symbol_count"] == 2
    assert [item["symbol"] for item in symbols] == ["BTCUSDT", "DELISTEDUSDT"]


def test_refresh_all_discovers_usdt_symbols_before_refresh(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_symbols: list[tuple[str, ...]] = []
    monkeypatch.setattr(cli, "_discover_usdt_symbols", lambda: ("BTCUSDT", "DELISTEDUSDT"))
    monkeypatch.setattr(cli, "Httpx2SyncClient", FakeArchiveClient)
    monkeypatch.setattr(cli, "BinanceFundingRateClient", fake_funding_client)

    def fake_refresh_market_data(
        request: RefreshRequest,
        *,
        archive_client: ArchiveHttpClient,
        funding_client: FundingRateClient,
    ) -> RefreshResult:
        _ = (archive_client, funding_client)
        captured_symbols.append(request.symbols)
        return RefreshResult(
            run_id="run-all",
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
            "all",
            "--start-day",
            "2026-06-09",
            "--end-day",
            "2026-06-09",
        ],
    )

    assert result.exit_code == 0
    assert captured_symbols == [("BTCUSDT", "DELISTEDUSDT")]
