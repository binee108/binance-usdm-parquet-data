from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from binance_usdm_parquet_data.archive_index import ArchivePrefixPage
from binance_usdm_parquet_data.symbol_universe import (
    ArchiveDataset,
    ArchiveListingClient,
    build_symbol_universe,
    publish_symbol_universe,
    read_symbol_universe,
)


class FakeListingClient:
    def __init__(self) -> None:
        self._pages: dict[str, list[ArchivePrefixPage]] = {
            "data/futures/um/monthly/klines/": [
                ArchivePrefixPage(
                    prefixes=(
                        "data/futures/um/monthly/klines/BTCUSDT/",
                        "data/futures/um/monthly/klines/DELISTEDUSDT/",
                        "data/futures/um/monthly/klines/BTCUSDC/",
                    ),
                    next_token="continuation",
                ),
                ArchivePrefixPage(
                    prefixes=(
                        "data/futures/um/monthly/klines/ETHBUSD/",
                        "data/futures/um/monthly/klines/%CE%A9USDT/",
                    ),
                    next_token=None,
                ),
            ],
            "data/futures/um/monthly/premiumIndexKlines/": [
                ArchivePrefixPage(
                    prefixes=("data/futures/um/monthly/premiumIndexKlines/BTCUSDT/",),
                    next_token=None,
                )
            ],
        }

    def list_prefix_pages(self, prefix: str) -> list[ArchivePrefixPage]:
        return self._pages[prefix]


def test_archive_first_universe_keeps_delisted_usdt_and_excludes_other_quotes() -> None:
    client: ArchiveListingClient = FakeListingClient()

    universe = build_symbol_universe(client, quote_assets=("USDT",))

    symbols = [symbol.symbol for symbol in universe.symbols]
    assert symbols == ["BTCUSDT", "DELISTEDUSDT", "ΩUSDT"]
    assert all(symbol.quote_asset == "USDT" for symbol in universe.symbols)
    delisted = next(symbol for symbol in universe.symbols if symbol.symbol == "DELISTEDUSDT")
    assert delisted.datasets[ArchiveDataset.KLINES] is True
    assert delisted.datasets[ArchiveDataset.PREMIUM_INDEX_KLINES] is False


def test_archive_universe_can_be_reconfigured_without_changing_default_scope() -> None:
    client: ArchiveListingClient = FakeListingClient()

    universe = build_symbol_universe(client, quote_assets=("USDT", "USDC"))

    assert [symbol.symbol for symbol in universe.symbols] == [
        "BTCUSDT",
        "BTCUSDC",
        "DELISTEDUSDT",
        "ΩUSDT",
    ]


def test_read_symbol_universe_parses_parbp_simple_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifests" / "binance" / "usdm" / "symbol_universe.json"
    manifest.parent.mkdir(parents=True)
    _ = manifest.write_text(
        json.dumps({"quote_asset": "USDT", "symbols": ["ETHUSDT", "BTCUSDT"]}) + "\n",
        encoding="utf-8",
    )

    universe = read_symbol_universe(tmp_path)

    assert [symbol.symbol for symbol in universe.symbols] == ["ETHUSDT", "BTCUSDT"]
    assert [symbol.quote_asset for symbol in universe.symbols] == ["USDT", "USDT"]


def test_publish_symbol_universe_accepts_symbols_and_writes_readable_manifest(
    tmp_path: Path,
) -> None:
    publish_symbol_universe(tmp_path, ("ETHUSDT", "BTCUSDT"))

    universe = read_symbol_universe(tmp_path)
    payload = cast(
        "dict[str, object]",
        json.loads(
            (tmp_path / "manifests" / "binance" / "usdm" / "symbol_universe.json").read_text(
                encoding="utf-8"
            )
        ),
    )

    assert [symbol.symbol for symbol in universe.symbols] == ["BTCUSDT", "ETHUSDT"]
    assert payload["quote_asset"] == "USDT"
