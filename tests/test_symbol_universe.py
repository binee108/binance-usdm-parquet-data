from __future__ import annotations

from binance_usdm_parquet_data.archive_index import ArchivePrefixPage
from binance_usdm_parquet_data.symbol_universe import (
    ArchiveDataset,
    ArchiveListingClient,
    build_symbol_universe,
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
