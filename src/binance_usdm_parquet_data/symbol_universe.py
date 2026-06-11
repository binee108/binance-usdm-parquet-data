from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol
from urllib.parse import unquote

from binance_usdm_parquet_data.archive_index import ArchivePrefixPage, encode_path_segment


class ArchiveDataset(StrEnum):
    KLINES = "klines"
    PREMIUM_INDEX_KLINES = "premium_index_klines"


@dataclass(frozen=True, slots=True)
class SymbolMetadata:
    symbol: str
    quote_asset: str
    storage_key: str
    datasets: dict[ArchiveDataset, bool]


@dataclass(frozen=True, slots=True)
class SymbolUniverse:
    symbols: tuple[SymbolMetadata, ...]


class ArchiveListingClient(Protocol):
    def list_prefix_pages(self, prefix: str) -> list[ArchivePrefixPage]: ...


PREFIX_BY_DATASET = {
    ArchiveDataset.KLINES: "data/futures/um/monthly/klines/",
    ArchiveDataset.PREMIUM_INDEX_KLINES: "data/futures/um/monthly/premiumIndexKlines/",
}


def build_symbol_universe(
    client: ArchiveListingClient,
    *,
    quote_assets: tuple[str, ...] = ("USDT",),
) -> SymbolUniverse:
    dataset_symbols = {
        dataset: _symbols_from_pages(client.list_prefix_pages(prefix))
        for dataset, prefix in PREFIX_BY_DATASET.items()
    }
    canonical = dataset_symbols[ArchiveDataset.KLINES]
    selected = sorted(
        (symbol for symbol in canonical if _quote_asset(symbol, quote_assets) is not None),
        key=lambda symbol: _symbol_sort_key(symbol, quote_assets),
    )
    return SymbolUniverse(
        symbols=tuple(
            SymbolMetadata(
                symbol=symbol,
                quote_asset=_quote_asset(symbol, quote_assets) or "",
                storage_key=encode_path_segment(symbol),
                datasets={
                    dataset: symbol in symbols for dataset, symbols in dataset_symbols.items()
                },
            )
            for symbol in selected
        )
    )


def _symbols_from_pages(pages: list[ArchivePrefixPage]) -> set[str]:
    symbols: set[str] = set()
    for page in pages:
        for prefix in page.prefixes:
            normalized = prefix.rstrip("/")
            raw_symbol = normalized.rsplit("/", maxsplit=1)[-1]
            symbols.add(unquote(raw_symbol))
    return symbols


def _quote_asset(symbol: str, quote_assets: tuple[str, ...]) -> str | None:
    for quote_asset in quote_assets:
        if symbol.endswith(quote_asset):
            return quote_asset
    return None


def _symbol_sort_key(symbol: str, quote_assets: tuple[str, ...]) -> tuple[str, int, str]:
    quote_asset = _quote_asset(symbol, quote_assets)
    if quote_asset is None:
        return symbol, len(quote_assets), symbol
    return symbol.removesuffix(quote_asset), quote_assets.index(quote_asset), symbol
