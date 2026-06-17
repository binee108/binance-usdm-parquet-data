from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Final, Protocol, cast
from urllib.parse import unquote

from binance_usdm_parquet_data.archive_index import ArchivePrefixPage
from binance_usdm_parquet_data.paths import manifest_root
from binance_usdm_parquet_data.storage_keys import symbol_storage_key

type JsonValue = str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]


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
SYMBOL_UNIVERSE_FILE: Final = "symbol_universe.json"


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
                storage_key=symbol_storage_key(symbol),
                datasets={
                    dataset: symbol in symbols for dataset, symbols in dataset_symbols.items()
                },
            )
            for symbol in selected
        )
    )


def read_symbol_universe(root: Path) -> SymbolUniverse:
    path = manifest_root(root) / SYMBOL_UNIVERSE_FILE
    if not path.exists():
        return SymbolUniverse(symbols=())
    raw = cast("JsonValue", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(raw, dict):
        msg = f"symbol universe manifest is not an object: {path}"
        raise TypeError(msg)
    return _universe_from_dict(cast("JsonObject", raw))


def publish_symbol_universe(root: Path, universe: SymbolUniverse | tuple[str, ...]) -> None:
    match universe:
        case SymbolUniverse():
            payload = _universe_dict(universe)
        case tuple():
            payload = _simple_universe_dict(universe)
    _atomic_write_json(manifest_root(root) / SYMBOL_UNIVERSE_FILE, payload)


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


def _universe_from_dict(payload: JsonObject) -> SymbolUniverse:
    raw_symbols = payload.get("symbols")
    if not isinstance(raw_symbols, list):
        msg = "symbol universe field symbols must be a list"
        raise TypeError(msg)
    quote_asset = payload.get("quote_asset")
    default_quote_asset = quote_asset if isinstance(quote_asset, str) else "USDT"
    return SymbolUniverse(
        symbols=tuple(_symbol_metadata_from_json(item, default_quote_asset) for item in raw_symbols)
    )


def _symbol_metadata_from_json(item: JsonValue, default_quote_asset: str) -> SymbolMetadata:
    if isinstance(item, str):
        return SymbolMetadata(
            symbol=item,
            quote_asset=default_quote_asset,
            storage_key=symbol_storage_key(item),
            datasets={
                ArchiveDataset.KLINES: True,
                ArchiveDataset.PREMIUM_INDEX_KLINES: False,
            },
        )
    if isinstance(item, dict):
        payload = cast("JsonObject", item)
        symbol = _required_str(payload, "symbol")
        quote_asset = _optional_str(payload, "quote_asset") or default_quote_asset
        storage_key = _optional_str(payload, "storage_key") or symbol_storage_key(symbol)
        return SymbolMetadata(
            symbol=symbol,
            quote_asset=quote_asset,
            storage_key=storage_key,
            datasets=_datasets_from_json(payload.get("datasets")),
        )
    msg = "symbol universe symbols must be strings or objects"
    raise TypeError(msg)


def _datasets_from_json(raw: JsonValue) -> dict[ArchiveDataset, bool]:
    if not isinstance(raw, dict):
        return {
            ArchiveDataset.KLINES: True,
            ArchiveDataset.PREMIUM_INDEX_KLINES: False,
        }
    payload = cast("JsonObject", raw)
    return {
        ArchiveDataset.KLINES: _dataset_flag(payload, "klines"),
        ArchiveDataset.PREMIUM_INDEX_KLINES: _dataset_flag(payload, "premium_index_klines")
        or _dataset_flag(payload, "premiumIndexKlines"),
    }


def _dataset_flag(payload: JsonObject, key: str) -> bool:
    value = payload.get(key)
    return value if isinstance(value, bool) else False


def _universe_dict(universe: SymbolUniverse) -> JsonObject:
    quote_assets = _json_strings(sorted({symbol.quote_asset for symbol in universe.symbols}))
    symbols: list[JsonValue] = [_metadata_dict(symbol) for symbol in universe.symbols]
    return {
        "quote_assets": quote_assets,
        "symbol_count": len(universe.symbols),
        "symbols": symbols,
    }


def _simple_universe_dict(symbols: tuple[str, ...]) -> JsonObject:
    return {"quote_asset": "USDT", "symbols": _json_strings(sorted(symbols))}


def _json_strings(values: Iterable[str]) -> list[JsonValue]:
    items: list[JsonValue] = []
    items.extend(values)
    return items


def _metadata_dict(symbol: SymbolMetadata) -> JsonObject:
    datasets: JsonObject = {
        dataset.value: available for dataset, available in symbol.datasets.items()
    }
    return {
        "symbol": symbol.symbol,
        "quote_asset": symbol.quote_asset,
        "storage_key": symbol.storage_key,
        "datasets": datasets,
    }


def _required_str(item: JsonObject, key: str) -> str:
    value = item.get(key)
    if isinstance(value, str):
        return value
    msg = f"symbol universe field {key} must be a string"
    raise TypeError(msg)


def _optional_str(item: JsonObject, key: str) -> str | None:
    value = item.get(key)
    if value is None or isinstance(value, str):
        return value
    msg = f"symbol universe field {key} must be a string or null"
    raise TypeError(msg)


def _atomic_write_json(path: Path, payload: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n"
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        _ = handle.write(text)
        temp_path = Path(handle.name)
    _ = temp_path.replace(path)
