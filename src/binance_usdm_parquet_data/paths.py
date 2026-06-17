from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final, override
from urllib.parse import quote

from binance_usdm_parquet_data.config import DEFAULT_ROOT

DEFAULT_CONTAINER_MARKET_DATA_ROOT: Final = Path("/market-data")
SYMBOL_RE: Final = re.compile(r"^[A-Z0-9_%=\-]+$")
INTERVAL_RE: Final = re.compile(r"^[0-9]+[mhdwM]$")
SYMBOL_SAFE_CHARS: Final = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_%=-"


@dataclass(frozen=True, slots=True)
class InvalidMarketDataKeyError(ValueError):
    field: str
    value: str

    @override
    def __str__(self) -> str:
        return f"Invalid market data {self.field}: {self.value}"


def market_data_root(root: Path | None = None) -> Path:
    if root is not None:
        return root.expanduser()
    configured = os.environ.get("MARKET_DATA_ROOT")
    if configured:
        return Path(configured).expanduser()
    if DEFAULT_CONTAINER_MARKET_DATA_ROOT.exists():
        return DEFAULT_CONTAINER_MARKET_DATA_ROOT
    return DEFAULT_ROOT


def raw_futures_root(root: Path | None = None) -> Path:
    return market_data_root(root) / "binance" / "futures"


def optimized_futures_root(root: Path | None = None) -> Path:
    return market_data_root(root) / "parbp_optimized" / "binance" / "futures"


def manifest_root(root: Path | None = None) -> Path:
    return market_data_root(root) / "manifests" / "binance" / "usdm"


def require_symbol(value: str) -> str:
    symbol = quote(value.strip().upper(), safe=SYMBOL_SAFE_CHARS)
    if not SYMBOL_RE.match(symbol):
        field = "symbol"
        raise InvalidMarketDataKeyError(field, value)
    return symbol


def require_interval(value: str) -> str:
    if not INTERVAL_RE.match(value):
        field = "interval"
        raise InvalidMarketDataKeyError(field, value)
    return value


def optimized_klines_file(root: Path | None, symbol: str, interval: str) -> Path:
    return optimized_klines_file_from_futures_root(
        optimized_futures_root(root),
        symbol,
        interval,
    )


def optimized_klines_file_from_futures_root(root: Path, symbol: str, interval: str) -> Path:
    checked_symbol = require_symbol(symbol)
    checked_interval = require_interval(interval)
    return (
        root
        / "klines"
        / f"symbol={checked_symbol}"
        / f"interval={checked_interval}"
        / "candles.parquet"
    )


def raw_klines_files(root: Path | None, symbol: str, interval: str) -> list[Path]:
    return raw_klines_files_from_futures_root(raw_futures_root(root), symbol, interval)


def raw_klines_files_from_futures_root(root: Path, symbol: str, interval: str) -> list[Path]:
    checked_symbol = require_symbol(symbol)
    checked_interval = require_interval(interval)
    directories = [root / "klines" / checked_symbol]
    legacy_symbol = _legacy_raw_symbol_dir_name(symbol, checked_symbol)
    if legacy_symbol is not None:
        directories.append(root / "klines" / legacy_symbol)
    return sorted(
        path
        for directory in directories
        for path in directory.glob(f"{directory.name}_klines_{checked_interval}_*.parquet")
    )


def kline_files_for_read(root: Path | None, symbol: str, interval: str) -> list[Path]:
    return kline_files_for_read_from_optimized_root(
        optimized_futures_root(root),
        symbol,
        interval,
    )


def kline_files_for_read_from_optimized_root(root: Path, symbol: str, interval: str) -> list[Path]:
    optimized = optimized_klines_file_from_futures_root(root, symbol, interval)
    return [optimized] if optimized.exists() else []


def funding_rate_files(root: Path | None, symbol: str) -> list[Path]:
    checked_symbol = require_symbol(symbol)
    return sorted((raw_futures_root(root) / "fundingRate" / checked_symbol).glob("*.parquet"))


def premium_index_kline_files(root: Path | None, symbol: str, interval: str) -> list[Path]:
    checked_symbol = require_symbol(symbol)
    checked_interval = require_interval(interval)
    directory = raw_futures_root(root) / "premiumIndexKlines" / checked_symbol
    pattern = f"{checked_symbol}_premiumIndexKlines_{checked_interval}_*.parquet"
    return sorted(directory.glob(pattern))


def _legacy_raw_symbol_dir_name(symbol: str, checked_symbol: str) -> str | None:
    legacy_symbol = symbol.strip().upper()
    if legacy_symbol == checked_symbol or legacy_symbol in {"", ".", ".."}:
        return None
    if "/" in legacy_symbol or "\\" in legacy_symbol:
        return None
    if quote(legacy_symbol, safe=SYMBOL_SAFE_CHARS) != checked_symbol:
        return None
    return legacy_symbol
