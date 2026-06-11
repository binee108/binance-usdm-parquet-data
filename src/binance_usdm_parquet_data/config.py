from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

DEFAULT_ROOT: Final = Path("~/Desktop/data").expanduser()


@dataclass(frozen=True, slots=True)
class CollectorConfig:
    root: Path = DEFAULT_ROOT
    quote_assets: tuple[str, ...] = ("USDT",)
    archive_base_url: str = "https://data.binance.vision/"
    futures_api_base_url: str = "https://fapi.binance.com"

    @property
    def raw_futures_root(self) -> Path:
        return self.root / "binance" / "futures"

    @property
    def optimized_futures_root(self) -> Path:
        return self.root / "parbp_optimized" / "binance" / "futures"

    @property
    def manifest_root(self) -> Path:
        return self.root / "manifests" / "binance" / "usdm"
