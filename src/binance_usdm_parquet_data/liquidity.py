from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from binance_usdm_parquet_data.funding_rate import JsonValue


@dataclass(frozen=True, slots=True)
class QuoteLiquidity:
    quote_asset: str
    symbol_count: int
    aggregate_quote_volume: float


class JsonHttpClient(Protocol):
    def get_json(self, url: str, params: dict[str, str]) -> JsonValue: ...


def fetch_current_quote_liquidity(
    http: JsonHttpClient,
    *,
    quote_assets: tuple[str, ...] = ("USDT", "USDC"),
) -> tuple[QuoteLiquidity, ...]:
    payload = http.get_json("https://fapi.binance.com/fapi/v1/ticker/24hr", {})
    if not isinstance(payload, list):
        msg = "24hr ticker response must be a JSON array"
        raise TypeError(msg)
    totals = {quote_asset: [0, 0.0] for quote_asset in quote_assets}
    for item in payload:
        if not isinstance(item, dict):
            continue
        symbol = item.get("symbol")
        quote_volume = item.get("quoteVolume")
        if not isinstance(symbol, str) or not isinstance(quote_volume, str | int | float):
            continue
        quote_asset = _matching_quote(symbol, quote_assets)
        if quote_asset is None:
            continue
        totals[quote_asset][0] += 1
        totals[quote_asset][1] += float(quote_volume)
    return tuple(
        QuoteLiquidity(
            quote_asset=quote_asset,
            symbol_count=int(totals[quote_asset][0]),
            aggregate_quote_volume=totals[quote_asset][1],
        )
        for quote_asset in quote_assets
    )


def _matching_quote(symbol: str, quote_assets: tuple[str, ...]) -> str | None:
    for quote_asset in quote_assets:
        if symbol.endswith(quote_asset):
            return quote_asset
    return None
