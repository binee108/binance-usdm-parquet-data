from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

type JsonValue = str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class FundingRateRecord:
    funding_time: int
    funding_rate: float
    mark_price: float | None


@dataclass(frozen=True, slots=True)
class FundingRateRequest:
    symbol: str
    start_time_ms: int
    end_time_ms: int
    limit: int = 1000


class FundingRateClient(Protocol):
    def get_funding_rates(
        self,
        *,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int,
    ) -> list[FundingRateRecord]: ...


class JsonHttpClient(Protocol):
    def get_json(self, url: str, params: dict[str, str]) -> JsonValue: ...


@dataclass(frozen=True, slots=True)
class BinanceFundingRateClient:
    http: JsonHttpClient
    base_url: str = "https://fapi.binance.com"

    def get_funding_rates(
        self,
        *,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int,
    ) -> list[FundingRateRecord]:
        payload = self.http.get_json(
            f"{self.base_url.rstrip('/')}/fapi/v1/fundingRate",
            {
                "symbol": symbol,
                "startTime": str(start_time_ms),
                "endTime": str(end_time_ms),
                "limit": str(limit),
            },
        )
        if not isinstance(payload, list):
            msg = "fundingRate response must be a JSON array"
            raise TypeError(msg)
        return [_parse_funding_item(item) for item in payload]


def collect_funding_rates(
    client: FundingRateClient,
    request: FundingRateRequest,
) -> list[FundingRateRecord]:
    cursor = request.start_time_ms
    by_time: dict[int, FundingRateRecord] = {}
    while cursor <= request.end_time_ms:
        page = client.get_funding_rates(
            symbol=request.symbol,
            start_time_ms=cursor,
            end_time_ms=request.end_time_ms,
            limit=min(request.limit, 1000),
        )
        if not page:
            break
        for record in page:
            if request.start_time_ms <= record.funding_time <= request.end_time_ms:
                by_time[record.funding_time] = record
        next_cursor = max(record.funding_time for record in page) + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(page) < min(request.limit, 1000):
            break
    return [by_time[key] for key in sorted(by_time)]


def _parse_funding_item(item: JsonValue) -> FundingRateRecord:
    if not isinstance(item, dict):
        msg = f"fundingRate item must be an object: {item!r}"
        raise TypeError(msg)
    funding_time = item.get("fundingTime")
    funding_rate = item.get("fundingRate")
    mark_price = item.get("markPrice")
    if not isinstance(funding_time, int):
        msg = f"invalid fundingTime: {funding_time!r}"
        raise TypeError(msg)
    if not isinstance(funding_rate, str | int | float):
        msg = f"invalid fundingRate: {funding_rate!r}"
        raise TypeError(msg)
    if mark_price is not None and not isinstance(mark_price, str | int | float):
        msg = f"invalid markPrice: {mark_price!r}"
        raise TypeError(msg)
    return FundingRateRecord(
        funding_time=funding_time,
        funding_rate=float(funding_rate),
        mark_price=None if mark_price is None else float(mark_price),
    )
