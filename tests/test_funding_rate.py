from __future__ import annotations

from binance_usdm_parquet_data.funding_rate import (
    BinanceFundingRateClient,
    FundingRateClient,
    FundingRateRecord,
    FundingRateRequest,
    JsonHttpClient,
    JsonValue,
    collect_funding_rates,
)


class FakeFundingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int, int]] = []

    def get_funding_rates(
        self,
        *,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int,
    ) -> list[FundingRateRecord]:
        del symbol
        self.calls.append((start_time_ms, end_time_ms, limit))
        if start_time_ms == 1000:
            return [
                FundingRateRecord(funding_time=1000, funding_rate=0.0001, mark_price=100.0),
                FundingRateRecord(funding_time=2000, funding_rate=0.0002, mark_price=101.0),
            ]
        if start_time_ms == 2001:
            return [
                FundingRateRecord(funding_time=2000, funding_rate=0.0002, mark_price=101.0),
                FundingRateRecord(funding_time=3000, funding_rate=0.0003, mark_price=102.0),
            ]
        return []


def test_funding_rate_pagination_advances_without_duplicates() -> None:
    client: FundingRateClient = FakeFundingClient()
    request = FundingRateRequest(symbol="BTCUSDT", start_time_ms=1000, end_time_ms=4000, limit=2)

    records = collect_funding_rates(client, request)

    assert [record.funding_time for record in records] == [1000, 2000, 3000]
    assert client.calls == [(1000, 4000, 2), (2001, 4000, 2), (3001, 4000, 2)]


class FakeJsonHttpClient:
    def get_json(self, url: str, params: dict[str, str]) -> JsonValue:
        assert url == "https://fapi.binance.com/fapi/v1/fundingRate"
        assert params == {
            "symbol": "BTCUSDT",
            "startTime": "1000",
            "endTime": "2000",
            "limit": "1000",
        }
        return [
            {"fundingTime": 1000, "fundingRate": "0.00010000", "markPrice": "100.5"},
            {"fundingTime": 2000, "fundingRate": "0.00020000", "markPrice": None},
        ]


def test_binance_rest_funding_client_parses_string_payload() -> None:
    client = BinanceFundingRateClient(http=FakeJsonHttpClient())
    json_client: JsonHttpClient = client.http

    records = client.get_funding_rates(
        symbol="BTCUSDT",
        start_time_ms=1000,
        end_time_ms=2000,
        limit=1000,
    )

    assert json_client is not None
    assert records == [
        FundingRateRecord(funding_time=1000, funding_rate=0.0001, mark_price=100.5),
        FundingRateRecord(funding_time=2000, funding_rate=0.0002, mark_price=None),
    ]
