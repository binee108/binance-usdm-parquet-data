from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QueryWindow:
    start_ts: int | None
    end_ts: int | None


@dataclass(frozen=True, slots=True)
class KlineRecord:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True, slots=True)
class IntrabarKlineRecord:
    timestamp: int
    open: float
    high: float
    low: float
    close: float


@dataclass(frozen=True, slots=True)
class FundingRateEventRecord:
    timestamp: int
    funding_rate: float


@dataclass(frozen=True, slots=True)
class MissingKlineRange:
    symbol: str
    interval: str
    missing_start_ts: int
    missing_end_ts: int
    missing_count: int
    observed_before_ts: int
    observed_after_ts: int
    reason: str = "timestamp_gap"


@dataclass(frozen=True, slots=True)
class FreshnessSummary:
    dataset: str
    interval: str | None
    symbol_count: int
    latest_complete_utc_day: str | None


@dataclass(frozen=True, slots=True)
class LocalStatusSummary:
    source_count: int
    freshness: tuple[FreshnessSummary, ...]
