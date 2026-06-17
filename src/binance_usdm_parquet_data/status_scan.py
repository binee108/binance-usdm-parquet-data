from __future__ import annotations

import json
import re
from calendar import monthrange
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from binance_usdm_parquet_data.records import FreshnessSummary, LocalStatusSummary

KLINE_PATTERN_PREFIX: Final = r"^(?P<symbol>.+)_klines_(?P<interval>[A-Za-z0-9]+)_"
KLINE_PATTERN_SUFFIX: Final = r"(?P<month>\d{4}-\d{2})\.parquet$"
PREMIUM_PATTERN_PREFIX: Final = (
    r"^(?P<symbol>.+)_premiumIndexKlines_(?P<interval>[A-Za-z0-9]+)_"
)
PREMIUM_PATTERN_SUFFIX: Final = r"(?P<period>\d{4}-\d{2}(?:-\d{2})?)\.parquet$"
KLINE_FILE_RE: Final = re.compile(f"{KLINE_PATTERN_PREFIX}{KLINE_PATTERN_SUFFIX}")
PREMIUM_FILE_RE: Final = re.compile(f"{PREMIUM_PATTERN_PREFIX}{PREMIUM_PATTERN_SUFFIX}")
FUNDING_FILE_RE: Final = re.compile(
    r"^(?P<symbol>.+)_fundingRate_(?P<day>\d{4}-\d{2}-\d{2})\.parquet$"
)

type JsonValue = str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class LocalObservation:
    dataset: str
    interval: str | None
    symbol: str
    latest_complete_utc_day: str | None
    source_count: int


def scan_local_status(root: Path) -> LocalStatusSummary:
    observations = [
        *_optimized_kline_observations(root),
        *_premium_index_kline_observations(root),
        *_funding_rate_observations(root),
    ]
    return LocalStatusSummary(
        source_count=sum(observation.source_count for observation in observations),
        freshness=_summarize_observations(observations),
    )


def _optimized_kline_observations(root: Path) -> tuple[LocalObservation, ...]:
    optimized_root = root / "parbp_optimized" / "binance" / "futures" / "klines"
    if not optimized_root.exists():
        return ()
    observations: list[LocalObservation] = []
    for candles_path in optimized_root.glob("symbol=*/interval=*/candles.parquet"):
        symbol = candles_path.parent.parent.name.removeprefix("symbol=")
        interval = candles_path.parent.name.removeprefix("interval=")
        manifest = _read_json_object(candles_path.with_name("candles.manifest.json"))
        observations.append(
            LocalObservation(
                dataset="klines",
                interval=interval,
                symbol=symbol,
                latest_complete_utc_day=_latest_day_from_optimized_manifest(manifest),
                source_count=_source_count_from_optimized_manifest(manifest),
            )
        )
    return tuple(observations)


def _premium_index_kline_observations(root: Path) -> tuple[LocalObservation, ...]:
    raw_root = root / "binance" / "futures" / "premiumIndexKlines"
    if not raw_root.exists():
        return ()
    observations: list[LocalObservation] = []
    for parquet_path in raw_root.glob("*/*.parquet"):
        match = PREMIUM_FILE_RE.match(parquet_path.name)
        if match is not None:
            observations.append(
                LocalObservation(
                    dataset="premiumIndexKlines",
                    interval=match.group("interval"),
                    symbol=match.group("symbol"),
                    latest_complete_utc_day=_period_end_day(match.group("period")),
                    source_count=1,
                )
            )
    return tuple(observations)


def _funding_rate_observations(root: Path) -> tuple[LocalObservation, ...]:
    raw_root = root / "binance" / "futures" / "fundingRate"
    if not raw_root.exists():
        return ()
    observations: list[LocalObservation] = []
    for parquet_path in raw_root.glob("*/*.parquet"):
        match = FUNDING_FILE_RE.match(parquet_path.name)
        if match is not None:
            observations.append(
                LocalObservation(
                    dataset="fundingRate",
                    interval=None,
                    symbol=match.group("symbol"),
                    latest_complete_utc_day=match.group("day"),
                    source_count=1,
                )
            )
    return tuple(observations)


def _summarize_observations(observations: list[LocalObservation]) -> tuple[FreshnessSummary, ...]:
    grouped: dict[tuple[str, str | None], list[LocalObservation]] = {}
    for observation in observations:
        grouped.setdefault((observation.dataset, observation.interval), []).append(observation)
    summaries: list[FreshnessSummary] = []
    for (dataset, interval), group in grouped.items():
        summaries.append(
            FreshnessSummary(
                dataset=dataset,
                interval=interval,
                symbol_count=len({observation.symbol for observation in group}),
                latest_complete_utc_day=max(
                    (
                        item.latest_complete_utc_day
                        for item in group
                        if item.latest_complete_utc_day
                    ),
                    default=None,
                ),
            )
        )
    return tuple(sorted(summaries, key=lambda item: (item.dataset, item.interval or "")))


def _read_json_object(path: Path) -> JsonObject:
    if not path.exists():
        return {}
    raw = cast("JsonValue", json.loads(path.read_text(encoding="utf-8")))
    return raw if isinstance(raw, dict) else {}


def _latest_day_from_optimized_manifest(manifest: JsonObject) -> str | None:
    last_source = manifest.get("last_source")
    if not isinstance(last_source, str):
        return None
    match = KLINE_FILE_RE.match(Path(last_source).name)
    return _month_end_day(match.group("month")) if match else None


def _source_count_from_optimized_manifest(manifest: JsonObject) -> int:
    source_files = manifest.get("source_files")
    return source_files if isinstance(source_files, int) and source_files > 0 else 1


def _month_end_day(month: str) -> str:
    year, month_number = month.split("-", maxsplit=1)
    day = monthrange(int(year), int(month_number))[1]
    return f"{year}-{month_number}-{day:02d}"


def _period_end_day(period: str) -> str:
    if len(period) == len("YYYY-MM-DD"):
        return period
    return _month_end_day(period)
