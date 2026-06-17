from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Final, TypedDict, cast

from binance_usdm_parquet_data.locks import (
    DEFAULT_LOCK_POLL_SECONDS,
    DEFAULT_LOCK_STALE_SECONDS,
    DEFAULT_LOCK_WAIT_SECONDS,
    LockOptions,
    SharedFileLockTimeoutError,
    SharedFileStaleLockError,
    shared_file_lock,
)
from binance_usdm_parquet_data.paths import manifest_root
from binance_usdm_parquet_data.records import MissingKlineRange

MISSING_KLINES_FILENAME: Final = "missing_klines.jsonl"
LOCK_WAIT_SECONDS: Final = DEFAULT_LOCK_WAIT_SECONDS
LOCK_STALE_SECONDS: Final = DEFAULT_LOCK_STALE_SECONDS
LOCK_POLL_SECONDS: Final = DEFAULT_LOCK_POLL_SECONDS
MissingKlinesLockTimeoutError = SharedFileLockTimeoutError
MissingKlinesStaleLockError = SharedFileStaleLockError


class MissingKlinePayload(TypedDict):
    dataset: str
    symbol: str
    interval: str
    missing_start_ts: int
    missing_end_ts: int
    missing_count: int
    observed_before_ts: int
    observed_after_ts: int
    reason: str


def missing_klines_path(root: Path | None) -> Path:
    return manifest_root(root) / MISSING_KLINES_FILENAME


def read_missing_klines(root: Path | None) -> list[MissingKlinePayload]:
    path = missing_klines_path(root)
    if not path.exists():
        return []
    records: list[MissingKlinePayload] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = cast("object", json.loads(line))
        if isinstance(payload, dict):
            parsed = _parse_missing_payload(cast("dict[str, object]", payload))
            if parsed is not None:
                records.append(parsed)
    return records


def record_missing_klines(root: Path | None, ranges: list[MissingKlineRange]) -> None:
    if not ranges:
        return
    path = missing_klines_path(root)
    with shared_file_lock(
        path,
        "record_missing_klines",
        options=LockOptions(
            wait_seconds=LOCK_WAIT_SECONDS,
            stale_seconds=LOCK_STALE_SECONDS,
            poll_seconds=LOCK_POLL_SECONDS,
        ),
    ):
        existing = {_payload_key(payload): payload for payload in read_missing_klines(root)}
        for missing_range in ranges:
            payload = _payload_from_range(missing_range)
            _ = existing.setdefault(_payload_key(payload), payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            ordered = sorted(existing.values(), key=_payload_key)
            with tmp_path.open("w", encoding="utf-8") as handle:
                for payload in ordered:
                    _ = handle.write(
                        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
                    )
            _ = tmp_path.replace(path)
        finally:
            tmp_path.unlink(missing_ok=True)


def append_missing_klines(root: Path | None, ranges: list[MissingKlineRange]) -> None:
    record_missing_klines(root, ranges)


def _payload_from_range(missing_range: MissingKlineRange) -> MissingKlinePayload:
    return {
        "dataset": "klines",
        "symbol": missing_range.symbol,
        "interval": missing_range.interval,
        "missing_start_ts": missing_range.missing_start_ts,
        "missing_end_ts": missing_range.missing_end_ts,
        "missing_count": missing_range.missing_count,
        "observed_before_ts": missing_range.observed_before_ts,
        "observed_after_ts": missing_range.observed_after_ts,
        "reason": missing_range.reason,
    }


def _payload_key(payload: MissingKlinePayload) -> tuple[str, str, int, int]:
    return (
        payload["symbol"],
        payload["interval"],
        payload["missing_start_ts"],
        payload["missing_end_ts"],
    )


def _parse_missing_payload(payload: dict[str, object]) -> MissingKlinePayload | None:
    values = (
        payload.get("dataset"),
        payload.get("symbol"),
        payload.get("interval"),
        payload.get("missing_start_ts"),
        payload.get("missing_end_ts"),
        payload.get("missing_count"),
        payload.get("observed_before_ts"),
        payload.get("observed_after_ts"),
        payload.get("reason"),
    )
    match values:
        case (
            str(dataset),
            str(symbol),
            str(interval),
            int(start),
            int(end),
            int(count),
            int(before),
            int(after),
            str(reason),
        ):
            return {
                "dataset": dataset,
                "symbol": symbol,
                "interval": interval,
                "missing_start_ts": start,
                "missing_end_ts": end,
                "missing_count": count,
                "observed_before_ts": before,
                "observed_after_ts": after,
                "reason": reason,
            }
        case _:
            return None
