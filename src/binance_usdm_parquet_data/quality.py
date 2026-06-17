from __future__ import annotations

import json
import os
import socket
import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Final, TypedDict, cast

from binance_usdm_parquet_data.paths import manifest_root
from binance_usdm_parquet_data.records import MissingKlineRange

MISSING_KLINES_FILENAME: Final = "missing_klines.jsonl"
LOCK_WAIT_SECONDS: Final = 30.0
LOCK_STALE_SECONDS: Final = 300.0
LOCK_POLL_SECONDS: Final = 0.1


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


class MissingKlinesLockTimeoutError(RuntimeError):
    pass


class MissingKlinesStaleLockError(RuntimeError):
    pass


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
    with _missing_klines_lock(path):
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


def _missing_lock_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.lock")


@contextmanager
def _missing_klines_lock(path: Path) -> Generator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _missing_lock_path(path)
    owner_token = uuid.uuid4().hex
    deadline = time.monotonic() + LOCK_WAIT_SECONDS
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            metadata = _read_lock_metadata(lock_path)
            if _lock_is_stale(lock_path, metadata):
                if metadata is not None and not _lock_targets_path(metadata, path):
                    msg = f"stale missing klines lock targets a different file: {lock_path}"
                    raise MissingKlinesStaleLockError(msg) from None
                lock_path.unlink(missing_ok=True)
                continue
            if time.monotonic() >= deadline:
                msg = f"timed out waiting for lock: {lock_path}"
                raise MissingKlinesLockTimeoutError(msg) from None
            time.sleep(LOCK_POLL_SECONDS)
            continue
        else:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(_lock_metadata(path, owner_token), handle, sort_keys=True)
                _ = handle.write("\n")
            break
    try:
        yield
    finally:
        metadata = _read_lock_metadata(lock_path)
        if metadata is None or metadata.get("owner_token") == owner_token:
            lock_path.unlink(missing_ok=True)


def _lock_metadata(path: Path, owner_token: str) -> dict[str, str | int | float]:
    return {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "created_at_epoch": time.time(),
        "target_path": str(path.resolve()),
        "operation": "record_missing_klines",
        "owner_token": owner_token,
    }


def _read_lock_metadata(path: Path) -> dict[str, str | int | float] | None:
    try:
        payload = cast("object", json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return {
        str(key): value
        for key, value in cast("dict[object, object]", payload).items()
        if isinstance(value, str | int | float)
    }


def _lock_is_stale(path: Path, metadata: dict[str, str | int | float] | None) -> bool:
    created_at = metadata.get("created_at_epoch") if metadata else None
    if isinstance(created_at, int | float):
        return time.time() - float(created_at) >= LOCK_STALE_SECONDS
    try:
        return time.time() - path.stat().st_mtime >= LOCK_STALE_SECONDS
    except OSError:
        return False


def _lock_targets_path(metadata: dict[str, str | int | float], path: Path) -> bool:
    target = metadata.get("target_path")
    return isinstance(target, str) and target == str(path.resolve())


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
