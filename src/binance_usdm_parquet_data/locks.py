from __future__ import annotations

import json
import os
import socket
import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

DEFAULT_LOCK_WAIT_SECONDS: Final = 30.0
DEFAULT_LOCK_STALE_SECONDS: Final = 300.0
DEFAULT_LOCK_POLL_SECONDS: Final = 0.1

type LockMetadata = dict[str, str | int | float]


@dataclass(frozen=True, slots=True)
class LockOptions:
    wait_seconds: float = DEFAULT_LOCK_WAIT_SECONDS
    stale_seconds: float = DEFAULT_LOCK_STALE_SECONDS
    poll_seconds: float = DEFAULT_LOCK_POLL_SECONDS


class SharedFileLockTimeoutError(RuntimeError):
    pass


class SharedFileStaleLockError(RuntimeError):
    pass


@contextmanager
def shared_file_lock(
    target_path: Path,
    operation: str,
    *,
    lock_path: Path | None = None,
    options: LockOptions | None = None,
) -> Generator[None]:
    lock_options = options or LockOptions()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_lock_path = lock_path or default_lock_path(target_path)
    resolved_lock_path.parent.mkdir(parents=True, exist_ok=True)
    owner_token = uuid.uuid4().hex
    metadata = lock_metadata(target_path, operation, owner_token)
    deadline = time.monotonic() + lock_options.wait_seconds
    while True:
        try:
            fd = os.open(str(resolved_lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            existing = read_lock_metadata(resolved_lock_path)
            stale_mtime_ns = _mtime_ns(resolved_lock_path) if existing is None else None
            if lock_is_stale(
                resolved_lock_path,
                existing,
                stale_seconds=lock_options.stale_seconds,
            ):
                if existing is not None and not lock_targets_path(existing, target_path):
                    msg = f"stale lock targets a different file: {resolved_lock_path}"
                    raise SharedFileStaleLockError(msg) from None
                if claim_stale_lock(
                    resolved_lock_path,
                    expected=existing,
                    stale_mtime_ns=stale_mtime_ns,
                ):
                    continue
            if time.monotonic() >= deadline:
                msg = f"timed out waiting for lock: {resolved_lock_path}"
                raise SharedFileLockTimeoutError(msg) from None
            time.sleep(lock_options.poll_seconds)
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, sort_keys=True)
            _ = handle.write("\n")
        break
    try:
        yield
    finally:
        release_lock(resolved_lock_path, expected=metadata)


def default_lock_path(target_path: Path) -> Path:
    return target_path.with_name(f".{target_path.name}.lock")


def lock_metadata(target_path: Path, operation: str, owner_token: str) -> LockMetadata:
    return {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "created_at_epoch": time.time(),
        "target_path": str(target_path.resolve()),
        "operation": operation,
        "owner_token": owner_token,
    }


def release_lock(lock_path: Path, *, expected: LockMetadata) -> None:
    tombstone = _claim_lock(lock_path, expected=expected, stale_mtime_ns=None)
    if tombstone is not None:
        tombstone.unlink(missing_ok=True)


def claim_stale_lock(
    lock_path: Path,
    *,
    expected: LockMetadata | None,
    stale_mtime_ns: int | None,
) -> bool:
    tombstone = _claim_lock(lock_path, expected=expected, stale_mtime_ns=stale_mtime_ns)
    if tombstone is None:
        return False
    tombstone.unlink(missing_ok=True)
    return True


def read_lock_metadata(path: Path) -> LockMetadata | None:
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


def lock_is_stale(
    path: Path,
    metadata: LockMetadata | None,
    *,
    stale_seconds: float,
) -> bool:
    created_at = metadata.get("created_at_epoch") if metadata else None
    if isinstance(created_at, int | float):
        return time.time() - float(created_at) >= stale_seconds
    try:
        return time.time() - path.stat().st_mtime >= stale_seconds
    except OSError:
        return False


def lock_targets_path(metadata: LockMetadata, path: Path) -> bool:
    target = metadata.get("target_path")
    return isinstance(target, str) and target == str(path.resolve())


def _claim_lock(
    lock_path: Path,
    *,
    expected: LockMetadata | None,
    stale_mtime_ns: int | None,
) -> Path | None:
    tombstone = lock_path.with_name(f".{lock_path.name}.{os.getpid()}.{uuid.uuid4().hex}.stale")
    try:
        _ = lock_path.replace(tombstone)
    except FileNotFoundError:
        return None
    except OSError:
        return None
    if _claimed_lock_matches(tombstone, expected=expected, stale_mtime_ns=stale_mtime_ns):
        return tombstone
    try:
        _ = tombstone.replace(lock_path)
    except OSError:
        tombstone.unlink(missing_ok=True)
    return None


def _claimed_lock_matches(
    tombstone: Path,
    *,
    expected: LockMetadata | None,
    stale_mtime_ns: int | None,
) -> bool:
    if expected is not None:
        return read_lock_metadata(tombstone) == expected
    if stale_mtime_ns is None:
        return False
    try:
        return tombstone.stat().st_mtime_ns == stale_mtime_ns
    except OSError:
        return False


def _mtime_ns(path: Path) -> int | None:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None
