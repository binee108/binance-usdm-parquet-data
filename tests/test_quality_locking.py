from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from binance_usdm_parquet_data.quality import (
    MissingKlinesLockTimeoutError,
    missing_klines_path,
    record_missing_klines,
)
from binance_usdm_parquet_data.records import MissingKlineRange


def test_missing_klines_stale_lock_claim_does_not_delete_competing_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = missing_klines_path(tmp_path)
    lock_path = path.with_name(f".{path.name}.lock")
    lock_path.parent.mkdir(parents=True)
    stale_metadata = _lock_metadata(path, "stale-owner", time.time() - 1_000)
    competing_metadata = _lock_metadata(path, "competing-owner", time.time())
    _ = lock_path.write_text(json.dumps(stale_metadata), encoding="utf-8")
    real_replace = Path.replace

    def replace_then_compete(source: Path, target: Path | str) -> Path:
        result = real_replace(source, target)
        if source == lock_path:
            _ = lock_path.write_text(json.dumps(competing_metadata), encoding="utf-8")
        return result

    monkeypatch.setattr("binance_usdm_parquet_data.quality.LOCK_STALE_SECONDS", 0.01)
    monkeypatch.setattr("binance_usdm_parquet_data.quality.LOCK_WAIT_SECONDS", 0.0)
    monkeypatch.setattr("pathlib.Path.replace", replace_then_compete)

    with pytest.raises(MissingKlinesLockTimeoutError):
        record_missing_klines(tmp_path, [_missing_range()])

    assert json.loads(lock_path.read_text(encoding="utf-8")) == competing_metadata
    assert not path.exists()


def _lock_metadata(path: Path, owner_token: str, created_at: float) -> dict[str, str | int | float]:
    return {
        "pid": 999999,
        "hostname": "test",
        "created_at_epoch": created_at,
        "target_path": str(path.resolve()),
        "operation": "test",
        "owner_token": owner_token,
    }


def _missing_range() -> MissingKlineRange:
    return MissingKlineRange(
        symbol="BTCUSDT",
        interval="1m",
        missing_start_ts=1_700_000_000,
        missing_end_ts=1_700_000_060,
        missing_count=1,
        observed_before_ts=1_699_999_940,
        observed_after_ts=1_700_000_120,
    )
