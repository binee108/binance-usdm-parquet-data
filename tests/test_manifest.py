from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from binance_usdm_parquet_data.manifest import (
    CollectorFailure,
    CollectorRun,
    DatasetFreshness,
    ManifestStore,
)


def test_manifest_store_publishes_json_status_and_failures(tmp_path: Path) -> None:
    store = ManifestStore(tmp_path)
    run = CollectorRun(
        run_id="run-1",
        mode="manual",
        status="failed",
        started_at="2026-06-12T00:00:00+00:00",
        finished_at="2026-06-12T00:01:00+00:00",
        symbol_count=2,
        item_count=3,
        success_count=2,
        failure_count=1,
        last_error="checksum mismatch",
    )
    failure = CollectorFailure(
        dataset="klines",
        symbol="BTCUSDT",
        interval="1m",
        target_date="2026-06-11",
        source_url="https://data.binance.vision/example.parquet",
        attempt_count=2,
        error_code="checksum_mismatch",
        error_message="expected abc got def",
        retryable=True,
    )

    store.publish_status(
        last_run=run,
        freshness=(
            DatasetFreshness(
                dataset="klines",
                interval="1m",
                symbol_count=2,
                latest_complete_utc_day="2026-06-11",
            ),
        ),
        failures=(failure,),
    )

    status_path = tmp_path / "manifests" / "binance" / "usdm" / "status.json"
    status = cast("dict[str, object]", json.loads(status_path.read_text(encoding="utf-8")))
    last_run = cast("dict[str, object]", status["last_run"])
    freshness = cast("list[dict[str, object]]", status["freshness"])
    failures = cast("list[dict[str, object]]", status["failures"])
    assert last_run["id"] == "run-1"
    assert status["failed_item_count"] == 1
    assert freshness[0]["dataset"] == "klines"
    assert failures[0]["error_code"] == "checksum_mismatch"

    jsonl_path = tmp_path / "manifests" / "binance" / "usdm" / "failures.jsonl"
    jsonl_payload = cast(
        "dict[str, object]",
        json.loads(jsonl_path.read_text(encoding="utf-8").strip()),
    )
    assert jsonl_payload["retryable"] is True
