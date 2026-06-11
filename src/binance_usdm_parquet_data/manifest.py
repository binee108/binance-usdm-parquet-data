from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import cast

type JsonValue = str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class DatasetFreshness:
    dataset: str
    interval: str | None
    symbol_count: int
    latest_complete_utc_day: str | None


@dataclass(frozen=True, slots=True)
class CollectorFailure:
    dataset: str
    symbol: str
    interval: str | None
    target_date: str
    source_url: str
    attempt_count: int
    error_code: str
    error_message: str
    retryable: bool


@dataclass(frozen=True, slots=True)
class CollectorRun:
    run_id: str
    mode: str
    status: str
    started_at: str
    finished_at: str | None
    symbol_count: int
    item_count: int
    success_count: int
    failure_count: int
    last_error: str | None


@dataclass(frozen=True, slots=True)
class CollectorSource:
    dataset: str
    symbol: str
    interval: str | None
    target_date: str
    source_url: str
    output_path: str
    checksum: str | None
    row_count: int


@dataclass(frozen=True, slots=True)
class ManifestStore:
    root: Path

    @property
    def manifest_root(self) -> Path:
        return self.root / "manifests" / "binance" / "usdm"

    def publish_status(
        self,
        *,
        last_run: CollectorRun | None,
        freshness: tuple[DatasetFreshness, ...],
        failures: tuple[CollectorFailure, ...],
        sources: tuple[CollectorSource, ...] = (),
    ) -> None:
        self.manifest_root.mkdir(parents=True, exist_ok=True)
        payload: JsonObject = {
            "status": "ok" if not failures else "degraded",
            "last_run": None if last_run is None else _run_dict(last_run),
            "failed_item_count": len(failures),
            "source_count": len(sources),
            "freshness": [_freshness_dict(item) for item in freshness],
            "failures": [_failure_dict(item) for item in failures],
        }
        _atomic_write_json(self.manifest_root / "status.json", payload)
        failure_lines = "".join(
            json.dumps(_failure_dict(item), sort_keys=True) + "\n" for item in failures
        )
        _atomic_write_text(self.manifest_root / "failures.jsonl", failure_lines)
        source_lines = "".join(
            json.dumps(_source_dict(item), sort_keys=True) + "\n" for item in sources
        )
        _atomic_write_text(self.manifest_root / "sources.jsonl", source_lines)


def read_status(root: Path) -> JsonObject:
    path = root / "manifests" / "binance" / "usdm" / "status.json"
    if not path.exists():
        return {
            "status": "missing",
            "last_run": None,
            "failed_item_count": 0,
            "source_count": 0,
            "freshness": [],
            "failures": [],
        }
    raw = cast("JsonValue", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(raw, dict):
        msg = f"status manifest is not an object: {path}"
        raise TypeError(msg)
    return cast("JsonObject", raw)


def _run_dict(run: CollectorRun) -> JsonObject:
    return {
        "id": run.run_id,
        "mode": run.mode,
        "status": run.status,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "symbol_count": run.symbol_count,
        "item_count": run.item_count,
        "success_count": run.success_count,
        "failure_count": run.failure_count,
        "last_error": run.last_error,
    }


def _freshness_dict(item: DatasetFreshness) -> JsonObject:
    return {
        "dataset": item.dataset,
        "interval": item.interval,
        "symbol_count": item.symbol_count,
        "latest_complete_utc_day": item.latest_complete_utc_day,
    }


def _failure_dict(item: CollectorFailure) -> JsonObject:
    return {
        "dataset": item.dataset,
        "symbol": item.symbol,
        "interval": item.interval,
        "target_date": item.target_date,
        "source_url": item.source_url,
        "attempt_count": item.attempt_count,
        "error_code": item.error_code,
        "error_message": item.error_message,
        "retryable": item.retryable,
    }


def _source_dict(item: CollectorSource) -> JsonObject:
    return {
        "dataset": item.dataset,
        "symbol": item.symbol,
        "interval": item.interval,
        "target_date": item.target_date,
        "source_url": item.source_url,
        "output_path": item.output_path,
        "checksum": item.checksum,
        "row_count": item.row_count,
    }


def _atomic_write_json(path: Path, payload: JsonObject) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        _ = handle.write(text)
        temp_path = Path(handle.name)
    _ = temp_path.replace(path)
