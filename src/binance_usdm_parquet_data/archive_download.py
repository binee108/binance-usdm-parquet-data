from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol

import polars as pl

from binance_usdm_parquet_data.parquet_storage import normalize_kline_frame
from binance_usdm_parquet_data.paths import require_interval, require_symbol

SHA256_HEX_LENGTH = 64


class ArchiveHttpClient(Protocol):
    def get_bytes(self, url: str) -> bytes: ...

    def get_text(self, url: str) -> str: ...


@dataclass(frozen=True, slots=True)
class DailyArchiveRequest:
    dataset: str
    symbol: str
    interval: str
    day: date
    base_url: str = "https://data.binance.vision"


@dataclass(frozen=True, slots=True)
class MonthlyArchiveRequest:
    dataset: str
    symbol: str
    interval: str
    month: str
    base_url: str = "https://data.binance.vision"


@dataclass(frozen=True, slots=True)
class ArchiveDownloadFailure:
    source_url: str
    error_code: str
    error_message: str
    retryable: bool


@dataclass(frozen=True, slots=True)
class DownloadedArchiveFile:
    source_url: str
    output_path: Path
    checksum: str
    row_count: int


def download_daily_archive_to_parquet(
    client: ArchiveHttpClient,
    request: DailyArchiveRequest,
    root: Path,
) -> DownloadedArchiveFile | ArchiveDownloadFailure:
    source_url = _daily_zip_url(request)
    checksum_text = client.get_text(f"{source_url}.CHECKSUM")
    archive_bytes = client.get_bytes(source_url)
    expected_checksum = _parse_checksum(checksum_text)
    actual_checksum = hashlib.sha256(archive_bytes).hexdigest()
    if actual_checksum != expected_checksum:
        return ArchiveDownloadFailure(
            source_url=source_url,
            error_code="checksum_mismatch",
            error_message=f"expected {expected_checksum} got {actual_checksum}",
            retryable=True,
        )
    frame = _archive_zip_to_kline_frame(archive_bytes)
    output_path = _daily_parquet_path(root, request)
    _atomic_write_parquet(output_path, frame)
    return DownloadedArchiveFile(
        source_url=source_url,
        output_path=output_path,
        checksum=actual_checksum,
        row_count=frame.height,
    )


def download_monthly_archive_to_parquet(
    client: ArchiveHttpClient,
    request: MonthlyArchiveRequest,
    root: Path,
) -> DownloadedArchiveFile | ArchiveDownloadFailure:
    source_url = _monthly_zip_url(request)
    checksum_text = client.get_text(f"{source_url}.CHECKSUM")
    archive_bytes = client.get_bytes(source_url)
    expected_checksum = _parse_checksum(checksum_text)
    actual_checksum = hashlib.sha256(archive_bytes).hexdigest()
    if actual_checksum != expected_checksum:
        return ArchiveDownloadFailure(
            source_url=source_url,
            error_code="checksum_mismatch",
            error_message=f"expected {expected_checksum} got {actual_checksum}",
            retryable=True,
        )
    frame = _archive_zip_to_kline_frame(archive_bytes)
    output_path = _monthly_parquet_path(root, request)
    _atomic_write_parquet(output_path, frame)
    return DownloadedArchiveFile(
        source_url=source_url,
        output_path=output_path,
        checksum=actual_checksum,
        row_count=frame.height,
    )


def _daily_zip_url(request: DailyArchiveRequest) -> str:
    dataset_path = _dataset_archive_path(request.dataset)
    remote_symbol = require_symbol(request.symbol)
    interval = require_interval(request.interval)
    filename = f"{remote_symbol}-{interval}-{request.day.isoformat()}.zip"
    return (
        f"{request.base_url.rstrip('/')}/data/futures/um/daily/{dataset_path}/"
        f"{remote_symbol}/{interval}/{filename}"
    )


def _monthly_zip_url(request: MonthlyArchiveRequest) -> str:
    dataset_path = _dataset_archive_path(request.dataset)
    remote_symbol = require_symbol(request.symbol)
    interval = require_interval(request.interval)
    filename = f"{remote_symbol}-{interval}-{request.month}.zip"
    return (
        f"{request.base_url.rstrip('/')}/data/futures/um/monthly/{dataset_path}/"
        f"{remote_symbol}/{interval}/{filename}"
    )


def _dataset_archive_path(dataset: str) -> str:
    match dataset:
        case "klines":
            return "klines"
        case "premium_index_klines" | "premiumIndexKlines":
            return "premiumIndexKlines"
        case _:
            msg = f"unsupported archive dataset: {dataset}"
            raise ValueError(msg)


def _daily_parquet_path(root: Path, request: DailyArchiveRequest) -> Path:
    dataset_dir = "klines" if request.dataset == "klines" else "premiumIndexKlines"
    dataset_name = "klines" if request.dataset == "klines" else "premiumIndexKlines"
    storage_key = require_symbol(request.symbol)
    interval = require_interval(request.interval)
    filename = (
        f"{storage_key}_{dataset_name}_{interval}_{request.day.isoformat()}.parquet"
    )
    return root / "binance" / "futures" / dataset_dir / storage_key / filename


def _monthly_parquet_path(root: Path, request: MonthlyArchiveRequest) -> Path:
    dataset_dir = "klines" if request.dataset == "klines" else "premiumIndexKlines"
    dataset_name = "klines" if request.dataset == "klines" else "premiumIndexKlines"
    storage_key = require_symbol(request.symbol)
    interval = require_interval(request.interval)
    filename = f"{storage_key}_{dataset_name}_{interval}_{request.month}.parquet"
    return root / "binance" / "futures" / dataset_dir / storage_key / filename


def _parse_checksum(text: str) -> str:
    token = text.strip().split(maxsplit=1)[0]
    if len(token) != SHA256_HEX_LENGTH:
        msg = f"invalid checksum payload: {text!r}"
        raise ValueError(msg)
    return token


def _archive_zip_to_kline_frame(archive_bytes: bytes) -> pl.DataFrame:
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        csv_names = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(csv_names) != 1:
            msg = f"expected exactly one CSV in archive, got {csv_names}"
            raise ValueError(msg)
        csv_bytes = archive.read(csv_names[0])
    if csv_bytes.startswith(b"open_time,"):
        frame = pl.read_csv(io.BytesIO(csv_bytes), has_header=True).rename({"count": "trade_count"})
    else:
        frame = pl.read_csv(
            io.BytesIO(csv_bytes),
            has_header=False,
            new_columns=(
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trade_count",
            "taker_buy_volume",
            "taker_buy_quote_volume",
            "ignore",
            ),
        )
    return normalize_kline_frame(frame)


def _atomic_write_parquet(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(suffix=".parquet", dir=path.parent, delete=False) as handle:
        temp_path = Path(handle.name)
    frame.write_parquet(temp_path)
    _ = temp_path.replace(path)
