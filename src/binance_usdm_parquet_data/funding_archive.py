from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Final

import polars as pl

from binance_usdm_parquet_data.archive_download import (
    ArchiveDownloadFailure,
    ArchiveHttpClient,
    DownloadedArchiveFile,
)
from binance_usdm_parquet_data.storage_keys import symbol_storage_key

SHA256_HEX_LENGTH: Final = 64


@dataclass(frozen=True, slots=True)
class FundingRateArchiveRequest:
    symbol: str
    month: str
    base_url: str = "https://data.binance.vision"


def download_monthly_funding_rate_archive_to_parquet(
    client: ArchiveHttpClient,
    request: FundingRateArchiveRequest,
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
    frame = _archive_zip_to_funding_frame(archive_bytes)
    output_path = _monthly_parquet_path(root, request)
    _atomic_write_parquet(output_path, frame)
    return DownloadedArchiveFile(
        source_url=source_url,
        output_path=output_path,
        checksum=actual_checksum,
        row_count=frame.height,
    )


def _monthly_zip_url(request: FundingRateArchiveRequest) -> str:
    storage_key = symbol_storage_key(request.symbol)
    filename = f"{storage_key}-fundingRate-{request.month}.zip"
    return (
        f"{request.base_url.rstrip('/')}/data/futures/um/monthly/fundingRate/"
        f"{storage_key}/{filename}"
    )


def _monthly_parquet_path(root: Path, request: FundingRateArchiveRequest) -> Path:
    storage_key = symbol_storage_key(request.symbol)
    return (
        root
        / "binance"
        / "futures"
        / "fundingRate"
        / storage_key
        / f"{storage_key}_fundingRate_{request.month}.parquet"
    )


def _parse_checksum(text: str) -> str:
    token = text.strip().split(maxsplit=1)[0]
    if len(token) != SHA256_HEX_LENGTH:
        msg = f"invalid checksum payload: {text!r}"
        raise ValueError(msg)
    return token


def _archive_zip_to_funding_frame(archive_bytes: bytes) -> pl.DataFrame:
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        csv_names = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(csv_names) != 1:
            msg = f"expected exactly one CSV in archive, got {csv_names}"
            raise ValueError(msg)
        csv_bytes = archive.read(csv_names[0])
    frame = pl.read_csv(io.BytesIO(csv_bytes), has_header=True)
    return frame.select(
        pl.from_epoch(pl.col("calc_time").cast(pl.Int64), time_unit="ms")
        .dt.replace_time_zone("UTC")
        .cast(pl.Datetime(time_unit="ms", time_zone="UTC"))
        .alias("funding_time"),
        pl.col("last_funding_rate").cast(pl.Float64).alias("funding_rate"),
        pl.lit(None, dtype=pl.Float64).alias("mark_price"),
    )


def _atomic_write_parquet(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(suffix=".parquet", dir=path.parent, delete=False) as handle:
        temp_path = Path(handle.name)
    frame.write_parquet(temp_path)
    _ = temp_path.replace(path)
