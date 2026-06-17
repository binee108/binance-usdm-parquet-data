from __future__ import annotations

import os
from pathlib import Path

import pytest

from binance_usdm_parquet_data.paths import (
    find_optimized_klines,
    find_raw_archives,
    kline_files_for_read,
    market_data_root,
    optimized_klines_file,
    raw_klines_files,
    require_symbol,
)


def test_market_data_root_prefers_explicit_root_then_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_root = tmp_path / "env-root"
    monkeypatch.setenv("MARKET_DATA_ROOT", os.fspath(env_root))

    assert market_data_root(tmp_path / "explicit") == tmp_path / "explicit"
    assert market_data_root() == env_root


def test_path_helpers_encode_symbols_and_never_fallback_to_raw_for_runtime_reads(
    tmp_path: Path,
) -> None:
    encoded_symbol = require_symbol("1000왜USDT")
    raw_dir = tmp_path / "binance" / "futures" / "klines" / "1000왜USDT"
    raw_dir.mkdir(parents=True)
    expected_raw = raw_dir / "1000왜USDT_klines_1m_2026-06.parquet"
    expected_raw.touch()

    assert encoded_symbol == "1000%EC%99%9CUSDT"
    assert raw_klines_files(tmp_path, "1000왜USDT", "1m") == [expected_raw]
    assert optimized_klines_file(tmp_path, "/ABSUSDT", "1m").is_relative_to(tmp_path)
    assert kline_files_for_read(tmp_path, "1000왜USDT", "1m") == []


def test_find_optimized_klines_discovers_existing_symbol_interval_files(
    tmp_path: Path,
) -> None:
    btc = optimized_klines_file(tmp_path, "BTCUSDT", "1m")
    eth = optimized_klines_file(tmp_path, "ETHUSDT", "5m")
    btc.parent.mkdir(parents=True)
    eth.parent.mkdir(parents=True)
    btc.touch()
    eth.touch()

    discovered = find_optimized_klines(tmp_path, symbol="BTCUSDT")

    assert discovered == (btc,)


def test_find_market_data_rejects_traversal_interval(tmp_path: Path) -> None:
    optimized_root = tmp_path / "parbp_optimized"

    with pytest.raises(ValueError, match="Invalid market data interval"):
        _ = find_optimized_klines(tmp_path, interval="1m/../../outside")

    with pytest.raises(ValueError, match="Invalid market data interval"):
        _ = find_raw_archives(tmp_path, dataset="klines", interval="1m/../../outside")

    assert not optimized_root.exists()
    assert not (tmp_path / "outside").exists()


def test_find_raw_archives_discovers_supported_archive_datasets(tmp_path: Path) -> None:
    kline = (
        tmp_path
        / "binance"
        / "futures"
        / "klines"
        / "BTCUSDT"
        / "BTCUSDT_klines_1m_2026-06.parquet"
    )
    premium = (
        tmp_path
        / "binance"
        / "futures"
        / "premiumIndexKlines"
        / "BTCUSDT"
        / "BTCUSDT_premiumIndexKlines_1m_2026-06.parquet"
    )
    funding = (
        tmp_path
        / "binance"
        / "futures"
        / "fundingRate"
        / "BTCUSDT"
        / "BTCUSDT_fundingRate_2026-06-01.parquet"
    )
    for path in (kline, premium, funding):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    assert find_raw_archives(tmp_path, dataset="premiumIndexKlines", symbol="BTCUSDT") == (
        premium,
    )
    assert find_raw_archives(tmp_path, symbol="BTCUSDT") == (funding, kline, premium)
