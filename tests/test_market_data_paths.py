from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

from binance_usdm_parquet_data.paths import (
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
