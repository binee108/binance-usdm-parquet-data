from __future__ import annotations

import ast
from pathlib import Path

from typer.testing import CliRunner

from binance_usdm_parquet_data.cli import app
from binance_usdm_parquet_data.config import CollectorConfig


def test_default_config_collects_usdt_only(tmp_path: Path) -> None:
    config = CollectorConfig(root=tmp_path)

    assert config.quote_assets == ("USDT",)
    assert config.raw_futures_root == tmp_path / "binance" / "futures"
    assert config.optimized_futures_root == tmp_path / "parbp_optimized" / "binance" / "futures"


def test_cli_help_exposes_required_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ("discover", "liquidity", "plan", "refresh", "optimize", "status", "validate"):
        assert command in result.output


def test_package_does_not_import_parbp() -> None:
    package_root = Path(__file__).resolve().parents[1] / "src" / "binance_usdm_parquet_data"
    imported_roots: set[str] = set()
    for path in package_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", maxsplit=1)[0])

    assert {"src", "apps", "autoresearch_platform"}.isdisjoint(imported_roots)
