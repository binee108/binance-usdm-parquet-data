from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from binance_usdm_parquet_data.archive_index import S3ArchiveIndexClient
from binance_usdm_parquet_data.config import CollectorConfig
from binance_usdm_parquet_data.duckdb_optimize import optimize_klines
from binance_usdm_parquet_data.funding_rate import BinanceFundingRateClient
from binance_usdm_parquet_data.http_client import Httpx2SyncClient
from binance_usdm_parquet_data.liquidity import fetch_current_quote_liquidity
from binance_usdm_parquet_data.manifest import JsonObject, JsonValue, read_status
from binance_usdm_parquet_data.refresh import RefreshRequest, refresh_market_data
from binance_usdm_parquet_data.storage_keys import symbol_storage_key
from binance_usdm_parquet_data.symbol_universe import (
    SymbolMetadata,
    SymbolUniverse,
    build_symbol_universe,
)

app = typer.Typer(no_args_is_help=True)
DEFAULT_ROOT = CollectorConfig().root


@app.command()
def discover(
    root: Annotated[Path, typer.Option("--root", help="Storage root")] = DEFAULT_ROOT,
) -> None:
    universe = _discover_usdt_universe()
    _print_json(
        {
            "root": str(root),
            "quote_assets": ["USDT"],
            "symbol_count": len(universe.symbols),
            "symbols": [_symbol_metadata_json(symbol) for symbol in universe.symbols],
        }
    )


@app.command()
def liquidity() -> None:
    http = Httpx2SyncClient()
    result = fetch_current_quote_liquidity(http)
    _print_json(
        {
            "quote_assets": [
                {
                    "quote_asset": item.quote_asset,
                    "symbol_count": item.symbol_count,
                    "aggregate_quote_volume": item.aggregate_quote_volume,
                }
                for item in result
            ]
        }
    )


@app.command()
def plan(
    root: Annotated[Path, typer.Option("--root", help="Storage root")] = DEFAULT_ROOT,
) -> None:
    local_symbols = _local_raw_symbols(root)
    _print_json(
        {
            "root": str(root),
            "default_quote_assets": ["USDT"],
            "local_symbol_count": len(local_symbols),
            "status": read_status(root)["status"],
        }
    )


@app.command()
def refresh(
    root: Annotated[Path, typer.Option("--root", help="Storage root")] = DEFAULT_ROOT,
    symbols: Annotated[str, typer.Option("--symbols", help="Comma-separated symbols")] = "BTCUSDT",
    start_day: Annotated[str, typer.Option("--start-day", help="UTC start day")] = "",
    end_day: Annotated[str, typer.Option("--end-day", help="UTC end day")] = "",
    datasets: Annotated[str, typer.Option("--datasets", help="Comma-separated datasets")] = (
        "klines,premiumIndexKlines,fundingRate"
    ),
    interval: Annotated[str, typer.Option("--interval", help="Archive interval")] = "1m",
    archive_granularity: Annotated[
        str,
        typer.Option("--archive-granularity", help="daily or monthly archive files"),
    ] = "daily",
    optimize_output: Annotated[bool, typer.Option("--optimize/--no-optimize")] = True,
) -> None:
    today = datetime.now(UTC).date()
    target_day = today if not end_day else datetime.fromisoformat(end_day).date()
    requested_symbols = _parse_symbols(symbols)
    request = RefreshRequest(
        root=root,
        symbols=requested_symbols,
        start_day=target_day if not start_day else datetime.fromisoformat(start_day).date(),
        end_day=target_day,
        datasets=tuple(dataset.strip() for dataset in datasets.split(",") if dataset.strip()),
        interval=interval,
        optimize=optimize_output,
        archive_granularity=archive_granularity,
    )
    http = Httpx2SyncClient()
    result = refresh_market_data(
        request,
        archive_client=http,
        funding_client=BinanceFundingRateClient(http=http),
    )
    _print_json(
        {
            "run_id": result.run_id,
            "status": result.status,
            "success_count": result.success_count,
            "failure_count": result.failure_count,
        }
    )


@app.command()
def optimize(
    root: Annotated[Path, typer.Option("--root", help="Storage root")] = DEFAULT_ROOT,
    symbols: Annotated[
        str,
        typer.Option("--symbols", help="Comma-separated symbols or all-local"),
    ] = "all-local",
    interval: Annotated[str, typer.Option("--interval", help="Archive interval")] = "1m",
) -> None:
    optimized: list[str] = []
    for symbol in _parse_local_symbols(root, symbols):
        raw_files = _raw_kline_files(root, symbol, interval)
        if not raw_files:
            continue
        output = optimize_klines(
            raw_files=tuple(raw_files),
            output_root=root / "parbp_optimized" / "binance" / "futures",
            symbol=symbol,
            interval=interval,
        )
        optimized.append(str(output))
    outputs: list[JsonValue] = list(optimized)
    payload: JsonObject = {
        "root": str(root),
        "interval": interval,
        "optimized_count": len(optimized),
        "outputs": outputs,
    }
    _print_json(payload)


@app.command()
def status(
    root: Annotated[Path, typer.Option("--root", help="Storage root")] = DEFAULT_ROOT,
) -> None:
    _print_json(read_status(root))


@app.command()
def validate(
    root: Annotated[Path, typer.Option("--root", help="Storage root")] = DEFAULT_ROOT,
) -> None:
    payload = read_status(root)
    valid = payload["status"] in {"ok", "degraded"} and payload["last_run"] is not None
    _print_json(
        {
            "root": str(root),
            "valid": valid,
            "status": payload["status"],
            "failed_item_count": payload["failed_item_count"],
        }
    )
    if not valid:
        raise typer.Exit(1)


def main() -> None:
    app()


def _parse_symbols(value: str) -> tuple[str, ...]:
    if value.strip().lower() == "all":
        return _discover_usdt_symbols()
    return tuple(symbol.strip().upper() for symbol in value.split(",") if symbol.strip())


def _parse_local_symbols(root: Path, value: str) -> tuple[str, ...]:
    if value.strip().lower() == "all-local":
        return _local_raw_symbols(root)
    return _parse_symbols(value)


def _local_raw_symbols(root: Path) -> tuple[str, ...]:
    raw_root = root / "binance" / "futures" / "klines"
    if not raw_root.exists():
        return ()
    return tuple(sorted(path.name for path in raw_root.iterdir() if path.is_dir()))


def _raw_kline_files(root: Path, symbol: str, interval: str) -> tuple[Path, ...]:
    storage_key = symbol_storage_key(symbol)
    directory = root / "binance" / "futures" / "klines" / storage_key
    return tuple(sorted(directory.glob(f"{storage_key}_klines_{interval}_*.parquet")))


def _discover_usdt_symbols() -> tuple[str, ...]:
    universe = _discover_usdt_universe()
    return tuple(symbol.symbol for symbol in universe.symbols)


def _discover_usdt_universe() -> SymbolUniverse:
    http = Httpx2SyncClient()
    return build_symbol_universe(S3ArchiveIndexClient(http), quote_assets=("USDT",))


def _symbol_metadata_json(symbol: SymbolMetadata) -> JsonObject:
    return {
        "symbol": symbol.symbol,
        "quote_asset": symbol.quote_asset,
        "storage_key": symbol.storage_key,
        "datasets": {dataset.value: available for dataset, available in symbol.datasets.items()},
    }


def _print_json(payload: JsonValue) -> None:
    typer.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))
