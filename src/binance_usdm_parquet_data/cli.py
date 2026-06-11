from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from binance_usdm_parquet_data.archive_index import S3ArchiveIndexClient
from binance_usdm_parquet_data.config import CollectorConfig
from binance_usdm_parquet_data.funding_rate import BinanceFundingRateClient
from binance_usdm_parquet_data.http_client import Httpx2SyncClient
from binance_usdm_parquet_data.liquidity import fetch_current_quote_liquidity
from binance_usdm_parquet_data.manifest import JsonObject, JsonValue, read_status
from binance_usdm_parquet_data.refresh import RefreshRequest, refresh_market_data
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
    _print_json({"root": str(root), "default_quote_assets": ["USDT"]})


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
) -> None:
    _print_json({"root": str(root), "status": "not_configured"})


@app.command()
def status(
    root: Annotated[Path, typer.Option("--root", help="Storage root")] = DEFAULT_ROOT,
) -> None:
    _print_json(read_status(root))


@app.command()
def validate(
    root: Annotated[Path, typer.Option("--root", help="Storage root")] = DEFAULT_ROOT,
) -> None:
    _print_json({"root": str(root), "valid": True})


def main() -> None:
    app()


def _parse_symbols(value: str) -> tuple[str, ...]:
    if value.strip().lower() == "all":
        return _discover_usdt_symbols()
    return tuple(symbol.strip().upper() for symbol in value.split(",") if symbol.strip())


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
