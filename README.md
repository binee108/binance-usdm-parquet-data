# binance-usdm-parquet-data

Archive-first Binance USD-M market-data collector for USDT perpetual backtesting.

The collector discovers symbols from Binance public archive prefixes, not from the
current exchangeInfo response, so delisted USDT symbols remain eligible for
historical backtests. It writes deterministic raw Parquet, optimized
DuckDB-friendly Parquet layouts, and file-backed manifests for operator status.

Default production scope is USDT only.

## Shared data root

By default the library stores data under `~/Desktop/data`; set `MARKET_DATA_ROOT`
or pass an explicit `root: Path` to use another shared store. Applications should
read and refresh through this package instead of writing project-local Binance
data directories.

```python
from datetime import date
from pathlib import Path

from binance_usdm_parquet_data.http_client import Httpx2SyncClient
from binance_usdm_parquet_data.funding_rate import BinanceFundingRateClient
from binance_usdm_parquet_data.refresh import RefreshRequest, refresh_market_data

root = Path("~/Desktop/data").expanduser()
http = Httpx2SyncClient(timeout_seconds=30)

refresh_market_data(
    RefreshRequest(
        root=root,
        symbols=("BTCUSDT", "ETHUSDT"),
        start_day=date(2026, 6, 1),
        end_day=date(2026, 6, 16),
        datasets=("klines", "premiumIndexKlines", "fundingRate"),
        max_concurrent_downloads=4,
        http_timeout_seconds=30,
        funding_rest_sleep_seconds=0.2,
    ),
    archive_client=http,
    funding_client=BinanceFundingRateClient(http=http),
    premium_client=http,
)
```

## Public APIs

Use `binance_usdm_parquet_data.readers` for loaded data:

- `load_klines`
- `load_resampled_klines`
- `load_intrabar_buckets`
- `load_funding_rates`
- `load_premium_index_klines`

Use `binance_usdm_parquet_data.paths` for discovery of raw and optimized files,
`binance_usdm_parquet_data.manifest` for status/failures/sources, and
`binance_usdm_parquet_data.symbol_universe` for archive-discovered tradable
universe publication and reads.

Writes that can be shared by multiple projects use package-owned file locks:
refresh runs, manifest writes, optimized Parquet output, and missing-kline
quality records all claim stale locks atomically before replacing files.
