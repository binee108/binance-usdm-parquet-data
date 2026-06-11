# binance-usdm-parquet-data

Archive-first Binance USD-M market-data collector for USDT perpetual backtesting.

The collector discovers symbols from Binance public archive prefixes, not from the
current exchangeInfo response, so delisted USDT symbols remain eligible for
historical backtests. It writes deterministic raw Parquet, optimized
DuckDB-friendly Parquet layouts, and file-backed manifests for operator status.

Default production scope is USDT only.
