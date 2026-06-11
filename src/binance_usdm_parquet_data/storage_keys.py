from __future__ import annotations

from urllib.parse import quote


def symbol_storage_key(symbol: str) -> str:
    key = encode_path_segment(symbol)
    if not key or key in {".", ".."} or "/" in key:
        msg = f"unsafe market-data storage key: {symbol!r}"
        raise ValueError(msg)
    return key


def encode_path_segment(value: str) -> str:
    return quote(value, safe="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_=")
