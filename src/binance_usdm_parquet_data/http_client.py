from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from typing import cast

import httpx2

from binance_usdm_parquet_data.funding_rate import JsonValue

_LIMITS = httpx2.Limits(max_connections=200, max_keepalive_connections=40, keepalive_expiry=30.0)
_TIMEOUT = httpx2.Timeout(connect=5.0, read=30.0, write=10.0, pool=10.0)
_SOCKET_OPTIONS: list[tuple[int, int, int]] = [(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)]


@dataclass(frozen=True, slots=True)
class Httpx2SyncClient:
    base_url: str = ""

    def get_text(self, url: str, params: dict[str, str] | None = None) -> str:
        with create_client(base_url=self.base_url) as client:
            response = client.get(url, params=params or {})
            _ = response.raise_for_status()
            return response.text

    def get_bytes(self, url: str) -> bytes:
        with create_client(base_url=self.base_url) as client:
            response = client.get(url)
            _ = response.raise_for_status()
            return response.content

    def get_json(self, url: str, params: dict[str, str]) -> JsonValue:
        with create_client(base_url=self.base_url) as client:
            response = client.get(url, params=params)
            _ = response.raise_for_status()
            return cast("JsonValue", json.loads(response.text))


def create_client(*, base_url: str = "") -> httpx2.Client:
    transport = httpx2.HTTPTransport(
        http2=True,
        retries=3,
        limits=_LIMITS,
        socket_options=_SOCKET_OPTIONS,
    )
    return httpx2.Client(
        transport=transport,
        timeout=_TIMEOUT,
        base_url=base_url,
        follow_redirects=True,
        headers={"accept-encoding": "br, zstd, gzip"},
    )
