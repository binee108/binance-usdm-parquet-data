from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from typing import cast

import httpx2

from binance_usdm_parquet_data.funding_rate import JsonValue

_LIMITS = httpx2.Limits(max_connections=200, max_keepalive_connections=40, keepalive_expiry=30.0)
_CONNECT_TIMEOUT_SECONDS = 5.0
_READ_TIMEOUT_SECONDS = 30.0
_WRITE_TIMEOUT_SECONDS = 10.0
_POOL_TIMEOUT_SECONDS = 10.0
_TIMEOUT = httpx2.Timeout(
    connect=_CONNECT_TIMEOUT_SECONDS,
    read=_READ_TIMEOUT_SECONDS,
    write=_WRITE_TIMEOUT_SECONDS,
    pool=_POOL_TIMEOUT_SECONDS,
)
_SOCKET_OPTIONS: list[tuple[int, int, int]] = [(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)]


@dataclass(frozen=True, slots=True)
class Httpx2SyncClient:
    base_url: str = ""
    timeout_seconds: float | None = None

    def get_text(self, url: str, params: dict[str, str] | None = None) -> str:
        with create_client(base_url=self.base_url, timeout_seconds=self.timeout_seconds) as client:
            response = client.get(url, params=params or {})
            _ = response.raise_for_status()
            return response.text

    def get_bytes(self, url: str) -> bytes:
        with create_client(base_url=self.base_url, timeout_seconds=self.timeout_seconds) as client:
            response = client.get(url)
            _ = response.raise_for_status()
            return response.content

    def get_json(self, url: str, params: dict[str, str]) -> JsonValue:
        with create_client(base_url=self.base_url, timeout_seconds=self.timeout_seconds) as client:
            response = client.get(url, params=params)
            _ = response.raise_for_status()
            return cast("JsonValue", json.loads(response.text))


def create_client(*, base_url: str = "", timeout_seconds: float | None = None) -> httpx2.Client:
    read_timeout = _READ_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
    timeout = httpx2.Timeout(
        connect=_CONNECT_TIMEOUT_SECONDS,
        read=read_timeout,
        write=_WRITE_TIMEOUT_SECONDS,
        pool=_POOL_TIMEOUT_SECONDS,
    )
    transport = httpx2.HTTPTransport(
        http2=True,
        retries=3,
        limits=_LIMITS,
        socket_options=_SOCKET_OPTIONS,
    )
    return httpx2.Client(
        transport=transport,
        timeout=timeout,
        base_url=base_url,
        follow_redirects=True,
        headers={"accept-encoding": "br, zstd, gzip"},
    )
