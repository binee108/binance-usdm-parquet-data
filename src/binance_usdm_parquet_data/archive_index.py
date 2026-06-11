from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from urllib.parse import quote

from defusedxml import ElementTree


@dataclass(frozen=True, slots=True)
class ArchivePrefixPage:
    prefixes: tuple[str, ...]
    next_token: str | None


class SyncHttpClient(Protocol):
    def get_text(self, url: str, params: dict[str, str]) -> str: ...


@dataclass(frozen=True, slots=True)
class S3ArchiveIndexClient:
    http: SyncHttpClient
    base_url: str = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"

    def list_prefix_pages(self, prefix: str) -> list[ArchivePrefixPage]:
        pages: list[ArchivePrefixPage] = []
        continuation: str | None = None
        while True:
            params = {"delimiter": "/", "list-type": "2", "prefix": prefix}
            if continuation is not None:
                params["continuation-token"] = continuation
            text = self.http.get_text(self.base_url.rstrip("/"), params)
            page = parse_s3_prefix_page(text)
            pages.append(page)
            if page.next_token is None:
                return pages
            continuation = page.next_token


def parse_s3_prefix_page(xml_text: str) -> ArchivePrefixPage:
    root = ElementTree.fromstring(xml_text)
    namespace = _namespace(root.tag)
    prefix_tag = f"{namespace}CommonPrefixes"
    key_tag = f"{namespace}Prefix"
    token_tag = f"{namespace}NextContinuationToken"
    prefixes: list[str] = []
    for node in root.findall(prefix_tag):
        value = node.findtext(key_tag)
        if value:
            prefixes.append(value)
    return ArchivePrefixPage(
        prefixes=tuple(prefixes),
        next_token=root.findtext(token_tag),
    )


def encode_path_segment(value: str) -> str:
    return quote(value, safe="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_=")


def _namespace(tag: str) -> str:
    if tag.startswith("{"):
        return tag.split("}", maxsplit=1)[0] + "}"
    return ""
