from __future__ import annotations

from dataclasses import dataclass, field

from binance_usdm_parquet_data.archive_index import S3ArchiveIndexClient


@dataclass(slots=True)
class RecordingHttpClient:
    calls: list[tuple[str, dict[str, str]]] = field(default_factory=list)

    def get_text(self, url: str, params: dict[str, str]) -> str:
        self.calls.append((url, dict(params)))
        return (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
            "<CommonPrefixes><Prefix>data/futures/um/monthly/klines/BTCUSDT/</Prefix></CommonPrefixes>"
            "</ListBucketResult>"
        )


def test_s3_archive_index_uses_bucket_listing_endpoint() -> None:
    client = RecordingHttpClient()

    pages = S3ArchiveIndexClient(client).list_prefix_pages("data/futures/um/monthly/klines/")

    assert len(pages) == 1
    assert client.calls == [
        (
            "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision",
            {
                "delimiter": "/",
                "list-type": "2",
                "prefix": "data/futures/um/monthly/klines/",
            },
        )
    ]
