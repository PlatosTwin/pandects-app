from typing import List
from dataclasses import dataclass
import uuid
import pandas as pd


@dataclass
class FilingMetadata:
    agreement_uuid: str
    url: str
    target: str
    acquirer: str
    filing_date: str
    transaction_date: str
    transaction_price: int
    transaction_type: str
    transaction_consideration: str
    consideration_type: str
    target_type: str


def get_uuid(x):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, x))


def fetch_new_filings(since: str) -> List[FilingMetadata]:
    """
    Eventually:
    1. Fetch all new filings
    2. Identify definitive merger agreements
    3. Identify transaction metadata
    4. Aggregate into FilingMetadata object

    Currently:
    1. Pull the 10 oldest non-staged agreements from the DMA corpus
    2. Use DMA metadata where available, and nulls everywhere else
    """
    usecols = ["target", "acquirer", "date_announcement", "url", "filename"]
    df = pd.read_csv(
        "/Users/nikitabogdanov/Downloads/dma_corpus_metadata.csv",
        usecols=usecols,
        parse_dates=["date_announcement"],
    )

    # 2) Drop duplicate filings by filename
    df.drop_duplicates(subset="filename", inplace=True)

    # 3) Keep only filings announced after `since`
    cutoff = pd.to_datetime(since)
    df = df[df["date_announcement"] > cutoff]

    # 4) Sort oldest first and take only the 10 oldest new filings
    df.sort_values("date_announcement", ascending=True, inplace=True)
    df = df.head(10)

    # 5) Build our results list via a memory‑light iterator
    results: List[FilingMetadata] = []
    for row in df.itertuples(index=False):
        results.append(
            FilingMetadata(
                agreement_uuid=get_uuid(row.filename),
                url=row.url,
                target=row.target,
                acquirer=row.acquirer,
                filing_date="",
                transaction_date=row.date_announcement.strftime("%Y-%m-%d"),
                transaction_price=0,
                transaction_type="",
                transaction_consideration="",
                consideration_type="",
                target_type="",
            )
        )

    return results, df["date_announcement"].tolist()[-1]
