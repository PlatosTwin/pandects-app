from typing import List, Optional
from dataclasses import dataclass
import uuid
import pandas as pd


@dataclass
class FilingMetadata:
    """Metadata for a filing document."""
    agreement_uuid: str
    url: str
    target: str
    acquirer: str
    filing_date: str


def get_uuid(x: str) -> str:
    """Generate a UUID5 hash from the input string."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, x))


def fetch_new_filings(since: str) -> List[FilingMetadata]:
    """
    Fetch new filings from the DMA corpus.

    Eventually:
    1. Fetch all new filings
    2. Identify definitive merger agreements
    3. Identify transaction metadata
    4. Aggregate into FilingMetadata object

    Currently:
    1. Pull the 10 oldest non-staged agreements from the DMA corpus (simulate daily filings)
    2. Use DMA metadata where available, and nulls everywhere else

    Args:
        since: Date string to filter filings from.

    Returns:
        List of FilingMetadata objects.
    """
    usecols = ["target", "acquirer", "date_announcement", "url", "filename"]
    df = pd.read_csv(
        "/Users/nikitabogdanov/Downloads/dma_corpus_metadata.csv",
        usecols=usecols,
        parse_dates=["date_announcement"],
    )

    # Drop duplicate filings by filename
    df.drop_duplicates(subset="filename", inplace=True)

    # Keep only filings announced after `since`
    cutoff = pd.to_datetime(since)
    df = df[df["date_announcement"] > cutoff]

    # Sort oldest first and take only the 10 oldest new filings
    df.sort_values("date_announcement", ascending=True, inplace=True)
    # df = df.head(10)
    df = df.sample(frac=0.25)

    # Build our results list via a memory‑light iterator
    results: List[FilingMetadata] = []
    for row in df.itertuples(index=False):
        # Ensure all fields are strings, handle date formatting safely
        date_ann = row.date_announcement
        try:
            date_obj = pd.to_datetime(str(date_ann))
            filing_date = date_obj.strftime("%Y-%m-%d")
        except Exception:
            filing_date = str(date_ann)
        results.append(
            FilingMetadata(
                agreement_uuid=get_uuid(str(row.filename)),
                url=str(row.url),
                target=str(row.target),
                acquirer=str(row.acquirer),
                filing_date=filing_date,
            )
        )

    return results
