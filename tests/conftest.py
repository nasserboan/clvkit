from pathlib import Path

import matplotlib
import pandas as pd
import pytest

# Tests must never try to open a window.
matplotlib.use("Agg")

CDNOW_SAMPLE = Path(__file__).resolve().parents[1] / "CDNOW_sample.txt"


@pytest.fixture(scope="session")
def cdnow_sample() -> pd.DataFrame:
    """The CDNOW 1/10 systematic sample — 2,357 customers, 1997-01 to 1998-06.

    Columns in the raw file are: master-file customer id, sample customer id,
    date (YYYYMMDD), number of CDs, dollar value. Every published Fader-Hardie
    estimate is fit on this sample rather than the 23,570-customer master.
    """
    transactions = pd.read_csv(
        CDNOW_SAMPLE,
        sep=r"\s+",
        header=None,
        names=["master_id", "customer_id", "date", "quantity", "amount"],
    )
    transactions["date"] = pd.to_datetime(transactions["date"], format="%Y%m%d")
    return transactions
