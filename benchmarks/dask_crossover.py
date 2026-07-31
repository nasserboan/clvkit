"""Where does ``engine="dask"`` start to win?

Summarising is the only step in clvkit whose cost scales with transactions
rather than customers, so it is the only step Dask is offered for — and the
claim that it helps is worth a number rather than an assertion.

Each measurement runs in its own subprocess. Peak RSS is a high-water mark
that never comes back down inside a process, so measuring both engines in one
interpreter would report the larger of the two twice.

Both engines read the same Parquet file from disk. Handing Dask a pandas frame
would be a rigged comparison: the log would already be in memory, which is the
cost the whole exercise is trying to avoid.

    uv run --extra dask python benchmarks/dask_crossover.py
    uv run --extra dask python benchmarks/dask_crossover.py --sizes 100_000 1_000_000

Results, and what they mean, are in benchmarks/README.md.
"""

import argparse
import json
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Roughly CDNOW-shaped: a long tail of one-time buyers, a thin core of
# repeaters. The ratio matters, because it decides how much the summary shrinks
# the log — which is the whole quantity being measured.
TRANSACTIONS_PER_CUSTOMER = 3
SIZES = (100_000, 1_000_000, 4_000_000)
PARTITION_ROWS = 500_000

# `ru_maxrss` is bytes on macOS and kilobytes everywhere else, and getrusage
# will not tell you which. Getting this wrong is a silent factor of 1000 in the
# column the whole benchmark is read for.
_RSS_TO_MB = 1e6 if sys.platform == "darwin" else 1e3


def write_log(
    path: Path,
    n_transactions: int,
    per_customer: int = TRANSACTIONS_PER_CUSTOMER,
    seed: int = 0,
) -> None:
    """A synthetic transaction log in the documented input contract."""
    rng = np.random.default_rng(seed)
    n_customers = max(1, n_transactions // per_customer)
    frame = pd.DataFrame(
        {
            "customer_id": rng.integers(0, n_customers, n_transactions),
            "date": pd.Timestamp("2020-01-01")
            + pd.to_timedelta(rng.integers(0, 1461, n_transactions), unit="D"),
            "amount": np.round(rng.gamma(2.0, 25.0, n_transactions), 2),
        }
    )
    # One file per partition, so the Dask reader has something to stream.
    frame.to_parquet(
        path,
        partition_cols=None,
        index=False,
        row_group_size=PARTITION_ROWS,
    )


def measure(engine: str, path: Path) -> dict:
    """Summarise the log once, and report what it cost."""
    from clvkit import CustomerBase

    # The read is inside the timer for both. Dask's read is lazy and pandas'
    # is not, so leaving it out would charge Dask for work pandas had already
    # done — and reading the log is part of summarising it either way.
    started = time.perf_counter()
    if engine == "dask":
        import dask.dataframe as dd

        log = dd.read_parquet(path, split_row_groups=True)
    else:
        log = pd.read_parquet(path)
    base = CustomerBase.from_transactions(
        log, time_unit="W", collapse="D", engine=engine
    )
    elapsed = time.perf_counter() - started

    return {
        "engine": engine,
        "seconds": round(elapsed, 2),
        "peak_mb": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / _RSS_TO_MB, 1
        ),
        "customers": len(base.to_pandas()),
    }


def run_cell(engine: str, path: Path) -> dict:
    """One measurement, in a fresh interpreter — see the module docstring."""
    out = subprocess.run(
        [sys.executable, __file__, "--cell", engine, str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def main(sizes: tuple[int, ...], per_customer: int) -> None:
    print(f"{'rows':>12}  {'engine':>7}  {'seconds':>8}  {'peak MB':>8}  customers")
    with tempfile.TemporaryDirectory() as tmp:
        for size in sizes:
            path = Path(tmp) / f"log_{size}.parquet"
            write_log(path, size, per_customer)
            for engine in ("pandas", "dask"):
                cell = run_cell(engine, path)
                print(
                    f"{size:>12,}  {cell['engine']:>7}  {cell['seconds']:>8.2f}  "
                    f"{cell['peak_mb']:>8.1f}  {cell['customers']:,}"
                )
            path.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", nargs=2, metavar=("ENGINE", "PATH"))
    parser.add_argument("--sizes", nargs="+", type=int, default=list(SIZES))
    # How hard the summary compresses the log is the whole question, so it is
    # a knob: at 3 the RFM table is nearly as tall as the log, at 50 it is not.
    parser.add_argument("--per-customer", type=int, default=TRANSACTIONS_PER_CUSTOMER)
    args = parser.parse_args()

    if args.cell:
        engine, path = args.cell
        print(json.dumps(measure(engine, Path(path))))
    else:
        main(tuple(args.sizes), args.per_customer)
