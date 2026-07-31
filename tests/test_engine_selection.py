"""Asking for an engine you do not have must fail here, and say so plainly.

Everything below holds on the four-dependency install — nothing in this file
imports Dask, and the missing-extra case fakes its absence rather than
requiring it. That is deliberate: the worst version of an optional dependency
is one whose absence surfaces as an ``ImportError`` three frames inside
someone else's library.
"""

import sys

import pandas as pd
import pytest

from clvkit import CohortMatrix, CustomerBase

LOG = pd.DataFrame(
    [
        ("alice", "2024-01-01", 10.0),
        ("alice", "2024-02-05", 30.0),
        ("bob", "2024-01-02", 15.0),
    ],
    columns=["customer_id", "date", "amount"],
)


class TestImportingClvkitCostsNothingOptional:
    def test_dask_is_not_imported_by_importing_clvkit(self):
        # A fresh interpreter, because this one may have imported dask already
        # via another test module.
        import subprocess

        probe = "import sys, clvkit; assert 'dask' not in sys.modules"
        subprocess.run([sys.executable, "-c", probe], check=True)


class TestTheEngineArgumentIsCheckedBeforeAnyWork:
    def test_an_unknown_engine_is_named(self):
        with pytest.raises(ValueError, match="engine must be one of"):
            CustomerBase.from_transactions(LOG, engine="polars")
        with pytest.raises(ValueError, match="engine must be one of"):
            CohortMatrix.from_transactions(LOG, engine="polars")

    def test_a_missing_extra_is_named_not_traced(self, monkeypatch):
        # None in sys.modules is how the import system spells "this module is
        # unavailable" — the same ImportError a machine without dask raises.
        monkeypatch.setitem(sys.modules, "dask.dataframe", None)
        with pytest.raises(ImportError, match=r"clvkit\[dask\]"):
            CustomerBase.from_transactions(LOG, engine="dask")
