"""The Dask engine must agree with the pandas one, exactly.

A second implementation of a documented input contract is only worth having if
it cannot drift from the first. So almost everything here is the same
assertion — build the same log both ways, demand identical values *and*
identical dtypes — repeated across the paths where the two heads diverge:
netting, dropping, timing-only logs, and a ruler coarser than the grain.

Row counts are compared, not just values. Under ``on_negative="net"`` a
customer whose every period nets non-positive vanishes from the index
entirely, and an engine that lost a different set of customers would still
pass a value-only comparison on the intersection.

This file skips wholesale without the ``dask`` extra, which is the point of it
being an extra. The checks that must hold on the four-dependency install —
including the error you get for asking for Dask without having it — live in
``test_engine_selection.py``, which imports nothing optional.
"""

import pandas as pd
import pytest

from clvkit import CohortMatrix, CustomerBase

dd = pytest.importorskip("dask.dataframe", reason="needs the optional dask extra")
dask = pytest.importorskip("dask")


@pytest.fixture(autouse=True)
def _hand_both_engines_the_same_dtypes():
    """Turn off Dask's Arrow-string conversion for the duration of a test.

    ``dd.from_pandas`` rewrites a str column to ``string[pyarrow]`` by default,
    which would leave the dask summary's index a different dtype from the
    pandas one for reasons that have nothing to do with this library. Storage
    is not semantics, and comparing dtypes is only meaningful when both engines
    were handed the same log — so it is turned off rather than asserted around.
    """
    with dask.config.set({"dataframe.convert-string": False}):
        yield


# Five customers with different shapes: alice repeats within and across months,
# bob repeats across months only, carol never repeats, and erin has a refund
# that only cancels one of her periods. dave's refund cancels his only
# purchase, so `on_negative="net"` must delete him from both engines' output
# rather than leave him behind at frequency 0.
LOG = pd.DataFrame(
    [
        ("alice", "2024-01-01", 10.0),
        ("alice", "2024-01-03", 20.0),
        ("alice", "2024-02-05", 30.0),
        ("alice", "2024-03-20", 40.0),
        ("bob", "2024-01-02", 15.0),
        ("bob", "2024-03-01", 25.0),
        ("carol", "2024-01-08", 40.0),
        ("dave", "2024-02-02", 50.0),
        ("dave", "2024-02-02", -50.0),
        ("erin", "2024-01-15", 5.0),
        ("erin", "2024-01-16", -1.0),
        ("erin", "2024-04-02", 12.0),
    ],
    columns=["customer_id", "date", "amount"],
)


@pytest.fixture
def log() -> pd.DataFrame:
    frame = LOG.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


@pytest.fixture
def lazy(log) -> "dd.DataFrame":
    # More than one partition on purpose: a per-partition period conversion
    # that read its own partition's minimum as an origin would pass at one.
    return dd.from_pandas(log, npartitions=3)


def both(log, lazy, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The same summary from both engines, index-sorted for comparison."""
    eager = CustomerBase.from_transactions(log, **kwargs)
    lazily = CustomerBase.from_transactions(lazy, engine="dask", **kwargs)
    assert lazily.engine == "dask"
    return eager.to_pandas().sort_index(), lazily.to_pandas().sort_index()


class TestTheTwoEnginesSummariseIdentically:
    def test_the_default_path(self, log, lazy):
        eager, lazily = both(log, lazy)
        pd.testing.assert_frame_equal(eager, lazily)

    def test_netting_drops_the_same_customers(self, log, lazy):
        eager, lazily = both(log, lazy, on_negative="net")
        # dave nets to zero across February and leaves the base entirely.
        assert "dave" not in eager.index
        pd.testing.assert_frame_equal(eager, lazily)

    def test_dropping_negatives(self, log, lazy):
        eager, lazily = both(log, lazy, on_negative="drop")
        assert "dave" in eager.index
        pd.testing.assert_frame_equal(eager, lazily)

    def test_both_refuse_a_negative_amount_under_raise(self, log, lazy):
        with pytest.raises(ValueError, match="negative amounts"):
            CustomerBase.from_transactions(log, on_negative="raise")
        with pytest.raises(ValueError, match="negative amounts"):
            CustomerBase.from_transactions(lazy, engine="dask", on_negative="raise")

    def test_a_timing_only_log(self, log, lazy):
        eager, lazily = both(log, lazy, amount_col=None)
        assert "monetary_value" not in eager.columns
        pd.testing.assert_frame_equal(eager, lazily)

    def test_a_ruler_coarser_than_the_grain(self, log, lazy):
        eager, lazily = both(log, lazy, time_unit="W", collapse="D")
        pd.testing.assert_frame_equal(eager, lazily)

    def test_an_explicit_observation_period_end(self, log, lazy):
        eager, lazily = both(log, lazy, observation_period_end="2024-06-30")
        pd.testing.assert_frame_equal(eager, lazily)

    def test_a_monthly_grain(self, log, lazy):
        eager, lazily = both(log, lazy, time_unit="M", collapse="M")
        pd.testing.assert_frame_equal(eager, lazily)

    @pytest.mark.parametrize("npartitions", [1, 2, 5, 12])
    def test_the_answer_does_not_depend_on_the_partitioning(self, log, npartitions):
        """The regression test for the bug a three-partition fixture missed.

        A per-customer figure is only correct if the engine sees all of that
        customer's rows together. At twelve partitions over twelve rows every
        repeat buyer is split, which is the condition that broke an earlier
        draft — and broke it silently at first, on values rather than an
        exception.
        """
        eager, lazily = both(log, dd.from_pandas(log, npartitions=npartitions))
        pd.testing.assert_frame_equal(eager, lazily)

    def test_an_inherited_coarse_grain_warns_the_same_either_way(self, log, lazy):
        """The warning path, which needs `collapse` left implicit to fire."""
        with pytest.warns(UserWarning, match="collapsed 3 of 12 transactions") as eager:
            CustomerBase.from_transactions(log, time_unit="M")
        with pytest.warns(
            UserWarning, match="collapsed 3 of 12 transactions"
        ) as lazily:
            CustomerBase.from_transactions(lazy, time_unit="M", engine="dask")
        assert str(eager[0].message) == str(lazily[0].message)

    def test_the_cdnow_sample(self, cdnow_sample):
        """The real log the published estimates are fit on, both ways.

        6,919 transactions and integer ids, so the dtype comparison here is
        strict without any help from the fixture above.
        """
        lazy = dd.from_pandas(cdnow_sample, npartitions=4)
        eager, lazily = both(
            cdnow_sample, lazy, time_unit="W", collapse="D", amount_col="amount"
        )
        # 2,357 customers in the sample, less the handful whose every period
        # nets to zero and is dropped by the default on_negative="net".
        assert len(eager) == 2349
        pd.testing.assert_frame_equal(eager, lazily)


class TestTheTwoEnginesPivotIdentically:
    def _both(self, log, lazy, **kwargs):
        eager = CohortMatrix.from_transactions(log, **kwargs)
        lazily = CohortMatrix.from_transactions(lazy, engine="dask", **kwargs)
        return eager.to_pandas(), lazily.to_pandas()

    def test_retention(self, log, lazy):
        eager, lazily = self._both(log, lazy)
        pd.testing.assert_frame_equal(eager, lazily)

    def test_revenue(self, log, lazy):
        eager, lazily = self._both(log, lazy, metric="revenue")
        pd.testing.assert_frame_equal(eager, lazily)

    def test_a_weekly_period(self, log, lazy):
        eager, lazily = self._both(log, lazy, period="W")
        pd.testing.assert_frame_equal(eager, lazily)

    @pytest.mark.parametrize("npartitions", [1, 2, 5, 12])
    def test_the_answer_does_not_depend_on_the_partitioning(self, log, npartitions):
        eager, lazily = self._both(log, dd.from_pandas(log, npartitions=npartitions))
        pd.testing.assert_frame_equal(eager, lazily)

    def test_the_cohort_index_is_periods_not_ordinals(self, lazy):
        matrix = CohortMatrix.from_transactions(lazy, engine="dask").to_pandas()
        assert isinstance(matrix.index, pd.PeriodIndex)

    def test_an_empty_log_is_refused_by_both(self, log):
        empty = log.iloc[:0]
        with pytest.raises(ValueError, match="empty transaction log"):
            CohortMatrix.from_transactions(empty)
        with pytest.raises(ValueError, match="empty transaction log"):
            CohortMatrix.from_transactions(
                dd.from_pandas(empty, npartitions=1), engine="dask"
            )


class TestTheTwoDivergencesThatDoExist:
    """Two things the engines genuinely differ on, pinned so they stay small.

    Both are documented on ``from_transactions``. Pinning them here is what
    stops a third one appearing quietly: a test suite that only asserts
    agreement cannot tell "still identical" from "nobody looked".
    """

    def test_row_order_differs_and_nothing_else_does(self, log, lazy):
        eager = CustomerBase.from_transactions(log).to_pandas()
        lazily = CustomerBase.from_transactions(lazy, engine="dask").to_pandas()
        # pandas keeps first-appearance order; the dask shuffle decides its own.
        assert set(eager.index) == set(lazily.index)
        pd.testing.assert_frame_equal(eager.sort_index(), lazily.sort_index())

    def test_dask_keeps_the_id_dtype_it_was_given(self, log):
        """Values agree under Dask's default Arrow-string conversion; storage
        does not, because the conversion happens to the *input* before clvkit
        sees it. The summary carries through whatever dtype it was handed."""
        with dask.config.set({"dataframe.convert-string": True}):
            lazily = CustomerBase.from_transactions(
                dd.from_pandas(log, npartitions=3), engine="dask"
            ).to_pandas()
        eager = CustomerBase.from_transactions(log).to_pandas()

        assert lazily.index.dtype != eager.index.dtype
        pd.testing.assert_frame_equal(
            eager.sort_index(), lazily.sort_index(), check_index_type=False
        )


class TestADaskBaseSaysWhatItCannotDo:
    def test_split_refuses_and_names_the_way_out(self, lazy):
        base = CustomerBase.from_transactions(lazy, engine="dask")
        with pytest.raises(ValueError, match="engine='pandas'"):
            base.split(calibration_period_end="2024-02-01")

    def test_split_still_works_on_the_pandas_engine(self, log):
        base = CustomerBase.from_transactions(log)
        calibration, holdout = base.split(calibration_period_end="2024-02-01")
        assert len(calibration.to_pandas()) == len(holdout)

    def test_the_long_form_states_it(self, lazy):
        base = CustomerBase.from_transactions(lazy, engine="dask")
        assert "engine='dask'" in str(base)
        assert ".split() is unavailable" in str(base)

    def test_a_pandas_base_says_nothing_about_engines(self, log):
        assert "engine=" not in str(CustomerBase.from_transactions(log))


class TestAPandasFrameUnderTheDaskEngineIsRefused:
    """Lives here, not in test_engine_selection: without the extra installed
    the missing-extra ImportError fires first, which is the right order."""

    def test_customer_base(self, log):
        with pytest.raises(TypeError, match="expects a dask DataFrame"):
            CustomerBase.from_transactions(log, engine="dask")

    def test_cohort_matrix(self, log):
        with pytest.raises(TypeError, match="expects a dask DataFrame"):
            CohortMatrix.from_transactions(log, engine="dask")
