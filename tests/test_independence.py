"""The independence-of-monetary-value assumption, made checkable.

Fader, Hardie & Lee (2005), "RFM and CLV: Using Iso-value Curves for Customer
Base Analysis", §2.1(iii) states the assumption the composed CLV rests on, and
§2.2 is the authors' own assessment of it on CDNOW. These tests reproduce that
assessment.
"""

from functools import cache
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

from clvkit import CustomerBase
from clvkit._result import Result
from clvkit.clv.independence import IndependenceCheck, check_monetary_independence

CDNOW_SAMPLE = Path(__file__).resolve().parents[1] / "CDNOW_sample.txt"
CDNOW_CALIBRATION_END = "1997-09-30"


@cache
def _cdnow_calibration() -> CustomerBase:
    """The CDNOW 1/10 sample over weeks 1-39 — the base §2.2 is computed on."""
    log = pd.read_csv(
        CDNOW_SAMPLE,
        sep=r"\s+",
        header=None,
        names=["customer_id", "sample_id", "date", "quantity", "amount"],
    )
    log["date"] = pd.to_datetime(log["date"], format="%Y%m%d")
    full = CustomerBase.from_transactions(
        log[["customer_id", "date", "amount"]], time_unit="D"
    )
    calibration, _ = full.split(calibration_period_end=CDNOW_CALIBRATION_END)
    return calibration


def _log(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["customer_id", "date", "amount"])


def _independent_base(n_customers: int = 400, seed: int = 7) -> CustomerBase:
    """Spend drawn with no reference to how often a customer buys."""
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2020-01-01")
    for customer in range(n_customers):
        repeats = 1 + rng.poisson(2.5)
        for offset in range(repeats + 1):
            rows.append(
                (
                    f"c{customer:04d}",
                    start + pd.Timedelta(days=7 * offset),
                    float(rng.gamma(shape=6.0, scale=6.0)),
                )
            )
    return CustomerBase.from_transactions(_log(rows))


def _dependent_base(n_customers: int = 400, seed: int = 7) -> CustomerBase:
    """Spend deliberately tied to buying rate — the assumption broken on purpose."""
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2020-01-01")
    for customer in range(n_customers):
        repeats = 1 + rng.poisson(2.5)
        # Heavy buyers spend proportionally more per transaction.
        scale = 6.0 * (1 + repeats)
        for offset in range(repeats + 1):
            rows.append(
                (
                    f"c{customer:04d}",
                    start + pd.Timedelta(days=7 * offset),
                    float(rng.gamma(shape=6.0, scale=scale)),
                )
            )
    return CustomerBase.from_transactions(_log(rows))


# --- the published §2.2 assessment --------------------------------------


def test_golden_cdnow_reproduces_the_papers_correlation():
    # §2.2: "the simple correlation between average transaction value and the
    # number of transactions is 0.11", over the 946 repeat buyers of weeks 1-39.
    check = check_monetary_independence(_cdnow_calibration())

    assert check.n_customers == 946
    assert round(check.pearson_r, 2) == 0.11


def test_golden_cdnow_reproduces_the_papers_outlier_adjusted_correlation():
    # §2.2 names the outlier — "a customer who made 21 transactions ... with an
    # average transaction value of $300" — and reports 0.06 (p = 0.08) without
    # it. Dropping that one customer here must land on the published numbers.
    summary = _cdnow_calibration().to_pandas()
    outlier = summary["monetary_value"].idxmax()

    assert summary.loc[outlier, "frequency"] == 21
    assert summary.loc[outlier, "monetary_value"] == pytest.approx(299.63, abs=0.01)

    without = check_monetary_independence(_cdnow_calibration(), exclude=[outlier])

    assert round(without.pearson_r, 2) == 0.06
    assert round(without.pearson_p, 2) == 0.08


def test_golden_cdnow_within_group_variation_dominates_between_group():
    # §2.2's reading of Figure 4: "the variation within each number-of-
    # transactions group dominates the between-group variation." eta squared is
    # that sentence as a number — the share of monetary-value variance that
    # frequency explains at all.
    check = check_monetary_independence(_cdnow_calibration())

    assert check.eta_squared < 0.15


def test_golden_cdnow_passes_the_independence_check():
    # The paper's verdict: "we do not feel that it represents a substantial
    # violation of our independence assumption."
    assert check_monetary_independence(_cdnow_calibration()).holds()


# --- the check as a behaviour -------------------------------------------


def test_independent_spend_holds():
    assert check_monetary_independence(_independent_base()).holds()


def test_spend_tied_to_buying_rate_fails_the_check():
    check = check_monetary_independence(_dependent_base())

    assert not check.holds()
    assert check.spearman_rho > 0.3


def test_thresholds_are_overridable():
    check = check_monetary_independence(_independent_base())

    assert not check.holds(max_correlation=0.0)
    assert not check.holds(max_eta_squared=0.0)


def test_check_uses_repeat_buyers_only():
    # A one-time buyer has no observed average transaction value, so §2.2's
    # statistics are computed over repeat buyers only — 946 of CDNOW's 2,357.
    summary = _cdnow_calibration().to_pandas()

    check = check_monetary_independence(_cdnow_calibration())

    assert len(summary) > check.n_customers
    assert check.n_customers == int((summary["frequency"] > 0).sum())


# --- the Result contract -------------------------------------------------


def test_check_satisfies_the_result_contract():
    assert isinstance(check_monetary_independence(_independent_base()), Result)


def test_to_pandas_carries_every_reported_statistic():
    check = check_monetary_independence(_independent_base())

    frame = check.to_pandas()

    assert list(frame.columns) == ["value"]
    assert set(frame.index) == {
        "n_customers",
        "pearson_r",
        "pearson_p",
        "spearman_rho",
        "spearman_p",
        "eta_squared",
    }
    assert frame.loc["pearson_r", "value"] == pytest.approx(check.pearson_r)


def test_to_json_round_trips():
    check = check_monetary_independence(_independent_base())

    restored = pd.read_json(StringIO(check.to_json()), orient="index")

    assert restored.loc["eta_squared", "value"] == pytest.approx(check.eta_squared)


def test_plot_draws_the_papers_figure_4():
    # Figure 4: "The Distribution of Average Transaction Value by Number of
    # Transactions" — a box-and-whisker per frequency level.
    check = check_monetary_independence(_independent_base())

    ax = check.plot()

    assert isinstance(ax, Axes)
    assert "transaction" in ax.get_ylabel().lower()
    assert "repeat" in ax.get_xlabel().lower()
    assert len(ax.get_xticklabels()) > 1


def test_plot_buckets_the_long_frequency_tail():
    # CDNOW's frequency runs to 21; one box per level would be unreadable.
    check = check_monetary_independence(_cdnow_calibration())

    ax = check.plot(max_frequency=5)

    labels = [tick.get_text() for tick in ax.get_xticklabels()]
    assert labels == ["1", "2", "3", "4", "5+"]


def test_repr_reports_the_verdict():
    assert "holds" in repr(check_monetary_independence(_independent_base()))
    assert "violated" in repr(check_monetary_independence(_dependent_base()))


# --- preconditions -------------------------------------------------------


def test_check_needs_monetary_value():
    cb = CustomerBase.from_transactions(
        _log([("A", "2020-01-01", 10), ("A", "2020-01-08", 20)]), amount_col=None
    )

    with pytest.raises(ValueError, match="amount_col"):
        check_monetary_independence(cb)


def test_check_needs_repeat_buyers():
    cb = CustomerBase.from_transactions(
        _log([("A", "2020-01-01", 10), ("B", "2020-01-01", 20)])
    )

    with pytest.raises(ValueError, match="repeat"):
        check_monetary_independence(cb)


def test_exclude_rejects_a_customer_the_base_has_never_heard_of():
    # A typo here would quietly answer a different question.
    with pytest.raises(KeyError, match="not in this base"):
        check_monetary_independence(_independent_base(), exclude=["nobody"])


def test_check_needs_more_than_one_frequency_level():
    # Every repeat buyer bought exactly twice: there is no variation in
    # frequency, so no correlation with it can be computed.
    cb = CustomerBase.from_transactions(
        _log(
            [
                ("A", "2020-01-01", 10),
                ("A", "2020-01-08", 12),
                ("B", "2020-01-01", 20),
                ("B", "2020-01-08", 22),
            ]
        )
    )

    with pytest.raises(ValueError, match="frequency"):
        check_monetary_independence(cb)


def test_construction_is_via_the_function_not_the_class():
    # IndependenceCheck holds already-computed statistics; the customer base
    # goes in through check_monetary_independence.
    assert isinstance(
        check_monetary_independence(_independent_base()), IndependenceCheck
    )
