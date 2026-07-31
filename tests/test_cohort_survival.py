"""CohortSurvival — non-contractual survival from aggregated fitted P(alive).

The hand-built base below is monthly, observed to 1998-04-30, so every
cohort, age and average is checkable by eye:

    customer  first buy   cohort    age (months)
    A         1998-01     1998-01        3
    B         1998-01     1998-01        3
    C         1998-02     1998-02        2
    D         1998-03     1998-03        1

A stub transaction model hands back a fixed P(alive) per customer, so the
expected curve is arithmetic a reader can do in their head.
"""

import json

import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

from clvkit import BGNBD, MBGNBD, CohortSurvival, CustomerBase
from clvkit._result import Prediction, Result


@pytest.fixture
def log() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["A", "A", "B", "B", "C", "C", "D"],
            "date": pd.to_datetime(
                [
                    "1998-01-05",
                    "1998-03-10",
                    "1998-01-20",
                    "1998-02-02",
                    "1998-02-14",
                    "1998-04-01",
                    "1998-03-08",
                ]
            ),
            "amount": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
        }
    )


@pytest.fixture
def base(log) -> CustomerBase:
    return CustomerBase.from_transactions(
        log, time_unit="M", observation_period_end="1998-04-30"
    )


class _StubAliveModel:
    """A stand-in transaction model with a P(alive) the test dictates."""

    def __init__(self, alive: dict[str, float]) -> None:
        self.alive = alive
        self.fitted_on: CustomerBase | None = None

    def fit(self, cb: CustomerBase) -> "_StubAliveModel":
        self.fitted_on = cb
        return self

    def probability_alive(self, cb: CustomerBase | None = None) -> Prediction:
        index = (cb or self.fitted_on).to_pandas().index
        return Prediction(
            pd.Series([self.alive[i] for i in index], index=index),
            name="probability_alive",
            description="Probability alive",
        )


def _stub() -> _StubAliveModel:
    return _StubAliveModel({"A": 0.8, "B": 0.6, "C": 0.5, "D": 1.0})


def _simulated_base(n_customers: int = 200, seed: int = 7) -> CustomerBase:
    """A small base drawn from the BG/NBD story, with staggered acquisitions.

    Customers are born across the first 20 weeks so cohorts, recency and T
    actually vary — a base acquired all at once is a single cohort and has no
    survival curve to draw.
    """
    rng = np.random.default_rng(seed)
    origin = pd.Timestamp("2020-01-06")
    horizon_weeks = 60.0

    rows = []
    for customer in range(n_customers):
        birth = float(rng.integers(0, 20))
        rate = float(rng.gamma(shape=0.8, scale=0.20))
        dropout = float(rng.beta(1.2, 3.0))

        week = birth
        while week <= horizon_weeks:
            rows.append((f"c{customer:04d}", origin + pd.Timedelta(weeks=round(week))))
            if rng.random() < dropout:
                break
            week += float(rng.exponential(1.0 / max(rate, 1e-6)))

    return CustomerBase.from_transactions(
        pd.DataFrame(rows, columns=["customer_id", "date"]),
        amount_col=None,
        time_unit="W",
        # These synthetic purchases are generated on a weekly clock, so a
        # weekly collapse loses nothing real. Said explicitly, because saying
        # nothing is what `from_transactions` warns about.
        collapse="W",
        observation_period_end=origin + pd.Timedelta(weeks=horizon_weeks),
    )


# --- the curve itself ----------------------------------------------------


def test_survival_is_the_mean_probability_alive_of_each_cohort(base):
    curve = CohortSurvival(transaction_model=_stub()).fit(base).predict()

    frame = curve.to_pandas()

    assert list(frame.index.astype(str)) == ["1998-01", "1998-02", "1998-03"]
    # (0.8 + 0.6) / 2, 0.5, 1.0
    assert list(frame["survival"]) == [0.7, 0.5, 1.0]


def test_each_cohort_carries_its_size(base):
    frame = CohortSurvival(transaction_model=_stub()).fit(base).predict().to_pandas()

    assert list(frame["customers"]) == [2, 1, 1]


def test_cohort_age_is_periods_lived_since_acquisition(base):
    # The 1998-01 cohort has been observed for three whole months by 1998-04.
    frame = CohortSurvival(transaction_model=_stub()).fit(base).predict().to_pandas()

    assert list(frame["age"]) == [3, 2, 1]


def test_the_curve_matches_a_directly_computed_aggregate_of_p_alive():
    """The acceptance criterion, computed the long way round.

    P(alive) comes from a separately fitted `BGNBD` through its own public
    verb, and the cohorts are re-derived here from `T` — so the two paths
    share nothing but the customer base.
    """
    cb = _simulated_base()
    summary = cb.to_pandas()
    alive = BGNBD().fit(cb).probability_alive().to_pandas()["probability_alive"]

    # Acquisition age is T weeks, so customers sharing a T share a cohort.
    expected = alive.groupby(summary["T"]).mean().sort_index(ascending=False)

    curve = CohortSurvival().fit(cb).predict().to_pandas()

    assert list(curve["age"]) == list(expected.index)
    assert curve["survival"].to_numpy() == pytest.approx(expected.to_numpy())


def test_every_customer_lands_in_exactly_one_cohort():
    cb = _simulated_base()

    curve = CohortSurvival().fit(cb).predict().to_pandas()

    assert curve["customers"].sum() == len(cb.to_pandas())


def test_survival_is_a_share_between_zero_and_one():
    cb = _simulated_base()

    survival = CohortSurvival().fit(cb).predict().to_pandas()["survival"]

    assert ((survival >= 0.0) & (survival <= 1.0)).all()


# --- model injection -----------------------------------------------------


def test_the_default_transaction_model_is_bgnbd():
    assert isinstance(CohortSurvival().transaction_model, BGNBD)


def test_an_injected_model_is_used_instead_of_the_default(base):
    stub = _stub()

    survival = CohortSurvival(transaction_model=stub).fit(base)

    assert survival.transaction_model is stub
    # The curve is the stub's own numbers, so the default cannot have run.
    assert survival.predict().to_pandas()["survival"].iloc[0] == 0.7


def test_mbgnbd_drops_into_the_transaction_seam():
    """The curve takes MBG/NBD with no model-specific code.

    `CohortSurvival` reaches the model only through `fit` and
    `probability_alive`, so the aggregation never learns which of the two it
    is holding. MBG/NBD lets a customer die at time zero, so it reads the
    same base as less alive than BG/NBD does.
    """
    cb = _simulated_base()

    bgnbd = CohortSurvival(BGNBD()).fit(cb).predict().to_pandas()
    mbgnbd = CohortSurvival(MBGNBD()).fit(cb).predict().to_pandas()

    assert list(bgnbd.index) == list(mbgnbd.index)
    assert not np.allclose(bgnbd["survival"], mbgnbd["survival"])
    assert (mbgnbd["survival"] <= bgnbd["survival"]).all()


def test_cohort_survival_is_exported_from_the_top_level_namespace():
    import clvkit

    assert clvkit.CohortSurvival is CohortSurvival
    assert "CohortSurvival" in clvkit.__all__


# --- the cohort grain ----------------------------------------------------


def test_cohorts_default_to_the_grain_the_base_was_summarised_at(base):
    frame = CohortSurvival(transaction_model=_stub()).fit(base).predict().to_pandas()

    assert frame.index.freqstr == pd.PeriodIndex(["1998-01"], freq="M").freqstr


def test_a_coarser_period_merges_cohorts(base):
    # January, February and March all fall in 1998Q1 — one cohort, one age.
    frame = (
        CohortSurvival(transaction_model=_stub())
        .fit(base)
        .predict(period="Q")
        .to_pandas()
    )

    assert list(frame.index.astype(str)) == ["1998Q1"]
    assert list(frame["customers"]) == [4]
    assert frame["survival"].iloc[0] == pytest.approx((0.8 + 0.6 + 0.5 + 1.0) / 4)
    assert list(frame["age"]) == [1]


def test_a_cohort_straddling_a_boundary_is_filed_by_where_it_starts():
    """A weekly cohort spanning a month-end belongs to the month it opened in.

    The week 1998-01-26/1998-02-01 is six-sevenths January. Filing it by
    where the *period* ends — pandas' own default when changing frequency —
    would post that whole cohort to February.
    """
    log = pd.DataFrame(
        {
            "customer_id": ["A", "A"],
            "date": pd.to_datetime(["1998-01-28", "1998-02-20"]),
        }
    )
    cb = CustomerBase.from_transactions(
        log, amount_col=None, time_unit="W", observation_period_end="1998-03-01"
    )

    frame = (
        CohortSurvival(transaction_model=_StubAliveModel({"A": 0.9}))
        .fit(cb)
        .predict(period="M")
        .to_pandas()
    )

    assert list(frame.index.astype(str)) == ["1998-01"]


def test_a_period_finer_than_the_time_unit_is_refused(base):
    # T dates an acquisition to the month; daily cohorts would be invented.
    survival = CohortSurvival(transaction_model=_stub()).fit(base)

    with pytest.raises(ValueError, match="finer"):
        survival.predict(period="D")


# --- preconditions -------------------------------------------------------


def test_predict_before_fit_is_an_error():
    with pytest.raises(RuntimeError, match="not fitted"):
        CohortSurvival().predict()


def test_fit_rejects_anything_that_is_not_a_customer_base(log):
    with pytest.raises(TypeError, match="CustomerBase"):
        CohortSurvival().fit(log)


# --- the Result contract -------------------------------------------------


def test_survival_curve_satisfies_the_result_protocol(base):
    curve = CohortSurvival(transaction_model=_stub()).fit(base).predict()

    assert isinstance(curve, Result)


def test_to_pandas_returns_a_copy(base):
    curve = CohortSurvival(transaction_model=_stub()).fit(base).predict()

    frame = curve.to_pandas()
    frame["survival"] = 0.0

    assert curve.to_pandas()["survival"].iloc[0] == 0.7


def test_to_json_keys_cohorts_by_label(base):
    curve = CohortSurvival(transaction_model=_stub()).fit(base).predict()

    payload = json.loads(curve.to_json())

    assert payload["1998-01"] == {"age": 3, "customers": 2, "survival": 0.7}


def test_plot_draws_survival_against_cohort_age(base):
    curve = CohortSurvival(transaction_model=_stub()).fit(base).predict()

    ax = curve.plot()

    assert isinstance(ax, Axes)
    assert "age" in ax.get_xlabel().lower()
    assert ax.get_ylim() == (0.0, 1.0)
    # Oldest cohort leftmost: ages ascending, whatever order the frame is in.
    line = ax.lines[0]
    assert list(line.get_xdata()) == [1, 2, 3]
    assert list(line.get_ydata()) == [1.0, 0.5, 0.7]


def test_plot_accepts_an_existing_axes(base):
    import matplotlib.pyplot as plt

    curve = CohortSurvival(transaction_model=_stub()).fit(base).predict()
    _, ax = plt.subplots()

    assert curve.plot(ax=ax) is ax


def test_repr_names_the_cohorts_and_the_model(base):
    survival = CohortSurvival(transaction_model=_stub())

    assert "unfitted" in repr(survival)

    curve = survival.fit(base).predict()

    assert "3" in repr(curve)
    assert "SurvivalCurve" in repr(curve)
