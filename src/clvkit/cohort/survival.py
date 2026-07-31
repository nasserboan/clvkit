"""Model-based cohort survival — P(alive), aggregated.

A non-contractual business never observes a customer leaving, so there is no
event to feed a survival analysis: churn is latent, and the only thing that
can say who is still alive is a model that infers it. That is exactly what
the BTYD P(alive) is — the denominator of eq. (10) in Fader, Hardie & Lee
(2005), "'Counting Your Customers' the Easy Way" — and this module does
nothing more than average it over acquisition cohorts.

Averaging is the whole trick. P(alive) is a per-customer posterior, so its
mean over a group of customers *is* the expected share of that group still
alive — no extra assumption, no contractual analogue, no hazard function.
Each cohort is measured at its own age (whole periods lived by the end of the
observation window), so the cohorts together trace a survival curve without
any of them being followed forward in time.

The cost of reading it as one curve: every point comes from a different set
of customers, so a cohort acquired during a bad campaign shows up as a dip in
survival that has nothing to do with age. It is a cross-section, not a
followed panel.
"""

from typing import TYPE_CHECKING, Protocol

import pandas as pd

from clvkit._result import Prediction, values_of
from clvkit.clv.bgnbd import BGNBD
from clvkit.customer_base import CustomerBase

if TYPE_CHECKING:
    from matplotlib.axes import Axes

_NOT_FITTED = "CohortSurvival is not fitted yet — call .fit(cb) first"


class TransactionModel(Protocol):
    """What `CohortSurvival` needs from a transaction-flow model.

    `fit` and `probability_alive` — a shorter list than `CLV` asks of the same
    models, because a survival curve never looks forward over a horizon. Each
    composition states its own minimum, which is what lets `MBGNBD` drop into
    both without inheriting anything.
    """

    def fit(self, cb: CustomerBase) -> "TransactionModel": ...

    def probability_alive(self, cb: CustomerBase | None = None) -> Prediction: ...


class SurvivalCurve:
    """The share of each acquisition cohort still alive, by cohort age.

    One row per cohort, indexed by the cohort's own period::

        age         whole periods the cohort had lived by the end of the
                    observation window
        customers   how many customers were acquired in that period
        survival    their mean fitted P(alive) — the expected share still active

    Cohorts run oldest-first down the index, so `survival` reads as a decay
    curve read right-to-left; `plot()` puts age on the x-axis instead.
    """

    def __init__(self, data: pd.DataFrame, *, period: str) -> None:
        self._data = data
        # The pandas offset alias cohorts are grained at ("M", "W", "Q"...).
        # `age` is counted in these, so it is the curve's unit as well.
        self.period = period

    def to_pandas(self) -> pd.DataFrame:
        """The curve as a DataFrame, one row per cohort."""
        return self._data.copy()

    def to_json(self) -> str:
        """Cohort-keyed JSON, the same shape as `to_pandas()`."""
        frame = self.to_pandas()
        frame.index = frame.index.astype(str)
        return frame.to_json(orient="index")

    def plot(self, ax: "Axes | None" = None, **kwargs) -> "Axes":
        """Draw survival against cohort age."""
        # Imported here, not at module scope, so `import clvkit` doesn't drag
        # in pyplot for anyone who only ever calls to_pandas().
        from clvkit.plotting import plot_survival_curve

        return plot_survival_curve(self, ax=ax, **kwargs)

    def __repr__(self) -> str:
        return (
            f"<SurvivalCurve {len(self._data)} {self.period!r} cohorts, "
            f"{int(self._data['customers'].sum())} customers>"
        )


class CohortSurvival:
    """Survival curves for a non-contractual base, from a fitted P(alive).

        >>> curve = CohortSurvival().fit(cb).predict()
        >>> curve.to_pandas()["survival"]
        >>> curve.plot()

    Defaults to `BGNBD()`; pass any model answering `fit` and
    `probability_alive` to swap it, e.g. `CohortSurvival(MBGNBD())`.
    """

    def __init__(self, transaction_model: TransactionModel | None = None) -> None:
        self.transaction_model: TransactionModel = transaction_model or BGNBD()

        self.time_unit_: str | None = None
        self._cb: CustomerBase | None = None

    def fit(self, cb: CustomerBase) -> "CohortSurvival":
        """Fit the transaction model on `cb` — everything else is aggregation."""
        if not isinstance(cb, CustomerBase):
            raise TypeError(
                f"CohortSurvival consumes a CustomerBase, got {type(cb).__name__}; "
                "build one with CustomerBase.from_transactions(...)"
            )

        self.transaction_model.fit(cb)
        self.time_unit_ = cb.time_unit
        self._cb = cb
        return self

    def predict(self, *, period: str | None = None) -> SurvivalCurve:
        """Aggregate the fitted P(alive) into a survival curve.

        `period` is the cohort grain, any pandas offset alias (`"M"`, `"Q"`,
        `"Y"`). It defaults to the base's own `time_unit`, which is the finest
        grain the RFM summary can resolve — coarsen it when that leaves too
        many cohorts to read (see `opinions.md`).
        """
        if self._cb is None:
            raise RuntimeError(_NOT_FITTED)

        cb = self._cb
        period = period or cb.time_unit
        _check_grain(cb, period)

        cohort = _cohort_of(cb, period)
        alive = values_of(self.transaction_model.probability_alive())

        # Sorted, so cohorts run oldest-first and the index is the curve's own
        # time axis reversed.
        grouped = alive.groupby(cohort.rename("cohort"), sort=True)
        survival = grouped.mean()
        cohorts = pd.PeriodIndex(survival.index)

        data = pd.DataFrame(
            {
                # Whole periods lived by the end of the observation window —
                # the same clock the cohorts themselves are bucketed on.
                "age": cb.observation_period_end.to_period(period).ordinal
                - cohorts.asi8,
                "customers": grouped.size().to_numpy(),
                "survival": survival.to_numpy(),
            },
            index=cohorts,
        )

        return SurvivalCurve(data, period=period)

    def __repr__(self) -> str:
        state = "unfitted" if self._cb is None else f"fitted on {self.time_unit_}"
        return f"<CohortSurvival {type(self.transaction_model).__name__} ({state})>"


def _cohort_of(cb: CustomerBase, period: str) -> pd.Series:
    """Each customer's acquisition period, recovered from the RFM summary.

    `CustomerBase` never stores a first-purchase date, but it doesn't have to:
    `T` is defined as the age of that purchase at `observation_period_end`, so
    subtracting it from the observation period's own ordinal lands back on the
    exact period the customer was acquired in. No transaction log needed, and
    no information the summary discarded.

    The arithmetic happens at the base's `collapse` grain, not its `time_unit`:
    those differ whenever the ruler is coarser than the events (the CDNOW fit
    collapses daily and reports weekly), which makes `T` fractional. Undoing
    that division first keeps the inversion exact instead of truncating a
    customer into the wrong period.

    Coarsening to `period` then places each acquisition period by where it
    *starts*: a week straddling a month boundary belongs to the month it began
    in. The summary cannot say which side of the boundary the purchase itself
    fell on, so this is a convention rather than a recovered fact — but it is
    the one that keeps a cohort label meaning "the bucket the customer's first
    period opened in", and pandas' own default (`how="E"`, by where the period
    ends) would file most of that week's customers under the wrong month.
    """
    summary = cb.to_pandas()
    obs_end = cb.observation_period_end.to_period(cb.collapse).ordinal
    age = summary["T"].to_numpy() * cb.periods_per_time_unit
    acquired = pd.PeriodIndex.from_ordinals(
        obs_end - age.round().astype("int64"), freq=cb.collapse
    )
    return pd.Series(acquired.asfreq(period, how="S"), index=summary.index)


def _check_grain(cb: CustomerBase, period: str) -> None:
    """Refuse a cohort grain finer than the base was summarised at.

    An acquisition is dated to within one `collapse` period — the grain events
    were bucketed at — and no finer. Asking for daily cohorts from a base
    collapsed weekly would place every customer on whichever day pandas happens
    to convert to: a precise-looking answer to a question the data cannot
    resolve. Note this is the *collapse* grain, not the ruler, so a base
    collapsed daily and reported weekly can still be cut into daily cohorts.
    """
    reference = cb.observation_period_end
    if _span(reference, period) < _span(reference, cb.collapse):
        raise ValueError(
            f"cohort period={period!r} is finer than the base's "
            f"collapse={cb.collapse!r}, which is as precisely as an "
            "acquisition can be dated"
        )


def _span(reference: pd.Timestamp, freq: str) -> pd.Timedelta:
    """How long the `freq` period containing `reference` lasts."""
    bucket = reference.to_period(freq)
    return bucket.end_time - bucket.start_time
