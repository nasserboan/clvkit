"""CustomerBase — the single input currency every clvkit model consumes."""

import warnings
from typing import Literal

import pandas as pd

from clvkit._display import describe, signature

OnNegative = Literal["net", "drop", "raise"]
_ON_NEGATIVE_MODES: tuple[OnNegative, ...] = ("net", "drop", "raise")

# How many `collapse` periods make one `time_unit`, for the pairs where that
# number is a constant. Only calendar units of fixed length qualify: a week is
# always 7 days, but a month is 28-31, so there is no exact day->month ruler
# and we refuse rather than pick an average and call it precision.
_RULER_RATIO: dict[tuple[str, str], int] = {("D", "W"): 7}

# Same-day purchases are one shopping trip, and collapsing them is BTYD canon
# (the counting process wants separated events). Collapsing anything *coarser*
# than a day is a modelling choice with teeth, so it is the boundary we warn at.
_CANONICAL_COLLAPSE = "D"


def _is_finer_than(freq: str, other: str) -> bool:
    """Does one period of `freq` span less time than one of `other`?

    Compared at a fixed reference date, because months and years vary in
    length and a bare alias cannot be ordered without one.
    """
    reference = pd.Timestamp("2001-01-01")
    return _span(reference, freq) < _span(reference, other)


def _span(reference: pd.Timestamp, freq: str) -> pd.Timedelta:
    period = reference.to_period(freq)
    return period.end_time - period.start_time


def _ruler_ratio(collapse: str, time_unit: str) -> int:
    """How many `collapse` periods fit in one `time_unit`."""
    if collapse == time_unit:
        return 1
    if _is_finer_than(time_unit, collapse):
        raise ValueError(
            f"time_unit={time_unit!r} is finer than collapse={collapse!r}. "
            "The ruler cannot resolve more detail than the events kept — "
            f"either collapse at {time_unit!r} or report in {collapse!r}."
        )
    if (collapse, time_unit) not in _RULER_RATIO:
        raise ValueError(
            f"no exact conversion from collapse={collapse!r} to "
            f"time_unit={time_unit!r}: one {time_unit!r} is not a fixed number "
            f"of {collapse!r} periods. Supported pairs: "
            f"{sorted(_RULER_RATIO)}, or collapse == time_unit."
        )
    return _RULER_RATIO[(collapse, time_unit)]


class CustomerBase:
    """A self-describing RFM summary built from a raw transaction log.

    `frequency` is the BTYD *repeat* purchase count (total purchases minus
    one), not the raw transaction count — the #1 gotcha for anyone new to
    the BTYD/probability-model tradition.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        time_unit: str,
        observation_period_end: pd.Timestamp,
        has_monetary: bool,
        on_negative: OnNegative,
        collapse: str | None = None,
        events: pd.DataFrame | None = None,
        customer_id_col: str = "customer_id",
        amount_col: str | None = "amount",
    ) -> None:
        self._data = data
        self.time_unit = time_unit
        # The grain events were collapsed into, which is what `_events` is
        # bucketed by. Defaults to `time_unit` — the two were one knob until
        # the CDNOW fit needed them apart.
        self.collapse = time_unit if collapse is None else collapse
        # How many collapse periods make one time_unit — 1 whenever the two
        # agree. Provenance, not configuration: anything reading `T` back into
        # calendar terms needs it to undo the division `_summarize` applied.
        self.periods_per_time_unit = _ruler_ratio(self.collapse, time_unit)
        self.observation_period_end = observation_period_end
        self.has_monetary = has_monetary
        self.on_negative = on_negative
        # Collapsed per-bucket events, retained so .split() can recompute
        # calibration/holdout RFM without re-parsing the raw log.
        self._events = events
        self._customer_id_col = customer_id_col
        self._amount_col = amount_col

    @classmethod
    def from_transactions(
        cls,
        transactions: pd.DataFrame,
        customer_id_col: str = "customer_id",
        datetime_col: str = "date",
        amount_col: str | None = "amount",
        *,
        time_unit: str = "D",
        collapse: str | None = None,
        observation_period_end: str | pd.Timestamp | None = None,
        datetime_format: str | None = None,
        on_negative: OnNegative = "net",
    ) -> "CustomerBase":
        """Summarise a raw transaction log into RFM plus provenance.

        `time_unit` is the ruler: the unit `recency` and `T` are reported in.
        `collapse` is the event grain: transactions falling in the same
        `collapse` period become one purchase, because the counting process
        the BTYD models assume wants separated events.

        They default to the same thing, which is what makes `time_unit="W"`
        alone a trap — it does not merely re-scale the ruler, it deletes every
        second purchase inside a week, and it deletes most from your heaviest
        buyers. Pass both to reproduce the published CDNOW fit, which collapses
        at the data's daily resolution and reports time in weeks::

            CustomerBase.from_transactions(log, time_unit="W", collapse="D")
        """
        if on_negative not in _ON_NEGATIVE_MODES:
            raise ValueError(
                f"on_negative must be one of {_ON_NEGATIVE_MODES}, got {on_negative!r}"
            )

        # Naming the grain is consent to it; inheriting it silently is the trap.
        collapse_was_implicit = collapse is None
        collapse = time_unit if collapse_was_implicit else collapse
        ratio = _ruler_ratio(collapse, time_unit)

        has_monetary = amount_col is not None
        columns = [customer_id_col, datetime_col] + (
            [amount_col] if has_monetary else []
        )
        df = transactions[columns].copy()
        df[datetime_col] = pd.to_datetime(df[datetime_col], format=datetime_format)

        if observation_period_end is None:
            observation_period_end = df[datetime_col].max()
        else:
            observation_period_end = pd.to_datetime(
                observation_period_end, format=datetime_format
            )

        if has_monetary:
            if on_negative == "raise" and (df[amount_col] < 0).any():
                raise ValueError(
                    "negative amounts found in transactions; "
                    "pass on_negative='net' or 'drop' to handle them"
                )
            if on_negative == "drop":
                df = df[df[amount_col] >= 0]

        df["_bucket"] = df[datetime_col].dt.to_period(collapse)

        if collapse_was_implicit and _is_finer_than(_CANONICAL_COLLAPSE, collapse):
            cls._warn_if_the_grain_ate_purchases(df, customer_id_col, collapse)

        if has_monetary:
            events = df.groupby(
                [customer_id_col, "_bucket"], sort=False, as_index=False
            )[amount_col].sum()
            if on_negative == "net":
                events = events[events[amount_col] > 0]
        else:
            events = df[[customer_id_col, "_bucket"]].drop_duplicates()

        summary = cls._summarize(
            events,
            customer_id_col,
            amount_col,
            has_monetary,
            observation_period_end,
            collapse,
            ratio,
        )

        return cls(
            summary,
            time_unit=time_unit,
            collapse=collapse,
            observation_period_end=observation_period_end,
            has_monetary=has_monetary,
            on_negative=on_negative,
            events=events,
            customer_id_col=customer_id_col,
            amount_col=amount_col,
        )

    @staticmethod
    def _warn_if_the_grain_ate_purchases(
        df: pd.DataFrame, customer_id_col: str, collapse: str
    ) -> None:
        """Say so, with a count, when a coarse grain merged real purchases.

        Silence here would be the whole problem: the summary still looks
        plausible, the fit still converges, and the loss falls hardest on the
        frequent buyers — the ones whose behaviour the model exists to capture.
        """
        total = len(df)
        kept = len(df[[customer_id_col, "_bucket"]].drop_duplicates())
        absorbed = total - kept
        if absorbed == 0:
            return

        # Only offer the daily grain when this ruler can actually be reached
        # from it — telling someone to pass collapse='D' with time_unit='M'
        # would send them into a ValueError.
        if (_CANONICAL_COLLAPSE, collapse) in _RULER_RATIO:
            remedy = (
                f"Pass collapse='{_CANONICAL_COLLAPSE}' to keep them and still "
                f"report time in {collapse!r}, or pass collapse={collapse!r} to "
                "say you meant this."
            )
        else:
            remedy = (
                f"There is no exact {_CANONICAL_COLLAPSE!r}-to-{collapse!r} "
                "ruler, so the only way to keep them is a finer time_unit. "
                f"Pass collapse={collapse!r} to say you meant this."
            )

        warnings.warn(
            f"time_unit={collapse!r} collapsed {absorbed} of {total} transactions "
            "into earlier purchases in the same period. This biases the fit "
            f"downward, and it takes the most from your most frequent buyers. {remedy}",
            UserWarning,
            stacklevel=3,
        )

    @staticmethod
    def _summarize(
        events: pd.DataFrame,
        customer_id_col: str,
        amount_col: str | None,
        has_monetary: bool,
        observation_period_end: pd.Timestamp,
        collapse: str,
        ratio: int = 1,
    ) -> pd.DataFrame:
        """Turn collapsed per-bucket events into an RFM summary.

        Ages are counted in `collapse` periods — the grain `_bucket` lives at —
        then divided by `ratio` to land in `time_unit`. When the two grains
        agree `ratio` is 1 and the ages stay exact integers.
        """
        grouped = events.groupby(customer_id_col, sort=False)["_bucket"]
        first = grouped.min()
        last = grouped.max()
        count = grouped.count()

        obs_end_ordinal = observation_period_end.to_period(collapse).ordinal
        first_ordinal = first.astype("int64")
        last_ordinal = last.astype("int64")

        recency = last_ordinal - first_ordinal
        age = obs_end_ordinal - first_ordinal
        if ratio != 1:
            recency = recency / ratio
            age = age / ratio

        summary = pd.DataFrame(
            {
                "frequency": count - 1,
                "recency": recency,
                "T": age,
            }
        )
        summary.index.name = customer_id_col

        if has_monetary:
            events_sorted = events.sort_values([customer_id_col, "_bucket"])
            is_repeat = (
                events_sorted.groupby(customer_id_col, sort=False).cumcount() > 0
            )
            monetary = (
                events_sorted[is_repeat]
                .groupby(customer_id_col, sort=False)[amount_col]
                .mean()
            )
            summary["monetary_value"] = monetary.reindex(summary.index, fill_value=0.0)

        return summary

    def split(
        self,
        *,
        calibration_period_end: str | pd.Timestamp,
        observation_period_end: str | pd.Timestamp | None = None,
    ) -> tuple["CustomerBase", pd.DataFrame]:
        """Split into a calibration CustomerBase and a holdout frame.

        Calibration RFM is computed against ``calibration_period_end`` exactly
        as ``from_transactions`` would; the holdout frame carries
        ``frequency_holdout`` (repeat purchases in the holdout window),
        ``monetary_value_holdout`` (mean holdout spend, only when monetary),
        and ``duration_holdout`` (holdout length in ``time_unit``). Customers
        whose first purchase falls in the holdout window have no calibration
        history and are excluded from both outputs.
        """
        if self._events is None:
            raise ValueError(
                "split() requires a CustomerBase built via from_transactions"
            )

        grain = self.collapse
        ratio = self.periods_per_time_unit
        cid = self._customer_id_col
        cal_end = pd.to_datetime(calibration_period_end)
        obs_end = (
            self.observation_period_end
            if observation_period_end is None
            else pd.to_datetime(observation_period_end)
        )

        cal_bucket = cal_end.to_period(grain)
        obs_bucket = obs_end.to_period(grain)

        events = self._events
        buckets = events["_bucket"]
        cal_events = events[buckets <= cal_bucket]
        holdout_events = events[(buckets > cal_bucket) & (buckets <= obs_bucket)]

        cal_summary = self._summarize(
            cal_events, cid, self._amount_col, self.has_monetary, cal_end, grain, ratio
        )
        calibration = CustomerBase(
            cal_summary,
            time_unit=self.time_unit,
            collapse=self.collapse,
            observation_period_end=cal_end,
            has_monetary=self.has_monetary,
            on_negative=self.on_negative,
            events=cal_events,
            customer_id_col=cid,
            amount_col=self._amount_col,
        )

        # Holdout stats align to the calibration index — this drops any
        # customer born in the holdout window (no calibration events).
        holdout = pd.DataFrame(index=cal_summary.index)
        grouped = holdout_events.groupby(cid, sort=False)
        holdout["frequency_holdout"] = (
            grouped["_bucket"].count().reindex(holdout.index, fill_value=0)
        )
        if self.has_monetary:
            holdout["monetary_value_holdout"] = (
                grouped[self._amount_col].mean().reindex(holdout.index, fill_value=0.0)
            )
        duration = obs_bucket.ordinal - cal_bucket.ordinal
        holdout["duration_holdout"] = duration if ratio == 1 else duration / ratio

        return calibration, holdout

    def to_pandas(self) -> pd.DataFrame:
        """Return the RFM summary as a DataFrame indexed by customer_id."""
        return self._data.copy()

    def __repr__(self) -> str:
        # One line, because this is what shows up inside a list, a dict, a
        # traceback, or a failed assertion. `print(cb)` gets the long form.
        return signature(self)

    def __str__(self) -> str:
        return describe(self)
