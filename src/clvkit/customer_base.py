"""CustomerBase — the single input currency every clvkit model consumes."""

import warnings
from functools import partial
from typing import TYPE_CHECKING, Literal

import pandas as pd

from clvkit import _engine
from clvkit._display import describe, signature
from clvkit._engine import Engine

if TYPE_CHECKING:
    from types import ModuleType

    from dask.dataframe import DataFrame as DaskDataFrame

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
        engine: Engine = "pandas",
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
        # Which dataframe library did the summarising. Provenance, and the
        # reason `_events` may be absent — see .split().
        self.engine = engine

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
        engine: Engine = "pandas",
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

        `on_negative` shapes `monetary_value` and nothing else. Under every
        mode, a period containing at least one transaction with a non-negative
        amount is a purchase event, and `frequency`, `recency` and `T` are
        computed from those events — switching the policy never moves them.
        `"net"` (the default) nets each period's transactions and carries the
        result into `monetary_value` only when it stays positive; `"drop"`
        discards negative rows before summing spend; `"raise"` refuses a log
        containing any negative amount. Whenever a policy nets a period away,
        discards rows, or erases a customer whose every transaction is
        negative, a warning says so with a count — as does a row with no
        customer id, which cannot join a per-customer summary.

        ``engine="dask"`` takes a ``dask.dataframe.DataFrame`` instead of a
        pandas one and never holds the whole log in memory. The summary it
        returns is an ordinary pandas-backed ``CustomerBase`` — per-customer
        and small — but it does not retain the event frame, so ``.split()``
        refuses on it. It needs the optional ``dask`` extra.

        The two engines agree on every value. They differ in exactly two ways,
        both cosmetic and both tested: rows come back in shuffle order rather
        than first-appearance order, and the customer id keeps whatever dtype
        the input frame carried — which Dask itself may have rewritten to
        ``string[pyarrow]`` on the way in.
        """
        if on_negative not in _ON_NEGATIVE_MODES:
            raise ValueError(
                f"on_negative must be one of {_ON_NEGATIVE_MODES}, got {on_negative!r}"
            )
        dd = _engine.resolve(engine, transactions)

        # Naming the grain is consent to it; inheriting it silently is the trap.
        collapse_was_implicit = collapse is None
        collapse = time_unit if collapse_was_implicit else collapse
        ratio = _ruler_ratio(collapse, time_unit)

        has_monetary = amount_col is not None
        columns = [customer_id_col, datetime_col] + (
            [amount_col] if has_monetary else []
        )
        df = transactions[columns].copy()
        df[datetime_col] = _engine.to_datetime(
            df[datetime_col], dd, format=datetime_format
        )

        # A row with no customer id cannot join a per-customer summary, and
        # groupby would drop it without a word — in a weekly scoring job that
        # is rows silently missing from the scored table.
        null_id_rows, total_rows = _engine.counts(
            (df[customer_id_col].isna().sum(), df.shape[0]), dd
        )
        if null_id_rows:
            warnings.warn(
                f"{null_id_rows} of {total_rows} rows have no "
                f"{customer_id_col!r} and were dropped: the summary is "
                "per-customer, and a row with no customer id cannot join one.",
                UserWarning,
                stacklevel=2,
            )
            df = df[df[customer_id_col].notna()]

        last_transaction = df[datetime_col].max()
        if dd is not None:
            last_transaction = last_transaction.compute()

        if observation_period_end is None:
            observation_period_end = last_transaction
        else:
            observation_period_end = pd.to_datetime(
                observation_period_end, format=datetime_format
            )
            # It records when the data was pulled, so it may sit after the last
            # transaction. Before it, T is measured to a date the log outruns,
            # and every customer buying later gets recency > T: impossible, and
            # caught two calls away inside fit(). Refuse it here, where the
            # cause is still visible.
            if observation_period_end < last_transaction:
                raise ValueError(
                    f"observation_period_end "
                    f"({observation_period_end.date()}) is before the last "
                    f"transaction ({last_transaction.date()}). It records when "
                    f"the data was pulled; it does not truncate the log. To "
                    f"fit on a calibration window, use "
                    f"CustomerBase.split(calibration_period_end=...)."
                )

        if has_monetary and on_negative == "raise":
            any_negative = (df[amount_col] < 0).any()
            if dd is not None:
                any_negative = any_negative.compute()
            if any_negative:
                raise ValueError(
                    "negative amounts found in transactions; "
                    "pass on_negative='net' or 'drop' to handle them"
                )

        df["_bucket"] = _engine.buckets(df[datetime_col], collapse, dd)

        if collapse_was_implicit and _is_finer_than(_CANONICAL_COLLAPSE, collapse):
            cls._warn_if_the_grain_ate_purchases(df, customer_id_col, collapse, dd)

        # One event construction for both engines — every operation in it is
        # API-identical in pandas and Dask, so there is no second set of
        # policy semantics to drift from the first.
        if has_monetary:
            # The timing basis: a period counts as a purchase event when at
            # least one of its transactions has a non-negative amount, under
            # every on_negative mode. A period of pure refunds corroborates
            # no purchase, and a refund never erases the event it lands next
            # to — the policy shapes the spend column below, nothing here.
            vouched = df[df[amount_col] >= 0]
            timing = vouched[[customer_id_col, "_bucket"]].drop_duplicates()

            # The monetary basis: per-period spend under the policy. A NaN
            # after the merge marks an event the policy excluded from spend
            # while its timing stayed; `_summarize` reads it as exactly that.
            spend = vouched if on_negative == "drop" else df
            sums = (
                spend.groupby([customer_id_col, "_bucket"])[amount_col]
                .sum()
                .reset_index()
            )
            if on_negative == "net":
                sums = sums[sums[amount_col] > 0]
            events = timing.merge(sums, on=[customer_id_col, "_bucket"], how="left")

            cls._warn_what_the_policy_took(
                df,
                timing,
                spend,
                events,
                customer_id_col=customer_id_col,
                amount_col=amount_col,
                on_negative=on_negative,
                dd=dd,
            )
        else:
            events = df[[customer_id_col, "_bucket"]].drop_duplicates()

        if dd is not None:
            # The whole reduction stays lazy down to one row per customer, and
            # only that lands in memory. There is no event frame to keep.
            summary = cls._summarize_dask(
                events,
                customer_id_col,
                amount_col,
                has_monetary,
                observation_period_end,
                collapse,
                ratio,
            )
            events = None
        else:
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
            engine=engine,
        )

    @staticmethod
    def _warn_if_the_grain_ate_purchases(
        df: pd.DataFrame,
        customer_id_col: str,
        collapse: str,
        dd: "ModuleType | None" = None,
    ) -> None:
        """Say so, with a count, when a coarse grain merged real purchases.

        Silence here would be the whole problem: the summary still looks
        plausible, the fit still converges, and the loss falls hardest on the
        frequent buyers — the ones whose behaviour the model exists to capture.
        """
        deduplicated = df[[customer_id_col, "_bucket"]].drop_duplicates()
        total, kept = _engine.row_counts((df, deduplicated), dd)
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
    def _warn_what_the_policy_took(
        df: pd.DataFrame,
        timing: pd.DataFrame,
        spend: pd.DataFrame,
        events: pd.DataFrame,
        *,
        customer_id_col: str,
        amount_col: str,
        on_negative: OnNegative,
        dd: "ModuleType | None" = None,
    ) -> None:
        """Count whatever the monetary policy excluded, and say so.

        Silence here was the original bug: a summary missing customers still
        looks plausible, still fits, still scores — it is just missing
        customers. Same counted-warning pattern as the collapse warning above,
        and the counts are materialised in one batch so a Dask log is not
        read once per number.
        """
        customers, customers_kept, timing_events, no_spend_events, spend_rows, rows = (
            _engine.counts(
                (
                    df[customer_id_col].nunique(),
                    timing[customer_id_col].nunique(),
                    timing.shape[0],
                    events[amount_col].isna().sum(),
                    spend.shape[0],
                    df.shape[0],
                ),
                dd,
            )
        )

        erased = customers - customers_kept
        if erased:
            warnings.warn(
                f"on_negative={on_negative!r} dropped {erased} of {customers} "
                "customers entirely: none of their transactions has a "
                "non-negative amount, so no purchase event anchors their "
                "history.",
                UserWarning,
                stacklevel=3,
            )
        if on_negative == "net" and no_spend_events:
            warnings.warn(
                f"on_negative='net' netted {no_spend_events} of {timing_events} "
                "purchase events to a non-positive total. They keep their "
                "place in frequency, recency and T; they contribute no spend "
                "to monetary_value.",
                UserWarning,
                stacklevel=3,
            )
        if on_negative == "drop" and rows - spend_rows:
            warnings.warn(
                f"on_negative='drop' discarded {rows - spend_rows} of {rows} "
                "transactions with negative amounts from the spend basis. "
                "frequency, recency and T are unaffected.",
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

        Every figure here depends on one customer's events and nothing else,
        which is what lets the Dask engine run this exact function per
        partition rather than reimplementing it. `_bucket` arrives as a period
        from the pandas head and as an ordinal from the Dask one; both answer
        `astype("int64")` with the ordinal, which is all this reads.
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
            # A NaN amount is an event the monetary policy excluded from the
            # spend basis while its timing stayed. `mean` skips it, and a
            # customer with no qualifying repeat spend lands on 0.0, the same
            # figure a one-time buyer gets.
            monetary = (
                events_sorted[is_repeat]
                .groupby(customer_id_col, sort=False)[amount_col]
                .mean()
            )
            summary["monetary_value"] = monetary.reindex(summary.index).fillna(0.0)

        return summary

    @staticmethod
    def _summarize_dask(
        events: "DaskDataFrame",
        customer_id_col: str,
        amount_col: str | None,
        has_monetary: bool,
        observation_period_end: pd.Timestamp,
        collapse: str,
        ratio: int = 1,
    ) -> pd.DataFrame:
        """The same summary, reduced lazily and materialised once.

        Takes the per-bucket events `from_transactions` built — the event
        construction is shared with the pandas path, so the policy semantics
        cannot drift between engines — and the reduction is not a Dask
        reimplementation of `_summarize` either: it *is* `_summarize`.
        Shuffling on the customer id puts every one of a customer's events
        into a single partition, and every figure in the summary depends on
        one customer's events alone, so the per-partition answer is the
        global answer.

        An earlier draft did reimplement the reduction, with a Dask
        `groupby.transform` standing in for the `cumcount` that finds a
        customer's repeat purchases. It agreed with pandas on three
        partitions and silently stopped aligning on thirty-two.

        Peak memory is one partition of collapsed events plus the finished
        summary; the raw log is never held whole.
        """
        events = events.shuffle(on=customer_id_col)
        summarize = partial(
            CustomerBase._summarize,
            customer_id_col=customer_id_col,
            amount_col=amount_col,
            has_monetary=has_monetary,
            observation_period_end=observation_period_end,
            collapse=collapse,
            ratio=ratio,
        )
        # `_meta` is Dask's own zero-row copy of the frame, carrying the real
        # dtypes. Summarising it is how the output schema gets derived from the
        # function rather than restated next to it.
        return events.map_partitions(summarize, meta=summarize(events._meta)).compute()

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
            if self.engine == "dask":
                raise ValueError(
                    "split() is unavailable on a base summarised with "
                    "engine='dask': the per-bucket event frame it would need is "
                    "the memory that engine exists to avoid, so it is not kept. "
                    "Either re-summarise with engine='pandas', or summarise the "
                    "calibration and holdout windows as two separate bases."
                )
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
            # `fillna` as well as `reindex`: a customer whose every holdout
            # event was excluded from the spend basis has a NaN mean, not a
            # missing row.
            holdout["monetary_value_holdout"] = (
                grouped[self._amount_col].mean().reindex(holdout.index).fillna(0.0)
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
