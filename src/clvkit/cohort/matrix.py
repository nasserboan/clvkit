"""The descriptive cohort matrix — a pivot, not a model.

This is the one part of clvkit with no likelihood, no fitting, and no
parameters: it reports what the transaction log already says. Every cell is
an observed count or an observed sum, so nothing here needs the BTYD canon
to be believed.

It reads the **raw transaction log**, not a :class:`~clvkit.CustomerBase`.
The RFM summary keeps only first/last purchase timing per customer, and a
cohort matrix needs the whole per-period activity pattern — a customer who
bought in months 0, 3 and 7 is indistinguishable in RFM from one who bought
in months 0, 5 and 7. That information is discarded by the summary, so the
pivot has to start from the log.
"""

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from clvkit import _engine
from clvkit._engine import Engine

if TYPE_CHECKING:
    from types import ModuleType

    from dask.dataframe import DataFrame as DaskDataFrame
    from matplotlib.axes import Axes

Metric = Literal["retention", "revenue"]
_METRICS: tuple[Metric, ...] = ("retention", "revenue")

#: Column-0 of the matrix is the cohort's own period, so this is the label
#: for "how many periods after acquisition".
_PERIOD_NUMBER = "period_number"

_EMPTY_LOG = "cannot build a CohortMatrix from an empty transaction log"


def _cohort_grid(
    part: pd.DataFrame, customer_id_col: str, value_col: str
) -> pd.DataFrame:
    """One partition's ``(cohort, period_number, value)`` rows.

    Module level so Dask can pickle it, and correct per partition only because
    the caller shuffled on the customer id first — see ``_cells_dask``.
    """
    cohort = part["_bucket"].groupby(part[customer_id_col]).transform("min")
    return pd.DataFrame(
        {
            "_cohort": cohort,
            _PERIOD_NUMBER: part["_bucket"] - cohort,
            "_value": part[value_col],
        }
    )


class CohortMatrix:
    """A cohort-by-period matrix of retention or revenue, and its heatmap.

    Rows are acquisition cohorts (the period of a customer's first observed
    purchase); columns are whole periods elapsed since that cohort's own
    period. Cell values are:

    ``metric="retention"``
        the number of distinct customers of that cohort active in that
        period (a customer buying three times in one period counts once).
    ``metric="revenue"``
        the total amount those customers spent in that period.

    Recent cohorts have been observed for fewer periods, so the matrix is a
    triangle: cells past a cohort's observation window are ``NaN``, never
    ``0``. A zero means "observed, nobody bought"; missing means "not yet
    knowable". Because of those ``NaN``s the retention matrix is float-typed
    even though it counts customers.

        >>> matrix = CohortMatrix.from_transactions(df, period="M")
        >>> matrix.to_pandas(relative=True)   # retention rates
        >>> matrix.plot()                     # the heatmap

    There is no ``fit()``: a pivot has no state to estimate.
    """

    def __init__(self, matrix: pd.DataFrame, *, metric: Metric, period: str) -> None:
        self._matrix = matrix
        self.metric = metric
        # The pandas offset alias the log was bucketed with ("M", "W", "Q"...).
        self.period = period

    @classmethod
    def from_transactions(
        cls,
        transactions: pd.DataFrame,
        *,
        period: str = "M",
        metric: Metric = "retention",
        customer_id_col: str = "customer_id",
        datetime_col: str = "date",
        amount_col: str | None = "amount",
        datetime_format: str | None = None,
        engine: Engine = "pandas",
    ) -> "CohortMatrix":
        """Pivot a raw transaction log into a cohort-by-period matrix.

        ``period`` is any pandas offset alias (``"M"``, ``"W"``, ``"Q"``,
        ``"Y"``); it sets both the cohort grain and the column grain, since a
        cohort matrix only makes sense when both use the same clock.

        The observation window ends at the last transaction in the log, and
        that is what makes a cell unobserved rather than zero.

        ``engine="dask"`` takes a ``dask.dataframe.DataFrame`` and reduces the
        log without holding it in memory; the matrix it returns is the same
        pandas one, since a cohort-by-period grid is small whatever built it.
        It needs the optional ``dask`` extra.
        """
        if metric not in _METRICS:
            raise ValueError(f"metric must be one of {_METRICS}, got {metric!r}")
        dd = _engine.resolve(engine, transactions)

        # Amounts are only touched for revenue, so a timing-only log needs no
        # amount column at all when the metric is retention.
        needs_amount = metric == "revenue"
        if needs_amount and amount_col is None:
            raise ValueError("metric='revenue' needs an amount_col, got None")

        columns = [customer_id_col, datetime_col] + (
            [amount_col] if needs_amount else []
        )
        df = transactions[columns].copy()
        # Under Dask, emptiness costs a pass over the log to establish, so it
        # is caught in `_cells_dask` where a pass is already being made.
        if dd is None and df.empty:
            raise ValueError(_EMPTY_LOG)
        df[datetime_col] = _engine.to_datetime(
            df[datetime_col], dd, format=datetime_format
        )

        if dd is None:
            bucket = _engine.buckets(df[datetime_col], period, dd)
            cohort = bucket.groupby(df[customer_id_col]).transform("min")
            # Period dtype casts to its integer ordinal, so the difference is a
            # whole number of periods regardless of calendar length.
            offset = bucket.astype("int64") - cohort.astype("int64")

            grid = pd.DataFrame(
                {"_cohort": cohort, _PERIOD_NUMBER: offset, "_id": df[customer_id_col]}
            )
            if needs_amount:
                grid["_amount"] = df[amount_col].to_numpy()
                cells = grid.groupby(["_cohort", _PERIOD_NUMBER])["_amount"].sum()
            else:
                cells = grid.groupby(["_cohort", _PERIOD_NUMBER])["_id"].nunique()
            cells = cells.unstack()
            end = bucket.max()
        else:
            cells, end = cls._cells_dask(
                df, dd, period, needs_amount, customer_id_col, datetime_col, amount_col
            )

        matrix = cls._to_triangle(cells, end)
        return cls(matrix, metric=metric, period=period)

    @staticmethod
    def _cells_dask(
        df: "DaskDataFrame",
        dd: "ModuleType",
        period: str,
        needs_amount: bool,
        customer_id_col: str,
        datetime_col: str,
        amount_col: str | None,
    ) -> tuple[pd.DataFrame, pd.Period]:
        """The same pivot, reduced lazily; returns the small pandas grid.

        Cohorts travel as period *ordinals* rather than periods, because Dask's
        period-dtype support is thin — they become a ``PeriodIndex`` again on
        the way out, so the matrix is indistinguishable from the pandas one.

        A customer's cohort is the minimum over that customer's own rows, so
        the frame is shuffled on the customer id first and the cohort assigned
        one partition at a time, in pandas. Dask's ``groupby.transform`` looks
        like it would do this in one line; it returns a result that stops
        aligning row-for-row once there are enough partitions to shuffle.
        """
        bucket = _engine.buckets(df[datetime_col], period, dd)
        df = df.assign(_bucket=bucket).shuffle(on=customer_id_col)

        value_col = amount_col if needs_amount else customer_id_col
        meta = _cohort_grid(df._meta, customer_id_col, value_col)
        grid = df.map_partitions(_cohort_grid, customer_id_col, value_col, meta=meta)
        grouped = grid.groupby(["_cohort", _PERIOD_NUMBER])["_value"]
        cells = (grouped.sum() if needs_amount else grouped.nunique()).compute()

        if cells.empty:
            raise ValueError(_EMPTY_LOG)

        # Every row's bucket is its cohort plus its period number, so the last
        # bucket in the log is already in this index — no second pass for it.
        end_ordinal = int(
            (cells.index.get_level_values(0) + cells.index.get_level_values(1)).max()
        )

        cells = cells.unstack()
        cells.index = pd.PeriodIndex.from_ordinals(cells.index, freq=period)
        return cells, pd.PeriodIndex.from_ordinals([end_ordinal], freq=period)[0]

    @staticmethod
    def _to_triangle(
        matrix: pd.DataFrame, observation_period_end: pd.Period
    ) -> pd.DataFrame:
        """Square off the pivot, separating observed zeros from unobserved cells.

        ``unstack`` leaves a hole wherever a (cohort, period) pair had no
        transactions — but that hole means two different things depending on
        whether the period had happened yet by the end of the log.
        """
        cohorts = pd.PeriodIndex(matrix.index).sort_values()
        # The oldest cohort has been observed the longest, and that sets the
        # width of the whole matrix.
        ages = observation_period_end.ordinal - cohorts.asi8
        columns = pd.RangeIndex(0, int(ages.max()) + 1, name=_PERIOD_NUMBER)

        matrix = matrix.reindex(index=cohorts, columns=columns).astype(float)
        matrix.index.name = "cohort"

        # A cell is observed when the cohort had lived that many periods by the
        # end of the log. Observed-but-absent means nobody bought (0);
        # unobserved stays missing, so a young cohort never reads as churned.
        observed = ages[:, None] >= columns.to_numpy()[None, :]
        values = matrix.to_numpy()
        return pd.DataFrame(
            np.where(observed, np.nan_to_num(values, nan=0.0), np.nan),
            index=matrix.index,
            columns=matrix.columns,
        )

    def to_pandas(self, *, relative: bool = False) -> pd.DataFrame:
        """The matrix as a DataFrame: cohorts down, periods across.

        ``relative=True`` divides every cohort by its own period-0 value —
        retention *rates* rather than counts, or a revenue index rather than
        currency. That is the comparable view, since cohorts differ in size.
        """
        matrix = self._matrix.copy()
        if relative:
            matrix = matrix.div(matrix[0], axis=0)
        return matrix

    def to_json(self, *, relative: bool = False) -> str:
        """Cohort-keyed JSON, the same shape as ``to_pandas()``.

        Unobserved cells serialise as ``null``, keeping the triangle's shape
        legible outside pandas.
        """
        matrix = self.to_pandas(relative=relative)
        matrix.index = matrix.index.astype(str)
        matrix.columns = matrix.columns.astype(str)
        return matrix.to_json(orient="index")

    def plot(
        self, ax: "Axes | None" = None, *, relative: bool = True, **kwargs
    ) -> "Axes":
        """Draw the cohort heatmap — the standard triangle chart.

        Defaults to the relative view because that is what the chart is read
        for: how fast each cohort decays, independent of how big it was.
        """
        # Imported here, not at module scope, so `import clvkit` doesn't drag
        # in pyplot for anyone who only ever calls to_pandas().
        from clvkit.plotting import plot_cohort_matrix

        return plot_cohort_matrix(self, ax=ax, relative=relative, **kwargs)

    def __repr__(self) -> str:
        rows, cols = self._matrix.shape
        return (
            f"<CohortMatrix {self.metric!r} "
            f"{rows} cohorts x {cols} {self.period!r} periods>"
        )
