"""Is monetary value independent of the transaction process? — the CLV premise.

Composing a transaction model with a monetary model into a single lifetime
value is only legitimate under one assumption. Fader, Hardie & Lee (2005),
"RFM and CLV: Using Iso-value Curves for Customer Base Analysis", state it as
assumption (iii) of §2.1:

    The distribution of average transaction values across customers is
    independent of the transaction process.

That assumption is what lets equation (1) factor,

    CLV = margin x revenue/transaction x DET

into a spend term and a transaction term that can be modelled — and fitted —
separately. Break it and the product of two separately-correct expectations is
not the expectation of the product, so the composed CLV is biased.

The authors do not merely assert the assumption; §2.2 assesses it, on CDNOW,
with three pieces of evidence:

1. The simple correlation between average transaction value and number of
   transactions across repeat buyers (0.11).
2. The observation that a single outlier drives most of it (drop the customer
   with 21 transactions averaging $300 and it falls to 0.06, p = 0.08).
3. Figure 4, box-and-whisker plots of average transaction value by number of
   repeat purchases, read as: "the variation within each number-of-transactions
   group dominates the between-group variation."

This module reproduces that assessment on *your* customer base, so the
assumption behind your CLV is a number you can look at rather than a sentence
in a docstring. Point (1) is `pearson_r`; point (2) is why `spearman_rho` is
reported alongside it (a rank correlation does the paper's manual outlier
removal automatically, and `exclude=` does it by hand); point (3) is
`eta_squared` and `plot()`.
"""

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from clvkit.customer_base import CustomerBase

# See opinions.md, "When is monetary independence violated?". Both are
# deliberately loose: the check exists to catch a customer base where spend and
# buying rate move together enough to bias CLV, not to police mild correlation
# the papers themselves shrug at (CDNOW passes both with room to spare).
DEFAULT_MAX_CORRELATION = 0.30
DEFAULT_MAX_ETA_SQUARED = 0.25


class MonetaryIndependenceWarning(UserWarning):
    """Raised by `CLV.fit` when the composition's premise looks violated.

    A warning rather than an error on purpose: the paper's own CDNOW data shows
    a slight positive correlation and the authors still proceed. Only the
    analyst can decide whether the relationship in *their* data is substantial
    enough to abandon the composition.
    """


class IndependenceCheck:
    """§2.2's assessment of the independence assumption, run on one base.

    Constructed by `check_monetary_independence`, not directly::

        check = check_monetary_independence(cb)
        check.holds()        # the verdict, at documented thresholds
        check.plot()         # the paper's Figure 4, on your data
        check.to_pandas()    # every statistic behind the verdict
    """

    def __init__(
        self,
        frequency: pd.Series,
        monetary_value: pd.Series,
        *,
        pearson: tuple[float, float],
        spearman: tuple[float, float],
        eta_squared: float,
    ) -> None:
        # The (x, m_x) pairs the statistics were computed on — kept so plot()
        # can draw the distribution the summary statistics compress.
        self._frequency = frequency
        self._monetary_value = monetary_value

        self.n_customers = len(frequency)
        #: §2.2's headline number: the simple correlation the paper reports.
        self.pearson_r, self.pearson_p = (float(value) for value in pearson)
        #: The outlier-robust counterpart — see the module docstring.
        self.spearman_rho, self.spearman_p = (float(value) for value in spearman)
        #: Share of monetary-value variance explained by frequency (Figure 4).
        self.eta_squared = float(eta_squared)

    def holds(
        self,
        *,
        max_correlation: float = DEFAULT_MAX_CORRELATION,
        max_eta_squared: float = DEFAULT_MAX_ETA_SQUARED,
    ) -> bool:
        """Does the independence assumption survive at these thresholds?

        `True` when the rank correlation between frequency and spend stays
        under `max_correlation` *and* frequency explains less than
        `max_eta_squared` of the variance in spend. Both defaults are opinions,
        not canon — see `opinions.md` — and both are arguments precisely so
        that a business with a known frequency/spend relationship can set its
        own bar instead of arguing with ours.
        """
        return (
            abs(self.spearman_rho) < max_correlation
            and self.eta_squared < max_eta_squared
        )

    def to_pandas(self) -> pd.DataFrame:
        """Every statistic behind the verdict, one row each, indexed by name."""
        statistics = {
            "n_customers": float(self.n_customers),
            "pearson_r": self.pearson_r,
            "pearson_p": self.pearson_p,
            "spearman_rho": self.spearman_rho,
            "spearman_p": self.spearman_p,
            "eta_squared": self.eta_squared,
        }
        return pd.DataFrame(
            {"value": list(statistics.values())},
            index=pd.Index(statistics, name="statistic"),
        )

    def to_json(self) -> str:
        """The statistics as JSON, the same shape as `to_pandas()`."""
        return self.to_pandas().to_json(orient="index")

    def plot(self, ax: "Axes | None" = None, **kwargs) -> "Axes":
        """Draw the paper's Figure 4 on this customer base."""
        from clvkit.plotting import plot_independence

        return plot_independence(self, ax=ax, **kwargs)

    def grouped_spend(self, max_frequency: int = 7) -> dict[str, np.ndarray]:
        """Average transaction value grouped by repeat-purchase count.

        The data behind Figure 4. Frequencies at or above `max_frequency` are
        pooled into one trailing group, because the tail of a real customer
        base is a long run of groups with one or two members each — which is
        noise, not a distribution anyone can read.
        """
        if max_frequency < 1:
            raise ValueError(f"max_frequency must be >= 1, got {max_frequency!r}")

        # Frequency is a count, so it labels as an integer even though it is
        # carried as a float for the correlation maths.
        capped = self._frequency.clip(upper=max_frequency).astype(int)
        return {
            f"{level}+" if level == max_frequency else str(level): (
                self._monetary_value[capped == level].to_numpy()
            )
            for level in sorted(capped.unique())
        }

    def __repr__(self) -> str:
        verdict = "holds" if self.holds() else "violated"
        return (
            f"<IndependenceCheck {verdict}: r={self.pearson_r:.3f}, "
            f"rho={self.spearman_rho:.3f}, eta2={self.eta_squared:.3f}, "
            f"n={self.n_customers}>"
        )


def check_monetary_independence(
    cb: "CustomerBase", *, exclude: Iterable = ()
) -> IndependenceCheck:
    """Assess §2.2's independence assumption on `cb`.

    Computed over repeat buyers only: a one-time buyer has no observed average
    transaction value, which is why the paper's statistics cover 946 of CDNOW's
    2,357 customers rather than all of them.

    `exclude` drops customer ids before computing anything — the paper's own
    move in §2.2, where removing a single outlier takes the correlation from
    0.11 to 0.06. Use it to ask "is this relationship real, or is it one
    customer?".
    """
    if not cb.has_monetary:
        raise ValueError(
            "the independence assumption is about spend, so checking it needs "
            "a CustomerBase built with an amount_col; this one has no "
            "monetary value"
        )

    summary = cb.to_pandas()
    excluded = list(exclude)
    unknown = [customer for customer in excluded if customer not in summary.index]
    if unknown:
        # Silently ignoring a mistyped id would answer a different question
        # than the one asked, and look exactly like the right answer.
        raise KeyError(f"exclude names customers not in this base: {unknown}")

    repeat = summary[summary["frequency"] > 0].drop(index=excluded, errors="ignore")
    if repeat.empty:
        raise ValueError(
            "the independence check compares spend against buying rate across "
            "repeat buyers; this customer base has none"
        )

    frequency = repeat["frequency"].astype(float)
    monetary_value = repeat["monetary_value"].astype(float)
    if frequency.nunique() < 2:
        raise ValueError(
            "every repeat buyer has the same frequency, so spend cannot be "
            "correlated with it; the independence check needs a customer base "
            "with variation in frequency"
        )

    return IndependenceCheck(
        frequency,
        monetary_value,
        pearson=stats.pearsonr(frequency, monetary_value),
        spearman=stats.spearmanr(frequency, monetary_value),
        eta_squared=_eta_squared(frequency, monetary_value),
    )


def _eta_squared(frequency: pd.Series, monetary_value: pd.Series) -> float:
    """Share of variance in spend explained by frequency — Figure 4 as a number.

    The paper reads Figure 4 as "the variation within each number-of-
    transactions group dominates the between-group variation". That is exactly
    a one-way variance decomposition of spend by frequency level: eta squared is
    the between-group sum of squares over the total, so small values *are* the
    paper's conclusion, stated numerically.
    """
    grand_mean = monetary_value.mean()
    groups = monetary_value.groupby(frequency, sort=False)

    between = (groups.count() * (groups.mean() - grand_mean) ** 2).sum()
    total = ((monetary_value - grand_mean) ** 2).sum()

    # A base where every repeat buyer spent identically has no variance to
    # apportion; nothing is explained, so nothing is violated.
    return 0.0 if total == 0 else float(between / total)
