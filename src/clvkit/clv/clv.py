"""CLV — a transaction model times a monetary model, discounted.

Implements the composition of Fader, Hardie & Lee (2005), "RFM and CLV: Using
Iso-value Curves for Customer Base Analysis" §2. Section and equation numbers
below refer to that paper.

The paper's central factorisation is equation (1):

    CLV = margin x revenue/transaction x DET

where DET is the number of *discounted expected transactions* — the present
value of a customer's future purchase stream. Its discrete form (p. 10) is

              n   E(Y(t) | X = x, t_x, T) - E(Y(t-1) | X = x, t_x, T)
    DET  =   sum  ---------------------------------------------------
             t=1                    (1 + d)^t

i.e. the expected purchases *within* each future period, each discounted back
from the end of that period at rate `d`. The paper goes on to derive a
continuous-time, infinite-horizon closed form (eq. 2), but that expression is
specific to the Pareto/NBD's parameters. The discrete sum above needs only
`E(Y(t))` — the one thing every transaction model in this library already
answers via `predict(t)` — so BG/NBD, MBG/NBD, or anything else with that verb
composes here without a line of model-specific code.

What makes the factorisation legal is assumption (iii) of §2.1: average
transaction value is independent of the transaction process. Without it the
product of two separately-correct expectations is not the expectation of the
product. §2.2 does not take that on faith, and neither does this module — `fit`
runs the paper's own assessment on your base and says so when it fails. See
`clvkit.clv.independence`.
"""

import warnings
from typing import TYPE_CHECKING, Protocol

import numpy as np
import pandas as pd

from clvkit._result import Prediction, values_of
from clvkit.clv.bgnbd import BGNBD
from clvkit.clv.gamma_gamma import GammaGamma
from clvkit.clv.independence import (
    IndependenceCheck,
    MonetaryIndependenceWarning,
    check_monetary_independence,
)
from clvkit.customer_base import CustomerBase

if TYPE_CHECKING:
    from matplotlib.axes import Axes

_NOT_FITTED = "CLV is not fitted yet — call .fit(cb) first"


class TransactionModel(Protocol):
    """What `CLV` needs from a transaction-flow model: `fit` and `predict(t)`.

    Structural, so `MBGNBD` and any future variant compose without inheriting
    anything — the DET sum only ever asks for cumulative expected purchases.
    """

    def fit(self, cb: CustomerBase) -> "TransactionModel": ...

    def predict(self, t: float, cb: CustomerBase | None = None) -> Prediction: ...


class MonetaryModel(Protocol):
    """What `CLV` needs from a spend model: `fit` and a horizon-free `predict`."""

    def fit(self, cb: CustomerBase) -> "MonetaryModel": ...

    def predict(self, cb: CustomerBase | None = None) -> Prediction: ...


class CLVResult:
    """Discounted expected residual lifetime value, per customer.

    Carries every factor of equation (1), not just their product, so the number
    can be interrogated: a CLV that looks wrong is usually a DET that looks
    wrong or a spend estimate that looks wrong, and separating them is the
    difference between a diagnosis and a shrug.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        horizon: int,
        discount_rate: float,
        margin: float,
        time_unit: str,
    ) -> None:
        self._data = data
        self.horizon = horizon
        self.discount_rate = discount_rate
        self.margin = margin
        self.time_unit = time_unit

    @property
    def description(self) -> str:
        """Unit-aware axis label — the horizon is not implied by the numbers."""
        return f"Lifetime value over {self.horizon} {self.time_unit}"

    def to_pandas(self) -> pd.DataFrame:
        """The four factors of equation (1), indexed by customer_id."""
        return self._data.copy()

    def to_json(self) -> str:
        """Customer-keyed JSON, the same shape as `to_pandas()`."""
        return self.to_pandas().to_json(orient="index")

    def plot(self, ax: "Axes | None" = None, **kwargs) -> "Axes":
        """Draw the distribution of lifetime value across the customer base."""
        from clvkit.plotting import plot_prediction

        # Lifetime value is "one predicted quantity per customer", which is
        # exactly what `Prediction` is for — so the clv column draws through
        # the same histogram as every other per-customer prediction rather
        # than growing a near-identical second one in plotting.py.
        return plot_prediction(
            Prediction(self._data["clv"], name="clv", description=self.description),
            ax=ax,
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"<CLVResult horizon={self.horizon} {self.time_unit}, "
            f"discount_rate={self.discount_rate:g}, margin={self.margin:g}, "
            f"{len(self._data)} customers>"
        )


class CLV:
    """The one-shot composition — a transaction model and a monetary model.

        >>> clv = CLV().fit(cb).predict(horizon=12, discount_rate=0.01)
        >>> clv.to_pandas()["clv"]

    Defaults to `BGNBD()` for the transaction flow and `GammaGamma()` for
    spend; pass either to swap it, e.g. `CLV(transaction_model=MBGNBD())`.
    """

    def __init__(
        self,
        transaction_model: TransactionModel | None = None,
        monetary_model: MonetaryModel | None = None,
        *,
        check_independence: bool = True,
    ) -> None:
        self.transaction_model: TransactionModel = transaction_model or BGNBD()
        self.monetary_model: MonetaryModel = monetary_model or GammaGamma()
        # The §2.2 assessment runs at fit time by default. Turn it off only if
        # you already know spend and buying rate are related in your business
        # and have decided to live with the bias.
        self.check_independence = check_independence

        self.time_unit_: str | None = None
        self._cb: CustomerBase | None = None
        self._independence: IndependenceCheck | None = None

    def fit(self, cb: CustomerBase) -> "CLV":
        """Fit both sub-models on `cb`, and assess the assumption joining them.

        Lifetime value needs amounts, so a `CustomerBase` built without an
        `amount_col` is refused here rather than several frames deep inside the
        monetary model.
        """
        if not isinstance(cb, CustomerBase):
            raise TypeError(
                f"CLV consumes a CustomerBase, got {type(cb).__name__}; "
                "build one with CustomerBase.from_transactions(...)"
            )
        if not cb.has_monetary:
            raise ValueError(
                "CLV is a monetary quantity, so it needs a CustomerBase built "
                "with an amount_col; this one carries only transaction timing. "
                "Rebuild it with CustomerBase.from_transactions(df, "
                "amount_col='amount'), or use the transaction model on its own "
                "if you only have timing data."
            )

        self.transaction_model.fit(cb)
        self.monetary_model.fit(cb)
        self.time_unit_ = cb.time_unit
        self._cb = cb
        self._independence = None

        if self.check_independence:
            self._warn_if_independence_is_violated()
        return self

    def predict(
        self,
        *,
        horizon: int,
        discount_rate: float = 0.0,
        margin: float = 1.0,
        cb: CustomerBase | None = None,
    ) -> CLVResult:
        """Discounted expected residual lifetime value over `horizon` periods.

        - `horizon` is a whole number of the fitted base's own `time_unit`; the
          DET sum runs one term per period, so half a period has no increment.
        - `discount_rate` is `d` in the DET sum: the rate **per `time_unit`**,
          not per year. At weekly granularity 0.01 is 1% a week, ~68% a year.
        - `margin` is the contribution margin of equation (1). The default of
          1.0 makes the result revenue-based lifetime value; pass your gross
          margin to get contribution-based CLV (see `opinions.md`).
        """
        if int(horizon) != horizon or horizon < 1:
            raise ValueError(
                f"horizon must be a positive whole number of periods, got {horizon!r}"
            )
        if discount_rate <= -1:
            raise ValueError(
                f"discount_rate must be greater than -1, got {discount_rate!r}"
            )
        if margin <= 0:
            raise ValueError(f"margin must be positive, got {margin!r}")
        if self._cb is None:
            raise RuntimeError(_NOT_FITTED)

        horizon = int(horizon)
        cumulative = self._cumulative_expected_purchases(horizon, cb)
        det = _discounted_expected_transactions(cumulative, discount_rate)
        spend = values_of(self.monetary_model.predict(cb))

        data = pd.DataFrame(
            {
                "expected_purchases": cumulative[-1],
                "discounted_expected_transactions": det,
                "expected_spend": spend,
                # Equation (1), assembled.
                "clv": margin * spend * det,
            }
        )

        return CLVResult(
            data,
            horizon=horizon,
            discount_rate=discount_rate,
            margin=margin,
            time_unit=self.time_unit_ or "",
        )

    def independence_check(self, cb: CustomerBase | None = None) -> IndependenceCheck:
        """The §2.2 assessment of the assumption this composition rests on.

        Defaults to the base the model was fitted on. Read `.holds()` for the
        verdict and `.plot()` for the paper's Figure 4 on your own data.
        """
        if cb is None:
            if self._cb is None:
                raise RuntimeError(_NOT_FITTED)
            if self._independence is None:
                self._independence = check_monetary_independence(self._cb)
            return self._independence
        return check_monetary_independence(cb)

    def _cumulative_expected_purchases(
        self, horizon: int, cb: CustomerBase | None
    ) -> list[pd.Series]:
        """E(Y(t) | X = x, t_x, T) for t = 1..horizon, per customer.

        One `predict(t)` call per period. That is the whole coupling to the
        transaction model — nothing here knows it is a BG/NBD.
        """
        return [
            values_of(self.transaction_model.predict(float(t), cb))
            for t in range(1, horizon + 1)
        ]

    def _warn_if_independence_is_violated(self) -> None:
        try:
            check = self.independence_check()
        except ValueError:
            # A base too small or too uniform to assess (no repeat buyers, or
            # no variation in frequency) is not evidence of a violation, and
            # the sub-models will complain about it far more usefully.
            return

        if not check.holds():
            warnings.warn(
                "monetary value does not look independent of the transaction "
                f"process on this customer base (Spearman rho={check.spearman_rho:.2f}, "
                f"eta2={check.eta_squared:.2f} over {check.n_customers} repeat buyers). "
                "CLV multiplies expected spend by expected transactions, which "
                "assumes exactly that independence (Fader, Hardie & Lee 2005, "
                "§2.1), so this composition may be biased. Inspect it with "
                ".independence_check().plot(), or pass check_independence=False "
                "to silence this.",
                MonetaryIndependenceWarning,
                stacklevel=3,
            )

    def __repr__(self) -> str:
        state = "unfitted" if self._cb is None else f"fitted on {self.time_unit_}"
        return (
            f"<CLV {type(self.transaction_model).__name__} x "
            f"{type(self.monetary_model).__name__} ({state})>"
        )


def _discounted_expected_transactions(
    cumulative: list[pd.Series], discount_rate: float
) -> pd.Series:
    """The DET sum of p. 10, from cumulative expected purchases.

    `cumulative[i]` is E(Y(i+1)); differencing turns the cumulative curve into
    the per-period increments the sum is written over, and E(Y(0)) = 0 supplies
    the first one. Each increment is discounted from the end of its period.

    With `discount_rate = 0` every weight is 1 and the sum telescopes back to
    E(Y(horizon)) exactly — the undiscounted forecast, as it should.
    """
    increments = np.diff(np.vstack([np.zeros(len(cumulative[0])), *cumulative]), axis=0)
    weights = 1.0 / (1.0 + discount_rate) ** np.arange(1, len(cumulative) + 1)

    return pd.Series(weights @ increments, index=cumulative[0].index)
