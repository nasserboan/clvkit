"""The Gamma-Gamma model of monetary value — expected spend per transaction.

Implements Fader & Hardie (2013), "The Gamma-Gamma Model of Monetary Value"
(<http://brucehardie.com/notes/025/>), the derivation note behind Fader,
Hardie & Lee (2005). Equation numbers in this module refer to that note.

The model rests on three assumptions (note §1):

1. The monetary value of a customer's given transaction varies randomly
   around their average transaction value.
2. Average transaction values vary across customers but do not vary over
   time for any given individual — which is why `predict` takes no horizon.
3. The distribution of average transaction values across customers is
   **independent of the transaction process** — so this model can be fit and
   composed with a BG/NBD-style transaction model without a joint likelihood.

Assumption 3 is worth checking on your own data before trusting a CLV built
on it: if `frequency` and `monetary_value` are strongly correlated in your
customer base, spend and buying rate are not independent and the composed
CLV will be biased.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from clvkit._result import Prediction
from clvkit.customer_base import CustomerBase

# `gamma` rather than `γ`: this is an index into a Series a caller may key by
# hand, and a name you cannot type is a name nobody uses.
_PARAM_NAMES = ("p", "q", "gamma")


class GammaGamma:
    """Spend per transaction, estimated by maximum likelihood.

    Individual transaction values are gamma(p, ν); ν itself is gamma(q, γ)
    across customers (note §2), which makes a customer's unobserved mean
    transaction value ζ inverse-gamma with shape q and scale pγ.

        >>> gg = GammaGamma().fit(cb)
        >>> gg.predict().to_pandas()
    """

    def __init__(self) -> None:
        self.p: float | None = None
        self.q: float | None = None
        self.gamma: float | None = None
        self._cb: CustomerBase | None = None

    @property
    def params_(self) -> pd.Series | None:
        """The fitted ``(p, q, γ)``, or ``None`` before fitting.

        A view over `p`, `q` and `gamma` rather than a fourth copy of them —
        the three named floats stay the source of truth, because the formulae
        in this module read them individually. It exists so that a caller
        holding a transaction model and a monetary model can read both the
        same way; `BGNBD` has carried `params_` since it was written, and
        anything wanting to render or compare the two had to special-case
        this class for want of it.
        """
        if self.p is None or self.q is None or self.gamma is None:
            return None
        return pd.Series([self.p, self.q, self.gamma], index=_PARAM_NAMES, dtype=float)

    def fit(self, cb: CustomerBase) -> "GammaGamma":
        """Estimate (p, q, γ) by maximising the sample log-likelihood (6).

        Only customers with at least one repeat transaction carry information
        about spend; one-time buyers have no observed average and contribute
        nothing to the likelihood (note §3, "Parameter Estimation").
        """
        frequency, monetary_value = self._repeat_buyers(cb)

        # Optimise on the log scale: (p, q, γ) are all strictly positive, and
        # this keeps the optimiser inside the support without a constraint set.
        # Start from (p, q) = (1, 2) and γ = mean(z̄), which by (3) puts the
        # initial population mean at the observed one — so the search begins
        # at the scale of the data whether spend is in cents or in euros.
        result = minimize(
            lambda log_params: (
                -self._log_likelihood(*np.exp(log_params), frequency, monetary_value)
            ),
            x0=np.log([1.0, 2.0, monetary_value.mean()]),
            method="L-BFGS-B",
            options={"maxiter": 10_000},
        )
        if not result.success:
            raise RuntimeError(
                f"Gamma-Gamma likelihood did not converge: {result.message}"
            )

        self.p, self.q, self.gamma = (float(v) for v in np.exp(result.x))
        self._cb = cb
        return self

    def predict(self, cb: CustomerBase | None = None) -> Prediction:
        """Expected spend per transaction, E(Z | p, q, γ; z̄, x), per customer.

        There is no horizon argument: assumption 2 makes a customer's average
        transaction value constant over time, so this is the same number
        whether you look one week or one year ahead.
        """
        p, q, gamma, fitted_on = self._fitted()
        summary = self._require_monetary(fitted_on if cb is None else cb).to_pandas()

        x = summary["frequency"].to_numpy(dtype=float)
        z_bar = summary["monetary_value"].to_numpy(dtype=float)

        # Equation (5): a weighted average of the population mean pγ/(q-1)
        # and the customer's own observed average z̄, with the weight shifting
        # towards z̄ as the number of observed transactions x grows. For a
        # one-time buyer (x = 0, z̄ = 0) it collapses to the population mean —
        # equation (3) — which is exactly the right prior-only answer.
        expected_spend = p * (gamma + x * z_bar) / (p * x + q - 1)

        return Prediction(
            pd.Series(expected_spend, index=summary.index),
            name="expected_spend",
            description="expected spend per transaction",
        )

    def population_mean(self) -> float:
        """E(Z | p, q, γ) = pγ/(q − 1) — mean spend across the population (3)."""
        p, q, gamma, _ = self._fitted()
        return p * gamma / (q - 1)

    @staticmethod
    def _log_likelihood(
        p: float,
        q: float,
        gamma: float,
        frequency: np.ndarray,
        monetary_value: np.ndarray,
    ) -> float:
        """Sample log-likelihood (6): Σ ln f(z̄ᵢ | p, q, γ; xᵢ).

        The summand is the log of equation (1a),

            f(z̄ | p, q, γ; x) = Γ(px+q)/(Γ(px)Γ(q)) · z̄^(px-1) x^(px) γ^q
                                / (γ + xz̄)^(px+q)

        i.e. a beta distribution of the second kind. (1a) itself overflows for
        large z̄ and x, but its logarithm — what we actually evaluate — does
        not, which is why the note calibrates on this form (note §3, fn. 1).
        """
        if p <= 0 or q <= 0 or gamma <= 0:
            return -np.inf

        px = p * frequency
        return float(
            np.sum(
                gammaln(px + q)
                - gammaln(px)
                - gammaln(q)
                + q * np.log(gamma)
                + (px - 1) * np.log(monetary_value)
                + px * np.log(frequency)
                - (px + q) * np.log(gamma + frequency * monetary_value)
            )
        )

    @staticmethod
    def _repeat_buyers(cb: CustomerBase) -> tuple[np.ndarray, np.ndarray]:
        """The (x, z̄) pairs the likelihood is evaluated on."""
        summary = GammaGamma._require_monetary(cb).to_pandas()
        repeat = summary[summary["frequency"] > 0]
        if repeat.empty:
            raise ValueError(
                "Gamma-Gamma needs customers with at least one repeat "
                "transaction; this customer base has none"
            )
        if (repeat["monetary_value"] <= 0).any():
            # ln z̄ is undefined at zero, and a customer who nets nothing has
            # no average spend to model. on_negative="net" is the only mode
            # that drops non-positive buckets outright.
            raise ValueError(
                "Gamma-Gamma requires a positive monetary_value for every "
                "repeat buyer; rebuild the CustomerBase with on_negative="
                "'net' so that non-positive transactions are not counted "
                "as purchase events"
            )
        return (
            repeat["frequency"].to_numpy(dtype=float),
            repeat["monetary_value"].to_numpy(dtype=float),
        )

    @staticmethod
    def _require_monetary(cb: CustomerBase) -> CustomerBase:
        if not cb.has_monetary:
            raise ValueError(
                "Gamma-Gamma models spend, so it needs a CustomerBase built "
                "with an amount_col; this one has no monetary value"
            )
        return cb

    def _fitted(self) -> tuple[float, float, float, CustomerBase]:
        """The estimates and the base they came from — `fit` sets all four."""
        if self.p is None or self.q is None or self.gamma is None or self._cb is None:
            raise RuntimeError("GammaGamma is not fitted; call fit(cb) first")
        return self.p, self.q, self.gamma, self._cb

    def __repr__(self) -> str:
        # Reads the sentinels, not `_fitted()`, which raises — an unfitted
        # model is exactly the thing you print while working out why.
        params = self.params_
        if params is None:
            return "<GammaGamma (unfitted)>"
        return (
            "<GammaGamma " + ", ".join(f"{k}={v:.4g}" for k, v in params.items()) + ">"
        )
