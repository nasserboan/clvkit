"""BG/NBD — the beta-geometric/NBD transaction-flow model, fit by maximum likelihood.

Implements Fader, Hardie & Lee (2005), "'Counting Your Customers' the Easy Way:
An Alternative to the Pareto/NBD Model", *Marketing Science* 24(2), 275-284.
Section and equation numbers in the comments below refer to that paper.

The behavioural story (paper section 3):

1. While active, a customer buys as a Poisson process with rate ``lambda``.
2. ``lambda`` is gamma-distributed across customers with shape ``r``, scale ``alpha``.
3. After *any* transaction a customer drops out with probability ``p``, so the
   dropout occasion is (shifted) geometric. This is the one departure from the
   Pareto/NBD, and the reason the whole model stays in closed form.
4. ``p`` is beta-distributed across customers with parameters ``a`` and ``b``.
5. ``lambda`` and ``p`` vary independently.

Everything the model can say follows from those five lines, so the code below
is short: one likelihood, one conditional expectation, one P(alive).
"""

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, minimize
from scipy.special import gammaln, hyp2f1, logsumexp

from clvkit._result import Prediction
from clvkit.customer_base import CustomerBase

_PARAM_NAMES = ("r", "alpha", "a", "b")
_REQUIRED_COLUMNS = ("frequency", "recency", "T")
_NOT_FITTED = "BGNBD is not fitted yet — call .fit(cb) first"


def _rfm_arrays(summary: pd.DataFrame) -> dict[str, np.ndarray]:
    """The three sufficient statistics as float arrays.

    Keyed to match the keyword parameters of the functions below, so callers
    can splat this straight in.
    """
    return {name: summary[name].to_numpy(dtype=float) for name in _REQUIRED_COLUMNS}


def _log_likelihood(
    r: float,
    alpha: float,
    a: float,
    b: float,
    *,
    frequency: np.ndarray,
    recency: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    """Per-customer log-likelihood — Fader et al. (2005) eq. (6).

    The paper writes the likelihood for a customer summarised by
    ``(X = x, t_x, T)`` as the sum of two terms::

        B(a, b+x)   Gamma(r+x) alpha^r                 B(a+1, b+x-1)   Gamma(r+x) alpha^r
        --------- . ---------------------  +  d(x>0) . ------------- . ---------------------
         B(a, b)    Gamma(r)(alpha+T)^(r+x)               B(a, b)      Gamma(r)(alpha+t_x)^(r+x)

    the first being "still alive at T", the second "dropped out at t_x".

    Evaluated as written this underflows for any realistic customer base, so we
    factor out the parts both terms share and add the rest in log space. That
    gives the four pieces of the worksheet the paper itself publishes in
    Figure 1 (the Excel screenshot in section 7):

        A1 = lnGamma(r+x) - lnGamma(r) + r ln(alpha)
        A2 = lnGamma(a+b) + lnGamma(b+x) - lnGamma(b) - lnGamma(a+b+x)   [= ln B(a,b+x)/B(a,b)]
        A3 = -(r+x) ln(alpha+T)
        A4 = ln(a) - ln(b+x-1) - (r+x) ln(alpha+t_x)                     [only when x > 0]

    with ``ln L = A1 + A2 + ln(exp(A3) + d(x>0) exp(A4))``. A4 carries the
    ``a/(b+x-1)`` factor because ``B(a+1, b+x-1)/B(a, b+x)`` simplifies to
    exactly that, which is also why eq. (10)'s denominator has the same ratio.
    """
    x = frequency

    a1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
    a2 = gammaln(a + b) + gammaln(b + x) - gammaln(b) - gammaln(a + b + x)
    a3 = -(r + x) * np.log(alpha + T)

    # b + x - 1 is only ever evaluated where x > 0, but numpy computes both
    # branches of the where, so clamp x away from the b - 1 pole first.
    safe_x = np.maximum(x, 1.0)
    a4 = np.where(
        x > 0,
        np.log(a) - np.log(b + safe_x - 1) - (r + x) * np.log(alpha + recency),
        -np.inf,
    )

    return a1 + a2 + logsumexp(np.stack([a3, a4]), axis=0)


def _probability_alive(
    r: float,
    alpha: float,
    a: float,
    b: float,
    *,
    frequency: np.ndarray,
    recency: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    """P(customer is still active at T) — the denominator of eq. (10).

    Mixing the individual-level P(active) of appendix eq. (A2) over the gamma
    and beta priors leaves the "alive" term of eq. (6) over the whole of
    eq. (6), which after cancelling the shared factors is::

                                          1
        P(alive) = ---------------------------------------------
                   1 + d(x>0) . a/(b+x-1) . ((alpha+T)/(alpha+t_x))^(r+x)

    For ``x = 0`` the delta term drops out and this is exactly 1: the paper's
    assumption that every customer is active at the start of the observation
    period means a customer who has never transacted cannot yet have dropped out.
    """
    x = frequency
    safe_x = np.maximum(x, 1.0)

    dropout_odds = np.where(
        x > 0,
        (a / (b + safe_x - 1)) * ((alpha + T) / (alpha + recency)) ** (r + x),
        0.0,
    )
    return 1.0 / (1.0 + dropout_odds)


def _conditional_expected_purchases(
    r: float,
    alpha: float,
    a: float,
    b: float,
    t: float,
    *,
    frequency: np.ndarray,
    recency: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    """E[Y(t) | X = x, t_x, T] — Fader et al. (2005) eq. (10).

    The expected number of transactions in the next ``t`` time units for a
    customer with history ``(x, t_x, T)``::

        (a+b+x-1)/(a-1) [1 - ((alpha+T)/(alpha+T+t))^(r+x) 2F1(r+x, b+x; a+b+x-1; t/(alpha+T+t))]
        ---------------------------------------------------------------------------------------
                          1 + d(x>0) . a/(b+x-1) . ((alpha+T)/(alpha+t_x))^(r+x)

    where 2F1 is the Gaussian hypergeometric function. It appears because the
    integral over the beta-distributed dropout probability (appendix eq. (A8))
    is Euler's integral representation of 2F1. The paper stresses that this is
    a *single* evaluation per customer, used only after the likelihood has
    already been maximised — it never enters the optimisation.

    The denominator is the same dropout-odds expression as ``_probability_alive``:
    the conditional expectation is the unconditional one scaled by P(alive).

    Note that ``a < 1`` (as on CDNOW, where a = .793) makes ``(a+b+x-1)/(a-1)``
    negative; the bracketed term is then negative too and the two signs cancel.
    That is the formula behaving as published, not a domain error.
    """
    x = frequency

    z = t / (alpha + T + t)
    hyp_term = hyp2f1(r + x, b + x, a + b + x - 1, z)
    unconditional = ((a + b + x - 1) / (a - 1)) * (
        1 - ((alpha + T) / (alpha + T + t)) ** (r + x) * hyp_term
    )

    return unconditional * _probability_alive(
        r, alpha, a, b, frequency=frequency, recency=recency, T=T
    )


class BGNBD:
    """The BG/NBD transaction-flow model.

    Three verbs, no aliases::

        model = BGNBD().fit(cb)
        model.predict(t=12).to_pandas()      # expected purchases in the next 12 time units
        model.probability_alive().to_pandas()

    Fitting is maximum likelihood — no sampler, no priors, no convergence
    diagnostics to read.
    """

    def __init__(self) -> None:
        self.params_: pd.Series | None = None
        self.log_likelihood_: float | None = None
        self.time_unit_: str | None = None
        self.collapse_: str | None = None
        self._cb: CustomerBase | None = None

    def fit(self, cb: CustomerBase) -> "BGNBD":
        """Estimate ``(r, alpha, a, b)`` by maximum likelihood on ``cb``.

        The optimiser works on ``log`` parameters so the search is
        unconstrained while the parameters stay strictly positive, which is
        what the gamma and beta priors of section 3 require.
        """
        rfm = _rfm_arrays(self._validated_summary(cb))

        def negative_log_likelihood(log_params: np.ndarray) -> float:
            r, alpha, a, b = np.exp(log_params)
            total = _log_likelihood(r, alpha, a, b, **rfm).sum()
            # A non-finite objective is a step the optimiser must back away
            # from, not a crash.
            return -total if np.isfinite(total) else np.inf

        result = self._minimize(negative_log_likelihood)

        self.params_ = pd.Series(np.exp(result.x), index=_PARAM_NAMES)
        self.log_likelihood_ = float(-result.fun)
        self.time_unit_ = cb.time_unit
        self.collapse_ = cb.collapse
        self._cb = cb
        return self

    @staticmethod
    def _minimize(objective) -> OptimizeResult:
        """Maximise the likelihood, then polish.

        Nelder-Mead from all-ones is robust on this surface but stops loose;
        L-BFGS-B from its answer tightens the result to the precision the
        published-estimate golden test needs.
        """
        coarse = minimize(
            objective,
            np.zeros(len(_PARAM_NAMES)),
            method="Nelder-Mead",
            options={"maxiter": 10_000, "maxfev": 10_000, "xatol": 1e-8, "fatol": 1e-8},
        )
        return minimize(
            objective, coarse.x, method="L-BFGS-B", options={"maxiter": 10_000}
        )

    def predict(self, t: float, cb: CustomerBase | None = None) -> Prediction:
        """Expected purchases in the next ``t`` time units, per customer.

        ``t`` is in the fitted base's own ``time_unit``. Pass ``cb`` to score a
        different customer base than the one fitted on — it must carry the same
        ``time_unit``, or the answer would silently be in the wrong units.
        """
        if t <= 0:
            raise ValueError(f"t must be positive, got {t!r}")

        r, alpha, a, b = self._fitted_params()
        summary = self._scoring_summary(cb)

        values = _conditional_expected_purchases(
            r, alpha, a, b, float(t), **_rfm_arrays(summary)
        )

        return Prediction(
            pd.Series(values, index=summary.index),
            name="expected_purchases",
            description=f"Expected purchases in the next {t:g} {self.time_unit_}",
        )

    def probability_alive(self, cb: CustomerBase | None = None) -> Prediction:
        """P(each customer is still active) at the base's observation period end."""
        r, alpha, a, b = self._fitted_params()
        summary = self._scoring_summary(cb)

        values = _probability_alive(r, alpha, a, b, **_rfm_arrays(summary))

        return Prediction(
            pd.Series(values, index=summary.index),
            name="probability_alive",
            description="Probability alive",
            plot_data=summary[["frequency", "recency", "T"]],
            plot_time_unit=self.time_unit_ or "",
        )

    def _fitted_params(self) -> tuple[float, float, float, float]:
        if self.params_ is None:
            raise RuntimeError(_NOT_FITTED)
        return tuple(self.params_[name] for name in _PARAM_NAMES)  # type: ignore[return-value]

    def _scoring_summary(self, cb: CustomerBase | None) -> pd.DataFrame:
        """The RFM table to score, defaulting to the one fitted on."""
        if cb is None:
            if self._cb is None:
                raise RuntimeError(_NOT_FITTED)
            return self._cb.to_pandas()

        summary = self._validated_summary(cb)
        # The provenance guard the CustomerBase carries exists for exactly this:
        # a model fit in weeks scoring a base measured in days is not an error
        # numpy can see, it is just a wrong answer.
        if cb.time_unit != self.time_unit_:
            raise ValueError(
                f"model was fit on a CustomerBase with time_unit={self.time_unit_!r}, "
                f"but was given one with time_unit={cb.time_unit!r}"
            )
        # Same ruler, different event grain, is the subtler version of the same
        # mistake: the summaries agree on units and disagree on how many
        # purchases each customer made.
        if cb.collapse != self.collapse_:
            raise ValueError(
                f"model was fit on a CustomerBase with collapse={self.collapse_!r}, "
                f"but was given one with collapse={cb.collapse!r}"
            )
        return summary

    @staticmethod
    def _validated_summary(cb: CustomerBase) -> pd.DataFrame:
        """Check the input contract before the likelihood turns violations into NaNs."""
        if not isinstance(cb, CustomerBase):
            raise TypeError(
                f"BGNBD consumes a CustomerBase, got {type(cb).__name__}; "
                "build one with CustomerBase.from_transactions(...)"
            )

        summary = cb.to_pandas()
        missing = [col for col in _REQUIRED_COLUMNS if col not in summary.columns]
        if missing:
            raise ValueError(f"CustomerBase is missing required columns: {missing}")
        if summary.empty:
            raise ValueError("CustomerBase is empty — nothing to fit")
        if (summary["recency"] > summary["T"]).any():
            raise ValueError("CustomerBase has customers with recency > T")
        if (summary[list(_REQUIRED_COLUMNS)] < 0).to_numpy().any():
            raise ValueError("CustomerBase has negative frequency, recency or T")

        return summary

    def __repr__(self) -> str:
        if self.params_ is None:
            return "<BGNBD (unfitted)>"
        params = ", ".join(f"{k}={v:.4g}" for k, v in self.params_.items())
        return f"<BGNBD {params}>"
