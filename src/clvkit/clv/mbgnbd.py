"""MBG/NBD — the never-returner variant of the BG/NBD, fit by maximum likelihood.

Implements Batislam, Denizel & Filiztekin (2007), "Empirical validation and
comparison of models for customer base analysis", *International Journal of
Research in Marketing* 24(3), 201-209. Equation numbers in the comments below
refer to that paper; it derives the model as a modification of Fader, Hardie &
Lee (2005), which `clvkit.clv.bgnbd` implements, and states only the formulas
that change.

The behavioural story (paper section 3) is the BG/NBD's, with one line added:

1. While active, a customer buys as a Poisson process with rate ``lambda``.
2. ``lambda`` is gamma-distributed across customers with shape ``r``, scale ``alpha``.
3. After *any* transaction a customer drops out with probability ``p`` —
   **including the very first one**, so a customer may die at time zero.
4. ``p`` is beta-distributed across customers with parameters ``a`` and ``b``.
5. ``lambda`` and ``p`` vary independently.

Point 3 is the whole difference, and it is not cosmetic. Under BG/NBD a
customer who never repeated cannot have dropped out — dropout requires a
transaction to follow, so P(alive | x = 0) = 1 identically. The paper's authors
found that this "results in treating the customers with zero repeat purchases
during the observation period as being active at time T and thereafter", which
on their grocery cohort (40-49% zero-repurchasers) badly understated the
never-returners. Adding a dropout opportunity immediately after the first
purchase gives that mass somewhere to go.

Mechanically the whole model is the BG/NBD with the dropout-count exponent
raised by one: every ``(1-p)^x`` becomes ``(1-p)^(x+1)``, so every ``B(a, b+x)``
becomes ``B(a, b+x+1)`` and the ``x > 0`` indicator on the dropout term
disappears. The code below is `bgnbd.py`'s sibling, deliberately duplicated
rather than shared: the two likelihoods differ by a single shifted argument,
and a parameterised "shift" would hide exactly the thing worth reading.
"""

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, minimize
from scipy.special import gammaln, hyp2f1

from clvkit._result import Prediction
from clvkit.customer_base import CustomerBase

_PARAM_NAMES = ("r", "alpha", "a", "b")
_REQUIRED_COLUMNS = ("frequency", "recency", "T")
_NOT_FITTED = "MBGNBD is not fitted yet — call .fit(cb) first"


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
    """Per-customer log-likelihood — Batislam et al. (2007) appendix A.

    The individual-level likelihood is eq. (1)::

        L(lambda, p | X = x, t_x, T) = (1-p)^(x+1) lambda^x e^(-lambda T)
                                     +  p (1-p)^x  lambda^x e^(-lambda t_x)

    — the customer survived all ``x + 1`` dropout opportunities (one after the
    initial purchase, one after each repeat), or survived ``x`` of them and
    died at the last. Compare Fader et al. (2005) eq. (3), which has
    ``(1-p)^x`` and ``p (1-p)^(x-1)`` and needs a ``x > 0`` indicator on the
    second term; here ``x = 0`` leaves a bare ``p``, the time-zero dropout.

    Integrating over the gamma prior on ``lambda`` and the beta prior on ``p``
    gives the appendix's aggregate form::

        B(a, b+x+1)   Gamma(r+x) alpha^r        B(a+1, b+x)   Gamma(r+x) alpha^r
        ----------- . ---------------------  +  ----------- . ---------------------
          B(a, b)     Gamma(r)(alpha+T)^(r+x)     B(a, b)     Gamma(r)(alpha+t_x)^(r+x)

    Evaluated as written this underflows for any realistic customer base, so we
    factor out what both terms share and add the rest in log space::

        A1 = lnGamma(r+x) - lnGamma(r) + r ln(alpha)
        A2 = lnGamma(a+b) + lnGamma(b+x+1) - lnGamma(b) - lnGamma(a+b+x+1)
        A3 = -(r+x) ln(alpha+T)
        A4 = ln(a) - ln(b+x) - (r+x) ln(alpha+t_x)

    with ``ln L = A1 + A2 + ln(exp(A3) + exp(A4))``. A4 carries ``a/(b+x)``
    because ``B(a+1, b+x)/B(a, b+x+1)`` simplifies to exactly that — which is
    also the dropout-odds factor in eq. (5). Unlike the BG/NBD's ``a/(b+x-1)``
    this has no pole at ``x = 0``, so no branch is needed: the dropout term is
    live for every customer.
    """
    x = frequency

    a1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
    a2 = gammaln(a + b) + gammaln(b + x + 1) - gammaln(b) - gammaln(a + b + x + 1)
    a3 = -(r + x) * np.log(alpha + T)
    a4 = np.log(a) - np.log(b + x) - (r + x) * np.log(alpha + recency)

    return a1 + a2 + np.logaddexp(a3, a4)


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
    """P(customer is still active at T) — Batislam et al. (2007) eq. (5).

    The "alive" term of the appendix likelihood over the whole of it, which
    after cancelling the shared factors is::

                                     1
        P(alive) = ------------------------------------------
                   1 + a/(b+x) . ((alpha+T)/(alpha+t_x))^(r+x)

    The paper writes the odds factor as ``Gamma(a+1)Gamma(b+x)/(Gamma(a)Gamma(b+x+1))``,
    which is ``a/(b+x)``.

    For ``x = 0`` (so ``t_x = 0``) this is ``1/(1 + a/b ((alpha+T)/alpha)^r)``,
    strictly below 1 and falling as ``T`` grows: the never-returner mass that
    the BG/NBD, where the same customer is alive with probability exactly 1,
    has nowhere to put.
    """
    x = frequency

    dropout_odds = (a / (b + x)) * ((alpha + T) / (alpha + recency)) ** (r + x)
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
    """E[Y(t) | X = x, t_x, T] — Batislam et al. (2007) appendix A.

    The expected number of transactions in the next ``t`` time units for a
    customer with history ``(x, t_x, T)``::

        (a+b+x)/(a-1) [1 - ((alpha+T)/(alpha+T+t))^(r+x) 2F1(r+x, b+x+1; a+b+x; t/(alpha+T+t))]
        ------------------------------------------------------------------------------------
                            1 + a/(b+x) . ((alpha+T)/(alpha+t_x))^(r+x)

    where 2F1 is the Gaussian hypergeometric function, arising from Euler's
    integral over the beta-distributed dropout probability. The denominator is
    the dropout odds of eq. (5), so — exactly as in the BG/NBD — this is an
    unconditional expectation scaled by P(alive).

    **The leading coefficient is not the one the paper prints.** Appendix A
    gives ``(b+x)/(a-1)``, by analogy with the ``b/(a-1)`` of the unconditional
    eq. (4). But conditioning divides by the alive branch, whose beta factor is
    ``B(a, b+x+1)``, so the coefficient is
    ``B(a-1, b+x+1)/B(a, b+x+1) = (a+b+x)/(a-1)`` — the BG/NBD's
    ``(a+b+x-1)/(a-1)`` with ``x`` raised by one, as every other term is. The
    printed version fails the model's own sanity check: at ``x = 0, t_x = 0,
    T = 0`` a customer is just a random draw from the population, so E[Y(t)]
    must collapse to eq. (4), and with ``(a+b+x)/(a-1)`` it does — the
    ``(a+b)/(a-1)`` meets ``P(alive) = b/(a+b)`` and leaves ``b/(a-1)``
    exactly. ``test_conditional_expectation_at_the_origin_reduces_to_equation_4``
    and the numerical-integration test beside it pin both readings.

    Note that ``a < 1`` (as in the paper's own estimates, where a = .40 to .49)
    makes ``(a+b+x)/(a-1)`` negative; the bracketed term is then negative too
    and the two signs cancel. That is the formula behaving as published, not a
    domain error.
    """
    x = frequency

    z = t / (alpha + T + t)
    hyp_term = hyp2f1(r + x, b + x + 1, a + b + x, z)
    unconditional = ((a + b + x) / (a - 1)) * (
        1 - ((alpha + T) / (alpha + T + t)) ** (r + x) * hyp_term
    )

    return unconditional * _probability_alive(
        r, alpha, a, b, frequency=frequency, recency=recency, T=T
    )


class MBGNBD:
    """The MBG/NBD transaction-flow model — BG/NBD's never-returner variant.

    The same three verbs as `BGNBD`, so it drops into the same places::

        model = MBGNBD().fit(cb)
        model.predict(t=12).to_pandas()      # expected purchases in the next 12 time units
        model.probability_alive().to_pandas()

    Reach for it over `BGNBD` when a large share of the base never repeated:
    BG/NBD calls every one of those customers alive with probability 1, while
    this model lets them have died right after their first purchase.

    Fitting is maximum likelihood — no sampler, no priors, no convergence
    diagnostics to read.
    """

    def __init__(self) -> None:
        self.params_: pd.Series | None = None
        self.log_likelihood_: float | None = None
        self.time_unit_: str | None = None
        self.collapse_: str | None = None
        self._cb: CustomerBase | None = None

    def fit(self, cb: CustomerBase) -> "MBGNBD":
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
        L-BFGS-B from its answer tightens the result.
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
                f"MBGNBD consumes a CustomerBase, got {type(cb).__name__}; "
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
            return "<MBGNBD (unfitted)>"
        params = ", ".join(f"{k}={v:.4g}" for k, v in self.params_.items())
        return f"<MBGNBD {params}>"
