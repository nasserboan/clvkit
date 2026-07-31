"""Pinpoint tests for the MBG/NBD special-function math.

The sibling of ``test_bgnbd_likelihood.py``, and held to the same standard:
every expected value comes from an independent source — a closed form the
paper states, beta/gamma functions evaluated in natural space, or numerical
integration of the individual-level model — never from re-running the code
path under test.

The MBG/NBD is one shifted beta parameter away from the BG/NBD, which is
exactly why it deserves its own pinpoint suite: an off-by-one in ``b + x``
would still fit, still predict, and still be wrong.

Sources: Batislam, Denizel & Filiztekin (2007), *IJRM* 24(3), 201-209 — eq. (1)
(individual-level likelihood), eq. (3) (individual-level E[X(t)]), eq. (4)
(E[X(t)] for a random customer), eq. (5) (P(active)), and the unnumbered
aggregate likelihood and E[Y(t)] of Appendix A.
"""

import numpy as np
from scipy.integrate import quad
from scipy.special import beta as beta_fn
from scipy.special import gamma as gamma_fn
from scipy.special import hyp2f1

from clvkit.clv.mbgnbd import (
    _conditional_expected_purchases,
    _log_likelihood,
    _probability_alive,
)


def test_zero_repeat_customer_likelihood_carries_the_never_returner_mass():
    # Batislam et al. (2007) eq. (1) with x = 0 and t_x = 0 is
    #     (1 - p) e^{-lambda T}  +  p,
    # so integrating over the gamma and beta priors leaves the closed form
    #     B(a, b+1)/B(a, b) (alpha/(alpha+T))^r  +  B(a+1, b)/B(a, b)
    #   = [ b (alpha/(alpha+T))^r  +  a ] / (a + b).
    # The lone `a/(a+b)` is the never-returner mass: under BG/NBD this term
    # does not exist and the likelihood is (alpha/(alpha+T))^r exactly.
    r, alpha, a, b = 0.5, 6.0, 1.5, 3.0
    T = np.array([10.0, 25.0])

    log_likelihood = _log_likelihood(
        r, alpha, a, b, frequency=np.zeros(2), recency=np.zeros(2), T=T
    )

    expected = np.log((b * (alpha / (alpha + T)) ** r + a) / (a + b))
    np.testing.assert_allclose(log_likelihood, expected, rtol=1e-13)
    # Strictly above the BG/NBD value for the same customer: the extra mass.
    assert (log_likelihood > r * np.log(alpha / (alpha + T))).all()


def test_repeat_customer_likelihood_matches_the_appendix_in_natural_space():
    # The implementation works in log space with a log-sum-exp; this recomputes
    # the Appendix A aggregate likelihood literally, in natural space, from
    # beta and gamma functions. Two independent arithmetic paths, one answer.
    # Note the beta arguments: (a, b+x+1) and (a+1, b+x) — one higher in the
    # second slot than the BG/NBD's (a, b+x) and (a+1, b+x-1).
    r, alpha, a, b = 0.243, 4.414, 0.793, 2.426
    x, t_x, T = 4.0, 30.7, 38.9

    log_likelihood = _log_likelihood(
        r,
        alpha,
        a,
        b,
        frequency=np.array([x]),
        recency=np.array([t_x]),
        T=np.array([T]),
    )

    gamma_ratio = gamma_fn(r + x) * alpha**r / gamma_fn(r)
    alive_term = (
        beta_fn(a, b + x + 1) / beta_fn(a, b) * gamma_ratio / (alpha + T) ** (r + x)
    )
    dead_term = (
        beta_fn(a + 1, b + x) / beta_fn(a, b) * gamma_ratio / (alpha + t_x) ** (r + x)
    )
    expected = np.log(alive_term + dead_term)

    np.testing.assert_allclose(log_likelihood, [expected], rtol=1e-12)


def test_repeat_customer_likelihood_is_pinned_to_a_literal():
    # Belt and braces: the value above, frozen. If a refactor changes both the
    # implementation and the natural-space check in the same way, this still
    # fails. It is also visibly *not* the BG/NBD value at the same point
    # (-15.32935235570709, in test_bgnbd_likelihood.py).
    log_likelihood = _log_likelihood(
        0.243,
        4.414,
        0.793,
        2.426,
        frequency=np.array([4.0]),
        recency=np.array([30.7]),
        T=np.array([38.9]),
    )

    np.testing.assert_allclose(log_likelihood, [-15.487436077892465], rtol=1e-11)


def test_a_zero_repeat_customer_can_already_be_dead():
    # Batislam et al. (2007) eq. (5): P(active) = 1 / (1 + a/(b+x) *
    # ((alpha+T)/(alpha+t_x))^(r+x)). At x = 0, t_x = 0 that is
    # 1 / (1 + a/b ((alpha+T)/alpha)^r) — strictly below 1, and the whole
    # point of the model: a customer may drop out at time zero, immediately
    # after the first purchase. BG/NBD pins this to 1 identically.
    r, alpha, a, b = 0.5, 6.0, 1.5, 3.0
    T = np.array([1.0, 20.0, 39.0])

    alive = _probability_alive(
        r, alpha, a, b, frequency=np.zeros(3), recency=np.zeros(3), T=T
    )

    expected = 1.0 / (1.0 + (a / b) * ((alpha + T) / alpha) ** r)
    np.testing.assert_allclose(alive, expected, rtol=1e-13)
    assert (alive < 1.0).all()
    # Longer-observed silence is stronger evidence of a time-zero dropout.
    assert (np.diff(alive) < 0).all()


def test_conditional_expectation_matches_integrating_the_individual_level_model():
    # The strongest independent check: rebuild E[Y(t) | x, t_x, T] from the
    # individual-level equations only, with no hypergeometric function and none
    # of the beta algebra the closed form relies on.
    #
    # Given alive at T, the forward process is a plain BG: eq. (3) without the
    # time-zero (1-p) factor, so E[Y(t) | lambda, p, alive] = (1 - e^{-lambda p t}) / p.
    # Weighting that by the "alive" branch of eq. (1) and integrating over the
    # gamma prior analytically leaves a single integral over p:
    #
    #   E[Y(t)] = int_0^1 (1/p) (1-p)^(x+1) [ (alpha+T)^-(r+x) - (alpha+T+pt)^-(r+x) ] g(p) dp
    #             / [ B(a,b+x+1)/B(a,b) (alpha+T)^-(r+x) + B(a+1,b+x)/B(a,b) (alpha+t_x)^-(r+x) ]
    #
    # This is what catches the coefficient the paper misprints: Appendix A
    # publishes (b+x)/(a-1), but the conditioning divides by B(a, b+x+1), which
    # makes the leading factor B(a-1,b+x+1)/B(a,b+x+1) = (a+b+x)/(a-1).
    r, alpha, a, b = 0.5, 6.0, 1.5, 3.0
    x, t_x, T, t = 3.0, 20.0, 26.0, 12.0

    predicted = _conditional_expected_purchases(
        r,
        alpha,
        a,
        b,
        t,
        frequency=np.array([x]),
        recency=np.array([t_x]),
        T=np.array([T]),
    )

    def beta_density(p: float) -> float:
        return p ** (a - 1) * (1 - p) ** (b - 1) / beta_fn(a, b)

    numerator, _ = quad(
        lambda p: (
            (1 / p)
            * (1 - p) ** (x + 1)
            * ((alpha + T) ** -(r + x) - (alpha + T + p * t) ** -(r + x))
            * beta_density(p)
        ),
        0,
        1,
        limit=500,
        epsabs=1e-14,
        epsrel=1e-14,
    )
    denominator = beta_fn(a, b + x + 1) / beta_fn(a, b) * (alpha + T) ** -(r + x) + (
        beta_fn(a + 1, b + x) / beta_fn(a, b) * (alpha + t_x) ** -(r + x)
    )

    np.testing.assert_allclose(predicted, [numerator / denominator], rtol=1e-10)


def test_conditional_expectation_hypergeometric_term_matches_eulers_integral():
    # Appendix A puts 2F1(r+x, b+x+1; a+b+x; z) in E[Y(t)] — each of the three
    # arguments one higher in `x` than the BG/NBD's 2F1(r+x, b+x; a+b+x-1; z)
    # except the first. Euler's integral,
    #     B(b+x+1, a-1) 2F1(r+x, b+x+1; a+b+x; z)
    #       = int_0^1 u^(b+x) (1-u)^(a-2) (1 - z u)^-(r+x) du,
    # evaluates it without scipy.special.hyp2f1, so a swapped or shifted
    # argument fails here. `a` must exceed 1 or the integrand blows up at u = 1.
    r, alpha, a, b = 0.5, 6.0, 1.5, 3.0
    x, t_x, T, t = 3.0, 20.0, 26.0, 12.0

    predicted = _conditional_expected_purchases(
        r,
        alpha,
        a,
        b,
        t,
        frequency=np.array([x]),
        recency=np.array([t_x]),
        T=np.array([T]),
    )

    z = t / (alpha + T + t)
    integral, _ = quad(
        lambda u: u ** (b + x) * (1 - u) ** (a - 2) * (1 - z * u) ** (-(r + x)),
        0,
        1,
    )
    hypergeometric = integral / beta_fn(b + x + 1, a - 1)
    numerator = (
        (a + b + x)
        / (a - 1)
        * (1 - ((alpha + T) / (alpha + T + t)) ** (r + x) * hypergeometric)
    )
    denominator = 1 + a / (b + x) * ((alpha + T) / (alpha + t_x)) ** (r + x)

    np.testing.assert_allclose(predicted, [numerator / denominator], rtol=1e-8)


def test_conditional_expectation_at_the_origin_reduces_to_equation_4():
    # A customer with no history observed for no time is just a random draw
    # from the population, so E[Y(t) | x=0, t_x=0, T=0] must equal the
    # unconditional E[X(t)] of eq. (4):
    #     b/(a-1) [1 - (alpha/(alpha+t))^r 2F1(r, b+1; a+b; t/(t+alpha))].
    # The 2F1 arguments coincide at x = 0, so what this pins is the leading
    # coefficient and the P(alive) factor: (a+b)/(a-1) * b/(a+b) = b/(a-1).
    # With the coefficient as Appendix A prints it, the identity fails.
    r, alpha, a, b = 0.5, 6.0, 1.5, 3.0
    t = 12.0

    predicted = _conditional_expected_purchases(
        r,
        alpha,
        a,
        b,
        t,
        frequency=np.zeros(1),
        recency=np.zeros(1),
        T=np.zeros(1),
    )

    expected = (
        b
        / (a - 1)
        * (1 - (alpha / (alpha + t)) ** r * hyp2f1(r, b + 1, a + b, t / (t + alpha)))
    )
    np.testing.assert_allclose(predicted, [expected], rtol=1e-12)
