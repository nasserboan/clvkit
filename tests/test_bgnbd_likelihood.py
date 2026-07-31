"""Pinpoint tests for the BG/NBD special-function math.

These are the one deliberate exception to "test the public API only": the
likelihood's beta/gamma ratio and the Gaussian-hypergeometric term in
E[Y(t)] are the highest-risk lines in the package, and a wrong special
function should fail a named test rather than surface as "parameters drifted".

Every expected value here comes from an independent source — a closed form
the paper states, or numerical integration of the Euler integral the paper
derives the hypergeometric term from — never from re-running the code path
under test.
"""

import numpy as np
from scipy.integrate import quad
from scipy.special import beta as beta_fn
from scipy.special import gamma as gamma_fn

from clvkit.clv.bgnbd import _conditional_expected_purchases, _log_likelihood


def test_zero_repeat_customer_likelihood_matches_the_closed_form():
    # Fader et al. (2005) eq. (6): with x = 0 the delta term vanishes and both
    # beta ratios collapse to 1, leaving L = (alpha / (alpha + T))^r exactly.
    r, alpha, a, b = 0.5, 6.0, 1.5, 3.0
    T = np.array([10.0, 25.0])

    log_likelihood = _log_likelihood(
        r, alpha, a, b, frequency=np.zeros(2), recency=np.zeros(2), T=T
    )

    expected = r * np.log(alpha / (alpha + T))
    np.testing.assert_allclose(log_likelihood, expected, rtol=0, atol=1e-12)


def test_repeat_customer_likelihood_matches_equation_6_in_natural_space():
    # The implementation works in log space with a log-sum-exp; this recomputes
    # Fader et al. (2005) eq. (6) literally, in natural space, from beta and
    # gamma functions. Two independent arithmetic paths, one answer.
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
        beta_fn(a, b + x) / beta_fn(a, b) * gamma_ratio / (alpha + T) ** (r + x)
    )
    dead_term = (
        beta_fn(a + 1, b + x - 1)
        / beta_fn(a, b)
        * gamma_ratio
        / (alpha + t_x) ** (r + x)
    )
    expected = np.log(alive_term + dead_term)

    np.testing.assert_allclose(log_likelihood, [expected], rtol=1e-12)


def test_repeat_customer_likelihood_is_pinned_to_a_literal():
    # Belt and braces: the value above, frozen. If a refactor changes both the
    # implementation and the natural-space check in the same way, this still fails.
    log_likelihood = _log_likelihood(
        0.243,
        4.414,
        0.793,
        2.426,
        frequency=np.array([4.0]),
        recency=np.array([30.7]),
        T=np.array([38.9]),
    )

    np.testing.assert_allclose(log_likelihood, [-15.32935235570709], rtol=1e-11)


def test_conditional_expectation_hypergeometric_term_matches_eulers_integral():
    # Fader et al. (2005) appendix, eq. (A8): the 2F1 in eq. (10) is exactly
    # Euler's integral
    #     B(a-1, b+x) * 2F1(r+x, b+x; a+b+x-1; z)
    #       = int_0^1 q^(b+x-1) (1-q)^(a-2) (1 - z q)^-(r+x) dq.
    # Integrating that numerically evaluates 2F1 without scipy.special.hyp2f1,
    # so this catches a wrong argument order or a wrong `c` parameter.
    # `a` must exceed 1 here or the integrand is not integrable at q = 1.
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
        lambda q: q ** (b + x - 1) * (1 - q) ** (a - 2) * (1 - z * q) ** (-(r + x)),
        0,
        1,
    )
    hyp2f1 = integral / beta_fn(a - 1, b + x)
    numerator = (
        (a + b + x - 1)
        / (a - 1)
        * (1 - ((alpha + T) / (alpha + T + t)) ** (r + x) * hyp2f1)
    )
    denominator = 1 + a / (b + x - 1) * ((alpha + T) / (alpha + t_x)) ** (r + x)

    np.testing.assert_allclose(predicted, [numerator / denominator], rtol=1e-8)


def test_probability_alive_is_one_for_a_zero_repeat_customer():
    # Fader et al. (2005) appendix: "a customer cannot drop out before he has
    # made any transactions", so P(active | X = 0) = 1 identically.
    from clvkit.clv.bgnbd import _probability_alive

    alive = _probability_alive(
        0.243,
        4.414,
        0.793,
        2.426,
        frequency=np.zeros(3),
        recency=np.zeros(3),
        T=np.array([1.0, 20.0, 39.0]),
    )

    np.testing.assert_allclose(alive, np.ones(3))
