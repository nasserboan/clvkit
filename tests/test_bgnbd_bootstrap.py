"""Parametric bootstrap for BG/NBD parameter uncertainty (issue #125, #140).

The seam under test is the public `BGNBD.parameter_uncertainty()` and the
`ParameterUncertainty` it returns. The deepest check is that the source-tree
generative simulator agrees with the likelihood: a base simulated from known
parameters, refit, recovers the behaviour it was drawn from — the analog of the
model's existing "fit recovers what it was simulated from" tests.
"""

import json

import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

from clvkit import BGNBD, CustomerBase
from clvkit._result import Result
from clvkit.clv.bgnbd import _simulate_summary

# A generative truth with genuine dropout, so a and b are estimable at all.
TRUE_R, TRUE_ALPHA, TRUE_A, TRUE_B = 0.4, 8.0, 0.8, 2.5


def _simulated_base(
    n: int,
    *,
    seed: int,
    r: float = TRUE_R,
    alpha: float = TRUE_ALPHA,
    a: float = TRUE_A,
    b: float = TRUE_B,
) -> CustomerBase:
    """A timing-only base drawn from known BG/NBD parameters.

    Tenures vary across customers so a never-returner's silence has a length —
    the only thing that pins the dropout parameters down at all.
    """
    rng = np.random.default_rng(seed)
    tenure = rng.uniform(26.0, 52.0, size=n)
    frequency, recency = _simulate_summary(r, alpha, a, b, T=tenure, rng=rng)
    summary = pd.DataFrame(
        {"frequency": frequency, "recency": recency, "T": tenure},
        index=pd.Index([f"c{i}" for i in range(n)], name="customer_id"),
    )
    return CustomerBase(
        summary,
        time_unit="W",
        collapse="W",
        observation_period_end=pd.Timestamp("2021-01-01"),
        has_monetary=False,
        on_negative="net",
    )


@pytest.fixture(scope="module")
def fitted() -> BGNBD:
    return BGNBD().fit(_simulated_base(300, seed=1))


# ---------------------------------------------------------------------------
# the result contract
# ---------------------------------------------------------------------------


def test_it_returns_one_row_per_parameter_in_the_models_order(fitted):
    frame = fitted.parameter_uncertainty(n=30, seed=0).to_pandas()

    assert list(frame.index) == ["r", "alpha", "a", "b"]
    assert list(frame.columns) == ["estimate", "se", "ci_low", "ci_high"]


def test_the_estimate_column_is_the_original_point_estimate(fitted):
    frame = fitted.parameter_uncertainty(n=30, seed=0).to_pandas()

    # check_names=False: the result names its index "parameter"; params_ does
    # not. The labels and the values are what must agree.
    pd.testing.assert_series_equal(frame["estimate"], fitted.params_, check_names=False)


def test_every_standard_error_is_positive_and_intervals_are_ordered(fitted):
    frame = fitted.parameter_uncertainty(n=40, seed=0).to_pandas()

    assert (frame["se"] > 0).all()
    assert (frame["ci_low"] <= frame["ci_high"]).all()


def test_it_satisfies_the_result_protocol(fitted):
    assert isinstance(fitted.parameter_uncertainty(n=20, seed=0), Result)


def test_to_json_round_trips_to_the_same_numbers(fitted):
    result = fitted.parameter_uncertainty(n=20, seed=0)

    payload = json.loads(result.to_json())

    assert set(payload) == {"r", "alpha", "a", "b"}
    assert payload["r"]["estimate"] == pytest.approx(fitted.params_["r"])


def test_it_plots_itself_onto_matplotlib_axes(fitted):
    ax = fitted.parameter_uncertainty(n=20, seed=0).plot()

    assert isinstance(ax, Axes)
    assert [label.get_text() for label in ax.get_yticklabels()] == [
        "r",
        "alpha",
        "a",
        "b",
    ]


def test_repr_names_the_model_and_the_replicate_count(fitted):
    text = repr(fitted.parameter_uncertainty(n=20, seed=0))

    assert "BGNBD" in text
    assert "20 replicates" in text


# ---------------------------------------------------------------------------
# reproducibility and validation
# ---------------------------------------------------------------------------


def test_the_same_seed_gives_an_identical_result(fitted):
    first = fitted.parameter_uncertainty(n=30, seed=7).to_pandas()
    second = fitted.parameter_uncertainty(n=30, seed=7).to_pandas()

    pd.testing.assert_frame_equal(first, second)


def test_a_different_seed_moves_the_intervals(fitted):
    first = fitted.parameter_uncertainty(n=30, seed=7).to_pandas()
    other = fitted.parameter_uncertainty(n=30, seed=8).to_pandas()

    assert not first["ci_low"].equals(other["ci_low"])


def test_calling_it_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        BGNBD().parameter_uncertainty(n=10, seed=0)


def test_bad_arguments_are_refused(fitted):
    with pytest.raises(ValueError, match="n must be"):
        fitted.parameter_uncertainty(n=1, seed=0)
    with pytest.raises(ValueError, match="confidence"):
        fitted.parameter_uncertainty(n=10, seed=0, confidence=1.5)


def test_the_simulator_and_likelihood_agree():
    # Simulate a large base from the known truth, refit, and recover the two
    # quantities the parameters exist to express: the population purchase rate
    # r/alpha and the population dropout probability a/(a+b). This pins the
    # source-tree simulator to the likelihood, exactly as the model's own
    # recovery tests do — a and b individually are too weakly identified to
    # assert on, which is the whole motivation for this feature.
    params = BGNBD().fit(_simulated_base(5000, seed=2)).params_

    assert params["r"] / params["alpha"] == pytest.approx(TRUE_R / TRUE_ALPHA, rel=0.15)
    assert params["a"] / (params["a"] + params["b"]) == pytest.approx(
        TRUE_A / (TRUE_A + TRUE_B), rel=0.2
    )


def test_the_interval_covers_the_truth_for_the_well_identified_parameters():
    # A single seeded draw, not a coverage study: on a large base the 95%
    # interval should comfortably contain the true r and alpha. a and b are
    # deliberately not asserted here — they are what the bootstrap exists to
    # show as uncertain.
    model = BGNBD().fit(_simulated_base(2000, seed=3))

    frame = model.parameter_uncertainty(n=40, seed=0).to_pandas()

    for name, truth in (("r", TRUE_R), ("alpha", TRUE_ALPHA)):
        assert frame.loc[name, "ci_low"] <= truth <= frame.loc[name, "ci_high"]


def test_intervals_tighten_as_the_base_grows():
    small = BGNBD().fit(_simulated_base(150, seed=4))
    large = BGNBD().fit(_simulated_base(1500, seed=5))

    small_frame = small.parameter_uncertainty(n=25, seed=0).to_pandas()
    large_frame = large.parameter_uncertainty(n=25, seed=0).to_pandas()

    small_width = small_frame["ci_high"] - small_frame["ci_low"]
    large_width = large_frame["ci_high"] - large_frame["ci_low"]

    assert large_width.sum() < small_width.sum()


def test_a_and_b_come_back_wider_than_r():
    # The methodologist's point, encoded: the dropout parameters are far less
    # identified than the purchase-rate shape, so their intervals are relatively
    # wider. This is the honest result the feature exists to surface.
    model = BGNBD().fit(_simulated_base(1500, seed=6))

    frame = model.parameter_uncertainty(n=40, seed=0).to_pandas()
    relative_width = (frame["ci_high"] - frame["ci_low"]) / frame["estimate"].abs()

    assert relative_width["a"] > relative_width["r"]
    assert relative_width["b"] > relative_width["r"]
