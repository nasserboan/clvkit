import inspect

import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

import clvkit
from clvkit import BGNBD, MBGNBD, CustomerBase
from clvkit._result import Prediction

# The MBG/NBD's own generative story, used to build the fixture below.
# `a`/`b` are only identifiable from data in which customers actually drop out,
# so the base has to be simulated from the model rather than from arbitrary
# purchase dates: a base where nobody ever dies pushes the MLE to the `a -> 0`
# boundary, where the model collapses to a plain NBD and every customer is
# alive with probability 1 — a degenerate fit that would hide every bug.
TRUE_R, TRUE_ALPHA, TRUE_A, TRUE_B = 0.8, 40.0, 0.8, 2.5
FIRST_COHORT_DAY = pd.Timestamp("2020-01-01")
OBSERVATION_PERIOD_END = pd.Timestamp("2021-01-01")


def _base(rows: list[tuple[str, str]], **kwargs) -> CustomerBase:
    transactions = pd.DataFrame(rows, columns=["customer_id", "date"])
    return CustomerBase.from_transactions(transactions, amount_col=None, **kwargs)


def _simulated_transactions(n: int = 500, seed: int = 11) -> pd.DataFrame:
    """A transaction log drawn from the MBG/NBD story of Batislam et al. (2007).

    Purchase rates are gamma, dropout probabilities beta, and — the modified
    part — a customer may drop out immediately after the *first* purchase, not
    only after a repeat. Customers are born over the first four months, so
    tenure ``T`` varies and a never-returner's silence has a length.
    """
    rng = np.random.default_rng(seed)
    purchase_rate = rng.gamma(shape=TRUE_R, scale=1 / TRUE_ALPHA, size=n)
    dropout_probability = rng.beta(TRUE_A, TRUE_B, size=n)
    born = rng.integers(0, 120, size=n)

    rows: list[tuple[str, pd.Timestamp]] = []
    for customer in range(n):
        first = FIRST_COHORT_DAY + pd.Timedelta(days=int(born[customer]))
        rows.append((f"C{customer:03d}", first))
        if rng.random() < dropout_probability[customer]:
            continue  # dead at time zero — the branch BG/NBD does not have

        elapsed = 0.0
        while True:
            elapsed += rng.exponential(1 / purchase_rate[customer])
            when = first + pd.Timedelta(days=elapsed)
            if when > OBSERVATION_PERIOD_END:
                break
            rows.append((f"C{customer:03d}", when))
            if rng.random() < dropout_probability[customer]:
                break

    return pd.DataFrame(rows, columns=["customer_id", "date"])


@pytest.fixture(scope="module")
def simulated_base() -> CustomerBase:
    return CustomerBase.from_transactions(
        _simulated_transactions(),
        amount_col=None,
        observation_period_end=str(OBSERVATION_PERIOD_END.date()),
    )


@pytest.fixture(scope="module")
def model(simulated_base) -> MBGNBD:
    return MBGNBD().fit(simulated_base)


def test_mbgnbd_is_exported_from_the_top_level_namespace():
    assert clvkit.MBGNBD is MBGNBD
    assert "MBGNBD" in clvkit.__all__


def test_fit_returns_self_so_the_three_verbs_chain(simulated_base):
    model = MBGNBD()

    fitted = model.fit(simulated_base)

    assert fitted is model


def test_fit_estimates_four_positive_parameters(model):
    assert list(model.params_.index) == ["r", "alpha", "a", "b"]
    assert (model.params_ > 0).all()
    assert np.isfinite(model.log_likelihood_)


def test_fit_recovers_the_behaviour_it_was_simulated_from(model):
    # The two quantities the parameters exist to express: the population mean
    # purchase rate r/alpha, and the population mean dropout probability
    # a/(a+b). Recovering both from a log simulated out of the model is the
    # end-to-end check that fit, likelihood and parameterisation agree.
    params = model.params_

    assert params["r"] / params["alpha"] == pytest.approx(TRUE_R / TRUE_ALPHA, rel=0.15)
    assert params["a"] / (params["a"] + params["b"]) == pytest.approx(
        TRUE_A / (TRUE_A + TRUE_B), rel=0.15
    )


def test_predict_returns_a_per_customer_prediction(model, simulated_base):
    prediction = model.predict(t=10)

    assert isinstance(prediction, Prediction)
    frame = prediction.to_pandas()
    assert list(frame.columns) == ["expected_purchases"]
    assert frame.index.equals(simulated_base.to_pandas().index)
    assert (frame["expected_purchases"] >= 0).all()


def test_predict_over_a_longer_horizon_expects_more_purchases(model):
    short = model.predict(t=5).to_pandas()["expected_purchases"]
    long = model.predict(t=50).to_pandas()["expected_purchases"]

    assert (long > short).all()


def test_probability_alive_returns_probabilities(model, simulated_base):
    prediction = model.probability_alive()

    frame = prediction.to_pandas()
    assert list(frame.columns) == ["probability_alive"]
    assert frame.index.equals(simulated_base.to_pandas().index)
    assert frame["probability_alive"].between(0, 1).all()


def test_a_customer_who_never_returned_may_already_be_dead(model, simulated_base):
    # The never-returner mass, and the whole reason this model exists. BG/NBD's
    # dropout can only follow a repeat purchase, so a zero-repeat customer is
    # alive with probability exactly 1 no matter how long the silence; the
    # MBG/NBD lets him have dropped out at time zero.
    summary = simulated_base.to_pandas()
    never_returned = summary["frequency"] == 0
    assert never_returned.any()

    modified = model.probability_alive().to_pandas()["probability_alive"]
    original = (
        BGNBD().fit(simulated_base).probability_alive().to_pandas()["probability_alive"]
    )

    assert (original[never_returned] == 1.0).all()
    assert (modified[never_returned] < 1.0).all()


def test_a_longer_silence_makes_a_never_returner_likelier_dead(model, simulated_base):
    # Among customers who never came back, the only thing that distinguishes
    # them is how long we have been watching. Under BG/NBD that distinction
    # cannot register at all.
    summary = simulated_base.to_pandas()
    never_returned = summary["frequency"] == 0

    alive = model.probability_alive().to_pandas()["probability_alive"][never_returned]
    tenure = summary["T"][never_returned]

    # Rank correlation, because the claim is monotonicity, not linearity.
    assert alive.corr(tenure, method="spearman") == pytest.approx(-1.0)


def test_the_two_models_disagree_about_a_base_with_never_returners(simulated_base):
    # Same data, different dropout story: the model the data was simulated from
    # should explain it better, and the parameters are not a reparameterisation
    # of each other.
    modified = MBGNBD().fit(simulated_base)
    original = BGNBD().fit(simulated_base)

    assert modified.log_likelihood_ > original.log_likelihood_
    assert not np.allclose(modified.params_.to_numpy(), original.params_.to_numpy())


def _recent_vs_lapsed_base() -> CustomerBase:
    """Two customers with the same repeat count and tenure, differing only in
    when they last bought."""
    return _base(
        [
            ("recent", "2020-01-01"),
            ("recent", "2020-11-01"),
            ("recent", "2020-12-01"),
            ("lapsed", "2020-01-01"),
            ("lapsed", "2020-01-20"),
            ("lapsed", "2020-02-01"),
        ],
        observation_period_end="2020-12-31",
    )


def test_a_recently_active_customer_is_likelier_alive_than_a_long_lapsed_one(model):
    # Scored, not fitted: two customers carry no information about population
    # heterogeneity, so this pair is a question to put to a model, not a base
    # to estimate one from.
    alive = model.probability_alive(cb=_recent_vs_lapsed_base()).to_pandas()

    assert (
        alive.loc["recent", "probability_alive"]
        > alive.loc["lapsed", "probability_alive"]
    )


def test_a_lapsed_customer_is_expected_to_buy_less_than_a_recently_active_one(model):
    expected = model.predict(t=30, cb=_recent_vs_lapsed_base()).to_pandas()

    assert (
        expected.loc["recent", "expected_purchases"]
        > expected.loc["lapsed", "expected_purchases"]
    )


def test_predictions_can_be_scored_on_a_different_customer_base(model):
    other = _base(
        [("new", "2020-05-01"), ("new", "2020-08-01")],
        observation_period_end="2020-12-31",
    )

    prediction = model.predict(t=10, cb=other)

    assert list(prediction.to_pandas().index) == ["new"]


def test_prediction_plot_renders(model):
    ax = model.probability_alive().plot()

    assert isinstance(ax, Axes)
    assert ax.get_xlabel() == "Days since last purchase"


def test_predict_before_fit_is_refused():
    with pytest.raises(RuntimeError, match="not fitted"):
        MBGNBD().predict(t=1)


def test_probability_alive_before_fit_is_refused():
    with pytest.raises(RuntimeError, match="not fitted"):
        MBGNBD().probability_alive()


def test_fit_refuses_anything_that_is_not_a_customer_base():
    with pytest.raises(TypeError, match="CustomerBase"):
        MBGNBD().fit(pd.DataFrame({"frequency": [1], "recency": [1], "T": [2]}))


def test_fit_refuses_an_empty_customer_base():
    empty = _base([("A", "2020-01-01")]).to_pandas().iloc[:0]
    cb = CustomerBase(
        empty,
        time_unit="D",
        observation_period_end=pd.Timestamp("2020-01-01"),
        has_monetary=False,
        on_negative="net",
    )

    with pytest.raises(ValueError, match="empty"):
        MBGNBD().fit(cb)


def test_scoring_a_base_with_a_different_time_unit_is_refused(model):
    # The provenance guard: daily parameters applied to a weekly base would
    # produce plausible-looking, wrong numbers.
    weekly = _base(
        [("new", "2020-05-01"), ("new", "2020-08-01")],
        observation_period_end="2020-12-31",
        time_unit="W",
    )

    with pytest.raises(ValueError, match="time_unit"):
        model.predict(t=10, cb=weekly)


def test_predict_refuses_a_non_positive_horizon(model):
    with pytest.raises(ValueError, match="positive"):
        model.predict(t=0)


def test_the_surface_is_interchangeable_with_bgnbd():
    # The promise of the ticket: an analyst swaps one for the other — into
    # `CLV(transaction_model=MBGNBD())`, say — without learning a new API.
    # Same three verbs, same parameters, no aliases. (Return annotations
    # differ, and should: each `fit` hands back its own type.)
    verbs = ("fit", "predict", "probability_alive")

    for verb in verbs:
        assert (
            inspect.signature(getattr(MBGNBD, verb)).parameters
            == inspect.signature(getattr(BGNBD, verb)).parameters
        )
    assert {name for name in vars(MBGNBD) if not name.startswith("_")} == set(verbs)
