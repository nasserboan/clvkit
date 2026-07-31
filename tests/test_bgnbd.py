import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

from clvkit import BGNBD, CustomerBase
from clvkit._result import Prediction


def _base(rows: list[tuple[str, str]], **kwargs) -> CustomerBase:
    transactions = pd.DataFrame(rows, columns=["customer_id", "date"])
    return CustomerBase.from_transactions(transactions, amount_col=None, **kwargs)


def _toy_base(**kwargs) -> CustomerBase:
    """A small base with a mix of loyal, lapsed and one-shot customers."""
    rng = np.random.default_rng(7)
    rows: list[tuple[str, str]] = []
    for customer in range(60):
        start = pd.Timestamp("2020-01-01")
        n_purchases = int(rng.integers(1, 12))
        gaps = rng.integers(1, 25, size=n_purchases).cumsum()
        for gap in gaps:
            rows.append(
                (f"C{customer:03d}", str((start + pd.Timedelta(days=int(gap))).date()))
            )
    return _base(rows, observation_period_end="2021-01-01", **kwargs)


def test_fit_returns_self_so_the_three_verbs_chain():
    model = BGNBD()

    fitted = model.fit(_toy_base())

    assert fitted is model


def test_fit_estimates_four_positive_parameters():
    model = BGNBD().fit(_toy_base())

    assert list(model.params_.index) == ["r", "alpha", "a", "b"]
    assert (model.params_ > 0).all()
    assert np.isfinite(model.log_likelihood_)


def test_predict_returns_a_per_customer_prediction():
    cb = _toy_base()
    model = BGNBD().fit(cb)

    prediction = model.predict(t=10)

    assert isinstance(prediction, Prediction)
    frame = prediction.to_pandas()
    assert list(frame.columns) == ["expected_purchases"]
    assert frame.index.equals(cb.to_pandas().index)
    assert (frame["expected_purchases"] >= 0).all()


def test_predict_over_a_longer_horizon_expects_more_purchases():
    model = BGNBD().fit(_toy_base())

    short = model.predict(t=5).to_pandas()["expected_purchases"]
    long = model.predict(t=50).to_pandas()["expected_purchases"]

    assert (long > short).all()


def test_probability_alive_returns_probabilities():
    cb = _toy_base()
    model = BGNBD().fit(cb)

    prediction = model.probability_alive()

    frame = prediction.to_pandas()
    assert list(frame.columns) == ["probability_alive"]
    assert frame.index.equals(cb.to_pandas().index)
    assert frame["probability_alive"].between(0, 1).all()


def _recent_vs_lapsed_base() -> CustomerBase:
    """Two customers with the same repeat count and tenure, differing only in
    when they last bought — plus padding so the fit has something to learn from."""
    return _base(
        [
            ("recent", "2020-01-01"),
            ("recent", "2020-11-01"),
            ("recent", "2020-12-01"),
            ("lapsed", "2020-01-01"),
            ("lapsed", "2020-01-20"),
            ("lapsed", "2020-02-01"),
        ]
        + [
            (f"pad{i:02d}", date)
            for i in range(20)
            for date in ("2020-01-01", "2020-06-01")
        ],
        observation_period_end="2020-12-31",
    )


def test_a_recently_active_customer_is_likelier_alive_than_a_long_lapsed_one():
    model = BGNBD().fit(_recent_vs_lapsed_base())

    alive = model.probability_alive().to_pandas()["probability_alive"]

    assert alive["recent"] > alive["lapsed"]


def test_a_lapsed_customer_is_expected_to_buy_less_than_a_recently_active_one():
    model = BGNBD().fit(_recent_vs_lapsed_base())

    expected = model.predict(t=30).to_pandas()["expected_purchases"]

    assert expected["recent"] > expected["lapsed"]


def test_predictions_can_be_scored_on_a_different_customer_base():
    model = BGNBD().fit(_toy_base())
    other = _base(
        [("new", "2020-05-01"), ("new", "2020-08-01")],
        observation_period_end="2020-12-31",
    )

    prediction = model.predict(t=10, cb=other)

    assert list(prediction.to_pandas().index) == ["new"]


def test_prediction_plot_renders():
    model = BGNBD().fit(_toy_base())

    ax = model.probability_alive().plot()

    assert isinstance(ax, Axes)
    assert ax.get_xlabel() == "Probability alive"


def test_predict_before_fit_is_refused():
    with pytest.raises(RuntimeError, match="not fitted"):
        BGNBD().predict(t=1)


def test_probability_alive_before_fit_is_refused():
    with pytest.raises(RuntimeError, match="not fitted"):
        BGNBD().probability_alive()


def test_fit_refuses_anything_that_is_not_a_customer_base():
    with pytest.raises(TypeError, match="CustomerBase"):
        BGNBD().fit(pd.DataFrame({"frequency": [1], "recency": [1], "T": [2]}))


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
        BGNBD().fit(cb)


def test_scoring_a_base_with_a_different_time_unit_is_refused():
    # The provenance guard: weekly parameters applied to a daily base would
    # produce plausible-looking, wrong numbers.
    model = BGNBD().fit(_toy_base(time_unit="W"))
    daily = _toy_base(time_unit="D")

    with pytest.raises(ValueError, match="time_unit"):
        model.predict(t=10, cb=daily)


def test_predict_refuses_a_non_positive_horizon():
    model = BGNBD().fit(_toy_base())

    with pytest.raises(ValueError, match="positive"):
        model.predict(t=0)
