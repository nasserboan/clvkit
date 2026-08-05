"""CLV — composing a transaction model with a monetary model.

Follows Fader, Hardie & Lee (2005), "RFM and CLV: Using Iso-value Curves for
Customer Base Analysis", §2: equation (1) factors lifetime value into a spend
term and a discounted-expected-transactions term, and the discrete DET sum on
p. 10 is what makes that computable over a finite horizon.
"""

import warnings
from functools import cache
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

import clvkit
from clvkit import BGNBD, CLV, MBGNBD, CustomerBase, GammaGamma
from clvkit._result import Result
from clvkit.clv.clv import CLVResult
from clvkit.clv.independence import IndependenceCheck, MonetaryIndependenceWarning

CDNOW_SAMPLE = Path(__file__).resolve().parents[1] / "CDNOW_sample.txt"


@cache
def _cdnow() -> CustomerBase:
    """The CDNOW 1/10 sample, weekly — the base the published estimates use."""
    log = pd.read_csv(
        CDNOW_SAMPLE,
        sep=r"\s+",
        header=None,
        names=["customer_id", "sample_id", "date", "quantity", "amount"],
    )
    log["date"] = pd.to_datetime(log["date"], format="%Y%m%d")
    return CustomerBase.from_transactions(
        log[["customer_id", "date", "amount"]], time_unit="W"
    )


@cache
def _fitted_cdnow_clv() -> CLV:
    return CLV().fit(_cdnow())


def _base(
    *, dependent_spend: bool = False, n_customers: int = 500, seed: int = 11
) -> CustomerBase:
    """A synthetic base drawn from the BG/NBD story itself.

    Purchases arrive as a Poisson process with a gamma-distributed rate, and
    after each one the customer drops out with a beta-distributed probability
    (Fader et al. 2005, §3). Customers are born at staggered dates, so recency
    and T actually vary — a base where every customer buys on consecutive
    weeks from a common start has no dropout signal at all and the likelihood
    runs off to infinity.
    """
    rng = np.random.default_rng(seed)
    origin = pd.Timestamp("2020-01-06")
    horizon_weeks = 78.0

    rows = []
    for customer in range(n_customers):
        birth = float(rng.integers(0, 30))
        rate = float(rng.gamma(shape=0.8, scale=0.20))
        dropout = float(rng.beta(1.2, 3.0))
        # Spend either tracks the buying rate or ignores it — the two sides of
        # the independence assumption, simulated.
        scale = 6.0 * (1 + 20 * rate) if dependent_spend else 6.0

        week = birth
        while week <= horizon_weeks:
            rows.append(
                (
                    f"c{customer:04d}",
                    origin + pd.Timedelta(weeks=round(week)),
                    float(rng.gamma(shape=6.0, scale=scale)),
                )
            )
            if rng.random() < dropout:
                break
            week += float(rng.exponential(1.0 / max(rate, 1e-6)))

    return CustomerBase.from_transactions(
        pd.DataFrame(rows, columns=["customer_id", "date", "amount"]),
        time_unit="W",
        observation_period_end=origin + pd.Timedelta(weeks=horizon_weeks),
    )


class _CountingTransactionModel:
    """A stand-in transaction model — the seam MBGNBD will drop into."""

    def __init__(self, per_period: float = 0.5) -> None:
        self.per_period = per_period
        self.fitted_on: CustomerBase | None = None

    def fit(self, cb: CustomerBase) -> "_CountingTransactionModel":
        self.fitted_on = cb
        return self

    def predict(self, t: float, cb: CustomerBase | None = None):
        from clvkit._result import Prediction

        index = (cb or self.fitted_on).to_pandas().index
        return Prediction(
            pd.Series(self.per_period * t, index=index),
            name="expected_purchases",
            description="stub",
        )


# --- the headline pitch --------------------------------------------------


def test_the_readme_pitch_runs_end_to_end(cdnow_sample, tmp_path, monkeypatch):
    # The <10-line pitch from the spec, executed verbatim on a raw log.
    monkeypatch.chdir(tmp_path)
    df = cdnow_sample[["customer_id", "date", "amount"]]

    cb = CustomerBase.from_transactions(df)
    clv = CLV().fit(cb).predict(horizon=12, discount_rate=0.01)
    clv.plot()
    clv.to_pandas().to_csv("clv.csv")

    assert len(pd.read_csv(tmp_path / "clv.csv")) == len(cb.to_pandas())


def test_clv_is_exported_from_the_top_level_namespace():
    assert clvkit.CLV is CLV
    assert "CLV" in clvkit.__all__


# --- the composition itself ---------------------------------------------


def test_predict_returns_the_factors_of_equation_1():
    # (1): CLV = margin x revenue/transaction x DET. Every factor is a column,
    # so the composed number can be audited rather than taken on faith.
    result = _fitted_cdnow_clv().predict(horizon=12, discount_rate=0.01)

    frame = result.to_pandas()

    assert list(frame.columns) == [
        "expected_purchases",
        "discounted_expected_transactions",
        "expected_spend",
        "clv",
    ]
    assert list(frame.index) == list(_cdnow().to_pandas().index)


def test_clv_is_the_product_of_its_factors():
    result = _fitted_cdnow_clv().predict(horizon=12, discount_rate=0.01, margin=0.3)

    frame = result.to_pandas()

    assert frame["clv"].to_numpy() == pytest.approx(
        0.3
        * frame["expected_spend"].to_numpy()
        * frame["discounted_expected_transactions"].to_numpy()
    )


def test_undiscounted_det_is_just_expected_purchases_over_the_horizon():
    # With d = 0 the DET sum telescopes: every increment is weighted 1, so what
    # is left is E(Y(horizon)) — the transaction model's own answer.
    clv = _fitted_cdnow_clv()

    frame = clv.predict(horizon=12).to_pandas()

    assert frame["discounted_expected_transactions"].to_numpy() == pytest.approx(
        frame["expected_purchases"].to_numpy()
    )


def test_discounting_lowers_lifetime_value():
    clv = _fitted_cdnow_clv()

    undiscounted = clv.predict(horizon=26).to_pandas()["clv"]
    discounted = clv.predict(horizon=26, discount_rate=0.01).to_pandas()["clv"]

    assert (discounted < undiscounted).all()


def test_det_matches_the_discrete_sum_the_paper_writes_down():
    # Pinpoint: DET = sum over t of [E(Y(t)) - E(Y(t-1))] / (1+d)^t (p. 10),
    # computed here straight from the transaction model's own predict().
    cb = _base()
    transaction_model = BGNBD().fit(cb)
    horizon, discount_rate = 5, 0.02

    clv = CLV(transaction_model=BGNBD(), monetary_model=GammaGamma()).fit(cb)
    det = clv.predict(horizon=horizon, discount_rate=discount_rate).to_pandas()[
        "discounted_expected_transactions"
    ]

    cumulative = [
        transaction_model.predict(t).to_pandas()["expected_purchases"]
        for t in range(1, horizon + 1)
    ]
    expected = sum(
        (cumulative[t - 1] - (cumulative[t - 2] if t > 1 else 0.0))
        / (1 + discount_rate) ** t
        for t in range(1, horizon + 1)
    )

    assert det.to_numpy() == pytest.approx(expected.to_numpy())


def test_expected_spend_comes_from_the_monetary_model():
    cb = _base()

    clv = CLV().fit(cb)
    spend = clv.predict(horizon=8).to_pandas()["expected_spend"]

    assert spend.to_numpy() == pytest.approx(
        GammaGamma().fit(cb).predict().to_pandas()["expected_spend"].to_numpy()
    )


def test_margin_scales_lifetime_value_linearly():
    clv = _fitted_cdnow_clv()

    revenue = clv.predict(horizon=12).to_pandas()["clv"]
    contribution = clv.predict(horizon=12, margin=0.25).to_pandas()["clv"]

    assert contribution.to_numpy() == pytest.approx(0.25 * revenue.to_numpy())


def test_a_longer_horizon_is_worth_more():
    clv = _fitted_cdnow_clv()

    assert (
        clv.predict(horizon=52).to_pandas()["clv"]
        > clv.predict(horizon=12).to_pandas()["clv"]
    ).all()


# --- sub-model injection -------------------------------------------------


def test_defaults_are_bgnbd_and_gamma_gamma():
    clv = CLV()

    assert isinstance(clv.transaction_model, BGNBD)
    assert isinstance(clv.monetary_model, GammaGamma)


def test_injected_models_are_used_instead_of_the_defaults():
    stub = _CountingTransactionModel(per_period=0.5)
    cb = _base()

    clv = CLV(transaction_model=stub).fit(cb)

    assert clv.transaction_model is stub
    assert stub.fitted_on is cb
    # 0.5 purchases per period, undiscounted, over 4 periods.
    assert clv.predict(horizon=4).to_pandas()[
        "discounted_expected_transactions"
    ].to_numpy() == pytest.approx(2.0)


def test_mbgnbd_drops_into_the_transaction_seam():
    """The composition takes MBG/NBD with no model-specific code.

    `CLV` reaches the transaction model only through `fit` and `predict(t)`,
    so the DET sum never learns which of the two it is holding. This is the
    acceptance criterion MBG/NBD (#19) could not assert on its own branch,
    because `CLV` did not exist there yet.
    """
    cb = _base()

    clv = CLV(transaction_model=MBGNBD()).fit(cb)
    frame = clv.predict(horizon=12, discount_rate=0.01).to_pandas()

    assert isinstance(clv.transaction_model, MBGNBD)
    assert clv.transaction_model.params_ is not None
    assert clv.monetary_model.p is not None

    assert list(frame.columns) == [
        "expected_purchases",
        "discounted_expected_transactions",
        "expected_spend",
        "clv",
    ]
    assert np.isfinite(frame.to_numpy()).all()
    assert (frame["clv"] > 0).all()
    # Equation (1) still factorises, at margin 1.0.
    assert frame["clv"].to_numpy() == pytest.approx(
        (frame["discounted_expected_transactions"] * frame["expected_spend"]).to_numpy()
    )


def test_the_transaction_model_actually_drives_lifetime_value():
    """Swapping BG/NBD for MBG/NBD moves the answer, and moves it downward.

    Guards the seam against a silent fallback to the default: identical
    output would mean the injected model is being ignored. MBG/NBD lets a
    customer die at time zero, so it holds less residual value than BG/NBD
    on the same base.
    """
    cb = _base()

    bgnbd = CLV(transaction_model=BGNBD()).fit(cb).predict(horizon=12).to_pandas()
    mbgnbd = CLV(transaction_model=MBGNBD()).fit(cb).predict(horizon=12).to_pandas()

    assert not np.allclose(bgnbd["clv"].to_numpy(), mbgnbd["clv"].to_numpy())
    assert mbgnbd["clv"].sum() < bgnbd["clv"].sum()


def test_fit_fits_both_sub_models():
    cb = _base()

    clv = CLV().fit(cb)

    assert clv.transaction_model.params_ is not None
    assert clv.monetary_model.p is not None


# --- the independence assumption ----------------------------------------


def test_fit_exposes_the_independence_check_behind_the_composition():
    clv = _fitted_cdnow_clv()

    check = clv.independence_check()

    assert isinstance(check, IndependenceCheck)
    assert check.holds()


def test_fit_warns_when_spend_and_buying_rate_move_together():
    # The composition is only legitimate under §2.1(iii). If a base plainly
    # violates it, saying so at fit time is the whole point of the check.
    cb = _base(dependent_spend=True)

    with pytest.warns(MonetaryIndependenceWarning, match="independen"):
        CLV().fit(cb)


def test_fit_is_quiet_on_a_base_that_satisfies_the_assumption():
    with warnings.catch_warnings():
        warnings.simplefilter("error", MonetaryIndependenceWarning)
        CLV().fit(_base())


def test_the_independence_warning_can_be_switched_off():
    cb = _base(dependent_spend=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error", MonetaryIndependenceWarning)
        CLV(check_independence=False).fit(cb)


def test_independence_check_before_fit_is_an_error():
    with pytest.raises(RuntimeError, match="not fitted"):
        CLV().independence_check()


# --- preconditions -------------------------------------------------------


def test_fit_rejects_a_customer_base_without_amounts():
    log = pd.DataFrame(
        [("A", "2020-01-01", 10.0), ("A", "2020-01-08", 20.0)],
        columns=["customer_id", "date", "amount"],
    )
    cb = CustomerBase.from_transactions(log, amount_col=None)

    with pytest.raises(ValueError, match="amount_col"):
        CLV().fit(cb)


def test_the_missing_amounts_error_names_clv_not_the_inner_model():
    log = pd.DataFrame(
        [("A", "2020-01-01", 10.0), ("A", "2020-01-08", 20.0)],
        columns=["customer_id", "date", "amount"],
    )
    cb = CustomerBase.from_transactions(log, amount_col=None)

    with pytest.raises(ValueError, match="CLV"):
        CLV().fit(cb)


def test_predict_before_fit_is_an_error():
    with pytest.raises(RuntimeError, match="not fitted"):
        CLV().predict(horizon=12)


@pytest.mark.parametrize("horizon", [0, -3, 2.5])
def test_horizon_must_be_a_positive_whole_number_of_periods(horizon):
    # The DET sum runs over discrete periods; half a period has no increment.
    clv = _fitted_cdnow_clv()

    with pytest.raises(ValueError, match="horizon"):
        clv.predict(horizon=horizon)


def test_discount_rate_must_be_above_minus_one():
    clv = _fitted_cdnow_clv()

    with pytest.raises(ValueError, match="discount_rate"):
        clv.predict(horizon=12, discount_rate=-1.0)


def test_margin_must_be_positive():
    clv = _fitted_cdnow_clv()

    with pytest.raises(ValueError, match="margin"):
        clv.predict(horizon=12, margin=0.0)


# --- the Result contract -------------------------------------------------


def test_result_satisfies_the_result_contract():
    result = _fitted_cdnow_clv().predict(horizon=12)

    assert isinstance(result, CLVResult)
    assert isinstance(result, Result)


def test_result_to_pandas_returns_a_copy():
    result = _fitted_cdnow_clv().predict(horizon=12)

    frame = result.to_pandas()
    frame["clv"] = 0.0

    assert (result.to_pandas()["clv"] > 0).any()


def test_result_round_trips_through_json():
    result = _fitted_cdnow_clv().predict(horizon=12)

    restored = pd.read_json(StringIO(result.to_json()), orient="index")

    assert restored["clv"].to_numpy() == pytest.approx(
        result.to_pandas()["clv"].to_numpy()
    )


def test_result_plot_is_labelled_with_the_horizon_and_time_unit():
    result = _fitted_cdnow_clv().predict(horizon=12, discount_rate=0.01)

    _, target = plt.subplots()
    ax = result.plot(ax=target)

    assert ax is target
    assert isinstance(ax, Axes)
    assert "12" in ax.get_xlabel()
    assert "weeks" in ax.get_xlabel()
    assert ax.get_ylabel() == "Expected spend per transaction ($)"
    assert ax.get_title() == "Where CLV comes from: how often × how much"
    assert ax.spines["top"].get_visible() is False
    assert ax.spines["right"].get_visible() is False
    assert len(ax.figure.axes) == 2
    assert ax.figure.axes[1].get_ylabel() == "12-week CLV ($)"
    assert len(ax.lines) > 0

    line = ax.lines[0]
    assert line.get_xdata() * line.get_ydata() == pytest.approx(
        line.get_xdata()[0] * line.get_ydata()[0]
    )


def test_result_plot_accepts_a_custom_title():
    result = _fitted_cdnow_clv().predict(horizon=12, discount_rate=0.01)

    ax = result.plot(title="My CLV")

    assert ax.get_title() == "My CLV"


def test_iso_curve_labels_follow_manual_axis_limits():
    # The labels are pinned to the top of each curve via the axis's own
    # limit-change events, so setting xlim/ylim by hand after the plot must
    # keep every visible label inside the frame rather than stranding it.
    result = _fitted_cdnow_clv().predict(horizon=12, discount_rate=0.01)

    _, target = plt.subplots()
    ax = result.plot(ax=target, curve_levels=4)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 60)

    labels = [text for text in ax.texts if text.get_text().startswith("$")]
    assert labels  # each curve carries its round-money CLV level
    visible = [label for label in labels if label.get_visible()]
    assert visible  # at least one curve crosses the manual frame
    for label in visible:
        x, y = label.xy
        assert -1e-6 <= x <= 10 + 1e-6
        assert -1e-6 <= y <= 60 + 1e-6


def test_iso_curves_are_labelled_at_round_money_levels():
    # The curves stand in for "everyone here is worth about the same", so they
    # sit at 1/2/5 round numbers rather than raw percentiles of the CLV column.
    result = _fitted_cdnow_clv().predict(horizon=12, discount_rate=0.01)

    ax = result.plot(curve_levels=4)

    levels = sorted(
        int(text.get_text().lstrip("$").replace(",", ""))
        for text in ax.texts
        if text.get_text().startswith("$")
    )
    assert levels  # curves are drawn and labelled
    nice_mantissas = {1, 2, 5}
    for level in levels:
        mantissa = level / 10 ** (len(str(level)) - 1)
        assert mantissa in nice_mantissas


def test_plot_accepts_a_currency_symbol():
    result = _fitted_cdnow_clv().predict(horizon=12, discount_rate=0.01)

    ax = result.plot(currency="€")

    assert ax.get_ylabel() == "Expected spend per transaction (€)"
    assert ax.figure.axes[1].get_ylabel().endswith("(€)")
    assert any(text.get_text().startswith("€") for text in ax.texts)


def test_explicit_levels_override_the_automatic_ones():
    result = _fitted_cdnow_clv().predict(horizon=12, discount_rate=0.01)

    ax = result.plot(levels=[25, 75])

    money = sorted(
        text.get_text() for text in ax.texts if text.get_text().startswith("$")
    )
    assert money == ["$25", "$75"]


def test_color_scale_defaults_clip_the_whale_tail():
    # Left to None, the ceiling is the 95th percentile of CLV, not the raw max
    # a couple of whales set.
    result = _fitted_cdnow_clv().predict(horizon=12, discount_rate=0.01)
    clv = result.to_pandas()["clv"].to_numpy()

    ax = result.plot()

    _, ceiling = ax.collections[0].get_clim()
    assert ceiling == pytest.approx(np.quantile(clv, 0.95))
    assert ceiling < clv.max()


def test_color_scale_limits_are_configurable():
    result = _fitted_cdnow_clv().predict(horizon=12, discount_rate=0.01)

    ax = result.plot(vmin=10, vmax=200)

    floor, ceiling = ax.collections[0].get_clim()
    assert (floor, ceiling) == (10, 200)


def test_plot_options_typeddict_tracks_plot_clv_signature():
    # CLVResult.plot forwards **kwargs typed as PlotCLVOptions; if plot_clv
    # grows or renames an option and the TypedDict is not updated, the option
    # silently stops being discoverable. Fail here instead.
    import inspect

    from clvkit.plotting import PlotCLVOptions, plot_clv

    keyword_only = {
        name
        for name, param in inspect.signature(plot_clv).parameters.items()
        if param.kind is inspect.Parameter.KEYWORD_ONLY
    }
    assert set(PlotCLVOptions.__annotations__) == keyword_only


def test_plot_forwards_marker_styling_through_scatter_kwargs():
    result = _fitted_cdnow_clv().predict(horizon=12, discount_rate=0.01)

    ax = result.plot(scatter_kwargs={"s": 9, "alpha": 0.3})

    collection = ax.collections[0]
    assert collection.get_sizes().tolist() == [9]
    assert collection.get_alpha() == 0.3


def test_result_repr_carries_the_terms_of_the_calculation():
    result = _fitted_cdnow_clv().predict(horizon=12, discount_rate=0.01)

    assert "horizon=12" in repr(result)
    assert "discount_rate=0.01" in repr(result)
