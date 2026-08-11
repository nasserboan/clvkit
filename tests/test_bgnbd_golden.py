"""The trust signal: reproduce the published CDNOW BG/NBD estimates.

Fader, Hardie & Lee (2005) section 7 calibrate the BG/NBD on a 1/10 systematic
sample of the CDNOW cohort — 2,357 customers — over the first 39 of the 78
weeks, and report ``r = .243, alpha = 4.414, a = .793, b = 2.426`` (the Excel
worksheet screenshot in their Figure 1 shows exactly these four cells).

**On time units.** The paper answers two different questions about time, and
answers them differently. Purchases are collapsed at the data's own *daily*
resolution — CDNOW records a date, and two orders on one date are one shopping
trip — while time itself is measured continuously *in weeks*: for customer *i*,
"``T_i = 39 - time of first purchase``". `CustomerBase` says exactly that with
``time_unit="W", collapse="D"``: collapse by day, report in weeks.

``time_unit="W"`` alone would say something else, and something wrong. It would
bucket at weekly grain, merging a Monday and a Wednesday purchase into one
event — a coarser sufficient statistic than the published fit used, and one
that loses the most from the most frequent buyers. It fits
``r = .291, alpha = 6.852, a = .665, b = 2.320``, and `CustomerBase` warns
when you ask for it without saying you meant it.
"""

import pandas as pd
import pytest
from matplotlib.axes import Axes

from clvkit import BGNBD, CustomerBase

# Fader, Hardie & Lee (2005), section 7 / Figure 1.
PUBLISHED = {"r": 0.243, "alpha": 4.414, "a": 0.793, "b": 2.426}
PUBLISHED_LOG_LIKELIHOOD = -9582.4

# 39 weeks after the 1997-01-01 cohort start: the paper's calibration period.
CALIBRATION_PERIOD_END = "1997-09-30"

# The paper's holdout window: weeks 40-78, the behaviour the fit never saw.
# In the base's own unit, because it is now summarised in weeks.
HOLDOUT_WEEKS = 39


@pytest.fixture(scope="module")
def cdnow_calibration(cdnow_sample) -> CustomerBase:
    # amount_col=None because BG/NBD is a timing-only model. It used to be a
    # workaround as well — the default netting silently dropped the eight
    # customers whose only transaction is $0.00 — but timing no longer
    # depends on the monetary policy, so passing the amount column now
    # produces this exact summary too.
    cb = CustomerBase.from_transactions(
        cdnow_sample, amount_col=None, time_unit="W", collapse="D"
    )
    calibration, _ = cb.split(calibration_period_end=CALIBRATION_PERIOD_END)
    return calibration


@pytest.fixture(scope="module")
def cdnow_model(cdnow_calibration) -> BGNBD:
    return BGNBD().fit(cdnow_calibration)


def test_the_calibration_base_is_the_published_1_in_10_sample(cdnow_calibration):
    assert len(cdnow_calibration.to_pandas()) == 2357


def test_the_amount_column_no_longer_moves_the_timing(cdnow_sample, cdnow_calibration):
    """The regression test for the silent CDNOW drop.

    The default netting used to erase the eight customers whose only
    transaction is $0.00 from frequency, recency and T — 2,349 rows instead
    of 2,357, and a fit that moved with them. Timing is policy-independent
    now, and the netting says what it excluded from spend instead.
    """
    with pytest.warns(UserWarning, match="netted 8 of"):
        with_amounts = CustomerBase.from_transactions(
            cdnow_sample, time_unit="W", collapse="D"
        )
    calibration, _ = with_amounts.split(calibration_period_end=CALIBRATION_PERIOD_END)

    timing = ["frequency", "recency", "T"]
    pd.testing.assert_frame_equal(
        calibration.to_pandas()[timing],
        cdnow_calibration.to_pandas()[timing],
    )


def test_fitted_parameters_reproduce_the_published_estimates(cdnow_model):
    params = cdnow_model.params_

    fitted = {
        "r": params["r"],
        "alpha": params["alpha"],
        "a": params["a"],
        "b": params["b"],
    }

    for name, published in PUBLISHED.items():
        # The paper reports three decimal places, so agreement to within half a
        # unit in the last published digit is the strongest claim available.
        assert fitted[name] == pytest.approx(published, abs=5e-4), (
            f"{name}: fitted {fitted[name]:.5f} vs published {published}"
        )


def test_maximised_log_likelihood_matches_the_published_value(cdnow_model):
    # The log-likelihood, unlike the parameters, is not scale-free: eq. (6)
    # carries a factor lambda^x with units of time^-x, so it is only comparable
    # to the published value when the base is already reported in the paper's
    # unit. It is — no Jacobian to apply.
    assert cdnow_model.log_likelihood_ == pytest.approx(
        PUBLISHED_LOG_LIKELIHOOD, abs=0.1
    )


def test_predictions_on_the_published_fit_are_per_customer(
    cdnow_model, cdnow_calibration
):
    # 39 weeks of holdout, in the base's own (weekly) time unit.
    expected = cdnow_model.predict(t=HOLDOUT_WEEKS).to_pandas()
    alive = cdnow_model.probability_alive().to_pandas()

    assert len(expected) == 2357
    assert len(alive) == 2357
    assert (expected["expected_purchases"] >= 0).all()
    assert alive["probability_alive"].between(0, 1).all()


def test_probability_alive_plot_renders_on_cdnow(cdnow_model):
    ax = cdnow_model.probability_alive().plot()

    assert isinstance(ax, Axes)
    assert ax.get_xlabel() == "Weeks since last purchase"
    assert ax.get_ylabel() == "P(alive) at observation end"
    assert ax.figure.axes[1].get_ylabel() == "Repeat purchases (frequency)"


def test_conditional_expectation_tracks_actual_holdout_purchasing(
    cdnow_sample, cdnow_model, cdnow_calibration
):
    # The paper's own validation of eq. (10): bucket customers by their number
    # of calibration-period repeat transactions and compare mean predicted
    # holdout purchases to mean actual, bucket by bucket. Parameters can print
    # correctly while eq. (10) is wired wrong; this is what catches that,
    # against 39 weeks of behaviour the model never saw.
    cb = CustomerBase.from_transactions(
        cdnow_sample, amount_col=None, time_unit="W", collapse="D"
    )
    _, holdout = cb.split(calibration_period_end=CALIBRATION_PERIOD_END)

    forecast = cdnow_model.predict(t=HOLDOUT_WEEKS).to_pandas()
    joined = (
        cdnow_calibration.to_pandas()
        .join(forecast)
        .join(holdout["frequency_holdout"])
        # The paper collapses the sparse right tail into a "7+" bucket.
        .assign(bucket=lambda d: d["frequency"].clip(upper=7))
    )

    by_bucket = joined.groupby("bucket").agg(
        predicted=("expected_purchases", "mean"),
        actual=("frequency_holdout", "mean"),
    )

    assert by_bucket["predicted"].is_monotonic_increasing
    # Within half a purchase over 39 weeks, in every bucket.
    assert (by_bucket["predicted"] - by_bucket["actual"]).abs().max() < 0.5
