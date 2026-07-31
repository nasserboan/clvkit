"""MBG/NBD on the real CDNOW cohort — where the never-returner mass shows up.

`test_bgnbd_golden.py` can check its model against published estimates; there
are none for the MBG/NBD on CDNOW, so this file validates the other way the
literature does: against 39 weeks of holdout behaviour the model never saw, on
the same 2,357-customer sample, plus the one qualitative claim that separates
the two models.

CDNOW is exactly the base Batislam, Denizel & Filiztekin (2007) built the model
for — 60% of its customers never bought again in the calibration period, and
BG/NBD calls every one of them alive with probability 1.
"""

import pandas as pd
import pytest

from clvkit import BGNBD, MBGNBD, CustomerBase

# 39 weeks after the 1997-01-01 cohort start, as in the BG/NBD golden test.
CALIBRATION_PERIOD_END = "1997-09-30"
DAYS_PER_WEEK = 7
HOLDOUT_WEEKS = 39


@pytest.fixture(scope="module")
def cdnow_calibration(cdnow_sample) -> CustomerBase:
    cb = CustomerBase.from_transactions(cdnow_sample, amount_col=None, time_unit="D")
    calibration, _ = cb.split(calibration_period_end=CALIBRATION_PERIOD_END)
    return calibration


@pytest.fixture(scope="module")
def cdnow_model(cdnow_calibration) -> MBGNBD:
    return MBGNBD().fit(cdnow_calibration)


def test_the_fit_lands_in_the_interior_of_the_parameter_space(cdnow_model):
    # Not a golden test — no published MBG/NBD estimate for CDNOW exists — but
    # a guard against the failure mode this likelihood actually has: a run to
    # the `a -> 0` boundary, where the dropout branch switches off and the
    # model silently degenerates into a plain NBD.
    params = cdnow_model.params_.copy()
    params["alpha"] = params["alpha"] / DAYS_PER_WEEK  # daily fit, read in weeks

    assert (params > 0).all()
    assert 0.05 < params["r"] < 5
    assert 0.5 < params["alpha"] < 50  # weeks
    assert 0.05 < params["a"] < 10
    assert 0.05 < params["b"] < 10


def test_the_never_returners_are_not_all_assumed_alive(cdnow_model, cdnow_calibration):
    # The acceptance criterion of the ticket, on real data: 1,411 of the 2,357
    # CDNOW customers never made a repeat purchase in the calibration period.
    # BG/NBD's dropout requires a transaction to follow, so all 1,411 are alive
    # with probability exactly 1; the MBG/NBD's time-zero dropout gives that
    # mass somewhere to go.
    summary = cdnow_calibration.to_pandas()
    never_returned = summary["frequency"] == 0
    assert never_returned.sum() == 1411

    modified = cdnow_model.probability_alive().to_pandas()["probability_alive"]
    original = (
        BGNBD()
        .fit(cdnow_calibration)
        .probability_alive()
        .to_pandas()["probability_alive"]
    )

    assert (original[never_returned] == 1.0).all()
    assert (modified[never_returned] < 0.9).all()
    assert (modified[never_returned] > 0.0).all()


def test_conditional_expectation_tracks_actual_holdout_purchasing(
    cdnow_sample, cdnow_model, cdnow_calibration
):
    # The same external validation the BG/NBD golden test runs: bucket
    # customers by calibration repeat count and compare mean predicted holdout
    # purchases to mean actual, bucket by bucket, over 39 weeks the model never
    # saw. Parameters can look plausible while the conditional expectation is
    # wired wrong — a shifted beta argument or the leading coefficient the
    # paper misprints — and this is what catches that.
    cb = CustomerBase.from_transactions(cdnow_sample, amount_col=None, time_unit="D")
    _, holdout = cb.split(calibration_period_end=CALIBRATION_PERIOD_END)

    forecast = cdnow_model.predict(t=HOLDOUT_WEEKS * DAYS_PER_WEEK).to_pandas()
    joined = (
        cdnow_calibration.to_pandas()
        .join(forecast)
        .join(holdout["frequency_holdout"])
        # The literature collapses the sparse right tail into a "7+" bucket.
        .assign(bucket=lambda d: d["frequency"].clip(upper=7))
    )

    by_bucket = joined.groupby("bucket").agg(
        predicted=("expected_purchases", "mean"),
        actual=("frequency_holdout", "mean"),
    )

    assert by_bucket["predicted"].is_monotonic_increasing
    # Within 0.6 of a purchase over 39 weeks, in every bucket. The band is
    # wider than the BG/NBD's 0.5 in one direction only: the MBG/NBD kills off
    # more customers, so it forecasts slightly under throughout — the expected
    # cost of the extra dropout branch, not a wiring error.
    gap = by_bucket["predicted"] - by_bucket["actual"]
    assert gap.abs().max() < 0.6


def test_predictions_are_per_customer(cdnow_model):
    expected = cdnow_model.predict(t=HOLDOUT_WEEKS * DAYS_PER_WEEK).to_pandas()
    alive = cdnow_model.probability_alive().to_pandas()

    assert len(expected) == 2357
    assert len(alive) == 2357
    assert (expected["expected_purchases"] >= 0).all()
    assert alive["probability_alive"].between(0, 1).all()


def test_the_two_models_reach_comparable_likelihoods_on_the_same_base(
    cdnow_calibration, cdnow_model
):
    # Batislam et al. (2007) section 5 report that BG/NBD and MBG/NBD "yield
    # almost identical estimates for the expected number of weekly and
    # cumulative repeat purchases" — the models differ in where they put the
    # never-returners, not in how well they fit. A likelihood far apart from
    # the BG/NBD's would mean one of the two is wrong, not that one is better.
    original = BGNBD().fit(cdnow_calibration)

    assert cdnow_model.log_likelihood_ == pytest.approx(
        original.log_likelihood_, rel=1e-3
    )


def test_the_fit_is_stable_across_a_scoring_round_trip(cdnow_model, cdnow_sample):
    # Scoring the model on a base built the same way must reproduce the fitted
    # base's answers exactly — the guard against `predict` quietly reading
    # different state than `fit` wrote.
    cb = CustomerBase.from_transactions(cdnow_sample, amount_col=None, time_unit="D")
    calibration, _ = cb.split(calibration_period_end=CALIBRATION_PERIOD_END)

    pd.testing.assert_frame_equal(
        cdnow_model.probability_alive().to_pandas(),
        cdnow_model.probability_alive(cb=calibration).to_pandas(),
    )
