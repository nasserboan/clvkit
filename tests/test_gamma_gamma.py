import inspect
from functools import cache
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import betaprime

from clvkit import CustomerBase, GammaGamma
from clvkit._result import Result

# Published Fader–Hardie estimates on the CDNOW 1/10 sample, from §3 of
# "The Gamma-Gamma Model of Monetary Value" (brucehardie.com/notes/025/).
PUBLISHED_P, PUBLISHED_Q, PUBLISHED_GAMMA = 6.25, 3.74, 15.44

CDNOW_SAMPLE = Path(__file__).resolve().parents[1] / "CDNOW_sample.txt"
# Fader, Hardie & Lee (2005) calibrate on the first 39 weeks of the CDNOW
# panel, i.e. everything up to and including 1997-09-30.
CDNOW_CALIBRATION_END = "1997-09-30"


@cache
def _cdnow_calibration() -> CustomerBase:
    """The CDNOW 1/10 sample (2,357 customers), 39-week calibration period.

    `time_unit="D"` because the published RFM summary collapses *same-day*
    transactions into one event; that reproduces the note's Table 1 exactly
    (946 repeat buyers, mean z̄ = $35.08). Gamma-Gamma is time-independent,
    so the unit only matters through that collapse.
    """
    log = pd.read_csv(
        CDNOW_SAMPLE,
        sep=r"\s+",
        header=None,
        names=["customer_id", "sample_id", "date", "quantity", "amount"],
    )
    log["date"] = pd.to_datetime(log["date"], format="%Y%m%d")
    full = CustomerBase.from_transactions(
        log[["customer_id", "date", "amount"]], time_unit="D"
    )
    calibration, _ = full.split(calibration_period_end=CDNOW_CALIBRATION_END)
    return calibration


def _simulated_log(
    *,
    p: float = 6.25,
    q: float = 3.74,
    gamma: float = 15.44,
    n_customers: int = 800,
    n_one_time: int = 0,
    mean_repeats: float = 2.0,
    seed: int = 20250724,
) -> pd.DataFrame:
    """A transaction log drawn from the Gamma-Gamma model itself.

    Each customer draws ν ~ gamma(q, γ), then spends z ~ gamma(p, ν) on every
    transaction — the generative story of §2 of the note, run forwards.
    """
    rng = np.random.default_rng(seed)
    nu = rng.gamma(shape=q, scale=1 / gamma, size=n_customers)
    n_repeat = 1 + rng.poisson(mean_repeats, size=n_customers)

    rows = []
    start = pd.Timestamp("2020-01-01")
    for customer, (rate, repeats) in enumerate(zip(nu, n_repeat, strict=True)):
        amounts = rng.gamma(shape=p, scale=1 / rate, size=repeats + 1)
        for offset, amount in enumerate(amounts):
            rows.append((f"c{customer:04d}", start + pd.Timedelta(days=offset), amount))
    for extra in range(n_one_time):
        rows.append((f"solo{extra:02d}", start, 42.0))

    return pd.DataFrame(rows, columns=["customer_id", "date", "amount"])


def _simulated_base(**kwargs) -> CustomerBase:
    return CustomerBase.from_transactions(_simulated_log(**kwargs))


def _log(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["customer_id", "date", "amount"])


# --- the public surface -------------------------------------------------


def test_fit_predict_returns_expected_spend_per_customer():
    cb = _simulated_base(n_customers=400)

    prediction = GammaGamma().fit(cb).predict()

    spend = prediction.to_pandas()
    assert list(spend.index) == list(cb.to_pandas().index)
    assert spend.columns.tolist() == ["expected_spend"]
    assert (spend["expected_spend"] > 0).all()


def test_predict_takes_no_horizon_argument():
    # Spend per transaction is time-independent (assumption 2 of the note),
    # so unlike the transaction models there is nothing to forecast over.
    parameters = inspect.signature(GammaGamma.predict).parameters

    assert "t" not in parameters
    assert "horizon" not in parameters


def test_fit_recovers_the_population_mean_it_was_simulated_from():
    # p and γ trade off against each other; pγ/(q−1) is what is identified.
    cb = _simulated_base()

    gg = GammaGamma().fit(cb)

    true_mean = PUBLISHED_P * PUBLISHED_GAMMA / (PUBLISHED_Q - 1)
    assert gg.population_mean() == pytest.approx(true_mean, rel=0.10)


def test_one_time_buyer_is_predicted_the_population_mean():
    # With no observed average, equation (5) collapses to equation (3).
    cb = _simulated_base(n_customers=400, n_one_time=3)

    gg = GammaGamma().fit(cb)
    spend = gg.predict().to_pandas()["expected_spend"]

    one_time = spend.loc[["solo00", "solo01", "solo02"]]
    assert one_time.tolist() == pytest.approx([gg.population_mean()] * 3)


def test_more_transactions_shift_the_estimate_towards_the_observed_average():
    # Equation (5) weights z̄ by px/(px + q − 1), so the weight grows with x.
    gg = GammaGamma().fit(_simulated_base(n_customers=400))
    # Two customers with the same average spend but different transaction
    # counts; the frequent one's estimate should sit closer to that average.
    same_average = CustomerBase.from_transactions(
        _log(
            [("rare", f"2020-01-0{day}", 200.0) for day in (1, 2, 3)]
            + [("frequent", f"2020-02-{day:02d}", 200.0) for day in range(1, 16)]
        )
    )

    spend = gg.predict(same_average).to_pandas()["expected_spend"]

    assert abs(spend["frequent"] - 200.0) < abs(spend["rare"] - 200.0)
    assert gg.population_mean() < spend["rare"] < spend["frequent"] < 200.0


def test_fit_is_invariant_to_the_currency_unit():
    # Only γ carries the money scale: p and q are shape parameters.
    log = _simulated_log(n_customers=400)
    in_cents = log.assign(amount=log["amount"] * 100.0)

    euros = GammaGamma().fit(CustomerBase.from_transactions(log))
    cents = GammaGamma().fit(CustomerBase.from_transactions(in_cents))

    assert cents.p == pytest.approx(euros.p, rel=1e-4)
    assert cents.q == pytest.approx(euros.q, rel=1e-4)
    assert cents.gamma == pytest.approx(euros.gamma * 100.0, rel=1e-4)


def test_prediction_satisfies_the_result_contract():
    prediction = GammaGamma().fit(_simulated_base(n_customers=400)).predict()

    assert isinstance(prediction, Result)


def test_prediction_round_trips_through_json():
    prediction = GammaGamma().fit(_simulated_base(n_customers=400)).predict()

    restored = pd.read_json(StringIO(prediction.to_json()), orient="index")

    assert restored["expected_spend"].to_numpy() == pytest.approx(
        prediction.to_pandas()["expected_spend"].to_numpy()
    )


# --- preconditions ------------------------------------------------------


def test_fit_rejects_a_customer_base_without_monetary_value():
    cb = CustomerBase.from_transactions(
        _log([("A", "2020-01-01", 10), ("A", "2020-01-02", 20)]), amount_col=None
    )

    with pytest.raises(ValueError, match="amount_col"):
        GammaGamma().fit(cb)


def test_fit_rejects_a_customer_base_with_no_repeat_buyers():
    cb = CustomerBase.from_transactions(
        _log([("A", "2020-01-01", 10), ("B", "2020-01-01", 20)])
    )

    with pytest.raises(ValueError, match="repeat"):
        GammaGamma().fit(cb)


def test_predict_before_fit_is_an_error():
    with pytest.raises(RuntimeError, match="not fitted"):
        GammaGamma().predict()


# --- golden test: the published CDNOW estimates -------------------------


def test_golden_cdnow_reproduces_published_estimates():
    # The headline trust signal: p = 6.25, q = 3.74, γ = 15.44 (note §3).
    gg = GammaGamma().fit(_cdnow_calibration())

    assert round(gg.p, 2) == PUBLISHED_P
    assert round(gg.q, 2) == PUBLISHED_Q
    assert round(gg.gamma, 2) == PUBLISHED_GAMMA


def test_golden_cdnow_summary_matches_the_notes_table_1():
    # Guards the fixture itself: 946 repeat buyers, mean z̄ = $35.08 (Table 1).
    summary = _cdnow_calibration().to_pandas()
    repeat = summary[summary["frequency"] > 0]

    assert len(repeat) == 946
    assert repeat["monetary_value"].mean() == pytest.approx(35.08, abs=0.005)
    assert repeat["monetary_value"].max() == pytest.approx(299.63, abs=0.005)


# --- pinpoint likelihood tests ------------------------------------------


def test_log_likelihood_matches_the_beta_prime_distribution():
    # Equation (1a) *is* a beta distribution of the second kind (note §2), and
    # scipy's `betaprime` is an independent implementation of that density. So
    # agreeing with it catches a mis-transcribed special-function term that a
    # parameter-drift test would only show as a vague wobble.
    p, q, gamma = 3.5, 2.25, 12.0
    frequency = np.array([1.0, 2.0, 5.0, 17.0])
    monetary_value = np.array([9.5, 40.0, 133.25, 3.75])

    # z̄ ~ betaprime(a = px, b = q, scale = γ/x); see the derivation of (1b).
    expected = betaprime.logpdf(
        monetary_value, p * frequency, q, scale=gamma / frequency
    ).sum()

    assert GammaGamma._log_likelihood(
        p, q, gamma, frequency, monetary_value
    ) == pytest.approx(expected)


def test_log_likelihood_at_the_published_cdnow_estimates():
    # Pinpoint: the exact sample log-likelihood (6) at (6.25, 3.74, 15.44).
    summary = _cdnow_calibration().to_pandas()
    repeat = summary[summary["frequency"] > 0]

    log_likelihood = GammaGamma._log_likelihood(
        PUBLISHED_P,
        PUBLISHED_Q,
        PUBLISHED_GAMMA,
        repeat["frequency"].to_numpy(dtype=float),
        repeat["monetary_value"].to_numpy(dtype=float),
    )

    assert log_likelihood == pytest.approx(-4055.919232049544, abs=1e-9)


class TestItDescribesItself:
    """`GammaGamma` was the one public class with no `__repr__` of its own.

    Falling back to Python's default leaks a memory address into every log
    line, traceback and notebook cell that touches the model.
    """

    def test_an_unfitted_model_says_so(self):
        assert repr(GammaGamma()) == "<GammaGamma (unfitted)>"

    def test_a_fitted_model_shows_its_parameters(self):
        text = repr(GammaGamma().fit(_simulated_base(n_customers=400)))
        assert text.startswith("<GammaGamma ")
        for name in ("p=", "q=", "gamma="):
            assert name in text, name

    def test_it_never_leaks_a_memory_address(self):
        for model in (GammaGamma(), GammaGamma().fit(_simulated_base(n_customers=400))):
            assert " object at 0x" not in repr(model)

    def test_params_matches_the_loose_attributes(self):
        gg = GammaGamma().fit(_simulated_base(n_customers=400))
        assert list(gg.params_.index) == ["p", "q", "gamma"]
        assert gg.params_["p"] == gg.p
        assert gg.params_["q"] == gg.q
        assert gg.params_["gamma"] == gg.gamma

    def test_params_is_none_before_fitting(self):
        assert GammaGamma().params_ is None

    def test_params_reads_the_same_way_bgnbd_does(self):
        # The asymmetry this closes: BGNBD kept a params_ Series while
        # GammaGamma kept three loose floats, so nothing could render both.
        from clvkit import BGNBD

        for model in (
            BGNBD().fit(_simulated_base(n_customers=400)),
            GammaGamma().fit(_simulated_base(n_customers=400)),
        ):
            assert isinstance(model.params_, pd.Series)
            assert model.params_.notna().all()
