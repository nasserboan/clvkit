import pandas as pd
import pytest

from clvkit import CustomerBase


def _log(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["customer_id", "date", "amount"])


def test_repeat_customer_rfm():
    transactions = _log(
        [
            ("A", "2020-01-01", 10),
            ("A", "2020-01-05", 20),
            ("A", "2020-01-10", 30),
            ("A", "2020-01-15", 40),
            ("A", "2020-01-20", 50),
        ]
    )

    cb = CustomerBase.from_transactions(transactions)
    row = cb.to_pandas().loc["A"]

    assert row["frequency"] == 4
    assert row["recency"] == 19
    assert row["T"] == 19
    assert row["monetary_value"] == 35.0


def test_single_purchase_customer_is_all_zero_except_T():
    transactions = _log(
        [
            ("A", "2020-01-01", 10),
            ("A", "2020-01-20", 20),
            ("B", "2020-01-01", 100),
        ]
    )

    cb = CustomerBase.from_transactions(transactions)
    row = cb.to_pandas().loc["B"]

    assert row["frequency"] == 0
    assert row["recency"] == 0
    assert row["monetary_value"] == 0
    assert row["T"] == 19


def test_same_time_unit_transactions_collapse_and_sum():
    transactions = _log(
        [
            ("C", "2020-02-01", 10),
            ("C", "2020-02-01", 15),
            ("C", "2020-02-05", 20),
        ]
    )

    cb = CustomerBase.from_transactions(transactions)
    row = cb.to_pandas().loc["C"]

    assert row["frequency"] == 1
    assert row["recency"] == 4
    assert row["monetary_value"] == 20.0


def test_on_negative_net_keeps_only_positive_net_buckets():
    transactions = _log(
        [
            ("D", "2020-01-01", 10),
            ("D", "2020-01-01", -3),
            ("D", "2020-01-10", 5),
            ("D", "2020-01-10", -8),
        ]
    )

    cb = CustomerBase.from_transactions(transactions, on_negative="net")
    row = cb.to_pandas().loc["D"]

    assert row["frequency"] == 0
    assert row["recency"] == 0
    assert row["monetary_value"] == 0


def test_on_negative_net_drops_customer_with_no_positive_bucket():
    transactions = _log(
        [
            ("D", "2020-01-01", 10),
            ("D", "2020-01-01", -15),
            ("Z", "2020-01-05", 100),
        ]
    )

    cb = CustomerBase.from_transactions(transactions, on_negative="net")
    summary = cb.to_pandas()

    assert "D" not in summary.index
    assert "Z" in summary.index


def test_on_negative_drop_discards_negative_rows_pre_aggregation():
    transactions = _log(
        [
            ("E", "2020-01-01", 10),
            ("E", "2020-01-10", -5),
            ("E", "2020-01-20", 7),
        ]
    )

    cb = CustomerBase.from_transactions(transactions, on_negative="drop")
    row = cb.to_pandas().loc["E"]

    assert row["frequency"] == 1
    assert row["recency"] == 19
    assert row["monetary_value"] == 7.0


def test_on_negative_raise_rejects_negative_amounts():
    transactions = _log([("F", "2020-01-01", -1)])

    with pytest.raises(ValueError):
        CustomerBase.from_transactions(transactions, on_negative="raise")


def test_observation_period_end_before_last_transaction_is_refused():
    transactions = _log([("A", "2020-01-01", 10), ("A", "2020-01-20", 20)])

    with pytest.raises(ValueError, match="does not truncate the log"):
        CustomerBase.from_transactions(
            transactions, observation_period_end="2020-01-10"
        )


def test_observation_period_end_after_last_transaction_extends_T():
    transactions = _log([("A", "2020-01-01", 10), ("A", "2020-01-11", 20)])

    cb = CustomerBase.from_transactions(
        transactions, observation_period_end="2020-01-21"
    )
    row = cb.to_pandas().loc["A"]

    assert row["recency"] == 10
    assert row["T"] == 20


def test_amount_col_none_omits_monetary_value():
    transactions = pd.DataFrame(
        {
            "customer_id": ["A", "A"],
            "date": ["2020-01-01", "2020-01-10"],
        }
    )

    cb = CustomerBase.from_transactions(transactions, amount_col=None)

    assert cb.has_monetary is False
    assert "monetary_value" not in cb.to_pandas().columns


def test_observation_period_end_defaults_to_latest_log_date():
    transactions = _log(
        [
            ("A", "2020-01-01", 10),
            ("A", "2020-01-10", 20),
        ]
    )

    cb = CustomerBase.from_transactions(transactions)

    assert cb.observation_period_end == pd.Timestamp("2020-01-10")


def test_observation_period_end_is_overridable():
    transactions = _log(
        [
            ("A", "2020-01-01", 10),
            ("A", "2020-01-10", 20),
        ]
    )

    cb = CustomerBase.from_transactions(
        transactions, observation_period_end="2020-02-01"
    )
    row = cb.to_pandas().loc["A"]

    assert cb.observation_period_end == pd.Timestamp("2020-02-01")
    assert row["T"] == 31


def test_provenance_fields_exposed():
    transactions = _log([("A", "2020-01-01", 10)])

    cb = CustomerBase.from_transactions(transactions, time_unit="W", on_negative="drop")

    assert cb.time_unit == "W"
    assert cb.has_monetary is True
    assert cb.on_negative == "drop"
    assert cb.observation_period_end == pd.Timestamp("2020-01-01")


def _split_log() -> pd.DataFrame:
    # calibration_period_end = 2020-03-31, observation_period_end = 2020-06-30
    return _log(
        [
            # A: repeat purchases spanning the boundary
            ("A", "2020-01-01", 10),
            ("A", "2020-02-01", 20),
            ("A", "2020-03-01", 30),
            ("A", "2020-04-01", 40),
            ("A", "2020-05-01", 50),
            # C: single calibration purchase, no holdout activity
            ("C", "2020-02-15", 60),
            # B: born in the holdout window — must be excluded
            ("B", "2020-05-01", 100),
        ]
    )


def test_split_returns_calibration_base_and_holdout_columns():
    cb = CustomerBase.from_transactions(_split_log())

    calibration, holdout = cb.split(
        calibration_period_end="2020-03-31",
        observation_period_end="2020-06-30",
    )

    assert isinstance(calibration, CustomerBase)
    assert calibration.observation_period_end == pd.Timestamp("2020-03-31")
    assert list(holdout.columns) == [
        "frequency_holdout",
        "monetary_value_holdout",
        "duration_holdout",
    ]

    a_hold = holdout.loc["A"]
    assert a_hold["frequency_holdout"] == 2
    assert a_hold["monetary_value_holdout"] == 45.0
    assert a_hold["duration_holdout"] == 91  # 2020-03-31 -> 2020-06-30, days

    c_hold = holdout.loc["C"]
    assert c_hold["frequency_holdout"] == 0
    assert c_hold["monetary_value_holdout"] == 0.0


def test_split_excludes_customers_born_in_holdout_window():
    cb = CustomerBase.from_transactions(_split_log())

    calibration, holdout = cb.split(
        calibration_period_end="2020-03-31",
        observation_period_end="2020-06-30",
    )

    assert "B" not in calibration.to_pandas().index
    assert "B" not in holdout.index


def test_split_calibration_rfm_matches_from_transactions_at_calibration_end():
    log = _split_log()
    cal_end = "2020-03-31"

    calibration, _ = CustomerBase.from_transactions(log).split(
        calibration_period_end=cal_end,
        observation_period_end="2020-06-30",
    )

    cal_only = log[pd.to_datetime(log["date"]) <= pd.Timestamp(cal_end)]
    expected = CustomerBase.from_transactions(cal_only, observation_period_end=cal_end)

    pd.testing.assert_frame_equal(calibration.to_pandas(), expected.to_pandas())


def test_split_duration_holdout_is_in_time_unit_for_weekly():
    cb = CustomerBase.from_transactions(_split_log(), time_unit="W")

    _, holdout = cb.split(
        calibration_period_end="2020-03-31",
        observation_period_end="2020-06-30",
    )

    # 2020-03-31 -> 2020-06-30 spans 13 whole weeks (bucket-ordinal diff)
    assert holdout.loc["A", "duration_holdout"] == 13


def test_split_observation_period_end_defaults_to_base_end():
    cb = CustomerBase.from_transactions(_split_log())  # end = 2020-05-01

    _, holdout = cb.split(calibration_period_end="2020-03-31")

    # holdout window is 2020-03-31 -> 2020-05-01 (base's own end)
    assert holdout.loc["A", "duration_holdout"] == 31
    assert holdout.loc["A", "frequency_holdout"] == 2
