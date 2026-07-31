"""CohortMatrix — the descriptive cohort pivot over a raw transaction log.

The log below is hand-built so every cell is checkable by eye. Monthly
periods, three cohorts, an observation window ending 1998-03-15:

    customer  date        amount   cohort   period_number
    A         1998-01-05    10     1998-01        0
    A         1998-02-10    20     1998-01        1
    A         1998-03-01    30     1998-01        2
    B         1998-01-20    40     1998-01        0
    B         1998-03-15    50     1998-01        2
    C         1998-02-02    60     1998-02        0
    C         1998-02-20     5     1998-02        0
    D         1998-03-10    70     1998-03        0

C never buys again, so cohort 1998-02 has an *observed* zero at period 1;
cohorts 1998-02 and 1998-03 have no period-2 (and 1998-03 no period-1) cell
at all — the incomplete triangle, which must read as missing, not as zero.
"""

import json

import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

from clvkit import CohortMatrix
from clvkit._result import Result


@pytest.fixture
def log() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["A", "A", "A", "B", "B", "C", "C", "D"],
            "date": pd.to_datetime(
                [
                    "1998-01-05",
                    "1998-02-10",
                    "1998-03-01",
                    "1998-01-20",
                    "1998-03-15",
                    "1998-02-02",
                    "1998-02-20",
                    "1998-03-10",
                ]
            ),
            "amount": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 5.0, 70.0],
        }
    )


def test_retention_matrix_counts_active_customers_per_cohort_period(log):
    frame = CohortMatrix.from_transactions(log, period="M").to_pandas()

    assert list(frame.index.astype(str)) == ["1998-01", "1998-02", "1998-03"]
    assert list(frame.columns) == [0, 1, 2]
    # A and B both start in January; only A buys in February; both buy in March.
    assert list(frame.iloc[0]) == [2, 1, 2]


def test_a_customer_buying_twice_in_one_period_is_counted_once(log):
    frame = CohortMatrix.from_transactions(log, period="M").to_pandas()

    # C bought twice in 1998-02 — one active customer, not two.
    assert frame.loc[pd.Period("1998-02", "M"), 0] == 1


def test_an_observed_period_with_no_activity_is_zero_not_missing(log):
    frame = CohortMatrix.from_transactions(log, period="M").to_pandas()

    # The 1998-02 cohort was observable through 1998-03; C simply didn't buy.
    assert frame.loc[pd.Period("1998-02", "M"), 1] == 0


def test_unobserved_cells_of_the_incomplete_triangle_are_missing(log):
    frame = CohortMatrix.from_transactions(log, period="M").to_pandas()

    assert np.isnan(frame.loc[pd.Period("1998-02", "M"), 2])
    assert np.isnan(frame.loc[pd.Period("1998-03", "M"), 1])
    assert np.isnan(frame.loc[pd.Period("1998-03", "M"), 2])


def test_revenue_matrix_sums_amounts_per_cohort_period(log):
    frame = CohortMatrix.from_transactions(
        log, period="M", metric="revenue"
    ).to_pandas()

    assert list(frame.iloc[0]) == [50.0, 20.0, 80.0]  # (10+40), 20, (30+50)
    assert frame.loc[pd.Period("1998-02", "M"), 0] == 65.0  # 60 + 5
    assert frame.loc[pd.Period("1998-02", "M"), 1] == 0.0
    assert np.isnan(frame.loc[pd.Period("1998-03", "M"), 1])


def test_relative_matrix_divides_each_cohort_by_its_own_first_period(log):
    matrix = CohortMatrix.from_transactions(log, period="M")

    frame = matrix.to_pandas(relative=True)

    assert list(frame.iloc[0]) == [1.0, 0.5, 1.0]
    assert frame.loc[pd.Period("1998-02", "M"), 1] == 0.0
    assert np.isnan(frame.loc[pd.Period("1998-03", "M"), 1])


def test_period_argument_changes_the_cohort_grain(log):
    frame = CohortMatrix.from_transactions(log, period="Y").to_pandas()

    # Every customer's first purchase falls in 1998 — one cohort, one period.
    assert frame.shape == (1, 1)
    assert frame.iloc[0, 0] == 4


def test_to_pandas_returns_a_copy(log):
    matrix = CohortMatrix.from_transactions(log, period="M")

    matrix.to_pandas().iloc[0, 0] = 99.0

    assert matrix.to_pandas().iloc[0, 0] == 2


def test_to_json_keys_cohorts_by_label_and_writes_missing_cells_as_null(log):
    matrix = CohortMatrix.from_transactions(log, period="M")

    payload = json.loads(matrix.to_json())

    assert payload["1998-01"] == {"0": 2, "1": 1, "2": 2}
    assert payload["1998-03"] == {"0": 1, "1": None, "2": None}


def test_custom_column_names_are_honoured():
    log = pd.DataFrame(
        {
            "cust": ["A", "A", "B"],
            "when": pd.to_datetime(["1998-01-05", "1998-02-05", "1998-02-07"]),
            "spend": [1.0, 2.0, 3.0],
        }
    )

    frame = CohortMatrix.from_transactions(
        log,
        period="M",
        metric="revenue",
        customer_id_col="cust",
        datetime_col="when",
        amount_col="spend",
    ).to_pandas()

    assert frame.loc[pd.Period("1998-01", "M"), 1] == 2.0
    assert frame.loc[pd.Period("1998-02", "M"), 0] == 3.0


def test_retention_needs_no_amount_column(log):
    timing_only = log[["customer_id", "date"]]

    frame = CohortMatrix.from_transactions(timing_only, period="M").to_pandas()

    assert list(frame.iloc[0]) == [2, 1, 2]


def test_revenue_without_an_amount_column_raises(log):
    with pytest.raises(ValueError, match="amount_col"):
        CohortMatrix.from_transactions(log, metric="revenue", amount_col=None)


def test_unknown_metric_raises(log):
    with pytest.raises(ValueError, match="metric"):
        CohortMatrix.from_transactions(log, metric="churn")


def test_an_empty_log_raises_rather_than_producing_an_empty_matrix():
    empty = pd.DataFrame({"customer_id": [], "date": [], "amount": []})

    with pytest.raises(ValueError, match="empty"):
        CohortMatrix.from_transactions(empty)


def test_string_dates_are_parsed():
    log = pd.DataFrame(
        {
            "customer_id": ["A", "A"],
            "date": ["1998-01-05", "1998-02-05"],
            "amount": [1.0, 2.0],
        }
    )

    frame = CohortMatrix.from_transactions(log, period="M").to_pandas()

    assert list(frame.iloc[0]) == [1, 1]


def test_plot_draws_a_heatmap(log):
    matrix = CohortMatrix.from_transactions(log, period="M")

    ax = matrix.plot()

    assert isinstance(ax, Axes)
    assert len(ax.images) == 1
    assert ax.get_ylabel() == "Cohort"
    assert [label.get_text() for label in ax.get_yticklabels()] == [
        "1998-01",
        "1998-02",
        "1998-03",
    ]


def test_plot_accepts_an_existing_axes(log):
    import matplotlib.pyplot as plt

    matrix = CohortMatrix.from_transactions(log, period="M")
    _, ax = plt.subplots()

    assert matrix.plot(ax=ax) is ax


def test_cohort_matrix_satisfies_the_result_protocol(log):
    assert isinstance(CohortMatrix.from_transactions(log, period="M"), Result)


def test_repr_names_the_metric_and_shape(log):
    matrix = CohortMatrix.from_transactions(log, period="M")

    assert "retention" in repr(matrix)
    assert "3" in repr(matrix)
