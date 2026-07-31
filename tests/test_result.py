import json

import pandas as pd
from matplotlib.axes import Axes

from clvkit._result import Prediction, Result


def _prediction() -> Prediction:
    values = pd.Series(
        [0.5, 1.5, 2.5], index=pd.Index(["A", "B", "C"], name="customer_id")
    )
    return Prediction(
        values, name="expected_purchases", description="Expected purchases"
    )


def test_prediction_to_pandas_is_per_customer():
    prediction = _prediction()

    frame = prediction.to_pandas()

    assert list(frame.columns) == ["expected_purchases"]
    assert frame.index.name == "customer_id"
    assert frame.loc["B", "expected_purchases"] == 1.5


def test_prediction_to_pandas_returns_a_copy():
    prediction = _prediction()

    frame = prediction.to_pandas()
    frame.loc["B", "expected_purchases"] = 99.0

    assert prediction.to_pandas().loc["B", "expected_purchases"] == 1.5


def test_prediction_to_json_round_trips_to_the_same_values():
    prediction = _prediction()

    payload = json.loads(prediction.to_json())

    assert payload == {
        "A": {"expected_purchases": 0.5},
        "B": {"expected_purchases": 1.5},
        "C": {"expected_purchases": 2.5},
    }


def test_prediction_plot_returns_matplotlib_axes():
    prediction = _prediction()

    ax = prediction.plot()

    assert isinstance(ax, Axes)
    assert ax.get_xlabel() == "Expected purchases"


def test_prediction_satisfies_the_result_protocol():
    assert isinstance(_prediction(), Result)
