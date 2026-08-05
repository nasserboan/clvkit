"""The shared result contract — what every clvkit prediction hands back.

Results are rich by default (they draw themselves) but never a trap:
``to_pandas()`` is the universal escape hatch.
"""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes


@runtime_checkable
class Result(Protocol):
    """Structural contract for every clvkit result type.

    A Protocol rather than a base class: result types are unrelated to each
    other and share only these three verbs, so there is nothing to inherit.
    """

    def to_pandas(self) -> pd.DataFrame: ...

    def to_json(self) -> str: ...

    def plot(self) -> "Axes": ...


class Prediction:
    """One predicted quantity per customer — the shared return of every model.

    `BGNBD.predict`, `BGNBD.probability_alive`, and (later) the monetary models
    all answer the same shape of question: a number per customer. They all
    return this.
    """

    def __init__(
        self,
        values: pd.Series,
        *,
        name: str,
        description: str,
        plot_data: pd.DataFrame | None = None,
        plot_time_unit: str = "",
    ) -> None:
        self._values = values.rename(name)
        self.name = name
        # Human-readable, unit-aware label — the axis title when plotted.
        self.description = description
        self._plot_data = None if plot_data is None else plot_data.copy()
        self._plot_time_unit = plot_time_unit

    def to_pandas(self) -> pd.DataFrame:
        """The prediction as a one-column DataFrame indexed by customer_id."""
        return self._values.to_frame().copy()

    def to_json(self) -> str:
        """Customer-keyed JSON, the same shape as ``to_pandas()``."""
        return self.to_pandas().to_json(orient="index")

    def plot(self, ax: "Axes | None" = None, **kwargs) -> "Axes":
        """Draw the distribution of the prediction across the customer base."""
        # Imported here, not at module scope, so `import clvkit` doesn't drag
        # in pyplot for anyone who only ever calls to_pandas().
        from clvkit.plotting import plot_prediction, plot_probability_alive

        if self.name == "probability_alive" and self._plot_data is not None:
            return plot_probability_alive(self, ax=ax, **kwargs)
        return plot_prediction(self, ax=ax, **kwargs)

    def __repr__(self) -> str:
        return f"<Prediction {self.name!r} over {len(self._values)} customers>"


def values_of(prediction: Prediction) -> pd.Series:
    """The single column of a `Prediction`, whatever the model named it.

    A `Prediction` is one quantity per customer by construction, so taking it
    positionally is what the protocol actually promises. Reading it by name
    would couple a composition to `BGNBD`'s choice of column label, and the
    whole point of the protocol is that an injected model needn't share it.
    """
    return prediction.to_pandas().iloc[:, 0]
