"""Parametric bootstrap — parameter uncertainty for the MLE transaction models.

The BG/NBD-family models hand back four point estimates and nothing beside
them, so a parameter the data pins down looks exactly like one it barely
constrains. On the canonical CDNOW cohort ``a`` and ``b`` are uncertain at
roughly ±40–50% while ``r`` and ``alpha`` are tight, and today's API cannot
show that.

A *parametric* bootstrap is the honest instrument here (issue #125): from the
fitted parameters, simulate many synthetic customer bases, refit the same
model on each, and read the spread of the refits. Asymptotic Hessian standard
errors were rejected — they mislead on this weakly identified surface. The
nonparametric customer resample was rejected too — the maintainer asked for the
model-based version, which lets the small-cohort display note point at a number
tied to the model rather than to a reshuffle of the observed rows.

This module is model-agnostic: it asks a fitted model only for its ``params_``
and a ``_simulate(rng)`` that draws a synthetic ``CustomerBase`` from those
parameters, then refits via the model's own ``fit``. So the same engine serves
BG/NBD and MBG/NBD (and any later model with those two things) without knowing
which it holds.
"""

from typing import TYPE_CHECKING, Protocol

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from clvkit.customer_base import CustomerBase

_COLUMNS = ("estimate", "se", "ci_low", "ci_high")


class Bootstrappable(Protocol):
    """What the engine needs from a fitted model: its estimates and a simulator.

    Structural, so BG/NBD and MBG/NBD qualify without inheriting anything — the
    bootstrap only ever reads ``params_``, draws a synthetic base, and refits.
    """

    @property
    def params_(self) -> "pd.Series | None": ...

    def fit(self, cb: "CustomerBase") -> "Bootstrappable": ...

    def _simulate(self, rng: np.random.Generator) -> "CustomerBase": ...


class ParameterUncertainty:
    """Per-parameter estimate, bootstrap standard error, and percentile interval.

    Rich by default like every clvkit result — it draws itself — but
    ``to_pandas()`` is the universal escape hatch. Indexed by parameter name, in
    the model's own order, with columns ``estimate, se, ci_low, ci_high``.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        model_name: str,
        n_replicates: int,
        confidence: float,
        seed: int | None,
    ) -> None:
        self._data = data
        self.model_name = model_name
        self.n_replicates = n_replicates
        self.confidence = confidence
        self.seed = seed

    def to_pandas(self) -> pd.DataFrame:
        """The four columns, indexed by parameter name."""
        return self._data.copy()

    def to_json(self) -> str:
        """Parameter-keyed JSON, the same shape as ``to_pandas()``."""
        return self.to_pandas().to_json(orient="index")

    def plot(self, ax: "Axes | None" = None, **kwargs) -> "Axes":
        """Draw each estimate with its bootstrap interval."""
        # Imported here, not at module scope, so `import clvkit` doesn't drag in
        # pyplot for anyone who only ever calls to_pandas().
        from clvkit.plotting import plot_parameter_uncertainty

        return plot_parameter_uncertainty(self, ax=ax, **kwargs)

    def __repr__(self) -> str:
        return (
            f"<ParameterUncertainty {self.model_name}: {len(self._data)} params, "
            f"{self.n_replicates} replicates, {self.confidence:.0%} CI, "
            f"seed={self.seed}>"
        )


def parametric_bootstrap(
    model: Bootstrappable,
    *,
    n: int,
    seed: int | None,
    confidence: float,
) -> ParameterUncertainty:
    """Bootstrap the parameter uncertainty of a fitted ``model``.

    ``n`` synthetic bases are drawn from the fitted parameters and the same
    model class refit on each; the spread of the refits gives the standard error
    and a percentile confidence interval. ``estimate`` is the original fit's
    value, not the bootstrap mean. Reproducible from ``seed``.
    """
    # The public entry point (each model's `.parameter_uncertainty()`) raises
    # that model's own not-fitted error before reaching here; this is the
    # engine's internal precondition, not a second user-facing message.
    estimate = model.params_
    if estimate is None:
        raise RuntimeError("parametric_bootstrap requires a fitted model")
    if int(n) != n or n < 2:
        raise ValueError(f"n must be a whole number >= 2, got {n!r}")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence!r}")

    n = int(n)
    param_names = list(estimate.index)
    rng = np.random.default_rng(seed)

    draws = np.empty((n, len(param_names)), dtype=float)
    for i in range(n):
        refit = type(model)().fit(model._simulate(rng))
        draws[i] = refit.params_.to_numpy(dtype=float)  # type: ignore[union-attr]

    # A refit that lands on a non-finite parameter (a degenerate synthetic base
    # the optimiser could not leave) would poison every percentile; drop those
    # replicates rather than let one NaN erase the interval.
    keep = np.isfinite(draws).all(axis=1)
    draws = draws[keep]
    if len(draws) < 2:
        raise RuntimeError(
            "parametric bootstrap produced fewer than two usable replicates; "
            "the base may be too small or too degenerate to resample"
        )

    tail = (1 - confidence) / 2
    lo, hi = np.percentile(draws, [100 * tail, 100 * (1 - tail)], axis=0)

    data = pd.DataFrame(
        {
            "estimate": estimate.to_numpy(dtype=float),
            "se": draws.std(axis=0, ddof=1),
            "ci_low": lo,
            "ci_high": hi,
        },
        index=pd.Index(param_names, name="parameter"),
        columns=list(_COLUMNS),
    )

    return ParameterUncertainty(
        data,
        model_name=type(model).__name__,
        n_replicates=len(draws),
        confidence=confidence,
        seed=seed,
    )
