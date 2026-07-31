"""Every result's ``.plot()`` delegates here.

One flat module, one matplotlib backend: restyling clvkit's charts means
editing this file, not hunting ``plot`` methods across the package.
"""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from clvkit._result import Prediction
    from clvkit.clv.independence import IndependenceCheck
    from clvkit.cohort.matrix import CohortMatrix
    from clvkit.cohort.survival import SurvivalCurve


def plot_prediction(
    prediction: "Prediction",
    *,
    ax: "Axes | None" = None,
    bins: int = 30,
    **kwargs,
) -> "Axes":
    """Histogram the prediction across the customer base.

    A per-customer prediction is a distribution, and its shape is the thing an
    analyst reads first — how many customers are near zero, where the tail is.
    """
    if ax is None:
        _, ax = plt.subplots()

    frame = prediction.to_pandas()
    ax.hist(frame[prediction.name], bins=bins, **kwargs)
    ax.set_xlabel(prediction.description)
    ax.set_ylabel("Customers")
    ax.set_title(prediction.description)
    return ax


def plot_independence(
    check: "IndependenceCheck",
    *,
    ax: "Axes | None" = None,
    max_frequency: int = 7,
    **kwargs,
) -> "Axes":
    """Figure 4 of Fader, Hardie & Lee (2005) — spend by repeat-purchase count.

    The paper's own way of eyeballing the independence assumption behind CLV:
    if average transaction value really is independent of the transaction
    process, the boxes sit at roughly the same height and the spread *within*
    each one dwarfs the drift *between* them.
    """
    if ax is None:
        _, ax = plt.subplots()

    grouped = check.grouped_spend(max_frequency)
    ax.boxplot(list(grouped.values()), tick_labels=list(grouped.keys()), **kwargs)
    ax.set_xlabel("Repeat purchases")
    ax.set_ylabel("Average transaction value")
    ax.set_title(
        f"Spend by repeat-purchase count "
        f"(rho={check.spearman_rho:.2f}, eta2={check.eta_squared:.2f})"
    )
    return ax


def plot_cohort_matrix(
    matrix: "CohortMatrix",
    *,
    ax: "Axes | None" = None,
    relative: bool = True,
    cmap: str = "viridis",
    annotate: bool = True,
    colorbar: bool = True,
    **kwargs,
) -> "Axes":
    """Draw a cohort matrix as the standard triangular heatmap.

    Unobserved cells — the periods a young cohort hasn't lived through yet —
    are drawn in flat grey and left unlabelled, so the eye never reads them
    as a cohort that dropped to zero.
    """
    if ax is None:
        _, ax = plt.subplots()

    frame = matrix.to_pandas(relative=relative)
    values = frame.to_numpy(dtype=float)

    colormap = plt.get_cmap(cmap).copy()
    colormap.set_bad(color="0.9")
    image = ax.imshow(
        np.ma.masked_invalid(values), aspect="auto", cmap=colormap, **kwargs
    )

    ax.set_xticks(range(len(frame.columns)), [str(c) for c in frame.columns])
    ax.set_yticks(range(len(frame.index)), [str(i) for i in frame.index])
    ax.set_xlabel(f"Periods since cohort start ({matrix.period})")
    ax.set_ylabel("Cohort")

    label = _cohort_value_label(matrix.metric, relative)
    ax.set_title(label)
    if colorbar:
        ax.figure.colorbar(image, ax=ax, label=label)

    if annotate:
        # Label against the local background so text stays readable at both
        # ends of the colour ramp.
        threshold = np.nanmean(values) if np.isfinite(values).any() else 0.0
        fmt = "{:.0%}" if relative else "{:,.0f}"
        for row, col in zip(*np.where(np.isfinite(values)), strict=True):
            value = values[row, col]
            ax.text(
                col,
                row,
                fmt.format(value),
                ha="center",
                va="center",
                fontsize="small",
                color="white" if value < threshold else "black",
            )

    return ax


def plot_survival_curve(
    curve: "SurvivalCurve",
    *,
    ax: "Axes | None" = None,
    marker: str = "o",
    **kwargs,
) -> "Axes":
    """Draw model-based survival against cohort age.

    Each marker is one cohort, placed at the age it had reached by the end of
    the observation window — so the line is read across cohorts, not along one.
    The y-axis is pinned to 0–1 because the quantity is a share, and a curve
    auto-scaled to a narrow band exaggerates decay that isn't there.
    """
    if ax is None:
        _, ax = plt.subplots()

    frame = curve.to_pandas().sort_values("age")
    ax.plot(frame["age"], frame["survival"], marker=marker, **kwargs)

    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(f"Cohort age ({curve.period})")
    ax.set_ylabel("Share still alive")
    ax.set_title("Model-based cohort survival")
    return ax


def _cohort_value_label(metric: str, relative: bool) -> str:
    if metric == "revenue":
        return "Revenue retention" if relative else "Revenue"
    return "Retention rate" if relative else "Active customers"
