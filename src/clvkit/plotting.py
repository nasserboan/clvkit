"""Every result's ``.plot()`` delegates here.

One flat module, one matplotlib backend: restyling clvkit's charts means
editing this file, not hunting ``plot`` methods across the package.
"""

from typing import TYPE_CHECKING, TypedDict

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from clvkit._result import Prediction
    from clvkit.clv._bootstrap import ParameterUncertainty
    from clvkit.clv.clv import CLVResult
    from clvkit.clv.independence import IndependenceCheck
    from clvkit.cohort.matrix import CohortMatrix
    from clvkit.cohort.survival import SurvivalCurve


class PlotCLVOptions(TypedDict, total=False):
    """Every keyword ``plot_clv`` accepts, as a type.

    Declared once here so ``CLVResult.plot`` can forward ``**kwargs`` and still
    surface each option — with its type — at the call site, instead of hiding
    everything behind an opaque ``**kwargs``. Keep it in lockstep with
    ``plot_clv``'s signature; a test asserts they match.
    """

    ax: "Axes | None"
    ax_kwargs: dict
    scatter_kwargs: dict
    curve_levels: int
    levels: "list[float] | None"
    cmap: str
    colorbar: bool
    vmin: "float | None"
    vmax: "float | None"
    title: "str | None"
    currency: str


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


def plot_parameter_uncertainty(
    result: "ParameterUncertainty",
    *,
    ax: "Axes | None" = None,
    **kwargs,
) -> "Axes":
    """Draw each fitted parameter as a point with its bootstrap interval.

    One row per parameter, the estimate marked and the percentile interval as a
    horizontal bar, so a reader sees at a glance which parameters the data pins
    down and which it barely constrains.
    """
    if ax is None:
        _, ax = plt.subplots()

    frame = result.to_pandas()
    estimate = frame["estimate"].to_numpy(dtype=float)
    positions = np.arange(len(frame))
    # A percentile interval need not straddle the point estimate, so errorbar's
    # (non-negative) half-widths are clamped rather than trusted to be positive.
    lower = np.clip(estimate - frame["ci_low"].to_numpy(dtype=float), 0.0, None)
    upper = np.clip(frame["ci_high"].to_numpy(dtype=float) - estimate, 0.0, None)

    ax.errorbar(estimate, positions, xerr=[lower, upper], fmt="o", capsize=4, **kwargs)
    ax.set_yticks(positions, list(frame.index))
    ax.invert_yaxis()
    ax.set_xlabel("Parameter value")
    ax.set_title(
        f"{result.model_name} parameters: "
        f"{result.confidence:.0%} bootstrap interval "
        f"({result.n_replicates} replicates)"
    )
    return ax


def plot_clv(
    result: "CLVResult",
    *,
    ax: "Axes | None" = None,
    ax_kwargs: "dict | None" = None,
    scatter_kwargs: "dict | None" = None,
    curve_levels: int = 4,
    levels: "list[float] | None" = None,
    cmap: str = "viridis",
    colorbar: bool = True,
    vmin: "float | None" = None,
    vmax: "float | None" = None,
    title: "str | None" = None,
    currency: str = "$",
) -> "Axes":
    """Show how discounted transactions and spend make up each CLV.

    The dashed lines are equal-CLV curves, drawn at round money levels so a
    reader traces "everyone on this line is worth about the same" rather than
    decoding a percentile. ``currency`` is the symbol on the axes and labels;
    it does not convert anything — the numbers are whatever unit ``expected_spend``
    already carries.

    Every knob is optional and falls back to a sensible default when left as
    ``None``:

    - ``levels`` — the CLV values the curves sit at. Default: ``curve_levels``
      round money levels spread across the bulk of the distribution.
    - ``vmin`` / ``vmax`` — the colour scale. Default: ``0`` and the 95th
      percentile of CLV, so a handful of whales don't wash the ramp out.

    Marker styling (size, alpha, edge colour, …) goes through ``scatter_kwargs``
    rather than loose ``**kwargs``, so the signature stays fully typed.
    """
    ax_kwargs = ax_kwargs or {}
    scatter_kwargs = scatter_kwargs or {}
    if ax is None:
        _, ax = plt.subplots(**ax_kwargs)

    frame = result.to_pandas()
    x = frame["discounted_expected_transactions"].to_numpy(dtype=float)
    y = frame["expected_spend"].to_numpy(dtype=float)
    clv = frame["clv"].to_numpy(dtype=float)
    resolved_vmin = 0.0 if vmin is None else vmin
    resolved_vmax = _default_color_ceiling(clv) if vmax is None else vmax
    points = ax.scatter(
        x, y, c=clv, cmap=cmap, vmin=resolved_vmin, vmax=resolved_vmax, **scatter_kwargs
    )

    positive_x = x[x > 0]
    curve_values = (
        levels if levels is not None else _round_clv_levels(clv, curve_levels)
    )
    if curve_values and positive_x.size:
        y_cap = float(np.nanmax(y)) * 1.05
        x_min, x_max = float(positive_x.min()), float(positive_x.max())
        curves = []
        for level in curve_values:
            # Every point on an iso-CLV curve satisfies margin * x * y = level,
            # so the whole curve is fixed by the constant k = level / margin.
            k = level / result.margin
            start = max(x_min, k / y_cap)
            curve_x = np.linspace(start, x_max, 200)
            ax.plot(curve_x, k / curve_x, color="0.5", linestyle="--", linewidth=1)
            annotation = ax.annotate(
                f"{currency}{level:,.0f}",
                xy=(start, k / start),
                xytext=(3, -3),
                textcoords="offset points",
                ha="left",
                va="top",
                fontsize="small",
                color="0.4",
            )
            curves.append((annotation, k))
        _pin_iso_labels_to_view(ax, curves)

    _style_scatter_axes(ax)
    ax.set_xlabel(
        f"Discounted expected transactions "
        f"(next {result.horizon} {_time_unit_word(result.time_unit, plural=True)})"
    )
    ax.set_ylabel(f"Expected spend per transaction ({currency})")
    ax.set_title(
        title if title is not None else "Where CLV comes from: how often × how much"
    )
    if colorbar:
        ax.figure.colorbar(
            points,
            ax=ax,
            label=(
                f"{result.horizon}-{_time_unit_word(result.time_unit)} CLV ({currency})"
            ),
        )
    return ax


def _round_clv_levels(clv: np.ndarray, count: int) -> list[float]:
    """Pick up to ``count`` round money levels for the equal-CLV curves.

    Iso-value curves crowd toward the origin, so evenly spaced levels bunch up;
    a geometric run across the bulk of the CLV distribution spreads the curves
    across the cloud. Each is then snapped to a 1/2/5 "nice" number so the
    labels read ``$50``, ``$100`` rather than ``$47.30``.
    """
    positive = clv[clv > 0]
    if count < 1 or positive.size == 0:
        return []
    lo, hi = np.quantile(positive, [0.4, 0.97])
    if not hi > lo > 0:
        return []
    levels = sorted({_nice_number(v) for v in np.geomspace(lo, hi, count)})
    return [level for level in levels if level > 0]


def _default_color_ceiling(clv: np.ndarray) -> "float | None":
    """A colour ceiling that ignores the long right tail of whales.

    Normalising the ramp to the raw maximum lets one or two extreme customers
    flatten everyone else into the bottom of the palette. Clipping at the 95th
    percentile keeps the gradient meaningful for the mass of the base. Returns
    ``None`` (matplotlib's own autoscaling) when there's nothing to clip.
    """
    finite = clv[np.isfinite(clv)]
    if finite.size == 0:
        return None
    return float(np.quantile(finite, 0.95))


def _nice_number(value: float) -> float:
    """Snap a positive value to the nearest 1/2/5 x 10^n round number."""
    exponent = np.floor(np.log10(value))
    base = 10.0**exponent
    mantissa = value / base
    if mantissa < 1.5:
        nice = 1.0
    elif mantissa < 3.5:
        nice = 2.0
    elif mantissa < 7.5:
        nice = 5.0
    else:
        nice = 10.0
    return nice * base


_TIME_UNIT_WORDS = {"D": "day", "W": "week", "M": "month", "Q": "quarter", "Y": "year"}


def _time_unit_word(time_unit: str, *, plural: bool = False) -> str:
    word = _TIME_UNIT_WORDS.get(time_unit)
    if word is None:
        return time_unit
    return f"{word}s" if plural else word


def _pin_iso_labels_to_view(ax: "Axes", curves: list) -> None:
    """Keep each iso-CLV label at the top of its curve, whatever the limits.

    An iso-CLV curve is ``y = k / x``; its highest *visible* point is the
    left end of the segment the current view actually shows. Placing the label
    in fixed data coordinates breaks the moment the caller sets their own
    ``xlim``/``ylim`` — the text drifts off the frame. So the position is
    recomputed from the live limits, wired to the axis's own change events
    (``xlim_changed``/``ylim_changed``), which also fire on autoscale.
    """

    def reposition(target: "Axes") -> None:
        x0, x1 = sorted(target.get_xlim())
        y0, y1 = sorted(target.get_ylim())
        count = len(curves)
        for rank, (annotation, k) in enumerate(curves):
            if x1 <= 0 or y1 <= 0:
                annotation.set_visible(False)
                continue
            # Every curve alone hugs the top-left corner, so pinning them all
            # to the top edge stacks the labels. Instead give each a distinct
            # height in the upper band and read x back off its own curve — the
            # labels fan out and stay legible even on an outlier-stretched axis.
            fraction = 0.9 if count == 1 else 0.9 - 0.45 * rank / (count - 1)
            y_label = y0 + (y1 - y0) * fraction
            x_label = k / y_label
            if not (x0 <= x_label <= x1):
                # The curve never reaches that height inside the x-window; fall
                # back to the highest point it does show, or hide it entirely.
                left = max(x0, k / y1)
                right = x1 if y0 <= 0 else min(x1, k / y0)
                if left > right:
                    annotation.set_visible(False)
                    continue
                x_label, y_label = left, k / left
            annotation.set_visible(True)
            annotation.xy = (x_label, y_label)

    reposition(ax)
    ax.callbacks.connect("xlim_changed", reposition)
    ax.callbacks.connect("ylim_changed", reposition)


def plot_probability_alive(
    prediction: "Prediction",
    *,
    ax: "Axes | None" = None,
    cmap: str = "plasma",
    colorbar: bool = True,
    **kwargs,
) -> "Axes":
    """Show probability alive against silence and repeat-purchase count."""
    if ax is None:
        _, ax = plt.subplots()

    frame = prediction._plot_data
    if frame is None:
        return plot_prediction(prediction, ax=ax, **kwargs)

    # A faint white rim separates overlapping points in the dense band without
    # drawing attention to itself. Defaults only — a caller can override both.
    kwargs.setdefault("edgecolors", (1.0, 1.0, 1.0, 0.4))
    kwargs.setdefault("linewidths", 0.3)
    points = ax.scatter(
        frame["T"] - frame["recency"],
        prediction.to_pandas()[prediction.name],
        c=frame["frequency"],
        cmap=cmap,
        **kwargs,
    )

    _style_scatter_axes(ax)
    ax.set_xlabel(f"{_time_unit_label(prediction._plot_time_unit)} since last purchase")
    ax.set_ylabel("P(alive) at observation end")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Gone, or just quiet? Same silence reads differently by buying rhythm")
    if colorbar:
        ax.figure.colorbar(points, ax=ax, label="Repeat purchases (frequency)")
    return ax


def _style_scatter_axes(ax: "Axes") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _time_unit_label(time_unit: str) -> str:
    return {
        "D": "Days",
        "W": "Weeks",
        "M": "Months",
        "Q": "Quarters",
        "Y": "Years",
    }.get(time_unit, time_unit)


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
