"""Figures for the manifesto article, fit on the real CDNOW sample.

Run: uv run python docs/articles/figures/make_figures.py

Both figures come from the same published fit the README benchmarks against,
so the article never shows a number the test suite is not already guarding.
Palette is the dataviz reference instance, validated for CVD separation
(blue/red adjacent pair, protan dE 21.6).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from clvkit import BGNBD, CustomerBase

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
CALIBRATION_END = "1997-09-30"

BLUE, RED = "#2a78d6", "#e34948"
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"


def _style(ax):
    """Recessive axes: no box, one soft grid, ink on text only."""
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d8d7d2")
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.grid(axis="y", color="#e8e7e2", linewidth=0.8)
    ax.set_axisbelow(True)


def load():
    raw = pd.read_csv(
        ROOT / "CDNOW_sample.txt",
        sep=r"\s+",
        header=None,
        names=["master_id", "customer_id", "date", "quantity", "amount"],
    )
    raw["date"] = pd.to_datetime(raw["date"], format="%Y%m%d")

    # amount_col=None reproduces the published fit exactly; spend is joined
    # back separately so the eight zero-value customers are not dropped.
    cb = CustomerBase.from_transactions(
        raw, amount_col=None, time_unit="W", collapse="D"
    )
    calibration, _ = cb.split(calibration_period_end=CALIBRATION_END)
    model = BGNBD().fit(calibration)

    summary = calibration.to_pandas().copy()
    summary["p_alive"] = model.probability_alive().to_pandas()["probability_alive"]

    spend = (
        raw[raw["date"] <= CALIBRATION_END]
        .groupby("customer_id")["amount"]
        .sum()
        .rename("spend")
    )
    return summary.join(spend, how="left").fillna({"spend": 0.0}), model


def figure_one(df):
    """The thesis: past spend and being alive are close to independent.

    Repeat buyers only. BG/NBD hands every frequency-zero customer
    P(alive) = 1 by construction, because a customer who never came back has
    produced no gap to infer a dropout from. Leaving those 1,411 in draws a
    dense band across the top of the chart that is an artefact of the model's
    definition rather than anything about CDNOW. Figure two is where they
    belong.
    """
    df = df[df["frequency"] >= 1]
    cut = df["spend"].quantile(0.90)
    doomed = (df["spend"] >= cut) & (df["p_alive"] < 0.30)

    # $1,050 holds every red point but one. The exception spent $6,553, and
    # letting the axis reach it would squash the 99% of the base under $200
    # into a stripe. It is pinned to the right edge with its own marker
    # instead, so the count in the title is the count on the chart.
    xmax = 1050.0
    off = df["spend"] > xmax
    plotted = df.assign(spend=df["spend"].clip(upper=xmax))

    fig, ax = plt.subplots(figsize=(10, 6), dpi=140)
    _style(ax)
    ax.scatter(
        plotted.loc[~doomed & ~off, "spend"],
        plotted.loc[~doomed & ~off, "p_alive"],
        s=14,
        c=BLUE,
        alpha=0.35,
        linewidths=0,
        label="Everyone else",
    )
    ax.scatter(
        plotted.loc[doomed & ~off, "spend"],
        plotted.loc[doomed & ~off, "p_alive"],
        s=26,
        c=RED,
        alpha=0.9,
        linewidths=0,
        label="Top 10% by spend, under 30% likely alive",
    )
    ax.scatter(
        plotted.loc[off, "spend"],
        plotted.loc[off, "p_alive"],
        s=70,
        marker=">",
        c=[RED if d else BLUE for d in doomed[off]],
        linewidths=0,
        clip_on=False,
    )
    # Only the red one gets a label. The four off-scale blue points are
    # ordinary big spenders who are fine, and naming them is noise.
    labelled = off & doomed
    for spend, p in zip(
        df.loc[labelled, "spend"], df.loc[labelled, "p_alive"], strict=True
    ):
        ax.annotate(
            f"${spend:,.0f}, P(active) {p:.0e}",
            xy=(xmax, p),
            xytext=(-12, 18),
            textcoords="offset points",
            ha="right",
            color=MUTED,
            fontsize=10,
        )
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(
        "Total spend in the first 39 weeks (USD), repeat buyers only",
        color=MUTED,
        fontsize=11,
    )
    ax.set_ylabel("P(still active)", color=MUTED, fontsize=11)
    ax.set_title(
        f"{int(doomed.sum())} of CDNOW's biggest spenders were probably already gone",
        color=INK,
        fontsize=15,
        loc="left",
        pad=16,
    )
    # Below the axes: the lower-right corner is where the red points live.
    leg = ax.legend(
        frameon=False,
        fontsize=10,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
    )
    for text in leg.get_texts():
        text.set_color(MUTED)
    fig.tight_layout()
    fig.savefig(OUT / "01-alive-vs-spend.png", facecolor=SURFACE)
    return int(doomed.sum())


def figure_two(df):
    """The 60%: most of the base is one purchase and no evidence."""
    counts = df["frequency"].clip(upper=10).value_counts().sort_index()
    zero = int(counts.loc[0])
    total = len(df)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=140)
    _style(ax)
    colors = [RED if f == 0 else BLUE for f in counts.index]
    ax.bar(counts.index, counts.values, color=colors, width=0.72)
    ax.annotate(
        f"{zero:,} of {total:,} customers\nnever came back ({zero / total:.0%})",
        xy=(0, zero),
        xytext=(1.4, zero * 0.86),
        color=INK,
        fontsize=12,
        arrowprops={"arrowstyle": "-", "color": "#d8d7d2", "linewidth": 1.2},
    )
    ax.set_xticks(range(11))
    ax.set_xticklabels([str(i) for i in range(10)] + ["10+"])
    ax.set_xlabel("Repeat purchases in the first 39 weeks", color=MUTED, fontsize=11)
    ax.set_ylabel("Customers", color=MUTED, fontsize=11)
    ax.set_title(
        "Frequency counts repeats, so most of CDNOW sits at zero",
        color=INK,
        fontsize=15,
        loc="left",
        pad=16,
    )
    fig.tight_layout()
    fig.savefig(OUT / "02-frequency-zero-mass.png", facecolor=SURFACE)
    return zero, total


if __name__ == "__main__":
    df, model = load()
    p = model.params_
    print(f"fit: r={p['r']:.6f} alpha={p['alpha']:.6f} a={p['a']:.6f} b={p['b']:.6f}")
    print(f"fig 1: {figure_one(df)} big spenders under 30% alive")
    zero, total = figure_two(df)
    print(f"fig 2: {zero} of {total} at frequency 0 ({zero / total:.1%})")
