#!/usr/bin/env python3
"""Plot solution quality vs. cluster size N from sweep/results_baseline.txt.

Produces a two-panel, paper-grade figure:
  top:    hit/miss per N (status color) + rolling hit-rate line
  bottom: percentage gap to the known global minimum

Usage: python3 sweep/plot_baseline.py [results_csv] [output_prefix]
"""
import sys

import matplotlib.pyplot as plt
import pandas as pd

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "sweep/results_baseline.txt"
OUT_PREFIX = sys.argv[2] if len(sys.argv) > 2 else "sweep/baseline_quality"

ROLLING_WINDOW = 9

# -- palette (status colors + chart chrome; see dataviz skill reference) -----
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"
BLUE = "#2a78d6"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "text.color": INK_PRIMARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def main():
    df = pd.read_csv(RESULTS, comment="#", na_values="NA",
                      names=["N", "BestEnergy", "KnownMin", "Gap", "GapPercent",
                             "Hit", "Time_s"])
    df = df.dropna(subset=["GapPercent"]).sort_values("N")

    df["RollingHitRate"] = (
        df["Hit"].rolling(ROLLING_WINDOW, center=True, min_periods=1).mean() * 100
    )

    fig, (ax_hit, ax_gap) = plt.subplots(
        2, 1, figsize=(7.5, 4.5), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.4]},
    )

    # -- top panel: hit/miss + rolling hit rate --------------------------------
    hits = df[df["Hit"] == 1]
    misses = df[df["Hit"] == 0]
    ax_hit.scatter(hits["N"], [1] * len(hits), s=10, color=GOOD, zorder=3, label="Hit")
    ax_hit.scatter(misses["N"], [0] * len(misses), s=10, color=CRITICAL, zorder=3, label="Miss")
    ax_hit.plot(df["N"], df["RollingHitRate"] / 100, color=INK_SECONDARY,
                linewidth=1.5, zorder=2, label=f"Rolling hit rate (window={ROLLING_WINDOW})")

    ax_hit.set_ylim(-0.15, 1.15)
    ax_hit.set_yticks([0, 1])
    ax_hit.set_yticklabels(["Miss", "Hit"])
    ax_hit.set_ylabel("Outcome")
    ax_hit.legend(loc="lower left", frameon=False, fontsize=8, ncol=3,
                  bbox_to_anchor=(0, 1.02), borderaxespad=0)

    # -- bottom panel: gap percent ----------------------------------------------
    ax_gap.plot(df["N"], df["GapPercent"].clip(lower=0), color=BLUE, linewidth=1.5, zorder=3)
    ax_gap.scatter(df["N"], df["GapPercent"].clip(lower=0), s=8, color=BLUE, zorder=3)
    ax_gap.set_ylabel("Gap to known minimum (%)")
    ax_gap.set_xlabel("Cluster size N")
    ax_gap.set_ylim(bottom=0)

    for outlier_n in (98,):
        row = df[df["N"] == outlier_n]
        if not row.empty:
            x, y = row["N"].iloc[0], row["GapPercent"].iloc[0]
            ax_gap.annotate(f"N={outlier_n}", (x, y), textcoords="offset points",
                             xytext=(0, 8), ha="center", fontsize=8, color=INK_SECONDARY)

    for ax in (ax_hit, ax_gap):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
        ax.tick_params(length=0)

    ax_gap.spines["bottom"].set_color(BASELINE)

    fig.tight_layout()
    fig.savefig(f"{OUT_PREFIX}.pdf")
    print(f"Wrote {OUT_PREFIX}.pdf")


if __name__ == "__main__":
    main()
