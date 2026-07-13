#!/usr/bin/env python3
"""Small multiples: gap-to-known-minimum across the 4 tested walkers x
iterations combos, one tiny panel per N that failed the baseline sweep.

Each panel is labeled by its N and shows that N's own trajectory from step 1
(2046w/500i) to step 4 (8190w/2000i), sharing a common y-scale so panels are
directly comparable. 24 of 25 panels drop to zero; N=131 (red) is the one
that doesn't.

Usage: python3 sweep/plot_param_sweep.py [results_csv] [output_prefix]
"""
import sys

import matplotlib.pyplot as plt
import pandas as pd

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "sweep/results_param_sweep.txt"
OUT_PREFIX = sys.argv[2] if len(sys.argv) > 2 else "sweep/param_sweep_multiples"

STEPS = [(2046, 500), (8190, 500), (2046, 2000), (8190, 2000)]

NORMAL = "#2a78d6"
ACCENT = "#d03b3b"
INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "text.color": INK_PRIMARY,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

EMPHASIZE_N = 131
NCOLS = 5

# N=131 didn't resolve at the standard grid's largest walkers/iterations
# combo; a dedicated follow-up with even more of both did. Shown as a 5th,
# separated point.
EXTRA_POINT = {"N": 131, "label": "16384 walkers\n4000 iterations", "gap": 0.0}
RESOLVED = "#0ca30c"


def main():
    df = pd.read_csv(RESULTS, comment="#", na_values="NA",
                      names=["N", "Walkers", "NTemps", "NEnsembles", "Iterations",
                             "BestEnergy", "KnownMin", "Gap", "GapPercent", "Hit", "Time_s"])
    df["Walkers"] = df["Walkers"].apply(lambda w: 2046 if w < 5000 else 8190)
    df["GapPercent"] = df["GapPercent"].clip(lower=0)

    per_step = [df[(df["Walkers"] == w) & (df["Iterations"] == it)].set_index("N")["GapPercent"]
                for w, it in STEPS]
    ns = sorted(per_step[0].index)

    nrows = -(-len(ns) // NCOLS)
    ymax = max(s.max() for s in per_step) * 1.08
    x = range(len(STEPS))

    fig, axes = plt.subplots(nrows, NCOLS, figsize=(NCOLS * 1.7, nrows * 1.5), sharey=True)
    axes_flat = axes.flatten()

    for ax, n in zip(axes_flat, ns):
        y = [per_step[i].get(n) for i in range(len(STEPS))]
        is_emphasized = n == EMPHASIZE_N
        color = ACCENT if is_emphasized else NORMAL
        ax.plot(x, y, color=color, linewidth=1.8, marker="o", markersize=2.5, zorder=3)
        ax.set_ylim(-0.02 * ymax, ymax)

        if is_emphasized and n == EXTRA_POINT["N"]:
            extra_x = len(STEPS) + 0.6
            ax.plot([x[-1], extra_x], [y[-1], EXTRA_POINT["gap"]], color=GRIDLINE,
                     linewidth=1, linestyle=":", zorder=2)
            ax.scatter([extra_x], [EXTRA_POINT["gap"]], color=RESOLVED, s=14, zorder=3)
            ax.text(extra_x, ymax * 0.5, EXTRA_POINT["label"], fontsize=6, color=RESOLVED,
                     ha="center", va="center", linespacing=1.4)
            ax.set_xlim(-0.3, extra_x + 0.5)
        else:
            ax.set_xlim(-0.3, len(STEPS) - 0.7)

        ax.set_title(f"N={n}", fontsize=8.5, color=color,
                     fontweight="bold" if is_emphasized else "normal", pad=2)
        ax.set_xticks([])
        ax.tick_params(axis="y", labelsize=6, length=0, colors=INK_MUTED)
        for spine in ("top", "right", "bottom"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color(GRIDLINE)
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.6, zorder=0)

    for ax in axes_flat[len(ns):]:
        ax.axis("off")

    fig.tight_layout(rect=[0, 0.05, 1, 0.93])

    fig.text(0.5, 0.015,
              "Left to right per panel: 2046 walkers, 500 iterations → 8190 walkers, "
              "500 iterations → 2046 walkers, 2000 iterations → 8190 walkers, 2000 iterations",
              ha="center", fontsize=8, color=INK_MUTED)
    fig.text(0.5, 0.97, "Gap to known minimum (%)", ha="center", fontsize=10, color=INK_PRIMARY)
    fig.savefig(f"{OUT_PREFIX}.pdf")
    print(f"Wrote {OUT_PREFIX}.pdf")


if __name__ == "__main__":
    main()
