#!/usr/bin/env python3
"""Two scaling plots: how runtime scales with problem size N, and how it
scales with walkers. Both log-log axes but with plain (non-scientific)
tick labels for direct comparison against config values.

Left:  wall-clock time vs. cluster size N (baseline sweep, 1024w/250i),
       against the theoretical N^2 curve (O(N^2) pairwise energy/force
       cost), full N range.
Right: wall-clock time vs. walkers, every (N, iterations) pair in the
       parameter sweep as a thin line (2046 -> 8190 walkers), against a
       linear reference — shows the sub-linear cost of adding walkers.

Usage: python3 sweep/plot_scaling.py [baseline_csv] [param_sweep_csv] [output_prefix]
"""
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

BASELINE_CSV = sys.argv[1] if len(sys.argv) > 1 else "sweep/results_baseline.txt"
PARAM_CSV = sys.argv[2] if len(sys.argv) > 2 else "sweep/results_param_sweep.txt"
OUT_PREFIX = sys.argv[3] if len(sys.argv) > 3 else "sweep/scaling"

LINE_COLOR = "#2a78d6"
REF_COLOR = "#898781"
CONTEXT = "#c3c2b7"
INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.edgecolor": CONTEXT,
    "text.color": INK_PRIMARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def plain_numbers(ax, axis="both"):
    fmt = mticker.FuncFormatter(lambda v, _: f"{v:g}")
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", color=GRIDLINE, linewidth=0.6, zorder=0)
    ax.tick_params(length=0)


def main():
    baseline = pd.read_csv(BASELINE_CSV, comment="#", na_values="NA",
                            names=["N", "BestEnergy", "KnownMin", "Gap", "GapPercent",
                                   "Hit", "Time_s"])
    param_sweep = pd.read_csv(PARAM_CSV, comment="#", na_values="NA",
                          names=["N", "Walkers", "NTemps", "NEnsembles", "Iterations",
                                 "BestEnergy", "KnownMin", "Gap", "GapPercent", "Hit", "Time_s"])
    param_sweep["Walkers"] = param_sweep["Walkers"].apply(lambda w: 2046 if w < 5000 else 8190)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # -- left: time vs N, measured vs theoretical N^2, full domain -------------
    n_vals = baseline["N"].to_numpy(dtype=float)
    t_vals = baseline["Time_s"].to_numpy(dtype=float)

    ax1.plot(n_vals, t_vals, color=LINE_COLOR, linewidth=1.8, label="Measured", zorder=3)

    n_full = np.array([n_vals.min(), n_vals.max()])
    anchor_n, anchor_t = 150.0, t_vals[-1]
    n2_coeff = anchor_t / anchor_n ** 2
    ax1.plot(n_full, n2_coeff * n_full ** 2, color=REF_COLOR, linewidth=1.5,
             linestyle="--", zorder=2, label="N^2 (theoretical)")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xticks([2, 10, 50, 150])
    ax1.set_yticks([0.1, 1, 10, 100, 300])
    plain_numbers(ax1)
    ax1.set_xlabel("Cluster size N")
    ax1.set_ylabel("Wall-clock time (s)")
    ax1.text(-0.14, 1.08, "(a)", transform=ax1.transAxes, fontsize=11,
              fontweight="bold", color=INK_PRIMARY)
    ax1.legend(loc="upper left", frameon=False, fontsize=8)
    style_axes(ax1)

    # -- right: time vs walkers, every (N, iterations) pair ----------------------
    all_x = []
    for (n, it), grp in param_sweep.groupby(["N", "Iterations"]):
        grp = grp.sort_values("Walkers")
        if len(grp) < 2:
            continue
        ax2.plot(grp["Walkers"], grp["Time_s"], color=CONTEXT, linewidth=0.8,
                  alpha=0.6, zorder=2)
        all_x.extend(grp["Walkers"].tolist())

    w_ref = np.array([min(all_x), max(all_x)])
    mid_y = param_sweep["Time_s"].median()
    ax2.plot(w_ref, mid_y * (w_ref / w_ref[0]), color=REF_COLOR, linewidth=1.5,
              linestyle="--", zorder=2, label="Linear (ideal)")

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xticks([2000, 4000, 8000])
    plain_numbers(ax2)
    ax2.set_xlabel("Walkers")
    ax2.set_ylabel("Wall-clock time (s)")
    ax2.text(-0.14, 1.08, "(b)", transform=ax2.transAxes, fontsize=11,
              fontweight="bold", color=INK_PRIMARY)
    ax2.legend(loc="upper left", frameon=False, fontsize=8)
    style_axes(ax2)

    fig.tight_layout()
    fig.savefig(f"{OUT_PREFIX}.pdf")
    print(f"Wrote {OUT_PREFIX}.pdf")


if __name__ == "__main__":
    main()
