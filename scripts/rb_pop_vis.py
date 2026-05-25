#!/usr/bin/env python3

import csv
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


path = sys.argv[1] if len(sys.argv) > 1 else "output.data"
generations = {}

with open(path, newline="") as f:
    reader = csv.DictReader(f)
    x_cols = [name for name in reader.fieldnames if name.startswith("x")]

    for row in reader:
        generation = int(row["generation"])
        generations.setdefault(generation, []).append([float(row[c]) for c in x_cols])

frames = [generations[g] for g in sorted(generations)]

if not frames:
    raise SystemExit(f"No data found in {path}")
if len(frames[0][0]) < 2:
    raise SystemExit("Need at least two coordinates to plot")

fig, ax = plt.subplots(figsize=(7, 7))
scatter = ax.scatter([], [], s=30, c="#ffcc33", edgecolors="black", linewidths=0.4, alpha=0.95, zorder=4)

xs = [point[0] for frame in frames for point in frame]
ys = [point[1] for frame in frames for point in frame]

x_min, x_max = -2.0, 2.0
y_min, y_max = -1.0, 3.0

grid_x = np.linspace(x_min, x_max, 400)
grid_y = np.linspace(y_min, y_max, 400)
X, Y = np.meshgrid(grid_x, grid_y)
Z = (1 - X) ** 2 + 100 * (Y - X ** 2) ** 2

low_levels = np.linspace(0.0, 11.0, 8)
high_levels = np.logspace(np.log10(11.0), np.log10(Z.max()), 10)
levels = np.unique(np.concatenate([low_levels, high_levels]))

ax.contourf(X, Y, Z, levels=levels, cmap="viridis", extend="max")
ax.contour(X, Y, Z, levels=levels, linewidths=0.7, colors="white", alpha=0.85)
ax.scatter([1.0], [1.0], marker="*", s=120, c="red", zorder=5)

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel(x_cols[0])
ax.set_ylabel(x_cols[1])
title = ax.set_title("")


def update(frame_index):
    scatter.set_offsets(frames[frame_index])
    title.set_text(f"Generation {frame_index}")
    return scatter, title


anim = FuncAnimation(fig, update, frames=len(frames), interval=200, blit=True, repeat=True)
plt.tight_layout()
output = str(Path(path).with_suffix(".gif"))
anim.save(output, writer="pillow", fps=5)
print(output)