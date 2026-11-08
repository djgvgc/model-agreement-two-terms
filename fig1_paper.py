#!/usr/bin/env python
"""Figure 1 of the paper.

Shaded band = +/- 1 s.d. of the reported reading across the 60 worlds at
each difficulty spread. The numbers come from minimal.py (numpy only);
this drawing script additionally needs matplotlib.

Run from this directory:

    python fig1_paper.py        # writes fig1_paper.png and .pdf beside this file
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import minimal as M

OUT = HERE / "fig1_paper.png"
rows = np.asarray(M.run_scan(), dtype=float)
x = rows[:, 0]
rep = rows[:, 1]          # reported reading: strength subtracted only
ari = rows[:, 2]          # arithmetic from scores + difficulty alone
both = rows[:, 3]         # after subtracting both
sd = rows[:, 5]           # s.d. of `rep` over the 60 worlds
gap = rep - ari

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "text.color": "#222222",
    "pdf.fonttype": 42,
})

fig = plt.figure(figsize=(8.0, 3.24))
gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.32,
                      left=0.14, right=0.95, top=0.94, bottom=0.22)

# ---------------------------------------------------------------- left
ax = fig.add_subplot(gs[0, 0])
ax.fill_between(x, rep - sd, rep + sd, color="#b9b9b9", alpha=0.55,
                linewidth=0, zorder=1)
ax.plot(x, rep, color="#7d7d7d", lw=1.9, zorder=3)
ax.plot(x, ari, color="#24476b", lw=2.1, ls=(0, (5, 3)), zorder=4)
ax.plot(x, both, color="#2e7d5b", lw=2.0, marker="o", ms=4.8, zorder=5)

handles = [
    Patch(facecolor="#b9b9b9", alpha=0.55, label="reported"),
    Line2D([0], [0], color="#24476b", lw=2.1, ls=(0, (5, 3)),
           label="from scores and difficulty alone"),
    Line2D([0], [0], color="#2e7d5b", lw=2.0, marker="o", ms=4.8,
           label="what is left after subtracting both"),
]
ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=10.5,
          handlelength=2.1, labelspacing=0.34, borderpad=0, borderaxespad=0.2)

ax.text(1.72, 0.176, "+.168", fontsize=12, color="#333333")
ax.text(0.02, -0.0105, "no two models share anything in these worlds",
        fontsize=10, color="#6b6b6b")

ax.set_xlim(-0.05, 2.05)
ax.set_ylim(-0.015, 0.19)
ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
ax.set_xticklabels(["0.0", "0.5", "1.0", "1.5", "2.0"])
ax.set_yticks([0.0, 0.05, 0.10, 0.15])
ax.set_yticklabels(["0", "+.05", "+.10", "+.15"])
ax.set_xlabel("item difficulty spread", labelpad=10)
ax.set_ylabel("agreement beyond\nwhat scores predict", labelpad=8, fontsize=9.5, linespacing=1.25)
ax.tick_params(axis="both", length=4, width=0.9, direction="out")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ---------------------------------------------------------------- right
ax2 = fig.add_subplot(gs[0, 1])
ax2.axhline(0.0, color="#8c8c8c", lw=1.1, ls=(0, (4, 4)), zorder=1)
ax2.plot(x, gap, color="#d17128", lw=2.1, marker="o", ms=5.2, zorder=3)
ax2.text(0.06, -0.00215, "never more\nthan .0008 apart",
         fontsize=10.5, color="#6b6b6b", linespacing=1.4)

ax2.set_xlim(-0.05, 2.05)
ax2.set_ylim(-0.0026, 0.0026)
ax2.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
ax2.set_xticklabels(["0.0", "0.5", "1.0", "1.5", "2.0"])
ax2.set_yticks([-0.002, 0.0, 0.002])
ax2.set_yticklabels(["\u2212.002", "0", "+.002"])
ax2.set_xlabel("item difficulty spread", labelpad=10)
ax2.set_ylabel("gap between\nthe two curves", labelpad=8, linespacing=1.4)
ax2.tick_params(axis="both", length=4, width=0.9, direction="out")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

fig.savefig(OUT, dpi=180, facecolor="white")
fig.savefig(OUT.with_suffix(".pdf"), facecolor="white")
print("wrote", OUT.name, "and", OUT.with_suffix(".pdf").name)
print("s.d. of reported reading across 60 worlds:", ", ".join("%.4f" % v for v in sd))
