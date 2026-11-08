#!/usr/bin/env python
"""Figure 1: one knob, one figure.

Turns a planted link from zero upwards and reads it back under two reference
tables. The numbers printed below the table are the ones quoted in the paper.
Imports `minimal.py` from the same directory, so no path outside the package is
used.

    python one_knob.py            # prints the table, writes fig_one_knob.png/.pdf
    python one_knob.py --outdir . # write the figure somewhere else

Needs matplotlib on top of numpy. CPU only, a few seconds.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import minimal as M  # noqa: E402

LAMS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
W = 60
SPREAD = 1.5

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 10.5,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
})


def collect():
    rows = []
    for li, lam in enumerate(LAMS):
        pair_true, pair_guess, should = [], [], []
        for w in range(W):
            rng = M.world_rng(700 + li, w)
            B, P, mask = M.make_world(rng, SPREAD, plant=lam, pair=(0, 1))
            Pg = M.fit_rasch(B)
            pair_true.append(float((M.agree(B) - M.expected(P))[0, 1]))
            pair_guess.append(float((M.agree(B) - M.expected(Pg))[0, 1]))
            if lam > 0:
                e = P[0] * P[1] + (1 - P[0]) * (1 - P[1])
                should.append(float((mask * (1 - e)).mean()))
            else:
                should.append(0.0)
        rows.append(dict(
            lam=lam,
            pt=np.mean(pair_true), pts=np.std(pair_true),
            pg=np.mean(pair_guess), pgs=np.std(pair_guess),
            should=np.mean(should),
        ))
    return rows


def crossing(rows):
    """Interpolate where the re-guessed curve crosses zero (in lambda)."""
    x = np.array([r["lam"] for r in rows])
    y = np.array([r["pg"] for r in rows])
    for i in range(len(x) - 1):
        if y[i] <= 0.0 <= y[i + 1]:
            t = -y[i] / (y[i + 1] - y[i])
            return float(x[i] + t * (x[i + 1] - x[i]))
    return float("nan")


def report(rows):
    print("world: %d models x %d items, %d repeats, spread %.1f, seed %d"
          % (M.M, M.N, W, SPREAD, M.SEED))
    print("%6s | %10s %10s | %8s" % ("link", "true", "re-guessed", "should"))
    for r in rows:
        print("%6.2f | %+10.4f %+10.4f | %+8.4f"
              % (r["lam"], r["pt"], r["pg"], r["should"]))
    r0, r3 = rows[0], next(r for r in rows if abs(r["lam"] - 0.30) < 1e-9)
    print("\n  no link, true table     : %+.4f" % r0["pt"])
    print("  no link, re-guessed     : %+.4f" % r0["pg"])
    print("  30%% link, true table   : %+.4f / %+.4f = %.0f%% read back"
          % (r3["pt"], r3["should"], 100 * r3["pt"] / r3["should"]))
    print("  30%% link, re-guessed   : %+.4f / %+.4f = %.0f%% read back"
          % (r3["pg"], r3["should"], 100 * r3["pg"] / r3["should"]))
    print("  re-guessed crosses zero at lambda = %.3f" % crossing(rows))


def draw(rows, stem="fig_one_knob", outdir=HERE):
    x = [r["lam"] for r in rows]
    yt = np.array([r["pt"] for r in rows])
    yg = np.array([r["pg"] for r in rows])
    st = np.array([r["pts"] for r in rows])
    sg = np.array([r["pgs"] for r in rows])
    fig, ax = plt.subplots(1, 1, figsize=(7.4, 4.2))

    ax.axhline(0.0, color="#999999", lw=1.0, ls=(0, (4, 4)), zorder=1)
    ax.fill_between(x, yt - st, yt + st, color="#2c5f8a", alpha=0.13,
                    linewidth=0, zorder=2)
    ax.fill_between(x, yg - sg, yg + sg, color="#c0562a", alpha=0.13,
                    linewidth=0, zorder=2)
    ax.plot(x, yt, color="#2c5f8a", lw=2.1, marker="o", ms=4.2, zorder=4,
            label="true difficulty table")
    ax.plot(x, yg, color="#c0562a", lw=2.1, marker="o", ms=4.2, zorder=4,
            label="difficulty re-guessed")
    ax.plot(x, [r["should"] for r in rows], color="#777777", lw=1.5,
            ls=(0, (2, 3)), zorder=3, label="should read back")

    ax.annotate("no link, yet reading is negative",
                xy=(0, rows[0]["pg"]), xytext=(0.13, rows[0]["pg"] - 0.022),
                fontsize=9, color="#c0562a",
                arrowprops=dict(arrowstyle="->", color="#c0562a", lw=1.0))
    ax.annotate("a link exists, yet the reading sits near zero",
                xy=(0.20, rows[4]["pg"]), xytext=(0.235, -0.020),
                fontsize=9, color="#444444",
                arrowprops=dict(arrowstyle="->", color="#888888", lw=1.0))
    ax.annotate("true reads all, re-guessed reads a third",
                xy=(0.30, rows[5]["pt"]), xytext=(0.235, 0.152),
                fontsize=9, color="#444444",
                arrowprops=dict(arrowstyle="->", color="#888888", lw=1.0))

    ax.set_xlim(-0.02, 0.52)
    ax.set_xlabel("planted link strength (fraction of items copied)",
                  fontsize=9.5, labelpad=8)
    ax.set_ylabel("reading left after subtraction", fontsize=9.5, labelpad=8)
    ax.tick_params(length=4, width=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=9,
              labelspacing=0.34, borderpad=0, borderaxespad=0.2)

    fig.suptitle("One knob, one figure: turn the link from zero upwards",
                 fontsize=12.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in (".png", ".pdf"):
        p = Path(outdir) / (stem + ext)
        fig.savefig(p, dpi=180, facecolor="white")
        print("wrote", p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="fig_one_knob")
    ap.add_argument("--outdir", default=str(HERE))
    a = ap.parse_args()
    res = collect()
    report(res)
    draw(res, stem=a.stem, outdir=a.outdir)
