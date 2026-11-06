#!/usr/bin/env python
"""One panel: the phenomenon others reported, and why it happens.

    python bridge.py

Only numpy. Reuses the same answer-known world and the same functions as
minimal.py; no new data, no new paper, no new experiment design.

Two lines (nothing planted / a link of known size planted) walk through three
readings: subtract strength only -> subtract strength and difficulty with the
KNOWN table -> same but with a table GUESSED back from the answers.
"""
from pathlib import Path

import numpy as np

import minimal as M

HERE = Path(__file__).resolve().parent
W = 60
SPREAD = 1.5
PAIR = (0, 1)
PLANTS = [0.0, 0.30]


def read_pair(B, P_known, P_guessed):
    """Three readings of ONE pair, so the right answer stays in view."""
    A = M.agree(B)
    p = B.mean(1)
    E_strength = p[:, None] * p[None, :] + (1 - p[:, None]) * (1 - p[None, :])
    only = A[PAIR] - E_strength[PAIR]
    both_known = A[PAIR] - M.expected(P_known)[PAIR]
    both_guess = A[PAIR] - M.expected(P_guessed)[PAIR]
    return float(only), float(both_known), float(both_guess)


def arith_pair(P):
    """What item difficulty alone would produce for this one pair."""
    return float(2 * np.mean((P[0] - P[0].mean()) * (P[1] - P[1].mean())))


def run():
    rows = []
    for pi, plant in enumerate(PLANTS):
        vals, shoulds, ariths = [], [], []
        for w in range(W):
            rng = M.world_rng(70 + pi, w)
            B, P, mask = M.make_world(rng, SPREAD, plant=plant, pair=PAIR)
            Pg = M.fit_rasch(B)
            vals.append(read_pair(B, P, Pg))
            ariths.append(arith_pair(P))
            if mask is not None:
                e_indep = P[0] * P[1] + (1 - P[0]) * (1 - P[1])
                shoulds.append(float((mask * (1 - e_indep)).mean()))
        v = np.mean(np.array(vals), axis=0)
        rows.append(dict(plant=plant, only=float(v[0]), known=float(v[1]),
                         guess=float(v[2]), sd=float(np.std(np.array(vals)[:, 1])),
                         should=float(np.mean(shoulds)) if shoulds else 0.0,
                         arith=float(np.mean(ariths))))
    return rows


def draw(rows, path, should):
    Wd, Ht = 720, 400
    x0, x1, y0, y1 = 70, 470, 55, 285
    ymin, ymax = -0.09, 0.25
    xs = [x0 + (i + 0.5) * (x1 - x0) / 3 for i in range(3)]

    def yOf(v):
        return y1 - (v - ymin) / (ymax - ymin) * (y1 - y0)

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {Wd} {Ht}" '
         f'width="{Wd}" height="{Ht}">',
         '<rect width="100%" height="100%" fill="white"/>',
         '<text x="20" y="24" font-family="sans-serif" font-size="15" fill="#222">'
         'Why the excess agreement disappears, and what the negative reading is</text>',
         '<text x="20" y="42" font-family="sans-serif" font-size="11" fill="#777">'
         'one answer-known world, 8 models x 240 items, 60 repeats, watching one pair</text>']
    for v in [-0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25]:
        y = yOf(v)
        col = "#bbb" if v == 0 else "#eee"
        s.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="{col}"/>')
        s.append(f'<text x="{x0-38}" y="{y+4:.1f}" font-family="sans-serif" '
                 f'font-size="10" fill="#666">{v:+.2f}</text>')
    labels = ["strength only", "+ difficulty (known)", "+ difficulty (guessed)"]
    for x, lab in zip(xs, labels):
        s.append(f'<text x="{x:.1f}" y="{y1+20}" text-anchor="middle" '
                 f'font-family="sans-serif" font-size="10.5" fill="#444">{lab}</text>')
    series = [
        (0, "#5b7288", "nothing planted"),
        (1, "#d17128", f"30% link planted (should read {should:+.3f})"),
    ]
    for idx, col, name in series:
        r = rows[idx]
        vals = [r["only"], r["known"], r["guess"]]
        pts = " ".join(f"{x:.1f},{yOf(v):.1f}" for x, v in zip(xs, vals))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.4"/>')
        for x, v in zip(xs, vals):
            s.append(f'<circle cx="{x:.1f}" cy="{yOf(v):.1f}" r="4" fill="{col}"/>')
            s.append(f'<text x="{x:.1f}" y="{yOf(v)-11:.1f}" text-anchor="middle" '
                     f'font-family="sans-serif" font-size="11" fill="{col}">{v:+.4f}</text>')
        yy = y0 + 10 + 26 * idx
        s.append(f'<line x1="{x1+22}" y1="{yy}" x2="{x1+46}" y2="{yy}" '
                 f'stroke="{col}" stroke-width="2.4"/>')
        s.append(f'<text x="{x1+52}" y="{yy+4}" font-family="sans-serif" '
                 f'font-size="10.5" fill="#333">{name}</text>')
    s.append(f'<text x="{x0}" y="{Ht-40}" font-family="sans-serif" font-size="11" '
             f'fill="#444">The drop is arithmetic: with the known table, the planted link '
             f'is still read back in full.</text>')
    s.append(f'<text x="{x0}" y="{Ht-22}" font-family="sans-serif" font-size="11" '
             f'fill="#444">The dip below zero appears only when difficulty is guessed '
             f'from the same answers.</text>')
    s.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="#444"/>')
    s.append('</svg>')
    path.write_text("\n".join(s), encoding="utf-8")


def main():
    rows = run()
    print("world: 8 x 240, spread %.1f, %d repeats, pair %s" % (SPREAD, W, PAIR))
    print("%-22s %12s %12s %12s" % ("", "strength only", "known table", "guessed table"))
    for r in rows:
        tag = "nothing planted" if r["plant"] == 0 else "30% link planted"
        print("%-22s %+12.4f %+12.4f %+12.4f" % (tag, r["only"], r["known"], r["guess"]))
    should = rows[1]["should"]
    arith = rows[0]["arith"]
    print("\nshould read back %+.4f | pair's own arithmetic term %+.4f" % (should, arith))

    a, b = rows[0], rows[1]
    checks = [
        ("B1 known table reads the link back", abs(b["known"] - should) < 0.01 and abs(a["known"]) < 0.01),
        ("B2 flip only under guessed table", a["guess"] < -0.02 and b["guess"] > 0 and abs(a["known"]) < 0.01),
        ("B3 identity holds for one pair", abs(b["only"] - (a["only"] + should)) < 0.02),
        ("B4 first step is the arithmetic term", abs(a["only"] - arith) < 0.01 and abs(a["known"]) < 0.01),
    ]
    print("\nledger")
    for name, ok in checks:
        print("  %-40s %s" % (name, "holds" if ok else "BROKEN"))
    draw(rows, HERE / "fig_bridge.svg", should)
    print("\nfigure: fig_bridge.svg")


if __name__ == "__main__":
    main()
