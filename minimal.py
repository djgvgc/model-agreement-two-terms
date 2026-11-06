#!/usr/bin/env python
"""A small world whose right answer is known in advance.

    python minimal.py            # artificial worlds, no data needed
    python minimal.py --real     # also check the public right/wrong table

Two figures, one action: before reporting how alike two models are, first work
out how much item difficulty alone would have produced, and only explain the rest.

Only numpy is needed. No training, no download, a few seconds on a laptop.
The --real path additionally needs ../check/answers.npz and ../check/check.py.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEED = 20261105
M, N = 8, 240
W = 60
A_TRUE = np.linspace(-1.4, 1.4, M)
IU = np.triu_indices(M, 1)
SPREADS = [0.0, 0.5, 1.0, 1.5, 2.0]


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_world(rng, spread, plant=0.0, pair=(0, 1)):
    """Answer table with NO relation between models except strength + difficulty."""
    d = rng.normal(0.0, spread, N)
    P = sig(A_TRUE[:, None] - d[None, :])
    B = (rng.random((M, N)) < P).astype(float)
    mask = None
    if plant > 0:
        mask = rng.random(N) < plant
        B[pair[1], mask] = B[pair[0], mask]
    return B, P, mask


def agree(B):
    n = B.shape[1]
    return (B @ B.T + (1 - B) @ (1 - B).T) / n


def expected(P):
    n = P.shape[1]
    return (P @ P.T + (1 - P) @ (1 - P).T) / n


def only_strength(B):
    p = B.mean(1)
    E = p[:, None] * p[None, :] + (1 - p[:, None]) * (1 - p[None, :])
    return float((agree(B) - E)[IU].mean())


def arithmetic(P):
    """What item difficulty alone would produce, without looking at any pair."""
    pm = P.mean(1)
    cov = ((P[IU[0]] - pm[IU[0], None]) * (P[IU[1]] - pm[IU[1], None])).mean()
    return float(2 * cov)


def both(B, P):
    return float((agree(B) - expected(P))[IU].mean())


def fit_rasch(X, iters=8, steps=15):
    """Guess strength and difficulty back out of the answers."""
    mm, nn = X.shape
    acc, pas = X.mean(1), X.mean(0)
    a, d = np.zeros(mm), np.zeros(nn)
    for _ in range(iters):
        lo, hi = np.full(mm, -8.0), np.full(mm, 8.0)
        for _ in range(steps):
            mid = (lo + hi) / 2
            go = sig(mid[:, None] - d[None, :]).mean(1) < acc
            lo, hi = np.where(go, mid, lo), np.where(go, hi, mid)
        a = (lo + hi) / 2
        lo, hi = np.full(nn, -8.0), np.full(nn, 8.0)
        for _ in range(steps):
            mid = (lo + hi) / 2
            go = sig(a[:, None] - mid[None, :]).mean(0) > pas
            lo, hi = np.where(go, mid, lo), np.where(go, hi, mid)
        d = (lo + hi) / 2
        sh = a.mean()
        a -= sh
        d -= sh
    return sig(a[:, None] - d[None, :])


def world_rng(tag, i):
    return np.random.default_rng(SEED + 100000 * tag + i)


# ------------------------------------------------------------------ P1

def run_scan():
    rows = []
    for si, spread in enumerate(SPREADS):
        a, b, c, d = [], [], [], []
        for w in range(W):
            rng = world_rng(10 + si, w)
            B, P, _ = make_world(rng, spread)
            a.append(only_strength(B))
            b.append(arithmetic(P))
            c.append(both(B, P))
            d.append(both(B, fit_rasch(B)))
        rows.append((spread, float(np.mean(a)), float(np.mean(b)),
                     float(np.mean(c)), float(np.mean(d)),
                     float(np.std(a)), float(np.std(c))))
    return rows


# ------------------------------------------------------------------ P2

def run_plant():
    out = []
    for si, spread in enumerate([0.0, 1.5]):
        should, read, read_fit = [], [], []
        for w in range(W):
            rng = world_rng(50 + si, w)
            B, P, mask = make_world(rng, spread, plant=0.30, pair=(0, 1))
            e_indep = P[0] * P[1] + (1 - P[0]) * (1 - P[1])
            should.append(float((mask * (1 - e_indep)).mean()))
            read.append(float((agree(B) - expected(P))[0, 1]))
            read_fit.append(float((agree(B) - expected(fit_rasch(B)))[0, 1]))
        out.append((spread, float(np.mean(should)), float(np.mean(read)),
                    float(np.mean(read_fit)), float(np.std(read))))
    return out


# ------------------------------------------------------- figures (plain SVG)

def fig_scan(rows, path):
    Wd, Ht = 720, 400
    x0, x1, y0, y1 = 70, 470, 55, 285
    ymin, ymax = -0.07, 0.22

    def xy(s, v):
        return (x0 + (s / 2.0) * (x1 - x0),
                y1 - (v - ymin) / (ymax - ymin) * (y1 - y0))

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {Wd} {Ht}" '
         f'width="{Wd}" height="{Ht}">',
         '<rect width="100%" height="100%" fill="white"/>',
         '<text x="20" y="24" font-family="sans-serif" font-size="15" fill="#222">'
         'Where the excess agreement comes from</text>',
         '<text x="20" y="42" font-family="sans-serif" font-size="11" fill="#777">'
         '8 models x 240 items, 60 worlds per setting; no link between models is planted</text>']
    for v in [-0.05, 0.0, 0.05, 0.10, 0.15, 0.20]:
        _, y = xy(0, v)
        col = "#bbb" if v == 0 else "#eee"
        s.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" '
                 f'stroke="{col}" stroke-width="1"/>')
        s.append(f'<text x="{x0-38}" y="{y+4:.1f}" font-family="sans-serif" '
                 f'font-size="10" fill="#666">{v:+.2f}</text>')
    for sp in SPREADS:
        x, _ = xy(sp, 0)
        s.append(f'<text x="{x:.1f}" y="{y1+18}" text-anchor="middle" '
                 f'font-family="sans-serif" font-size="10" fill="#555">{sp:.1f}</text>')
    s.append(f'<text x="{(x0+x1)/2:.1f}" y="{y1+38}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="11" fill="#444">'
             f'how much items differ in difficulty</text>')
    series = [
        (1, "#2c5f8a", "4 0", "subtract strength only"),
        (2, "#d17128", "6 4", "arithmetic from difficulty alone"),
        (3, "#2e7d5b", "4 0", "subtract both (known table)"),
        (4, "#9aa0a6", "2 4", "subtract both (guessed table)"),
    ]
    for idx, col, dash, label in series:
        pts = " ".join(f"{xy(r[0], r[idx])[0]:.1f},{xy(r[0], r[idx])[1]:.1f}" for r in rows)
        s.append(f'<polyline points="{pts}" fill="none" stroke="{col}" '
                 f'stroke-width="2.2" stroke-dasharray="{dash}"/>')
        lx, ly = xy(rows[-1][0], rows[-1][idx])
        s.append(f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="3.2" fill="{col}"/>')
        yy = y0 + 8 + 24 * (idx - 1)
        s.append(f'<line x1="{x1+22}" y1="{yy}" x2="{x1+46}" y2="{yy}" '
                 f'stroke="{col}" stroke-width="2.2" stroke-dasharray="{dash}"/>')
        s.append(f'<text x="{x1+52}" y="{yy+4}" font-family="sans-serif" '
                 f'font-size="10.5" fill="#333">{label}</text>')
    s.append(f'<text x="{x0}" y="{Ht-18}" font-family="sans-serif" font-size="11" '
             f'fill="#444">The blue and orange lines coincide: nothing about model '
             f'pairing is needed to produce that number.</text>')
    s.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="#444" stroke-width="1"/>')
    s.append('</svg>')
    path.write_text("\n".join(s), encoding="utf-8")


def fig_plant(res, path):
    Wd, Ht = 720, 380
    x0, x1, y0, y1 = 80, 470, 60, 270
    ymax = max(max(r[1], r[2], r[3]) for r in res) * 1.25

    def yOf(v):
        return y1 - v / ymax * (y1 - y0)

    labels = ["all items equally hard", "items differ in difficulty"]
    cols = ["#c9d6e2", "#2c5f8a", "#9aa0a6"]
    names = ["should read", "read back (known table)", "read back (guessed table)"]
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {Wd} {Ht}" '
         f'width="{Wd}" height="{Ht}">',
         '<rect width="100%" height="100%" fill="white"/>',
         '<text x="20" y="24" font-family="sans-serif" font-size="15" fill="#222">'
         'The check does not swallow a real link</text>',
         '<text x="20" y="42" font-family="sans-serif" font-size="11" fill="#777">'
         'one model copies the other on 30% of items; 60 worlds per setting</text>']
    for v in np.linspace(0, ymax, 5):
        y = yOf(v)
        s.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="#eee"/>')
        s.append(f'<text x="{x0-38}" y="{y+4:.1f}" font-family="sans-serif" '
                 f'font-size="10" fill="#666">{v:.2f}</text>')
    for gi, r in enumerate(res):
        cx = x0 + (gi + 0.5) * (x1 - x0) / 2
        for bi, key in enumerate([1, 2, 3]):
            v = r[key]
            bw, x = 44, cx - 78 + bi * 52
            y = yOf(v)
            s.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw}" height="{y1-y:.1f}" '
                     f'fill="{cols[bi]}"/>')
            s.append(f'<text x="{x+bw/2:.1f}" y="{y-6:.1f}" text-anchor="middle" '
                     f'font-family="sans-serif" font-size="11" fill="#222">{v:.4f}</text>')
        s.append(f'<text x="{cx:.1f}" y="{y1+22}" text-anchor="middle" '
                 f'font-family="sans-serif" font-size="11" fill="#444">{labels[gi]}</text>')
    for bi, name in enumerate(names):
        xx, yy = x1 + 22, y0 + 10 + 22 * bi
        s.append(f'<rect x="{xx}" y="{yy-9}" width="16" height="12" fill="{cols[bi]}"/>')
        s.append(f'<text x="{xx+22}" y="{yy+1}" font-family="sans-serif" '
                 f'font-size="10.5" fill="#333">{name}</text>')
    s.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="#444"/>')
    s.append('</svg>')
    path.write_text("\n".join(s), encoding="utf-8")


# ------------------------------------------------------------- P3 / P4

def run_real():
    chk = HERE.parent / "check"
    if not (chk / "answers.npz").exists() or not (chk / "check.py").exists():
        print("--real needs ../check/answers.npz and ../check/check.py")
        return None
    sys.path.insert(0, str(chk))
    import check as C

    B, names = C.from_npz(chk / "answers.npz")
    m, n = B.shape
    iu = np.triu_indices(m, 1)
    P = C.guess_difficulty(B)
    p = B.mean(1)

    only = float((C.alike(B) - (p[:, None] * p[None, :]
                  + (1 - p[:, None]) * (1 - p[None, :])))[iu].mean())
    arith = float(2 * ((P[iu[0]] - P.mean(1)[iu[0], None])
                       * (P[iu[1]] - P.mean(1)[iu[1], None])).mean())
    both_ = float((C.alike(B) - C.expected_alike(P))[iu].mean())

    # plant a link of known size on the same pair the existing check uses
    rel = np.array([C.relation(C.short(names[j]), C.short(names[k]))
                    for j, k in zip(*iu)])
    cand = [(abs(p[j] - p[k]), j, k) for (j, k), L in zip(zip(*iu), rel) if L == 0]
    _, j, k = min(cand)
    rng = np.random.default_rng(SEED)
    mask = rng.random(n) < 0.12
    B2 = B.copy()
    B2[k, mask] = B2[j, mask]
    should = float((mask & (B[j] != B[k])).mean())
    D0 = C.alike(B) - C.expected_alike(P)
    D2 = C.alike(B2) - C.expected_alike(P)
    read_back = float(D2[j, k] - D0[j, k])

    # P3: is the changing reading caused by re-fitting, or by averaging a
    # different set of pairs? Keep ONE difficulty table and ONE answer table.
    D = D0
    base = float(D[iu].mean())
    perm = np.random.default_rng(SEED).permutation(m)
    ladder = []
    for kk in [8, 16, 32, m]:
        sel = np.sort(perm[:kk])
        ii = np.triu_indices(kk, 1)
        fixed = float(D[np.ix_(sel, sel)][ii].mean())
        refit = float((C.alike(B[sel])
                       - C.expected_alike(C.guess_difficulty(B[sel])))[ii].mean()) if kk < m else both_
        ladder.append((kk, fixed, refit))
    rng2 = np.random.default_rng(SEED + 1)
    sub = []
    for _ in range(300):
        sel = np.sort(rng2.choice(m, 8, replace=False))
        ii = np.triu_indices(8, 1)
        sub.append(float(D[np.ix_(sel, sel)][ii].mean()))
    return dict(shape=(m, n), only=only, arith=arith, both=both_,
                should=should, read_back=read_back, base=base,
                ladder=ladder, sub=sub)


# ---------------------------------------------------------------- ledger

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="also check the public table")
    a = ap.parse_args()

    print("world: %d models x %d items, %d repeats per setting, seed %d" % (M, N, W, SEED))
    rows = run_scan()
    plant = run_plant()

    print("\nP1  item difficulty OFF -> ON   (nothing planted)")
    print("  %-8s %12s %12s %12s %12s" % ("spread", "strength", "arithmetic", "both(true)", "both(guess)"))
    for r in rows:
        print("  %-8.1f %+12.4f %+12.4f %+12.4f %+12.4f" % (r[0], r[1], r[2], r[3], r[4]))
    gap = max(abs(r[1] - r[2]) for r in rows)
    flat = max(abs(r[3]) for r in rows)
    rise = rows[-1][1] - rows[0][1]
    p1 = gap < 0.01 and flat < 0.01 and rise > 0.08
    print("  max |strength - arithmetic| = %.4f | max |both(true)| = %.4f | rise = %+.4f"
          % (gap, flat, rise))
    print("  P1: %s" % ("holds" if p1 else "BROKEN"))

    print("\nP2  plant a 30% link, then read it back")
    print("  %-8s %12s %12s %12s" % ("spread", "should", "read(true)", "read(guess)"))
    for r in plant:
        print("  %-8.1f %+12.4f %+12.4f %+12.4f" % (r[0], r[1], r[2], r[3]))
    errs = [abs(r[2] - r[1]) for r in plant]
    p2 = max(errs) < 0.01
    print("  |read - should| = %.4f / %.4f" % tuple(errs))
    print("  P2: %s" % ("holds" if p2 else "BROKEN"))

    real = run_real() if a.real else None
    if real:
        print("\nP4  public table %d x %d" % real["shape"])
        print("  strength %+.4f | arithmetic %+.4f | both %+.4f"
              % (real["only"], real["arith"], real["both"]))
        print("  planted %+.4f -> read back %+.4f"
              % (real["should"], real["read_back"]))
        p4 = (abs(real["only"] - 0.1350) < 0.001
              and abs(real["arith"] - 0.1395) < 0.001
              and abs(real["both"] + 0.0044) < 0.001
              and abs(real["read_back"] - 0.0307) < 0.001
              and abs(real["should"] - 0.0307) < 0.001)
        print("  P4: %s" % ("holds" if p4 else "BROKEN / check before going on"))

        print("\nP3  one fixed difficulty table, one fixed answer table")
        print("  all %d pairs average = %+.4f" % (real["shape"][0], real["base"]))
        print("  %-8s %14s %14s" % ("models", "fixed table", "re-guessed"))
        for kk, fixed, refit in real["ladder"]:
            print("  %-8d %+14.4f %+14.4f" % (kk, fixed, refit))
        sub = np.array(real["sub"])
        print("  random 8-model subsets (n=300): mean %+.4f sd %.4f  [%.4f, %.4f]"
              % (sub.mean(), sub.std(), np.quantile(sub, .05), np.quantile(sub, .95)))
        spread_fixed = abs(real["ladder"][0][1] - real["ladder"][-1][1])
        p3 = spread_fixed < 0.005
        print("  fixed-table spread 8 -> %d = %.4f" % (real["shape"][0], spread_fixed))
        print("  P3: %s" % ("holds" if p3 else "BROKEN"))

    fig_scan(rows, HERE / "fig_arithmetic.svg")
    fig_plant(plant, HERE / "fig_injection.svg")
    print("\nfigures: fig_arithmetic.svg, fig_injection.svg")


if __name__ == "__main__":
    main()
