#!/usr/bin/env python
"""One file, one command.

Input : a table of right/wrong answers, rows = models, columns = items (1 = right).
Output: five numbers and one figure.

    python check.py                          # uses answers.npz next to this file
    python check.py --npz OTHER.npz          # any other right/wrong table
    python check.py --samples RELEASE_DIR    # rebuild from the public release

Only numpy is needed. No training. A few seconds on a laptop.
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RNG = np.random.default_rng(20261105)


# ---------------------------------------------------------------- input

def _number(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def from_release(d):
    """Rebuild a right/wrong table from the public release, no cached files."""
    subs = sorted(p for p in Path(d).iterdir() if p.is_dir())
    names, per = [], []
    for p in subs:
        f = p / "samples.json"
        if not f.exists():
            continue
        rows = json.loads(f.read_text(encoding="utf-8"))
        rec = {}
        for r in rows:
            vals = [_number(v) for k, v in r.items() if k.startswith("score_")]
            vals = [v for v in vals if v is not None]
            rec[r["question_id"]] = float(np.mean(vals)) if vals else np.nan
        names.append(p.name)
        per.append(rec)
    common = set(per[0])
    for r in per[1:]:
        common &= set(r)
    qids = sorted(common)
    X = np.array([[r[q] for q in qids] for r in per], dtype=float)
    return (np.nan_to_num(X, nan=0.0) >= 0.5).astype(float), names


def from_npz(p):
    z = np.load(p, allow_pickle=True)
    B = z["B"].astype(float)
    names = [str(s) for s in z["models"]] if "models" in z else [str(i) for i in range(len(B))]
    return B, names


# ------------------------------------------------------- the one model fit

def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def guess_difficulty(B, iters=8, steps=15):
    """Guess how strong each model is and how hard each item is, from the answers."""
    m, n = B.shape
    acc, pas = B.mean(1), B.mean(0)
    a, d = np.zeros(m), np.zeros(n)
    for _ in range(iters):
        lo, hi = np.full(m, -8.0), np.full(m, 8.0)
        for _ in range(steps):
            mid = (lo + hi) / 2
            go = sig(mid[:, None] - d[None, :]).mean(1) < acc
            lo, hi = np.where(go, mid, lo), np.where(go, hi, mid)
        a = (lo + hi) / 2
        lo, hi = np.full(n, -8.0), np.full(n, 8.0)
        for _ in range(steps):
            mid = (lo + hi) / 2
            go = sig(a[:, None] - mid[None, :]).mean(0) > pas
            lo, hi = np.where(go, mid, lo), np.where(go, hi, mid)
        d = (lo + hi) / 2
        sh = a.mean()
        a -= sh
        d -= sh
    return sig(a[:, None] - d[None, :])


def alike(B):
    n = B.shape[1]
    return (B @ B.T + (1 - B) @ (1 - B).T) / n


def expected_alike(P):
    n = P.shape[1]
    return (P @ P.T + (1 - P) @ (1 - P).T) / n


# ------------------------------------------------- which models are related

LINEAGE = {}
for s in ["0.5B", "1.5B", "3B", "7B", "14B", "32B", "72B"]:
    LINEAGE["Qwen2.5-" + s] = ("Qwen2.5", "Qwen2.5-" + s, None)
    LINEAGE["Qwen2.5-" + s + "-Instruct"] = ("Qwen2.5", "Qwen2.5-" + s, None)
for s in ["2b", "9b", "27b"]:
    t = "gemma-2-27b" if s in ("2b", "9b") else None
    LINEAGE["gemma-2-" + s] = ("Gemma-2", "gemma-2-" + s, t)
    LINEAGE["gemma-2-" + s + "-it"] = ("Gemma-2", "gemma-2-" + s, t)
LINEAGE.update({
    "Llama-3.1-8B": ("Llama-3", "Llama-3.1-8B", None),
    "Llama-3.1-8B-Instruct": ("Llama-3", "Llama-3.1-8B", None),
    "Llama-3.1-70B": ("Llama-3", "Llama-3.1-70B", None),
    "Llama-3.1-70B-Instruct": ("Llama-3", "Llama-3.1-70B", None),
    "Llama-3.2-1B": ("Llama-3", "Llama-3.2-1B", "Llama-3.1-8B"),
    "Llama-3.2-1B-Instruct": ("Llama-3", "Llama-3.2-1B", "Llama-3.1-8B"),
    "Llama-3.2-3B": ("Llama-3", "Llama-3.2-3B", "Llama-3.1-8B"),
    "Llama-3.2-3B-Instruct": ("Llama-3", "Llama-3.2-3B", "Llama-3.1-8B"),
    "Llama-3.3-70B-Instruct": ("Llama-3", "Llama-3.1-70B", "Llama-3.1-70B"),
    "SmolLM2-135M-Instruct": ("SmolLM2", "SmolLM2-135M-Instruct", None),
    "SmolLM2-360M": ("SmolLM2", "SmolLM2-360M", None),
    "SmolLM2-360M-Instruct": ("SmolLM2", "SmolLM2-360M", None),
    "SmolLM2-1.7B": ("SmolLM2", "SmolLM2-1.7B", None),
    "SmolLM2-1.7B-Instruct": ("SmolLM2", "SmolLM2-1.7B", None),
    "Phi-3.5-MoE-instruct": ("Phi", "Phi-3.5-MoE-instruct", None),
    "Phi-3.5-mini-instruct": ("Phi", "Phi-3.5-mini-instruct", None),
    "phi-4": ("Phi", "phi-4", None),
    "Falcon3-7B-Base": ("Falcon3", "Falcon3-7B-Base", None),
    "Falcon3-7B-Instruct": ("Falcon3", "Falcon3-7B-Base", None),
    "Falcon3-10B-Base": ("Falcon3", "Falcon3-10B-Base", None),
    "Falcon3-10B-Instruct": ("Falcon3", "Falcon3-10B-Base", None),
    "Falcon3-1B-Instruct": ("Falcon3", "Falcon3-1B-Instruct", "Falcon3-7B-Base"),
})


def short(name):
    name = str(name).replace("__", "/")
    for p in ["HuggingFaceTB/", "Qwen/", "google/", "meta-llama/", "microsoft/", "tiiuae/"]:
        name = name.replace(p, "")
    return name


def relation(a, b):
    if a not in LINEAGE or b not in LINEAGE:
        return 0
    fa, ba, ta = LINEAGE[a]
    fb, bb, tb = LINEAGE[b]
    if fa != fb:
        return 0
    if ba == b or bb == a:
        return 3
    if ba == bb:
        return 2
    if ta is not None and (ta == b or ta == bb or tb == a or tb == ba):
        return 2
    if ta is not None and ta == tb:
        return 2
    return 1


# --------------------------------------------------------------- the check

def run(B, names):
    m, n = B.shape
    iu = np.triu_indices(m, 1)
    A = alike(B)
    P = guess_difficulty(B)
    E = expected_alike(P)
    p = B.mean(1)

    # 1. the usual reading: subtract only how strong each model is
    only = (A - (p[:, None] * p[None, :] + (1 - p[:, None]) * (1 - p[None, :])))[iu].mean()

    # 2. how much of that appears from item difficulty alone, without looking at answers
    pm = P.mean(1)
    cov = ((P[iu[0]] - pm[iu[0], None]) * (P[iu[1]] - pm[iu[1], None])).mean()
    arithmetic = 2 * cov

    # 3. subtract both strength and difficulty
    both = (A - E)[iu].mean()

    # 4. plant a link of known size and see whether the ruler reads it back
    rel = np.array([relation(short(names[j]), short(names[k])) for j, k in zip(*iu)])
    acc = p
    cand = [(abs(acc[j] - acc[k]), j, k) for (j, k), L in zip(zip(*iu), rel) if L == 0]
    if not cand:
        cand = [(abs(acc[j] - acc[k]), j, k) for j, k in zip(*iu)]
    _, j, k = min(cand)
    mask = RNG.random(n) < 0.12
    B2 = B.copy()
    B2[k, mask] = B2[j, mask]
    A2 = alike(B2)
    should = (mask & (B[j] != B[k])).mean()
    read_back = (A2 - E)[j, k] - (A - E)[j, k]
    P2 = guess_difficulty(B2)
    E2 = expected_alike(P2)
    read_back_reguess = (A2 - E2)[j, k] - (A - E)[j, k]

    # 5. how much two models differ, compared with two unrelated ones
    d_ind = p[:, None] * (1 - p[None, :]) + p[None, :] * (1 - p[:, None])
    ratio = (1 - A) / d_ind
    r = ratio[iu]
    same = [ratio[a, b] for a, b in zip(*iu)
            if short(names[a]) in LINEAGE and short(names[b]) in LINEAGE
            and (LINEAGE[short(names[a])][1] == short(names[b])
                 or LINEAGE[short(names[b])][1] == short(names[a]))]

    groups = {L: np.sort(r[rel == L]) for L in [3, 2, 1, 0] if (rel == L).any()}

    print("table: %d models x %d items" % (m, n))
    print("1  subtract strength only .................... %+.4f" % only)
    print("2  arithmetic from item difficulty alone ..... %+.4f" % arithmetic)
    print("3  subtract both ............................. %+.4f" % both)
    print("4  planted link: should %+.4f, read back %+.4f (re-guessed %+.4f)"
          % (should, read_back, read_back_reguess))
    if same:
        print("5  same-checkpoint pairs n=%d, median ratio %.3f" % (len(same), np.median(same)))
    for L in [3, 2, 1, 0]:
        if L in groups:
            g = groups[L]
            print("   relation %d: n=%3d  median %.3f" % (L, len(g), np.median(g)))

    if groups:
        draw_figure(groups, HERE / "fig_ratio.svg")
        print("figure: %s" % (HERE / "fig_ratio.svg"))


def draw_figure(groups, path):
    W, H = 680, 340
    x0, x1, y0, y1 = 70, 640, 30, 270
    labels = {3: "same weights", 2: "shared teacher", 1: "same maker", 0: "different makers"}

    def xy(i, v):
        return x0 + (i + 0.5) * (x1 - x0) / 4, y1 - min(v, 1.05) / 1.05 * (y1 - y0)

    s = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d">'
         % (W, H, W, H),
         '<rect width="100%%" height="100%%" fill="white"/>',
         '<text x="20" y="20" font-family="sans-serif" font-size="14" fill="#222">'
         'How much two models differ, relative to two unrelated ones</text>']
    for v in [0, 0.25, 0.5, 0.75, 1.0]:
        _, y = xy(0, v)
        s.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#ddd" stroke-width="1"/>'
                 % (x0, y, x1, y))
        s.append('<text x="%d" y="%.1f" font-family="sans-serif" font-size="11" fill="#666">%.2f</text>'
                 % (x0 - 32, y + 4, v))
    _, yref = xy(0, 1.0)
    s.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#c0392b" stroke-width="1.5" stroke-dasharray="6 4"/>'
             % (x0, yref, x1, yref))
    s.append('<text x="%d" y="%.1f" font-family="sans-serif" font-size="11" fill="#c0392b">unrelated = 1</text>'
             % (x1 - 92, yref - 6))
    for i, L in enumerate([3, 2, 1, 0]):
        if L not in groups:
            continue
        g = groups[L]
        q25, med, q75 = np.quantile(g, [.25, .5, .75])
        x, _ = xy(i, 0)
        _, ya = xy(i, q25)
        _, yb = xy(i, q75)
        _, ym = xy(i, med)
        s.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#2c5f8a" stroke-width="2"/>'
                 % (x, ya, x, yb))
        s.append('<circle cx="%.1f" cy="%.1f" r="6" fill="#2c5f8a"/>' % (x, ym))
        s.append('<text x="%.1f" y="%.1f" text-anchor="middle" font-family="sans-serif" '
                 'font-size="12" fill="#222">%.3f</text>' % (x, ym - 12, med))
        s.append('<text x="%.1f" y="%d" text-anchor="middle" font-family="sans-serif" '
                 'font-size="11" fill="#444">%s</text>' % (x, y1 + 20, labels[L]))
        s.append('<text x="%.1f" y="%d" text-anchor="middle" font-family="sans-serif" '
                 'font-size="10" fill="#888">n=%d</text>' % (x, y1 + 36, len(g)))
    s.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#444" stroke-width="1"/>'
             % (x0, y1, x1, y1))
    s.append('</svg>')
    path.write_text("\n".join(s), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", help="right/wrong table in .npz, key B")
    ap.add_argument("--samples", help="public release directory with per-model samples.json")
    a = ap.parse_args()
    if a.samples:
        B, names = from_release(a.samples)
    else:
        B, names = from_npz(a.npz or HERE / "answers.npz")
    run(B, names)


if __name__ == "__main__":
    main()
