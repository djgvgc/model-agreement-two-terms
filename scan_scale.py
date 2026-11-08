#!/usr/bin/env python
"""Sweep the read-back scale over the knobs the paper left fixed.

No new claim: we ask how much of a planted link the guessed difficulty table
reads back when the number of items, the shape of the difficulty
distribution, or the way the link is planted changes. Only numpy.

Run from anywhere:

    python scan_scale.py

Writes scan_scale_results.txt and scan_scale_results.json beside this file.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import minimal as MM

W = 60
MODELS = MM.M
A_TRUE = MM.A_TRUE
SPREAD = 1.5
AMPS = [0.05, 0.10, 0.20, 0.30, 0.50]
TARGET = 0.0307
PAIR = (0, 1)


def draw_d(rng, n, shape):
    """All three shapes have the same standard deviation SPREAD."""
    if shape == "normal":
        return rng.normal(0.0, SPREAD, n)
    if shape == "uniform":
        s = SPREAD * np.sqrt(3.0)
        return rng.uniform(-s, s, n)
    if shape == "bimodal":
        return rng.choice([-SPREAD, SPREAD], size=n).astype(float)
    raise ValueError(shape)


def make_world(rng, n, shape, amp=0.0, target=None):
    d = draw_d(rng, n, shape)
    P = 1.0 / (1.0 + np.exp(-(A_TRUE[:, None] - d[None, :])))
    B = (rng.random((MODELS, n)) < P).astype(float)
    j, k = PAIR
    if target is not None:
        # "Target reading" plant: copy on disagreeing items until the pair's
        # excess agreement against the known table rises by `target`.
        diff = np.flatnonzero(B[j] != B[k])
        rng.shuffle(diff)
        need = min(int(round(target * n)), len(diff))
        B[k, diff[:need]] = B[j, diff[:need]]
        return B, P, float(need / n)
    if amp > 0:
        # "Behavioural" plant: one model copies the other on a fraction of items.
        mask = rng.random(n) < amp
        e_indep = P[j] * P[k] + (1 - P[j]) * (1 - P[k])
        B[k, mask] = B[j, mask]
        return B, P, float((mask * (1 - e_indep)).mean())
    return B, P, 0.0


def readings(B, P):
    j, k = PAIR
    known = float((MM.agree(B) - MM.expected(P))[j, k])
    guess = float((MM.agree(B) - MM.expected(MM.fit_rasch(B)))[j, k])
    return known, guess


def cell(n, shape, amp=0.0, target=None, tag=0):
    shoulds, knowns, guesses = [], [], []
    for w in range(W):
        rng = np.random.default_rng(MM.SEED + 100000 * tag + w)
        B, P, should = make_world(rng, n, shape, amp=amp, target=target)
        kn, gu = readings(B, P)
        shoulds.append(should)
        knowns.append(kn)
        guesses.append(gu)
    return (float(np.mean(shoulds)), float(np.mean(knowns)),
            float(np.mean(guesses)))


def floor_from(rows):
    """Planted excess where the guessed-table reading crosses zero."""
    xs = [r[0] for r in rows]
    ys = [r[2] for r in rows]
    for i in range(len(xs) - 1):
        if ys[i] <= 0.0 <= ys[i + 1]:
            t = -ys[i] / (ys[i + 1] - ys[i])
            return xs[i] + t * (xs[i + 1] - xs[i])
    return float("nan")


def main():
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    say("world: %d models, %d repeats per setting, seed %d, spread %.1f"
        % (MODELS, W, MM.SEED, SPREAD))

    say("\nSELF-CHECK (240 items, normal difficulty, 30% behavioural plant)")
    say("  must reproduce the paper's +0.0903 / +0.0912 / +0.0315")
    s, kn, gu = cell(240, "normal", amp=0.30, tag=51)
    say("  should %+.4f | known %+.4f | guessed %+.4f | share %.1f%%"
        % (s, kn, gu, 100 * gu / s))
    selfcheck = dict(should=s, known=kn, guessed=gu, share=100 * gu / s)

    data = {"selfcheck": selfcheck, "tableA": [], "tableB": [], "tableC": []}

    say("\nTABLE A  number of items (normal difficulty, spread 1.5)")
    say("  %-6s %12s %12s %10s %12s" %
        ("items", "should(30%)", "guessed(30%)", "share", "floor"))
    for i, n in enumerate([60, 120, 240, 480, 960]):
        rs = [cell(n, "normal", amp=a, tag=100 + i) for a in AMPS]
        share = rs[3][2] / rs[3][0]
        fl = floor_from(rs)
        say("  %-6d %+12.4f %+12.4f %9.1f%% %12.4f"
            % (n, rs[3][0], rs[3][2], 100 * share, fl))
        data["tableA"].append(dict(items=n, should=rs[3][0],
                                   guessed=rs[3][2], share=100 * share,
                                   floor=fl, amps=[list(r) for r in rs]))

    say("\nTABLE B  difficulty shape (240 items, spread 1.5)")
    say("  %-10s %12s %12s %10s %12s" %
        ("shape", "should(30%)", "guessed(30%)", "share", "floor"))
    for i, shape in enumerate(["normal", "uniform", "bimodal"]):
        rs = [cell(240, shape, amp=a, tag=200 + i) for a in AMPS]
        share = rs[3][2] / rs[3][0]
        fl = floor_from(rs)
        say("  %-10s %+12.4f %+12.4f %9.1f%% %12.4f"
            % (shape, rs[3][0], rs[3][2], 100 * share, fl))
        data["tableB"].append(dict(shape=shape, should=rs[3][0],
                                   guessed=rs[3][2], share=100 * share,
                                   floor=fl, amps=[list(r) for r in rs]))

    say("\nTABLE C  two plant definitions (normal difficulty, spread 1.5)")
    say("  %-6s %-14s %12s %12s %10s" %
        ("items", "plant", "should", "guessed", "share"))
    for i, n in enumerate([240, 960]):
        beh = cell(n, "normal", amp=0.30, tag=300 + i)
        tgt = cell(n, "normal", target=TARGET, tag=310 + i)
        for label, r in [("behavioural", beh), ("target", tgt)]:
            say("  %-6d %-14s %+12.4f %+12.4f %9.1f%%"
                % (n, label, r[0], r[2], 100 * r[2] / r[0]))
            data["tableC"].append(dict(items=n, plant=label, should=r[0],
                                       guessed=r[2], share=100 * r[2] / r[0]))

    (HERE / "scan_scale_results.txt").write_text("\n".join(lines) + "\n",
                                                  encoding="utf-8")
    (HERE / "scan_scale_results.json").write_text(json.dumps(data, indent=2),
                                                   encoding="utf-8")
    say("\nwrote scan_scale_results.txt / .json")


if __name__ == "__main__":
    main()
