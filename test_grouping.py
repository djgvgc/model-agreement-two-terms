#!/usr/bin/env python
"""Grouping check with resampling that respects shared models.

The paper's intervals resample PAIRS and treat them as independent, which the
paper itself flags as optimistic because pairs share models. This script
resamples MODELS instead (cluster bootstrap), so every resample keeps the
dependence between pairs that share a model. It also reproduces the original
pair-level interval as a self-check.

    python test_grouping.py

Writes test_grouping_results.txt / .json beside this file. Only numpy.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check as C

R = 2000
SEED = 20261105
LABELS = {3: "one checkpoint", 2: "shared teacher",
          1: "same maker", 0: "different makers"}


def main():
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    B, names = C.from_npz(HERE / "answers.npz")
    m, n = B.shape
    iu = np.triu_indices(m, 1)
    P = C.guess_difficulty(B)
    D = C.alike(B) - C.expected_alike(P)          # pair-wise remainder B
    d = D[iu]

    short = [C.short(x) for x in names]
    REL = np.zeros((m, m), dtype=int)
    for i in range(m):
        for j in range(m):
            if i != j:
                REL[i, j] = C.relation(short[i], short[j])
    rel = REL[iu]

    groups = {L: d[rel == L] for L in [3, 2, 1, 0]}
    obs = {L: float(g.mean()) for L, g in groups.items()}
    gap = obs[3] - obs[0]

    say("table: %d models x %d items, %d pairs" % (m, n, len(d)))
    say("SELF-CHECK (must match the paper's group means and counts)")
    for L in [3, 2, 1, 0]:
        say("  %-18s n=%3d  mean %+.4f  all positive: %s"
            % (LABELS[L], len(groups[L]), obs[L], bool((groups[L] > 0).all())))
    say("  gap (one checkpoint - different makers) = %+.4f" % gap)
    say("  paper reports +0.0304 / +0.0059 / +0.0054 / -0.0074 and gap +0.0378")

    # ---- pair-level bootstrap (reproduces the paper's optimistic intervals)
    rng = np.random.default_rng(SEED)
    pair_gaps, pair_means = [], {L: [] for L in [3, 2, 1, 0]}
    for _ in range(R):
        idx = rng.integers(0, len(d), len(d))
        dd, rr = d[idx], rel[idx]
        pg = {L: dd[rr == L].mean() for L in [3, 2, 1, 0] if (rr == L).any()}
        for L, v in pg.items():
            pair_means[L].append(float(v))
        if 3 in pg and 0 in pg:
            pair_gaps.append(float(pg[3] - pg[0]))

    def ci(vals):
        return [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))]

    say("\nPAIR-LEVEL bootstrap (pairs treated as independent; the paper's version)")
    say("  gap 95%% interval [%+.4f, %+.4f]  (paper: [+0.0311, +0.0453])"
        % tuple(ci(pair_gaps)))

    # ---- model-level cluster bootstrap (keeps pairs sharing a model together)
    rng2 = np.random.default_rng(SEED + 1)
    pos_i, pos_j = np.triu_indices(m, 1)
    mod_gaps, mod_means = [], {L: [] for L in [3, 2, 1, 0]}
    for _ in range(R):
        sel = rng2.integers(0, m, m)              # resample MODELS with replacement
        a, b = sel[pos_i], sel[pos_j]
        ok = a != b                              # drop a model paired with itself
        a, b = a[ok], b[ok]
        dd, rr = D[a, b], REL[a, b]
        mg = {}
        for L in [3, 2, 1, 0]:
            v = dd[rr == L]
            if len(v):
                mg[L] = float(v.mean())
                mod_means[L].append(mg[L])
        if 3 in mg and 0 in mg:
            mod_gaps.append(float(mg[3] - mg[0]))

    say("\nMODEL-LEVEL cluster bootstrap (resample models, keep shared-model dependence)")
    for L in [3, 2, 1, 0]:
        lo, hi = ci(mod_means[L])
        say("  %-18s mean %+.4f  95%% interval [%+.4f, %+.4f]"
            % (LABELS[L], obs[L], lo, hi))
    lo, hi = ci(mod_gaps)
    say("  gap                 mean %+.4f  95%% interval [%+.4f, %+.4f]"
        % (gap, lo, hi))
    frac_below_zero = float(np.mean(np.asarray(mod_gaps) <= 0.0))
    say("  fraction of resamples with gap <= 0: %.4f" % frac_below_zero)

    data = dict(
        counts={LABELS[L]: int(len(groups[L])) for L in [3, 2, 1, 0]},
        means={LABELS[L]: obs[L] for L in [3, 2, 1, 0]},
        gap=gap,
        pair_ci=ci(pair_gaps),
        model_ci={LABELS[L]: ci(mod_means[L]) for L in [3, 2, 1, 0]},
        model_gap_ci=ci(mod_gaps),
        frac_gap_le_zero=frac_below_zero,
    )
    (HERE / "test_grouping_results.txt").write_text("\n".join(lines) + "\n",
                                                     encoding="utf-8")
    (HERE / "test_grouping_results.json").write_text(json.dumps(data, indent=2),
                                                      encoding="utf-8")
    say("\nwrote test_grouping_results.txt / .json")


if __name__ == "__main__":
    main()
