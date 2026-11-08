# Reproducibility package

Scripts and the derived answer table used in the paper. Prepared for anonymous
review; no author identity and no local path appear in these files.

## Single entry point

Python 3 with numpy only, CPU, no training, no external services:

    python check.py
    python minimal.py
    python bridge.py

Run `bridge.py` from this directory (it imports `minimal.py` beside it).

## What each script prints

- `check.py`: the readings on the public 42 x 8707 correctness table
  (+0.1350 / +0.1395 / -0.0044; a planted 0.0307 reads back 0.0307; the four
  ratio readings). With `--npz` it works on a supplied table and writes a
  figure.
- `minimal.py`: answer-known worlds (8 models, 240 items). The default run
  needs no external data; `--real` additionally uses the public table. It also
  prints the s.d. of the reported reading across the 60 worlds at each
  difficulty spread (0.0055-0.0144): that is the shaded band in Figure 1.
- `bridge.py`: the 3 x 2 table of readings with and without a planted 30%
  link, together with the B1-B4 checks.
- `fig1_paper.py`: redraws Figure 1 from those numbers. It needs matplotlib
  in addition to numpy; the scripts above stay numpy-only.
- `scan_scale.py`: the scale sweep behind the read-back table in Section 2
  (share and crossing over item counts, difficulty shapes, and both plant
  definitions). It imports `minimal.py` and writes `scan_scale_results.txt`
  and `scan_scale_results.json`.
- `test_grouping.py`: the 2000 model-level cluster bootstrap behind Section 3.
  It imports `check.py`, reads `answers.npz`, and writes
  `test_grouping_results.txt` and `test_grouping_results.json`.

The reported readings in the paper are produced by these scripts; the
`*_results.txt` files hold the printed output they were read from.

## Data

`answers.npz` is derived from the public release `bethgelab/lm-similarity`,
split `judge_w_gt_mmlu_pro_free_filtered`
(https://huggingface.co/datasets/bethgelab/lm-similarity). The underlying
dataset is an input only and is NOT a contribution of this work; no new dataset
is introduced. Preprocessing is contained in `check.py`, and
`check.py --samples <path-to-public-release>` rebuilds the table directly from
that release.

The native `answers.npz` is in this repository, so the scripts run directly
with no restore step. For hosting platforms that accept text files only,
`answers.npz.b64` is kept as a byte-identical text fallback (63,440 B; decode
verified byte-identical). Decode it once only if the native file is unavailable:

    base64 -d answers.npz.b64 > answers.npz        # Git Bash / Linux / macOS
    # PowerShell:
    # [IO.File]::WriteAllBytes("answers.npz", [Convert]::FromBase64String((Get-Content -Raw answers.npz.b64)))

## Public link

https://anonymous.4open.science/r/model-agreement-two-terms-1637

## Notes

- This package is anonymized: no author name, affiliation or local path appears
  in these files, and the repository history contains only anonymous commits.
- All runs are CPU-only and complete in seconds.
