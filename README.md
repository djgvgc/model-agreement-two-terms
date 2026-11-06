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
  needs no external data; `--real` additionally uses the public table.
- `bridge.py`: the 3 x 2 table of readings with and without a planted 30%
  link, together with the B1-B4 checks.

The reported readings in the paper are produced by these scripts.

## Data

`answers.npz` is derived from the public release `bethgelab/lm-similarity`,
split `judge_w_gt_mmlu_pro_free_filtered`
(https://huggingface.co/datasets/bethgelab/lm-similarity). The underlying
dataset is an input only and is NOT a contribution of this work; no new dataset
is introduced. Preprocessing is contained in `check.py`, and
`check.py --samples <path-to-public-release>` rebuilds the table directly from
that release.

For hosting platforms that accept text files only, `answers.npz.b64` is a
base64 copy of the same bytes (63,440 B; decode verified byte-identical).
Restore it once before running the scripts:

    base64 -d answers.npz.b64 > answers.npz        # Git Bash / Linux / macOS
    # PowerShell:
    # [IO.File]::WriteAllBytes("answers.npz", [Convert]::FromBase64String((Get-Content -Raw answers.npz.b64)))

The local package already contains `answers.npz`, so no restore step is needed
locally.

## Notes

- This package is local and anonymized. A public hosting URL will be added once
  the anonymous repository exists; no public link is claimed here yet.
- All runs are CPU-only and complete in seconds.
