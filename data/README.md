# Data

## What is committed here

| Path | What it is |
|---|---|
| `clean_split_assignment.csv` | The 61/20 channel partition — one row per channel, with its spacecraft, primary anomaly class, and `train`/`holdout` assignment. Regenerate it any time with `python src/holdout_split.py`. |
| `synthetic/gan_v4_clean_synth_point.npy` | 2000 synthetic point-anomaly windows, shape `(2000, 64)`, in `[-1, 1]`. Output of Phase 3. |
| `synthetic/gan_v4_clean_synth_contextual.npy` | 2000 synthetic contextual-anomaly windows, same shape and scaling. |

The two `.npy` banks are what let Phase 4 run **without** retraining the GAN.

## What is NOT committed: the NASA SMAP/MSL dataset

The telemetry benchmark (Hundman et al., 2018) is not redistributed here, for size and
licensing reasons. It is the property of its original authors.

Download it from Kaggle:
<https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl>

The archive is laid out like this:

```
archive/
├── labeled_anomalies.csv
└── data/
    └── data/
        ├── train/   # <chan_id>.npy — nominal only
        └── test/    # <chan_id>.npy — nominal + anomalous
```

Point the code at it in either of two ways:

**Option 1 — environment variable (recommended, nothing to move):**

```bash
export SMAP_MSL_ROOT="/path/to/archive/data/data"          # bash / zsh
$env:SMAP_MSL_ROOT = "D:\...\archive\data\data"            # PowerShell
```

`labeled_anomalies.csv` is found automatically two levels up from that folder, which is where
the Kaggle archive puts it. If your copy is arranged differently, set `SMAP_MSL_LABELS` to the
file directly.

**Option 2 — drop it in place:** copy `train/`, `test/` and `labeled_anomalies.csv` into
`data/raw/`, which is the default and is git-ignored.

Every script calls `require_dataset()` at startup, so a missing or misconfigured dataset fails
immediately with a message naming the exact path it looked for, rather than part-way through a
long run.

## Conventions used everywhere

Identical across all four phases, so results stay comparable:

- **81 unique channels** after dropping the duplicate `P-2` row (`keep='first'`).
- **Primary sensor only** (column 0 of each `.npy`).
- Window **64**, step **16**, overlap threshold **0.1**.
- Channel-level z-score using *training-split* statistics, clipped at ±3σ and divided by 3 → `[-1, 1]`.
- **Seed 42** everywhere.
- Point-adjust F1 for the VAE and LSTM; window-level F1 for the SVM.
