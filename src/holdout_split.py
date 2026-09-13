"""
================================================================================
holdout_split.py  
================================================================================
Author:     Mehrab Jamshidi — Politecnico di Milano
Supervisor: Prof. Andrea Colagrossi

WHY THIS FILE EXISTS
--------------------
Progress Report No. 4 was reviewed and one methodological problem was identified:
the Phase 3 WGAN-GP was trained on anomaly windows extracted from the NASA test
split, and Phase 4 then evaluated synthetic-to-real detection on that SAME test
split. The synthetic data was therefore shaped, indirectly, by the very anomalies
used for evaluation. This breaks the "train on synthetic, test on UNSEEN real"
claim and inflates the data-efficiency curve.

THE FIX (channel-level held-out protocol)
-----------------------------------------
A per-channel anomaly holdout is impossible: 63 of the 81 channels have only ONE
labelled anomaly sequence, so holding it out would leave that channel with zero
anomalies to either train or test on. The only clean unit is the CHANNEL.

This module partitions the 81 channels into:
  - TRAIN channels  : their anomaly windows form the GAN's training pool, and
                      (for the data-efficiency curve) the real-anomaly windows
                      that may be added to the SVM training set.
  - HOLDOUT channels: their anomalies are NEVER seen by the GAN and NEVER added
                      to any training set. They are used only for the final
                      synthetic-to-real evaluation.

The split is stratified by primary anomaly class (point / contextual / mixed)
and fixed by SEED so it is identical across Phase 3, Phase 4, and every re-run.

This is a STRICTER and more defensible protocol than the original: it answers
"can a GAN that never saw channel X's faults still produce synthetic data that
detects them?" — a genuine generalisation question.
================================================================================
"""

import ast
import os
import sys

import numpy as np
import pandas as pd

# Make `src/` importable regardless of which phase folder the caller lives in.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import LABELS_FILE, SPLIT_ASSIGNMENT_CSV, require_dataset  # noqa: E402

SEED = 42
HOLDOUT_FRACTION = 0.25     # ~25% of channels per class go to the held-out test set


def _parse_classes(s):
    """The 'class' column looks like '[contextual, contextual]' (no quotes)."""
    inner = str(s).strip("[]")
    return [x.strip() for x in inner.split(",") if x.strip()]


def _primary_class(class_str):
    cl = _parse_classes(class_str)
    has_point = "point" in cl
    has_ctx = "contextual" in cl
    if has_point and has_ctx:
        return "mixed"
    if has_point:
        return "point"
    return "contextual"


def compute_split(labels_file=LABELS_FILE, seed=SEED, holdout_fraction=HOLDOUT_FRACTION):
    """Return (train_channels, holdout_channels) as sorted lists of chan_id.

    Deterministic given the seed. Stratified by primary anomaly class so both
    classes are represented in the held-out set.
    """
    df = pd.read_csv(labels_file).drop_duplicates("chan_id", keep="first")
    df["primary_class"] = df["class"].apply(_primary_class)

    rng = np.random.RandomState(seed)
    train, holdout = [], []
    for cls in ["point", "contextual", "mixed"]:
        chans = sorted(df[df.primary_class == cls]["chan_id"].tolist())
        chans = list(chans)
        rng.shuffle(chans)
        n_hold = max(1, round(holdout_fraction * len(chans)))
        holdout += chans[:n_hold]
        train += chans[n_hold:]
    return sorted(train), sorted(holdout)


def is_holdout(chan_id, holdout_channels):
    return chan_id in holdout_channels


if __name__ == "__main__":
    require_dataset()
    tr, ho = compute_split()
    print("=" * 70)
    print("CLEAN HELD-OUT SPLIT  (channel-level, stratified by class, seed=%d)" % SEED)
    print("=" * 70)
    print(f"GAN-training channels : {len(tr)}")
    print(f"Held-out test channels: {len(ho)}")
    print()
    print("Held-out channels (anomalies NEVER seen by the GAN or any training set):")
    print("  " + ", ".join(ho))
    print()
    # Class balance report
    df = pd.read_csv(LABELS_FILE).drop_duplicates("chan_id", keep="first")
    df["primary_class"] = df["class"].apply(_primary_class)
    print("Held-out class balance:")
    print(df[df.chan_id.isin(ho)].primary_class.value_counts().to_string())
    print()
    print("Train class balance:")
    print(df[df.chan_id.isin(tr)].primary_class.value_counts().to_string())

    # Record the assignment so the split is auditable without re-running anything.
    assignment = df[["chan_id", "spacecraft"]].copy()
    assignment["primary_class"] = df["primary_class"]
    assignment["split"] = np.where(assignment.chan_id.isin(ho), "holdout", "train")
    assignment = assignment.sort_values(["split", "chan_id"], ascending=[True, True])
    assignment.to_csv(SPLIT_ASSIGNMENT_CSV, index=False)
    print(f"\nWrote {SPLIT_ASSIGNMENT_CSV}")
