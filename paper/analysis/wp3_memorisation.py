"""
================================================================================
WP3-I — Nearest-neighbour memorisation check on the generative fault source
================================================================================
Paper:  Detectability of learned spacecraft anomaly detectors: minimum-
        detectable-fault envelopes by controlled fault injection
Author: Mehrab Jamshidi — Politecnico di Milano

WHY THIS IS NOT OPTIONAL
   The whole method rests on the injected stimulus being GENERATED rather than
   RETRIEVED. If the generator were largely reproducing its training windows,
   the detection envelope would be a measurement of how well the detector
   responds to a handful of memorised real faults, dressed up as a continuous
   parameter sweep, and the severity axis would be close to meaningless. The
   thesis explicitly recorded that no such check had been run. This runs it.

   Note the check is CPU-only and needs no trained model: it compares the
   committed synthetic windows against the real windows the generator was
   trained on.

THE DESIGN — TWO REFERENCE SCALES, NOT ONE
   A raw nearest-neighbour distance means nothing on its own; it has to be read
   against how close real windows get to each other. Two baselines are used.

   1. REAL-TO-REAL. For each real training window, the distance to its nearest
      OTHER real training window. This is the scale at which genuine fault
      windows in this dataset resemble one another. If synthetic-to-real
      distances sat well below this, the generator would be interpolating
      inside the training set more tightly than the training set is spaced —
      the signature of memorisation.

   2. HELDOUT-TO-TRAIN. For each real anomaly window from the 20 HELD-OUT
      channels, the distance to the nearest real training window. These are
      genuine, unseen real faults, so this is what "a novel real window" looks
      like from the training pool's point of view. It is the more informative
      of the two: if synthetic windows sit at a distance comparable to or
      greater than genuinely unseen real windows, they are not retrieved copies.

   Windows are compared in the generator's own output space (tanh-scaled,
   64 steps), which is the space in which retrieval would have to occur.

INPUTS   NASA SMAP/MSL dataset; the repo's committed synthetic pools and split
OUTPUT   ../results/wp3_memorisation.csv
================================================================================
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
from _paths import REPO  # noqa: E402
OUT = HERE.parent / "results"
OUT.mkdir(parents=True, exist_ok=True)
from _paths import DATA_ROOT, LABELS  # noqa: E402

WINDOW, STEP, CLIP = 64, 16, 3.0
SEED = 42


def parse_windows(s):
    n = [int(x) for x in re.findall(r"\d+", str(s))]
    return [[n[i], n[i + 1]] for i in range(0, len(n) - 1, 2)]


def parse_classes(s):
    return [c.strip() for c in str(s).strip("[]").split(",")]


def scale_to_tanh(w):
    return (np.clip(w, -CLIP, CLIP) / CLIP).astype(np.float32)


def extract_pool(channels: set[str]):
    """Reproduce the generator's window-extraction loop exactly: channel z-score
    on train statistics, per-anomaly-sequence windowing at stride 16, short
    sequences padded with surrounding context, then tanh scaling."""
    labels = pd.read_csv(LABELS).drop_duplicates("chan_id", keep="first")
    pools = {"point": [], "contextual": []}
    for _, row in labels.iterrows():
        chan = row["chan_id"]
        if chan not in channels:
            continue
        tp = DATA_ROOT / "test" / f"{chan}.npy"
        rp = DATA_ROOT / "train" / f"{chan}.npy"
        if not (tp.exists() and rp.exists()):
            continue
        tr = np.load(rp)[:, 0]
        te = np.load(tp)[:, 0]
        mu, sd = tr.mean(), tr.std()
        if sd < 1e-9:
            sd = 1.0
        test_norm = (te - mu) / sd

        for (s, e), cls in zip(parse_windows(row["anomaly_sequences"]),
                               parse_classes(row["class"])):
            s, e = max(0, s), min(len(test_norm), e)
            if e - s < WINDOW:
                pad = WINDOW - (e - s)
                a = max(0, s - pad // 2)
                b = min(len(test_norm), e + (pad - pad // 2))
                w = test_norm[a:b]
                if len(w) < WINDOW:
                    continue
                wins = [w[:WINDOW]]
            else:
                seq = test_norm[s:e]
                wins = [seq[i:i + WINDOW]
                        for i in range(0, len(seq) - WINDOW + 1, STEP)]
            key = "point" if cls == "point" else "contextual"
            pools[key].extend(scale_to_tanh(w) for w in wins)
    return {k: (np.stack(v) if v else np.empty((0, WINDOW), np.float32))
            for k, v in pools.items()}


def nn_distances(A: np.ndarray, B: np.ndarray, exclude_self=False,
                 block=512) -> np.ndarray:
    """Min Euclidean distance from each row of A to any row of B."""
    out = np.empty(len(A), dtype=np.float64)
    b2 = (B ** 2).sum(1)
    for i in range(0, len(A), block):
        chunk = A[i:i + block]
        d2 = (chunk ** 2).sum(1)[:, None] + b2[None, :] - 2.0 * chunk @ B.T
        np.maximum(d2, 0, out=d2)
        if exclude_self:
            for j in range(len(chunk)):
                d2[j, i + j] = np.inf
        out[i:i + block] = np.sqrt(d2.min(axis=1))
    return out


def main():
    split = pd.read_csv(REPO / "data" / "clean_split_assignment.csv")
    train_ch = set(split[split.split == "train"].chan_id)
    hold_ch = set(split[split.split == "holdout"].chan_id)
    print(f"train channels {len(train_ch)} | held-out {len(hold_ch)}\n")

    real_train = extract_pool(train_ch)
    real_hold = extract_pool(hold_ch)
    synth = {
        "point": np.load(REPO / "data" / "synthetic" /
                         "gan_v4_clean_synth_point.npy").astype(np.float32),
        "contextual": np.load(REPO / "data" / "synthetic" /
                              "gan_v4_clean_synth_contextual.npy").astype(np.float32),
    }

    rows = []
    for cls in ("point", "contextual"):
        R, H, S = real_train[cls], real_hold[cls], synth[cls]
        print(f"[{cls}] real train {R.shape} | real held-out {H.shape} | "
              f"synthetic {S.shape}")
        if len(R) < 2 or len(S) == 0:
            print("   insufficient data, skipped")
            continue

        # The tanh scaling clips at +-3 sigma, so any window that spends its
        # whole length outside the nominal band collapses to a CONSTANT vector
        # of +-1. Such windows are exact duplicates of one another, which drives
        # the real-to-real nearest-neighbour baseline to zero and makes it
        # meaningless. Report the saturation rate and deduplicate the real pools
        # before computing baselines.
        # Threshold at 1e-3 rather than machine epsilon: the generator's tanh
        # output approaches but never exactly reaches +-1, so its saturated
        # windows have std of order 1e-4. Treating those as non-saturated would
        # misreport them as near-duplicate "retrievals" of the real constant
        # windows, when in fact both are just the fully-clipped excursion.
        def sat_rate(X):
            return float((X.std(axis=1) < 1e-3).mean()) if len(X) else float("nan")

        sat_R, sat_H, sat_S = sat_rate(R), sat_rate(H), sat_rate(S)
        Rd = np.unique(R, axis=0)
        Hd = np.unique(H, axis=0) if len(H) else H
        print(f"   saturated (constant) windows: real train {sat_R*100:.1f}%  "
              f"real held-out {sat_H*100:.1f}%  synthetic {sat_S*100:.1f}%")
        print(f"   unique real train windows: {len(Rd)}/{len(R)}")

        d_syn = nn_distances(S, Rd)
        d_real = nn_distances(Rd, Rd, exclude_self=True)
        d_hold = nn_distances(Hd, Rd) if len(Hd) else np.array([np.nan])
        R, H = Rd, Hd

        # A memorised generator would place synthetic windows abnormally close
        # to the training pool: below the tight tail of real-to-real spacing.
        # Saturated windows are excluded from the test on both sides, because a
        # fully-clipped excursion is a single degenerate point in this space
        # that any generator reaches without retrieving anything.
        thr = np.percentile(d_real, 1)
        live = S.std(axis=1) >= 1e-3
        frac_below = float((d_syn[live] < thr).mean()) if live.any() else float("nan")

        rows.append(dict(
            anomaly_class=cls,
            n_real_train=len(R), n_real_holdout=len(H), n_synth=len(S),
            sat_rate_real_train=sat_R, sat_rate_real_holdout=sat_H,
            sat_rate_synth=sat_S,
            syn_to_train_median=float(np.median(d_syn)),
            syn_to_train_p05=float(np.percentile(d_syn, 5)),
            syn_to_train_min=float(d_syn.min()),
            real_to_real_median=float(np.median(d_real)),
            real_to_real_p01=float(thr),
            holdout_to_train_median=float(np.median(d_hold)),
            ratio_syn_over_holdout=float(np.median(d_syn) / np.median(d_hold)),
            ratio_syn_over_realreal=float(np.median(d_syn) / np.median(d_real)),
            frac_synth_below_realreal_p01=frac_below))

        print(f"   synthetic -> train   NN distance: median {np.median(d_syn):.3f}"
              f"  p05 {np.percentile(d_syn,5):.3f}  min {d_syn.min():.3f}")
        print(f"   real      -> real    NN distance: median {np.median(d_real):.3f}"
              f"  p01 {thr:.3f}")
        print(f"   held-out  -> train   NN distance: median {np.median(d_hold):.3f}")
        print(f"   ratio synthetic/held-out = {np.median(d_syn)/np.median(d_hold):.2f}"
              f"   (>= ~1 means synthetic sits no closer than genuinely unseen real)")
        print(f"   synthetic below the real-to-real 1st percentile: "
              f"{frac_below*100:.2f}%\n")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "wp3_memorisation.csv", index=False)

    print("=" * 74)
    print("READING")
    for r in df.itertuples():
        verdict = ("no evidence of memorisation"
                   if r.ratio_syn_over_holdout >= 0.9 and
                   r.frac_synth_below_realreal_p01 < 0.01
                   else "REVIEW — synthetic windows sit unusually close to training data")
        print(f"  {r.anomaly_class:<11} {verdict}")
    print(f"\nwritten -> {OUT / 'wp3_memorisation.csv'}")


if __name__ == "__main__":
    main()
