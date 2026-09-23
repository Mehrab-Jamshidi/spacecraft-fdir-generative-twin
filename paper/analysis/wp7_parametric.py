"""
================================================================================
WP7 — Does the LEARNED fault model matter? Parametric-injection control
================================================================================
Paper:  Detectability of learned spacecraft anomaly detectors: minimum-
        detectable-fault envelopes by controlled fault injection
Author: Mehrab Jamshidi — Politecnico di Milano

THE OBJECTION THIS ANSWERS
   The paper's stated novelty against the fault-injection literature is that the
   stimulus comes from a generative model fitted to the real fault record rather
   than from a hand-specified parametric model. That is an architectural
   difference. Whether it MATTERS is an empirical question the paper does not
   currently answer, and it is the first thing a reviewer will ask: why not just
   inject a step, a ramp and a spike, as every fault-injection campaign since
   the 1990s has done?

   If a parametric envelope were indistinguishable from the generative one, the
   generator would be an expensive way to reproduce a simple result and the
   contribution would be substantially weaker.

THE TEST
   The identical campaign is re-run with the generator replaced by the four
   parametric fault families of the classical spacecraft FDIR taxonomy that have
   a severity parameter -- step (hard-over), ramp (drift), spike train, and
   noise burst (erratic) -- scaled into the same z-score units so that alpha
   means the same thing as in the generative campaign.

   Two questions are then asked.

   Q1  Do the envelopes differ? A different contour means the choice of fault
       model changes the design statement, which already matters.

   Q2  THE DECISIVE ONE. Which envelope better predicts detection of the REAL
       labelled faults? The generative envelope's whole justification is that
       its stimulus resembles the faults the spacecraft actually suffers. That
       is testable: place each real fault on each envelope and compare rank
       agreement with observed detection. If the generative envelope predicts
       real faults better than the parametric ones, the learned fault model is
       doing work that a hand-specified one cannot.

   The parametric campaign reuses the same trained detector, the same clean
   streams, the same grid and the same replicate count, so the fault model is
   the only thing that changes.

USAGE   set SMAP_MSL_ROOT, then
        python analysis/wp7_parametric.py
OUTPUT  ../results/wp7_parametric_envelope.csv
================================================================================
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from wp3_envelope_extended import (ALPHAS, DURATIONS, LOOKBACK, MIN_CLEAN_LEN,
                                   N_INJ, N_REPLICAS, SEED, evaluate,
                                   load_channel, longest_nominal_segment,
                                   lstm_errors, make_sequences, parse_windows,
                                   pick_positions, train_lstm)

HERE = Path(__file__).resolve().parent
from _paths import REPO  # noqa: E402
OUT = HERE.parent / "results"
import os
from _paths import DATA_ROOT, LABELS  # noqa: E402

# The parametric families of the classical spacecraft FDIR taxonomy that carry a
# severity parameter. Amplitudes are in z-score units and scaled by 3 so that
# alpha = 1 means the same excursion magnitude as the generator's tanh output
# rescaled by 3 -- i.e. the severity axes are directly comparable.
FAMILIES = ("step", "ramp", "spike", "noise")
SCALE = 3.0


def parametric_segment(family: str, duration: int, rng) -> np.ndarray:
    """A fault waveform of the requested family, in z-score units at unit
    severity (the caller scales by alpha)."""
    t = np.arange(duration, dtype=np.float32)
    sign = 1.0 if rng.random() < 0.5 else -1.0
    if family == "step":                      # hard-over: constant offset
        return np.full(duration, sign * SCALE, dtype=np.float32)
    if family == "ramp":                      # drift: linear accumulation
        return (sign * SCALE * (t + 1) / duration).astype(np.float32)
    if family == "spike":                     # spike train
        seg = np.zeros(duration, dtype=np.float32)
        n = max(1, duration // 16)
        for p in rng.choice(duration, size=n, replace=False):
            seg[p] = sign * SCALE
        return seg
    if family == "noise":                     # erratic: variance increase
        return (rng.standard_normal(duration) * SCALE).astype(np.float32)
    raise ValueError(family)


def inject_parametric(clean, family, alpha, duration, n_inj, rng):
    """Identical blend to the generative campaign; only the waveform differs."""
    inj = clean.copy().astype(np.float32)
    y = np.zeros(len(inj), dtype=int)
    for loc in pick_positions(len(inj), duration, n_inj, rng):
        seg = parametric_segment(family, duration, rng)
        inj[loc:loc + duration] = inj[loc:loc + duration] * (1.0 - alpha) + seg * alpha
        y[loc:loc + duration] = 1
    return inj, y


def main():
    print("=" * 74)
    print("WP7 — parametric-injection control")
    print(f"device: {'cuda' if torch.cuda.is_available() else 'cpu'}   "
          f"families: {', '.join(FAMILIES)}")
    print("=" * 74)

    labels = pd.read_csv(LABELS).drop_duplicates("chan_id", keep="first")
    split = pd.read_csv(REPO / "data" / "clean_split_assignment.csv")
    holdout = set(split[split.split == "holdout"].chan_id)

    rows = []
    t0 = time.time()
    done = 0
    for idx, row in labels.iterrows():
        chan, sc, ac = row["chan_id"], row["spacecraft"], row["class"]
        if chan not in holdout:
            continue
        ch = load_channel(chan)
        if ch is None:
            continue
        train_norm, test_norm = ch
        seqs = parse_windows(row["anomaly_sequences"])
        a, b = longest_nominal_segment(test_norm, seqs)
        clean = test_norm[a:b]
        if len(clean) < MIN_CLEAN_LEN:
            continue
        lstm, thr = train_lstm(train_norm)
        if lstm is None:
            continue

        for family in FAMILIES:
            rng = np.random.default_rng(SEED + idx)     # same stream per family
            for alpha in ALPHAS:
                for duration in DURATIONS:
                    if len(clean) < 2 * (LOOKBACK + 16) + duration * N_INJ + 64 * N_INJ:
                        continue
                    acc = []
                    for _ in range(N_REPLICAS):
                        s, y = inject_parametric(clean, family, alpha, duration,
                                                 N_INJ, rng)
                        if y.sum() == 0:
                            continue
                        Xs, ys = make_sequences(s)
                        if len(Xs) == 0:
                            continue
                        e = lstm_errors(lstm, torch.FloatTensor(Xs),
                                        torch.FloatTensor(ys))
                        acc.append(evaluate(e, y, thr, LOOKBACK))
                    if not acc:
                        continue
                    lats = [r["lat"] for r in acc if not np.isnan(r["lat"])]
                    rows.append(dict(
                        chan_id=chan, spacecraft=sc, anomaly_class=ac,
                        family=family, alpha=alpha, duration=duration,
                        f1_pa=round(float(np.mean([r["f1_pa"] for r in acc])), 4),
                        f1_pw=round(float(np.mean([r["f1_pw"] for r in acc])), 4),
                        latency=round(float(np.mean(lats)), 2) if lats else np.nan,
                        n_replicas=len(acc)))
        done += 1
        print(f"  [{done:2d}/20] {chan:<5}  ({(time.time()-t0)/60:.1f} min)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "wp7_parametric_envelope.csv", index=False)
    print(f"\nwrote {len(df)} rows -> {OUT / 'wp7_parametric_envelope.csv'}")
    print(f"total {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
