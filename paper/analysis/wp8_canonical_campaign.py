"""
================================================================================
WP8 — The canonical injection campaign
================================================================================
Paper:  Detectability of learned spacecraft anomaly detectors: minimum-
        detectable-fault envelopes by controlled fault injection
Author: Mehrab Jamshidi — Politecnico di Milano

WHY THIS RUN EXISTS

   The paper previously drew its numbers from three separate executions:

     A  phase4sens_per_channel.csv   (thesis campaign)  — Tables 1-3
     B  wp3_envelope_extended.csv    (re-run)           — Tables 4-6, learned family
     C  wp7_parametric_envelope.csv  (parametric)       — Table 6, four families

   That created two defects a referee would find.

   D1  TWO CONTOURS FOR ONE DETECTOR. Table 1 came from run A and Tables 5-6
       from run B, so the same detector, stimulus and requirement appeared at
       alpha* = 1.047 / 0.172 in one table and 0.796 / 0.115 in another. The
       abstract quoted the first and the introduction the second.

   D2  A DETECTOR-TRAINING CONFOUND IN THE FAULT-MODEL COMPARISON. wp7 calls
       train_lstm() afresh, so the four parametric families were scored on a
       different detector draw from the learned family they are compared with.
       Run-to-run detector variation moves the grid mean by ~0.018 and single
       channel-cells by up to 0.70, so part of the reported 0.51-0.68 spread
       across fault models was not attributable to the fault model.

   This script replaces all three with ONE campaign. Each channel's LSTM and VAE
   are trained once and then scored against all five fault models, so the fault
   model is the only thing that varies within a detector, and the detector is the
   only thing that varies within a fault model.

WHAT IS ADDED BEYOND THE THREE RUNS IT REPLACES

   TIMELY-DETECTION PROBABILITY. The latency surface L(alpha, d) is a mean over
   injections the detector actually flagged, and the latency of a detected
   injection is bounded above by d by construction. Both properties push the
   mean upward with duration independently of the detector, so the published
   "latency grows with duration" result was confounded. This run therefore also
   records, per cell,

       n_seg          injected fault segments presented
       n_det          segments flagged at any point inside the fault window
       n_timely_L     segments flagged within L steps of onset, L in {5,10,20}

   from which P(alarm within L steps of onset) = n_timely_L / n_seg follows.
   Undetected segments count as failures, nothing is conditioned on detection,
   and the quantity is not bounded by d. That is the quantity a fault-management
   deadline actually asks for, and it is what the latency-budget claim is now
   tested against.

FIDELITY TO THE RUNS IT REPLACES

   * Injection RNG. numpy's stream is created fresh per fault model as
     default_rng(SEED + idx), exactly as wp7 does, and is consumed only by
     inject(). For the learned family the (alpha, duration, replicate) iteration
     order is identical to wp3, so the injected streams are BIT-IDENTICAL to
     runs A and B and the only thing that differs is the detector draw. The
     learned-family comparison against runs A and B is therefore a clean
     measurement of detector-training variability, with the stimulus held fixed.

   * Detector seeding. torch is re-seeded per channel as SEED + idx and cuDNN is
     put in deterministic mode, so this run is reproducible in a way runs A and B
     were not. That also makes it an independent third draw: with A, B and C the
     contour's run-to-run spread is measured over three executions, not two.

   * Everything else — channel loading, z-scoring, nominal-segment extraction,
     the blend model, the grid, the replicate count, the architectures, the
     training loops and the deployable k=3 threshold — is imported from
     wp3_envelope_extended so it cannot drift from the run it supersedes.

USAGE   set SMAP_MSL_ROOT, then
        python analysis/wp8_canonical_campaign.py
OUTPUT  ../results/wp8_canonical_envelope.csv
================================================================================
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, f1_score

from wp3_envelope_extended import (ALPHAS, DURATIONS, LOOKBACK, MIN_CLEAN_LEN,
                                   N_INJ, N_REPLICAS, SEED, inject,
                                   load_channel, longest_nominal_segment,
                                   lstm_errors, make_sequences, parse_windows,
                                   point_adjust, train_lstm, train_vae,
                                   vae_scores)
from wp7_parametric import inject_parametric

HERE = Path(__file__).resolve().parent
from _paths import REPO  # noqa: E402
OUT = HERE.parent / "results"
from _paths import DATA_ROOT, LABELS  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The learned model is listed last so that, for each channel, the parametric
# families are scored first and the learned family's RNG stream is the final one
# consumed. Order does not matter -- each family gets its own fresh stream --
# but keeping "learned" adjacent to the reporting code makes the diff readable.
FAULT_MODELS = ("step", "ramp", "spike", "noise", "learned")
LATENCY_BUDGETS = (5, 10, 20)


def evaluate_full(errors, y_true, threshold, offset):
    """As wp3_envelope_extended.evaluate, plus the segment-level counts needed
    for a censoring-free timely-detection probability.

    The extra fields are counts, not rates, so that aggregation across
    replicates and channels is a sum over segments rather than a mean of means.
    """
    base = dict(f1_pa=0., p_pa=0., r_pa=0., f1_pw=0., p_pw=0., r_pw=0.,
                lat=np.nan, n_seg=0, n_det=0,
                **{f"n_timely_{L}": 0 for L in LATENCY_BUDGETS})
    if len(np.unique(y_true)) < 2:
        return base

    T = len(y_true)
    pl = min(len(errors), T - offset)
    y_pred = np.zeros(T, dtype=int)
    y_pred[offset:offset + pl] = (errors[:pl] > threshold).astype(int)

    # Walk the ground-truth segments once, recording for each whether it was
    # flagged at all and, if so, how many steps after onset.
    lats, n_seg, in_seg, s = [], 0, False, 0
    for t in range(T):
        if y_true[t] == 1 and not in_seg:
            in_seg, s = True, t
        elif y_true[t] == 0 and in_seg:
            in_seg = False
            n_seg += 1
            det = np.where(y_pred[s:t] == 1)[0]
            if len(det):
                lats.append(int(det[0]))
    if in_seg:
        n_seg += 1
        det = np.where(y_pred[s:] == 1)[0]
        if len(det):
            lats.append(int(det[0]))

    ya = point_adjust(y_true, y_pred)
    out = dict(
        f1_pa=f1_score(y_true, ya, zero_division=0),
        p_pa=precision_score(y_true, ya, zero_division=0),
        r_pa=recall_score(y_true, ya, zero_division=0),
        f1_pw=f1_score(y_true, y_pred, zero_division=0),
        p_pw=precision_score(y_true, y_pred, zero_division=0),
        r_pw=recall_score(y_true, y_pred, zero_division=0),
        lat=float(np.mean(lats)) if lats else np.nan,
        n_seg=n_seg, n_det=len(lats))
    for L in LATENCY_BUDGETS:
        out[f"n_timely_{L}"] = int(sum(1 for v in lats if v <= L))
    return out


def inject_any(clean, model, pool, alpha, duration, rng):
    """Dispatch to the generative or the parametric injector. Both apply the
    identical severity blend of Eq. (5); only the waveform differs."""
    if model == "learned":
        return inject(clean, pool, alpha, duration, N_INJ, rng)
    return inject_parametric(clean, model, alpha, duration, N_INJ, rng)


def main():
    print("=" * 78)
    print("WP8 — canonical injection campaign")
    print(f"device: {DEVICE}   seed: {SEED}   "
          f"fault models: {', '.join(FAULT_MODELS)}")
    print("=" * 78, flush=True)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    labels = pd.read_csv(LABELS).drop_duplicates("chan_id", keep="first")
    split = pd.read_csv(REPO / "data" / "clean_split_assignment.csv")
    holdout = set(split[split.split == "holdout"].chan_id)
    pool_pt = np.load(REPO / "data" / "synthetic" /
                      "gan_v4_clean_synth_point.npy").astype(np.float32)
    pool_cx = np.load(REPO / "data" / "synthetic" /
                      "gan_v4_clean_synth_contextual.npy").astype(np.float32)

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
            print(f"  {chan}: clean stream too short ({len(clean)}), skipped")
            continue

        # Per-channel seeding: the detector draw no longer depends on the order
        # in which channels happen to be visited, which is what makes this run
        # reproducible where runs A and B were not.
        torch.manual_seed(SEED + idx)
        np.random.seed(SEED + idx)
        lstm, thr_l = train_lstm(train_norm)
        if lstm is None:
            continue
        torch.manual_seed(SEED + idx)
        vae, thr_v = train_vae(train_norm)

        has_cx, has_pt = "contextual" in str(ac), "point" in str(ac)
        pool = (np.vstack([pool_pt, pool_cx]) if (has_cx and has_pt)
                else pool_cx if has_cx else pool_pt)

        # ---- alpha = 0 no-injection control (one per detector, not per model)
        Xs0, ys0 = make_sequences(clean)
        if len(Xs0):
            e0 = lstm_errors(lstm, torch.FloatTensor(Xs0), torch.FloatTensor(ys0))
            fp_l = float((e0 > thr_l).sum()) / max(1, len(e0))
            if vae is not None:
                v0 = vae_scores(vae, clean)
                fp_v = float((v0 > thr_v).sum()) / max(1, len(v0))
            else:
                fp_v = np.nan
            for det, fp in (("lstm", fp_l), ("vae", fp_v)):
                rows.append(dict(
                    chan_id=chan, spacecraft=sc, anomaly_class=ac, detector=det,
                    fault_model="none", alpha=0.0, duration=0,
                    f1_pa=0.0, f1_pw=0.0, p_pa=0.0, p_pw=0.0, r_pa=0.0, r_pw=0.0,
                    latency=np.nan, n_seg=0, n_det=0,
                    **{f"n_timely_{L}": 0 for L in LATENCY_BUDGETS},
                    n_replicas=1, fp_rate_no_injection=round(fp, 5)))

        for model in FAULT_MODELS:
            # Fresh stream per fault model, exactly as wp7 does. For "learned"
            # this reproduces runs A and B's injected streams bit-for-bit.
            rng = np.random.default_rng(SEED + idx)
            for alpha in ALPHAS:
                for duration in DURATIONS:
                    if len(clean) < 2 * (LOOKBACK + 16) + duration * N_INJ + 64 * N_INJ:
                        continue
                    acc = {d: [] for d in ("lstm", "vae")}
                    for _ in range(N_REPLICAS):
                        inj_stream, y_true = inject_any(
                            clean, model, pool, alpha, duration, rng)
                        if y_true.sum() == 0:
                            continue
                        Xs, ys = make_sequences(inj_stream)
                        if len(Xs) == 0:
                            continue
                        el = lstm_errors(lstm, torch.FloatTensor(Xs),
                                         torch.FloatTensor(ys))
                        acc["lstm"].append(
                            evaluate_full(el, y_true, thr_l, LOOKBACK))
                        if vae is not None:
                            ev = vae_scores(vae, inj_stream)
                            acc["vae"].append(
                                evaluate_full(ev, y_true, thr_v, 0))
                    for det, lst in acc.items():
                        if not lst:
                            continue
                        lats = [r["lat"] for r in lst if not np.isnan(r["lat"])]
                        rec = dict(
                            chan_id=chan, spacecraft=sc, anomaly_class=ac,
                            detector=det, fault_model=model,
                            alpha=alpha, duration=duration,
                            f1_pa=round(float(np.mean([r["f1_pa"] for r in lst])), 4),
                            f1_pw=round(float(np.mean([r["f1_pw"] for r in lst])), 4),
                            p_pa=round(float(np.mean([r["p_pa"] for r in lst])), 4),
                            p_pw=round(float(np.mean([r["p_pw"] for r in lst])), 4),
                            r_pa=round(float(np.mean([r["r_pa"] for r in lst])), 4),
                            r_pw=round(float(np.mean([r["r_pw"] for r in lst])), 4),
                            latency=(round(float(np.mean(lats)), 2)
                                     if lats else np.nan),
                            n_seg=int(sum(r["n_seg"] for r in lst)),
                            n_det=int(sum(r["n_det"] for r in lst)),
                            n_replicas=len(lst), fp_rate_no_injection=np.nan)
                        for L in LATENCY_BUDGETS:
                            rec[f"n_timely_{L}"] = int(
                                sum(r[f"n_timely_{L}"] for r in lst))
                        rows.append(rec)

        done += 1
        print(f"  [{done:2d}/20] {chan:<5} done  "
              f"({(time.time() - t0) / 60:.1f} min elapsed)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "wp8_canonical_envelope.csv", index=False)
    print(f"\nwrote {len(df)} rows -> {OUT / 'wp8_canonical_envelope.csv'}")
    print(f"total time {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
