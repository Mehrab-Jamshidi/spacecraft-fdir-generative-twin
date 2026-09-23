"""
================================================================================
WP9 — Canonical campaign, re-executed with same-instance real-fault scoring
================================================================================
Paper:  Detectability of learned spacecraft anomaly detectors: minimum-
        detectable-fault envelopes by controlled fault injection
Author: Mehrab Jamshidi — Politecnico di Milano

WHY THIS RUN EXISTS

   The validity test places each real labelled anomaly on its channel's
   envelope and compares the predicted detection with the detection actually
   observed. Until now the two sides came from DIFFERENT trained detectors:
   the prediction from the envelope campaign's per-channel LSTM, the
   observation from the thesis Phase-2 LSTM (phase4_master_comparison.csv,
   column p2_lstm_k3_f1), trained separately. Retraining moves single
   channel-cells by up to 0.70 in F1 and one channel's nominal false-alarm rate
   from 0.11 to 0.72, so a comparison across instances mixes detector-draw
   variability into the quantity being validated.

   This run repeats the canonical campaign (wp8_canonical_campaign.py) exactly
   — same channel order, same per-channel torch/numpy seeding, cuDNN in
   deterministic mode, same injection streams — and, with the SAME per-channel
   detector instances, additionally records:

     1  real-fault detection on the channel's full test split, under the
        deployable k=3 threshold, point-adjust and point-wise
        -> wp9_real_detection.csv          (one row per channel x detector)
        -> wp9_real_segments.csv           (one row per labelled anomaly)

     2  chance-alarm rates on the un-injected clean stream: the fraction of
        windows of w steps that contain at least one alarm, for the window
        lengths used by the timely-detection budgets (L+1 = 6, 11, 21) and by
        the injected durations (16 ... 256). This is the false-call baseline
        against which a probability of detection must be read.
        -> wp9_chance_windows.csv          (one row per channel x detector)

   The injection envelope is written again to wp9_canonical_envelope.csv so it
   can be compared cell-by-cell with wp8_canonical_envelope.csv. If they are
   identical the campaign is bit-reproducible; either way, every quantity in
   the three files above comes from one detector instance per channel.

USAGE   set SMAP_MSL_ROOT, then
        python analysis/wp9_same_instance_campaign.py
================================================================================
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from wp3_envelope_extended import (ALPHAS, DURATIONS, LOOKBACK, MIN_CLEAN_LEN,
                                   N_INJ, N_REPLICAS, SEED, load_channel,
                                   longest_nominal_segment, lstm_errors,
                                   make_sequences, parse_windows, train_lstm,
                                   train_vae, vae_scores)
from wp8_canonical_campaign import (FAULT_MODELS, LATENCY_BUDGETS,
                                    evaluate_full, inject_any)

HERE = Path(__file__).resolve().parent
from _paths import REPO  # noqa: E402
OUT = HERE.parent / "results"
from _paths import DATA_ROOT, LABELS  # noqa: E402

CHANCE_WINDOWS = (6, 11, 21, 16, 32, 64, 128, 256)


def ground_truth(n, seqs):
    """Same convention as the thesis Phase-2 scripts: y[start:end] = 1."""
    y = np.zeros(n, dtype=int)
    for s, e in seqs:
        y[max(0, s):min(n, e)] = 1
    return y


def alarm_train(scores, threshold, n, offset):
    """Per-step alarm vector of length n, scores aligned from `offset`."""
    a = np.zeros(n, dtype=int)
    pl = min(len(scores), n - offset)
    if pl > 0:
        a[offset:offset + pl] = (scores[:pl] > threshold).astype(int)
    return a


def chance_rates(alarms, offset):
    """Fraction of w-step windows, starting where the detector has an output,
    that contain at least one alarm. Alarms on the un-injected stream are false
    by construction, so this is the probability that a window is 'detected' by
    chance alone."""
    a = alarms[offset:].astype(int)
    c = np.concatenate([[0], np.cumsum(a)])
    out = {}
    for w in CHANCE_WINDOWS:
        if len(a) < w:
            out[f"chance_w{w}"] = np.nan
            continue
        hits = (c[w:] - c[:-w]) > 0
        out[f"chance_w{w}"] = float(hits.mean())
    out["fp_rate_stepwise"] = float(a.mean()) if len(a) else np.nan
    return out


def segment_records(chan, det, alarms, seqs, cls):
    rows = []
    n = len(alarms)
    for k, (s, e) in enumerate(seqs):
        s0, e0 = max(0, s), min(n, e)
        hit = np.where(alarms[s0:e0] == 1)[0]
        rows.append(dict(chan_id=chan, detector=det, anomaly_idx=k, cls=cls,
                         start=s0, end=e0, duration=e0 - s0,
                         detected=int(len(hit) > 0),
                         latency=int(hit[0]) if len(hit) else np.nan))
    return rows


def main():
    print("=" * 78)
    print("WP9 — canonical campaign with same-instance real-fault scoring")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}   seed: {SEED}   fault models: {', '.join(FAULT_MODELS)}")
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

    env_rows, real_rows, seg_rows, chance_rows = [], [], [], []
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

        # ---- identical to wp8_canonical_campaign.main from here -----------
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

        Xs0, ys0 = make_sequences(clean)
        e0 = lstm_errors(lstm, torch.FloatTensor(Xs0), torch.FloatTensor(ys0))
        fp_l = float((e0 > thr_l).sum()) / max(1, len(e0))
        v0 = vae_scores(vae, clean) if vae is not None else None
        fp_v = (float((v0 > thr_v).sum()) / max(1, len(v0))
                if v0 is not None else np.nan)
        for det, fp in (("lstm", fp_l), ("vae", fp_v)):
            env_rows.append(dict(
                chan_id=chan, spacecraft=sc, anomaly_class=ac, detector=det,
                fault_model="none", alpha=0.0, duration=0,
                f1_pa=0.0, f1_pw=0.0, p_pa=0.0, p_pw=0.0, r_pa=0.0, r_pw=0.0,
                latency=np.nan, n_seg=0, n_det=0,
                **{f"n_timely_{L}": 0 for L in LATENCY_BUDGETS},
                n_replicas=1, fp_rate_no_injection=round(fp, 5)))

        for model in FAULT_MODELS:
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
                            acc["vae"].append(evaluate_full(ev, y_true, thr_v, 0))
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
                        env_rows.append(rec)
        # ---- end of the part identical to wp8 ------------------------------

        cls = ("contextual" if ("contextual" in str(ac) and "point" not in str(ac))
               else "point")

        # (2) chance-alarm rates on the un-injected clean stream
        al0 = alarm_train(e0, thr_l, len(clean), LOOKBACK)
        chance_rows.append(dict(chan_id=chan, detector="lstm", cls=cls,
                                clean_len=len(clean), **chance_rates(al0, LOOKBACK)))
        if v0 is not None:
            av0 = alarm_train(v0, thr_v, len(clean), 0)
            chance_rows.append(dict(chan_id=chan, detector="vae", cls=cls,
                                    clean_len=len(clean), **chance_rates(av0, 0)))

        # (1) real-fault detection on the full test split, same instances
        n = len(test_norm)
        y_real = ground_truth(n, seqs)
        Xt, yt = make_sequences(test_norm)
        et = lstm_errors(lstm, torch.FloatTensor(Xt), torch.FloatTensor(yt))
        r_l = evaluate_full(et, y_real, thr_l, LOOKBACK)
        real_rows.append(dict(chan_id=chan, spacecraft=sc, anomaly_class=ac,
                              cls=cls, detector="lstm", threshold=float(thr_l),
                              fp_rate_no_injection=fp_l,
                              **{k: v for k, v in r_l.items()}))
        seg_rows += segment_records(chan, "lstm",
                                    alarm_train(et, thr_l, n, LOOKBACK), seqs, cls)
        if vae is not None:
            vt = vae_scores(vae, test_norm)
            r_v = evaluate_full(vt, y_real, thr_v, 0)
            real_rows.append(dict(chan_id=chan, spacecraft=sc, anomaly_class=ac,
                                  cls=cls, detector="vae", threshold=float(thr_v),
                                  fp_rate_no_injection=fp_v,
                                  **{k: v for k, v in r_v.items()}))
            seg_rows += segment_records(chan, "vae",
                                        alarm_train(vt, thr_v, n, 0), seqs, cls)

        done += 1
        print(f"  [{done:2d}/20] {chan:<5} done  real F1_pa lstm={r_l['f1_pa']:.3f}"
              f"  FP={fp_l:.4f}  ({(time.time() - t0) / 60:.1f} min)", flush=True)

    pd.DataFrame(env_rows).to_csv(OUT / "wp9_canonical_envelope.csv", index=False)
    pd.DataFrame(real_rows).to_csv(OUT / "wp9_real_detection.csv", index=False)
    pd.DataFrame(seg_rows).to_csv(OUT / "wp9_real_segments.csv", index=False)
    pd.DataFrame(chance_rows).to_csv(OUT / "wp9_chance_windows.csv", index=False)
    print(f"\nwrote wp9_* files; total time {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
