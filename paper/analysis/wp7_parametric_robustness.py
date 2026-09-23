"""
================================================================================
WP7 — Is the parametric advantage real, or an artefact of the severity inversion?
================================================================================
Paper:  Detectability of learned spacecraft anomaly detectors: minimum-
        detectable-fault envelopes by controlled fault injection
Author: Mehrab Jamshidi — Politecnico di Milano

THE FINDING BEING STRESS-TESTED
   wp7_analyse_parametric.py finds that a STEP stimulus predicts real-fault
   detection better than the learned generator (Spearman +0.957 against +0.783),
   which falsifies the manuscript's novelty claim as originally written. Before
   rewriting the paper around that, the result has to survive the obvious
   confound.

THE CONFOUND
   Each fault model is placed on the severity axis by inverting its own
   calibration curve T_ar(alpha). That inversion is only well posed where the
   curve is strictly increasing. The generative calibration is monotone on just
   13 of 20 channels. If the parametric curves are monotone on more channels,
   the step could be winning because its severity estimate is better posed --- a
   property of the CALIBRATION, not of fault realism --- and the conclusion
   would be about our measurement procedure rather than about fault models.

WHAT IS MEASURED
   1. Monotonicity of T_ar(alpha) per fault model, per channel.
   2. The prediction comparison restricted to channels where EVERY fault model
      inverts cleanly, so no model is advantaged by a better-posed inversion.
   3. Stability across AR order.
   4. Whether the ranking survives dropping the two channels that flag their
      entire clean stream.

OUTPUT   ../results/wp7_parametric_robustness.csv
================================================================================
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from wp2_severity_metric import ar_severity, fit_ar
from wp2_validity import (ALPHAS, CONTEXTUAL, N_CAL_WINDOWS, SEED, WINDOW,
                          bilinear_surface, build_synth_segment,
                          invert_calibration, load_channel,
                          longest_nominal_segment, parse_windows, windowise)
from wp7_parametric import FAMILIES, parametric_segment

HERE = Path(__file__).resolve().parent
from _paths import REPO  # noqa: E402
RES = HERE.parent / "results"
from _paths import DATA_ROOT, LABELS  # noqa: E402

MODELS = ["generative"] + list(FAMILIES)


def calib_curve(kind, clean, starts, pool, coef, base_rms, order, rng):
    curve = []
    for a in ALPHAS:
        wins = np.empty((N_CAL_WINDOWS, WINDOW))
        for j, st in enumerate(starts):
            nom = clean[st:st + WINDOW]
            seg = (build_synth_segment(pool, WINDOW, rng) if kind == "generative"
                   else parametric_segment(kind, WINDOW, rng))
            wins[j] = nom * (1.0 - a) + seg * a
        curve.append(float(np.median(ar_severity(wins, coef, base_rms, order))))
    return np.array(curve)


def run(order: int):
    rng = np.random.default_rng(SEED)
    labels = pd.read_csv(LABELS).drop_duplicates("chan_id", keep="first")
    if _canonical():
        can = pd.read_csv(RES / "wp8_canonical_envelope.csv")
        can = can[(can.detector == "lstm") & (can.duration > 0)]
        par = can[can.fault_model != "learned"].rename(
            columns={"fault_model": "family"})
        gen = can[can.fault_model == "learned"]
        sev = pd.read_csv(RES / "wp2_severity_metric_comparison_runC.csv")
        print("envelope source: run C (wp8_canonical_envelope.csv)")
    else:
        par = pd.read_csv(RES / "wp7_parametric_envelope.csv")
        gen = pd.read_csv(RES / "wp3_envelope_extended.csv")
        gen = gen[(gen.detector == "lstm") & (gen.duration > 0)]
        sev = pd.read_csv(RES / "wp2_severity_metric_comparison.csv")
        print("envelope source: runs B+C (wp3 + wp7)")
    pool_pt = np.load(REPO / "data" / "synthetic" / "gan_v4_clean_synth_point.npy")
    pool_cx = np.load(REPO / "data" / "synthetic" / "gan_v4_clean_synth_contextual.npy")

    rows = []
    for ch in sorted(par.chan_id.unique()):
        lab = labels[labels.chan_id == ch]
        if lab.empty:
            continue
        ac = str(lab["class"].iloc[0])
        seqs = parse_windows(lab["anomaly_sequences"].iloc[0])
        _, test = load_channel(ch)
        a0, b0 = longest_nominal_segment(len(test), seqs)
        clean = test[a0:b0]
        if len(clean) < WINDOW + 40:
            continue
        coef, base_rms = fit_ar(clean, order)
        if coef is None:
            continue
        starts = rng.integers(0, len(clean) - WINDOW, size=N_CAL_WINDOWS)
        has_cx, has_pt = "contextual" in ac, "point" in ac
        pool = (np.vstack([pool_pt, pool_cx]) if (has_cx and has_pt)
                else pool_cx if has_cx else pool_pt)

        reals = []
        for s, e in seqs:
            s, e = max(0, s), min(len(test), e)
            if e - s < 8:
                continue
            if e - s >= WINDOW:
                wins = windowise(test[s:e])
            else:
                c = (s + e) // 2
                lo = max(0, min(len(test) - WINDOW, c - WINDOW // 2))
                wins = test[lo:lo + WINDOW][None, :]
            reals.append((float(np.median(ar_severity(wins, coef, base_rms, order))),
                          e - s))
        if not reals:
            continue

        rec = dict(chan_id=ch, ar_order=order,
                   cls="contextual" if ch in CONTEXTUAL else "point",
                   observed=float(sev[sev.chan_id == ch].observed_f1.iloc[0]))
        for m in MODELS:
            curve = calib_curve(m, clean, starts, pool, coef, base_rms, order, rng)
            rec[f"mono_{m}"] = bool(np.all(np.diff(curve) > 0))
            if m == "generative":
                sub = gen[gen.chan_id == ch].rename(columns={"f1_pa": "f1"})
            else:
                sub = par[(par.chan_id == ch) & (par.family == m)
                          & (par.duration > 0)].rename(columns={"f1_pa": "f1"})
            preds, ws = [], []
            for t_real, d_real in reals:
                a_hat, _ = invert_calibration(curve, t_real)
                p, _ = bilinear_surface(sub, a_hat, d_real)
                preds.append(p)
                ws.append(d_real)
            rec[f"pred_{m}"] = float(np.average(preds, weights=ws))
        rows.append(rec)
    return pd.DataFrame(rows)


def report(df, label):
    print(f"\n  {label}  (n={len(df)})")
    print(f"    {'model':<14}{'rho':>9}{'p':>9}{'MAE':>9}")
    out = {}
    for m in MODELS:
        c = f"pred_{m}"
        if c not in df or df[c].isna().all() or len(df) < 4:
            continue
        rho, p = spearmanr(df[c], df.observed)
        mae = (df[c] - df.observed).abs().mean()
        out[m] = rho
        print(f"    {m:<14}{rho:>+9.3f}{p:>9.3f}{mae:>9.3f}")
    return out


def _canonical() -> bool:
    return os.environ.get("ENVELOPE_SOURCE", "canonical").lower() in (
        "canonical", "wp8", "c")


def main():
    print("=" * 74)
    print("IS THE PARAMETRIC ADVANTAGE REAL, OR A CALIBRATION ARTEFACT?")
    print("=" * 74)

    d8 = run(8)
    d8.to_csv(RES / ("wp7_parametric_robustness" + ("_runC" if _canonical() else "") + ".csv"), index=False)

    print("\n  MONOTONICITY OF THE SEVERITY CALIBRATION (AR order 8)")
    print("  a model whose curve inverts cleanly on more channels is advantaged")
    for m in MODELS:
        k = int(d8[f"mono_{m}"].sum())
        print(f"    {m:<14} {k:>2}/{len(d8)} channels")

    report(d8, "ALL CHANNELS")

    clean_mask = np.logical_and.reduce([d8[f"mono_{m}"].to_numpy() for m in MODELS])
    report(d8[clean_mask], "ONLY CHANNELS WHERE EVERY MODEL INVERTS CLEANLY")

    patho = {"D-5", "R-1"}
    report(d8[~d8.chan_id.isin(patho)], "EXCLUDING D-5 AND R-1 (FP = 1.0)")

    print("\n  STABILITY ACROSS AR ORDER")
    ar_rows = [dict(ar_order=8, **{m: spearmanr(d8[f"pred_{m}"], d8.observed).statistic
                                   for m in MODELS})]
    for order in (4, 16):
        dd = run(order)
        r = {}
        for m in MODELS:
            r[m] = spearmanr(dd[f"pred_{m}"], dd.observed).statistic
        ar_rows.append(dict(ar_order=order, **r))
        print(f"    p={order:<3} " + "   ".join(f"{m} {v:+.3f}" for m, v in r.items()))
    pd.DataFrame(ar_rows).to_csv(
        RES / ("wp7_parametric_ar_orders" + ("_runC" if _canonical() else "") + ".csv"),
        index=False)

    print("\n" + "=" * 74)


if __name__ == "__main__":
    main()
