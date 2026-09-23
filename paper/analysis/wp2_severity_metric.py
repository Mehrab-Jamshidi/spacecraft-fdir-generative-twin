"""
================================================================================
WP2 E3 — The severity axis must be defined in the metric the detector uses
================================================================================
Paper:  Detectability of learned spacecraft anomaly detectors: minimum-
        detectable-fault envelopes by controlled fault injection
Author: Mehrab Jamshidi — Politecnico di Milano

WHERE THIS COMES FROM
   wp2_validity.py placed each real labelled anomaly on the envelope's severity
   axis by inverting a calibration curve built in the SIX-FEATURE space (mean,
   sd, min, max, range, lag-1 autocorrelation) that the thesis uses for its
   supervised classifier and its Frechet fidelity metric. The envelope then
   over-predicted real detection, and the over-prediction was concentrated in
   the contextual channels (bias +0.31; three of six contextual channels have
   essentially zero real detection against predictions of 0.26 to 0.92).

   wp2_failure_mechanism.py ruled out the obvious explanation: the shift between
   the nominal context around the real anomalies and the clean segment used to
   build the envelope does not predict the error (rho = +0.16, p = 0.50).

THE HYPOTHESIS TESTED HERE
   The mismatch is in the severity metric, not the envelope. The six-feature
   space is DISTRIBUTIONAL: it asks how displaced a window is. The detector is
   PREDICTIVE: it scores one-step-ahead prediction error. For point faults the
   two agree, because an amplitude excursion is both displaced and
   unpredictable. For contextual faults they come apart — a window can be
   modestly displaced yet highly unpredictable, or vice versa. If severity is
   inverted in the wrong metric, a real contextual anomaly is placed at the
   wrong alpha and the envelope is read at the wrong point.

WHAT IS DONE
   The severity calibration is rebuilt with a PREDICTIVE statistic, chosen to
   mirror the detector's own scoring mechanism without requiring the detector:
   an AR(p) model fitted by least squares on the channel's clean nominal segment,
   with

       T_ar(window) = RMS one-step-ahead AR residual on the window,
                      normalised by the AR residual RMS on clean nominal.

   T_ar = 1 means "as predictable as nominal". This is a linear surrogate for
   what the LSTM does, needs no GPU, and introduces no free parameter beyond the
   AR order. The two severity metrics are then compared head to head on the same
   predictive-validity test.

INTERPRETATION FIXED IN ADVANCE
   If the predictive metric materially improves the transfer — especially on the
   contextual channels — then the finding is methodological and positive: the
   severity axis of a detection envelope must be defined in the metric the
   detector scores in. If it does not, the transfer failure is a property of the
   envelope rather than of the severity parameterisation, and is reported as
   such.

INPUTS   NASA SMAP/MSL dataset; the repo's synthetic pools and per-channel surface
OUTPUT   ../results/wp2_severity_metric_comparison.csv
================================================================================
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from wp2_validity import (ALPHAS, CONTEXTUAL, DURATIONS, FP_GATE, N_CAL_WINDOWS,
                          SEED, WINDOW, bilinear_surface, build_synth_segment,
                          invert_calibration, load_channel,
                          longest_nominal_segment, parse_windows, windowise)

HERE = Path(__file__).resolve().parent
from _paths import REPO  # noqa: E402
OUT = HERE.parent / "results"
from _paths import DATA_ROOT, LABELS  # noqa: E402

AR_ORDER = 8


# --------------------------------------------------------------------------
def fit_ar(sig: np.ndarray, p: int = AR_ORDER):
    """Least-squares AR(p) on a nominal signal. Returns (coeffs, residual RMS)."""
    if len(sig) < p + 20:
        return None, None
    X = np.stack([sig[i:len(sig) - p + i] for i in range(p)], axis=1)
    y = sig[p:]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    rms = float(np.sqrt(np.mean(resid ** 2)))
    return coef, (rms if rms > 1e-9 else 1e-9)


def ar_severity(win: np.ndarray, coef: np.ndarray, base_rms: float,
                p: int = AR_ORDER) -> np.ndarray:
    """Normalised one-step-ahead AR residual RMS, per row of `win`."""
    X = np.stack([win[:, i:win.shape[1] - p + i] for i in range(p)], axis=2)
    y = win[:, p:]
    pred = np.einsum("ntk,k->nt", X, coef)
    return np.sqrt(np.mean((y - pred) ** 2, axis=1)) / base_rms


# --------------------------------------------------------------------------
def _canonical() -> bool:
    return os.environ.get("ENVELOPE_SOURCE", "canonical").lower() in (
        "canonical", "wp8", "c")


def _sfx() -> str:
    return "_runC" if _canonical() else ""


def main() -> None:
    rng = np.random.default_rng(SEED)
    labels = pd.read_csv(LABELS).drop_duplicates("chan_id", keep="first")
    if _canonical():
        can = pd.read_csv(OUT / "wp8_canonical_envelope.csv")
        per = can[(can.detector == "lstm") & (can.fault_model.isin(["learned", "none"]))
                  ].rename(columns={"f1_pa": "f1"})
        print("envelope source: run C (wp8_canonical_envelope.csv)")
    else:
        per = pd.read_csv(REPO / "results" / "phase4sens_per_channel.csv")
        print("envelope source: run A (phase4sens_per_channel.csv)")
    master = pd.read_csv(REPO / "results" / "phase4_master_comparison.csv")
    pool_pt = np.load(REPO / "data" / "synthetic" / "gan_v4_clean_synth_point.npy")
    pool_cx = np.load(REPO / "data" / "synthetic" / "gan_v4_clean_synth_contextual.npy")

    rows = []
    for ch in sorted(per.chan_id.unique()):
        lab = labels[labels.chan_id == ch]
        if lab.empty:
            continue
        ac = str(lab["class"].iloc[0])
        seqs = parse_windows(lab["anomaly_sequences"].iloc[0])
        _, test = load_channel(ch)
        a0, b0 = longest_nominal_segment(len(test), seqs)
        clean = test[a0:b0]
        if len(clean) < WINDOW + 10:
            continue

        coef, base_rms = fit_ar(clean)
        if coef is None:
            continue

        has_cx, has_pt = "contextual" in ac, "point" in ac
        pool = (np.vstack([pool_pt, pool_cx]) if (has_cx and has_pt)
                else pool_cx if has_cx else pool_pt)

        # ---- predictive calibration curve T_ar(alpha) ----
        starts = rng.integers(0, len(clean) - WINDOW, size=N_CAL_WINDOWS)
        curve = []
        for a in ALPHAS:
            wins = np.empty((N_CAL_WINDOWS, WINDOW))
            for j, st in enumerate(starts):
                nom = clean[st:st + WINDOW]
                syn = build_synth_segment(pool, WINDOW, rng)
                wins[j] = nom * (1.0 - a) + syn * a
            curve.append(float(np.median(ar_severity(wins, coef, base_rms))))
        curve = np.array(curve)
        monotone = bool(np.all(np.diff(curve) > 0))

        sub = per[(per.chan_id == ch) & (per.duration > 0)]
        fp = per[(per.chan_id == ch) & (per.duration == 0)]["fp_rate_no_injection"]
        fp = float(fp.iloc[0]) if len(fp) else np.nan

        preds, weights, alphas_hat = [], [], []
        for s, e in seqs:
            s, e = max(0, s), min(len(test), e)
            d_real = e - s
            if d_real < 8:
                continue
            if d_real >= WINDOW:
                wins = windowise(test[s:e])
            else:
                c = (s + e) // 2
                lo = max(0, min(len(test) - WINDOW, c - WINDOW // 2))
                wins = test[lo:lo + WINDOW][None, :]
            t_real = float(np.median(ar_severity(wins, coef, base_rms)))
            a_hat, _ = invert_calibration(curve, t_real)
            pred, _ = bilinear_surface(sub, a_hat, d_real)
            preds.append(pred)
            weights.append(d_real)
            alphas_hat.append(a_hat)

        if not preds:
            continue
        rows.append(dict(
            chan_id=ch, cls="contextual" if ch in CONTEXTUAL else "point",
            fp_rate=fp, admissible=bool(np.isfinite(fp) and fp <= FP_GATE),
            calibration_monotone=monotone,
            T_ar_alpha_min=curve[0], T_ar_alpha_max=curve[-1],
            alpha_hat_ar=float(np.average(alphas_hat, weights=weights)),
            predicted_f1_ar=float(np.average(preds, weights=weights))))

    ar = pd.DataFrame(rows)
    dist = pd.read_csv(OUT / f"wp2_predictive_validity{_sfx()}.csv")[
        ["chan_id", "alpha_hat_mean", "predicted_f1", "observed_f1"]].rename(
        columns={"alpha_hat_mean": "alpha_hat_dist", "predicted_f1": "predicted_f1_dist"})
    comp = ar.merge(dist, on="chan_id", how="left")
    comp["abs_err_dist"] = (comp.predicted_f1_dist - comp.observed_f1).abs()
    comp["abs_err_ar"] = (comp.predicted_f1_ar - comp.observed_f1).abs()
    comp.to_csv(OUT / f"wp2_severity_metric_comparison{_sfx()}.csv", index=False)

    print("PREDICTIVE (AR) SEVERITY CALIBRATION")
    print(f"  channels calibrated : {len(ar)}")
    print(f"  monotone T_ar(alpha): {int(ar.calibration_monotone.sum())}/{len(ar)}")
    print(f"  T_ar at alpha=0.10  : median {ar.T_ar_alpha_min.median():.2f}  "
          f"(1.0 = as predictable as nominal)")
    print(f"  T_ar at alpha=1.50  : median {ar.T_ar_alpha_max.median():.2f}\n")

    def report(df, name):
        for tag, col in (("distributional", "predicted_f1_dist"),
                         ("predictive(AR)", "predicted_f1_ar")):
            d = df.dropna(subset=[col, "observed_f1"])
            if len(d) < 4:
                continue
            rho, p_s = spearmanr(d[col], d.observed_f1)
            mae = (d[col] - d.observed_f1).abs().mean()
            bias = float(np.mean(d[col] - d.observed_f1))
            print(f"  {name:<24} {tag:<16} n={len(d):>2}  "
                  f"rho={rho:+.3f} (p={p_s:.3f})  MAE={mae:.3f}  bias={bias:+.3f}")
        print()

    print("HEAD-TO-HEAD — does defining severity in the detector's metric help?")
    report(comp, "all channels")
    report(comp[comp.cls == "point"], "point channels")
    report(comp[comp.cls == "contextual"], "contextual channels")
    report(comp[comp.admissible], "admissible channels")

    print("PER-CHANNEL")
    show = comp.sort_values("observed_f1", ascending=False)[
        ["chan_id", "cls", "alpha_hat_dist", "alpha_hat_ar",
         "predicted_f1_dist", "predicted_f1_ar", "observed_f1",
         "abs_err_dist", "abs_err_ar"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwritten -> {OUT / 'wp2_severity_metric_comparison.csv'}")


if __name__ == "__main__":
    main()
