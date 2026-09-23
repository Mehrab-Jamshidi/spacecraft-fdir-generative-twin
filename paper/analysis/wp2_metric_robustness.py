"""
================================================================================
WP2 E4 — Robustness of the severity-metric finding
================================================================================
Paper:  Detectability of learned spacecraft anomaly detectors: minimum-
        detectable-fault envelopes by controlled fault injection
Author: Mehrab Jamshidi — Politecnico di Milano

THE CLAIM BEING STRESS-TESTED
   wp2_severity_metric.py found that re-parameterising the envelope's severity
   axis from a DISTRIBUTIONAL statistic (six-feature displacement) to a
   PREDICTIVE one (normalised one-step-ahead AR residual, mirroring the
   detector's own scoring) raises the rank agreement between predicted and
   observed real-fault detection from rho = +0.52 to +0.78 overall, and from
   +0.12 to +0.93 on the contextual channels.

   With n = 20 channels that is a strong claim from a small sample, and it would
   be easy to arrive at by metric-shopping. Three specific threats are tested.

   T1  METRIC-SHOPPING. The predictive metric was chosen on mechanistic grounds
       — the detector scores one-step-ahead prediction error, so severity should
       be measured in that metric — and the hypothesis was written down before
       the comparison was run. That is an argument, not evidence. The evidence
       is whether the result is stable across the one free parameter the metric
       has: the AR order. If rho swings wildly with p, the finding is an
       artefact of a lucky choice.

   T2  ILL-POSED INVERSION. Only 13 of 20 channels have a strictly monotone
       T_ar(alpha) calibration curve, against 20 of 20 for the distributional
       statistic. Severity inversion on a non-monotone curve is not well defined,
       so the comparison is re-run on the monotone subset alone.

   T3  SAMPLE FRAGILITY. With n = 20, one channel can carry a correlation. The
       difference in Spearman rho between the two metrics is bootstrapped over
       channels, and the leave-one-out influence of each channel is reported.

OUTPUT   ../results/wp2_metric_robustness.csv
================================================================================
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from wp2_severity_metric import ar_severity, fit_ar
from wp2_validity import (ALPHAS, CONTEXTUAL, FP_GATE, N_CAL_WINDOWS, SEED, WINDOW,
                          bilinear_surface, build_synth_segment, invert_calibration,
                          load_channel, longest_nominal_segment, parse_windows,
                          windowise)

HERE = Path(__file__).resolve().parent
from _paths import REPO  # noqa: E402
OUT = HERE.parent / "results"
from _paths import DATA_ROOT, LABELS  # noqa: E402

AR_ORDERS = [2, 4, 8, 16, 32]
N_BOOT = 10_000


def run_for_order(p: int, labels, per, pool_pt, pool_cx) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
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
        if len(clean) < max(WINDOW, p) + 30:
            continue
        coef, base_rms = fit_ar(clean, p)
        if coef is None:
            continue

        has_cx, has_pt = "contextual" in ac, "point" in ac
        pool = (np.vstack([pool_pt, pool_cx]) if (has_cx and has_pt)
                else pool_cx if has_cx else pool_pt)

        starts = rng.integers(0, len(clean) - WINDOW, size=N_CAL_WINDOWS)
        curve = []
        for a in ALPHAS:
            wins = np.empty((N_CAL_WINDOWS, WINDOW))
            for j, st in enumerate(starts):
                wins[j] = (clean[st:st + WINDOW] * (1.0 - a)
                           + build_synth_segment(pool, WINDOW, rng) * a)
            curve.append(float(np.median(ar_severity(wins, coef, base_rms, p))))
        curve = np.array(curve)

        sub = per[(per.chan_id == ch) & (per.duration > 0)]
        fp = per[(per.chan_id == ch) & (per.duration == 0)]["fp_rate_no_injection"]
        fp = float(fp.iloc[0]) if len(fp) else np.nan

        preds, weights = [], []
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
            t_real = float(np.median(ar_severity(wins, coef, base_rms, p)))
            a_hat, _ = invert_calibration(curve, t_real)
            pred, _ = bilinear_surface(sub, a_hat, e - s)
            preds.append(pred)
            weights.append(e - s)
        if not preds:
            continue
        rows.append(dict(chan_id=ch, ar_order=p,
                         cls="contextual" if ch in CONTEXTUAL else "point",
                         admissible=bool(np.isfinite(fp) and fp <= FP_GATE),
                         monotone=bool(np.all(np.diff(curve) > 0)),
                         predicted=float(np.average(preds, weights=weights))))
    return pd.DataFrame(rows)


def _canonical() -> bool:
    return os.environ.get("ENVELOPE_SOURCE", "canonical").lower() in (
        "canonical", "wp8", "c")


def _sfx() -> str:
    return "_runC" if _canonical() else ""


def main() -> None:
    labels = pd.read_csv(LABELS).drop_duplicates("chan_id", keep="first")
    if _canonical():
        can = pd.read_csv(OUT / "wp8_canonical_envelope.csv")
        per = can[(can.detector == "lstm") & (can.fault_model.isin(["learned", "none"]))
                  ].rename(columns={"f1_pa": "f1"})
        print("envelope source: run C (wp8_canonical_envelope.csv)")
    else:
        per = pd.read_csv(REPO / "results" / "phase4sens_per_channel.csv")
        print("envelope source: run A (phase4sens_per_channel.csv)")
    pool_pt = np.load(REPO / "data" / "synthetic" / "gan_v4_clean_synth_point.npy")
    pool_cx = np.load(REPO / "data" / "synthetic" / "gan_v4_clean_synth_contextual.npy")

    base = pd.read_csv(OUT / f"wp2_severity_metric_comparison{_sfx()}.csv")
    obs = base[["chan_id", "observed_f1", "predicted_f1_dist"]]

    # ---- T1: AR-order sensitivity ----
    print("T1  AR-ORDER SENSITIVITY  (distributional baseline: rho=+0.518 all, "
          "+0.116 contextual)")
    all_orders = []
    for p in AR_ORDERS:
        df = run_for_order(p, labels, per, pool_pt, pool_cx).merge(obs, on="chan_id")
        all_orders.append(df)
        r_all = spearmanr(df.predicted, df.observed_f1)
        c = df[df.cls == "contextual"]
        r_cx = spearmanr(c.predicted, c.observed_f1)
        adm = df[df.admissible]
        r_ad = spearmanr(adm.predicted, adm.observed_f1)
        mae = (df.predicted - df.observed_f1).abs().mean()
        print(f"  p={p:>2}  monotone={int(df.monotone.sum()):>2}/{len(df)}  "
              f"all rho={r_all.statistic:+.3f} (p={r_all.pvalue:.3f})  "
              f"ctx rho={r_cx.statistic:+.3f} (p={r_cx.pvalue:.3f})  "
              f"adm rho={r_ad.statistic:+.3f} (p={r_ad.pvalue:.3f})  MAE={mae:.3f}")
    print()

    full = pd.concat(all_orders, ignore_index=True)
    full.to_csv(OUT / f"wp2_metric_robustness{_sfx()}.csv", index=False)

    # ---- T2: monotone-calibration subset (reference order p=8) ----
    d8 = all_orders[AR_ORDERS.index(8)]
    mono = d8[d8.monotone]
    print("T2  WELL-POSED INVERSION ONLY (monotone calibration curve, p=8)")
    for name, sub in (("predictive(AR)", mono.predicted),
                      ("distributional", mono.predicted_f1_dist)):
        r = spearmanr(sub, mono.observed_f1)
        mae = (sub - mono.observed_f1).abs().mean()
        print(f"  {name:<16} n={len(mono):>2}  rho={r.statistic:+.3f} "
              f"(p={r.pvalue:.3f})  MAE={mae:.3f}")
    print()

    # ---- T3: bootstrap the rho difference, and leave-one-out influence ----
    d = d8.dropna(subset=["predicted", "predicted_f1_dist", "observed_f1"])
    rng = np.random.default_rng(SEED)
    n = len(d)
    pa = d.predicted.to_numpy()
    pd_ = d.predicted_f1_dist.to_numpy()
    ob = d.observed_f1.to_numpy()

    diffs = np.empty(N_BOOT)
    for b in range(N_BOOT):
        i = rng.integers(0, n, n)
        if len(np.unique(ob[i])) < 3:
            diffs[b] = np.nan
            continue
        diffs[b] = (spearmanr(pa[i], ob[i]).statistic
                    - spearmanr(pd_[i], ob[i]).statistic)
    diffs = diffs[np.isfinite(diffs)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    obs_diff = spearmanr(pa, ob).statistic - spearmanr(pd_, ob).statistic
    print("T3  BOOTSTRAP OF THE RHO DIFFERENCE (predictive - distributional)")
    print(f"  observed difference : {obs_diff:+.3f}")
    print(f"  95% CI              : [{lo:+.3f}, {hi:+.3f}]")
    print(f"  P(difference > 0)   : {(diffs > 0).mean():.3f}\n")

    print("    leave-one-out influence on the predictive rho")
    base_rho = spearmanr(pa, ob).statistic
    infl = []
    for i, ch in enumerate(d.chan_id):
        m = np.ones(n, bool)
        m[i] = False
        infl.append((ch, spearmanr(pa[m], ob[m]).statistic - base_rho))
    infl.sort(key=lambda t: abs(t[1]), reverse=True)
    for ch, dd in infl[:5]:
        print(f"      drop {ch:<5} -> rho {base_rho + dd:+.3f}  (change {dd:+.3f})")
    print(f"\nwritten -> {OUT / 'wp2_metric_robustness.csv'}")


if __name__ == "__main__":
    main()
