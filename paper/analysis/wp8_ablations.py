"""
================================================================================
WP8 — Ablations closing the remaining objections to the validity result
================================================================================
Paper:  Detectability of learned spacecraft anomaly detectors: minimum-
        detectable-fault envelopes by controlled fault injection
Author: Mehrab Jamshidi — Politecnico di Milano

THREE OBJECTIONS THIS ANSWERS

  A1  "THE ENVELOPE IS DOING NO WORK." The predictive severity statistic T_ar
      measures how much less predictable a window is than nominal, and the LSTM
      scores on one-step-ahead prediction error. A sceptic can therefore argue
      that the agreement in Section 5.11 is a correlation between two measures of
      the same thing, and that the envelope -- the surface the severity estimate
      is read through -- contributes nothing. The existing circularity control
      (wp7_circularity.py) ablates the CHANNEL and the ENVELOPE SHAPE but never
      ablates the envelope itself, so it does not answer this.

      The missing ablation is to use alpha_hat DIRECTLY as the predictor, with no
      envelope at all. If alpha_hat alone ranks real-fault detection as well as
      the envelope does, the envelope is redundant.

  A2  "THE CONTEXTUAL RESULT AT n=6 IS ONE CHANNEL." Leave-one-out is reported in
      the paper for the overall rho but not for the contextual subset, which is
      the result carrying the claim and the one with the smallest sample.

  A3  "IT IS THE DURATION AXIS IN DISGUISE." The six contextual channels differ
      in real-anomaly duration as well as in estimated severity, so duration
      alone is a competing explanation and is tested here directly.

ALSO RECORDED: the censoring rates. Half of the severity estimates under the
predictive statistic sit on a grid boundary, and the paper should say so rather
than let a referee discover it.

NOTE ON A PREDICTOR THAT IS *NOT* USED HERE. An earlier draft of this analysis
used the column T_ar_alpha_min as "the raw severity statistic of the real
anomaly". It is not that: it is curve[0], the value of the channel's CALIBRATION
curve at alpha = 0.10, and therefore a channel property rather than a property of
the anomaly. It is excluded.

INPUTS   ../results/wp7_circularity.csv
         ../results/wp2_severity_metric_comparison.csv
         ../results/wp2_validity_per_anomaly.csv
OUTPUT   ../results/wp8_validity_ablations.csv
================================================================================
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"


def main() -> None:
    import os
    sfx = ("_runC" if os.environ.get("ENVELOPE_SOURCE", "canonical").lower()
           in ("canonical", "wp8", "c") else "")
    circ = pd.read_csv(RES / f"wp7_circularity{sfx}.csv")
    sev = pd.read_csv(RES / f"wp2_severity_metric_comparison{sfx}.csv")
    anom = pd.read_csv(RES / f"wp2_validity_per_anomaly{sfx}.csv")

    df = circ.merge(
        sev[["chan_id", "alpha_hat_ar", "predicted_f1_ar",
             "alpha_hat_dist", "predicted_f1_dist"]],
        on="chan_id", how="left")
    dur = anom.groupby("chan_id").d_real.max().rename("d_real_max")
    df = df.merge(dur, on="chan_id", how="left")

    obs = df["observed"].to_numpy(float)

    # ---------------------------------------------------------- A1: the ladder
    ladder = [
        ("ENVELOPE  channel x operating point", df["specific"]),
        ("ALPHA-HAT severity estimate alone, no envelope", df["alpha_hat_ar"]),
        ("SHAPE     pooled envelope, no channel", df["gridmean"]),
        ("CHANNEL   channel difficulty alone", df["channel"]),
    ]
    rows = []
    print("=" * 78)
    print("A1 — DOES THE ENVELOPE DO ANY WORK? (all 20 held-out channels)")
    print("=" * 78)
    for name, v in ladder:
        v = v.to_numpy(float)
        rho, p = spearmanr(v, obs)
        mae = (float(np.mean(np.abs(v - obs)))
               if name.startswith(("ENVELOPE", "SHAPE", "CHANNEL")) else np.nan)
        print(f"  {name:<50} rho = {rho:+.3f}   p = {p:.4f}"
              + (f"   MAE = {mae:.3f}" if np.isfinite(mae) else ""))
        rows.append(dict(test="ladder_all", predictor=name.split()[0],
                         n=len(obs), rho=rho, p=p, mae=mae))

    ctx = df[df.cls == "contextual"]
    obs_c = ctx["observed"].to_numpy(float)
    print("\n  restricted to the 6 contextual channels")
    for name, col in (("ENVELOPE", "specific"), ("ALPHA-HAT", "alpha_hat_ar"),
                      ("SHAPE", "gridmean"), ("CHANNEL", "channel")):
        rho, p = spearmanr(ctx[col].to_numpy(float), obs_c)
        print(f"  {name:<50} rho = {rho:+.3f}   p = {p:.4f}")
        rows.append(dict(test="ladder_contextual", predictor=name,
                         n=len(obs_c), rho=rho, p=p, mae=np.nan))

    # ------------------------------------------------ A2: contextual leave-one-out
    print("\n" + "=" * 78)
    print("A2 — LEAVE-ONE-OUT ON THE 6 CONTEXTUAL CHANNELS")
    print("=" * 78)
    pred_c = ctx["specific"].to_numpy(float)
    ids = ctx["chan_id"].tolist()
    loo = []
    for i, cid in enumerate(ids):
        keep = [j for j in range(len(ids)) if j != i]
        rho, p = spearmanr(pred_c[keep], obs_c[keep])
        loo.append(rho)
        print(f"  drop {cid:<6} rho = {rho:+.3f}   p = {p:.4f}")
        rows.append(dict(test="contextual_loo", predictor=f"drop_{cid}",
                         n=len(keep), rho=rho, p=p, mae=np.nan))
    print(f"  range: {min(loo):+.3f} to {max(loo):+.3f}   "
          f"(full sample {spearmanr(pred_c, obs_c)[0]:+.3f})")

    # ------------------------------------------------------ A3: duration alone
    print("\n" + "=" * 78)
    print("A3 — IS IT THE DURATION AXIS IN DISGUISE?")
    print("=" * 78)
    for lbl, sub in (("all 20", df), ("contextual 6", ctx)):
        rho, p = spearmanr(sub["d_real_max"].to_numpy(float),
                           sub["observed"].to_numpy(float))
        print(f"  real-anomaly duration alone, {lbl:<14} rho = {rho:+.3f}   p = {p:.4f}")
        rows.append(dict(test="duration_alone", predictor=lbl, n=len(sub),
                         rho=rho, p=p, mae=np.nan))

    # --------------------------------------------------------- censoring rates
    print("\n" + "=" * 78)
    print("CENSORING OF THE SEVERITY ESTIMATE (reported, not hidden)")
    print("=" * 78)
    for lbl, col in (("predictive  T_ar", "alpha_hat_ar"),
                     ("distributional  ", "alpha_hat_dist")):
        v = sev[col].to_numpy(float)
        lo = int((v <= 0.1001).sum())
        hi = int((v >= 1.4999).sum())
        print(f"  {lbl}: {lo}/20 at the 0.10 floor, {hi}/20 at the 1.50 ceiling,"
              f" {20 - lo - hi}/20 interior")
        rows.append(dict(test="censoring", predictor=col, n=20,
                         rho=np.nan, p=np.nan, mae=np.nan,
                         n_floor=lo, n_ceiling=hi, n_interior=20 - lo - hi))
    clamped = int((anom.predict_status == "duration_clamped").sum())
    print(f"  duration clamped to the tested grid: {clamped}/{len(anom)} anomalies"
          f"  (grid max 256 steps; real durations reach"
          f" {int(anom.d_real.max())})")
    rows.append(dict(test="censoring", predictor="duration_clamped",
                     n=len(anom), rho=np.nan, p=np.nan, mae=np.nan,
                     n_floor=clamped))

    out = pd.DataFrame(rows)
    out.to_csv(RES / f"wp8_validity_ablations{sfx}.csv", index=False)
    print(f"\nwrote {RES / f'wp8_validity_ablations{sfx}.csv'}")


if __name__ == "__main__":
    main()
