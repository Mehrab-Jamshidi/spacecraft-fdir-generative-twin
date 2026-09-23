"""
================================================================================
WP7 analysis — does a LEARNED fault model beat hand-specified parametric faults?
================================================================================
Paper:  Detectability of learned spacecraft anomaly detectors: minimum-
        detectable-fault envelopes by controlled fault injection
Author: Mehrab Jamshidi — Politecnico di Milano

CONSUMES  ../results/wp7_parametric_envelope.csv  (from wp7_parametric.py)
PRODUCES  ../results/wp7_parametric_comparison.csv

Q1  DO THE ENVELOPES DIFFER?
    If a step, a ramp, a spike train and a noise burst produce the same contour
    as the generator, the generator is an expensive way to get a simple answer.

Q2  WHICH ENVELOPE PREDICTS THE REAL FAULTS BETTER?
    This is the decisive test, and the one that justifies the whole approach.
    The generative envelope's claim to attention is that its stimulus resembles
    the faults the spacecraft actually suffers. That is testable: place each real
    labelled anomaly on each candidate envelope -- each with its OWN severity
    calibration, so no fault model is handicapped -- and compare rank agreement
    with what the detector actually scored.

    Severity is inverted with the predictive (AR-residual) statistic throughout,
    since Section 5.8 establishes that as the correct parameterisation. Using
    the distributional statistic for one model and the predictive one for the
    other would rig the comparison.

    If the parametric envelope predicts real faults as well as the generative
    one, the paper's novelty claim against the fault-injection literature must be
    weakened, and this script says so.
================================================================================
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from wp2_severity_metric import ar_severity, fit_ar
from wp2_validity import (ALPHAS, CONTEXTUAL, DURATIONS, N_CAL_WINDOWS, SEED,
                          WINDOW, bilinear_surface, invert_calibration,
                          load_channel, longest_nominal_segment, parse_windows,
                          windowise)
from wp7_parametric import FAMILIES, parametric_segment

HERE = Path(__file__).resolve().parent
from _paths import REPO  # noqa: E402
RES = HERE.parent / "results"
from _paths import DATA_ROOT, LABELS  # noqa: E402
AR_ORDER = 8


def first_upcross(vals, target):
    if vals[0] >= target:
        return ALPHAS[0], "left"
    for i in range(1, len(ALPHAS)):
        if vals[i] >= target:
            y0, y1 = vals[i - 1], vals[i]
            if y1 == y0:
                return ALPHAS[i], "ok"
            return ALPHAS[i - 1] + (target - y0) * (ALPHAS[i] - ALPHAS[i - 1]) / (y1 - y0), "ok"
    return None, "right"


def contour_str(piv, target):
    out = []
    for d in DURATIONS:
        if d not in piv.columns:
            out.append("--")
            continue
        a, st = first_upcross(piv[d].reindex(ALPHAS).to_numpy(), target)
        out.append(">1.50" if st == "right" else ("<=0.10" if st == "left" else f"{a:.3f}"))
    return out


def main():
    if os.environ.get("ENVELOPE_SOURCE", "canonical").lower() in ("canonical", "wp8", "c"):
        # One campaign, one detector training per channel, all five fault models
        # scored on it. This removes the detector-training confound that the
        # separate wp7 / wp3 runs carried: there the four parametric families
        # were scored on a freshly trained detector and the learned family on a
        # different one, so part of any spread between them was attributable to
        # the detector draw rather than to the fault model.
        can = pd.read_csv(RES / "wp8_canonical_envelope.csv")
        can = can[(can.detector == "lstm") & (can.duration > 0)]
        par = can[can.fault_model != "learned"].rename(
            columns={"fault_model": "family"})
        gen = can[can.fault_model == "learned"]
        print("envelope source: run C (wp8_canonical_envelope.csv)")
    else:
        par = pd.read_csv(RES / "wp7_parametric_envelope.csv")
        gen = pd.read_csv(RES / "wp3_envelope_extended.csv")
        gen = gen[(gen.detector == "lstm") & (gen.duration > 0)]
        print("envelope source: runs B+C (wp3 + wp7)")
    # The learned stimulus's prediction and the observed detection must come
    # from the same execution as the parametric envelopes, or the comparison
    # re-introduces the detector-draw confound the canonical run exists to
    # remove. (An earlier version read the run-A file here unconditionally.)
    canonical = os.environ.get("ENVELOPE_SOURCE", "canonical").lower() in ("canonical", "wp8", "c")
    sev = pd.read_csv(RES / ("wp2_severity_metric_comparison_runC.csv" if canonical
                             else "wp2_severity_metric_comparison.csv"))
    labels = pd.read_csv(LABELS).drop_duplicates("chan_id", keep="first")

    # ------------------------------------------------ Q1: contours
    print("=" * 78)
    print("Q1  DO THE ENVELOPES DIFFER?   (mean F1, point-adjust, 20 held-out channels)")
    print("=" * 78)
    gp = gen.groupby(["alpha", "duration"]).f1_pa.mean().unstack()
    print("\n  generative")
    print(gp.round(3).to_string())
    print(f"    contour F1>=0.50 : {'  '.join(contour_str(gp, 0.50))}")

    rows = []
    for fam in FAMILIES:
        pp = (par[par.family == fam].groupby(["alpha", "duration"])
              .f1_pa.mean().unstack())
        print(f"\n  {fam}")
        print(pp.round(3).to_string())
        print(f"    contour F1>=0.50 : {'  '.join(contour_str(pp, 0.50))}")
        common = gp.reindex(index=pp.index, columns=pp.columns)
        rows.append(dict(family=fam, grid_mean=float(pp.values.mean()),
                         gen_minus_par=float((common - pp).values.mean())))
    print("\n  grid-mean F1")
    print(f"    generative {gp.values.mean():.3f}")
    for r in rows:
        print(f"    {r['family']:<10} {r['grid_mean']:.3f}   "
              f"(generative - {r['family']} = {r['gen_minus_par']:+.3f})")

    # ------------------------------------------------ Q2: real-fault prediction
    print("\n" + "=" * 78)
    print("Q2  WHICH ENVELOPE PREDICTS THE REAL FAULTS BETTER?")
    print("=" * 78)
    rng = np.random.default_rng(SEED)
    pred_rows = []

    for ch in sorted(par.chan_id.unique()):
        lab = labels[labels.chan_id == ch]
        if lab.empty:
            continue
        seqs = parse_windows(lab["anomaly_sequences"].iloc[0])
        _, test = load_channel(ch)
        a0, b0 = longest_nominal_segment(len(test), seqs)
        clean = test[a0:b0]
        if len(clean) < WINDOW + 30:
            continue
        coef, base_rms = fit_ar(clean, AR_ORDER)
        if coef is None:
            continue

        starts = rng.integers(0, len(clean) - WINDOW, size=N_CAL_WINDOWS)

        # real anomalies, measured once in the predictive statistic
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
            reals.append((float(np.median(ar_severity(wins, coef, base_rms, AR_ORDER))),
                          e - s))
        if not reals:
            continue

        rec = dict(chan_id=ch, cls="contextual" if ch in CONTEXTUAL else "point",
                   observed=float(sev[sev.chan_id == ch].observed_f1.iloc[0]))

        # each fault model gets its OWN calibration curve, so none is handicapped
        for fam in FAMILIES:
            curve = []
            for a in ALPHAS:
                wins = np.empty((N_CAL_WINDOWS, WINDOW))
                for j, stt in enumerate(starts):
                    seg = parametric_segment(fam, WINDOW, rng)
                    wins[j] = clean[stt:stt + WINDOW] * (1.0 - a) + seg * a
                curve.append(float(np.median(ar_severity(wins, coef, base_rms, AR_ORDER))))
            curve = np.array(curve)
            sub = par[(par.chan_id == ch) & (par.family == fam) & (par.duration > 0)]
            sub = sub.rename(columns={"f1_pa": "f1"})
            preds, ws = [], []
            for t_real, d_real in reals:
                a_hat, _ = invert_calibration(curve, t_real)
                p, _ = bilinear_surface(sub, a_hat, d_real)
                preds.append(p)
                ws.append(d_real)
            rec[f"pred_{fam}"] = float(np.average(preds, weights=ws))

        rec["pred_generative"] = float(sev[sev.chan_id == ch].predicted_f1_ar.iloc[0])
        pred_rows.append(rec)

    df = pd.DataFrame(pred_rows).dropna()
    sfx = ("_runC"
           if os.environ.get("ENVELOPE_SOURCE", "canonical").lower()
           in ("canonical", "wp8", "c") else "")
    df.to_csv(RES / f"wp7_parametric_comparison{sfx}.csv", index=False)

    print(f"\n  channels: {len(df)}\n")
    print(f"  {'fault model':<16}{'rho':>9}{'p':>9}{'MAE':>9}{'bias':>9}")
    res = {}
    for name in ["generative"] + list(FAMILIES):
        col = f"pred_{name}"
        if col not in df:
            continue
        rho, p = spearmanr(df[col], df.observed)
        mae = (df[col] - df.observed).abs().mean()
        bias = float((df[col] - df.observed).mean())
        res[name] = (rho, mae)
        star = " <--" if name == "generative" else ""
        print(f"  {name:<16}{rho:>+9.3f}{p:>9.3f}{mae:>9.3f}{bias:>+9.3f}{star}")

    best_par = max((v[0] for k, v in res.items() if k != "generative"), default=np.nan)
    gen_rho = res.get("generative", (np.nan,))[0]
    print()
    if gen_rho > best_par + 0.05:
        v = ("the learned fault model predicts real faults better than any "
             "parametric family tested")
    elif abs(gen_rho - best_par) <= 0.05:
        v = ("WEAKEN THE CLAIM: a parametric family predicts real faults as well "
             "as the generator")
    else:
        v = ("WITHDRAW THE CLAIM: a parametric family predicts real faults BETTER "
             "than the generator")
    print(f"  VERDICT: {v}")

    print("\n  by class:")
    for c in ("point", "contextual"):
        s = df[df.cls == c]
        if len(s) < 4:
            continue
        line = []
        for name in ["generative"] + list(FAMILIES):
            col = f"pred_{name}"
            if col in s:
                line.append(f"{name} {spearmanr(s[col], s.observed).statistic:+.2f}")
        print(f"    {c:<12} n={len(s):>2}  " + "   ".join(line))

    print(f"\nwritten -> {RES / 'wp7_parametric_comparison.csv'}")


if __name__ == "__main__":
    main()
