"""
================================================================================
WP2 D/E — Per-channel design sheets, an admissibility gate, and the
           predictive-validity test of the envelope against REAL faults
================================================================================
Paper:  Detectability of learned spacecraft anomaly detectors: minimum-
        detectable-fault envelopes by controlled fault injection
Author: Mehrab Jamshidi — Politecnico di Milano

THE QUESTION THIS SETTLES
   The thesis states, as its fifth limitation, that the sensitivity surface
   "characterises detectability of faults produced by the generator and
   parameterised by the injection model, not of arbitrary physical faults;
   extrapolating it to real graded faults rests on the realism of the generator
   and remains to be validated against real fault data."

   That is the single largest opening a reviewer has. This script closes it as
   far as the data allows: it places each REAL labelled anomaly of each held-out
   channel on the same (severity, duration) axes the synthetic sweep uses, reads
   the predicted detection off that channel's own envelope, and compares the
   prediction against what the detector actually scored on that real anomaly.

HOW A REAL ANOMALY IS PLACED ON THE SEVERITY AXIS
   Duration is read directly from the ground-truth labels — no estimation.

   Severity is the hard part, and it is estimated by CALIBRATION rather than
   assumption. The injection model is

       x_inj = (1-a)*x_nom + a*(3*g),      g ~ generator output in tanh space

   so alpha is not a physical unit; it is a blend coefficient. To invert it for a
   real fault we need a statistic that moves with alpha and can also be measured
   on a real anomaly. We use the displacement of a window, in the SIX-FEATURE
   space the thesis already commits to (mean, sd, min, max, range, lag-1
   autocorrelation) — the same representation used by the supervised classifier
   and by the Frechet fidelity metric. Crucially it includes lag-1
   autocorrelation, so it responds to contextual faults, which change temporal
   structure without changing amplitude. A purely amplitude-based statistic
   would score every contextual anomaly at zero severity and the whole test
   would be vacuous.

   Per channel:
     1. Window the channel's longest clean nominal segment -> reference mean/sd
        per feature.
     2. For each alpha on the grid, synthesise injected windows exactly as the
        sweep does, and measure T(alpha) = median standardised distance of the
        injected windows from the nominal reference.  -> a calibration curve.
     3. Window each real labelled anomaly segment, measure T_real the same way.
     4. alpha_hat = invert T(alpha) at T_real by first upcrossing.

   The curve must be increasing in alpha for the inversion to mean anything;
   monotonicity is CHECKED per channel and reported, not assumed.

WHAT IS REPORTED, AND THE INTERPRETATION FIXED IN ADVANCE
   Spearman rho (does the envelope RANK real detectability?), Pearson r, and MAE
   (does it predict the LEVEL?). These are different claims and are reported
   separately. The outcome is written up whichever way it falls: a strong rho
   supports the envelope as a relative design instrument; a weak one bounds it to
   a characterisation of generator-produced faults. n = 20 channels, so this is
   evidence about ranking, not a calibrated predictive model.

INPUTS   NASA SMAP/MSL dataset (SMAP_MSL_ROOT or --data)
         <thesis repository>/data/synthetic/gan_v4_clean_synth_*.npy
         <thesis repository>/results/phase4sens_per_channel.csv
         <thesis repository>/results/phase4_master_comparison.csv
OUTPUTS  ../results/wp2_channel_design_sheet.csv
         ../results/wp2_severity_calibration.csv
         ../results/wp2_predictive_validity.csv
================================================================================
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
from _paths import REPO  # noqa: E402
OUT = HERE.parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

from _paths import DATA_ROOT, LABELS  # noqa: E402

WINDOW = 64
STEP = 16
ALPHAS = [0.10, 0.25, 0.50, 0.75, 1.00, 1.50]
DURATIONS = [16, 32, 64, 128, 256]
SEED = 42
N_CAL_WINDOWS = 400          # injected windows per alpha, per channel
CONTEXTUAL = {"A-8", "E-10", "F-3", "G-1", "M-2", "P-1"}
FP_GATE = 0.05               # admissibility: a channel whose un-injected clean
                             # stream is flagged above this rate cannot support
                             # a meaningful detection claim.


# --------------------------------------------------------------------------
# Data plumbing — mirrors phase4_sensitivity.py exactly so the severity scale
# is the same one the published surface was measured on.
# --------------------------------------------------------------------------
def parse_windows(s):
    n = [int(x) for x in re.findall(r"\d+", str(s))]
    return [[n[i], n[i + 1]] for i in range(0, len(n) - 1, 2)]


def load_channel(chan: str):
    tr = np.load(DATA_ROOT / "train" / f"{chan}.npy")[:, 0]
    te = np.load(DATA_ROOT / "test" / f"{chan}.npy")[:, 0]
    mu, sd = tr.mean(), tr.std()
    if sd < 1e-9:
        sd = 1.0
    return (tr - mu) / sd, (te - mu) / sd


def longest_nominal_segment(n: int, anom_seqs):
    mask = np.ones(n, dtype=bool)
    for s, e in anom_seqs:
        mask[max(0, s):min(n, e)] = False
    best = (0, 0)
    a = None
    for i in range(n):
        if mask[i]:
            if a is None:
                a = i
        elif a is not None:
            if i - a > best[1] - best[0]:
                best = (a, i)
            a = None
    if a is not None and n - a > best[1] - best[0]:
        best = (a, n)
    return best


# --------------------------------------------------------------------------
# The six-feature representation (thesis Appendix A.1)
# --------------------------------------------------------------------------
def features(win: np.ndarray) -> np.ndarray:
    """mean, sd, min, max, range, lag-1 autocorrelation — per row of `win`."""
    m = win.mean(axis=1)
    s = win.std(axis=1)
    lo = win.min(axis=1)
    hi = win.max(axis=1)
    rng = hi - lo
    a = win[:, :-1] - m[:, None]
    b = win[:, 1:] - m[:, None]
    den = (a * a).sum(axis=1)
    ac = np.divide((a * b).sum(axis=1), den, out=np.zeros_like(den), where=den > 1e-12)
    return np.column_stack([m, s, lo, hi, rng, ac])


def windowise(sig: np.ndarray, window=WINDOW, step=STEP) -> np.ndarray:
    if len(sig) < window:
        return np.empty((0, window))
    return np.stack([sig[i:i + window] for i in range(0, len(sig) - window + 1, step)])


def standardised_distance(F: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Diagonal-covariance Mahalanobis distance from the nominal reference."""
    return np.linalg.norm((F - mu) / sd, axis=1)


# --------------------------------------------------------------------------
def build_synth_segment(pool: np.ndarray, duration: int, rng) -> np.ndarray:
    """Identical construction to inject(): tanh space x3, concatenated if the
    requested duration exceeds the 64-step generator window."""
    scaled = pool * 3.0
    if duration <= scaled.shape[1]:
        return scaled[rng.integers(len(scaled)), :duration].copy()
    k = (duration + scaled.shape[1] - 1) // scaled.shape[1]
    idx = rng.integers(0, len(scaled), size=k)
    return np.concatenate([scaled[i] for i in idx])[:duration]


def calibrate_severity(clean: np.ndarray, pool: np.ndarray, mu, sd, rng):
    """T(alpha): median standardised feature displacement of injected windows."""
    curve = []
    starts = rng.integers(0, len(clean) - WINDOW, size=N_CAL_WINDOWS)
    for a in ALPHAS:
        wins = np.empty((N_CAL_WINDOWS, WINDOW))
        for j, st in enumerate(starts):
            nom = clean[st:st + WINDOW]
            syn = build_synth_segment(pool, WINDOW, rng)
            wins[j] = nom * (1.0 - a) + syn * a
        curve.append(float(np.median(standardised_distance(features(wins), mu, sd))))
    return np.array(curve)


def invert_calibration(curve: np.ndarray, t_real: float):
    """alpha_hat by first upcrossing of the calibration curve at t_real."""
    if t_real <= curve[0]:
        return ALPHAS[0], "left_censored"
    for i in range(1, len(ALPHAS)):
        if t_real <= curve[i]:
            x0, x1 = ALPHAS[i - 1], ALPHAS[i]
            y0, y1 = curve[i - 1], curve[i]
            if y1 == y0:
                return x1, "ok"
            return x0 + (t_real - y0) * (x1 - x0) / (y1 - y0), "ok"
    return ALPHAS[-1], "right_censored"


def bilinear_surface(sub: pd.DataFrame, a: float, d: float):
    """Read a channel's own (alpha,duration) F1 surface at an arbitrary point.
    Grid-edge values are held constant outside the tested range; every such
    clamp is flagged so it can be excluded in a sensitivity check."""
    piv = sub.pivot_table(index="alpha", columns="duration", values="f1")
    A = [x for x in ALPHAS if x in piv.index]
    D = [x for x in DURATIONS if x in piv.columns]
    if not A or not D:
        return np.nan, "no_surface"

    flags = []
    ac = min(max(a, A[0]), A[-1])
    dc = min(max(d, D[0]), D[-1])
    if ac != a:
        flags.append("alpha_clamped")
    if dc != d:
        flags.append("duration_clamped")

    def bracket(v, grid):
        if v <= grid[0]:
            return 0, 0, 0.0
        if v >= grid[-1]:
            return len(grid) - 1, len(grid) - 1, 0.0
        i = int(np.searchsorted(grid, v)) - 1
        return i, i + 1, (v - grid[i]) / (grid[i + 1] - grid[i])

    i0, i1, ta = bracket(ac, A)
    j0, j1, td = bracket(dc, D)
    v = (piv.iloc[i0, j0] * (1 - ta) * (1 - td) + piv.iloc[i1, j0] * ta * (1 - td)
         + piv.iloc[i0, j1] * (1 - ta) * td + piv.iloc[i1, j1] * ta * td)
    return float(v), ("ok" if not flags else "+".join(flags))


# --------------------------------------------------------------------------
def load_envelope() -> pd.DataFrame:
    """The per-channel envelope the real anomalies are placed on.

    Default (ENVELOPE_SOURCE=canonical) is the canonical execution, whose
    detector instances also produced the real-fault observations in
    wp9_real_detection.csv, so prediction and observation come from the same
    trained network. ENVELOPE_SOURCE=thesis reproduces an earlier draft that used
    run A (the thesis campaign) and a separately trained Phase-2 detector for the
    observation; it is kept only for traceability.

    Both are reduced to the same four columns, so nothing downstream changes.
    """
    src = os.environ.get("ENVELOPE_SOURCE", "canonical").lower()
    if src in ("canonical", "wp8", "c"):
        df = pd.read_csv(OUT / "wp8_canonical_envelope.csv")
        df = df[(df.detector == "lstm")
                & (df.fault_model.isin(["learned", "none"]))].copy()
        df = df.rename(columns={"f1_pa": "f1"})
        print("envelope source: run C (wp8_canonical_envelope.csv)")
        return df[["chan_id", "alpha", "duration", "f1",
                   "fp_rate_no_injection"]]
    print("envelope source: run A (phase4sens_per_channel.csv)")
    return pd.read_csv(REPO / "results" / "phase4sens_per_channel.csv")


def main() -> None:
    if not LABELS.exists():
        sys.exit(f"labels not found: {LABELS}\nSet SMAP_MSL_ROOT.")

    rng = np.random.default_rng(SEED)
    labels = pd.read_csv(LABELS).drop_duplicates("chan_id", keep="first")
    per = load_envelope()
    master = pd.read_csv(REPO / "results" / "phase4_master_comparison.csv")
    pool_pt = np.load(REPO / "data" / "synthetic" / "gan_v4_clean_synth_point.npy")
    pool_cx = np.load(REPO / "data" / "synthetic" / "gan_v4_clean_synth_contextual.npy")

    channels = sorted(per["chan_id"].unique())
    print(f"Held-out channels: {len(channels)}")
    print(f"Synthetic pools   : point {pool_pt.shape}, contextual {pool_cx.shape}\n")

    cal_rows, val_rows, sheet_rows = [], [], []

    for ch in channels:
        lab = labels[labels.chan_id == ch]
        if lab.empty:
            print(f"  {ch}: no label row, skipped")
            continue
        ac = str(lab["class"].iloc[0])
        seqs = parse_windows(lab["anomaly_sequences"].iloc[0])
        _, test = load_channel(ch)

        a0, b0 = longest_nominal_segment(len(test), seqs)
        clean = test[a0:b0]
        if len(clean) < WINDOW + 10:
            print(f"  {ch}: clean segment too short ({len(clean)}), skipped")
            continue

        # Nominal reference distribution in feature space.
        F_nom = features(windowise(clean))
        mu = F_nom.mean(axis=0)
        sd = F_nom.std(axis=0)
        sd[sd < 1e-9] = 1.0

        # The channel's synthetic pool — same rule as the sweep.
        has_cx, has_pt = "contextual" in ac, "point" in ac
        pool = (np.vstack([pool_pt, pool_cx]) if (has_cx and has_pt)
                else pool_cx if has_cx else pool_pt)

        curve = calibrate_severity(clean, pool, mu, sd, rng)
        monotone = bool(np.all(np.diff(curve) > 0))
        cal_rows.append(dict(chan_id=ch, monotone=monotone,
                             **{f"T_alpha_{a}": c for a, c in zip(ALPHAS, curve)}))

        sub = per[(per.chan_id == ch) & (per.duration > 0)]
        fp = per[(per.chan_id == ch) & (per.duration == 0)]["fp_rate_no_injection"]
        fp = float(fp.iloc[0]) if len(fp) else np.nan

        # ---- place each real anomaly on the axes ----
        for k, (s, e) in enumerate(seqs):
            s, e = max(0, s), min(len(test), e)
            d_real = e - s
            if d_real < 8:
                continue
            seg = test[s:e]
            wins = windowise(seg) if len(seg) >= WINDOW else seg[None, :]
            if len(seg) < WINDOW:                       # centre a 64-window on it
                c = (s + e) // 2
                lo = max(0, min(len(test) - WINDOW, c - WINDOW // 2))
                wins = test[lo:lo + WINDOW][None, :]
            t_real = float(np.median(standardised_distance(features(wins), mu, sd)))
            a_hat, a_status = invert_calibration(curve, t_real)
            pred, pred_status = bilinear_surface(sub, a_hat, d_real)

            val_rows.append(dict(
                chan_id=ch, anomaly_idx=k,
                cls="contextual" if ch in CONTEXTUAL else "point",
                d_real=d_real, T_real=t_real, alpha_hat=a_hat,
                alpha_status=a_status, predicted_f1=pred,
                predict_status=pred_status, fp_rate=fp,
                calibration_monotone=monotone))

        sheet_rows.append(dict(chan_id=ch,
                               cls="contextual" if ch in CONTEXTUAL else "point",
                               fp_rate_no_injection=fp,
                               admissible=bool(np.isfinite(fp) and fp <= FP_GATE),
                               n_real_anomalies=len(seqs),
                               calibration_monotone=monotone))

    cal = pd.DataFrame(cal_rows)
    val = pd.DataFrame(val_rows)
    sheet = pd.DataFrame(sheet_rows)

    # ---- per-channel aggregation, duration-weighted ----
    # The observation must come from the SAME trained detector as the envelope
    # it is compared against. In canonical mode it does: wp9 scored each
    # channel's real test split with the per-channel instance that produced the
    # canonical envelope. The legacy (run A) path compares against the thesis
    # Phase-2 LSTM, a different training draw, and is kept only so the earlier
    # draft's numbers remain reproducible.
    if os.environ.get("ENVELOPE_SOURCE", "canonical").lower() in ("canonical", "wp8", "c"):
        real = pd.read_csv(OUT / "wp9_real_detection.csv")
        obs = real[real.detector == "lstm"][["chan_id", "f1_pa"]].rename(
            columns={"f1_pa": "observed_f1"})
        print("observed real-fault F1: same detector instance (wp9_real_detection.csv)")
    else:
        obs = master[["chan_id", "p2_lstm_k3_f1"]].rename(
            columns={"p2_lstm_k3_f1": "observed_f1"})
    g = (val.groupby("chan_id")
            .apply(lambda x: pd.Series({
                "predicted_f1": np.average(x.predicted_f1, weights=x.d_real),
                "alpha_hat_mean": np.average(x.alpha_hat, weights=x.d_real),
                "d_real_max": x.d_real.max(),
                "cls": x.cls.iloc[0],
                "fp_rate": x.fp_rate.iloc[0]}), include_groups=False)
            .reset_index())
    comp = g.merge(obs, on="chan_id", how="left")
    comp["admissible"] = comp.fp_rate <= FP_GATE
    comp["abs_err"] = (comp.predicted_f1 - comp.observed_f1).abs()

    sfx = ("_runC"
           if os.environ.get("ENVELOPE_SOURCE", "canonical").lower()
           in ("canonical", "wp8", "c") else "")
    cal.to_csv(OUT / f"wp2_severity_calibration{sfx}.csv", index=False)
    comp.to_csv(OUT / f"wp2_predictive_validity{sfx}.csv", index=False)
    sheet.to_csv(OUT / f"wp2_channel_design_sheet{sfx}.csv", index=False)
    # Per-anomaly frame: the robustness subsets in the paper are defined by the
    # censoring and clamping flags, which live here rather than in the
    # per-channel aggregate. Saved so the claims audit can recompute them.
    val.to_csv(OUT / f"wp2_validity_per_anomaly{sfx}.csv", index=False)

    # ---------------- report ----------------
    print("SEVERITY CALIBRATION")
    print(f"  channels calibrated      : {len(cal)}")
    print(f"  monotone T(alpha)        : {int(cal.monotone.sum())}/{len(cal)}")
    print(f"  median T at alpha=0.10   : {cal['T_alpha_0.1'].median():.2f}")
    print(f"  median T at alpha=1.50   : {cal['T_alpha_1.5'].median():.2f}\n")

    print("ADMISSIBILITY GATE (no-injection false-positive rate <= "
          f"{FP_GATE:.2f})")
    bad = sheet[~sheet.admissible]
    print(f"  admissible channels      : {int(sheet.admissible.sum())}/{len(sheet)}")
    print("  excluded                 : "
          + ", ".join(f"{r.chan_id} (FP={r.fp_rate_no_injection:.3f})"
                      for r in bad.itertuples()) + "\n")

    print("REAL ANOMALIES PLACED ON THE ENVELOPE")
    print(f"  anomalies placed         : {len(val)}")
    print(f"  duration range           : {val.d_real.min()} - {val.d_real.max()} steps")
    print(f"  alpha_hat range          : {val.alpha_hat.min():.2f} - {val.alpha_hat.max():.2f}")
    print("  severity censoring       : "
          + ", ".join(f"{k}={v}" for k, v in val.alpha_status.value_counts().items()))
    print("  surface clamping         : "
          + ", ".join(f"{k}={v}" for k, v in val.predict_status.value_counts().items()) + "\n")

    def report(df, name):
        d = df.dropna(subset=["predicted_f1", "observed_f1"])
        if len(d) < 4:
            print(f"  {name}: n={len(d)} — too few to test")
            return
        rho, p_s = spearmanr(d.predicted_f1, d.observed_f1)
        r, p_p = pearsonr(d.predicted_f1, d.observed_f1)
        print(f"  {name:<28} n={len(d):>2}  "
              f"Spearman rho={rho:+.3f} (p={p_s:.3f})  "
              f"Pearson r={r:+.3f} (p={p_p:.3f})  "
              f"MAE={d.abs_err.mean():.3f}  bias={np.mean(d.predicted_f1 - d.observed_f1):+.3f}")

    print("PREDICTIVE VALIDITY — envelope prediction vs observed real-fault detection")
    report(comp, "all held-out channels")
    report(comp[comp.admissible], "admissible channels only")
    report(comp[comp.cls == "point"], "point channels")
    report(comp[comp.cls == "contextual"], "contextual channels")
    print()

    # ---- robustness: what is actually carrying the correlation? ----
    # A single headline rho would be misleading here. Two channels (D-5, R-1)
    # flag their entire clean stream and are correctly predicted near zero, so
    # they are "easy" agreements that inflate rho; and 9 of 25 real anomalies are
    # longer than the longest tested duration, so their prediction is a
    # grid-edge extrapolation. Both are isolated rather than buried.
    pathological = {"D-5", "R-1"}
    within_grid = set(val[val.predict_status == "ok"].chan_id)
    uncensored = set(val[val.alpha_status == "ok"].chan_id)

    print("ROBUSTNESS OF THE CORRELATION — which observations carry it")
    report(comp[~comp.chan_id.isin(pathological)], "excl. D-5, R-1 (FP=1.0)")
    report(comp[comp.chan_id.isin(within_grid)], "excl. duration-extrapolated")
    report(comp[comp.chan_id.isin(uncensored)], "excl. severity-censored")
    report(comp[comp.chan_id.isin(within_grid & uncensored)
                & ~comp.chan_id.isin(pathological)], "strictest subset")
    print()

    print("PER-CHANNEL DETAIL")
    show = comp.sort_values("observed_f1", ascending=False)[
        ["chan_id", "cls", "admissible", "alpha_hat_mean", "d_real_max",
         "predicted_f1", "observed_f1", "abs_err"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    for f in ("wp2_severity_calibration.csv", "wp2_predictive_validity.csv",
              "wp2_channel_design_sheet.csv"):
        print(f"written -> {OUT / f}")


if __name__ == "__main__":
    main()
