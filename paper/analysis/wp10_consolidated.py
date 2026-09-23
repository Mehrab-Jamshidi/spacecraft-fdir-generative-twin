"""
================================================================================
WP10 — Consolidated envelope analysis on the single canonical execution
================================================================================
Paper:  Detectability of learned spacecraft anomaly detectors: minimum-
        detectable-fault envelopes by controlled fault injection
Author: Mehrab Jamshidi — Politecnico di Milano

Every envelope-side number in the revised manuscript comes from ONE execution,
the deterministic canonical campaign (wp9_canonical_envelope.csv, which
reproduces wp8_canonical_envelope.csv). Runs A and B are used only as
replicates, to measure run-to-run variability in the units of the artefact.

What this script adds to wp8_analyse.py

  1  PROBABILITY OF DETECTION WITHIN A DEADLINE, WITH ITS SAMPLING INTERVAL.
     P_L(alpha, d) = mean over channels of n_timely_L / n_seg, bootstrapped over
     channels, for L in {5, 10, 20}; and the whole-segment hit rate n_det/n_seg.

  2  DEADLINE-CONSTRAINED CONTOURS AT REQUIREMENTS THAT CAN BIND.
     The earlier draft tested P_L >= 0.5, which the faintest tested fault
     already meets (0.51 at L=5), so it could not bind. A POD requirement in
     practice is 0.9. Contours are computed for P_L >= 0.7, 0.8, 0.9, alone and
     jointly with F1 >= 0.50, on the mean surface and on the bootstrap bounds.

  3  THE FALSE-CALL BASELINE. The chance that a window of the deadline's length
     contains an alarm on un-injected nominal data (wp9_chance_windows.csv), so
     that a probability of detection is never read without its false-call rate.

  4  ADMISSIBLE-ONLY SENSITIVITY of the headline contour.

  5  REPRODUCIBILITY IN ALPHA UNITS across runs A, B, C, set against the
     bootstrap interval of C.

  6  PREVALENCE of injected faults in the test streams, and of real faults in
     the real test splits: F1 depends on it, POD does not.

OUTPUTS  ../results/wp10_pod_bootstrap.csv
         ../results/wp10_deadline_contours.csv
         ../results/wp10_numbers.json          (every number quoted by the text)
================================================================================
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from wp2_envelope import first_upcrossing

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
from _paths import REPO  # noqa: E402

ALPHAS = [0.10, 0.25, 0.50, 0.75, 1.00, 1.50]
DURATIONS = [16, 32, 64, 128, 256]
BUDGETS = (5, 10, 20)
N_BOOT = 10_000
SEED = 42
FP_GATE = 0.05
NUM: dict = {}


def cls_of(ac: str) -> str:
    return ("contextual" if ("contextual" in str(ac) and "point" not in str(ac))
            else "point")


def fmt(a, s):
    if s == "right_censored":
        return ">1.50"
    if s == "left_censored":
        return "<=0.10"
    return f"{a:.3f}"


def canonical() -> pd.DataFrame:
    p9 = RES / "wp9_canonical_envelope.csv"
    p8 = RES / "wp8_canonical_envelope.csv"
    df = pd.read_csv(p9 if p9.exists() else p8)
    df["cls"] = df.anomaly_class.map(cls_of)
    return df


def boot_cells(ref: pd.DataFrame, col: str) -> pd.DataFrame:
    """Channel-level percentile bootstrap of the per-cell mean of `col`."""
    rng = np.random.default_rng(SEED)
    rows = []
    for d in DURATIONS:
        for a in ALPHAS:
            v = ref[(ref.duration == d) & np.isclose(ref.alpha, a)][col].to_numpy(float)
            v = v[np.isfinite(v)]
            if not len(v):
                continue
            m = v[rng.integers(0, len(v), size=(N_BOOT, len(v)))].mean(axis=1)
            rows.append(dict(alpha=a, duration=d, n=len(v), mean=v.mean(),
                             lo=np.percentile(m, 2.5), hi=np.percentile(m, 97.5),
                             sd=v.std(ddof=1)))
    return pd.DataFrame(rows)


def contour(b: pd.DataFrame, col: str, target: float):
    out = {}
    for d in DURATIONS:
        g = b[b.duration == d].sort_values("alpha")
        if g.empty:
            continue
        out[d] = first_upcrossing(g.alpha.tolist(), g[col].to_numpy(float),
                                  target, rising=True)
    return out


def joint(c1: dict, c2: dict) -> dict:
    out = {}
    for d in c1:
        (a1, s1), (a2, s2) = c1[d], c2[d]
        if "right_censored" in (s1, s2):
            out[d] = (None, "right_censored")
        elif s1 == "left_censored" and s2 == "left_censored":
            out[d] = (0.10, "left_censored")
        else:
            out[d] = (max(a1 if a1 is not None else 0.10,
                          a2 if a2 is not None else 0.10), "ok")
    return out


def main():
    df = canonical()
    lstm = df[(df.detector == "lstm")]
    ref = lstm[(lstm.fault_model == "learned") & (lstm.duration > 0)].copy()
    ctrl = lstm[lstm.fault_model == "none"].set_index("chan_id")
    fp = ctrl.fp_rate_no_injection
    adm = set(fp[fp <= FP_GATE].index)
    NUM["n_channels"] = int(ref.chan_id.nunique())
    NUM["fp_median"] = float(fp.median())
    NUM["fp_mean"] = float(fp.mean())
    NUM["n_admissible"] = len(adm)
    NUM["inadmissible"] = {k: float(v) for k, v in fp[fp > FP_GATE].sort_values(
        ascending=False).items()}
    NUM["n_channels_d256"] = int(ref[ref.duration == 256].chan_id.nunique())

    for L in BUDGETS:
        ref[f"p{L}"] = ref[f"n_timely_{L}"] / ref.n_seg.clip(lower=1)
    ref["pdet"] = ref.n_det / ref.n_seg.clip(lower=1)

    # ------------------------------------------------ 1  F1 and POD surfaces
    bF = boot_cells(ref, "f1_pa")
    NUM["f1_corner_lo"] = float(bF[(bF.alpha == 0.10) & (bF.duration == 16)]["mean"].iloc[0])
    NUM["f1_corner_hi"] = float(bF[(bF.alpha == 1.50) & (bF.duration == 256)]["mean"].iloc[0])
    NUM["f1_op_a1_d64"] = float(bF[(bF.alpha == 1.00) & (bF.duration == 64)]["mean"].iloc[0])
    NUM["f1_cell_sd_min"] = float(bF.sd.min())
    NUM["f1_cell_sd_max"] = float(bF.sd.max())
    NUM["f1_ci_halfwidth_mean"] = float(((bF.hi - bF.lo) / 2).mean())
    NUM["f1_grid_mean"] = float(bF["mean"].mean())
    mono = all(np.all(np.diff(bF[bF.duration == d].sort_values("alpha")["mean"].to_numpy()) > 0)
               for d in DURATIONS)
    NUM["f1_monotone_in_alpha_every_d"] = bool(mono)
    mono_d = all(np.all(np.diff(bF[np.isclose(bF.alpha, a)].sort_values("duration")["mean"].to_numpy()) > 0)
                 for a in ALPHAS)
    NUM["f1_monotone_in_d_every_alpha"] = bool(mono_d)

    pods = {}
    rows = []
    for col in ("pdet", "p5", "p10", "p20"):
        b = boot_cells(ref, col)
        b["quantity"] = col
        pods[col] = b
        rows.append(b)
    pd.concat(rows).to_csv(RES / "wp10_pod_bootstrap.csv", index=False)

    print("=" * 78)
    print("PROBABILITY OF DETECTION WITHIN A DEADLINE (mean over 20 channels)")
    print("=" * 78)
    for col in ("p5", "p20", "pdet"):
        b = pods[col]
        print(f"\n  {col}")
        print("   alpha " + "".join(f"{d:>9}" for d in DURATIONS))
        for a in ALPHAS:
            print(f"   {a:5.2f} " + "".join(
                f"{float(b[(b.duration == d) & np.isclose(b.alpha, a)]['mean'].iloc[0]):9.3f}"
                for d in DURATIONS))
        # movement across each axis
        dur_span = [float(b[np.isclose(b.alpha, a)]["mean"].max()
                          - b[np.isclose(b.alpha, a)]["mean"].min()) for a in ALPHAS]
        sev_gain = [float(b[(b.duration == d) & np.isclose(b.alpha, 1.5)]["mean"].iloc[0]
                          - b[(b.duration == d) & np.isclose(b.alpha, 0.1)]["mean"].iloc[0])
                    for d in DURATIONS]
        print(f"   max range across duration at fixed alpha: {max(dur_span):.3f}"
              f"   severity gain 0.10->1.50: {min(sev_gain):.3f} to {max(sev_gain):.3f}")
        NUM[f"{col}_max_duration_range"] = max(dur_span)
        NUM[f"{col}_sev_gain_min"] = min(sev_gain)
        NUM[f"{col}_sev_gain_max"] = max(sev_gain)
        NUM[f"{col}_grid"] = {f"{a}_{d}": float(b[(b.duration == d) & np.isclose(b.alpha, a)]["mean"].iloc[0])
                              for a in ALPHAS for d in DURATIONS}
        NUM[f"{col}_max"] = float(b["mean"].max())

    # conditional mean latency, for the three-sentence contrast in the text
    lat = ref.groupby(["alpha", "duration"]).latency.mean()
    NUM["latency_a01"] = {int(d): float(lat.loc[(0.1, d)]) for d in DURATIONS}
    NUM["latency_a15"] = {int(d): float(lat.loc[(1.5, d)]) for d in DURATIONS}
    print("\n  conditional mean latency at alpha=0.10:",
          {k: round(v, 2) for k, v in NUM["latency_a01"].items()})

    # ------------------------------------------------ 2  contours
    print("\n" + "=" * 78)
    print("CONTOURS: quality (F1), timeliness (P_L), joint")
    print("=" * 78)
    crow = []
    cF = {s: contour(bF, s, 0.50) for s in ("mean", "lo", "hi")}
    for s in ("mean", "lo", "hi"):
        NUM[f"mdf_f1_050_{s}"] = {d: fmt(*cF[s][d]) for d in DURATIONS}
    for t in (0.70, 0.80):
        for s in ("mean", "lo", "hi"):
            NUM[f"mdf_f1_{int(t*100):03d}_{s}"] = {d: fmt(*contour(bF, s, t)[d]) for d in DURATIONS}
    print("  F1>=0.50  mean:", NUM["mdf_f1_050_mean"])
    print("            cons:", NUM["mdf_f1_050_lo"])
    print("            opt :", NUM["mdf_f1_050_hi"])
    for L in BUDGETS:
        b = pods[f"p{L}"]
        for P in (0.7, 0.8, 0.9):
            for s in ("mean", "lo"):
                cP = contour(b, s, P)
                cJ = joint(cF[s], cP)
                for d in DURATIONS:
                    crow.append(dict(L=L, P_req=P, surface=s, duration=d,
                                     timeliness=fmt(*cP[d]), joint=fmt(*cJ[d])))
                NUM[f"mdf_P{L}_{int(P*100)}_{s}"] = {d: fmt(*cP[d]) for d in DURATIONS}
                NUM[f"mdf_joint_F050_P{L}_{int(P*100)}_{s}"] = {d: fmt(*cJ[d]) for d in DURATIONS}
    cdf = pd.DataFrame(crow)
    cdf.to_csv(RES / "wp10_deadline_contours.csv", index=False)
    for L in (5, 20):
        for P in (0.7, 0.8, 0.9):
            print(f"  P_{L}>={P:.1f}  timeliness mean: {NUM[f'mdf_P{L}_{int(P*100)}_mean']}")
            print(f"            joint w/ F1>=0.5 : {NUM[f'mdf_joint_F050_P{L}_{int(P*100)}_mean']}")
            print(f"            timeliness cons: {NUM[f'mdf_P{L}_{int(P*100)}_lo']}")

    # ------------------------------------------------ 3  false-call baseline
    ch = pd.read_csv(RES / "wp9_chance_windows.csv")
    chl = ch[ch.detector == "lstm"]
    print("\n" + "=" * 78)
    print("FALSE-CALL BASELINE: P(window of w steps on nominal data holds an alarm)")
    print("=" * 78)
    for w in (6, 11, 21, 16, 64, 256):
        v = chl[f"chance_w{w}"]
        va = chl[chl.chan_id.isin(adm)][f"chance_w{w}"]
        print(f"  w={w:3d}: mean {v.mean():.3f}  median {v.median():.3f}   "
              f"admissible mean {va.mean():.3f}")
        NUM[f"chance_w{w}_mean"] = float(v.mean())
        NUM[f"chance_w{w}_median"] = float(v.median())
        NUM[f"chance_w{w}_adm_mean"] = float(va.mean())

    # POD over admissible channels only, at the faintest and strongest severity
    ra = ref[ref.chan_id.isin(adm)]
    for L in (5, 20):
        g = ra.groupby(["alpha", "duration"])[f"p{L}"].mean()
        NUM[f"p{L}_adm_a01"] = {int(d): float(g.loc[(0.1, d)]) for d in DURATIONS}
        NUM[f"p{L}_adm_a15"] = {int(d): float(g.loc[(1.5, d)]) for d in DURATIONS}
        print(f"  admissible P_{L}: alpha=0.10 {NUM[f'p{L}_adm_a01']}")
        print(f"                  alpha=1.50 {NUM[f'p{L}_adm_a15']}")

    # ------------------------------------------------ 4  admissible-only F1 contour
    bA = boot_cells(ra, "f1_pa")
    NUM["mdf_f1_050_adm_mean"] = {d: fmt(*contour(bA, "mean", 0.50)[d]) for d in DURATIONS}
    NUM["mdf_f1_050_adm_lo"] = {d: fmt(*contour(bA, "lo", 0.50)[d]) for d in DURATIONS}
    NUM["f1_grid_mean_adm"] = float(bA["mean"].mean())
    print("\n  admissible-only F1>=0.50 contour, mean:", NUM["mdf_f1_050_adm_mean"],
          " cons:", NUM["mdf_f1_050_adm_lo"])

    # ------------------------------------------------ 5  reproducibility, alpha units
    runs = {"C": ref.groupby(["alpha", "duration"]).f1_pa.mean()}
    b = pd.read_csv(RES / "wp3_envelope_extended.csv")
    runs["B"] = b[(b.detector == "lstm") & (b.duration > 0)].groupby(["alpha", "duration"]).f1_pa.mean()
    a = pd.read_csv(REPO / "results" / "phase4sens_per_channel.csv")
    runs["A"] = a[a.duration > 0].groupby(["alpha", "duration"]).f1.mean()
    print("\n" + "=" * 78)
    print("REPRODUCIBILITY ACROSS THREE EXECUTIONS (identical injected streams)")
    print("=" * 78)
    for other in ("A", "B"):
        dlt = (runs[other] - runs["C"]).dropna()
        NUM[f"surface_{other}_minus_C_meanabs"] = float(dlt.abs().mean())
        NUM[f"surface_{other}_minus_C_max"] = float(dlt.abs().max())
        NUM[f"surface_{other}_minus_C_signed"] = float(dlt.mean())
        NUM[f"surface_{other}_higher_cells"] = int((dlt > 0).sum())
        print(f"  {other} - C: mean|d| {dlt.abs().mean():.3f}  max {dlt.abs().max():.3f}"
              f"  signed {dlt.mean():+.3f}  cells higher {(dlt > 0).sum()}/30")
    rep = {}
    for r, s in runs.items():
        vals = {d: first_upcrossing(ALPHAS, np.array([s.get((al, d), np.nan) for al in ALPHAS]),
                                    0.50, rising=True) for d in DURATIONS}
        rep[r] = {d: fmt(*vals[d]) for d in DURATIONS}
        print(f"  run {r}: F1>=0.50 contour {rep[r]}")
    NUM["repro_contour_f1_050"] = rep
    for d in (32, 64):
        xs = [float(rep[r][d]) for r in ("A", "B", "C")
              if rep[r][d] not in (">1.50", "<=0.10")]
        lo = cF["hi"][d]
        hi = cF["lo"][d]
        NUM[f"repro_spread_d{d}"] = max(xs) - min(xs)
        NUM[f"repro_interval_d{d}"] = [fmt(*lo), fmt(*hi)]
        print(f"  d={d}: run spread {max(xs) - min(xs):.3f}; bootstrap interval "
              f"[{fmt(*lo)}, {fmt(*hi)}]")

    # per-channel instability, same streams, different detector draw
    pc = {}
    for r, frame, col in (("A", a[a.duration > 0], "f1"),
                          ("B", b[(b.detector == "lstm") & (b.duration > 0)], "f1_pa"),
                          ("C", ref, "f1_pa")):
        pc[r] = frame.set_index(["chan_id", "alpha", "duration"])[col]
    worst = 0.0
    for x, y in (("A", "B"), ("A", "C"), ("B", "C")):
        dd = (pc[x] - pc[y]).dropna().abs()
        worst = max(worst, float(dd.max()))
    NUM["per_channel_cell_max_move"] = worst
    fpa = a[a.duration == 0].set_index("chan_id").fp_rate_no_injection
    NUM["fp_T2_runA"] = float(fpa.get("T-2", np.nan))
    NUM["fp_T2_runC"] = float(fp.get("T-2", np.nan))
    print(f"  largest single channel-cell move between runs: {worst:.3f}")
    print(f"  T-2 nominal false-alarm rate: run A {NUM['fp_T2_runA']:.3f}, run C {NUM['fp_T2_runC']:.3f}")

    # ------------------------------------------------ 6  prevalence
    cl = chl.set_index("chan_id").clean_len
    prev = {d: (3 * d / cl).describe()[["min", "50%", "max"]].to_dict() for d in DURATIONS}
    NUM["injected_prevalence"] = {int(d): {k: float(v) for k, v in p.items()} for d, p in prev.items()}
    seg = pd.read_csv(RES / "wp9_real_segments.csv")
    seg = seg[seg.detector == "lstm"]
    real = pd.read_csv(RES / "wp9_real_detection.csv")
    NUM["n_real_anomalies"] = int(len(seg))
    NUM["real_segments_detected_lstm"] = int(seg.detected.sum())
    NUM["real_duration_max"] = int(seg.duration.max())
    NUM["real_longer_than_grid"] = int((seg.duration > 256).sum())
    print("\n  injected-fault prevalence (3 d / clean length), median by d:",
          {d: round(p['50%'], 4) for d, p in prev.items()})
    print(f"  real anomalies: {len(seg)}; detected (any alarm in segment) "
          f"{int(seg.detected.sum())}; longer than 256 steps {NUM['real_longer_than_grid']}")

    rl = real[real.detector == "lstm"]
    NUM["real_f1_lstm_mean"] = float(rl.f1_pa.mean())
    NUM["real_f1_lstm_median"] = float(rl.f1_pa.median())
    rv = real[real.detector == "vae"]
    NUM["real_f1_vae_mean"] = float(rv.f1_pa.mean())

    # ------------------------------------------------ class contours
    for c in ("point", "contextual"):
        bc = boot_cells(ref[ref.cls == c], "f1_pa")
        NUM[f"mdf_f1_070_{c}_mean"] = {d: fmt(*contour(bc, "mean", 0.70)[d]) for d in DURATIONS}
        g = bc.set_index(["alpha", "duration"])["mean"]
        NUM[f"f1_{c}_d256_a01"] = float(g.loc[(0.1, 256)])
        NUM[f"f1_{c}_d256_a15"] = float(g.loc[(1.5, 256)])
        NUM[f"n_{c}"] = int(ref[ref.cls == c].chan_id.nunique())
        print(f"  {c:<10} F1>=0.70 contour {NUM[f'mdf_f1_070_{c}_mean']}   "
              f"d=256 sweep {g.loc[(0.1, 256)]:.3f} -> {g.loc[(1.5, 256)]:.3f}")

    # ------------------------------------------------ point-wise LSTM contour
    bW = boot_cells(ref, "f1_pw")
    NUM["mdf_pw_f1_030_mean"] = {d: fmt(*contour(bW, "mean", 0.30)[d]) for d in DURATIONS}
    print("  point-wise F1>=0.30 contour:", NUM["mdf_pw_f1_030_mean"])

    # ------------------------------------------------ d = 16 cross-section
    g = ref.groupby(["alpha", "duration"])[["p_pa", "r_pa"]].mean()
    NUM["d16_recall"] = [float(g.loc[(0.1, 16)].r_pa), float(g.loc[(1.5, 16)].r_pa)]
    NUM["d16_precision"] = [float(g.loc[(0.1, 16)].p_pa), float(g.loc[(1.5, 16)].p_pa)]

    # ------------------------------------------------ detectors at d = 16
    vae = df[(df.detector == "vae") & (df.fault_model == "learned") & (df.duration > 0)]
    gv = vae.groupby(["alpha", "duration"]).f1_pa.mean()
    gl = ref.groupby(["alpha", "duration"]).f1_pa.mean()
    NUM["vae_d16_range"] = [float(gv.xs(16, level="duration").min()),
                            float(gv.xs(16, level="duration").max())]
    NUM["lstm_d16_range"] = [float(gl.xs(16, level="duration").min()),
                             float(gl.xs(16, level="duration").max())]
    diff = (gl - gv).dropna()
    NUM["lstm_minus_vae_mean"] = float(diff.mean())
    NUM["lstm_beats_vae_cells"] = int((diff > 0).sum())
    NUM["vae_grid_mean"] = float(gv.mean())
    NUM["vae_fp_median"] = float(df[(df.detector == "vae") & (df.fault_model == "none")]
                                 .fp_rate_no_injection.median())

    (RES / "wp10_numbers.json").write_text(json.dumps(NUM, indent=1, default=str))
    print(f"\nwrote wp10_numbers.json ({len(NUM)} entries)")


if __name__ == "__main__":
    main()
