"""
================================================================================
WP8 — Analysis of the canonical campaign
================================================================================
Paper:  Detectability of learned spacecraft anomaly detectors: minimum-
        detectable-fault envelopes by controlled fault injection
Author: Mehrab Jamshidi — Politecnico di Milano

Derives every envelope-side table in the paper from ONE campaign
(wp8_canonical_envelope.csv), replacing the three-run patchwork described in the
header of wp8_canonical_campaign.py.

WHAT IS NEW HERE, BEYOND RE-DERIVING THE OLD TABLES FROM ONE RUN

  1  THE CONTOUR IS REPORTED AS AN INTERVAL, NOT A POINT.
     The envelope surface is nearly flat in alpha near a requirement crossing,
     so a small uncertainty in F1 becomes a large uncertainty in alpha*. The
     previous draft reported reproducibility in F1 units (mean |diff| 0.020)
     while selling an artefact denominated in alpha, where the same runs differ
     by up to 0.25. Here the channel-level bootstrap is propagated THROUGH the
     inversion: alpha* is computed on the mean surface, on the CI lower bound
     and on the CI upper bound, giving a contour interval directly. The
     run-to-run spread across the three independent executions is then reported
     as an external check that the interval is honest rather than as the
     headline uncertainty.

  2  TIMELY-DETECTION PROBABILITY REPLACES CONDITIONAL MEAN LATENCY AS THE
     DESIGN QUANTITY.
     L(alpha,d) is a mean over injections the detector flagged, and a flagged
     injection's latency is bounded above by d. Both push the mean upward with
     duration for reasons that have nothing to do with the detector. The
     campaign now records, per cell, the number of fault segments presented and
     the number flagged within L steps of onset, so

         P_timely(alpha, d | L) = mean over channels of  n_timely_L / n_seg

     counts undetected segments as failures, conditions on nothing, and is not
     bounded by d. The latency-constrained contour is recomputed against it.

  3  THE AXIS DECOMPOSITION IS REPORTED WITH THE RECALL LEVELS THAT QUALIFY IT.
     The previous draft defended the d=16 floor as "a recall statement". At
     d=16 recall runs 0.56-0.90 while precision runs 0.22-0.39, so the floor is
     precision-bound, i.e. it is the MOST protocol-conditioned cell on the grid
     rather than the least. The decomposition table now carries recall so the
     reader can see this rather than take a claim about it.

  4  THE FAULT-MODEL COMPARISON IS NO LONGER CONFOUNDED BY DETECTOR TRAINING.
     All five fault models are now scored on the same per-channel detector
     instance, so the fault model is the only thing that varies.

INPUTS   ../results/wp8_canonical_envelope.csv        (canonical, run C)
         ../results/wp3_envelope_extended.csv         (run B, reproducibility)
         <thesis repository>/results/phase4sens_per_channel.csv
                                                      (run A, reproducibility)
OUTPUTS  ../results/wp8_cell_bootstrap.csv
         ../results/wp8_mdf_contours.csv
         ../results/wp8_timely_detection.csv
         ../results/wp8_axis_decomposition.csv
         ../results/wp8_protocol_comparison.csv
         ../results/wp8_detector_comparison.csv
         ../results/wp8_faultmodel_comparison.csv
         ../results/wp8_run_reproducibility.csv
================================================================================
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from wp2_envelope import first_upcrossing

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
from _paths import REPO  # noqa: E402

ALPHAS = [0.10, 0.25, 0.50, 0.75, 1.00, 1.50]
DURATIONS = [16, 32, 64, 128, 256]
F1_TARGETS = (0.50, 0.70, 0.80)
LAT_BUDGETS = (5, 10, 20)
N_BOOT = 10_000
SEED = 42

# The mixed-class channel C-1 is grouped with point throughout, as Section 4.3
# states, so that class-disaggregated results partition the held-out set.
def class_of(anomaly_class: str) -> str:
    return "contextual" if ("contextual" in str(anomaly_class)
                            and "point" not in str(anomaly_class)) else "point"


# ------------------------------------------------------------------ bootstrap
def bootstrap_cells(df: pd.DataFrame, value_col: str, group: str) -> pd.DataFrame:
    """Percentile bootstrap over CHANNELS (the unit of independence), per cell.

    Column names follow the legacy wp2_cell_bootstrap schema so that the figure
    code and the claims audit can be repointed at this file without change.
    """
    rng = np.random.default_rng(SEED)
    rows = []
    for d in DURATIONS:
        for a in ALPHAS:
            cell = df[(df.duration == d) & (np.isclose(df.alpha, a))]
            v = cell[value_col].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if len(v) == 0:
                continue
            idx = rng.integers(0, len(v), size=(N_BOOT, len(v)))
            means = v[idx].mean(axis=1)
            lat = cell["latency"].to_numpy(dtype=float)
            lat = lat[np.isfinite(lat)]
            rows.append(dict(
                group=group, alpha=a, duration=d, n_channels=len(v),
                f1_mean=float(v.mean()), f1_median=float(np.median(v)),
                f1_sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                f1_ci_lo=float(np.percentile(means, 2.5)),
                f1_ci_hi=float(np.percentile(means, 97.5)),
                latency_mean=float(lat.mean()) if len(lat) else np.nan))
    return pd.DataFrame(rows)


def surface(boot: pd.DataFrame, d: int, col: str):
    g = boot[boot.duration == d].sort_values("alpha")
    return g[col].to_numpy(dtype=float), g["alpha"].tolist()


def contour_interval(boot: pd.DataFrame, target: float) -> list[dict]:
    """alpha* on the mean surface and on both bootstrap bounds.

    A HIGHER F1 surface crosses the requirement at a LOWER severity, so the
    conservative (larger) alpha* comes from the CI LOWER bound and the
    optimistic (smaller) one from the CI upper bound. Reporting the triple is
    the point: it is the interval, not the point estimate, that a design margin
    should be read from.
    """
    out = []
    for d in DURATIONS:
        if (boot.duration == d).sum() == 0:
            continue
        rec = dict(duration=d, requirement=f"F1>={target:.2f}", f1_target=target,
                   latency_budget=np.nan)
        # Legacy column names (alpha_mdf / alpha_mdf_conservative) are kept so the
        # figure code and the audit read this file unchanged; the optimistic bound
        # is new and is appended rather than substituted.
        for name, col in (("alpha_mdf", "f1_mean"),
                          ("alpha_mdf_conservative", "f1_ci_lo"),
                          ("alpha_mdf_optimistic", "f1_ci_hi")):
            vals, agrid = surface(boot, d, col)
            a, s = first_upcrossing(agrid, vals, target, rising=True)
            rec[name] = a
            rec["status" if name == "alpha_mdf"
                else name.replace("alpha_mdf", "status")] = s
        out.append(rec)
    return out


def fmt(alpha, status) -> str:
    """The paper's printed convention for a censored contour."""
    if status == "right_censored":
        return ">1.50"
    if status == "left_censored":
        return "<=0.10"
    return f"{alpha:.3f}"


# ------------------------------------------------------------------ loading
def load_canonical() -> pd.DataFrame:
    df = pd.read_csv(RES / "wp8_canonical_envelope.csv")
    df["cls"] = df["anomaly_class"].map(class_of)
    return df


def lstm_learned(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df.detector == "lstm") & (df.fault_model == "learned")
              & (df.duration > 0)].copy()


# ------------------------------------------------------------------ sections
def do_envelope_and_contours(df, out):
    ref = lstm_learned(df)
    boots = {}
    for group, sub in (("all", ref),
                       ("point", ref[ref.cls == "point"]),
                       ("contextual", ref[ref.cls == "contextual"])):
        boots[group] = bootstrap_cells(sub, "f1_pa", group)
    boot_all = pd.concat(boots.values(), ignore_index=True)
    boot_all.to_csv(out / "wp8_cell_bootstrap.csv", index=False)

    rows = []
    for group, b in boots.items():
        for t in F1_TARGETS:
            for rec in contour_interval(b, t):
                rec["group"] = group
                rows.append(rec)
    con = pd.DataFrame(rows)
    con.to_csv(out / "wp8_mdf_contours.csv", index=False)

    print("\n" + "=" * 78)
    print("MDF CONTOUR, with the bootstrap propagated through the inversion")
    print("=" * 78)
    for t in F1_TARGETS:
        sub = con[(con.group == "all") & (con.f1_target == t)]
        print(f"\n  F1 >= {t:.2f}")
        print("     d     mean      conservative (CI lo)   optimistic (CI hi)")
        for _, r in sub.iterrows():
            print(f"   {int(r.duration):4d}   {fmt(r.alpha_mdf, r.status):>8}"
                  f"   {fmt(r.alpha_mdf_conservative, r.status_conservative):>12}"
                  f"        {fmt(r.alpha_mdf_optimistic, r.status_optimistic):>10}")
    return boots, con


def do_timely(df, out):
    """Censoring-free latency: P(alarm within L steps of onset), per channel,
    averaged over channels. Undetected segments count as failures."""
    ref = lstm_learned(df)
    ref = ref.copy()
    for L in LAT_BUDGETS:
        ref[f"p_timely_{L}"] = ref[f"n_timely_{L}"] / ref["n_seg"].clip(lower=1)
    ref["p_det"] = ref["n_det"] / ref["n_seg"].clip(lower=1)

    rows = []
    for d in DURATIONS:
        for a in ALPHAS:
            c = ref[(ref.duration == d) & (np.isclose(ref.alpha, a))]
            if c.empty:
                continue
            rec = dict(alpha=a, duration=d, n_channels=len(c),
                       latency_mean=float(np.nanmean(c["latency"])),
                       p_detected=float(c["p_det"].mean()))
            for L in LAT_BUDGETS:
                rec[f"p_timely_{L}"] = float(c[f"p_timely_{L}"].mean())
            rows.append(rec)
    tim = pd.DataFrame(rows)
    tim.to_csv(out / "wp8_timely_detection.csv", index=False)

    print("\n" + "=" * 78)
    print("TIMELY DETECTION  P(alarm within L steps of onset) — not conditioned")
    print("on detection, not bounded by d. Compare with conditional mean latency.")
    print("=" * 78)
    for L in LAT_BUDGETS:
        print(f"\n  L <= {L} steps")
        print("   alpha  " + "".join(f"{d:>9}" for d in DURATIONS))
        for a in ALPHAS:
            r = [tim[(np.isclose(tim.alpha, a)) & (tim.duration == d)]
                 for d in DURATIONS]
            cells = "".join(f"{float(x[f'p_timely_{L}'].iloc[0]):>9.3f}"
                            if not x.empty else f"{'-':>9}" for x in r)
            print(f"   {a:5.2f}  {cells}")
    print("\n  conditional mean latency (the confounded quantity, for contrast)")
    print("   alpha  " + "".join(f"{d:>9}" for d in DURATIONS))
    for a in ALPHAS:
        r = [tim[(np.isclose(tim.alpha, a)) & (tim.duration == d)] for d in DURATIONS]
        cells = "".join(f"{float(x['latency_mean'].iloc[0]):>9.2f}"
                        if not x.empty else f"{'-':>9}" for x in r)
        print(f"   {a:5.2f}  {cells}")
    return tim


def do_timely_contours(boots, tim, out):
    """MDF contour under a joint (F1, timely-detection) requirement.

    The latency constraint is now 'at least P_req of injected faults are flagged
    within L steps', which is a requirement on a probability rather than on a
    conditional mean, and therefore survives the censoring critique.
    """
    b = boots["all"]
    rows = []
    for t in F1_TARGETS:
        for L in LAT_BUDGETS:
            for p_req in (0.50, 0.70):
                for d in DURATIONS:
                    vals, agrid = surface(b, d, "f1_mean")
                    a_f1, s_f1 = first_upcrossing(agrid, vals, t, rising=True)
                    sub = tim[tim.duration == d].sort_values("alpha")
                    if sub.empty:
                        continue
                    a_t, s_t = first_upcrossing(
                        sub["alpha"].tolist(),
                        sub[f"p_timely_{L}"].to_numpy(dtype=float),
                        p_req, rising=True)
                    if s_f1 == "right_censored" or s_t == "right_censored":
                        a, s = None, "right_censored"
                    else:
                        a = max(a_f1, a_t)
                        s = ("left_censored"
                             if s_f1 == "left_censored" and s_t == "left_censored"
                             else "ok")
                    rows.append(dict(duration=d, f1_target=t, lat_budget=L,
                                     p_timely_req=p_req, alpha=a, status=s))
    df = pd.DataFrame(rows)
    df.to_csv(out / "wp8_mdf_timely_contours.csv", index=False)

    print("\n" + "=" * 78)
    print("MDF CONTOUR under a joint requirement: F1 >= 0.50 AND at least 50% of")
    print("injected faults flagged within L steps of onset")
    print("=" * 78)
    print("   budget       " + "".join(f"{d:>10}" for d in DURATIONS))
    base = df[(df.f1_target == 0.50) & (df.p_timely_req == 0.50)]
    unc = [first_upcrossing(*(lambda v, g: (g, v))(
        *surface(boots["all"], d, "f1_mean")), 0.50, rising=True) for d in DURATIONS]
    print("   none         " + "".join(f"{fmt(a, s):>10}" for a, s in unc))
    for L in LAT_BUDGETS:
        sub = base[base.lat_budget == L].set_index("duration")
        print(f"   L <= {L:<2d} steps " + "".join(
            f"{fmt(sub.loc[d, 'alpha'], sub.loc[d, 'status']):>10}"
            if d in sub.index else f"{'-':>10}" for d in DURATIONS))
    return df


def do_axes(df, out):
    ref = lstm_learned(df)
    g = ref.groupby(["alpha", "duration"]).agg(
        f1_pa=("f1_pa", "mean"), p_pa=("p_pa", "mean"), r_pa=("r_pa", "mean"),
        f1_pw=("f1_pw", "mean"), p_pw=("p_pw", "mean"), r_pw=("r_pw", "mean")
    ).reset_index()

    rows = []
    for a in ALPHAS:
        lo = g[(np.isclose(g.alpha, a)) & (g.duration == 16)]
        hi = g[(np.isclose(g.alpha, a)) & (g.duration == 256)]
        if lo.empty or hi.empty:
            continue
        dP = float(hi.p_pa.iloc[0] - lo.p_pa.iloc[0])
        dR = float(hi.r_pa.iloc[0] - lo.r_pa.iloc[0])
        rows.append(dict(axis="duration", held_fixed=f"alpha={a}", frm=16.0, to=256.0,
                         d_f1=float(hi.f1_pa.iloc[0] - lo.f1_pa.iloc[0]),
                         d_precision=dP, d_recall=dR,
                         precision_share=dP / (dP + dR) if (dP + dR) else np.nan,
                         recall_lo=float(lo.r_pa.iloc[0]),
                         recall_hi=float(hi.r_pa.iloc[0]),
                         precision_lo=float(lo.p_pa.iloc[0]),
                         precision_hi=float(hi.p_pa.iloc[0])))
    for d in DURATIONS:
        lo = g[(g.duration == d) & (np.isclose(g.alpha, 0.10))]
        hi = g[(g.duration == d) & (np.isclose(g.alpha, 1.50))]
        if lo.empty or hi.empty:
            continue
        dP = float(hi.p_pa.iloc[0] - lo.p_pa.iloc[0])
        dR = float(hi.r_pa.iloc[0] - lo.r_pa.iloc[0])
        rows.append(dict(axis="severity", held_fixed=f"d={d}", frm=0.1, to=1.5,
                         d_f1=float(hi.f1_pa.iloc[0] - lo.f1_pa.iloc[0]),
                         d_precision=dP, d_recall=dR,
                         precision_share=dP / (dP + dR) if (dP + dR) else np.nan,
                         recall_lo=float(lo.r_pa.iloc[0]),
                         recall_hi=float(hi.r_pa.iloc[0]),
                         precision_lo=float(lo.p_pa.iloc[0]),
                         precision_hi=float(hi.p_pa.iloc[0])))
    ax = pd.DataFrame(rows)
    ax.to_csv(out / "wp8_axis_decomposition.csv", index=False)

    print("\n" + "=" * 78)
    print("AXIS DECOMPOSITION, with the precision/recall LEVELS that qualify it")
    print("=" * 78)
    for _, r in ax.iterrows():
        print(f"  {r.axis:<10} {r.held_fixed:<14} dF1={r.d_f1:+.3f}  dP={r.d_precision:+.3f}"
              f"  share={r.precision_share:.3f}   "
              f"P {r.precision_lo:.3f}->{r.precision_hi:.3f}   "
              f"R {r.recall_lo:.3f}->{r.recall_hi:.3f}")

    # The d=16 floor: is it recall-bound or precision-bound?
    print("\n  d=16 cross-section (the 'hard floor'):")
    print("    alpha   F1_pa   precision   recall")
    for a in ALPHAS:
        c = g[(g.duration == 16) & (np.isclose(g.alpha, a))]
        if c.empty:
            continue
        print(f"    {a:5.2f}  {c.f1_pa.iloc[0]:6.3f}   {c.p_pa.iloc[0]:9.3f}"
              f"   {c.r_pa.iloc[0]:6.3f}")
    return g


def do_protocol(df, out):
    rows = []
    for det in ("lstm", "vae"):
        sub = df[(df.detector == det) & (df.fault_model == "learned")
                 & (df.duration > 0)]
        g = sub.groupby(["alpha", "duration"]).agg(
            f1_pa=("f1_pa", "mean"), f1_pw=("f1_pw", "mean")).reset_index()
        for proto in ("f1_pa", "f1_pw"):
            dur = np.mean([
                float(g[(np.isclose(g.alpha, a)) & (g.duration == 256)][proto].iloc[0]
                      - g[(np.isclose(g.alpha, a)) & (g.duration == 16)][proto].iloc[0])
                for a in ALPHAS
                if not g[(np.isclose(g.alpha, a)) & (g.duration == 256)].empty
                and not g[(np.isclose(g.alpha, a)) & (g.duration == 16)].empty])
            sev = np.mean([
                float(g[(g.duration == d) & (np.isclose(g.alpha, 1.50))][proto].iloc[0]
                      - g[(g.duration == d) & (np.isclose(g.alpha, 0.10))][proto].iloc[0])
                for d in DURATIONS
                if not g[(g.duration == d) & (np.isclose(g.alpha, 1.50))].empty
                and not g[(g.duration == d) & (np.isclose(g.alpha, 0.10))].empty])
            rows.append(dict(detector=det, protocol=proto,
                             duration_effect=dur, severity_effect=sev,
                             grid_mean=float(g[proto].mean())))
    pr = pd.DataFrame(rows)
    # Re-emit in the legacy wp3_protocol_comparison schema so the figure code
    # and the claims audit can be repointed at this file without change.
    legacy = []
    for det in ("lstm", "vae"):
        pa = pr[(pr.detector == det) & (pr.protocol == "f1_pa")].iloc[0]
        pw = pr[(pr.detector == det) & (pr.protocol == "f1_pw")].iloc[0]
        legacy.append(dict(
            detector=det, level_pa=pa.grid_mean, level_pw=pw.grid_mean,
            duration_effect_pa=pa.duration_effect,
            duration_effect_pw=pw.duration_effect,
            severity_effect_pa=pa.severity_effect,
            severity_effect_pw=pw.severity_effect,
            duration_retained=pw.duration_effect / pa.duration_effect,
            severity_retained=pw.severity_effect / pa.severity_effect,
            verdict=("prediction HOLDS"
                     if (pw.duration_effect / pa.duration_effect)
                     < (pw.severity_effect / pa.severity_effect)
                     else "prediction FAILS")))
    pd.DataFrame(legacy).to_csv(out / "wp8_protocol_comparison.csv", index=False)

    print("\n" + "=" * 78)
    print("PROTOCOL: point-adjust vs point-wise, same alarm trains")
    print("=" * 78)
    for det in ("lstm", "vae"):
        pa = pr[(pr.detector == det) & (pr.protocol == "f1_pa")].iloc[0]
        pw = pr[(pr.detector == det) & (pr.protocol == "f1_pw")].iloc[0]
        print(f"  {det.upper():5s} duration  PA {pa.duration_effect:+.3f}"
              f"   PW {pw.duration_effect:+.3f}"
              f"   retained {100*pw.duration_effect/pa.duration_effect:5.1f}%")
        print(f"        severity  PA {pa.severity_effect:+.3f}"
              f"   PW {pw.severity_effect:+.3f}"
              f"   retained {100*pw.severity_effect/pa.severity_effect:5.1f}%")
        print(f"        grid mean PA {pa.grid_mean:.3f}   PW {pw.grid_mean:.3f}")
    return pr


def do_detectors(df, out):
    rows = []
    for det in ("lstm", "vae"):
        sub = df[(df.detector == det) & (df.fault_model == "learned")
                 & (df.duration > 0)]
        b = bootstrap_cells(sub, "f1_pa", det)
        for t in F1_TARGETS:
            for rec in contour_interval(b, t):
                rec["detector"] = det
                rows.append(rec)
    dc = pd.DataFrame(rows)
    dc.to_csv(out / "wp8_detector_comparison.csv", index=False)

    lst = df[(df.detector == "lstm") & (df.fault_model == "learned") & (df.duration > 0)]
    vae = df[(df.detector == "vae") & (df.fault_model == "learned") & (df.duration > 0)]
    gl = lst.groupby(["alpha", "duration"]).f1_pa.mean()
    gv = vae.groupby(["alpha", "duration"]).f1_pa.mean()
    common = gl.index.intersection(gv.index)
    diff = (gl.loc[common] - gv.loc[common])

    print("\n" + "=" * 78)
    print("DETECTOR COMPARISON (identical campaign, identical streams)")
    print("=" * 78)
    print(f"  LSTM exceeds VAE at {100*(diff>0).mean():.1f}% of {len(common)} cells,"
          f" mean margin {diff.mean():+.3f}")
    for t in (0.50, 0.70):
        print(f"\n  MDF at F1 >= {t:.2f}")
        for det in ("lstm", "vae"):
            s = dc[(dc.detector == det) & (dc.f1_target == t)]
            print(f"    {det.upper():5s} " + "".join(
                f"{fmt(r.alpha_mdf, r.status):>10}" for _, r in s.iterrows()))
    return dc


def do_faultmodels(df, out):
    models = ["noise", "spike", "step", "learned", "ramp"]
    rows = []
    for m in models:
        sub = df[(df.detector == "lstm") & (df.fault_model == m) & (df.duration > 0)]
        if sub.empty:
            continue
        b = bootstrap_cells(sub, "f1_pa", m)
        gm = float(sub.groupby(["alpha", "duration"]).f1_pa.mean().mean())
        rec = dict(fault_model=m, grid_mean_f1=gm)
        for r in contour_interval(b, 0.50):
            rec[f"mdf_d{int(r['duration'])}"] = fmt(r["alpha_mdf"], r["status"])
        rows.append(rec)
    fm = pd.DataFrame(rows).sort_values("grid_mean_f1", ascending=False)
    fm.to_csv(out / "wp8_faultmodel_comparison.csv", index=False)

    print("\n" + "=" * 78)
    print("FAULT MODELS — one detector instance, one campaign, one training")
    print("=" * 78)
    print(f"  {'model':<10}{'grid-mean F1':>14}   " + "".join(f"{d:>9}" for d in DURATIONS))
    for _, r in fm.iterrows():
        print(f"  {r.fault_model:<10}{r.grid_mean_f1:>14.3f}   " + "".join(
            f"{r.get(f'mdf_d{d}', '-'):>9}" for d in DURATIONS))
    return fm


def do_reproducibility(df, out):
    """Three independent executions of the identical campaign, compared where it
    matters: in the units of the artefact (alpha), not only in F1."""
    runs = {}
    c = lstm_learned(df)
    runs["C (canonical)"] = c.groupby(["alpha", "duration"]).f1_pa.mean()

    b = pd.read_csv(RES / "wp3_envelope_extended.csv")
    b = b[(b.detector == "lstm") & (b.duration > 0)]
    runs["B (wp3 re-run)"] = b.groupby(["alpha", "duration"]).f1_pa.mean()

    a = pd.read_csv(REPO / "results" / "phase4sens_per_channel.csv")
    acol = "f1" if "f1" in a.columns else "f1_pa"
    a = a[a.duration > 0] if "duration" in a.columns else a
    runs["A (thesis)"] = a.groupby(["alpha", "duration"])[acol].mean()

    rows = []
    for name, s in runs.items():
        for t in F1_TARGETS:
            for d in DURATIONS:
                vals = np.array([s.get((al, d), np.nan) for al in ALPHAS], float)
                if not np.isfinite(vals).all():
                    continue
                al, st = first_upcrossing(ALPHAS, vals, t, rising=True)
                rows.append(dict(run=name, f1_target=t, duration=d,
                                 alpha=al, status=st, printed=fmt(al, st)))
    rep = pd.DataFrame(rows)
    rep.to_csv(out / "wp8_run_reproducibility.csv", index=False)

    print("\n" + "=" * 78)
    print("REPRODUCIBILITY ACROSS THREE INDEPENDENT EXECUTIONS")
    print("=" * 78)
    ks = list(runs)
    base = runs[ks[0]]
    for k in ks[1:]:
        common = base.index.intersection(runs[k].index)
        dd = (runs[k].loc[common] - base.loc[common])
        print(f"  surface  {k:<16} vs {ks[0]}:  mean |diff| {dd.abs().mean():.4f}"
              f"   max {dd.abs().max():.4f}   signed {dd.mean():+.4f}")
    for t in F1_TARGETS:
        print(f"\n  contour alpha* at F1 >= {t:.2f}")
        print(f"    {'run':<18}" + "".join(f"{d:>10}" for d in DURATIONS))
        for name in runs:
            s = rep[(rep.run == name) & (rep.f1_target == t)].set_index("duration")
            print(f"    {name:<18}" + "".join(
                f"{s.loc[d,'printed']:>10}" if d in s.index else f"{'-':>10}"
                for d in DURATIONS))
    return rep


def do_fp_control(df):
    ctl = df[(df.duration == 0) & (df.detector == "lstm")]
    fp = ctl["fp_rate_no_injection"].astype(float)
    print("\n" + "=" * 78)
    print("NO-INJECTION CONTROL (LSTM)")
    print("=" * 78)
    print(f"  median FP {fp.median():.4f}   mean FP {fp.mean():.4f}   n={len(fp)}")
    bad = ctl[fp > 0.05][["chan_id", "fp_rate_no_injection"]]
    print(f"  inadmissible at eps=0.05: {len(bad)} channels")
    for _, r in bad.sort_values("fp_rate_no_injection", ascending=False).iterrows():
        print(f"     {r.chan_id:<6} {float(r.fp_rate_no_injection):.5f}")
    print(f"  admissible: {len(ctl) - len(bad)}/{len(ctl)}")


def main():
    df = load_canonical()
    print("=" * 78)
    print(f"WP8 ANALYSIS — {len(df)} rows, "
          f"{df.chan_id.nunique()} channels, "
          f"{df.fault_model.nunique() - 1} fault models, "
          f"{df.detector.nunique()} detectors")
    print("=" * 78)

    do_fp_control(df)
    boots, con = do_envelope_and_contours(df, RES)
    tim = do_timely(df, RES)
    do_timely_contours(boots, tim, RES)
    do_axes(df, RES)
    do_protocol(df, RES)
    do_detectors(df, RES)
    do_faultmodels(df, RES)
    do_reproducibility(df, RES)
    print("\nall outputs written to", RES)


if __name__ == "__main__":
    main()
