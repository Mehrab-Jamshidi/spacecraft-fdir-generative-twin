"""
================================================================================
WP2 A/B/C — The detection envelope, its confidence bounds, and the
             minimum-detectable-fault (MDF) contours
================================================================================
Paper:  Detectability of learned spacecraft anomaly detectors: minimum-
        detectable-fault envelopes by controlled fault injection
Author: Mehrab Jamshidi — Politecnico di Milano

WHAT THIS DOES, AND WHY IT IS THE CENTRE OF THE PAPER
   The thesis reported a 6x5 grid of mean F1 over fault severity (alpha) and
   duration (d). A grid of numbers is an experiment report. The object an FDIR
   designer actually needs is the inverse: given a required detection quality
   and a latency budget, WHAT IS THE SMALLEST FAULT THE DETECTOR WILL CATCH?

   That object is the minimum detectable fault contour

       alpha*(d | F1_req)  =  min { alpha : D(alpha, d) >= F1_req }

   and, with a latency budget L_req,

       alpha*(d | F1_req, L_req) = min { alpha : D >= F1_req AND L <= L_req }

   MDF is not a new idea — it is a named, established quantity in model-based
   FDI, computed there via interval observers, set-invariance/zonotopic
   characterisation and residual sensitivity. Every one of those routes needs an
   analytical plant model with bounded uncertainty. A learned detector has no
   such model. This script computes the empirical analogue.

DEFINITIONS AND THE CARE THEY NEED
   * FIRST UPCROSSING, not global threshold. The aggregate surface happens to be
     monotone in alpha at every d, but the per-class and per-channel surfaces are
     NOT (e.g. contextual at d=256 peaks at alpha=0.50 then dips). Taking a
     "minimum alpha above target" over a non-monotone curve is ill-defined, so
     the contour is defined as the first upcrossing scanning alpha upward, with
     linear interpolation between bracketing grid points.
   * CENSORING IS REPORTED, NOT HIDDEN. If the target is already met at the
     smallest tested severity the contour is left-censored (reported as
     "<=0.10"); if it is never met at the largest the contour is right-censored
     (">1.50"). Silently clipping either would fabricate a design limit.
   * CONSERVATIVE CONTOUR. Per-cell dispersion across channels is large
     (SD 0.22-0.35), so the contour is computed twice: on the mean surface, and
     on the lower bound of a channel-level bootstrap CI. A design sheet should
     quote the conservative one.

REPRODUCTION GATE
   Before computing anything new, the script rebuilds the published aggregate
   surface from the per-channel file and asserts it against the committed
   aggregated CSV. If the substrate does not reproduce, nothing downstream is
   trustworthy and the script stops.

INPUTS   <thesis repository>/results/phase4sens_per_channel.csv
         <thesis repository>/results/phase4sens_aggregated.csv
OUTPUTS  ../results/wp2_cell_bootstrap.csv     per-cell mean, CI, n
         ../results/wp2_mdf_contours.csv       the design sheet
         ../results/wp2_axis_decomposition.csv precision/recall split per axis
================================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Paths. The thesis repo is read-only source of truth; we only write to ours.
# --------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
from _paths import REPO  # noqa: E402
SRC_PER_CHANNEL = REPO / "results" / "phase4sens_per_channel.csv"
SRC_AGGREGATED = REPO / "results" / "phase4sens_aggregated.csv"
OUT = HERE.parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

ALPHAS = [0.10, 0.25, 0.50, 0.75, 1.00, 1.50]
DURATIONS = [16, 32, 64, 128, 256]

# Held-out channels whose primary class is contextual. Every other held-out
# channel counts as point, which places the one mixed channel (C-1, labelled
# [point, contextual]) in the point group — the convention used throughout the
# thesis, kept here so the numbers stay comparable.
CONTEXTUAL = {"A-8", "E-10", "F-3", "G-1", "M-2", "P-1"}

N_BOOT = 10_000
SEED = 42


# --------------------------------------------------------------------------
# Load + reproduction gate
# --------------------------------------------------------------------------
def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    per = pd.read_csv(SRC_PER_CHANNEL)
    agg = pd.read_csv(SRC_AGGREGATED)
    per["cls"] = np.where(per["chan_id"].isin(CONTEXTUAL), "contextual", "point")
    return per, agg


def reproduction_gate(per: pd.DataFrame, agg: pd.DataFrame) -> None:
    """Rebuild the published surface from per-channel rows and assert it."""
    inj = per[per["duration"] > 0]
    rebuilt = (
        inj.groupby(["alpha", "duration"])
        .agg(f1_mean=("f1", "mean"), n=("f1", "size"))
        .reset_index()
    )
    ref = agg[agg["duration"] > 0][["alpha", "duration", "f1_mean", "n_channels"]]
    merged = rebuilt.merge(ref, on=["alpha", "duration"], suffixes=("_new", "_pub"))

    if len(merged) != len(ALPHAS) * len(DURATIONS):
        sys.exit(f"GATE FAILED: expected 30 cells, matched {len(merged)}")

    dev = (merged["f1_mean_new"] - merged["f1_mean_pub"]).abs()
    n_ok = (merged["n"] == merged["n_channels"]).all()

    print("REPRODUCTION GATE")
    print(f"  cells matched            : {len(merged)}/30")
    print(f"  max |rebuilt - published|: {dev.max():.6f}")
    print(f"  per-cell channel counts  : {'match' if n_ok else 'MISMATCH'}")

    # Spot-check the three values the paper quotes in prose.
    def cell(a, d):
        return float(ref[(ref.alpha == a) & (ref.duration == d)]["f1_mean"].iloc[0])

    checks = {
        "faint/short corner (a=0.10,d=16)": (cell(0.10, 16), 0.2551),
        "large/long corner  (a=1.50,d=256)": (cell(1.50, 256), 0.8454),
        "operational anchor (a=1.00,d=64)": (cell(1.00, 64), 0.5907),
    }
    for name, (got, want) in checks.items():
        flag = "ok" if abs(got - want) < 5e-4 else "MISMATCH"
        print(f"  {name}: {got:.4f} (expect {want:.4f}) [{flag}]")
        if flag != "ok":
            sys.exit("GATE FAILED: headline cell does not match the thesis.")

    if dev.max() > 5e-4 or not n_ok:
        sys.exit("GATE FAILED: surface does not rebuild from per-channel data.")
    print("  -> gate passed; substrate is intact.\n")


# --------------------------------------------------------------------------
# B. Channel-level bootstrap
# --------------------------------------------------------------------------
def bootstrap_cells(per: pd.DataFrame) -> pd.DataFrame:
    """Percentile CI for each cell mean, resampling CHANNELS (the unit of
    independence) rather than injections."""
    rng = np.random.default_rng(SEED)
    rows = []
    inj = per[per["duration"] > 0]

    for group, sub_all in (("all", inj), ("point", inj[inj.cls == "point"]),
                           ("contextual", inj[inj.cls == "contextual"])):
        for a in ALPHAS:
            for d in DURATIONS:
                cell = sub_all[(sub_all.alpha == a) & (sub_all.duration == d)]
                vals = cell["f1"].to_numpy(dtype=float)
                lat = cell["latency_steps"].to_numpy(dtype=float)
                if vals.size == 0:
                    continue
                idx = rng.integers(0, vals.size, size=(N_BOOT, vals.size))
                boot = vals[idx].mean(axis=1)
                lo, hi = np.percentile(boot, [2.5, 97.5])
                rows.append(dict(
                    group=group, alpha=a, duration=d, n_channels=vals.size,
                    f1_mean=vals.mean(), f1_median=np.median(vals),
                    f1_sd=vals.std(ddof=1) if vals.size > 1 else np.nan,
                    f1_ci_lo=lo, f1_ci_hi=hi,
                    latency_mean=np.nanmean(lat) if np.isfinite(lat).any() else np.nan,
                ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# A. MDF contour
# --------------------------------------------------------------------------
def first_upcrossing(alphas: list[float], values: np.ndarray, target: float,
                     rising: bool = True) -> tuple[float | None, str]:
    """Smallest alpha at which `values` first reaches `target`, scanning upward
    with linear interpolation between bracketing grid points.

    rising=True  : we need values >= target (detection quality)
    rising=False : we need values <= target (latency budget)

    Returns (alpha*, status) with status one of: ok | left_censored |
    right_censored.  Censoring is returned explicitly so the caller reports a
    design limit rather than inventing one.
    """
    met = values >= target if rising else values <= target

    if met[0]:
        return alphas[0], "left_censored"
    for i in range(1, len(alphas)):
        if met[i]:
            x0, x1 = alphas[i - 1], alphas[i]
            y0, y1 = values[i - 1], values[i]
            if not np.isfinite(y0) or not np.isfinite(y1) or y1 == y0:
                return x1, "ok"
            return x0 + (target - y0) * (x1 - x0) / (y1 - y0), "ok"
    return None, "right_censored"


def mdf_contours(boot: pd.DataFrame,
                 f1_targets=(0.50, 0.70, 0.80),
                 lat_budgets=(5.0, 10.0, 20.0)) -> pd.DataFrame:
    rows = []
    for group in ("all", "point", "contextual"):
        g = boot[boot.group == group]
        for d in DURATIONS:
            col = g[g.duration == d].sort_values("alpha")
            if col.empty:
                continue
            a_grid = col["alpha"].tolist()
            f1_mean = col["f1_mean"].to_numpy()
            f1_lo = col["f1_ci_lo"].to_numpy()
            lat = col["latency_mean"].to_numpy()

            for t in f1_targets:
                a_mean, s_mean = first_upcrossing(a_grid, f1_mean, t, rising=True)
                a_cons, s_cons = first_upcrossing(a_grid, f1_lo, t, rising=True)
                row = dict(group=group, duration=d, requirement=f"F1>={t:.2f}",
                           f1_target=t, latency_budget=np.nan,
                           alpha_mdf=a_mean, status=s_mean,
                           alpha_mdf_conservative=a_cons, status_conservative=s_cons)
                rows.append(row)

            # Latency-constrained variant: the binding constraint is whichever
            # of the two demands the larger severity.
            for t in f1_targets:
                for L in lat_budgets:
                    a_f1, s_f1 = first_upcrossing(a_grid, f1_mean, t, rising=True)
                    a_l, s_l = first_upcrossing(a_grid, lat, L, rising=False)
                    if s_f1 == "right_censored" or s_l == "right_censored":
                        a_joint, status = None, "right_censored"
                    else:
                        a_joint = max(a_f1, a_l)
                        status = "ok" if "ok" in (s_f1, s_l) else "left_censored"
                    rows.append(dict(
                        group=group, duration=d,
                        requirement=f"F1>={t:.2f} & lat<={L:g}",
                        f1_target=t, latency_budget=L,
                        alpha_mdf=a_joint, status=status,
                        alpha_mdf_conservative=np.nan, status_conservative=""))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# C. Axis decomposition — how much of each axis effect is precision vs recall
# --------------------------------------------------------------------------
def axis_decomposition(agg: pd.DataFrame) -> pd.DataFrame:
    """Point-adjust credits a whole d-step segment when any one step inside it
    is flagged, while false positives are fixed by the clean stream. So
    precision can rise with duration BY CONSTRUCTION. Splitting each axis
    effect into its precision and recall components says how much of the
    'duration dominates' headline is protocol and how much is detection."""
    a = agg[agg.duration > 0]
    rows = []

    for alpha in ALPHAS:                       # duration sweep at fixed severity
        s = a[a.alpha == alpha].sort_values("duration")
        rows.append(dict(
            axis="duration", held_fixed=f"alpha={alpha}",
            frm=int(s.duration.iloc[0]), to=int(s.duration.iloc[-1]),
            d_f1=s.f1_mean.iloc[-1] - s.f1_mean.iloc[0],
            d_precision=s.precision_mean.iloc[-1] - s.precision_mean.iloc[0],
            d_recall=s.recall_mean.iloc[-1] - s.recall_mean.iloc[0]))

    for d in DURATIONS:                        # severity sweep at fixed duration
        s = a[a.duration == d].sort_values("alpha")
        rows.append(dict(
            axis="severity", held_fixed=f"d={d}",
            frm=s.alpha.iloc[0], to=s.alpha.iloc[-1],
            d_f1=s.f1_mean.iloc[-1] - s.f1_mean.iloc[0],
            d_precision=s.precision_mean.iloc[-1] - s.precision_mean.iloc[0],
            d_recall=s.recall_mean.iloc[-1] - s.recall_mean.iloc[0]))

    out = pd.DataFrame(rows)
    out["precision_share"] = out.d_precision / (out.d_precision + out.d_recall)
    return out


# --------------------------------------------------------------------------
def main() -> None:
    per, agg = load()
    reproduction_gate(per, agg)

    boot = bootstrap_cells(per)
    boot.to_csv(OUT / "wp2_cell_bootstrap.csv", index=False)

    contours = mdf_contours(boot)
    contours.to_csv(OUT / "wp2_mdf_contours.csv", index=False)

    decomp = axis_decomposition(agg)
    decomp.to_csv(OUT / "wp2_axis_decomposition.csv", index=False)

    # ---------------- report ----------------
    pd.set_option("display.width", 160)

    print("CELL DISPERSION (aggregate surface, channel-level bootstrap)")
    allg = boot[boot.group == "all"]
    print(f"  per-cell SD range        : {allg.f1_sd.min():.3f} - {allg.f1_sd.max():.3f}")
    print(f"  mean CI half-width       : {((allg.f1_ci_hi - allg.f1_ci_lo) / 2).mean():.3f}")
    wide = allg.loc[(allg.f1_ci_hi - allg.f1_ci_lo).idxmax()]
    print(f"  widest cell              : a={wide.alpha}, d={int(wide.duration)} "
          f"-> {wide.f1_mean:.3f} [{wide.f1_ci_lo:.3f}, {wide.f1_ci_hi:.3f}]\n")

    print("MDF CONTOUR — alpha*(d), aggregate surface, mean vs conservative (CI lower bound)")
    c = contours[(contours.group == "all") & contours.latency_budget.isna()]
    for t in (0.50, 0.70, 0.80):
        sub = c[c.f1_target == t].sort_values("duration")
        fm = "  ".join(
            f"d={int(r.duration):>3}:{_fmt(r.alpha_mdf, r.status):>7}" for r in sub.itertuples())
        fc = "  ".join(
            f"d={int(r.duration):>3}:{_fmt(r.alpha_mdf_conservative, r.status_conservative):>7}"
            for r in sub.itertuples())
        print(f"  F1>={t:.2f}  mean  {fm}")
        print(f"            cons. {fc}")
    print()

    print("MDF CONTOUR BY CLASS (mean surface, F1>=0.70)")
    for group in ("point", "contextual"):
        sub = contours[(contours.group == group) & (contours.f1_target == 0.70)
                       & contours.latency_budget.isna()].sort_values("duration")
        line = "  ".join(f"d={int(r.duration):>3}:{_fmt(r.alpha_mdf, r.status):>7}"
                         for r in sub.itertuples())
        print(f"  {group:>10}  {line}")
    print()

    print("AXIS DECOMPOSITION — is 'duration dominates' a protocol artefact?")
    print(decomp.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()

    print(f"written -> {OUT / 'wp2_cell_bootstrap.csv'}")
    print(f"written -> {OUT / 'wp2_mdf_contours.csv'}")
    print(f"written -> {OUT / 'wp2_axis_decomposition.csv'}")


def _fmt(a, status) -> str:
    if status == "right_censored" or a is None or (isinstance(a, float) and np.isnan(a)):
        return ">1.50"
    if status == "left_censored":
        return "<=0.10"
    return f"{a:.3f}"


if __name__ == "__main__":
    main()
