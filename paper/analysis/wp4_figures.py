"""
================================================================================
WP4 — Manuscript figures (vector PDF + PNG preview)
================================================================================
Paper:  Detectability of learned spacecraft anomaly detectors: minimum-
        detectable-fault envelopes by controlled fault injection

DESIGN RULES APPLIED
  * Categorical hues are the first three validated slots of the reference
    palette: #2a78d6, #eb6834, #1baf7a. Checked with the palette validator on the
    ALL-PAIRS list (worst CVD dE 9.2 deutan, worst normal-vision dE 24.0) — so the
    figures are safe for colour-vision deficiency.
  * Their greyscale luminance gaps are small (0.045-0.135), which a printed
    journal will flatten. Every series therefore ALSO carries a distinct
    linestyle and marker, and contour lines are directly labelled, so identity
    never rests on hue. This is the secondary-encoding requirement, not decoration.
  * Magnitude over the 2-D grid uses a SINGLE-HUE sequential ramp, monotone in
    lightness, so the heatmap survives greyscale by construction. No rainbow.
  * One axis per panel; no dual scales. Recessive grid and spines. Thin marks.
  * Output is vector PDF for submission; PNG at 200 dpi for reading here.
================================================================================
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import os

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

mpl.use("Agg")

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
FIG = HERE.parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1a1a19", "#5c5c58", "#d8d8d4"
ALPHAS = [0.10, 0.25, 0.50, 0.75, 1.00, 1.50]
DURATIONS = [16, 32, 64, 128, 256]

# Single-hue sequential ramp, monotone in lightness (greyscale-safe).
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue",
    ["#f4f8fd", "#cfe0f6", "#9dc1ec", "#6a9fdf", "#3d7ecb", "#1d5a9b", "#12385f"])

mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 9.5, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.7, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.alpha": 0.8,
    "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42,
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf (+ .png)")


def _contour_xy(sub: pd.DataFrame, col: str, status_col: str):
    """Return (x, y, censored_x) with right-censored points split out so they are
    drawn as bounds rather than silently clipped into the plotted line."""
    x, y, cens = [], [], []
    for r in sub.sort_values("duration").itertuples():
        a = getattr(r, col)
        s = getattr(r, status_col)
        if s == "right_censored" or a is None or not np.isfinite(a):
            cens.append(r.duration)
        else:
            x.append(r.duration)
            y.append(a)
    return np.array(x), np.array(y), np.array(cens)


# --------------------------------------------------------------------------
def fig_mdf_contour(contours: pd.DataFrame):
    """THE LEAD FIGURE. The minimum-detectable-fault contour: for a required
    detection quality, the smallest fault severity the detector resolves at each
    duration. Below/left of a contour the requirement is not met."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    targets = [(0.50, BLUE, "-", "o"), (0.70, ORANGE, "--", "s"), (0.80, AQUA, "-.", "^")]

    for ax, (col, stat, title) in zip(axes, [
            ("alpha_mdf", "status",
             "(a) mean, shaded to the bootstrap interval"),
            ("alpha_mdf_conservative", "status_conservative",
             "(b) conservative (CI lower bound)")]):
        for k, (t, c, ls, mk) in enumerate(targets):
            sub = contours[(contours.group == "all") & (contours.f1_target == t)
                           & contours.latency_budget.isna()]
            x, y, cens = _contour_xy(sub, col, stat)
            if len(x):
                ax.plot(x, y, ls, color=c, marker=mk, lw=1.6, ms=5,
                        label=f"$F_1 \\geq {t:.2f}$", zorder=3)
                # Direct label at the LEFTMOST point: the contours converge on the
                # right, so labelling the last point makes them collide.
                ax.annotate(f"$F_1\\geq{t:.2f}$", (x[0], y[0]),
                            textcoords="offset points",
                            xytext=((6, 7) if len(x) > 1 else (-50, 9)),
                            color=c, fontsize=7.5, weight="bold", zorder=4)
            # Requirement unreachable at any tested severity. Arrows are offset
            # horizontally per series, otherwise the three overplot and the
            # figure implies only one requirement is censored.
            off = 2.0 ** ((k - 1) * 0.16)
            for d in cens:
                ax.annotate("", xy=(d * off, 2.22), xytext=(d * off, 1.54),
                            arrowprops=dict(arrowstyle="-|>", color=c, lw=1.1,
                                            shrinkA=0, shrinkB=0), zorder=3)
        # The interval the contour is actually determined to: the band between
        # the contour taken on the bootstrap upper bound and on the lower bound.
        # Drawn only on the mean panel, where the mean line sits inside it.
        if col == "alpha_mdf" and "alpha_mdf_optimistic" in contours.columns:
            for k, (t, c, ls, mk) in enumerate(targets):
                sub = contours[(contours.group == "all")
                               & (contours.f1_target == t)
                               & contours.latency_budget.isna()].sort_values("duration")
                xs, lo, hi = [], [], []
                for r in sub.itertuples():
                    o, os_ = r.alpha_mdf_optimistic, r.status_optimistic
                    cv, cs = r.alpha_mdf_conservative, r.status_conservative
                    if os_ == "right_censored":
                        continue
                    xs.append(r.duration)
                    lo.append(0.10 if os_ == "left_censored" else o)
                    hi.append(1.50 if cs == "right_censored" else cv)
                if len(xs) > 1:
                    ax.fill_between(xs, lo, hi, color=c, alpha=0.11,
                                    lw=0, zorder=1)
        ax.axhspan(1.5, 2.8, color=GRID, alpha=0.45, zorder=0)
        ax.text(13.4, 2.66, "requirement not reachable at any tested severity",
                fontsize=6.6, color=MUTED, style="italic", va="center")
        ax.set_xscale("log", base=2)
        ax.set_xticks(DURATIONS)
        ax.set_xticklabels(DURATIONS)
        ax.set_xlim(12.6, 340)
        ax.set_ylim(0, 2.8)
        ax.set_xlabel("fault duration $d$ (time steps)")
        ax.set_title(title, loc="left", color=INK, pad=6)
    axes[0].set_ylabel("minimum detectable severity $\\alpha^*$")
    # One shared legend below both panels: an in-axes legend collides with the
    # direct labels, and identity must not rest on hue alone.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.09))
    save(fig, "fig_mdf_contour")


def fig_mdf_by_class(contours: pd.DataFrame):
    """Why the two fault families need different design margins."""
    fig, ax = plt.subplots(figsize=(3.7, 3.0))
    for k, (grp, c, ls, mk, lbl) in enumerate(
            [("point", BLUE, "-", "o", "point"),
             ("contextual", ORANGE, "--", "s", "contextual")]):
        sub = contours[(contours.group == grp) & (contours.f1_target == 0.70)
                       & contours.latency_budget.isna()]
        x, y, cens = _contour_xy(sub, "alpha_mdf", "status")
        if len(x):
            ax.plot(x, y, ls, color=c, marker=mk, lw=1.6, ms=5, label=lbl, zorder=3)
        # Offset per class: both classes are censored at d=16, and without this
        # only the class drawn last is visible.
        off = 2.0 ** ((k - 0.5) * 0.18)
        for d in cens:
            ax.annotate("", xy=(d * off, 2.22), xytext=(d * off, 1.54),
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=1.1,
                                        shrinkA=0, shrinkB=0), zorder=3)
    ax.axhspan(1.5, 2.6, color=GRID, alpha=0.45, zorder=0)
    ax.set_xscale("log", base=2)
    ax.set_xticks(DURATIONS)
    ax.set_xticklabels(DURATIONS)
    ax.set_xlim(14, 300)
    ax.set_ylim(0, 2.6)
    ax.set_xlabel("fault duration $d$ (time steps)")
    ax.set_ylabel("minimum detectable severity $\\alpha^*$")
    ax.set_title("MDF contour at $F_1 \\geq 0.70$, by fault class",
                 loc="left", color=INK)
    ax.legend(loc="upper right")
    save(fig, "fig_mdf_by_class")


def fig_surface(boot: pd.DataFrame):
    """The envelope itself: magnitude over the grid (sequential, greyscale-safe),
    and the severity cross-sections with bootstrap CIs so the dispersion the
    heatmap hides is visible."""
    fig = plt.figure(figsize=(7.2, 3.1))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.12], wspace=0.42)

    g = boot[boot.group == "all"]
    piv = g.pivot_table(index="alpha", columns="duration", values="f1_mean")
    ax = fig.add_subplot(gs[0])
    im = ax.imshow(piv.values, cmap=SEQ, vmin=0, vmax=1, aspect="auto", origin="lower")
    ax.set_xticks(range(len(DURATIONS)), DURATIONS)
    ax.set_yticks(range(len(ALPHAS)), [f"{a:.2f}" for a in ALPHAS])
    ax.set_xlabel("duration $d$ (time steps)")
    ax.set_ylabel("severity $\\alpha$")
    ax.grid(False)
    for i in range(len(ALPHAS)):
        for j in range(len(DURATIONS)):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if v > 0.58 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("$F_1$", fontsize=8)
    cb.outline.set_edgecolor(MUTED)
    ax.set_title("(a) detection envelope", loc="left", color=INK)

    ax2 = fig.add_subplot(gs[1])
    # Start the ramp darker than 0.25: the lightest steps vanish on paper.
    shades = [SEQ(0.38), SEQ(0.52), SEQ(0.66), SEQ(0.80), SEQ(0.94)]
    marks = ["o", "s", "^", "D", "v"]
    styles = ["-", "--", "-.", ":", "-"]
    for d, c, mk, ls in zip(DURATIONS, shades, marks, styles):
        s = g[g.duration == d].sort_values("alpha")
        ax2.plot(s.alpha, s.f1_mean, ls, color=c, marker=mk, ms=4, lw=1.4,
                 label=f"$d={d}$", zorder=3)
        ax2.fill_between(s.alpha, s.f1_ci_lo, s.f1_ci_hi, color=c, alpha=0.16, lw=0)
    ax2.set_xlabel("severity $\\alpha$")
    ax2.set_ylabel("mean $F_1$")
    ax2.set_ylim(0, 1.0)
    ax2.set_title("(b) severity cross-sections, 95% bootstrap CI",
                  loc="left", color=INK)
    ax2.legend(ncol=2, loc="lower right")
    save(fig, "fig_surface")


def fig_axis_decomposition(dec: pd.DataFrame):
    """How much of each axis effect is precision (which point-adjust inflates by
    construction) and how much is recall (which it does not)."""
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    dur = dec[dec.axis == "duration"].copy()
    sev = dec[dec.axis == "severity"].copy()
    labels = ([f"$\\alpha$={r.held_fixed.split('=')[1]}" for r in dur.itertuples()]
              + [f"$d$={r.held_fixed.split('=')[1]}" for r in sev.itertuples()])
    prec = list(dur.d_precision) + list(sev.d_precision)
    rec = list(dur.d_recall) + list(sev.d_recall)
    x = np.arange(len(labels))
    split = len(dur)

    ax.bar(x, prec, 0.68, color=BLUE, label="$\\Delta$ precision", zorder=3)
    ax.bar(x, rec, 0.68, bottom=prec, color=ORANGE, hatch="///",
           edgecolor="white", linewidth=0.6, label="$\\Delta$ recall", zorder=3)
    ax.axvline(split - 0.5, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.text(split / 2 - 0.5, 0.78, "duration sweep\n$d$: 16 $\\to$ 256", ha="center",
            fontsize=8, color=INK)
    ax.text(split + len(sev) / 2 - 0.5, 0.78,
            "severity sweep\n$\\alpha$: 0.10 $\\to$ 1.50", ha="center",
            fontsize=8, color=INK)
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylabel("change in metric across the sweep")
    ax.set_ylim(0, 0.92)
    ax.set_title("Decomposition of each axis effect", loc="left", color=INK)
    # Below the axes: an in-axes legend sits on the sweep annotations.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2)
    save(fig, "fig_axis_decomposition")


def fig_predictive_validity(cmp_: pd.DataFrame):
    """Does the envelope predict detection of REAL faults? Two severity
    parameterisations, same envelope, same real anomalies."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), sharex=True, sharey=True)
    from scipy.stats import spearmanr

    panels = [("predicted_f1_dist", "(a) distributional severity\n(six-feature displacement)"),
              ("predicted_f1_ar", "(b) predictive severity\n(AR one-step residual)")]
    for ax, (col, title) in zip(axes, panels):
        d = cmp_.dropna(subset=[col, "observed_f1"])
        ax.plot([0, 1], [0, 1], color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
        for cls, c, mk in [("point", BLUE, "o"), ("contextual", ORANGE, "s")]:
            s = d[d.cls == cls]
            ax.scatter(s.observed_f1, s[col], s=42, marker=mk, facecolor=c,
                       edgecolor="white", linewidth=0.8, label=cls, zorder=3)
        rho = spearmanr(d[col], d.observed_f1).statistic
        mae = (d[col] - d.observed_f1).abs().mean()
        # Lower right: the upper left is where the badly-predicted contextual
        # channels sit, and the stats box would sit on top of G-1.
        ax.text(0.96, 0.06, f"$\\rho$ = {rho:+.2f}\nMAE = {mae:.3f}",
                transform=ax.transAxes, va="bottom", ha="right",
                fontsize=8.5, color=INK)
        ax.set_xlabel("observed $F_1$ on the real anomaly")
        ax.set_title(title, loc="left", color=INK, fontsize=8.8)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

    # G-1: the case that shows the mechanism.
    d2 = cmp_[cmp_.chan_id == "G-1"]
    if len(d2):
        r = d2.iloc[0]
        axes[0].annotate("G-1", (r.observed_f1, r.predicted_f1_dist),
                         textcoords="offset points", xytext=(8, -2),
                         fontsize=8, color=ORANGE, weight="bold")
        axes[1].annotate("G-1", (r.observed_f1, r.predicted_f1_ar),
                         textcoords="offset points", xytext=(8, 4),
                         fontsize=8, color=ORANGE, weight="bold")
    axes[0].set_ylabel("$F_1$ predicted by the envelope")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.07))
    save(fig, "fig_predictive_validity")


def _upcross(vals, target):
    a = ALPHAS
    if vals[0] >= target:
        return a[0], "left"
    for i in range(1, len(a)):
        if vals[i] >= target:
            y0, y1 = vals[i - 1], vals[i]
            if y1 == y0:
                return a[i], "ok"
            return a[i - 1] + (target - y0) * (a[i] - a[i - 1]) / (y1 - y0), "ok"
    return None, "right"


def fig_detector_comparison(ext: pd.DataFrame):
    """Same fault model, same streams, same channels, same deployable rule —
    only the detector differs. If the contours separate, the envelope is a
    property of the detector and the method is a comparison instrument."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    specs = [("lstm", BLUE, "-", "o", "LSTM (prediction error)"),
             ("vae", ORANGE, "--", "s", "VAE (reconstruction error)")]

    ax = axes[0]
    for k, (det, c, ls, mk, lbl) in enumerate(specs):
        piv = (ext[(ext.detector == det) & (ext.duration > 0)]
               .groupby(["alpha", "duration"]).f1_pa.mean().unstack())
        xs, ys, cens = [], [], []
        for d in DURATIONS:
            if d not in piv.columns:
                continue
            a, st = _upcross(piv[d].reindex(ALPHAS).to_numpy(), 0.50)
            (cens if st == "right" else xs).append(d)
            if st != "right":
                ys.append(a)
        if xs:
            ax.plot(xs, ys, ls, color=c, marker=mk, lw=1.6, ms=5, label=lbl, zorder=3)
        off = 2.0 ** ((k - 0.5) * 0.18)
        for d in cens:
            ax.annotate("", xy=(d * off, 2.22), xytext=(d * off, 1.54),
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=1.1,
                                        shrinkA=0, shrinkB=0), zorder=3)
    ax.axhspan(1.5, 2.75, color=GRID, alpha=0.45, zorder=0)
    ax.text(13.6, 2.6, "not reachable at any tested severity", fontsize=6.6,
            color=MUTED, style="italic", va="center")
    ax.set_xscale("log", base=2)
    ax.set_xticks(DURATIONS, DURATIONS)
    ax.set_xlim(12.6, 340)
    ax.set_ylim(0, 2.75)
    ax.set_xlabel("fault duration $d$ (time steps)")
    ax.set_ylabel("minimum detectable severity $\\alpha^*$")
    ax.set_title("(a) MDF contour at $F_1 \\geq 0.50$", loc="left", color=INK)

    ax2 = axes[1]
    for det, c, ls, mk, lbl in specs:
        piv = (ext[(ext.detector == det) & (ext.duration > 0)]
               .groupby(["alpha", "duration"]).f1_pa.mean().unstack())
        s = piv.loc[:, 256] if 256 in piv.columns else piv.iloc[:, -1]
        ax2.plot(ALPHAS, s.reindex(ALPHAS), ls, color=c, marker=mk, lw=1.6,
                 ms=5, label=lbl, zorder=3)
    ax2.set_xlabel("severity $\\alpha$")
    ax2.set_ylabel("mean $F_1$")
    ax2.set_ylim(0, 1.0)
    ax2.set_title("(b) severity response at $d = 256$", loc="left", color=INK)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.09))
    save(fig, "fig_detector_comparison")


def fig_protocol(prot: pd.DataFrame):
    """Direct test of the prediction made by the precision/recall decomposition:
    scored point-wise, severity should survive and duration should compress."""
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    dets = ["lstm", "vae"]
    x = np.arange(len(dets))
    w = 0.34

    dur = [float(prot[prot.detector == d].duration_retained.iloc[0]) * 100 for d in dets]
    sev = [float(prot[prot.detector == d].severity_retained.iloc[0]) * 100 for d in dets]

    ax.bar(x - w / 2, dur, w, color=BLUE, label="duration axis", zorder=3)
    ax.bar(x + w / 2, sev, w, color=ORANGE, hatch="///", edgecolor="white",
           linewidth=0.6, label="severity axis", zorder=3)
    for xi, v in zip(x - w / 2, dur):
        ax.text(xi, v + 3, f"{v:.0f}%", ha="center", fontsize=8, color=INK)
    for xi, v in zip(x + w / 2, sev):
        ax.text(xi, v + 3, f"{v:.0f}%", ha="center", fontsize=8, color=INK)
    ax.axhline(100, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.text(1.46, 103, "fully retained", fontsize=7, color=MUTED,
            style="italic", ha="right")
    ax.set_xticks(x, ["LSTM\n(prediction error)", "VAE\n(reconstruction error)"])
    ax.set_ylabel("axis effect retained under\npoint-wise scoring (%)")
    ax.set_ylim(0, 125)
    ax.set_title("What survives when point-adjust is removed", loc="left", color=INK)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2)
    save(fig, "fig_protocol")


def fig_fault_models(par: pd.DataFrame, gen: pd.DataFrame, cmp_: pd.DataFrame):
    """Five fault models, one detector, one campaign.

    Five series exceeds the three categorical slots that validate on the
    all-pairs CVD list, so identity is NOT carried by hue here. The rhetorical
    contrast is learned-versus-parametric, so the learned model takes the
    emphasis colour and the four parametric families share a muted ink,
    separated by linestyle, marker and direct labels. That keeps the figure
    legible in greyscale and colourblind-safe by construction."""
    from scipy.stats import spearmanr

    PARAM = "#5c5c58"
    specs = [("noise", "-", "v"), ("spike", "--", "^"),
             ("step", "-.", "s"), ("ramp", (0, (1, 1.6)), "D")]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))

    ax = axes[0]
    gp = gen.groupby(["alpha", "duration"]).f1_pa.mean().unstack()
    series = [("learned (GAN)", gp, BLUE, "-", "o", 2.0)]
    for fam, ls, mk in specs:
        pp = (par[par.family == fam].groupby(["alpha", "duration"])
              .f1_pa.mean().unstack())
        series.append((fam, pp, PARAM, ls, mk, 1.2))

    for k, (name, piv, c, ls, mk, lw) in enumerate(series):
        xs, ys, cens = [], [], []
        for d in DURATIONS:
            if d not in piv.columns:
                continue
            a, st = _upcross(piv[d].reindex(ALPHAS).to_numpy(), 0.50)
            if st == "right":
                cens.append(d)
            else:
                xs.append(d)
                ys.append(a)
        if xs:
            # linestyle as a keyword: one of these is a dash tuple, which
            # matplotlib cannot parse as a positional format string.
            ax.plot(xs, ys, linestyle=ls, color=c, marker=mk, lw=lw, ms=4.5,
                    label=name, zorder=4 if name.startswith("learned") else 3)
            ax.annotate(name, (xs[0], ys[0]), textcoords="offset points",
                        xytext=((8, -13) if name.startswith("learned") else (5, 6)),
                        fontsize=6.8, color=c,
                        weight="bold" if name.startswith("learned") else "normal")
        off = 2.0 ** ((k - 2) * 0.1)
        for d in cens:
            ax.annotate("", xy=(d * off, 2.2), xytext=(d * off, 1.54),
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=1.0,
                                        shrinkA=0, shrinkB=0), zorder=3)
    ax.axhspan(1.5, 2.75, color=GRID, alpha=0.45, zorder=0)
    ax.text(13.4, 2.62, "not reachable at any tested severity", fontsize=6.4,
            color=MUTED, style="italic", va="center")
    ax.set_xscale("log", base=2)
    ax.set_xticks(DURATIONS, DURATIONS)
    ax.set_xlim(12.6, 360)
    ax.set_ylim(0, 2.75)
    ax.set_xlabel("fault duration $d$ (time steps)")
    ax.set_ylabel("minimum detectable severity $\\alpha^*$")
    ax.set_title("(a) MDF contour at $F_1 \\geq 0.50$", loc="left", color=INK)

    ax2 = axes[1]
    # Robustness run: all five models computed in one loop off one RNG stream.
    cmp_ = cmp_[cmp_.ar_order == 8] if "ar_order" in cmp_ else cmp_
    order = ["step", "spike", "generative", "ramp", "noise"]
    labels = {"generative": "learned (GAN)"}
    vals, cols, lo, hi = [], [], [], []
    rng = np.random.default_rng(42)
    n = len(cmp_)
    idx = rng.integers(0, n, size=(2000, n))
    for m in order:
        col = f"pred_{m}"
        x_, y_ = cmp_[col].to_numpy(float), cmp_.observed.to_numpy(float)
        v = spearmanr(x_, y_).statistic
        bs = np.array([spearmanr(x_[i], y_[i]).statistic for i in idx])
        bs = bs[np.isfinite(bs)]
        vals.append(v)
        lo.append(v - np.percentile(bs, 2.5))
        hi.append(np.percentile(bs, 97.5) - v)
        cols.append(BLUE if m == "generative" else PARAM)
    y = np.arange(len(order))[::-1]
    ax2.barh(y, vals, 0.6, color=cols, zorder=3)
    ax2.errorbar(vals, y, xerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=0.9,
                 capsize=2.5, zorder=4)
    for yi, v, h in zip(y, vals, hi):
        ax2.text(min(v + h + 0.03, 1.02), yi, f"{v:+.2f}", va="center", fontsize=7.5,
                 color=INK)
    ax2.set_yticks(y, [labels.get(m, m) for m in order])
    ax2.set_xlim(-0.3, 1.2)
    ax2.set_xlabel("rank agreement $\\rho$ with observed\ndetection of the REAL faults")
    ax2.set_title("(b) which stimulus predicts real faults", loc="left", color=INK)
    ax2.grid(axis="y", visible=False)

    fig.tight_layout()
    save(fig, "fig_fault_models")


def fig_axes_protocol(dec: pd.DataFrame, prot: pd.DataFrame):
    """What point-adjust contributes to each axis, in one figure.
    (a) each axis effect split into precision and recall; (b) the fraction of
    each axis effect that survives when the same alarm trains are re-scored
    point-wise."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0),
                             gridspec_kw=dict(width_ratios=[1.55, 1], wspace=0.34))
    ax = axes[0]
    dur = dec[dec.axis == "duration"].copy()
    sev = dec[dec.axis == "severity"].copy()
    labels = ([f"$\\alpha$={float(r.held_fixed.split('=')[1]):g}" for r in dur.itertuples()]
              + [f"$d$={r.held_fixed.split('=')[1]}" for r in sev.itertuples()])
    prec = list(dur.d_precision) + list(sev.d_precision)
    rec = list(dur.d_recall) + list(sev.d_recall)
    x = np.arange(len(labels))
    split = len(dur)
    ax.bar(x, prec, 0.68, color=BLUE, label="$\\Delta$ precision", zorder=3)
    ax.bar(x, rec, 0.68, bottom=prec, color=ORANGE, hatch="///",
           edgecolor="white", linewidth=0.6, label="$\\Delta$ recall", zorder=3)
    ax.axvline(split - 0.5, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.text(split / 2 - 0.5, 0.80, "duration sweep\n$d$: 16 $\\to$ 256", ha="center",
            fontsize=7.5, color=INK)
    ax.text(split + len(sev) / 2 - 0.5, 0.80,
            "severity sweep\n$\\alpha$: 0.10 $\\to$ 1.50", ha="center",
            fontsize=7.5, color=INK)
    ax.set_xticks(x, labels, rotation=50, ha="right", fontsize=7)
    ax.set_ylabel("change across the sweep")
    ax.set_ylim(0, 0.95)
    ax.set_title("(a) precision and recall components", loc="left", color=INK)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.33), ncol=2)

    ax = axes[1]
    dets = ["lstm", "vae"]
    xx = np.arange(len(dets))
    w = 0.34
    dr = [float(prot[prot.detector == d].duration_retained.iloc[0]) * 100 for d in dets]
    sr = [float(prot[prot.detector == d].severity_retained.iloc[0]) * 100 for d in dets]
    ax.bar(xx - w / 2, dr, w, color=BLUE, label="duration axis", zorder=3)
    ax.bar(xx + w / 2, sr, w, color=ORANGE, hatch="///", edgecolor="white",
           linewidth=0.6, label="severity axis", zorder=3)
    for xi, v in list(zip(xx - w / 2, dr)) + list(zip(xx + w / 2, sr)):
        ax.text(xi, v + 3, f"{v:.0f}%", ha="center", fontsize=7.5, color=INK)
    ax.axhline(100, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.set_xticks(xx, ["LSTM", "VAE"])
    ax.set_ylabel("effect retained point-wise (%)")
    ax.set_ylim(0, 125)
    ax.set_title("(b) re-scored point-wise", loc="left", color=INK)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=1)
    save(fig, "fig_axes_protocol")


def fig_timely(pod: pd.DataFrame, chance: pd.DataFrame, can: pd.DataFrame):
    """Detection within a deadline is flat in duration; the whole-segment hit
    rate that point-adjust rewards rises with duration about as fast as the
    chance that a nominal window of that length already holds a false alarm."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw=dict(wspace=0.3))
    shades = [SEQ(0.38), SEQ(0.52), SEQ(0.66), SEQ(0.80), SEQ(0.94)]
    marks = ["o", "s", "^", "D", "v"]
    styles = ["-", "--", "-.", ":", "-"]
    ax = axes[0]
    p5 = pod[pod.quantity == "p5"]
    for d, c, mk, ls in zip(DURATIONS, shades, marks, styles):
        s = p5[p5.duration == d].sort_values("alpha")
        ax.plot(s.alpha, s["mean"], ls, color=c, marker=mk, ms=4, lw=1.3,
                label=f"$d={d}$", zorder=3)
    lst = chance[chance.detector == "lstm"]
    cm = float(lst.chance_w6.mean())
    ax.axhline(cm, color=ORANGE, lw=1.1, ls=(0, (4, 2)), zorder=2)
    ax.text(1.49, cm + 0.02, "false-call probability, 6-step window", ha="right",
            fontsize=6.8, color=ORANGE)
    ax.set_xlabel("severity $\\alpha$")
    ax.set_ylabel("$\\mathcal{P}_5$: flagged within 5 steps")
    ax.set_ylim(0, 1.0)
    ax.set_title("(a) detection within a 5-step deadline", loc="left", color=INK)
    ax.legend(ncol=2, loc="upper left", fontsize=7)

    ax = axes[1]
    pdet = pod[(pod.quantity == "pdet") & np.isclose(pod.alpha, 0.10)].sort_values("duration")
    p5a = p5[np.isclose(p5.alpha, 0.10)].sort_values("duration")
    ch = [float(lst[f"chance_w{d}"].mean()) for d in DURATIONS]
    ax.plot(DURATIONS, pdet["mean"], "-", color=BLUE, marker="o", ms=4.5, lw=1.6,
            label="flagged anywhere in the fault\n(what point-adjust credits)")
    ax.plot(DURATIONS, ch, "--", color=ORANGE, marker="s", ms=4.5, lw=1.4,
            label="nominal window of length $d$\nholding a false alarm")
    ax.plot(DURATIONS, p5a["mean"], "-.", color=AQUA, marker="^", ms=4.5, lw=1.4,
            label="flagged within 5 steps")
    ax.set_xscale("log", base=2)
    ax.set_xticks(DURATIONS, DURATIONS)
    ax.set_xlabel("fault duration $d$ (time steps)")
    ax.set_ylabel("probability")
    ax.set_ylim(0, 1.0)
    ax.set_title("(b) what a longer fault buys, at $\\alpha = 0.10$", loc="left", color=INK)
    ax.legend(loc="lower right", fontsize=6.6)
    save(fig, "fig_timely")


def _canonical() -> bool:
    """Figures are drawn from the canonical single campaign (run C) by default.

    Set ENVELOPE_SOURCE=thesis to redraw them from the original three-run
    patchwork, which is retained only so the two can be compared.
    """
    return os.environ.get("ENVELOPE_SOURCE", "canonical").lower() in (
        "canonical", "wp8", "c")


def main():
    canon = _canonical() and (RES / "wp8_canonical_envelope.csv").exists()
    pre = "wp8_" if canon else "wp2_"
    print("figure source:", "run C (canonical)" if canon else "runs A+B (legacy)")

    # The band file carries the same run-A contours plus the optimistic bound,
    # so the lead figure can draw the interval that Table 1 reports.
    con_p = RES / (pre + "mdf_contours.csv")
    if not canon and (RES / "wp2_mdf_contours_band.csv").exists():
        con_p = RES / "wp2_mdf_contours_band.csv"
    contours = pd.read_csv(con_p)
    boot = pd.read_csv(RES / (pre + "cell_bootstrap.csv"))
    dec = pd.read_csv(RES / (pre + "axis_decomposition.csv"))
    sev_p = RES / "wp2_severity_metric_comparison_runC.csv"
    if not (canon and sev_p.exists()):
        sev_p = RES / "wp2_severity_metric_comparison.csv"
    cmp_ = pd.read_csv(sev_p)

    print("building figures ...")
    fig_mdf_contour(contours)
    fig_mdf_by_class(contours)
    fig_surface(boot)
    fig_predictive_validity(cmp_)

    if canon:
        can = pd.read_csv(RES / "wp8_canonical_envelope.csv")
        fig_detector_comparison(can[can.fault_model.isin(["learned", "none"])])
        fig_axes_protocol(dec, pd.read_csv(RES / "wp8_protocol_comparison.csv"))
        fig_timely(pd.read_csv(RES / "wp10_pod_bootstrap.csv"),
                   pd.read_csv(RES / "wp9_chance_windows.csv"), can)
        lstm = can[(can.detector == "lstm") & (can.duration > 0)]
        par = lstm[lstm.fault_model != "learned"].rename(
            columns={"fault_model": "family"})
        gen = lstm[lstm.fault_model == "learned"]
        rob_p = RES / "wp7_parametric_robustness_runC.csv"
        if not rob_p.exists():
            rob_p = RES / "wp7_parametric_robustness.csv"
        if rob_p.exists():
            fig_fault_models(par, gen, pd.read_csv(rob_p))
        else:
            print("  (parametric robustness absent, skipping fault-model figure)")
        print("all figures ->", FIG)
        return

    ext_p = RES / "wp3_envelope_extended.csv"
    prot_p = RES / "wp3_protocol_comparison.csv"
    if ext_p.exists() and prot_p.exists():
        fig_detector_comparison(pd.read_csv(ext_p))
        fig_protocol(pd.read_csv(prot_p))
    par_p = RES / "wp7_parametric_envelope.csv"
    cmp_p = RES / "wp7_parametric_robustness.csv"
    if par_p.exists() and cmp_p.exists() and ext_p.exists():
        g = pd.read_csv(ext_p)
        g = g[(g.detector == "lstm") & (g.duration > 0)]
        fig_fault_models(pd.read_csv(par_p), g, pd.read_csv(cmp_p))
    print("all figures ->", FIG)


if __name__ == "__main__":
    main()
