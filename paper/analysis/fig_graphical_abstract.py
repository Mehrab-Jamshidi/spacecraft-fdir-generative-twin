"""
Graphical abstract (Elsevier: readable at 13 x 5 cm; >= 1328 x 531 px).

Left:   a real nominal telemetry segment from a held-out channel with one
        injected fault, the object the campaign presents to the detector.
Right:  the resulting minimum-detectable-fault contour of the LSTM at F1 >= 0.50,
        shown as the bootstrap interval it is, with the deadline-constrained
        contour (80 % of faults flagged within five steps) that binds for long
        faults.
All numbers come from the canonical execution (wp10_numbers.json).
"""
import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.use("Agg")
HERE = Path(__file__).resolve().parent
RES, FIG = HERE.parent / "results", HERE.parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)
from _paths import DATA_ROOT as DATA  # noqa: E402
BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#1a1a19", "#5c5c58", "#d8d8d4"
DUR = [16, 32, 64, 128, 256]
mpl.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 10,
                     "axes.edgecolor": MUTED, "axes.linewidth": 0.7, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK,
                     "pdf.fonttype": 42, "savefig.dpi": 300, "savefig.bbox": "tight",
                     "mathtext.fontset": "dejavuserif"})
from fig_qa import check  # noqa: E402


def val(s):
    return np.nan if s == ">1.50" else (0.10 if s == "<=0.10" else float(s))


def main():
    N = json.loads((RES / "wp10_numbers.json").read_text())
    fig = plt.figure(figsize=(13 / 2.54 * 1.6, 5 / 2.54 * 1.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1], wspace=0.28)

    # ---- left: a real nominal segment with one injected fault
    ax = fig.add_subplot(gs[0])
    # P-1 (SMAP): a held-out channel with visible nominal dynamics; the window
    # [200, 520) lies well clear of its labelled anomalies.
    tr = np.load(DATA / "train" / "P-1.npy")[:, 0]
    te = np.load(DATA / "test" / "P-1.npy")[:, 0]
    z = (te - tr.mean()) / (tr.std() or 1.0)
    seg = z[200:520].astype(float).copy()
    t0, d, a = 150, 64, 0.75
    t = np.arange(d)
    g = 3.0 * (t + 1) / d                                  # a drift (ramp) fault
    inj = seg.copy()
    inj[t0:t0 + d] = (1 - a) * seg[t0:t0 + d] + a * g
    lo_y, hi_y = min(seg.min(), inj.min()), max(seg.max(), inj.max())
    span = hi_y - lo_y
    ax.plot(seg, color=MUTED, lw=0.9, label="nominal telemetry")
    ax.plot(np.arange(t0, t0 + d), inj[t0:t0 + d], color=ORANGE, lw=1.8,
            label=r"injected fault ($\alpha$, $d$)")
    ax.axvspan(t0, t0 + d, color=ORANGE, alpha=0.08, lw=0)
    ya = lo_y - 0.12 * span
    ax.annotate("", xy=(t0 + d, ya), xytext=(t0, ya),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9))
    ax.text(t0 + d / 2, ya - 0.13 * span, "duration $d$", ha="center", fontsize=8.5)
    ax.annotate("severity $\\alpha$", xy=(t0 + d - 2, inj[t0 + d - 2]),
                xytext=(t0 + d + 30, hi_y + 0.05 * span),
                fontsize=8.5, arrowprops=dict(arrowstyle="->", color=INK, lw=0.8))
    ax.set_ylim(lo_y - 0.32 * span, hi_y + 0.42 * span)
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title("Inject faults of controlled size into\nnominal telemetry; score a black-box detector",
                 fontsize=9.2, loc="left")
    ax.legend(loc="upper left", fontsize=7.5, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # ---- right: the contour as an interval, and the deadline regime
    ax = fig.add_subplot(gs[1])
    mean = [val(N["mdf_f1_050_mean"][str(x)]) for x in DUR]
    lo = [val(N["mdf_f1_050_hi"][str(x)]) for x in DUR]
    hi = [min(val(N["mdf_f1_050_lo"][str(x)]) if np.isfinite(val(N["mdf_f1_050_lo"][str(x)])) else 1.5, 1.5)
          for x in DUR]
    lo = [v if np.isfinite(v) else 1.5 for v in lo]
    ax.fill_between(DUR, lo, hi, color=BLUE, alpha=0.15, lw=0, label="bootstrap interval")
    xs = [x for x, m in zip(DUR, mean) if np.isfinite(m)]
    ys = [m for m in mean if np.isfinite(m)]
    ax.plot(xs, ys, "-o", color=BLUE, lw=1.8, ms=4, label=r"$F_1 \geq 0.50$, mean")
    dl = [val(N["mdf_joint_F050_P5_80_mean"][str(x)]) for x in DUR]
    ax.plot([x for x, v in zip(DUR, dl) if np.isfinite(v)], [v for v in dl if np.isfinite(v)],
            "--s", color=AQUA, lw=1.6, ms=4, label="+ 80% flagged in 5 steps")
    ax.axhspan(1.5, 1.75, color=GRID, alpha=0.6, lw=0)
    # 1 < alpha <= 1.5 is an extrapolation of the blend model (Section 4.5)
    ax.axhspan(1.0, 1.5, facecolor="none", edgecolor="#c9c9c3", hatch="////", lw=0, zorder=0)
    ax.text(15, 1.58, "not reached in tested range", fontsize=7, color=MUTED, style="italic")
    ax.set_xscale("log", base=2)
    ax.set_xticks(DUR, DUR)
    ax.set_ylim(0, 1.75)
    ax.set_xlabel("fault duration $d$ (steps)", fontsize=9)
    ax.set_ylabel(r"minimum detectable $\alpha^\star$", fontsize=9)
    ax.set_title("Minimum-detectable-fault contour:\nan interval, and a deadline that binds",
                 fontsize=9.2, loc="left")
    ax.legend(loc="lower left", fontsize=7.2, frameon=False)
    ax.grid(color=GRID, lw=0.5)
    fig.text(0.5, -0.06, "20 held-out NASA SMAP/MSL channels  ·  validated against real faults "
             r"with the same trained detector ($\rho = 0.80$)", ha="center", fontsize=8.3, color=MUTED)
    if check(fig, "graphical_abstract"):
        raise SystemExit("graphical_abstract: layout problems, not written")
    fig.savefig(FIG / "graphical_abstract.pdf")
    fig.savefig(FIG / "graphical_abstract.png", dpi=300)
    print("wrote figures/graphical_abstract.pdf (+ .png)")


if __name__ == "__main__":
    main()
