"""
Figure 1 — the characterisation procedure, as a schematic.

Top row: the injection campaign that produces the envelope and its inversion.
Bottom row: the validation branch that places real labelled faults on the same
axes. Drawn with matplotlib so it shares fonts and palette with the data figures.
Output: ../figures/fig_schematic.pdf (+ .png)
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

mpl.use("Agg")
FIG = Path(__file__).resolve().parent.parent / "figures"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, FILL, FILL2 = "#1a1a19", "#5c5c58", "#eef4fc", "#fdf1eb"

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
    "text.color": INK, "pdf.fonttype": 42, "savefig.dpi": 220,
    "savefig.bbox": "tight",
})


def box(ax, x, y, w, h, title, body, fc=FILL, ec=BLUE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.015",
                                fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h - 0.03, title, ha="center", va="top",
            fontsize=7.4, weight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.10, body, ha="center", va="top",
            fontsize=6.3, color=MUTED, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, color=MUTED, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=9, lw=1.0, color=color,
                                 linestyle=ls, shrinkA=2, shrinkB=2))


def main():
    fig, ax = plt.subplots(figsize=(7.3, 3.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    W, H = 0.163, 0.34
    top, bot = 0.585, 0.10
    xs = [0.008 + i * 0.2045 for i in range(5)]

    box(ax, xs[0], top, W, H, "Nominal telemetry",
        "clean test segment\nof a held-out channel,\nnever seen in training")
    box(ax, xs[1], top, W, H, "Fault injection",
        r"$x^{\mathrm{inj}}=(1-\alpha)x^{\mathrm{nom}}+\alpha g$" "\n"
        r"severity $\alpha$, duration $d$" "\n"
        "$g$: step, ramp, spike,\nnoise or learned")
    box(ax, xs[2], top, W, H, "Detector",
        "black box: score $s_t$\n"
        r"threshold $\mu_E+3\sigma_E$," "\n"
        "trained on nominal\ndata only")
    box(ax, xs[3], top, W, H, "Detection envelope",
        r"quality $\mathcal{D}(\alpha,d)$" "\n"
        r"timeliness $\mathcal{P}_L(\alpha,d)$" "\n"
        r"false alarms at $\alpha=0$")
    box(ax, xs[4], top, W, H, "MDF contour",
        r"smallest $\alpha$ meeting" "\n"
        "a requirement, as an\ninterval over a channel\nbootstrap",
        fc="#e9f7f1", ec=AQUA)
    for i in range(4):
        arrow(ax, xs[i] + W, top + H / 2, xs[i + 1], top + H / 2, color=BLUE)

    # validation branch
    box(ax, xs[0], bot, W, H, "Real faults",
        "25 labelled anomalies,\nscored by the same\ntrained detector",
        fc=FILL2, ec=ORANGE)
    box(ax, xs[1], bot, W, H, "Severity placement",
        r"calibrate $T(\alpha)$ on" "\n"
        r"injected windows," "\n"
        r"$\hat\alpha=T^{-1}(T^{\mathrm{real}})$")
    ax.patches[-1].set_facecolor(FILL2)
    ax.patches[-1].set_edgecolor(ORANGE)
    box(ax, xs[3], bot, W, H, "Prediction",
        "envelope read at\n"
        r"$(\hat\alpha,\ d^{\mathrm{real}})$",
        fc=FILL2, ec=ORANGE)
    box(ax, xs[4], bot, W, H, "Validity test",
        "agreement with the\nobserved detection;\nablations, controls",
        fc=FILL2, ec=ORANGE)
    arrow(ax, xs[0] + W, bot + H / 2, xs[1], bot + H / 2, color=ORANGE)
    arrow(ax, xs[1] + W, bot + H / 2, xs[3], bot + H / 2, color=ORANGE)
    arrow(ax, xs[3] + W, bot + H / 2, xs[4], bot + H / 2, color=ORANGE)
    # the envelope feeds the prediction
    arrow(ax, xs[3] + W / 2, top, xs[3] + W / 2, bot + H, color=MUTED, ls="--")
    # injection model feeds the calibration
    arrow(ax, xs[1] + W / 2, top, xs[1] + W / 2, bot + H, color=MUTED, ls="--")

    ax.text(0.5, 0.985, "characterisation by controlled injection", ha="center",
            va="top", fontsize=8, color=BLUE, style="italic")
    ax.text(0.5, 0.005, "validation against real faults", ha="center",
            va="bottom", fontsize=8, color=ORANGE, style="italic")

    fig.savefig(FIG / "fig_schematic.pdf")
    fig.savefig(FIG / "fig_schematic.png")
    print("wrote figures/fig_schematic.pdf (+ .png)")


if __name__ == "__main__":
    main()
