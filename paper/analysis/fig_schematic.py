"""
Figure 1 — the characterisation procedure, as a schematic.

Top row: the injection campaign that produces the envelope and its contour.
Bottom row: the validation branch that places real labelled faults on the same
axes.

The figure is drawn at exactly the manuscript's text width (6.47 in) and saved
without cropping, so LaTeX includes it at 100 % and every font keeps its stated
size. Each text is fitted to its box: if it does not fit with a margin, its size
is reduced in 0.25 pt steps, and the script fails if any text still escapes its
box or overlaps another (fig_qa.check).

Output: ../figures/fig_schematic.pdf (+ .png)
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from fig_qa import check

mpl.use("Agg")
FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED = "#1a1a19", "#4a4a47"
FILL_TOP, FILL_BOT, FILL_OUT = "#eef4fc", "#fdf1eb", "#e9f7f1"

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif", "text.color": INK, "pdf.fonttype": 42,
})

W_IN, H_IN = 6.47, 2.95          # manuscript text width; height of the diagram
BW, BH, GAP = 1.10, 0.98, 0.2    # box width, height, horizontal gap (inches)
X0 = (W_IN - (5 * BW + 4 * GAP)) / 2
TOP_Y, BOT_Y = 1.72, 0.26        # lower edge of each row (inches)
MARGIN_PT = 3.0                  # text must clear the box edge by this much

TOP = [
    ("Telemetry", ["held-out channel,", "clean test segment,", "unseen in training"], FILL_TOP, BLUE),
    ("Fault injection", [r"$x=(1-\alpha)\,x^{\mathrm{nom}}+\alpha\,g$", r"severity $\alpha$, duration $d$",
                         r"$g$: step, ramp, spike,", "noise or learned"], FILL_TOP, BLUE),
    ("Detector", ["black box: a score", r"and a threshold", r"$\mu+3\sigma$ set on", "nominal data only"],
     FILL_TOP, BLUE),
    ("Envelope", [r"quality $\mathcal{D}(\alpha,d)$", r"deadline $\mathcal{P}_L(\alpha,d)$",
                  r"false alarms at $\alpha=0$"], FILL_TOP, BLUE),
    ("MDF contour", [r"smallest $\alpha$ that", "meets a requirement,", "with a bootstrap", "interval"],
     FILL_OUT, AQUA),
]
BOTTOM = {
    0: ("Real faults", ["25 labelled", "anomalies, scored", "by the same", "trained detector"]),
    1: ("Severity", [r"calibrate $T(\alpha)$,", "then invert:", r"$\hat\alpha=T^{-1}(T^{\mathrm{real}})$"]),
    3: ("Prediction", ["envelope read at", r"$(\hat\alpha,\ d^{\mathrm{real}})$"]),
    4: ("Validity", ["agreement with the", "observed detection;", "ablations, controls"]),
}


def fit_text(ax, x, y, s, box, size, weight="normal", color=INK, min_size=6.0, **kw):
    """Place text and shrink it until it clears the box edges by MARGIN_PT."""
    fig = ax.figure
    r = fig.canvas.get_renderer()
    while True:
        t = ax.text(x, y, s, fontsize=size, weight=weight, color=color, **kw)
        bb = t.get_window_extent(renderer=r)
        pb = box.get_window_extent(renderer=r)
        m = MARGIN_PT * fig.dpi / 72
        if bb.x0 >= pb.x0 + m and bb.x1 <= pb.x1 - m:
            return t
        t.remove()
        if size - 0.25 < min_size:
            raise RuntimeError(f"text does not fit its box even at {min_size} pt: {s!r}")
        size -= 0.25


def draw_box(ax, col, row_y, title, lines, fill, edge):
    x = X0 + col * (BW + GAP)
    box = FancyBboxPatch((x, row_y), BW, BH,
                         boxstyle="round,pad=0,rounding_size=0.06",
                         fc=fill, ec=edge, lw=1.0)
    ax.add_patch(box)
    ax.figure.canvas.draw()
    texts = [fit_text(ax, x + BW / 2, row_y + BH - 0.08, title, box, 8.0, weight="bold",
                      ha="center", va="top")]
    body = "\n".join(lines)
    texts.append(fit_text(ax, x + BW / 2, row_y + BH - 0.31, body, box, 7.0, color=MUTED,
                          ha="center", va="top", linespacing=1.28))
    return box, texts


def arrow(ax, x0, y0, x1, y1, color, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=11, lw=1.2, color=color,
                                 linestyle=ls, shrinkA=1.5, shrinkB=1.5))


def main():
    fig = plt.figure(figsize=(W_IN, H_IN), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W_IN)
    ax.set_ylim(0, H_IN)
    ax.axis("off")

    pairs = []
    for c, (title, lines, fill, edge) in enumerate(TOP):
        box, texts = draw_box(ax, c, TOP_Y, title, lines, fill, edge)
        pairs += [(t, box) for t in texts]
    for c, (title, lines) in BOTTOM.items():
        box, texts = draw_box(ax, c, BOT_Y, title, lines, FILL_BOT, ORANGE)
        pairs += [(t, box) for t in texts]

    xr = lambda c: X0 + c * (BW + GAP) + BW          # right edge of column c
    xl = lambda c: X0 + c * (BW + GAP)               # left edge of column c
    xm = lambda c: X0 + c * (BW + GAP) + BW / 2      # centre of column c
    for c in range(4):
        arrow(ax, xr(c), TOP_Y + BH / 2, xl(c + 1), TOP_Y + BH / 2, BLUE)
    arrow(ax, xr(0), BOT_Y + BH / 2, xl(1), BOT_Y + BH / 2, ORANGE)
    arrow(ax, xr(1), BOT_Y + BH / 2, xl(3), BOT_Y + BH / 2, ORANGE)
    arrow(ax, xr(3), BOT_Y + BH / 2, xl(4), BOT_Y + BH / 2, ORANGE)
    # the injection model feeds the calibration; the envelope feeds the prediction
    arrow(ax, xm(1), TOP_Y, xm(1), BOT_Y + BH, MUTED, ls="--")
    arrow(ax, xm(3), TOP_Y, xm(3), BOT_Y + BH, MUTED, ls="--")

    ax.text(W_IN / 2, TOP_Y + BH + 0.08, "characterisation by controlled injection",
            ha="center", va="bottom", fontsize=8, color=BLUE, style="italic")
    ax.text(W_IN / 2, BOT_Y - 0.07, "validation against real faults",
            ha="center", va="top", fontsize=8, color=ORANGE, style="italic")

    problems = check(fig, "fig_schematic", contained=pairs)
    if problems:
        raise SystemExit("fig_schematic: layout problems, figure not written")
    fig.savefig(FIG / "fig_schematic.pdf")          # no tight cropping: exact size
    fig.savefig(FIG / "fig_schematic.png", dpi=220)
    plt.close(fig)
    print("wrote figures/fig_schematic.pdf (+ .png), 6.47 x 2.95 in")


if __name__ == "__main__":
    main()
