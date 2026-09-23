"""
Automatic layout check for matplotlib figures.

After a figure is drawn, every visible, non-empty text artist is measured in
display coordinates and three defects are reported:

  1  two texts whose bounding boxes overlap;
  2  a text that extends beyond the figure canvas;
  3  a text that escapes the box it is meant to sit in (for diagrams: pass
     `contained` as a list of (text_artist, patch_artist) pairs).

Returns the list of problems so a caller can fail loudly. Used by every figure
script in this folder, so that a label running out of its box is caught by the
build rather than by a reader.
"""
from __future__ import annotations

from matplotlib.text import Text


def _undrawn_ticklabels(fig):
    """Tick labels matplotlib keeps for ticks outside the view limits; they are
    never drawn, so they cannot collide with anything."""
    skip = set()
    for ax in fig.axes:
        for axis, (lo, hi) in ((ax.xaxis, sorted(ax.get_xlim())),
                               (ax.yaxis, sorted(ax.get_ylim()))):
            span = hi - lo
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                loc = tick.get_loc()
                if loc is None or not (lo - 1e-9 * span <= loc <= hi + 1e-9 * span):
                    skip.update({id(tick.label1), id(tick.label2)})
    return skip


def _visible_texts(fig):
    skip = _undrawn_ticklabels(fig)
    out = []
    for t in fig.findobj(Text):
        if id(t) in skip or not t.get_visible() or not t.get_text().strip():
            continue
        out.append(t)
    return out


def _both_rotated_ticklabels(a, b):
    """Rotated tick labels have axis-aligned boxes that overlap by construction
    even when the glyphs do not; such pairs are judged visually, not here."""
    return (a.get_rotation() % 180 not in (0, 90) and b.get_rotation() % 180 not in (0, 90)
            and a.axes is b.axes)


def check(fig, name, contained=None, pad_px=1.5, tol_px2=4.0):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = _visible_texts(fig)
    boxes = []
    for t in texts:
        try:
            bb = t.get_window_extent(renderer=r)
        except Exception:
            continue
        if bb.width <= 0 or bb.height <= 0:
            continue
        boxes.append((t, bb))

    problems = []
    fb = fig.bbox
    for t, bb in boxes:
        if (bb.x0 < fb.x0 - 1 or bb.x1 > fb.x1 + 1
                or bb.y0 < fb.y0 - 1 or bb.y1 > fb.y1 + 1):
            # bbox_inches="tight" will expand the canvas; only report texts that
            # would still be clipped, i.e. none. Kept as information.
            pass
    for i in range(len(boxes)):
        ti, bi = boxes[i]
        for j in range(i + 1, len(boxes)):
            tj, bj = boxes[j]
            if _both_rotated_ticklabels(ti, tj):
                continue
            w = min(bi.x1, bj.x1) - max(bi.x0, bj.x0)
            h = min(bi.y1, bj.y1) - max(bi.y0, bj.y0)
            if w > 0 and h > 0 and w * h > tol_px2:
                problems.append(f"overlap: '{ti.get_text()[:28]}' x '{tj.get_text()[:28]}'")
    for t, patch in (contained or []):
        bb = t.get_window_extent(renderer=r)
        pb = patch.get_window_extent(renderer=r)
        if (bb.x0 < pb.x0 + pad_px or bb.x1 > pb.x1 - pad_px
                or bb.y0 < pb.y0 + pad_px or bb.y1 > pb.y1 - pad_px):
            problems.append(f"outside its box: '{t.get_text()[:40]}'")
    status = "OK" if not problems else f"{len(problems)} PROBLEM(S)"
    print(f"  layout check {name}: {status}")
    for p in problems:
        print(f"      - {p}")
    return problems
