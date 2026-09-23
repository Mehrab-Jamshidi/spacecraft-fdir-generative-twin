"""
================================================================================
WP7 — Does the envelope add anything beyond "some channels are easier"?
================================================================================
Paper:  Detectability of learned spacecraft anomaly detectors: minimum-
        detectable-fault envelopes by controlled fault injection
Author: Mehrab Jamshidi — Politecnico di Milano

THE OBJECTION THIS ANSWERS
   The predictive-validity test in Section 5.8 predicts a real fault's detection
   by reading that channel's OWN envelope at the fault's estimated (alpha, d).
   The observed value is the same detector on the same channel. So a channel on
   which the detector is simply good will tend to score high on both sides, and
   a rank correlation across channels could be produced entirely by
   between-channel difficulty, with the severity and duration axes contributing
   nothing at all.

   If that were so, the claim "the envelope predicts real-fault detection" would
   collapse into "some channels are easier than others", which is not a claim
   about the envelope and would not support using it as a design instrument.
   A reviewer will raise this, and it has to be measured rather than argued.

THE TEST
   Three predictors are compared on the same 20 channels and the same observed
   values:

     SPECIFIC   the channel's envelope read at the real fault's own
                (alpha_hat, d_real) — the paper's actual predictor;
     CHANNEL    the channel's envelope averaged over the whole grid — carries
                channel difficulty and NOTHING about where the fault sits;
     GRIDMEAN   the pooled envelope read at (alpha_hat, d_real) but averaged
                across all channels — carries the shape of the envelope and
                nothing channel-specific.

   If SPECIFIC does not beat CHANNEL, the axes add nothing and the validity
   claim must be withdrawn. If SPECIFIC beats both, the prediction needs both
   the channel and the operating point, which is what the instrument asserts.

   A partial Spearman correlation is also reported: the association between the
   specific prediction and the observation, controlling for the channel-level
   baseline. This is the cleanest single number for the objection.

OUTPUT   ../results/wp7_circularity.csv
================================================================================
"""

from __future__ import annotations

from pathlib import Path

import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
from _paths import REPO  # noqa: E402

ALPHAS = [0.10, 0.25, 0.50, 0.75, 1.00, 1.50]
DURATIONS = [16, 32, 64, 128, 256]


def bilinear(piv: pd.DataFrame, a: float, d: float) -> float:
    A = [x for x in ALPHAS if x in piv.index]
    D = [x for x in DURATIONS if x in piv.columns]
    ac, dc = min(max(a, A[0]), A[-1]), min(max(d, D[0]), D[-1])

    def br(v, g):
        if v <= g[0]:
            return 0, 0, 0.0
        if v >= g[-1]:
            return len(g) - 1, len(g) - 1, 0.0
        i = int(np.searchsorted(g, v)) - 1
        return i, i + 1, (v - g[i]) / (g[i + 1] - g[i])

    i0, i1, ta = br(ac, A)
    j0, j1, td = br(dc, D)
    return float(piv.iloc[i0, j0] * (1 - ta) * (1 - td) + piv.iloc[i1, j0] * ta * (1 - td)
                 + piv.iloc[i0, j1] * (1 - ta) * td + piv.iloc[i1, j1] * ta * td)


def partial_spearman(x, y, z):
    """Spearman association between x and y controlling for z, via residuals of
    the rank-transformed variables."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    Z = np.column_stack([np.ones_like(rz), rz])
    bx = np.linalg.lstsq(Z, rx, rcond=None)[0]
    by = np.linalg.lstsq(Z, ry, rcond=None)[0]
    ex, ey = rx - Z @ bx, ry - Z @ by
    if ex.std() < 1e-12 or ey.std() < 1e-12:
        return np.nan, np.nan
    from scipy.stats import pearsonr
    return pearsonr(ex, ey)


def _canonical() -> bool:
    return os.environ.get("ENVELOPE_SOURCE", "canonical").lower() in (
        "canonical", "wp8", "c")


def main():
    if _canonical():
        can = pd.read_csv(RES / "wp8_canonical_envelope.csv")
        per = can[(can.detector == "lstm") & (can.fault_model.isin(["learned", "none"]))
                  ].rename(columns={"f1_pa": "f1"})
        sev = pd.read_csv(RES / "wp2_severity_metric_comparison_runC.csv")
        anom = pd.read_csv(RES / "wp2_validity_per_anomaly_runC.csv")
        print("envelope source: run C (wp8_canonical_envelope.csv)")
    else:
        per = pd.read_csv(REPO / "results" / "phase4sens_per_channel.csv")
        sev = pd.read_csv(RES / "wp2_severity_metric_comparison.csv")
        anom = pd.read_csv(RES / "wp2_validity_per_anomaly.csv")
        print("envelope source: run A (phase4sens_per_channel.csv)")

    # Pooled envelope across channels: the shape, with no channel identity.
    pooled = (per[per.duration > 0]
              .groupby(["alpha", "duration"]).f1.mean().unstack())

    rows = []
    for r in sev.itertuples():
        ch = r.chan_id
        sub = per[(per.chan_id == ch) & (per.duration > 0)]
        piv = sub.pivot_table(index="alpha", columns="duration", values="f1")
        an = anom[anom.chan_id == ch]
        if an.empty or piv.empty:
            continue
        w = an.d_real.to_numpy(dtype=float)
        d_eff = float(np.average(an.d_real, weights=w))

        rows.append(dict(
            chan_id=ch, cls=r.cls, observed=r.observed_f1,
            # the paper's predictor, under the predictive severity metric
            specific=r.predicted_f1_ar,
            # channel difficulty only: the channel's whole-grid mean
            channel=float(np.nanmean(piv.to_numpy())),
            # envelope shape only: pooled grid at this operating point
            gridmean=bilinear(pooled, r.alpha_hat_ar, d_eff),
        ))

    df = pd.DataFrame(rows).dropna()
    df.to_csv(RES / f"wp7_circularity{'_runC' if _canonical() else ''}.csv", index=False)

    print("=" * 76)
    print("DOES THE ENVELOPE ADD ANYTHING BEYOND CHANNEL DIFFICULTY?")
    print("=" * 76)
    print(f"  channels: {len(df)}\n")

    print(f"  {'predictor':<38}{'rho':>8}{'p':>9}{'MAE':>9}")
    for name, col, desc in (
            ("SPECIFIC  envelope at (alpha,d)", "specific", ""),
            ("CHANNEL   channel grid mean only", "channel", ""),
            ("GRIDMEAN  pooled envelope at (alpha,d)", "gridmean", "")):
        rho, p = spearmanr(df[col], df.observed)
        mae = (df[col] - df.observed).abs().mean()
        print(f"  {name:<38}{rho:>+8.3f}{p:>9.3f}{mae:>9.3f}")

    print()
    r_pc, p_pc = partial_spearman(df.specific, df.observed, df.channel)
    print(f"  partial rho (specific vs observed | channel difficulty): "
          f"{r_pc:+.3f}  (p={p_pc:.3f})")
    r_pg, p_pg = partial_spearman(df.specific, df.observed, df.gridmean)
    print(f"  partial rho (specific vs observed | envelope shape)    : "
          f"{r_pg:+.3f}  (p={p_pg:.3f})")

    rs = spearmanr(df.specific, df.observed).statistic
    rc = spearmanr(df.channel, df.observed).statistic
    print()
    if rs > rc + 0.05 and np.isfinite(r_pc) and r_pc > 0.3:
        v = "the operating point carries information beyond channel difficulty"
    elif rs <= rc + 0.05:
        v = ("WITHDRAW: the channel baseline predicts as well as the envelope; "
             "the axes add nothing")
    else:
        v = "AMBIGUOUS — report both and bound the claim accordingly"
    print(f"  VERDICT: {v}")

    print("\n  by class (specific vs channel baseline):")
    for c in ("point", "contextual"):
        s = df[df.cls == c]
        if len(s) < 4:
            print(f"    {c:<12} n={len(s)} too few")
            continue
        a = spearmanr(s.specific, s.observed).statistic
        b = spearmanr(s.channel, s.observed).statistic
        print(f"    {c:<12} n={len(s):>2}  specific {a:+.3f}   channel {b:+.3f}   "
              f"difference {a-b:+.3f}")

    print(f"\nwritten -> {RES / 'wp7_circularity.csv'}")


if __name__ == "__main__":
    main()
