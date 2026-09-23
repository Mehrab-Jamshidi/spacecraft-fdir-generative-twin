"""
================================================================================
WP11 — Uncertainty on the comparative claims, canonical execution
================================================================================
Two comparative claims rest on differences between Spearman coefficients
computed on the same 20 channels. With n = 20 such differences are noisy, so the
text may state a difference as established only if its paired bootstrap interval
excludes zero. This script supplies those intervals.

  1  Fault models: rho(parametric family) - rho(learned), paired over channels.
  2  Severity statistic: rho(predictive) - rho(distributional) on the admissible
     subset (the full-sample difference is in wp2_metric_robustness).
  3  The admissible-subset validity coefficients themselves.

INPUTS   ../results/wp7_parametric_robustness_runC.csv
         ../results/wp2_severity_metric_comparison_runC.csv
OUTPUT   ../results/wp11_uncertainty.json
================================================================================
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

RES = Path(__file__).resolve().parent.parent / "results"
N_BOOT, SEED = 10_000, 42
OUT = {}


def rho(x, y):
    return spearmanr(x, y).statistic


def paired_boot(df, a, b, obs="observed"):
    rng = np.random.default_rng(SEED)
    n = len(df)
    A, B, Y = df[a].to_numpy(float), df[b].to_numpy(float), df[obs].to_numpy(float)
    diffs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        ra, rb = rho(A[i], Y[i]), rho(B[i], Y[i])
        if np.isfinite(ra) and np.isfinite(rb):
            diffs.append(ra - rb)
    diffs = np.array(diffs)
    return dict(observed=float(rho(A, Y) - rho(B, Y)),
                ci_lo=float(np.percentile(diffs, 2.5)),
                ci_hi=float(np.percentile(diffs, 97.5)),
                p_gt0=float((diffs > 0).mean()))


def main():
    par = pd.read_csv(RES / "wp7_parametric_robustness_runC.csv")
    print("FAULT MODELS — paired bootstrap of rho(family) - rho(learned), n =", len(par))
    for fam in ("step", "spike", "ramp", "noise"):
        r = paired_boot(par, f"pred_{fam}", "pred_generative")
        OUT[f"faultmodel_{fam}_minus_learned"] = r
        print(f"  {fam:<6} {r['observed']:+.3f}  95% CI [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]"
              f"  P(>0) = {r['p_gt0']:.3f}")

    sev = pd.read_csv(RES / "wp2_severity_metric_comparison_runC.csv")
    adm = sev[sev.admissible].dropna(subset=["predicted_f1_ar", "predicted_f1_dist", "observed_f1"])
    for col, tag in (("predicted_f1_dist", "dist"), ("predicted_f1_ar", "pred")):
        s = spearmanr(adm[col], adm.observed_f1)
        OUT[f"admissible_{tag}"] = dict(rho=float(s.statistic), p=float(s.pvalue), n=len(adm),
                                        mae=float((adm[col] - adm.observed_f1).abs().mean()),
                                        bias=float((adm[col] - adm.observed_f1).mean()))
        print(f"  admissible {tag}: rho {s.statistic:+.3f} (p={s.pvalue:.3f})  n={len(adm)}"
              f"  MAE {OUT[f'admissible_{tag}']['mae']:.3f}  bias {OUT[f'admissible_{tag}']['bias']:+.3f}")
    r = paired_boot(adm.rename(columns={"observed_f1": "observed"}),
                    "predicted_f1_ar", "predicted_f1_dist")
    OUT["admissible_pred_minus_dist"] = r
    print(f"  admissible pred - dist: {r['observed']:+.3f}  CI [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]")

    # contextual placement under each statistic
    ctx = sev[sev.cls == "contextual"]
    OUT["ctx_alpha_hat_ar"] = dict(zip(ctx.chan_id, ctx.alpha_hat_ar.round(3)))
    OUT["ctx_alpha_hat_dist"] = dict(zip(ctx.chan_id, ctx.alpha_hat_dist.round(3)))
    n_bound = int(((ctx.alpha_hat_ar <= 0.1001) | (ctx.alpha_hat_ar >= 1.4999)).sum())
    OUT["ctx_boundary_placements_ar"] = n_bound
    print("  contextual alpha_hat (predictive):", OUT["ctx_alpha_hat_ar"], " boundary:", n_bound)

    (RES / "wp11_uncertainty.json").write_text(json.dumps(OUT, indent=1))
    print("wrote wp11_uncertainty.json")


if __name__ == "__main__":
    main()
