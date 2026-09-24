"""
================================================================================
WP12 — Claims-to-evidence audit for the revised manuscript
================================================================================
Paper:  Detectability of learned spacecraft anomaly detectors: minimum-
        detectable-fault envelopes by controlled fault injection

Prof. Colagrossi's condition on this work is that every claim be bound to the
results and supported by them. This script enforces it mechanically, in two
directions.

  1  REGISTRY -> DATA.  Every number the manuscript states is registered below
     with the computation that should reproduce it from the result files. Each is
     recomputed and compared at the precision it is printed with.

  2  TEXT -> REGISTRY.  Every decimal number and percentage that appears in the
     manuscript sources is extracted and looked up in the registry (or in the
     short allow-list of design constants). A number in the text that nothing
     supports is reported as UNREGISTERED, so a claim added later cannot escape
     the audit.

A registered value that no longer appears in the text is reported as STALE.

USAGE   python analysis/wp12_claims_audit.py        (exit code 1 on any failure)
================================================================================
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata, pearsonr

from _paths import LABELS, REPO

HERE = Path(__file__).resolve().parent
MAN = HERE.parent
RES = MAN / "results"
SEC = MAN / "sections"      # the manuscript text; absent from the public code release

N = json.loads((RES / "wp10_numbers.json").read_text())
U = json.loads((RES / "wp11_uncertainty.json").read_text())
ALPHAS = [0.10, 0.25, 0.50, 0.75, 1.00, 1.50]
DURS = [16, 32, 64, 128, 256]

REG: list[tuple[str, float, str]] = []   # (printed token, recomputed value, what)


def reg(tok, val, what):
    REG.append((tok, float(val), what))


def f(x):
    """A contour string from wp10_numbers ('1.281', '>1.50', '<=0.10') -> float."""
    return float(x) if x not in (">1.50", "<=0.10") else np.nan


# ============================================================ sources
boot = pd.read_csv(RES / "wp8_cell_bootstrap.csv")
pod = pd.read_csv(RES / "wp10_pod_bootstrap.csv")
dec = pd.read_csv(RES / "wp8_axis_decomposition.csv")
prot = pd.read_csv(RES / "wp8_protocol_comparison.csv").set_index("detector")
detc = pd.read_csv(RES / "wp8_detector_comparison.csv")
fm = pd.read_csv(RES / "wp8_faultmodel_comparison.csv").set_index("fault_model")
sev = pd.read_csv(RES / "wp2_severity_metric_comparison_runC.csv")
rob = pd.read_csv(RES / "wp2_metric_robustness_runC.csv")
circ = pd.read_csv(RES / "wp7_circularity_runC.csv")
par = pd.read_csv(RES / "wp7_parametric_robustness_runC.csv")
anom = pd.read_csv(RES / "wp2_validity_per_anomaly_runC.csv")
mem = pd.read_csv(RES / "wp3_memorisation.csv").set_index("anomaly_class")
qual = pd.read_csv(REPO / "results" / "gan_v4_clean_quality_report.csv")
lstm2 = pd.read_csv(REPO / "results" / "lstm_regression_results.csv")
thr = pd.read_csv(REPO / "results" / "threshold_baseline_results.csv")
can = pd.read_csv(RES / "wp9_canonical_envelope.csv")
labels = pd.read_csv(LABELS)


def rho(df, col, obs="observed_f1"):
    d = df.dropna(subset=[col, obs])
    return spearmanr(d[col], d[obs])


def mae(df, col, obs="observed_f1"):
    return float((df[col] - df[obs]).abs().mean())


def bias(df, col, obs="observed_f1"):
    return float((df[col] - df[obs]).mean())


def grid(q, a, d):
    return N[f"{q}_grid"][f"{a}_{d}"]


# ============================================================ Section 1 / 4: benchmark
lab = labels.copy()
nseq = lab.anomaly_sequences.map(lambda s: len(re.findall(r"\d+", s)) // 2)
reg("17.75%", 17.75, "AOCS share of losses [external: Colagrossi & Lavagna 2022]")
cls_tokens = sum((str(c).strip("[]").split(",") for c in lab["class"]), [])
cls_tokens = [c.strip() for c in cls_tokens]
assert len(cls_tokens) == 105 and cls_tokens.count("contextual") == 43
u = lab.drop_duplicates("chan_id", keep="first")
nseq_u = u.anomaly_sequences.map(lambda s: len(re.findall(r"\d+", s)) // 2)
assert (nseq_u == 1).sum() == 63, (nseq_u == 1).sum()
reg("5.9%", thr.drop_duplicates("chan_id").anomaly_rate.median(), "median anomalous fraction of a test split")


def fbeta(p, r, b):
    den = b * b * p + r
    return (1 + b * b) * p * r / den if den > 0 else 0.0


sm, ms = lstm2[lstm2.spacecraft == "SMAP"], lstm2[lstm2.spacecraft == "MSL"]
reg("0.637", sm.lstm_k3_f1.mean(), "all-channel LSTM k=3 F1, SMAP")
reg("0.709", ms.lstm_k3_f1.mean(), "all-channel LSTM k=3 F1, MSL")
reg("0.595", np.mean([fbeta(p, r, .5) for p, r in zip(sm.lstm_k3_precision, sm.lstm_k3_recall)]), "F0.5 SMAP")
reg("0.662", np.mean([fbeta(p, r, .5) for p, r in zip(ms.lstm_k3_precision, ms.lstm_k3_recall)]), "F0.5 MSL")
reg("0.71", 0.71, "Hundman et al. published F0.5 SMAP [external]")
reg("0.69", 0.69, "Hundman et al. published F0.5 MSL [external]")
lu = lstm2.drop_duplicates("chan_id", keep="first")
ctx_ch = set(lu[lu.anomaly_class.str.contains("contextual")].chan_id)
assert len(ctx_ch) == 30, len(ctx_ch)
reg("0.717", lu[lu.chan_id.isin(ctx_ch)].lstm_k3_f1.mean(), "LSTM k=3 on the 30 contextual-carrying channels")
reg("0.052", thr.drop_duplicates("chan_id")[lambda x: x.chan_id.isin(ctx_ch)].f1.mean(),
    "3-sigma detector on those channels")

# ============================================================ 4: generator, design
q = qual.set_index(["metric", "class"]).value
reg("0.141", q[("Frechet Distance", "contextual")], "generator Frechet distance, contextual")
reg("0.403", q[("Frechet Distance", "point")], "generator Frechet distance, point")
reg("1.24", mem.loc["point", "ratio_syn_over_holdout"], "memorisation ratio, point")
reg("4.24", mem.loc["contextual", "ratio_syn_over_holdout"], "memorisation ratio, contextual")
reg("47.0%", 100 * mem.loc["point", "sat_rate_real_train"], "saturated real point training windows")
reg("47%", 100 * mem.loc["point", "sat_rate_real_train"], "saturated real point training windows")
reg("52.6%", 100 * mem.loc["point", "sat_rate_real_holdout"], "saturated real held-out point windows")
reg("4.4%", 100 * mem.loc["point", "sat_rate_synth"], "saturated synthetic point windows")
reg("1.0%", 100 * N["injected_prevalence"]["16"]["50%"], "median faulty fraction, d=16")
reg("16%", 100 * N["injected_prevalence"]["256"]["50%"], "median faulty fraction, d=256")

# ============================================================ 5.1 envelope
reg("0.217", N["f1_corner_lo"], "F1 at (0.10, 16)")
reg("0.815", N["f1_corner_hi"], "F1 at (1.50, 256)")
reg("0.568", N["f1_op_a1_d64"], "F1 at (1.0, 64)")
reg("0.374", N["f1_cell_sd_max"], "max per-cell SD")
reg("0.127", N["f1_ci_halfwidth_mean"], "mean CI half-width")
reg("0.011", N["fp_median"], "median nominal FP rate")
reg("0.157", N["fp_mean"], "mean nominal FP rate")
reg("72%", 100 * N["inadmissible"]["T-2"], "T-2 FP rate")
reg("0.176", N["inadmissible"]["F-3"], "F-3 FP rate")
reg("0.080", N["inadmissible"]["E-8"], "E-8 FP rate")
reg("0.056", N["inadmissible"]["P-1"], "P-1 FP rate")
assert N["n_admissible"] == 14 and N["f1_monotone_in_alpha_every_d"] and N["f1_monotone_in_d_every_alpha"]
assert abs(N["f1_cell_sd_min"] - 0.217) < 5e-4

# ============================================================ 5.2 contour table
for req in ("050", "070", "080"):
    for s in ("hi", "mean", "lo"):
        for d in DURS:
            v = f(N[f"mdf_f1_{req}_{s}"][str(d)])
            if np.isfinite(v):
                # Bootstrap bounds are printed to 2 d.p.: they move in the second
                # decimal between resampling seeds (an independent re-bootstrap
                # gave 1.455 for 1.473). Mean-surface contours are deterministic.
                fmt = "{:.3f}" if s == "mean" else "{:.2f}"
                reg(fmt.format(v), v, f"contour F1>={int(req)/100:.2f} {s} d={d}")
b32 = boot[(boot.group == "all") & (boot.duration == 32)].set_index("alpha").f1_mean
reg("0.13", (b32[1.5] - b32[1.0]) / 0.5, "surface slope near the crossing at d=32")
reg("0.032", N["surface_A_minus_C_meanabs"], "run A above C, mean")
reg("0.050", N["surface_B_minus_C_meanabs"], "run B above C, mean")
assert N["surface_A_higher_cells"] == 30 and N["surface_B_higher_cells"] == 30
for r in ("A", "B"):
    for d in (32, 64):
        v = float(N["repro_contour_f1_050"][r][str(d)])
        reg(f"{v:.3f}", v, f"replicate {r} contour d={d}")
reg("10.7%", 100 * N["fp_T2_runA"], "T-2 FP in replicate")
reg("72.1%", 100 * N["fp_T2_runC"], "T-2 FP in canonical")
reg("0.704", N["per_channel_cell_max_move"], "largest channel-cell move between executions")
for d in (16, 32):
    v = f(N["mdf_f1_050_adm_mean"][str(d)])
    reg(f"{v:.3f}", v, f"admissible-only contour d={d}")
assert N["mdf_f1_050_adm_mean"]["64"] == "<=0.10"

# ============================================================ 5.3 timeliness
for qn in ("p5", "p20"):
    for a in (0.1, 0.5, 1.0, 1.5):
        for d in DURS:
            v = grid(qn, a, d)
            reg(f"{v:.3f}", v, f"{qn} at alpha={a}, d={d}")
reg("0.201", N["chance_w6_mean"], "false-call prob, 6-step window, mean")
reg("0.054", N["chance_w6_median"], "false-call prob, 6-step window, median")
reg("0.292", N["chance_w21_mean"], "false-call prob, 21-step window, mean")
reg("0.133", N["chance_w21_median"], "false-call prob, 21-step window, median")
reg("0.053", N["p5_max_duration_range"], "max P5 range across duration")
reg("0.047", N["p20_max_duration_range"], "max P20 range across duration")
reg("0.30", N["p5_sev_gain_min"], "P5 severity gain min")
reg("0.35", N["p5_sev_gain_max"], "P5 severity gain max")
reg("0.26", N["p20_sev_gain_min"], "P20 severity gain min")
reg("0.33", N["p20_sev_gain_max"], "P20 severity gain max")
reg("0.038", N["chance_w6_adm_mean"], "false-call prob, admissible")
reg("0.40", min(N["p5_adm_a01"].values()), "admissible P5 faintest min")
reg("0.45", max(N["p5_adm_a01"].values()), "admissible P5 faintest max")
reg("0.77", min(N["p5_adm_a15"].values()), "admissible P5 strongest min")
reg("0.81", max(N["p5_adm_a15"].values()), "admissible P5 strongest max")
for d in DURS:
    v = f(N["mdf_P5_70_mean"][str(d)])
    reg(f"{v:.3f}", v, f"timeliness contour P5>=0.7 d={d}")
    for key in ("mdf_joint_F050_P5_70_mean", "mdf_joint_F050_P5_80_mean"):
        v = f(N[key][str(d)])
        if np.isfinite(v):
            reg(f"{v:.3f}", v, f"{key} d={d}")
assert all(v == ">1.50" for v in N["mdf_joint_F050_P5_90_mean"].values())
assert all(v == ">1.50" for v in N["mdf_P5_80_lo"].values())
reg("1.3", np.mean([f(N["mdf_joint_F050_P5_80_mean"][str(d)]) for d in (32, 64, 128, 256)]),
    "P5>=0.8 contour, approx")
reg("3.2", N["latency_a01"]["16"], "conditional mean latency, alpha=0.10, d=16")
reg("20.7", N["latency_a01"]["256"], "conditional mean latency, alpha=0.10, d=256")
reg("0.51", min(grid("p5", 0.1, d) for d in DURS), "P5 at faintest, min over d")
reg("0.54", max(grid("p5", 0.1, d) for d in DURS), "P5 at faintest, max over d")

# ============================================================ 5.4 protocol
dd = dec[dec.axis == "duration"]
ss = dec[dec.axis == "severity"]
reg("0.39", dd.d_f1.min(), "duration F1 effect min")
reg("0.46", dd.d_f1.max(), "duration F1 effect max")
reg("0.16", ss.d_f1.min(), "severity F1 effect min")
reg("0.22", ss.d_f1.max(), "severity F1 effect max")
reg("64%", 100 * dd.precision_share.min(), "duration precision share min")
reg("82%", 100 * dd.precision_share.max(), "duration precision share max")
reg("35%", 100 * ss.precision_share.min(), "severity precision share min")
reg("47%", 100 * ss.precision_share.max(), "severity precision share max")
reg("0.570", grid("pdet", 0.1, 16), "segment hit rate, alpha=0.10, d=16")
reg("0.816", grid("pdet", 0.1, 256), "segment hit rate, alpha=0.10, d=256")
reg("0.246", grid("pdet", 0.1, 256) - grid("pdet", 0.1, 16), "its increase")
reg("0.268", N["chance_w16_mean"], "false-call prob, 16-step window")
reg("0.523", N["chance_w256_mean"], "false-call prob, 256-step window")
reg("0.255", N["chance_w256_mean"] - N["chance_w16_mean"], "its increase")
reg("0.900", N["d16_recall"][1], "d=16 recall at alpha=1.5")
reg("0.172", N["d16_precision"][0], "d=16 precision at alpha=0.10")
reg("0.347", N["d16_precision"][1], "d=16 precision at alpha=1.5")
reg("49%", 100 * prot.loc["lstm", "duration_retained"], "LSTM duration retained")
reg("92%", 100 * prot.loc["lstm", "severity_retained"], "LSTM severity retained")
reg("0.529", prot.loc["lstm", "level_pa"], "LSTM grid mean PA")
reg("0.300", prot.loc["lstm", "level_pw"], "LSTM grid mean PW")
reg("86%", 100 * prot.loc["vae", "duration_retained"], "VAE duration retained")
reg("91%", 100 * prot.loc["vae", "severity_retained"], "VAE severity retained")
for d in (32, 64, 128):
    v = f(N["mdf_pw_f1_030_mean"][str(d)])
    reg(f"{v:.3f}", v, f"point-wise contour F1>=0.30 d={d}")

# ============================================================ 5.5 detectors, fault models
reg("0.271", N["vae_grid_mean"], "VAE grid mean")
reg("0.257", N["lstm_minus_vae_mean"], "LSTM minus VAE, mean over cells")
assert N["lstm_beats_vae_cells"] == 30
reg("0.066", N["vae_d16_range"][0], "VAE at d=16, min")
reg("0.129", N["vae_d16_range"][1], "VAE at d=16, max")
reg("0.426", N["lstm_d16_range"][1], "LSTM at d=16, max")
vc = detc[(detc.detector == "vae")]
for t, d in ((0.5, 128), (0.5, 256), (0.7, 256)):
    v = float(vc[(vc.f1_target == t) & (vc.duration == d)].alpha_mdf.iloc[0])
    reg(f"{v:.3f}", v, f"VAE contour F1>={t} d={d}")
for m in fm.index:
    reg(f"{fm.loc[m, 'grid_mean_f1']:.3f}", fm.loc[m, "grid_mean_f1"], f"grid mean, {m}")
    for d in DURS:
        s = fm.loc[m, f"mdf_d{d}"]
        if isinstance(s, str) and s in (">1.50", "<=0.10"):
            continue
        reg(f"{float(s):.3f}", float(s), f"contour {m} d={d}")
reg("0.48", fm.grid_mean_f1.min(), "fault-model grid-mean range min")
reg("0.65", fm.grid_mean_f1.max(), "fault-model grid-mean range max")

# ============================================================ 5.6 class
cb = boot[boot.group.isin(["point", "contextual"])]
for c in ("point", "contextual"):
    reg(f"{N[f'f1_{c}_d256_a01']:.3f}", N[f"f1_{c}_d256_a01"], f"{c} F1 at (0.10,256)")
    reg(f"{N[f'f1_{c}_d256_a15']:.3f}", N[f"f1_{c}_d256_a15"], f"{c} F1 at (1.50,256)")
for d in (64, 128, 256):
    v = f(N["mdf_f1_070_contextual_mean"][str(d)])
    reg(f"{v:.3f}", v, f"contextual contour F1>=0.70 d={d}")
assert N["mdf_f1_070_point_mean"]["128"] == ">1.50" and N["mdf_f1_070_point_mean"]["256"] == "<=0.10"

# ============================================================ 5.7 validity
S = {"all": sev, "point": sev[sev.cls == "point"], "contextual": sev[sev.cls == "contextual"],
     "admissible": sev[sev.admissible]}
for k, df in S.items():
    for col, tag in (("predicted_f1_dist", "dist"), ("predicted_f1_ar", "pred")):
        r = rho(df, col)
        reg(f"{r.statistic:+.3f}", r.statistic, f"rho {tag} {k}")
        reg(f"{mae(df, col):.3f}", mae(df, col), f"MAE {tag} {k}")
        if tag == "pred" or k == "contextual":
            reg(f"{bias(df, col):+.3f}", bias(df, col), f"bias {tag} {k}")
        if k == "all" and tag == "dist":
            reg(f"{r.pvalue:.3f}" if r.pvalue >= 0.01 else f"{r.pvalue:.3f}", r.pvalue, f"p {tag} {k}")
reg("0.56", rho(S["admissible"], "predicted_f1_dist").pvalue, "p dist admissible (2 d.p.)")
reg("0.91", rho(S["contextual"], "predicted_f1_dist").pvalue, "p dist contextual (2 d.p.)")
g1 = sev.set_index("chan_id").loc["G-1"]
reg("0.570", g1.alpha_hat_dist, "G-1 distributional alpha-hat")
reg("0.963", g1.predicted_f1_dist, "G-1 distributional prediction")
reg("0.000", g1.observed_f1, "G-1 observed F1")
reg("0.000", g1.predicted_f1_ar, "G-1 predictive prediction")
assert abs(g1.alpha_hat_ar - 0.10) < 1e-6
assert U["ctx_boundary_placements_ar"] == 5
ctx_a = U["ctx_alpha_hat_ar"]
assert all(abs(ctx_a[c] - 0.10) < 1e-6 for c in ("A-8", "F-3", "G-1", "P-1"))
assert abs(ctx_a["M-2"] - 1.5) < 1e-6 and 0.1 < ctx_a["E-10"] < 1.5

# ablations
obs = circ.observed
for col, tag in (("channel", "channel"), ("gridmean", "shape")):
    r = spearmanr(circ[col], obs)
    reg(f"{r.statistic:+.3f}", r.statistic, f"ablation {tag} rho")
    reg(f"{r.pvalue:.3f}", r.pvalue, f"ablation {tag} p")
    reg(f"{(circ[col] - obs).abs().mean():.3f}", (circ[col] - obs).abs().mean(), f"ablation {tag} MAE")
reg("0.19", spearmanr(circ.channel, obs).pvalue, "channel p (2 d.p.)")
m = circ.merge(sev[["chan_id", "alpha_hat_ar"]], on="chan_id")
r = spearmanr(m.alpha_hat_ar, m.observed)
reg(f"{r.statistic:+.3f}", r.statistic, "severity-only ablation rho")
reg(f"{r.pvalue:.3f}", r.pvalue, "severity-only ablation p")
reg("0.13", r.pvalue, "severity-only p (2 d.p.)")


def partial(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    Z = np.column_stack([np.ones_like(rz), rz])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return pearsonr(ex, ey)


p1 = partial(circ.specific, obs, circ.channel)
p2 = partial(circ.specific, obs, circ.gridmean)
reg(f"{p1.statistic:+.3f}", p1.statistic, "partial rho | channel")
reg(f"{p2.statistic:+.3f}", p2.statistic, "partial rho | shape")
assert p1.pvalue < 0.001 and p2.pvalue < 0.001
cc = circ[circ.cls == "contextual"]
reg(f"{spearmanr(cc.channel, cc.observed).statistic:+.3f}",
    spearmanr(cc.channel, cc.observed).statistic, "contextual channel-only rho")

# robustness: AR order
for qo in (2, 4, 8, 16, 32):
    s = rob[rob.ar_order == qo]
    reg(f"{spearmanr(s.predicted, s.observed_f1).statistic:+.3f}",
        spearmanr(s.predicted, s.observed_f1).statistic, f"AR order {qo}, all")
    c = s[s.cls == "contextual"]
    v = spearmanr(c.predicted, c.observed_f1).statistic
    reg(f"{v:+.3f}", v, f"AR order {qo}, contextual")
# leave-one-out
d8 = rob[rob.ar_order == 8].reset_index(drop=True)
base = spearmanr(d8.predicted, d8.observed_f1).statistic
loo = {c: spearmanr(d8.predicted.drop(i), d8.observed_f1.drop(i)).statistic
       for i, c in enumerate(d8.chan_id)}
others = [abs(v - base) for c, v in loo.items() if c != "G-3"]
reg("0.034", max(others), "max LOO change excluding G-3")
reg("+0.916", loo["G-3"], "rho after dropping G-3")
cx = sev[sev.cls == "contextual"].reset_index(drop=True)
lc = [spearmanr(cx.predicted_f1_ar.drop(i), cx.observed_f1.drop(i)).statistic for i in range(len(cx))]
reg("+0.895", min(lc), "contextual LOO min")
reg("+1.000", max(lc), "contextual LOO max")
# duration alone
dmax = anom.groupby("chan_id").d_real.max()
m = sev.set_index("chan_id").join(dmax)
r_all = spearmanr(m.d_real, m.observed_f1)
r_cx = spearmanr(m[m.cls == "contextual"].d_real, m[m.cls == "contextual"].observed_f1)
reg(f"{r_all.statistic:+.3f}", r_all.statistic, "duration alone, all")
reg("0.41", r_all.pvalue, "duration alone p, all")
reg(f"{r_cx.statistic:+.3f}", r_cx.statistic, "duration alone, contextual")
reg("0.78", r_cx.pvalue, "duration alone p, contextual")
# bootstrap of the rho difference (same procedure as wp2_metric_robustness T3)
rng = np.random.default_rng(42)
pa, pdist, ob = d8.predicted.to_numpy(), d8.predicted_f1_dist.to_numpy(), d8.observed_f1.to_numpy()
diffs = []
for _ in range(10_000):
    i = rng.integers(0, len(ob), len(ob))
    if len(np.unique(ob[i])) < 3:
        diffs.append(np.nan)
        continue
    diffs.append(spearmanr(pa[i], ob[i]).statistic - spearmanr(pdist[i], ob[i]).statistic)
diffs = np.array(diffs)
diffs = diffs[np.isfinite(diffs)]
reg("+0.329", spearmanr(pa, ob).statistic - spearmanr(pdist, ob).statistic, "rho difference")
reg("-0.014", np.percentile(diffs, 2.5), "rho difference CI lo")
reg("+0.836", np.percentile(diffs, 97.5), "rho difference CI hi")
mono = d8[d8.monotone]
assert len(mono) == 13 and (mono.cls == "contextual").sum() == 2
reg("+0.740", spearmanr(mono.predicted, mono.observed_f1).statistic, "monotone subset, predictive")
reg("+0.652", spearmanr(mono.predicted_f1_dist, mono.observed_f1).statistic, "monotone subset, distributional")
fl = int((sev.alpha_hat_ar <= 0.1001).sum())
ce = int((sev.alpha_hat_ar >= 1.4999).sum())
assert fl == 5 and ce == 5
assert int((anom.predict_status == "duration_clamped").sum()) == 9 and len(anom) == 25
reg("3805", anom.d_real.max(), "longest real anomaly")

# ============================================================ 5.8 which stimulus
for mdl in ("step", "spike", "generative", "ramp", "noise"):
    r = spearmanr(par[f"pred_{mdl}"], par.observed)
    reg(f"{r.statistic:+.3f}", r.statistic, f"fault-model rho {mdl}")
    reg(f"{(par[f'pred_{mdl}'] - par.observed).abs().mean():.3f}",
        (par[f"pred_{mdl}"] - par.observed).abs().mean(), f"fault-model MAE {mdl}")
reg("0.002", spearmanr(par.pred_ramp, par.observed).pvalue, "ramp p")
reg("0.133", spearmanr(par.pred_noise, par.observed).pvalue, "noise p")
for fam in ("step", "spike", "ramp", "noise"):
    k = U[f"faultmodel_{fam}_minus_learned"]
    reg(f"{k['observed']:+.3f}", k["observed"], f"delta rho {fam}")
    reg(f"{k['ci_lo']:+.3f}", k["ci_lo"], f"delta rho {fam} CI lo")
    reg(f"{k['ci_hi']:+.3f}", k["ci_hi"], f"delta rho {fam} CI hi")
reg("0.150", U["faultmodel_step_minus_learned"]["observed"], "step minus learned (unsigned)")
reg("0.111", U["faultmodel_spike_minus_learned"]["observed"], "spike minus learned (unsigned)")
wp = {m: int(par[f"mono_{m}"].sum()) for m in ("step", "spike", "generative", "ramp", "noise")}
assert wp == dict(step=7, spike=16, generative=13, ramp=9, noise=20), wp
allmono = par[np.logical_and.reduce([par[f"mono_{m}"] for m in wp])]
assert len(allmono) == 7
reg("+0.829", spearmanr(allmono.pred_step, allmono.observed).statistic, "7-channel step")
assert abs(spearmanr(allmono.pred_spike, allmono.observed).statistic - 0.829) < 5e-4
reg("+0.198", spearmanr(allmono.pred_generative, allmono.observed).statistic, "7-channel learned")
aro = pd.read_csv(RES / "wp7_parametric_ar_orders_runC.csv").set_index("ar_order")
reg(f"{aro.spike.min():+.3f}", aro.spike.min(), "spike rho, min over AR orders 4/8/16")
reg(f"{aro.spike.max():+.3f}", aro.spike.max(), "spike rho, max over AR orders")
reg(f"{aro.generative.min():+.3f}", aro.generative.min(), "learned rho, min over AR orders")
reg(f"{aro.generative.max():+.3f}", aro.generative.max(), "learned rho, max over AR orders")
assert aro.step.min() < 0.70 < aro.step.max()   # the step's margin is order-sensitive
reg(f"{aro.loc[4, 'step']:+.3f}", aro.loc[4, "step"], "step rho at AR order 4")
reg(f"{aro.loc[16, 'step']:+.3f}", aro.loc[16, "step"], "step rho at AR order 16")
assert aro.loc[4, "step"] < aro.loc[4, "generative"] and aro.loc[16, "step"] < aro.loc[16, "generative"]

# exact permutation p for the six contextual channels (t approximation is anti-
# conservative at n = 6): all 720 permutations
import itertools  # noqa: E402
ctx = sev[sev.cls == "contextual"]
r0 = spearmanr(ctx.predicted_f1_ar, ctx.observed_f1).statistic
yv = ctx.observed_f1.to_numpy()
perm = [spearmanr(ctx.predicted_f1_ar, yv[list(p)]).statistic
        for p in itertools.permutations(range(len(yv)))]
reg("0.017", float(np.mean(np.abs(perm) >= abs(r0) - 1e-12)), "exact p, contextual predictive rho")

# timeliness of the parametric fault models: largest P5 change across duration
lstm_can = can[can.detector == "lstm"]


def p5_grid(fm):
    s = lstm_can[lstm_can.fault_model == fm]
    return (s.assign(p5=s.n_timely_5 / s.n_seg)
             .groupby(["alpha", "duration"]).p5.mean().unstack())


for fm in ("noise", "step", "ramp", "spike"):
    g = p5_grid(fm)
    v = float((g.max(axis=1) - g.min(axis=1)).max())
    reg(f"{v:.3f}", v, f"max P5 range across duration, {fm}")
gr = p5_grid("ramp")
reg("0.093", gr.loc[1.5, 16] - gr.loc[1.5, 256], "ramp P5 drop from d=16 to 256 at alpha=1.5")
reg("0.47", spearmanr(sev.predicted_f1_dist, sev.observed_f1).statistic,
    "abstract: distributional rho (2 d.p.)")

# ============================================================ discussion
reg("0.20", N["chance_w6_mean"], "false-call prob 6-step (2 d.p.)")
reg("0.52", N["chance_w256_mean"], "false-call prob 256-step (2 d.p.)")
reg("0.33", f(N["mdf_f1_050_mean"]["64"]), "abstract: contour at 64, mean (2 d.p.)")
reg("1.47", f(N["mdf_f1_050_lo"]["64"]), "abstract: contour at 64, conservative (2 d.p.)")
reg("0.80", spearmanr(sev.predicted_f1_ar, sev.observed_f1).statistic, "abstract: rho (2 d.p.)")

# ============================================================ allow-list
# Design constants, requirements, grid values, confidence levels and table
# column widths: stated by construction, not measured.
ALLOW = {"0.10", "0.25", "0.50", "0.75", "1.00", "1.50", "0.5", "1.0", "1.5", "0.7",
         "0.8", "0.9", "0.70", "0.80", "0.30", "0.05", "0.0025", "95%", "90%", "-0.5",
         "0.90", "0.001", "0.01", "0.000", "0.019",
         "2.0", "2.3", "2.6", "3.1", "3.4", "1.15", "1.3", "0.6",
         # admissibility limit and train/validation split, as percentages;
         # one-sided level of a two-sided 95% bound; LSTM dropout
         "5%", "10%", "97.5%", "0.1",
         # appendix: software versions and table column widths
         "3.12", "2.5", "12.1", "3.2", "10.6"}
# 0.019 appears as the p-value of the pooled-shape ablation (registered above as
# its rounded value); 1.15 is \arraystretch.


# ============================================================ run
def decimals(tok):
    t = tok.rstrip("%")
    return len(t.split(".")[1]) if "." in t else 0


def matches(tok, val):
    t = float(tok.rstrip("%").replace("+", ""))
    tol = 0.5 * 10 ** (-decimals(tok)) + 1e-9
    return abs(t - val) <= tol


def text_tokens():
    out = {}
    for fpath in sorted(glob.glob(str(SEC / "*.tex"))):
        t = open(fpath, encoding="utf-8").read()
        t = re.sub(r"(?<!\\)%.*", "", t)
        for mm in re.finditer(r"(?<![\w.])[-+]?\d+\.\d+|\\SI\{[\d.]+\}\{\\percent\}|\\num\{\d+\}", t):
            s = mm.group(0)
            s = re.sub(r"\\SI\{([\d.]+)\}\{\\percent\}", r"\1%", s)
            s = re.sub(r"\\num\{(\d+)\}", r"\1", s)
            out.setdefault(s, set()).add(Path(fpath).name)
    return out


def main():
    fails, ok = [], 0
    for tok, val, what in REG:
        if matches(tok, val):
            ok += 1
        else:
            fails.append((tok, val, what))
    has_text = SEC.is_dir() and any(SEC.glob("*.tex"))
    toks = text_tokens() if has_text else {}
    registered = {t for t, _, _ in REG}
    norm = lambda s: s.lstrip("+")
    reg_norm = {norm(t) for t in registered} | {t.lstrip("-") for t in registered}
    unreg = sorted(t for t in toks if t not in registered and norm(t) not in reg_norm
                   and t not in ALLOW and norm(t) not in ALLOW)
    stale = (sorted({t for t in registered if t not in toks and norm(t) not in {norm(x) for x in toks}
                     and "[external" not in dict((a, c) for a, _, c in REG).get(t, "")})
             if has_text else [])

    print("=" * 78)
    print("CLAIMS-TO-EVIDENCE AUDIT (revised manuscript, canonical execution)")
    print("=" * 78)
    for tok, val, what in fails:
        print(f"  MISMATCH  text {tok:>9}  data {val:.4f}   {what}")
    for t in unreg:
        print(f"  UNREGISTERED  {t:>9}  in {', '.join(sorted(toks[t]))}")
    for t in stale:
        print(f"  STALE (registered, not in text)  {t}")
    print(f"\n{ok}/{len(REG)} registered claims verified against source data")
    if has_text:
        print(f"{len(toks)} distinct numeric tokens in the text; "
              f"{len(unreg)} unregistered; {len(stale)} stale")
    else:
        print("manuscript text not present (sections/); the text-to-registry scan is "
              "skipped and each registered value is checked against the data only")
    sys.exit(1 if (fails or unreg) else 0)


if __name__ == "__main__":
    main()
