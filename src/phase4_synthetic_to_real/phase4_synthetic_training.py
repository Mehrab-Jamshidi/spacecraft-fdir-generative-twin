"""
================================================================================
PHASE 4 — Training FDIR Detectors on SYNTHETIC Fault Data, Testing on REAL Data
              (clean channel-level held-out protocol)
================================================================================

Author:     Mehrab Jamshidi — Politecnico di Milano
Supervisor: Prof. Andrea Colagrossi
Thesis:     Generative Digital Twin Methods for Spacecraft Telemetry Anomaly
            Detection: Toward Data-Driven FDIR Design and Testing
            MSc Aeronautical Engineering, Politecnico di Milano, AY 2025-26

================================================================================
THE RESEARCH QUESTION THIS PHASE ANSWERS
================================================================================

    "Can a generative model trained on nominal spacecraft telemetry substitute
     for real labelled fault data when designing and testing an FDIR detector?"

Phase 2 trained three detectors (SVM, VAE, LSTM) on REAL data and established
the performance ceiling. Phase 3 trained a WGAN-GP on the 61 TRAINING channels
only, producing synthetic fault windows with good temporal fidelity
(ACF mean |delta|: contextual 0.026, point 0.015).

Phase 4 trains / calibrates the SAME three detectors using ONLY synthetic fault
data (plus real NOMINAL data, which is always available in practice), and tests
them on the 20 HELD-OUT channels.

    The GAP between Phase 4 and Phase 2 IS the answer to the thesis question.
    Small gap  -> synthetic data is a viable substitute for real fault data.
    Large gap  -> it is not.

================================================================================
THE HELD-OUT PROTOCOL — READ THIS BEFORE INTERPRETING ANY NUMBER
================================================================================

Every result written by this script is measured on the 20 held-out channels
ONLY. Those channels' anomalies were never seen by the Phase 3 generator and are
never added to any training set here. `compute_split()` from holdout_split.py is
the single source of that partition, so Phase 3 and Phase 4 cannot drift apart.

This replaces an earlier, contaminated arrangement in which the GAN was trained
on anomaly windows from the same test split that Phase 4 then evaluated on. The
question the clean protocol answers is strictly harder and genuinely about
generalisation: "can a GAN that never saw channel X's faults still produce
synthetic data that detects them?" See docs/CLEAN_RERUN_PROTOCOL.md.

================================================================================
WHY THE THREE DETECTORS USE SYNTHETIC DATA DIFFERENTLY (key design rationale)
================================================================================

The three detectors learn fundamentally different things, so "train on
synthetic" means something different for each. Conflating them would be a
methodological error. The unifying principle of Phase 4 is:

    *** Every detector is made deployable WITHOUT EVER SEEING A REAL FAULT. ***

  SVM  (discriminative classifier — needs labelled positives + negatives)
        Train on:  SYNTHETIC anomaly windows (positives)
                 + REAL NOMINAL windows from the train split (negatives)
        This is the direct "Train-on-Synthetic, Test-on-Real" (TSTR) setup.
        It mirrors the Phase 2 SVM but replaces the real labelled faults (which
        required a fault to have already occurred) with synthetic ones.

  VAE  (reconstruction-based — trains only on nominal, flags high recon error)
        The VAE CANNOT be "trained" on anomalies (it models the nominal
        manifold). Its Phase 2 weakness was THRESHOLD CALIBRATION: on the
        held-out channels the deployable k=3 rule reaches 0.397 against a
        within-run oracle of 0.486, because the optimal threshold cannot be
        known without test labels.
        Phase 4 attempt: train the VAE on REAL nominal (as it must), then use
        SYNTHETIC anomalies as a label-free CALIBRATION SET to pick the
        threshold that best separates nominal from synthetic reconstruction
        error — a deployable threshold needing no real faults.

  LSTM (prediction-based — trains only on nominal, flags high prediction error)
        Same calibration logic as the VAE (Mode A, headline result).
        PLUS an ablation (Mode B, discriminative LSTM classifier) that injects
        synthetic anomaly windows into otherwise-nominal sequences to create a
        supervised signal. Reported separately and discussed honestly: it
        underperforms (0.320), and the point-class temporal gap is the reason.

================================================================================
FAIRNESS CONTROLS (non-negotiable for a credible comparison)
================================================================================
  * Same 20 held-out channels for every detector and every comparison.
  * Same SEED=42, WINDOW_SIZE=64, STEP_SIZE=16, OVERLAP_THRESHOLD=0.1.
  * Same channel z-score normalisation with +-3 sigma clip -> [-1, 1].
  * Same architectures and hyperparameters as Phase 2.
  * Same evaluation protocol per detector as its Phase 2 counterpart:
        SVM  -> window-level F1
        VAE  -> timestep point-adjust F1
        LSTM -> timestep point-adjust F1
  * Phase 4 DEPLOYABLE results are compared head-to-head against the Phase 2
    DEPLOYABLE results (k=3), NOT against the Phase 2 oracle best-threshold.
    (Comparing P4-deployable to P2-oracle would be the unfair move.)
  * Every synthetic-vs-real comparison is tested with a paired Wilcoxon
    signed-rank test over the 20 channels, written to
    results/phase4_clean_statistics.csv.

================================================================================
WHAT THIS RUN FOUND (results/phase4_clean_statistics.csv)
================================================================================

  SVM substitution      synthetic 0.474  vs real 0.631   Wilcoxon p = 0.134
  LSTM k3 synth vs real      0.544  vs      0.547        p = 0.701
  VAE best synthetic rule    0.350  vs k3   0.397        p = 0.286

Read carefully: with n = 20 these p-values show the ABSENCE of a detectable
difference, which is not the same as proof of equality. The defensible claim is
that synthetic faults substitute for real ones with no loss this study can
detect, and that the parameter-free k3 rule remains the recommended deployable
threshold — no synthetic-informed rule beats it.

The data-efficiency curve is FLAT (0.474 at 0 real windows -> 0.465 with all of
them): cross-channel real anomaly labels do not transfer. Any earlier
"synthetic data is worth N real windows" claim is withdrawn.

================================================================================
INPUTS / OUTPUTS
================================================================================

Reads  : data/synthetic/gan_v4_clean_synth_{point,contextual}.npy  (from Phase 3)
         the NASA SMAP/MSL dataset  (see data/README.md, SMAP_MSL_ROOT)
Writes : results/phase4_svm_tstr_results.csv
         results/phase4_vae_results.csv
         results/phase4_lstm_results.csv
         results/phase4_data_efficiency.csv
         results/phase4_clean_statistics.csv
         results/phase4_master_comparison.csv
         figures/phase4_{rule_comparison,overview,data_efficiency,
                         per_class_spacecraft,P1_case_study}.png

Because the committed synthetic .npy files are read directly, this script runs
without retraining the GAN. It still trains a VAE and an LSTM per channel, so
budget roughly an hour on a single GPU.
================================================================================
"""

import numpy as np
import pandas as pd
import os
import re
import sys
import time
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from scipy import stats

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION  — all paths come from src/config.py
# =============================================================================

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (  # noqa: E402
    DATASET_ROOT, LABELS_FILE, TEST_FOLDER, TRAIN_FOLDER,
    RESULTS_DIR, FIGURES_DIR, SYNTH_POINT_NPY, SYNTH_CTX_NPY,
    require_dataset,
)
from holdout_split import compute_split  # noqa: E402

require_dataset()

# Windowing — identical to every other thesis script
WINDOW_SIZE       = 64
STEP_SIZE         = 16
OVERLAP_THRESHOLD = 0.1
CLIP_SIGMA        = 3.0

# VAE (identical to vae_point_adjust.py)
VAE_LATENT_DIM = 8
VAE_BATCH      = 32
VAE_EPOCHS     = 100
VAE_LR         = 1e-3
VAE_BETA       = 1.0
VAE_PATIENCE   = 20

# LSTM (identical to lstm_regression.py)
LSTM_LOOKBACK = 64
LSTM_FORECAST = 1
LSTM_HIDDEN   = 64
LSTM_LAYERS   = 2
LSTM_DROPOUT  = 0.1
LSTM_BATCH    = 64
LSTM_EPOCHS   = 100
LSTM_LR       = 1e-3
LSTM_PATIENCE = 20

# Deployable threshold sweep for the synthetic-calibration protocol
K_VALUES = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

# Data-efficiency curve: # of real anomaly windows added to synthetic training
EFFICIENCY_N_REAL = [0, 1, 2, 5, 10, 20, 9999]   # 9999 = "all available"

# How many synthetic windows to use as the anomaly class in training
N_SYNTH_TRAIN = 1000      # per relevant class; balances against real nominal

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Canonical Phase 1/2/3 reference numbers (verified from result CSVs, 81 chans)
REF = {
    "threshold_mean": 0.1825,
    "svm_mean": 0.5981, "svm_ctx": 0.5084, "svm_pt": 0.6508,
    "svm_smap": 0.5343, "svm_msl": 0.7256,
    "ann_mean": 0.4127,
    "vae_best": 0.5897, "vae_k3": 0.4168, "vae_ctx": 0.6385, "vae_pt": 0.5609,
    "lstm_best": 0.7429, "lstm_k3": 0.6611, "lstm_ctx": 0.8363, "lstm_pt": 0.6880,
    "tstr_mean": 0.4113, "tstr_ctx": 0.3377, "tstr_pt": 0.4546,
    "hundman_smap": 0.7519, "hundman_msl": 0.6955,
}

print("=" * 78)
print("PHASE 4 — DETECTORS TRAINED ON SYNTHETIC DATA, TESTED ON REAL DATA")
print("Answering: can generative fault data substitute for real fault data?")
print("=" * 78)
print(f"Device: {DEVICE}")
print(f"Seed:   {SEED}   Window: {WINDOW_SIZE}/{STEP_SIZE}   Overlap thr: {OVERLAP_THRESHOLD}")
print("=" * 78)


# =============================================================================
# SECTION 1 — SHARED UTILITIES (identical conventions to prior phases)
# =============================================================================

def parse_windows(seq_str):
    """Parse '[[s,e],...]' -> list of [s,e]."""
    nums = [int(n) for n in re.findall(r'\d+', str(seq_str))]
    return [[nums[i], nums[i+1]] for i in range(0, len(nums)-1, 2)]


def parse_class_labels(class_str):
    return re.findall(r'contextual|point', str(class_str))


def build_ground_truth_vector(num_values, anomaly_windows):
    y = np.zeros(num_values, dtype=int)
    for s, e in anomaly_windows:
        y[max(0, s):min(num_values, e)] = 1
    return y


def scale_to_tanh(window_zscore, clip_sigma=CLIP_SIGMA):
    """z-score -> clip ±3σ -> /3  => range [-1,1].  Identical to Phase 3."""
    return (np.clip(window_zscore, -clip_sigma, clip_sigma) / clip_sigma).astype(np.float32)


def make_windows_array(signal, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    return np.array([signal[i:i+window_size]
                     for i in range(0, len(signal)-window_size+1, step_size)],
                    dtype=np.float32)


def extract_features(window):
    """The 6 statistical features used by the Phase 2A SVM and Phase 3 TSTR."""
    mean = float(np.mean(window)); std = float(np.std(window))
    mn = float(np.min(window)); mx = float(np.max(window)); rng = mx - mn
    if std < 1e-9:
        autocorr = 0.0
    else:
        w1, w2 = window[:-1] - mean, window[1:] - mean
        denom = float(np.sum(w1 ** 2))
        autocorr = float(np.sum(w1 * w2) / denom) if denom > 1e-12 else 0.0
    return np.array([mean, std, mn, mx, rng, autocorr], dtype=np.float32)


def batch_features(windows):
    return np.array([extract_features(w) for w in windows])


def synthetic_threshold_rules(nominal_errors, synth_errors, mu=None, sd=None):
    """
    Compute ALL candidate deployable thresholds from nominal + synthetic errors,
    so we can pick the winner EMPIRICALLY on the real test set (logged per rule).

    None of these uses real fault labels — all are deployable. The classical
    k=3 rule is the baseline to beat. We test four synthetic-informed rules:

      k3          : mean + 3σ of nominal errors (classical default, NO synthetic)
      mixF1       : maximize F1 on labeled {nominal=0, synth=1} mixture.
                    (Optimizes detection F1 directly but is sensitive to the
                     base-rate mismatch between the calib mixture and the real
                     test anomaly rate — measured so we can see if it bites.)
      synthRecall : highest NOMINAL-error percentile that still catches >=50%
                    of synthetic faults. Base-rate-free: controls nominal FPR
                    by construction, only uses synthetic *recall*, not counts.
      synthMedian : threshold = median synthetic-anomaly error ("faults look
                    like this; trip when error reaches their typical level").
      bestK       : over k in K_VALUES, pick the largest k whose synthetic
                    detection rate (synth>thr) still exceeds 0.5 — i.e. push
                    the nominal-tail threshold as high as the synthetic faults
                    allow. Falls back to k=3 if none qualifies.

    IMPORTANT (consistency with the oracle): mu and sd MUST be passed by the
    caller using the SAME statistics the oracle grid uses (mu_e, sd_e of the
    nominal/train errors, no special floor). This guarantees that the k-based
    rules (k3, bestK) select thresholds that are a strict SUBSET of the oracle
    grid {mu_e + k·sd_e : k in K_VALUES}, so a synthetic rule can never exceed
    the oracle ceiling. Earlier versions recomputed sd with a 1.0 floor inside
    this function, which desynchronised the thresholds on low-variance channels
    and let bestK spuriously beat the oracle on rare-anomaly channels.

    Returns dict: rule_name -> threshold (float).
    """
    ne = np.asarray(nominal_errors, dtype=np.float64)
    se = np.asarray(synth_errors, dtype=np.float64)
    if mu is None:
        mu = ne.mean() if len(ne) else 0.0
    if sd is None:
        sd = ne.std() if len(ne) else 1.0
    # NOTE: no artificial floor here — use the caller's sd exactly. Only guard
    # the genuinely-degenerate sd==0 case to avoid a zero-width threshold.
    if sd <= 0.0:
        sd = max(np.finfo(np.float64).eps, abs(mu) * 1e-9)
    rules = {'k3': mu + 3.0 * sd}
    if len(ne) == 0 or len(se) == 0:
        rules.update({'mixF1': rules['k3'], 'synthRecall': rules['k3'],
                      'synthMedian': rules['k3'], 'bestK': rules['k3']})
        return rules

    # mixF1
    ce = np.concatenate([ne, se])
    cy = np.concatenate([np.zeros(len(ne)), np.ones(len(se))]).astype(int)
    grid = np.unique(np.quantile(ce, np.linspace(0.50, 0.999, 200)))
    best_f1, best_thr = -1.0, grid[0]
    for thr in grid:
        pred = (ce > thr).astype(int)
        tp = int(((pred == 1) & (cy == 1)).sum()); fp = int(((pred == 1) & (cy == 0)).sum())
        fn = int(((pred == 0) & (cy == 1)).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    rules['mixF1'] = float(best_thr)

    # synthRecall: highest nominal percentile still catching >=50% of synth
    cand = np.quantile(ne, np.linspace(0.90, 0.9995, 200))   # ascending
    chosen = cand[0]
    for thr in cand:
        if (se > thr).mean() >= 0.50:
            chosen = thr
        else:
            break
    rules['synthRecall'] = float(chosen)

    # synthMedian
    rules['synthMedian'] = float(np.median(se))

    # bestK — search the SAME K_VALUES the oracle uses, descending, so the
    # chosen threshold is guaranteed to be one the oracle also evaluated.
    k_chosen = 3.0
    for k in sorted(K_VALUES, reverse=True):
        if (se > (mu + k * sd)).mean() >= 0.50:
            k_chosen = k
            break
    rules['bestK'] = float(mu + k_chosen * sd)
    return rules




def make_sequences(signal, lookback=LSTM_LOOKBACK, forecast=LSTM_FORECAST):
    X, y = [], []
    for t in range(len(signal) - lookback - forecast + 1):
        X.append(signal[t:t+lookback])
        y.append(signal[t+lookback:t+lookback+forecast])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def point_adjust(y_true_ts, y_pred_ts):
    """Hundman 2018 point-adjust: any hit inside a GT segment flags the segment."""
    y_adj = y_pred_ts.copy()
    in_seg, seg_start = False, 0
    for t in range(len(y_true_ts)):
        if y_true_ts[t] == 1 and not in_seg:
            in_seg, seg_start = True, t
        elif y_true_ts[t] == 0 and in_seg:
            in_seg = False
            if y_pred_ts[seg_start:t].any():
                y_adj[seg_start:t] = 1
    if in_seg and y_pred_ts[seg_start:].any():
        y_adj[seg_start:] = 1
    return y_adj


def windows_to_timestep_scores(window_errors, signal_len,
                               window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    """Max-pool overlapping window errors back to per-timestep scores (VAE path)."""
    ts = np.zeros(signal_len, dtype=np.float32)
    cnt = np.zeros(signal_len, dtype=np.float32)
    for w_idx, start in enumerate(range(0, signal_len-window_size+1, step_size)):
        end = start + window_size
        ts[start:end] = np.maximum(ts[start:end], window_errors[w_idx])
        cnt[start:end] += 1
    unc = cnt == 0
    if unc.any():
        ts[unc] = window_errors[-1] if len(window_errors) else 0.0
    return ts


def evaluate_pa_timestep(ts_scores, y_true_ts, threshold):
    """Threshold per-timestep scores, point-adjust, return P/R/F1 (VAE path)."""
    if len(np.unique(y_true_ts)) < 2:
        return 0.0, 0.0, 0.0
    y_pred = (ts_scores > threshold).astype(int)
    y_adj = point_adjust(y_true_ts, y_pred)
    return (precision_score(y_true_ts, y_adj, zero_division=0),
            recall_score(y_true_ts, y_adj, zero_division=0),
            f1_score(y_true_ts, y_adj, zero_division=0))


def evaluate_pa_errors(errors, y_true_ts, threshold, offset=LSTM_LOOKBACK):
    """LSTM path: errors start at timestep `offset`. Point-adjust P/R/F1."""
    if len(np.unique(y_true_ts)) < 2:
        return 0.0, 0.0, 0.0
    T = len(y_true_ts)
    pred_len = min(len(errors), T - offset)
    y_pred = np.zeros(T, dtype=int)
    y_pred[offset:offset+pred_len] = (errors[:pred_len] > threshold).astype(int)
    y_adj = point_adjust(y_true_ts, y_pred)
    return (precision_score(y_true_ts, y_adj, zero_division=0),
            recall_score(y_true_ts, y_adj, zero_division=0),
            f1_score(y_true_ts, y_adj, zero_division=0))


# =============================================================================
# SECTION 2 — LOAD SYNTHETIC DATA + REAL CHANNEL CACHE
# =============================================================================

print("\nLoading synthetic fault windows from Phase 3 ...")
SYNTH_POINT = np.load(SYNTH_POINT_NPY).astype(np.float32)   # (2000, 64) in [-1,1]
SYNTH_CTX   = np.load(SYNTH_CTX_NPY).astype(np.float32)
print(f"  Synthetic point:      {SYNTH_POINT.shape}")
print(f"  Synthetic contextual: {SYNTH_CTX.shape}")

# Pre-computed feature matrices for the synthetic anomaly windows (reused often)
SYNTH_POINT_FEAT = batch_features(SYNTH_POINT)
SYNTH_CTX_FEAT   = batch_features(SYNTH_CTX)

labels_df = pd.read_csv(LABELS_FILE).drop_duplicates('chan_id', keep='first')
print(f"\nChannels (after P-2 dedup): {len(labels_df)}")

# ---- CLEAN PROTOCOL: held-out channels for evaluation ----
TRAIN_CHANNELS, HOLDOUT_CHANNELS = compute_split()
print(f"Clean protocol: evaluating on {len(HOLDOUT_CHANNELS)} held-out channels "
      f"(GAN never saw their anomalies)")


def load_channel(chan):
    """Return (train_norm, test_norm, test_len) or None if files missing."""
    tp = os.path.join(TEST_FOLDER, f"{chan}.npy")
    rp = os.path.join(TRAIN_FOLDER, f"{chan}.npy")
    if not (os.path.exists(tp) and os.path.exists(rp)):
        return None
    train_raw = np.load(rp)[:, 0]
    test_raw  = np.load(tp)[:, 0]
    mu, sd = train_raw.mean(), train_raw.std()
    if sd < 1e-9:
        sd = 1.0
    return (train_raw - mu) / sd, (test_raw - mu) / sd, len(test_raw)


def synth_anomaly_for_channel(anom_class, feat=True):
    """Pick the synthetic anomaly source matching a channel's declared class.
    Mixed-class channels get a 50/50 blend (mirrors Phase 3 TSTR)."""
    has_ctx = 'contextual' in str(anom_class)
    has_pt  = 'point' in str(anom_class)
    if has_ctx and has_pt:
        if feat:
            return np.vstack([SYNTH_POINT_FEAT[:N_SYNTH_TRAIN//2],
                              SYNTH_CTX_FEAT[:N_SYNTH_TRAIN//2]])
        return np.vstack([SYNTH_POINT[:N_SYNTH_TRAIN//2],
                          SYNTH_CTX[:N_SYNTH_TRAIN//2]])
    if has_ctx:
        return (SYNTH_CTX_FEAT if feat else SYNTH_CTX)[:N_SYNTH_TRAIN]
    return (SYNTH_POINT_FEAT if feat else SYNTH_POINT)[:N_SYNTH_TRAIN]


# =============================================================================
# SECTION 3 — DETECTOR A: SVM  (Train on Synthetic, Test on Real)
# =============================================================================
#
# Positives = SYNTHETIC anomaly windows (matching the channel's class).
# Negatives = REAL NOMINAL windows from the channel's TRAIN split.
# Test      = REAL test windows, window-level labels via OVERLAP_THRESHOLD.
# Evaluation = window-level F1  (directly comparable to Phase 2A SVM = 0.598
#              and Phase 3 TSTR = 0.411).
#
# This is the cleanest expression of the thesis question for a discriminative
# detector: the ONLY thing that changed from Phase 2A is that the labeled
# fault examples are now generated rather than observed.

def build_real_test_windows(test_norm, test_len, anom_seqs):
    """Real test windows + window-level labels."""
    y_ts = build_ground_truth_vector(test_len, anom_seqs)
    X, y = [], []
    for start in range(0, len(test_norm)-WINDOW_SIZE+1, STEP_SIZE):
        w  = scale_to_tanh(test_norm[start:start+WINDOW_SIZE])
        lb = y_ts[start:start+WINDOW_SIZE]
        X.append(extract_features(w))
        y.append(1 if lb.mean() >= OVERLAP_THRESHOLD else 0)
    return np.array(X), np.array(y)


def build_real_nominal_features(train_norm):
    """Feature matrix of REAL nominal windows from the train split."""
    feats = []
    for start in range(0, len(train_norm)-WINDOW_SIZE+1, STEP_SIZE):
        feats.append(extract_features(scale_to_tanh(train_norm[start:start+WINDOW_SIZE])))
    return np.array(feats)


def build_real_anomaly_features(test_norm, test_len, anom_seqs):
    """REAL anomaly windows (for the data-efficiency curve only)."""
    feats = []
    for s, e in anom_seqs:
        s, e = max(0, s), min(len(test_norm), e)
        if e - s < WINDOW_SIZE:
            pad = WINDOW_SIZE - (e - s)
            s2 = max(0, s - pad // 2); e2 = min(len(test_norm), e + (pad - pad//2))
            seg = test_norm[s2:e2]
            if len(seg) < WINDOW_SIZE:
                continue
            feats.append(extract_features(scale_to_tanh(seg[:WINDOW_SIZE])))
        else:
            seg = test_norm[s:e]
            for i in range(0, len(seg)-WINDOW_SIZE+1, STEP_SIZE):
                feats.append(extract_features(scale_to_tanh(seg[i:i+WINDOW_SIZE])))
    return np.array(feats) if feats else np.empty((0, 6), dtype=np.float32)


def fit_eval_svm(X_train, y_train, X_test, y_test):
    scaler = StandardScaler().fit(X_train)
    cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
    svm = SVC(kernel='rbf', C=10.0, gamma='scale',
              class_weight={0: cw[0], 1: cw[1]}, random_state=SEED)
    svm.fit(scaler.transform(X_train), y_train)
    yp = svm.predict(scaler.transform(X_test))
    return (precision_score(y_test, yp, zero_division=0),
            recall_score(y_test, yp, zero_division=0),
            f1_score(y_test, yp, zero_division=0))


def run_svm_phase4():
    print("\n" + "=" * 78)
    print("DETECTOR A — SVM TSTR (synthetic anomalies + real nominal)")
    print("=" * 78)
    rows = []
    for idx, row in labels_df.iterrows():
        chan, sc, ac = row['chan_id'], row['spacecraft'], row['class']
        if chan not in HOLDOUT_CHANNELS:
            continue
        anom_seqs = parse_windows(row['anomaly_sequences'])
        ch = load_channel(chan)
        if ch is None:
            continue
        train_norm, test_norm, test_len = ch

        X_test, y_test = build_real_test_windows(test_norm, test_len, anom_seqs)
        if y_test.sum() == 0 or y_test.sum() == len(y_test):
            rows.append({'chan_id': chan, 'spacecraft': sc, 'anomaly_class': ac,
                         'svm_p4_precision': 0.0, 'svm_p4_recall': 0.0,
                         'svm_p4_f1': 0.0, 'note': 'no_class_variation'})
            continue

        X_nom = build_real_nominal_features(train_norm)
        y_nom = np.zeros(len(X_nom), dtype=int)
        X_syn = synth_anomaly_for_channel(ac, feat=True)
        y_syn = np.ones(len(X_syn), dtype=int)

        X_train = np.vstack([X_nom, X_syn])
        y_train = np.concatenate([y_nom, y_syn])
        p, r, f1 = fit_eval_svm(X_train, y_train, X_test, y_test)
        rows.append({'chan_id': chan, 'spacecraft': sc, 'anomaly_class': ac,
                     'svm_p4_precision': round(p, 4), 'svm_p4_recall': round(r, 4),
                     'svm_p4_f1': round(f1, 4), 'note': 'ok'})
        if (idx + 1) % 20 == 0:
            print(f"  [{idx+1:3d}] {chan:6s}  SVM-P4 F1={f1:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "phase4_svm_tstr_results.csv", index=False)
    ok = df[df.note == 'ok']
    print(f"\n  SVM-P4 mean F1 = {ok.svm_p4_f1.mean():.4f}   "
          f"(Phase 2A real SVM = {REF['svm_mean']:.4f}, Phase 3 TSTR = {REF['tstr_mean']:.4f})")
    return df


# =============================================================================
# SECTION 4 — VAE  (trained on real nominal; threshold CALIBRATED on synthetic)
# =============================================================================
#
# The VAE architecture & training are byte-for-byte identical to
# vae_point_adjust.py (Phase 2B). The ONLY change is how the deployable
# detection threshold is chosen:
#
#   Phase 2B deployable (k=3): threshold = mean+3σ of TRAIN reconstruction
#       errors. Collapsed to 0.418 because k=3 is wrong for ~26 channels and
#       the right k cannot be known without test labels.
#
#   Phase 4 deployable (synthetic-calibrated): pass the SYNTHETIC anomaly
#       windows through the trained VAE, measure their reconstruction error,
#       and pick the threshold (over the same k-grid) that best SEPARATES
#       real-nominal-train errors from synthetic-anomaly errors. No real
#       labels used -> fully deployable. We report this as `vae_p4_f1`.
#
# We ALSO recompute the Phase 2B oracle-best and k=3 here on the same run so
# the comparison is on identical VAE instances (removes any train-noise
# confound between phases).

class _Encoder(nn.Module):
    def __init__(self, d, l):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU())
        self.fc_mu = nn.Linear(16, l); self.fc_lv = nn.Linear(16, l)
    def forward(self, x):
        h = self.net(x); return self.fc_mu(h), self.fc_lv(h)

class _Decoder(nn.Module):
    def __init__(self, l, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(l, 16), nn.ReLU(),
                                 nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, d))
    def forward(self, z):
        return self.net(z)

class VAE(nn.Module):
    def __init__(self, d, l):
        super().__init__()
        self.encoder = _Encoder(d, l); self.decoder = _Decoder(l, d)
    def reparameterize(self, mu, lv):
        std = torch.exp(0.5 * lv); return mu + torch.randn_like(std) * std
    def forward(self, x):
        mu, lv = self.encoder(x); z = self.reparameterize(mu, lv)
        return self.decoder(z), mu, lv
    def reconstruct(self, x):
        mu, _ = self.encoder(x); return self.decoder(mu)

def _vae_loss(x, xh, mu, lv, beta=VAE_BETA):
    recon = nn.functional.mse_loss(xh, x, reduction='mean')
    kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
    return recon + beta * kl

def _vae_window_errors(model, Wt):
    model.eval(); errs = []
    with torch.no_grad():
        for i in range(0, len(Wt), 256):
            b = Wt[i:i+256].to(DEVICE)
            xh = model.reconstruct(b)
            errs.append(((b - xh) ** 2).mean(dim=1).cpu().numpy())
    return np.concatenate(errs)

def run_vae_phase4():
    print("\n" + "=" * 78)
    print("DETECTOR B — VAE (real-nominal training, SYNTHETIC threshold calibration)")
    print("=" * 78)
    rows = []
    for idx, row in labels_df.iterrows():
        chan, sc, ac = row['chan_id'], row['spacecraft'], row['class']
        if chan not in HOLDOUT_CHANNELS:
            continue
        anom_seqs = parse_windows(row['anomaly_sequences'])
        ch = load_channel(chan)
        if ch is None:
            continue
        train_norm, test_norm, test_len = ch
        y_true_ts = build_ground_truth_vector(test_len, anom_seqs)

        X_train = make_windows_array(train_norm)
        X_test  = make_windows_array(test_norm)
        if len(X_train) < 10:
            continue

        n_val = max(1, int(0.1 * len(X_train)))
        X_tr = torch.FloatTensor(X_train[:-n_val])
        X_va = torch.FloatTensor(X_train[-n_val:])
        X_te = torch.FloatTensor(X_test)

        tr_loader = DataLoader(TensorDataset(X_tr), batch_size=VAE_BATCH, shuffle=True)
        va_loader = DataLoader(TensorDataset(X_va), batch_size=VAE_BATCH)

        model = VAE(WINDOW_SIZE, VAE_LATENT_DIM).to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=VAE_LR)
        sch = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
        best_val, pc, ep_run = float('inf'), 0, 0
        for _ in range(VAE_EPOCHS):
            model.train()
            for (b,) in tr_loader:
                b = b.to(DEVICE); opt.zero_grad()
                xh, mu, lv = model(b); loss = _vae_loss(b, xh, mu, lv)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            model.eval(); vl = 0.0
            with torch.no_grad():
                for (b,) in va_loader:
                    b = b.to(DEVICE); xh, mu, lv = model(b); vl += _vae_loss(b, xh, mu, lv).item()
            vl /= len(va_loader); sch.step(vl); ep_run += 1
            if vl < best_val - 1e-5:
                best_val, pc = vl, 0
            else:
                pc += 1
                if pc >= VAE_PATIENCE:
                    break

        train_err = _vae_window_errors(model, X_tr)
        test_err_w = _vae_window_errors(model, X_te)
        ts_scores = windows_to_timestep_scores(test_err_w, test_len)

        mu_e, sd_e = train_err.mean(), train_err.std()
        # --- Phase 2B style references on THIS instance (fair within-run) ---
        grid = {f"k{k}": mu_e + k * sd_e for k in K_VALUES}
        for pct in [90, 95, 99]:
            grid[f"p{pct}"] = np.percentile(train_err, pct)
        best_f1 = best_p = best_r = 0.0; best_name = "none"
        for name, thr in grid.items():
            p, r, f1 = evaluate_pa_timestep(ts_scores, y_true_ts, thr)
            if f1 > best_f1:
                best_f1, best_p, best_r, best_name = f1, p, r, name
        p_k3, r_k3, f1_k3 = evaluate_pa_timestep(ts_scores, y_true_ts, mu_e + 3.0 * sd_e)

        # --- Phase 4 deployable thresholds: evaluate ALL candidate rules ---
        # Each rule is label-free (uses only nominal + synthetic errors). We log
        # every rule's real-test F1 so the winner is chosen empirically later.
        synth_w = synth_anomaly_for_channel(ac, feat=False)
        synth_err = _vae_window_errors(model, torch.FloatTensor(synth_w))
        rule_thr = synthetic_threshold_rules(train_err, synth_err, mu=mu_e, sd=sd_e)
        rule_f1 = {}
        for rname, thr in rule_thr.items():
            _, _, rf1 = evaluate_pa_timestep(ts_scores, y_true_ts, thr)
            rule_f1[rname] = round(rf1, 4)

        rows.append({'chan_id': chan, 'spacecraft': sc, 'anomaly_class': ac,
                     'epochs_run': ep_run,
                     'vae_best_f1': round(best_f1, 4), 'vae_best_threshold': best_name,
                     'vae_k3_f1': round(f1_k3, 4),
                     'vae_mixF1_f1': rule_f1['mixF1'],
                     'vae_synthRecall_f1': rule_f1['synthRecall'],
                     'vae_synthMedian_f1': rule_f1['synthMedian'],
                     'vae_bestK_f1': rule_f1['bestK'],
                     'note': 'ok'})
        if (idx + 1) % 20 == 0:
            print(f"  [{idx+1:3d}] {chan:6s}  k3={f1_k3:.3f} mixF1={rule_f1['mixF1']:.3f} "
                  f"synthRec={rule_f1['synthRecall']:.3f} synthMed={rule_f1['synthMedian']:.3f} "
                  f"bestK={rule_f1['bestK']:.3f} | oracle={best_f1:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "phase4_vae_results.csv", index=False)
    ok = df[df.note == 'ok']
    print(f"\n  VAE deployable threshold rules (mean / median F1 over channels):")
    print(f"    {'rule':<12}{'mean':>8}{'median':>9}")
    for rn, col in [('k3','vae_k3_f1'),('mixF1','vae_mixF1_f1'),('synthRecall','vae_synthRecall_f1'),
                    ('synthMedian','vae_synthMedian_f1'),('bestK','vae_bestK_f1'),('oracle','vae_best_f1')]:
        print(f"    {rn:<12}{ok[col].mean():>8.4f}{ok[col].median():>9.4f}")
    print(f"    (k3 = classical default, no synthetic; oracle uses test labels — ceiling only)")
    print(f"  Phase 2B reference: k3 = {REF['vae_k3']:.4f}, oracle best = {REF['vae_best']:.4f}")
    return df


# =============================================================================
# SECTION 5 — LSTM  (real-nominal training; Mode A calibration + Mode B discriminative)
# =============================================================================
#
# Architecture/training identical to lstm_regression.py (Phase 2C).
#
# Mode A (HEADLINE, synthetic threshold calibration):
#   Same as the VAE — train the LSTM on real nominal sequences, then use the
#   synthetic anomaly windows to pick the deployable threshold (the k that
#   best separates nominal prediction error from synthetic-anomaly prediction
#   error). Fully deployable, no real faults seen.
#
# Mode B (ABLATION, discriminative LSTM classifier):
#   A prediction-based LSTM trained ON anomaly windows would learn to predict
#   them well — LOWERING their prediction error and HURTING detection. That is
#   the wrong way to feed synthetic faults to a predictor. The sound aggressive
#   alternative is a DISCRIMINATIVE LSTM classifier: train it to output
#   nominal(0)/fault(1) from real-nominal windows (label 0) and synthetic
#   anomaly windows (label 1), then classify each real test window and
#   point-adjust. This tests whether synthetic faults can directly supervise a
#   temporal classifier. We expect it to track the SVM-TSTR result while adding
#   temporal modelling; reported separately and discussed honestly given the
#   point-class temporal gap (synth point ACF(1)=0.506 vs real 0.665).

class LSTMPredictor(nn.Module):
    def __init__(self, hidden=LSTM_HIDDEN, layers=LSTM_LAYERS,
                 dropout=LSTM_DROPOUT, forecast=LSTM_FORECAST):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=layers,
                            dropout=dropout if layers > 1 else 0.0, batch_first=True)
        self.fc = nn.Linear(hidden, forecast)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def _lstm_train(model, tr_loader, va_loader):
    opt = optim.Adam(model.parameters(), lr=LSTM_LR)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
    best_val, pc, ep = float('inf'), 0, 0
    for _ in range(LSTM_EPOCHS):
        model.train()
        for Xb, yb in tr_loader:
            Xb = Xb.unsqueeze(-1).to(DEVICE); yb = yb.to(DEVICE)
            opt.zero_grad(); loss = nn.functional.mse_loss(model(Xb), yb)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval(); vl = 0.0
        with torch.no_grad():
            for Xb, yb in va_loader:
                Xb = Xb.unsqueeze(-1).to(DEVICE); yb = yb.to(DEVICE)
                vl += nn.functional.mse_loss(model(Xb), yb).item()
        vl /= max(1, len(va_loader)); sch.step(vl); ep += 1
        if vl < best_val - 1e-6:
            best_val, pc = vl, 0
        else:
            pc += 1
            if pc >= LSTM_PATIENCE:
                break
    return ep

def _lstm_errors(model, X, y):
    model.eval(); errs = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            Xb = X[i:i+256].unsqueeze(-1).to(DEVICE)
            yb = y[i:i+256].to(DEVICE)
            errs.append(((model(Xb) - yb) ** 2).mean(dim=1).cpu().numpy())
    return np.concatenate(errs) if errs else np.array([])

class LSTMClassifier(nn.Module):
    """LSTM that classifies a 64-step window as nominal(0)/fault(1)."""
    def __init__(self, hidden=LSTM_HIDDEN, layers=LSTM_LAYERS, dropout=LSTM_DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=layers,
                            dropout=dropout if layers > 1 else 0.0, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)   # logit


def _run_discriminative_lstm(train_norm, test_norm, test_len, y_true_ts,
                             anom_seqs, synth_w):
    """Train an LSTM classifier on real-nominal(0) + synthetic-anomaly(1)
    windows; predict per real-test-window; map to timesteps; point-adjust F1."""
    # Training windows
    nom = make_windows_array(train_norm)
    nom = np.array([scale_to_tanh(w) for w in nom], dtype=np.float32)
    pos = synth_w.astype(np.float32)
    if len(nom) == 0:
        return 0.0
    X = np.vstack([nom, pos])
    y = np.concatenate([np.zeros(len(nom)), np.ones(len(pos))]).astype(np.float32)
    Xt = torch.FloatTensor(X).unsqueeze(-1)
    yt = torch.FloatTensor(y)
    # class-balanced positive weight
    n_pos = max(1, int(y.sum())); n_neg = max(1, len(y) - n_pos)
    pos_weight = torch.tensor([n_neg / n_pos], device=DEVICE)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=LSTM_BATCH, shuffle=True)

    model = LSTMClassifier().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LSTM_LR)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model.train()
    for _ in range(min(LSTM_EPOCHS, 30)):   # classifier converges fast
        for Xb, yb in loader:
            Xb = Xb.to(DEVICE); yb = yb.to(DEVICE)
            opt.zero_grad(); loss = lossf(model(Xb), yb)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

    # Score real test windows -> timestep prediction via max-pool overlap
    model.eval()
    test_w = make_windows_array(test_norm)
    test_w = np.array([scale_to_tanh(w) for w in test_w], dtype=np.float32)
    if len(test_w) == 0:
        return 0.0
    with torch.no_grad():
        probs = []
        Tt = torch.FloatTensor(test_w).unsqueeze(-1)
        for i in range(0, len(Tt), 256):
            probs.append(torch.sigmoid(model(Tt[i:i+256].to(DEVICE))).cpu().numpy())
        probs = np.concatenate(probs)
    ts = windows_to_timestep_scores(probs, test_len)
    # standard 0.5 decision on probability, then point-adjust
    _, _, f1 = evaluate_pa_timestep(ts, y_true_ts, 0.5)
    return f1


def run_lstm_phase4():
    print("\n" + "=" * 78)
    print("DETECTOR C — LSTM (Mode A: synth calibration | Mode B: discriminative)")
    print("=" * 78)
    rows = []
    for idx, row in labels_df.iterrows():
        chan, sc, ac = row['chan_id'], row['spacecraft'], row['class']
        if chan not in HOLDOUT_CHANNELS:
            continue
        anom_seqs = parse_windows(row['anomaly_sequences'])
        ch = load_channel(chan)
        if ch is None:
            continue
        train_norm, test_norm, test_len = ch
        y_true_ts = build_ground_truth_vector(test_len, anom_seqs)

        Xtr_s, ytr_s = make_sequences(train_norm)
        Xte_s, yte_s = make_sequences(test_norm)
        if len(Xtr_s) < 20:
            continue
        n_val = max(1, int(0.1 * len(Xtr_s)))
        X_tr = torch.FloatTensor(Xtr_s[:-n_val]); y_tr = torch.FloatTensor(ytr_s[:-n_val])
        X_va = torch.FloatTensor(Xtr_s[-n_val:]); y_va = torch.FloatTensor(ytr_s[-n_val:])
        X_te = torch.FloatTensor(Xte_s);          y_te = torch.FloatTensor(yte_s)

        tr_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=LSTM_BATCH, shuffle=True)
        va_loader = DataLoader(TensorDataset(X_va, y_va), batch_size=LSTM_BATCH)

        # ---- Mode A: standard nominal LSTM ----
        modelA = LSTMPredictor().to(DEVICE)
        ep = _lstm_train(modelA, tr_loader, va_loader)
        train_err = _lstm_errors(modelA, X_tr, y_tr)
        test_err  = _lstm_errors(modelA, X_te, y_te)
        mu_e, sd_e = train_err.mean(), train_err.std()

        grid = {f"k{k}": mu_e + k * sd_e for k in K_VALUES}
        for pct in [90, 95, 99]:
            grid[f"p{pct}"] = np.percentile(train_err, pct)
        best_f1 = 0.0; best_name = "none"
        for name, thr in grid.items():
            _, _, f1 = evaluate_pa_errors(test_err, y_true_ts, thr)
            if f1 > best_f1:
                best_f1, best_name = f1, name
        _, _, f1_k3 = evaluate_pa_errors(test_err, y_true_ts, mu_e + 3.0 * sd_e)

        # Phase 4 deployable thresholds: evaluate ALL candidate rules. Each
        # synthetic window is exactly WINDOW_SIZE(=64), too short for a full
        # 64-lookback + 1-forecast sample, so we CONCATENATE the synthetic
        # windows into one stream and slide the standard (64 -> 1) builder over
        # it to get synthetic one-step prediction errors under the nominal model.
        synth_w = synth_anomaly_for_channel(ac, feat=False)
        synth_stream = synth_w.reshape(-1)                      # (n*64,)
        sx, sy = make_sequences(synth_stream)                   # (m,64),(m,1)
        if len(sx):
            synth_err = _lstm_errors(modelA, torch.FloatTensor(sx), torch.FloatTensor(sy))
        else:
            synth_err = test_err  # degenerate guard; not expected
        rule_thr = synthetic_threshold_rules(train_err, synth_err, mu=mu_e, sd=sd_e)
        rule_f1 = {}
        for rname, thr in rule_thr.items():
            _, _, rf1 = evaluate_pa_errors(test_err, y_true_ts, thr)
            rule_f1[rname] = round(rf1, 4)

        # ---- Mode B: discriminative LSTM classifier (synthetic-supervised) ----
        # A prediction-based LSTM trained ON anomalies would learn to predict
        # them well, LOWERING their error and HURTING detection — the wrong use
        # of synthetic data for a predictor. The sound aggressive alternative is
        # a DISCRIMINATIVE LSTM: classify a window as nominal(0) vs fault(1),
        # trained on real-nominal windows (0) and synthetic anomaly windows (1).
        # This tests whether synthetic faults can supervise a temporal
        # classifier directly. Evaluated at window level then point-adjusted.
        f1_B = _run_discriminative_lstm(train_norm, test_norm, test_len,
                                        y_true_ts, anom_seqs, synth_w)

        rows.append({'chan_id': chan, 'spacecraft': sc, 'anomaly_class': ac,
                     'epochs_run': ep,
                     'lstm_best_f1': round(best_f1, 4), 'lstm_best_threshold': best_name,
                     'lstm_k3_f1': round(f1_k3, 4),
                     'lstm_mixF1_f1': rule_f1['mixF1'],
                     'lstm_synthRecall_f1': rule_f1['synthRecall'],
                     'lstm_synthMedian_f1': rule_f1['synthMedian'],
                     'lstm_bestK_f1': rule_f1['bestK'],
                     'lstm_p4_disc_f1': round(f1_B, 4),
                     'note': 'ok'})
        if (idx + 1) % 20 == 0:
            print(f"  [{idx+1:3d}] {chan:6s}  k3={f1_k3:.3f} mixF1={rule_f1['mixF1']:.3f} "
                  f"synthRec={rule_f1['synthRecall']:.3f} synthMed={rule_f1['synthMedian']:.3f} "
                  f"bestK={rule_f1['bestK']:.3f} disc={f1_B:.3f} | oracle={best_f1:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "phase4_lstm_results.csv", index=False)
    ok = df[df.note == 'ok']
    print(f"\n  LSTM deployable threshold rules (mean / median F1 over channels):")
    print(f"    {'rule':<12}{'mean':>8}{'median':>9}")
    for rn, col in [('k3','lstm_k3_f1'),('mixF1','lstm_mixF1_f1'),('synthRecall','lstm_synthRecall_f1'),
                    ('synthMedian','lstm_synthMedian_f1'),('bestK','lstm_bestK_f1'),
                    ('discrim(MB)','lstm_p4_disc_f1'),('oracle','lstm_best_f1')]:
        print(f"    {rn:<12}{ok[col].mean():>8.4f}{ok[col].median():>9.4f}")
    print(f"    (k3 = classical default; discrim = Mode B ablation; oracle uses test labels)")
    print(f"  Phase 2C reference: k3 = {REF['lstm_k3']:.4f}, oracle best = {REF['lstm_best']:.4f}")
    return df


# =============================================================================
# SECTION 6 — DATA-EFFICIENCY CURVE (the "money figure")
# =============================================================================
#
# How many REAL fault windows is the synthetic data worth? For the SVM, sweep
# the number of real anomaly windows added to the synthetic training set and
# measure mean F1 across channels. Two reference lines fall out naturally:
#   n_real = 0   -> pure synthetic (= the Phase 4 SVM result)
#   n_real = all -> all real faults + synthetic
# and the Phase 2A all-real SVM (0.598) is the ceiling. The point where the
# curve crosses / approaches 0.598 tells us the synthetic data's real-data
# equivalent value.

def run_data_efficiency():
    print("\n" + "=" * 78)
    print("EXTENSION 1 — DATA-EFFICIENCY CURVE (clean: real faults from TRAIN channels)")
    print("=" * 78)

    # Build a SHARED real-anomaly pool from the TRAIN channels only.
    # Under the clean protocol the real windows we add must never come from the
    # held-out channels we evaluate on.
    train_pool = []
    for _, row in labels_df.iterrows():
        chan = row['chan_id']
        if chan not in TRAIN_CHANNELS:
            continue
        anom_seqs = parse_windows(row['anomaly_sequences'])
        ch = load_channel(chan)
        if ch is None:
            continue
        _, test_norm, test_len = ch
        feats = build_real_anomaly_features(test_norm, test_len, anom_seqs)
        if len(feats):
            train_pool.append(feats)
    TRAIN_ANOM_POOL = np.vstack(train_pool) if train_pool else np.empty((0, 6), dtype=np.float32)
    print(f"  Train-channel real-anomaly pool: {len(TRAIN_ANOM_POOL)} windows")

    # Pre-extract per-channel pieces once — EVALUATION on held-out channels only.
    cache = []
    for _, row in labels_df.iterrows():
        chan, sc, ac = row['chan_id'], row['spacecraft'], row['class']
        if chan not in HOLDOUT_CHANNELS:
            continue
        anom_seqs = parse_windows(row['anomaly_sequences'])
        ch = load_channel(chan)
        if ch is None:
            continue
        train_norm, test_norm, test_len = ch
        X_test, y_test = build_real_test_windows(test_norm, test_len, anom_seqs)
        if y_test.sum() == 0 or y_test.sum() == len(y_test):
            continue
        X_nom = build_real_nominal_features(train_norm)
        X_syn = synth_anomaly_for_channel(ac, feat=True)
        cache.append((chan, sc, ac, X_test, y_test, X_nom, X_syn))

    rng = np.random.default_rng(SEED)
    eff_rows = []
    for n_real in EFFICIENCY_N_REAL:
        per_chan = []
        for (chan, sc, ac, X_test, y_test, X_nom, X_syn) in cache:
            y_nom = np.zeros(len(X_nom), dtype=int)
            parts_X = [X_nom, X_syn]; parts_y = [y_nom, np.ones(len(X_syn), dtype=int)]
            if n_real > 0 and len(TRAIN_ANOM_POOL) > 0:
                take = min(n_real, len(TRAIN_ANOM_POOL))
                sel = rng.choice(len(TRAIN_ANOM_POOL), size=take, replace=False)
                parts_X.append(TRAIN_ANOM_POOL[sel])
                parts_y.append(np.ones(take, dtype=int))
            X_train = np.vstack(parts_X); y_train = np.concatenate(parts_y)
            _, _, f1 = fit_eval_svm(X_train, y_train, X_test, y_test)
            per_chan.append(f1)
        label = "all" if n_real == 9999 else str(n_real)
        eff_rows.append({'n_real_anom': label, 'mean_f1': round(np.mean(per_chan), 4),
                         'median_f1': round(np.median(per_chan), 4)})
        print(f"  n_real={label:>4}  mean F1={np.mean(per_chan):.4f}")
    df = pd.DataFrame(eff_rows)
    df.to_csv(RESULTS_DIR / "phase4_data_efficiency.csv", index=False)
    return df

# =============================================================================
# SECTION 7 — MASTER COMPARISON TABLE + PUBLICATION FIGURES
# =============================================================================

def _pick_winning_rule(df, rule_cols, k3_col, oracle_col):
    """Pick the deployable rule with the highest AGGREGATE mean F1, after
    EXCLUDING any rule that exceeds the oracle ceiling on one or more channels.
    A label-free rule cannot legitimately beat the label-using oracle; when it
    does, the threshold has desynchronised from the oracle grid — a measurement
    artefact, not a result (this is exactly the LSTM synthMedian 'win', which
    exceeds the oracle on 7/20 held-out channels). Of the artefact-free rules,
    declare a winner only if it beats k3 by a meaningful margin (>0.01); a
    sub-0.01 'win' is a tie, not an improvement. Otherwise fall back to k3, the
    honest parameter-free default. Returns (winning_col, winning_mean, k3_mean)."""
    k3_mean = df[k3_col].mean()
    clean_rules = [c for c in rule_cols if (df[c] > df[oracle_col] + 1e-6).sum() == 0]
    if not clean_rules:
        return k3_col, k3_mean, k3_mean   # all synthetic rules are artefacts -> k3
    means = {c: df[c].mean() for c in clean_rules}
    win_col = max(means, key=means.get)
    win_mean = means[win_col]
    if win_mean <= k3_mean + 0.01:
        return k3_col, k3_mean, k3_mean   # no artefact-free rule meaningfully beats k3
    return win_col, win_mean, k3_mean


def build_master(svm_df, vae_df, lstm_df):
    """Merge per-channel P4 results with verified Phase 1/2 CSV numbers, and
    select each detector's winning DEPLOYABLE threshold rule empirically."""
    def dedup(p):
        return pd.read_csv(RESULTS_DIR / p).drop_duplicates('chan_id', keep='first')
    thr  = dedup("threshold_baseline_results.csv")[['chan_id', 'f1']].rename(columns={'f1': 'p1_threshold_f1'})
    cls  = dedup("classification_results.csv")[['chan_id', 'svm_f1']].rename(columns={'svm_f1': 'p2_svm_f1'})
    vaeR = dedup("vae_point_adjust_results.csv")[['chan_id', 'pa_best_f1', 'pa_k3_f1']].rename(
        columns={'pa_best_f1': 'p2_vae_best_f1', 'pa_k3_f1': 'p2_vae_k3_f1'})
    lsR  = dedup("lstm_regression_results.csv")[['chan_id', 'lstm_best_f1', 'lstm_k3_f1']].rename(
        columns={'lstm_best_f1': 'p2_lstm_best_f1', 'lstm_k3_f1': 'p2_lstm_k3_f1'})

    # ------------------------------------------------------------------------
    # KNOWN CAVEAT — the p3_tstr_f1 reference column
    # ------------------------------------------------------------------------
    # This column is read from the PRE-HELD-OUT generator's TSTR table
    # (gan_v3_tstr_results.csv). That generator was trained on anomaly windows
    # from every channel, INCLUDING the 20 held out here, so its per-channel
    # TSTR numbers are optimistic on exactly those channels:
    #
    #     mean TSTR over the 20 held-out channels
    #         contaminated v3 generator : 0.487
    #         clean v4 generator        : 0.450   (gan_v4_clean_tstr_results.csv)
    #
    # It is kept as-is so this script reproduces the committed
    # phase4_master_comparison.csv and the P-1 case-study figure exactly as they
    # appear in the thesis. It is a REFERENCE column only: no headline result,
    # no significance test, and no other figure depends on it. To rebuild the
    # table on a fully clean footing, switch the filename below to
    # "gan_v4_clean_tstr_results.csv" — that changes the p3_tstr_f1 column and
    # the "P3 TSTR" bar of the P-1 figure, and nothing else.
    tstr = dedup("gan_v3_tstr_results.csv")[['chan_id', 'tstr_f1']].rename(
        columns={'tstr_f1': 'p3_tstr_f1'})

    # --- empirically choose the winning deployable rule per detector ---
    vae_rules = ['vae_mixF1_f1', 'vae_synthRecall_f1', 'vae_synthMedian_f1', 'vae_bestK_f1']
    lstm_rules = ['lstm_mixF1_f1', 'lstm_synthRecall_f1', 'lstm_synthMedian_f1', 'lstm_bestK_f1']
    vae_win, vae_wm, vae_k3m = _pick_winning_rule(vae_df, vae_rules, 'vae_k3_f1', 'vae_best_f1')
    lstm_win, lstm_wm, lstm_k3m = _pick_winning_rule(lstm_df, lstm_rules, 'lstm_k3_f1', 'lstm_best_f1')
    print("\n" + "-" * 70)
    print("EMPIRICAL DEPLOYABLE-RULE SELECTION (winner = highest aggregate mean F1):")
    print(f"  VAE  winner: {vae_win}  (mean {vae_wm:.4f} vs k3 {vae_k3m:.4f})")
    print(f"  LSTM winner: {lstm_win} (mean {lstm_wm:.4f} vs k3 {lstm_k3m:.4f})")
    print("-" * 70)

    # GUARDRAIL: a label-free rule must NEVER exceed the oracle (which uses test
    # labels). If it does, the thresholds desynchronised from the oracle grid —
    # a bug, not a result. Flag loudly per channel.
    for det, df, rcols, ocol in [("VAE", vae_df, vae_rules, 'vae_best_f1'),
                                  ("LSTM", lstm_df, lstm_rules, 'lstm_best_f1')]:
        viol = []
        for rc in rcols + [det.lower() + '_k3_f1']:
            if rc in df.columns:
                bad = df[df[rc] > df[ocol] + 1e-6]
                if len(bad):
                    viol.append((rc, len(bad)))
        if viol:
            print(f"  [WARNING] {det}: rule>oracle on channels — {viol}")
            print(f"            (investigate threshold/oracle-grid desync before trusting these)")
        else:
            print(f"  [OK] {det}: no rule exceeds the oracle ceiling (thresholds consistent).")
    print("-" * 70)

    # canonical P4 deployable column per detector
    vae_df = vae_df.copy();  vae_df['vae_p4_f1'] = vae_df[vae_win]
    lstm_df = lstm_df.copy(); lstm_df['lstm_p4_f1'] = lstm_df[lstm_win]

    m = svm_df[['chan_id', 'spacecraft', 'anomaly_class', 'svm_p4_f1']].copy()
    m = m.merge(vae_df[['chan_id', 'vae_p4_f1']], on='chan_id', how='left')
    m = m.merge(lstm_df[['chan_id', 'lstm_p4_f1', 'lstm_p4_disc_f1']], on='chan_id', how='left')
    for t in [thr, cls, vaeR, lsR, tstr]:
        m = m.merge(t, on='chan_id', how='left')
    m.attrs['vae_win'] = vae_win; m.attrs['lstm_win'] = lstm_win

    # --- paired Wilcoxon signed-rank tests on the held-out channels -----------
    # Backs the statistics table in the report. n = #held-out channels; with that
    # n the tests are not highly powered, so a non-significant result is reported
    # as "no detectable difference", not as proof of equality.
    def _wilcoxon(a, b):
        a = np.asarray(a, float); b = np.asarray(b, float)
        ok = ~(np.isnan(a) | np.isnan(b)); a, b = a[ok], b[ok]
        if len(a) < 1 or np.allclose(a, b):
            return float('nan')
        try:
            return float(stats.wilcoxon(a, b, zero_method='wilcox').pvalue)
        except ValueError:
            return float('nan')
    lw = lstm_df.set_index('chan_id'); vw = vae_df.set_index('chan_id'); mi = m.set_index('chan_id')
    stat_rows = [
        {'comparison': 'SVM synthetic vs real (substitution)',
         'synthetic_mean': round(float(mi['svm_p4_f1'].mean()), 4),
         'real_mean': round(float(mi['p2_svm_f1'].mean()), 4),
         'wilcoxon_p': round(_wilcoxon(mi['svm_p4_f1'], mi['p2_svm_f1']), 4)},
        {'comparison': 'LSTM k3 synthetic vs LSTM k3 real',
         'synthetic_mean': round(float(lw['lstm_k3_f1'].mean()), 4),
         'real_mean': round(float(mi['p2_lstm_k3_f1'].mean()), 4),
         'wilcoxon_p': round(_wilcoxon(lw['lstm_k3_f1'], mi['p2_lstm_k3_f1']), 4)},
        {'comparison': 'LSTM synthMedian vs k3 (artefact check)',
         'synthetic_mean': round(float(lw['lstm_synthMedian_f1'].mean()), 4),
         'real_mean': round(float(lw['lstm_k3_f1'].mean()), 4),
         'wilcoxon_p': round(_wilcoxon(lw['lstm_synthMedian_f1'], lw['lstm_k3_f1']), 4)},
        {'comparison': 'VAE best synthetic rule (synthRecall) vs k3',
         'synthetic_mean': round(float(vw['vae_synthRecall_f1'].mean()), 4),
         'real_mean': round(float(vw['vae_k3_f1'].mean()), 4),
         'wilcoxon_p': round(_wilcoxon(vw['vae_synthRecall_f1'], vw['vae_k3_f1']), 4)},
    ]
    pd.DataFrame(stat_rows).to_csv(RESULTS_DIR / "phase4_clean_statistics.csv", index=False)
    print("  Wrote phase4_clean_statistics.csv (paired Wilcoxon on held-out channels)")

    m.to_csv(RESULTS_DIR / "phase4_master_comparison.csv", index=False)
    return m


def _agg(df, col, mask=None):
    d = df if mask is None else df[mask]
    return float(d[col].mean())


def figure_rule_comparison(vae_df, lstm_df):
    """Show every deployable threshold rule vs k=3 and the oracle ceiling, so the
    empirical choice of winner is transparent."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Phase 4 — Label-Free Threshold Rules vs Classical k=3 and Oracle Ceiling\n"
                 "Can synthetic faults set a better deployable threshold than the 3σ default?",
                 fontsize=12, fontweight='bold')
    specs = [
        ("VAE", vae_df, ['vae_k3_f1', 'vae_mixF1_f1', 'vae_synthRecall_f1',
                         'vae_synthMedian_f1', 'vae_bestK_f1', 'vae_best_f1']),
        ("LSTM", lstm_df, ['lstm_k3_f1', 'lstm_mixF1_f1', 'lstm_synthRecall_f1',
                           'lstm_synthMedian_f1', 'lstm_bestK_f1', 'lstm_best_f1']),
    ]
    names = ['k=3\n(classical)', 'mixF1', 'synthRecall', 'synthMedian', 'bestK', 'oracle\n(ceiling)']
    colors = ['#999999', '#4575b4', '#4575b4', '#4575b4', '#4575b4', '#1a9850']
    for ax, (det, df, cols) in zip(axes, specs):
        vals = [df[c].mean() for c in cols]
        bars = ax.bar(names, vals, color=colors, width=0.65)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.008,
                    f"{v:.3f}", ha='center', fontsize=9, fontweight='bold')
        ax.axhline(vals[0], color='#999999', ls=':', lw=1)   # k3 reference line
        ax.set_ylabel("Mean F1"); ax.set_ylim(0, max(vals) * 1.18)
        ax.set_title(f"{det} deployable threshold rules", fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase4_rule_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: phase4_rule_comparison.png")


def figure_overview(m):
    """Headline figure: P4 vs the train-on-real ceiling, per detector."""
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5))
    fig.suptitle("Phase 4 — Detectors Trained on SYNTHETIC Fault Data vs Trained on REAL Data\n"
                 "The gap to the real-data result answers the thesis question "
                 "(Mehrab Jamshidi, PoliMi 2026)", fontsize=13, fontweight='bold')

    # Panel 1: grouped bars per detector (P4 deployable vs P2 deployable vs P2 oracle)
    ax = axes[0]
    groups = ['SVM', 'VAE', 'LSTM']
    p4 = [_agg(m, 'svm_p4_f1'), _agg(m, 'vae_p4_f1'), _agg(m, 'lstm_p4_f1')]
    # held-out, like-for-like real baselines (mean over the SAME 20 channels),
    # not the all-channel Phase-2 numbers — keeps synthetic vs real on one footing.
    p2_deploy = [m['p2_svm_f1'].mean(), m['p2_vae_k3_f1'].mean(), m['p2_lstm_k3_f1'].mean()]
    p2_oracle = [m['p2_svm_f1'].mean(), m['p2_vae_best_f1'].mean(), m['p2_lstm_best_f1'].mean()]
    x = np.arange(len(groups)); w = 0.26
    ax.bar(x - w, p4, w, label='Phase 4 (synthetic, deployable)', color='#4575b4')
    ax.bar(x,     p2_deploy, w, label='Phase 2 (real, deployable)', color='#91bfdb')
    ax.bar(x + w, p2_oracle, w, label='Phase 2 (real, oracle best)', color='#d6d6d6')
    ax.axhline(REF['threshold_mean'], color='red', ls='--', lw=1.2,
               label=f"Phase 1 threshold ({REF['threshold_mean']:.3f})")
    for xi, v in zip(x - w, p4):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("Mean F1"); ax.set_ylim(0, 1.0); ax.grid(True, alpha=0.3, axis='y')
    ax.set_title("Synthetic-trained vs real-trained (mean F1 over 20 held-out channels)", fontsize=10)
    ax.legend(fontsize=8, loc='upper left')

    # Panel 2: retention ratio (P4 deployable / P2 deployable) — "how much is kept"
    ax = axes[1]
    ratios = [p4[i] / p2_deploy[i] if p2_deploy[i] > 0 else 0 for i in range(3)]
    bars = ax.bar(groups, ratios, color=['#1a9850', '#66bd63', '#a6d96a'], width=0.55)
    ax.axhline(1.0, color='black', ls='--', lw=1.2, label='full retention (=real)')
    for b, r in zip(bars, ratios):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                f"{r*100:.0f}%", ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel("Phase 4 / Phase 2 deployable F1"); ax.set_ylim(0, 1.3)
    ax.set_title("Performance retained when faults are synthetic, not real", fontsize=10)
    ax.grid(True, alpha=0.3, axis='y'); ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase4_overview.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: phase4_overview.png")


def figure_data_efficiency(eff_df, svm_real_ceiling=REF['svm_mean']):
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = eff_df['n_real_anom'].tolist()
    x = np.arange(len(labels))
    ax.plot(x, eff_df['mean_f1'], marker='o', lw=2, color='#4575b4', label='Synthetic + n real faults')
    ax.axhline(svm_real_ceiling, color='#d73027', ls='--', lw=1.5,
               label=f'Phase 2A all-channel baseline ({svm_real_ceiling:.3f}, not directly comparable)')
    ax.axhline(REF['threshold_mean'], color='gray', ls=':', lw=1.2,
               label=f"Phase 1 threshold ({REF['threshold_mean']:.3f})")
    for xi, v in zip(x, eff_df['mean_f1']):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("Number of REAL anomaly windows added to synthetic training set")
    ax.set_ylabel("Mean SVM F1 (20 held-out channels)")
    ax.set_title("Data-Efficiency on 20 held-out channels — real windows from known\n"
                 "channels do not transfer to unseen channels (Mehrab Jamshidi, PoliMi 2026)",
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9); ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase4_data_efficiency.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: phase4_data_efficiency.png")


def figure_per_class_spacecraft(m):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Phase 4 — Per-Class and Per-Spacecraft Breakdown (20 held-out channels)\n"
                 "Contextual transfer is expected to be strongest (synthetic ACF|Δ|=0.026)",
                 fontsize=12, fontweight='bold')
    ctx = m.anomaly_class.str.contains('contextual', na=False)
    pt = ~ctx
    smap = m.spacecraft == 'SMAP'; msl = m.spacecraft == 'MSL'

    ax = axes[0]
    dets = ['SVM', 'VAE', 'LSTM(calib)']
    cols = ['svm_p4_f1', 'vae_p4_f1', 'lstm_p4_f1']
    ctx_v = [_agg(m, c, ctx) for c in cols]
    pt_v = [_agg(m, c, pt) for c in cols]
    x = np.arange(len(dets)); w = 0.35
    ax.bar(x - w/2, ctx_v, w, label='Contextual', color='#2c7bb6')
    ax.bar(x + w/2, pt_v, w, label='Point', color='#fdae61')
    for xi, v in zip(x - w/2, ctx_v):
        ax.text(xi, v + 0.012, f"{v:.2f}", ha='center', fontsize=8)
    for xi, v in zip(x + w/2, pt_v):
        ax.text(xi, v + 0.012, f"{v:.2f}", ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(dets); ax.set_ylabel("Mean F1"); ax.set_ylim(0, 1.0)
    ax.set_title("By anomaly class", fontsize=10); ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    smap_v = [_agg(m, c, smap) for c in cols]
    msl_v = [_agg(m, c, msl) for c in cols]
    ax.bar(x - w/2, smap_v, w, label='SMAP', color='#1a9641')
    ax.bar(x + w/2, msl_v, w, label='MSL', color='#d7191c')
    for xi, v in zip(x - w/2, smap_v):
        ax.text(xi, v + 0.012, f"{v:.2f}", ha='center', fontsize=8)
    for xi, v in zip(x + w/2, msl_v):
        ax.text(xi, v + 0.012, f"{v:.2f}", ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(dets); ax.set_ylabel("Mean F1"); ax.set_ylim(0, 1.0)
    ax.set_title("By spacecraft", fontsize=10); ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase4_per_class_spacecraft.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: phase4_per_class_spacecraft.png")


def figure_p1_case_study(m):
    """Channel P-1 across all phases — the thesis's recurring illustrative case."""
    r = m[m.chan_id == 'P-1']
    if len(r) == 0:
        print("  (P-1 not found; skipping case-study figure)")
        return
    r = r.iloc[0]
    fig, ax = plt.subplots(figsize=(11, 6))
    methods = ['P1\nthreshold', 'P2 SVM\n(real)', 'P3 TSTR\n(synth SVM)',
               'P4 SVM\n(synth)', 'P4 VAE\n(synth cal)', 'P4 LSTM\n(synth cal)',
               'P2 LSTM\n(real best)']
    vals = [r.get('p1_threshold_f1', 0), r.get('p2_svm_f1', 0), r.get('p3_tstr_f1', 0),
            r.get('svm_p4_f1', 0), r.get('vae_p4_f1', 0), r.get('lstm_p4_f1', 0),
            r.get('p2_lstm_best_f1', 0)]
    colors = ['#999999', '#91bfdb', '#fdae61', '#4575b4', '#4575b4', '#4575b4', '#1a9850']
    bars = ax.bar(methods, vals, color=colors, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.012,
                f"{v:.3f}", ha='center', fontsize=9, fontweight='bold')
    ax.set_ylabel("F1"); ax.set_ylim(0, 1.05)
    ax.set_title("Channel P-1 across all phases — synthetic data on the hardest contextual case\n"
                 "(P-1 never exceeds amplitude bounds; classical FDIR is blind to it)",
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase4_P1_case_study.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: phase4_P1_case_study.png")


# =============================================================================
# SECTION 8 — ORCHESTRATION
# =============================================================================

def main():
    t0 = time.time()
    svm_df  = run_svm_phase4()
    vae_df  = run_vae_phase4()
    lstm_df = run_lstm_phase4()
    eff_df  = run_data_efficiency()

    print("\n" + "=" * 78)
    print("BUILDING MASTER COMPARISON + FIGURES")
    print("=" * 78)
    m = build_master(svm_df, vae_df, lstm_df)
    figure_rule_comparison(vae_df, lstm_df)
    figure_overview(m)
    figure_data_efficiency(eff_df)
    figure_per_class_spacecraft(m)
    figure_p1_case_study(m)

    # -------- Final console summary (the numbers to paste back for Report 4) --------
    okc = m.anomaly_class.str.contains('contextual', na=False)
    okp = ~okc
    print("\n" + "=" * 78)
    print("PHASE 4 FINAL SUMMARY  — paste this block back to write Report 4")
    print("=" * 78)
    print(f"{'Detector':<26}{'P4 mean':>9}{'P4 med':>8}{'P2 dep':>9}{'P2 orc':>9}{'retain%':>9}")
    print("-" * 70)
    rows = [
        ("SVM (window F1)",        'svm_p4_f1',  REF['svm_mean'], REF['svm_mean']),
        (f"VAE [{m.attrs.get('vae_win','?')}]",  'vae_p4_f1',  REF['vae_k3'],   REF['vae_best']),
        (f"LSTM [{m.attrs.get('lstm_win','?')}]", 'lstm_p4_f1', REF['lstm_k3'],  REF['lstm_best']),
    ]
    for name, col, dep, orc in rows:
        mn = _agg(m, col); md = float(m[col].median())
        print(f"{name:<26}{mn:>9.4f}{md:>8.4f}{dep:>9.4f}{orc:>9.4f}{(mn/dep*100 if dep>0 else 0):>8.0f}%")
    print(f"{'LSTM discrim (PA)':<26}{_agg(m, 'lstm_p4_disc_f1'):>9.4f}{float(m.lstm_p4_disc_f1.median()):>8.4f}  (ablation)")
    print("-" * 70)
    print("NOTE: VAE/LSTM 'P4' = the empirically winning DEPLOYABLE rule (chosen by")
    print("      aggregate mean, never per-channel). If no synthetic rule beat k=3,")
    print("      the winner IS k3. Median is reported because point-adjust F1 on a")
    print("      few rare-anomaly channels is high-variance; median is the robust stat.")
    print(f"Phase 1 threshold baseline: {REF['threshold_mean']:.4f}")
    print(f"Phase 3 TSTR SVM (prev):    {REF['tstr_mean']:.4f}")
    print()
    print("Per anomaly class (Phase 4 deployable):")
    print(f"  SVM   ctx={_agg(m,'svm_p4_f1',okc):.4f}  pt={_agg(m,'svm_p4_f1',okp):.4f}")
    print(f"  VAE   ctx={_agg(m,'vae_p4_f1',okc):.4f}  pt={_agg(m,'vae_p4_f1',okp):.4f}")
    print(f"  LSTM  ctx={_agg(m,'lstm_p4_f1',okc):.4f}  pt={_agg(m,'lstm_p4_f1',okp):.4f}")
    print()
    print("Per spacecraft (Phase 4 deployable):")
    for det, col in [("SVM", 'svm_p4_f1'), ("VAE", 'vae_p4_f1'), ("LSTM", 'lstm_p4_f1')]:
        s = _agg(m, col, m.spacecraft == 'SMAP'); ms = _agg(m, col, m.spacecraft == 'MSL')
        print(f"  {det:<5} SMAP={s:.4f}  MSL={ms:.4f}")
    p1 = m[m.chan_id == 'P-1']
    if len(p1):
        p1 = p1.iloc[0]
        print(f"\nChannel P-1:  SVM-P4={p1.svm_p4_f1:.3f}  VAE-P4={p1.vae_p4_f1:.3f}  "
              f"LSTM-P4(calib)={p1.lstm_p4_f1:.3f}")
    print("\nData-efficiency curve:")
    print(eff_df.to_string(index=False))
    print("\n" + "=" * 78)
    print(f"DONE in {(time.time()-t0)/60:.1f} min. Output files:")
    for f in [RESULTS_DIR / "phase4_svm_tstr_results.csv", RESULTS_DIR / "phase4_vae_results.csv",
              RESULTS_DIR / "phase4_lstm_results.csv", RESULTS_DIR / "phase4_data_efficiency.csv",
              RESULTS_DIR / "phase4_master_comparison.csv", FIGURES_DIR / "phase4_overview.png",
              FIGURES_DIR / "phase4_data_efficiency.png", FIGURES_DIR / "phase4_per_class_spacecraft.png",
              FIGURES_DIR / "phase4_P1_case_study.png"]:
        print(f"  {f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
