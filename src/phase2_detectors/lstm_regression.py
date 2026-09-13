"""
=============================================================================
LSTM Regression — Telemetry Forecasting and Anomaly Detection
NASA SMAP/MSL Dataset

Author: Mehrab Jamshidi — Politecnico di Milano
Supervisor: Prof. Andrea Colagrossi

PURPOSE
-------
This script implements the REGRESSION component of telemetry analysis.

WHAT IS REGRESSION HERE?
-------------------------
Regression in the context of time-series telemetry means:
  Given the last W timesteps of a sensor signal (the input window),
  PREDICT the next N timesteps (the forecast horizon).

This is called one-step-ahead or multi-step-ahead forecasting.
The model learns the normal dynamics of the signal during training
(on the nominal split only). During inference on the test split:
  - If the signal continues to follow its normal dynamics: prediction
    error is LOW → nominal
  - If the signal deviates from its learned dynamics (anomaly):
    prediction error is HIGH → anomalous

This is a fundamentally different approach from the VAE:
  - VAE asks: "Can I RECONSTRUCT this window from a compressed version?"
  - LSTM asks: "Can I PREDICT the next timestep given the history?"

Both use reconstruction/prediction error as the anomaly score.
The LSTM captures TEMPORAL DYNAMICS (how the signal evolves over time)
while the VAE captures DISTRIBUTIONAL STRUCTURE (what the signal looks like).
Together they provide complementary anomaly sensitivity.

ARCHITECTURE
------------
Input:  sequence of LOOKBACK=64 timesteps → shape (batch, 64, 1)
LSTM:   2 layers, hidden_size=64, dropout=0.1
Output: linear layer → FORECAST=1 timestep ahead

We use one-step-ahead prediction (FORECAST=1) because:
  1. It is the simplest and most interpretable regression formulation
  2. The prediction error at each timestep is directly comparable to the
     ground truth at the next timestep
  3. It is the basis of the Hundman et al. (2018) approach
     (they use an LSTM predictor, then threshold the prediction errors)

EVALUATION
----------
Same point-adjust protocol as vae_point_adjust.py to ensure all methods
are evaluated on the same basis and compared fairly with Hundman 2018.

WHY LSTM AND NOT TRANSFORMER?
------------------------------
The Anomaly Transformer (Xu et al. 2022) is the upper bound reference
in the thesis. Using an LSTM here keeps the regression baseline simple
and consistent with what Hundman 2018 used, so the comparison is clean.
The Anomaly Transformer comparison will be discussed in Phase 4.
=============================================================================
"""

import numpy as np
import pandas as pd
import os
import sys
import re
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (  # noqa: E402
    DATASET_ROOT, LABELS_FILE, TEST_FOLDER, TRAIN_FOLDER,
    RESULTS_DIR, FIGURES_DIR, require_dataset,
)

require_dataset()

LOOKBACK    = 64    # number of past timesteps used as input
FORECAST    = 1     # number of future timesteps to predict (one-step-ahead)
HIDDEN_SIZE = 64    # LSTM hidden state dimension
NUM_LAYERS  = 2     # LSTM depth
DROPOUT     = 0.1   # dropout between LSTM layers
BATCH_SIZE  = 64
EPOCHS      = 100
LR          = 1e-3
PATIENCE    = 20    # early stopping patience
SEED        = 42
K_VALUES    = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS_CSV = RESULTS_DIR / "lstm_regression_results.csv"
FIGURE_FILE = FIGURES_DIR / "lstm_regression_overview.png"
P1_FIGURE   = FIGURES_DIR / "lstm_P1_prediction.png"

print("=" * 70)
print("LSTM REGRESSION — TELEMETRY FORECASTING")
print("NASA SMAP/MSL — One-step-ahead prediction + point-adjust evaluation")
print("=" * 70)
print(f"Device:      {DEVICE}")
print(f"Lookback:    {LOOKBACK} timesteps")
print(f"Forecast:    {FORECAST} timestep ahead")
print(f"LSTM layers: {NUM_LAYERS} x {HIDDEN_SIZE} hidden units")

# =============================================================================
# HELPERS
# =============================================================================

def parse_windows(seq_str):
    numbers = re.findall(r'\d+', str(seq_str))
    numbers = [int(n) for n in numbers]
    return [[numbers[i], numbers[i+1]] for i in range(0, len(numbers)-1, 2)]

def build_ground_truth_vector(num_values, anomaly_windows):
    y = np.zeros(num_values, dtype=int)
    for start, end in anomaly_windows:
        y[max(0,start):min(num_values,end)] = 1
    return y

def make_sequences(signal, lookback, forecast):
    """
    Build (X, y) pairs for supervised sequence prediction.

    For each position t in the signal:
      X[t] = signal[t : t+lookback]       shape: (lookback,)
      y[t] = signal[t+lookback : t+lookback+forecast]  shape: (forecast,)

    The model learns: given the last `lookback` values, predict the
    next `forecast` values.

    INPUT:  signal of shape (T,)
    OUTPUT: X of shape (n_samples, lookback),
            y of shape (n_samples, forecast)
    """
    X, y = [], []
    for t in range(len(signal) - lookback - forecast + 1):
        X.append(signal[t : t+lookback])
        y.append(signal[t+lookback : t+lookback+forecast])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def point_adjust(y_true_ts, y_pred_ts):
    """Apply point-adjust rule (Hundman 2018): if any timestep in an
    anomaly window is detected, mark the whole window as detected."""
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


def evaluate_point_adjust(errors, y_true_ts, threshold):
    """Threshold errors, apply point-adjust, return P/R/F1."""
    # errors and y_true_ts may differ in length by LOOKBACK
    # (first LOOKBACK timesteps have no prediction)
    # Align: errors correspond to timesteps [LOOKBACK, LOOKBACK+len(errors)]
    offset = LOOKBACK
    T = len(y_true_ts)
    pred_len = min(len(errors), T - offset)

    y_pred_full = np.zeros(T, dtype=int)
    y_pred_full[offset:offset+pred_len] = (errors[:pred_len] > threshold).astype(int)

    if len(np.unique(y_true_ts)) < 2:
        return 0.0, 0.0, 0.0

    y_pred_adj = point_adjust(y_true_ts, y_pred_full)
    p  = precision_score(y_true_ts, y_pred_adj, zero_division=0)
    r  = recall_score(y_true_ts, y_pred_adj, zero_division=0)
    f1 = f1_score(y_true_ts, y_pred_adj, zero_division=0)
    return p, r, f1

# =============================================================================
# LSTM MODEL
# =============================================================================

class LSTMPredictor(nn.Module):
    """
    Two-layer LSTM for one-step-ahead telemetry prediction.

    Architecture:
      Input: (batch, lookback, 1) — univariate time series
      LSTM layer 1: hidden_size=64, returns all hidden states
      LSTM layer 2: hidden_size=64, returns last hidden state only
      Linear output: maps hidden_size → forecast timesteps

    The LSTM processes the sequence from left to right, updating its
    hidden state h_t at each timestep. The final hidden state h_{lookback}
    encodes the entire history seen in the lookback window and is used
    to predict the next FORECAST timesteps.

    WHY LSTM AND NOT GRU?
    LSTM has both cell state (long-term memory) and hidden state (short-term)
    which makes it better at capturing both slow drifts and fast oscillations
    in telemetry signals. For typical telemetry with mixed temporal scales,
    LSTM outperforms GRU slightly.
    """
    def __init__(self, hidden_size=64, num_layers=2, dropout=0.1, forecast=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True    # input shape: (batch, seq, features)
        )
        self.fc = nn.Linear(hidden_size, forecast)

    def forward(self, x):
        # x shape: (batch, lookback, 1)
        lstm_out, _ = self.lstm(x)      # (batch, lookback, hidden_size)
        last_hidden  = lstm_out[:, -1, :]  # (batch, hidden_size) — last timestep
        prediction   = self.fc(last_hidden)  # (batch, forecast)
        return prediction

# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.unsqueeze(-1).to(device)  # (batch, lookback, 1)
        y_batch = y_batch.to(device)                # (batch, forecast)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = nn.functional.mse_loss(pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.unsqueeze(-1).to(device)
            y_batch = y_batch.to(device)
            pred = model(X_batch)
            total_loss += nn.functional.mse_loss(pred, y_batch).item()
    return total_loss / len(loader)

def compute_prediction_errors(model, X_tensor, y_tensor, device):
    """
    Compute per-sample prediction MSE on the test sequences.
    Returns array of shape (n_samples,) — one error per predicted timestep.
    """
    model.eval()
    errors = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), 256):
            Xb = X_tensor[i:i+256].unsqueeze(-1).to(device)
            yb = y_tensor[i:i+256].to(device)
            pred = model(Xb)
            mse  = ((pred - yb)**2).mean(dim=1)  # per-sample MSE
            errors.append(mse.cpu().numpy())
    return np.concatenate(errors)

# =============================================================================
# MAIN LOOP
# =============================================================================

print(f"\nLoading labels...")
labels = pd.read_csv(LABELS_FILE)
print(f"  {len(labels)} channels\n")

results   = []
p1_detail = {}

for idx, row in labels.iterrows():
    chan       = row["chan_id"]
    sc         = row["spacecraft"]
    anom_class = row["class"]
    windows_gt = parse_windows(row["anomaly_sequences"])

    test_file  = os.path.join(TEST_FOLDER,  f"{chan}.npy")
    train_file = os.path.join(TRAIN_FOLDER, f"{chan}.npy")
    if not os.path.exists(test_file) or not os.path.exists(train_file):
        continue

    train_raw = np.load(train_file)[:, 0]
    test_raw  = np.load(test_file)[:,  0]
    test_len  = len(test_raw)

    # Normalize on training stats
    mu_tr = train_raw.mean()
    sd_tr = train_raw.std()
    if sd_tr < 1e-9: sd_tr = 1.0
    train_norm = (train_raw - mu_tr) / sd_tr
    test_norm  = (test_raw  - mu_tr) / sd_tr

    # Ground truth vector (timestep level)
    y_true_ts = build_ground_truth_vector(test_len, windows_gt)

    # Build sequences
    X_train_s, y_train_s = make_sequences(train_norm, LOOKBACK, FORECAST)
    X_test_s,  y_test_s  = make_sequences(test_norm,  LOOKBACK, FORECAST)

    if len(X_train_s) < 20:
        results.append({"chan_id": chan, "spacecraft": sc,
                         "anomaly_class": anom_class, "epochs_run": 0,
                         "lstm_best_f1": 0.0, "lstm_k3_f1": 0.0,
                         "lstm_best_precision": 0.0, "lstm_best_recall": 0.0,
                         "lstm_best_threshold": "none", "note": "too_short"})
        continue

    # Train/val split on training sequences
    n_val   = max(1, int(0.1 * len(X_train_s)))
    X_tr    = torch.FloatTensor(X_train_s[:-n_val])
    y_tr    = torch.FloatTensor(y_train_s[:-n_val])
    X_va    = torch.FloatTensor(X_train_s[-n_val:])
    y_va    = torch.FloatTensor(y_train_s[-n_val:])
    X_te    = torch.FloatTensor(X_test_s)
    y_te    = torch.FloatTensor(y_test_s)

    tr_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(TensorDataset(X_va, y_va), batch_size=BATCH_SIZE)

    # Train LSTM
    model     = LSTMPredictor(HIDDEN_SIZE, NUM_LAYERS, DROPOUT, FORECAST).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val   = float('inf')
    patience_c = 0
    epochs_run = 0

    for epoch in range(EPOCHS):
        train_epoch(model, tr_loader, optimizer, DEVICE)
        val_loss = eval_epoch(model, va_loader, DEVICE)
        scheduler.step(val_loss)
        epochs_run += 1
        if val_loss < best_val - 1e-6:
            best_val   = val_loss
            patience_c = 0
        else:
            patience_c += 1
            if patience_c >= PATIENCE: break

    # Compute prediction errors
    train_errors = compute_prediction_errors(model, X_tr, y_tr, DEVICE)
    test_errors  = compute_prediction_errors(model, X_te, y_te, DEVICE)

    # Calibrate thresholds from training errors
    mu_e  = train_errors.mean()
    sd_e  = train_errors.std()
    thresholds = {f"k{k}": mu_e + k * sd_e for k in K_VALUES}
    for pct in [90, 95, 99]:
        thresholds[f"p{pct}"] = np.percentile(train_errors, pct)

    # Evaluate with point-adjust
    best_f1  = 0.0; best_p = 0.0; best_r = 0.0; best_name = "none"
    for name, thr in thresholds.items():
        p, r, f1 = evaluate_point_adjust(test_errors, y_true_ts, thr)
        if f1 > best_f1:
            best_f1 = f1; best_p = p; best_r = r; best_name = name

    p_k3, r_k3, f1_k3 = evaluate_point_adjust(
        test_errors, y_true_ts, mu_e + 3.0*sd_e)

    results.append({
        "chan_id":              chan,
        "spacecraft":          sc,
        "anomaly_class":       anom_class,
        "epochs_run":          epochs_run,
        "lstm_best_precision": round(best_p,  4),
        "lstm_best_recall":    round(best_r,  4),
        "lstm_best_f1":        round(best_f1, 4),
        "lstm_best_threshold": best_name,
        "lstm_k3_precision":   round(p_k3, 4),
        "lstm_k3_recall":      round(r_k3, 4),
        "lstm_k3_f1":          round(f1_k3, 4),
        "note":                "ok"
    })

    # Save P-1 details for visualization
    if chan == "P-1":
        # Get actual predictions for plotting
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_te), 256):
                Xb = X_te[i:i+256].unsqueeze(-1).to(DEVICE)
                preds.append(model(Xb).cpu().numpy())
        preds = np.concatenate(preds).flatten()
        p1_detail = {
            "test_norm":    test_norm,
            "preds":        preds,
            "test_errors":  test_errors,
            "y_true_ts":    y_true_ts,
            "threshold_k3": mu_e + 3.0*sd_e,
            "best_f1":      best_f1,
            "windows_gt":   windows_gt,
            "train_len":    len(train_raw),
        }
        print(f"\n  P-1 LSTM RESULT:")
        print(f"    Epochs: {epochs_run}")
        print(f"    Best F1 (PA): {best_f1:.4f}  k=3 F1 (PA): {f1_k3:.4f}")
        print(f"    (VAE PA best was 0.102, threshold was 0.011)")

    if (idx + 1) % 10 == 0:
        print(f"  [{idx+1:3d}/82] {chan:6s} | LSTM F1={best_f1:.3f} | "
              f"k3 F1={f1_k3:.3f} | epochs={epochs_run}")

# =============================================================================
# AGGREGATE RESULTS
# =============================================================================

df = pd.DataFrame(results)
ok = df[df.note == "ok"]

print("\n" + "="*70)
print("AGGREGATE RESULTS — LSTM REGRESSION (POINT-ADJUST)")
print("="*70)

smap = ok[ok.spacecraft=="SMAP"]
msl  = ok[ok.spacecraft=="MSL"]

for col, label in [("lstm_best_f1","LSTM best threshold"),
                   ("lstm_k3_f1",  "LSTM k=3 threshold")]:
    v = ok[col]
    print(f"\n{label}:")
    print(f"  All  — mean={v.mean():.4f}  median={v.median():.4f}  "
          f"F1>0:{(v>0).sum()}/{len(ok)}  F1>0.5:{(v>0.5).sum()}/{len(ok)}")
    print(f"  SMAP — mean={smap[col].mean():.4f}")
    print(f"  MSL  — mean={msl[col].mean():.4f}")

ctx = ok[ok.anomaly_class.str.contains("contextual", na=False)]
pt  = ok[~ok.anomaly_class.str.contains("contextual", na=False)]
print(f"\nBy anomaly type (LSTM best):")
print(f"  Contextual (n={len(ctx)}): mean F1={ctx.lstm_best_f1.mean():.4f}")
print(f"  Point      (n={len(pt)}): mean F1={pt.lstm_best_f1.mean():.4f}")

print(f"\nHundman 2018 LSTM (published, point-adjust):")
print(f"  SMAP=0.7519  MSL=0.6955")

ok.to_csv(RESULTS_CSV, index=False)
print(f"\nSaved: {RESULTS_CSV}")

# =============================================================================
# P-1 VISUALIZATION
# =============================================================================

if p1_detail:
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.suptitle(
        "LSTM Regression — SMAP Channel P-1\n"
        "One-step-ahead prediction error as anomaly score (Mehrab Jamshidi, PoliMi 2026)",
        fontsize=13, fontweight="bold"
    )

    tn     = p1_detail["test_norm"]
    preds  = p1_detail["preds"]
    errs   = p1_detail["test_errors"]
    gt     = p1_detail["windows_gt"]
    tl     = p1_detail["train_len"]
    thr_k3 = p1_detail["threshold_k3"]
    bf1    = p1_detail["best_f1"]

    test_time = np.arange(tl, tl + len(tn))
    pred_time = np.arange(tl + LOOKBACK, tl + LOOKBACK + len(preds))

    ax = axes[0]
    ax.plot(test_time, tn, color="#2c7bb6", linewidth=0.6, label="True signal")
    ax.plot(pred_time, preds, color="#d73027", linewidth=0.6,
            alpha=0.7, label="LSTM prediction")
    for i, (s, e) in enumerate(gt):
        ax.axvspan(s+tl, e+tl, alpha=0.25, color="red",
                   label="Anomaly window" if i==0 else "")
    ax.set_title("Test signal vs LSTM one-step-ahead prediction", fontsize=10)
    ax.set_ylabel("Normalized value"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(pred_time, np.abs(tn[LOOKBACK:LOOKBACK+len(preds)] - preds),
            color="#2c7bb6", linewidth=0.6, label="|prediction error|")
    for i, (s, e) in enumerate(gt):
        ax.axvspan(s+tl, e+tl, alpha=0.25, color="red",
                   label="Anomaly window" if i==0 else "")
    ax.set_title("Absolute prediction error (|true - predicted|)", fontsize=10)
    ax.set_ylabel("Error"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[2]
    err_time = np.arange(tl + LOOKBACK, tl + LOOKBACK + len(errs))
    ax.plot(err_time, errs, color="#2c7bb6", linewidth=0.6,
            label="Prediction MSE (per timestep)")
    ax.axhline(thr_k3, color="red", linestyle="--", linewidth=1.5,
               label=f"k=3 threshold ({thr_k3:.4f})")
    for i, (s, e) in enumerate(gt):
        ax.axvspan(s+tl, e+tl, alpha=0.25, color="red",
                   label="Anomaly window" if i==0 else "")
    ax.set_title(f"LSTM Prediction MSE — P-1 | Best PA F1={bf1:.3f}", fontsize=10)
    ax.set_ylabel("MSE"); ax.set_xlabel("Time step (global)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(P1_FIGURE, dpi=150, bbox_inches="tight")
    print(f"P-1 figure saved: {P1_FIGURE}")

# =============================================================================
# OVERVIEW FIGURE
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(
    "LSTM Regression Results — All 82 NASA SMAP/MSL Channels\n"
    "One-step-ahead prediction, point-adjust evaluation (Mehrab Jamshidi, PoliMi 2026)",
    fontsize=13, fontweight="bold"
)

ax = axes[0]
df_s = ok.sort_values("lstm_best_f1", ascending=False)
colors = ["#2c7bb6" if "contextual" in str(r.anomaly_class) else "#fdae61"
          for _, r in df_s.iterrows()]
ax.bar(range(len(df_s)), df_s.lstm_best_f1.values, color=colors, width=0.8)
ax.axhline(0.7519, color="red",    linestyle="--", linewidth=1.5,
           label="Hundman SMAP=0.752")
ax.axhline(0.6955, color="orange", linestyle="--", linewidth=1.5,
           label="Hundman MSL=0.696")
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(facecolor="#2c7bb6", label="Contextual"),
    Patch(facecolor="#fdae61", label="Point"),
    plt.Line2D([0],[0], color="red",    linestyle="--", label="Hundman SMAP=0.752"),
    plt.Line2D([0],[0], color="orange", linestyle="--", label="Hundman MSL=0.696"),
], fontsize=8)
ax.set_title("LSTM F1 per channel (sorted, point-adjust)", fontsize=11)
ax.set_ylabel("F1"); ax.set_xlabel("Channel rank")
ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3)

ax = axes[1]
ctx_m = ok[ok.anomaly_class.str.contains("contextual", na=False)].lstm_best_f1.mean()
pt_m  = ok[~ok.anomaly_class.str.contains("contextual", na=False)].lstm_best_f1.mean()
cats  = ["LSTM\n(Contextual)", "LSTM\n(Point)", "Hundman\n(SMAP)", "Hundman\n(MSL)"]
vals  = [ctx_m, pt_m, 0.7519, 0.6955]
cols  = ["#2c7bb6", "#fdae61", "#d73027", "#d73027"]
bars  = ax.bar(cats, vals, color=cols, width=0.5, alpha=0.85)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
            f"{val:.3f}", ha="center", fontsize=11, fontweight="bold")
ax.set_title("LSTM mean F1 by anomaly type vs Hundman 2018", fontsize=11)
ax.set_ylabel("Mean F1"); ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURE_FILE, dpi=150, bbox_inches="tight")
print(f"Overview figure saved: {FIGURE_FILE}")

print("\n" + "="*70)
print("DONE.")
print(f"Files: {RESULTS_CSV}, {FIGURE_FILE}, {P1_FIGURE}")
print("Next: upload results and run report2_final.py")
print("="*70)
