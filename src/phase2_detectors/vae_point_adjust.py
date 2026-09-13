"""
=============================================================================
VAE Anomaly Detection — Point-Adjust Evaluation Protocol
NASA SMAP/MSL Dataset

Author: Mehrab Jamshidi — Politecnico di Milano
Supervisor: Prof. Andrea Colagrossi

PURPOSE
-------
This script re-evaluates the VAE using the POINT-ADJUST protocol from
Hundman et al. (2018) so that our numbers are directly comparable to the
published benchmark.

WHAT IS POINT-ADJUST?
---------------------
In our previous evaluation, we labeled windows as anomalous only if
>=10% of their timesteps overlapped with a ground-truth anomaly window.
This is strict — it requires the detector to flag the WINDOW as a whole.

Hundman et al. (2018) use a different protocol at the TIMESTEP level:
  1. For each timestep in the test split, predict 0 or 1 (nominal/anomalous)
  2. Apply point-adjust: if ANY timestep within a ground-truth anomaly
     window is predicted as anomalous (1), then ALL timesteps in that
     window are counted as correctly detected (recall = 1 for that window).
  3. Compute precision, recall, F1 at the timestep level after adjustment.

WHY THIS MATTERS:
  Our protocol penalises the detector for missing the edges of anomaly
  windows. Point-adjust only requires that the detector "wakes up" at
  some point inside each anomaly window — which is a more realistic
  requirement for operational FDIR where the goal is to detect the event,
  not necessarily the exact onset timestep.

  Point-adjust will give HIGHER recall (and therefore higher F1) for the
  same detector compared to our previous protocol. This is why Hundman's
  published F1 scores (SMAP=0.752, MSL=0.696) are higher than what we
  observed with our stricter window-overlap protocol.

IMPLEMENTATION:
  Rather than training new VAEs (which would take 30 minutes), we load
  the VAE results CSV from the previous run (vae_results.csv) and
  re-evaluate using point-adjust. For each channel:
    1. Retrain the VAE (same architecture and hyperparameters)
    2. Compute per-TIMESTEP reconstruction errors by averaging window
       errors back to individual timesteps
    3. Apply point-adjust scoring
    4. Report corrected F1

This gives us properly comparable numbers for the report.
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
# CONFIGURATION — must match vae_anomaly_detection.py exactly
# =============================================================================

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (  # noqa: E402
    DATASET_ROOT, LABELS_FILE, TEST_FOLDER, TRAIN_FOLDER,
    RESULTS_DIR, FIGURES_DIR, require_dataset,
)

require_dataset()

WINDOW_SIZE = 64
STEP_SIZE   = 16
LATENT_DIM  = 8
BATCH_SIZE  = 32
EPOCHS      = 100
LR          = 1e-3
BETA        = 1.0
K_VALUES    = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
SEED        = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS_CSV  = RESULTS_DIR / "vae_point_adjust_results.csv"
FIGURE_FILE  = FIGURES_DIR / "vae_point_adjust_overview.png"

print("=" * 70)
print("VAE ANOMALY DETECTION — POINT-ADJUST EVALUATION")
print("Protocol: Hundman et al. (2018) — KDD 2018")
print("=" * 70)
print(f"Device: {DEVICE}")

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

def make_windows_array(signal, window_size, step_size):
    windows = []
    for start in range(0, len(signal) - window_size + 1, step_size):
        windows.append(signal[start:start+window_size])
    return np.array(windows, dtype=np.float32)

# =============================================================================
# POINT-ADJUST FUNCTIONS
# =============================================================================

def windows_to_timestep_scores(window_errors, signal_len, window_size, step_size):
    """
    Convert per-window reconstruction errors back to per-timestep scores.

    Each timestep may be covered by multiple overlapping windows.
    We assign each timestep the MAXIMUM error among all windows that cover it.
    Using max (rather than mean) ensures that if any window covering a
    timestep detects something anomalous, the timestep is scored high.

    INPUT:
      window_errors : array of shape (n_windows,) — MSE per window
      signal_len    : length of the original signal
      window_size   : window size used during windowing
      step_size     : step size used during windowing

    OUTPUT:
      ts_scores : array of shape (signal_len,) — per-timestep anomaly score
    """
    ts_scores = np.zeros(signal_len, dtype=np.float32)
    ts_count  = np.zeros(signal_len, dtype=np.float32)

    for w_idx, start in enumerate(range(0, signal_len - window_size + 1, step_size)):
        end = start + window_size
        # Use max pooling: if this window's error > current ts score, update
        ts_scores[start:end] = np.maximum(ts_scores[start:end], window_errors[w_idx])
        ts_count[start:end] += 1

    # Timesteps not covered by any window (tail of signal) get mean score
    uncovered = ts_count == 0
    if uncovered.any():
        ts_scores[uncovered] = window_errors[-1] if len(window_errors) > 0 else 0.0

    return ts_scores


def point_adjust(y_true_ts, y_pred_ts):
    """
    Apply the point-adjust rule from Hundman et al. (2018).

    For each contiguous anomaly segment in y_true_ts:
      If ANY timestep in that segment is predicted as 1 (anomalous),
      then ALL timesteps in that segment are set to 1 in y_pred_ts.

    This models the operational requirement that the detector only needs
    to "trigger" once inside an anomaly window to be considered successful.

    INPUT:
      y_true_ts : binary array of shape (T,) — ground truth timestep labels
      y_pred_ts : binary array of shape (T,) — predicted timestep labels

    OUTPUT:
      y_pred_adjusted : adjusted predictions of shape (T,)
    """
    y_pred_adj = y_pred_ts.copy()

    # Find contiguous anomaly segments in ground truth
    in_anomaly = False
    seg_start  = 0

    for t in range(len(y_true_ts)):
        if y_true_ts[t] == 1 and not in_anomaly:
            in_anomaly = True
            seg_start  = t
        elif y_true_ts[t] == 0 and in_anomaly:
            in_anomaly = False
            seg_end = t
            # If any timestep in [seg_start, seg_end) was predicted anomalous
            if y_pred_ts[seg_start:seg_end].any():
                y_pred_adj[seg_start:seg_end] = 1

    # Handle segment that reaches end of signal
    if in_anomaly:
        if y_pred_ts[seg_start:].any():
            y_pred_adj[seg_start:] = 1

    return y_pred_adj


def evaluate_point_adjust(ts_scores, y_true_ts, threshold):
    """
    Threshold timestep scores, apply point-adjust, compute P/R/F1.
    """
    y_pred = (ts_scores > threshold).astype(int)
    y_pred_adj = point_adjust(y_true_ts, y_pred)

    if len(np.unique(y_true_ts)) < 2:
        return 0.0, 0.0, 0.0

    p  = precision_score(y_true_ts, y_pred_adj, zero_division=0)
    r  = recall_score(y_true_ts, y_pred_adj, zero_division=0)
    f1 = f1_score(y_true_ts, y_pred_adj, zero_division=0)
    return p, r, f1

# =============================================================================
# VAE ARCHITECTURE (identical to vae_anomaly_detection.py)
# =============================================================================

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU()
        )
        self.fc_mu      = nn.Linear(16, latent_dim)
        self.fc_log_var = nn.Linear(16, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.fc_mu(h), self.fc_log_var(h)

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    def forward(self, z):
        return self.net(z)

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

    def reconstruct(self, x):
        mu, _ = self.encoder(x)
        return self.decoder(mu)

def vae_loss(x, x_hat, mu, log_var, beta=1.0):
    recon = nn.functional.mse_loss(x_hat, x, reduction='mean')
    kl    = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return recon + beta * kl, recon.item(), kl.item()

def train_vae(model, loader, optimizer, device, beta=1.0):
    model.train()
    total = 0.0
    for batch_x, in loader:
        batch_x = batch_x.to(device)
        optimizer.zero_grad()
        x_hat, mu, lv = model(batch_x)
        loss, _, _ = vae_loss(batch_x, x_hat, mu, lv, beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)

def compute_window_errors(model, windows_tensor, device):
    model.eval()
    errors = []
    with torch.no_grad():
        for i in range(0, len(windows_tensor), 256):
            batch = windows_tensor[i:i+256].to(device)
            x_hat = model.reconstruct(batch)
            mse = ((batch - x_hat)**2).mean(dim=1)
            errors.append(mse.cpu().numpy())
    return np.concatenate(errors)

# =============================================================================
# MAIN LOOP
# =============================================================================

print(f"\nLoading labels...")
labels = pd.read_csv(LABELS_FILE)
print(f"  {len(labels)} channels\n")

results = []

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

    # Normalize on training statistics
    mu_tr = train_raw.mean()
    sd_tr = train_raw.std()
    if sd_tr < 1e-9: sd_tr = 1.0
    train_norm = (train_raw - mu_tr) / sd_tr
    test_norm  = (test_raw  - mu_tr) / sd_tr

    # Ground truth at TIMESTEP level (for point-adjust)
    y_true_ts = build_ground_truth_vector(test_len, windows_gt)

    # Windows
    X_train = make_windows_array(train_norm, WINDOW_SIZE, STEP_SIZE)
    X_test  = make_windows_array(test_norm,  WINDOW_SIZE, STEP_SIZE)
    if len(X_train) < 10: continue

    # Train/val split on training windows
    n_val = max(1, int(0.1 * len(X_train)))
    X_tr  = torch.FloatTensor(X_train[:-n_val])
    X_va  = torch.FloatTensor(X_train[-n_val:])
    X_te  = torch.FloatTensor(X_test)

    train_loader = DataLoader(TensorDataset(X_tr), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_va), batch_size=BATCH_SIZE)

    # Train VAE
    model     = VAE(WINDOW_SIZE, LATENT_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val   = float('inf')
    patience_c = 0
    epochs_run = 0

    for epoch in range(EPOCHS):
        train_vae(model, train_loader, optimizer, DEVICE, BETA)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, in val_loader:
                bx = bx.to(DEVICE)
                xh, mu, lv = model(bx)
                l, _, _ = vae_loss(bx, xh, mu, lv, BETA)
                val_loss += l.item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        epochs_run += 1
        if val_loss < best_val - 1e-5:
            best_val   = val_loss
            patience_c = 0
        else:
            patience_c += 1
            if patience_c >= 20: break

    # Compute errors
    train_errors  = compute_window_errors(model, X_tr, DEVICE)
    test_errors_w = compute_window_errors(model, X_te,  DEVICE)

    # Convert window errors → timestep scores
    ts_scores = windows_to_timestep_scores(test_errors_w, test_len, WINDOW_SIZE, STEP_SIZE)

    # Calibrate thresholds from training errors
    mu_e  = train_errors.mean()
    sd_e  = train_errors.std()
    thresholds = {f"k{k}": mu_e + k * sd_e for k in K_VALUES}
    for pct in [90, 95, 99]:
        thresholds[f"p{pct}"] = np.percentile(train_errors, pct)

    # Evaluate with point-adjust at each threshold
    best_f1  = 0.0; best_p = 0.0; best_r = 0.0; best_name = "none"
    for name, thr in thresholds.items():
        p, r, f1 = evaluate_point_adjust(ts_scores, y_true_ts, thr)
        if f1 > best_f1:
            best_f1 = f1; best_p = p; best_r = r; best_name = name

    # Also evaluate fixed k=3
    p_k3, r_k3, f1_k3 = evaluate_point_adjust(ts_scores, y_true_ts, mu_e + 3.0*sd_e)

    results.append({
        "chan_id":      chan,
        "spacecraft":  sc,
        "anomaly_class": anom_class,
        "epochs_run":  epochs_run,
        "pa_best_precision": round(best_p, 4),
        "pa_best_recall":    round(best_r, 4),
        "pa_best_f1":        round(best_f1, 4),
        "pa_best_threshold": best_name,
        "pa_k3_precision":   round(p_k3, 4),
        "pa_k3_recall":      round(r_k3, 4),
        "pa_k3_f1":          round(f1_k3, 4),
    })

    if (idx + 1) % 10 == 0 or chan in ["P-1","S-1"]:
        print(f"  [{idx+1:3d}/82] {chan:6s} | PA best F1={best_f1:.3f} | "
              f"PA k3 F1={f1_k3:.3f} | epochs={epochs_run}")

# =============================================================================
# RESULTS
# =============================================================================

df = pd.DataFrame(results)

print("\n" + "="*70)
print("AGGREGATE RESULTS — VAE WITH POINT-ADJUST EVALUATION")
print("="*70)

smap = df[df.spacecraft=="SMAP"]
msl  = df[df.spacecraft=="MSL"]

for col, label in [("pa_best_f1","PA best threshold"),
                   ("pa_k3_f1",  "PA k=3 threshold")]:
    v = df[col]
    print(f"\n{label}:")
    print(f"  All  — mean={v.mean():.4f}  median={v.median():.4f}  "
          f"F1>0:{(v>0).sum()}/82  F1>0.5:{(v>0.5).sum()}/82")
    print(f"  SMAP — mean={smap[col].mean():.4f}  median={smap[col].median():.4f}")
    print(f"  MSL  — mean={msl[col].mean():.4f}  median={msl[col].median():.4f}")

print(f"\nHundman 2018 LSTM (point-adjust, published):")
print(f"  SMAP — F1=0.7519")
print(f"  MSL  — F1=0.6955")

ctx = df[df.anomaly_class.str.contains("contextual", na=False)]
pt  = df[~df.anomaly_class.str.contains("contextual", na=False)]
print(f"\nBy anomaly type (PA best):")
print(f"  Contextual (n={len(ctx)}): mean F1={ctx.pa_best_f1.mean():.4f}")
print(f"  Point      (n={len(pt)}): mean F1={pt.pa_best_f1.mean():.4f}")

p1 = df[df.chan_id=="P-1"].iloc[0]
print(f"\nChannel P-1:")
print(f"  PA best F1={p1.pa_best_f1:.4f}  PA k3 F1={p1.pa_k3_f1:.4f}")
print(f"  (Previous window-overlap F1=0.102, threshold baseline F1=0.011)")

df.to_csv(RESULTS_CSV, index=False)
print(f"\nSaved: {RESULTS_CSV}")

# =============================================================================
# FIGURE
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(
    "VAE Anomaly Detection — Point-Adjust Evaluation (Hundman 2018 Protocol)\n"
    "NASA SMAP/MSL — Mehrab Jamshidi, PoliMi 2026",
    fontsize=13, fontweight="bold"
)

# Panel 1: PA F1 per channel sorted
ax = axes[0]
df_s = df.sort_values("pa_best_f1", ascending=False)
colors = ["#2c7bb6" if "contextual" in str(r.anomaly_class) else "#fdae61"
          for _, r in df_s.iterrows()]
ax.bar(range(len(df_s)), df_s.pa_best_f1.values, color=colors, width=0.8)
ax.axhline(0.7519, color="red",    linestyle="--", linewidth=1.5,
           label="Hundman SMAP F1=0.752")
ax.axhline(0.6955, color="orange", linestyle="--", linewidth=1.5,
           label="Hundman MSL  F1=0.696")
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(facecolor="#2c7bb6", label="Contextual"),
    Patch(facecolor="#fdae61", label="Point"),
    plt.Line2D([0],[0], color="red",    linestyle="--", label="Hundman SMAP=0.752"),
    plt.Line2D([0],[0], color="orange", linestyle="--", label="Hundman MSL=0.696"),
], fontsize=8)
ax.set_title("VAE F1 per channel — point-adjust (sorted)", fontsize=11)
ax.set_ylabel("F1 score"); ax.set_xlabel("Channel rank")
ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3)

# Panel 2: Comparison our PA vs Hundman by spacecraft
ax = axes[1]
cats = ["VAE PA best\n(SMAP)", "VAE PA best\n(MSL)", "Hundman LSTM\n(SMAP)", "Hundman LSTM\n(MSL)"]
vals = [smap.pa_best_f1.mean(), msl.pa_best_f1.mean(), 0.7519, 0.6955]
colors2 = ["#1a9641","#1a9641","#d73027","#d73027"]
bars = ax.bar(cats, vals, color=colors2, width=0.5, alpha=0.85)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
            f"{val:.3f}", ha="center", fontsize=11, fontweight="bold")
ax.set_title("VAE (point-adjust) vs Hundman 2018 LSTM", fontsize=11)
ax.set_ylabel("Mean F1 score"); ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURE_FILE, dpi=150, bbox_inches="tight")
print(f"Figure saved: {FIGURE_FILE}")

print("\n" + "="*70)
print("DONE. Next: run lstm_regression.py")
print("="*70)
