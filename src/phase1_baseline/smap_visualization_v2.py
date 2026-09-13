"""
NASA SMAP/MSL Telemetry Anomaly Visualization and Baseline Evaluation
Master Thesis Prototype — Background Simulation
Author: Mehrab Jamshidi
This script does two things:
  1. Loads SMAP channel P-1 and produces a visualization figure showing
     the telemetry signal with labeled anomaly windows highlighted in red.
  2. Computes a simple threshold-based anomaly detection baseline (z-score)
     on all 82 channels and reports precision, recall, and F1 score.
     This establishes the classical detection floor that the VAE must beat.

Figure output: smap_P-1_anomaly_visualization.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import ast
import os
import sys
import re
from sklearn.metrics import precision_score, recall_score, f1_score

# ─────────────────────────────────────────────────────────────────────────────
# PATHS — adjust if your folder structure is different
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (  # noqa: E402
    DATASET_ROOT, LABELS_FILE, TEST_FOLDER, TRAIN_FOLDER,
    RESULTS_DIR, FIGURES_DIR, require_dataset,
)

require_dataset()

# Channel to visualize
CHANNEL_ID = "P-1"

# ─────────────────────────────────────────────────────────────────────────────
# HELPER — parse anomaly sequences safely (handles 'point'/'contextual' labels)
# ─────────────────────────────────────────────────────────────────────────────

def parse_windows(seq_str):
    """Extract [[start,end],...] pairs from the anomaly_sequences column."""
    numbers = re.findall(r'\d+', seq_str)
    numbers = [int(n) for n in numbers]
    return [[numbers[i], numbers[i+1]] for i in range(0, len(numbers), 2)]

# ─────────────────────────────────────────────────────────────────────────────
# LOAD LABELS FILE
# ─────────────────────────────────────────────────────────────────────────────

print("Loading labels file...")
labels = pd.read_csv(LABELS_FILE)
print(f"  {len(labels)} channels loaded.")

# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — VISUALIZATION OF SMAP CHANNEL P-1
# ─────────────────────────────────────────────────────────────────────────────

print(f"\nGenerating visualization for channel {CHANNEL_ID}...")

channel_row  = labels[labels["chan_id"] == CHANNEL_ID].iloc[0]
spacecraft   = channel_row["spacecraft"]
anomaly_seqs = parse_windows(channel_row["anomaly_sequences"])
anomaly_class = channel_row["class"]
total_values  = channel_row["num_values"]

print(f"  Spacecraft:      {spacecraft}")
print(f"  Total timesteps: {total_values}")
print(f"  Anomaly type:    {anomaly_class}")
print(f"  Anomaly windows: {anomaly_seqs}")

# Load telemetry
test_data  = np.load(os.path.join(TEST_FOLDER,  f"{CHANNEL_ID}.npy"))
train_data = np.load(os.path.join(TRAIN_FOLDER, f"{CHANNEL_ID}.npy"))

print(f"  Train shape: {train_data.shape}")
print(f"  Test shape:  {test_data.shape}")

train_signal = train_data[:, 0]
test_signal  = test_data[:,  0]
train_len    = len(train_signal)
test_len     = len(test_signal)

full_signal  = np.concatenate([train_signal, test_signal])
time_axis    = np.arange(len(full_signal))

# Shift anomaly window indices from test-local to full-signal positions
anomaly_windows_global = [
    [start + train_len, end + train_len]
    for start, end in anomaly_seqs
]

# Build figure
fig = plt.figure(figsize=(16, 8))
fig.suptitle(
    f"NASA {spacecraft} — Channel {CHANNEL_ID}\n"
    f"Real Spacecraft Telemetry with Labeled Anomaly Windows",
    fontsize=14, fontweight="bold"
)

gs = gridspec.GridSpec(2, 1, hspace=0.4)

# Top panel — full channel
ax1 = fig.add_subplot(gs[0])
ax1.plot(time_axis, full_signal, color="#2c7bb6", linewidth=0.6, label="Telemetry signal")
ax1.axvline(x=train_len, color="gray", linestyle="--", linewidth=1.2, label="Train / Test split")
for i, (start, end) in enumerate(anomaly_windows_global):
    ax1.axvspan(start, end, alpha=0.35, color="#d73027",
                label="Anomaly window" if i == 0 else "")
ax1.set_ylabel("Sensor value", fontsize=10)
ax1.set_xlabel("Time step", fontsize=10)
ax1.set_title("Full channel — training (nominal) and test (with anomalies)", fontsize=10)
ax1.legend(loc="upper left", fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.annotate("← Train (nominal)    Test (with anomalies) →",
             xy=(train_len, 0), xycoords=("data", "axes fraction"),
             xytext=(10, -20), textcoords="offset points",
             fontsize=8, color="gray")

# Bottom panel — test split only
ax2 = fig.add_subplot(gs[1])
test_time = np.arange(train_len, train_len + test_len)
ax2.plot(test_time, test_signal, color="#2c7bb6", linewidth=0.8, label="Test signal")
for i, (start, end) in enumerate(anomaly_windows_global):
    ax2.axvspan(start, end, alpha=0.4, color="#d73027",
                label="Anomaly window" if i == 0 else "")
for i, (start, end) in enumerate(anomaly_windows_global):
    midpoint = (start + end) / 2
    ax2.annotate(f"Anomaly {i+1}",
                 xy=(midpoint, 1), xycoords=("data", "axes fraction"),
                 xytext=(0, -12), textcoords="offset points",
                 ha="center", fontsize=8, color="#d73027", fontweight="bold")
ax2.set_ylabel("Sensor value", fontsize=10)
ax2.set_xlabel("Time step", fontsize=10)
ax2.set_title("Test split only — anomaly windows highlighted in red", fontsize=10)
ax2.legend(loc="upper left", fontsize=8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig_file = f"smap_{CHANNEL_ID}_anomaly_visualization.png"
plt.savefig(fig_file, dpi=150, bbox_inches="tight")
print(f"\n  Figure saved as: {fig_file}")
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — THRESHOLD BASELINE EVALUATION ON ALL 82 CHANNELS
#
# Method: z-score threshold on the primary sensor value (column 0).
#   Train the threshold on the training split (nominal data only).
#   Flag a test timestep as anomalous if it exceeds train_mean ± 3*train_std.
#   Compare predictions against ground truth labels from labeled_anomalies.csv.
#
# This is the classical statistical detection floor. It is simpler than the
# LSTM method of Hundman et al. (2018) but establishes the lower bound that
# any meaningful ML method must exceed.
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*65)
print("PART 2 — THRESHOLD BASELINE EVALUATION (all 82 channels)")
print("="*65)
print("Method: z-score threshold trained on nominal split")
print("        flag if |value - train_mean| > 3 × train_std")
print("="*65)

results = []

for _, row in labels.iterrows():
    chan   = row["chan_id"]
    sc     = row["spacecraft"]
    nval   = row["num_values"]
    windows = parse_windows(row["anomaly_sequences"])

    # Load files — skip if missing
    test_file  = os.path.join(TEST_FOLDER,  f"{chan}.npy")
    train_file = os.path.join(TRAIN_FOLDER, f"{chan}.npy")
    if not os.path.exists(test_file) or not os.path.exists(train_file):
        print(f"  WARNING: files missing for {chan}, skipping.")
        continue

    tr = np.load(train_file)[:, 0]
    te = np.load(test_file)[:,  0]

    # Build ground truth vector (1 = anomalous, 0 = nominal)
    y_true = np.zeros(nval, dtype=int)
    for start, end in windows:
        y_true[start:end] = 1

    # Compute threshold from training split
    mu    = tr.mean()
    sigma = tr.std()
    if sigma == 0:
        sigma = 1e-9   # avoid division by zero on constant channels

    # Predict: anomalous if more than 3 standard deviations from training mean
    y_pred = (np.abs(te - mu) > 3 * sigma).astype(int)

    # Metrics
    p  = precision_score(y_true, y_pred, zero_division=0)
    r  = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    results.append({
        "chan_id":    chan,
        "spacecraft": sc,
        "precision":  p,
        "recall":     r,
        "f1":         f1,
        "anomaly_rate": y_true.mean() * 100
    })

df = pd.DataFrame(results)

# ── Per-channel results ───────────────────────────────────────────────────────
print("\nPer-channel results (sorted by F1):")
print(f"{'Chan':<8} {'SC':<5} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Anom%':>7}")
print("-" * 50)
for _, r in df.sort_values("f1", ascending=False).iterrows():
    print(f"{r.chan_id:<8} {r.spacecraft:<5} "
          f"{r.precision:>10.3f} {r.recall:>8.3f} "
          f"{r.f1:>8.3f} {r.anomaly_rate:>6.2f}%")

# ── Aggregate summary ─────────────────────────────────────────────────────────
print("\n" + "="*65)
print("AGGREGATE SUMMARY — Threshold Baseline")
print("="*65)
print(f"Channels evaluated:       {len(df)}")
print(f"\nAll channels (n={len(df)}):")
print(f"  Mean Precision:         {df.precision.mean():.3f}")
print(f"  Mean Recall:            {df.recall.mean():.3f}")
print(f"  Mean F1:                {df.f1.mean():.3f}")
print(f"  Median F1:              {df.f1.median():.3f}")

smap_df = df[df.spacecraft == "SMAP"]
msl_df  = df[df.spacecraft == "MSL"]
print(f"\nSMAP only (n={len(smap_df)}):")
print(f"  Mean F1:                {smap_df.f1.mean():.3f}")
print(f"  Median F1:              {smap_df.f1.median():.3f}")
print(f"\nMSL only (n={len(msl_df)}):")
print(f"  Mean F1:                {msl_df.f1.mean():.3f}")
print(f"  Median F1:              {msl_df.f1.median():.3f}")

# Channel P-1 specifically (to match report Section 6)
p1 = df[df.chan_id == "P-1"].iloc[0]
print(f"\nChannel P-1 specifically (used in report visualization):")
print(f"  Precision:              {p1.precision:.3f}")
print(f"  Recall:                 {p1.recall:.3f}")
print(f"  F1:                     {p1.f1:.3f}")
print(f"  (anomaly rate:          {p1.anomaly_rate:.2f}%)")

# Channels with zero F1 (baseline completely fails)
zero_f1 = df[df.f1 == 0]
print(f"\nChannels where baseline F1 = 0 (completely fails): {len(zero_f1)}")
print(f"  These are the channels where VAE improvement will be most visible.")

print("\n" + "="*65)
print("INTERPRETATION")
print("="*65)
print("A low mean F1 on contextual anomaly channels confirms the core")
print("thesis motivation: classical threshold detection is insufficient")
print("for the distributional anomalies that dominate real spacecraft")
print("telemetry. The VAE, trained on nominal data and detecting")
print("reconstruction error, is the proposed solution.")
print("="*65)

# ── Save results to CSV for report ───────────────────────────────────────────
csv_file = RESULTS_DIR / "threshold_baseline_results.csv"
df.to_csv(csv_file, index=False)
print(f"\nFull results saved to: {csv_file}")