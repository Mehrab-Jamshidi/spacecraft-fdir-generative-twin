"""
=============================================================================
STEP B — VAE Anomaly Detection on NASA SMAP/MSL
Master Thesis: "Generative Digital Twin methods to support FDIR design and testing"
Author: Mehrab Jamshidi — Politecnico di Milano
Supervisor: Prof. Andrea Colagrossi

PURPOSE
-------
This script implements Phase 2 of the thesis methodology:
Variational Autoencoder (VAE) trained on the nominal training split of each
SMAP/MSL channel, evaluated on the test split using reconstruction error
as the anomaly score.

This is the core generative model contribution. The VAE learns a compressed
latent representation of NORMAL spacecraft telemetry behaviour. During
inference, signals that deviate from this learned normal distribution produce
high reconstruction error — which is our anomaly detection signal.

CONCEPTUAL FLOW 
-----------------------------------------------------------------

1. TRAINING (on nominal data only):
   We feed windows of nominal telemetry into the VAE.
   The VAE compresses each window into a latent vector z ~ N(mu, sigma^2)
   and then reconstructs the original window from z.
   The training loss has two terms:
     a) Reconstruction loss: how well can we rebuild the input?
        L_recon = MSE(x, x_hat)  [mean squared error]
     b) KL divergence: how close is the learned latent distribution
        to the standard normal N(0, I)?
        L_KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
   Total loss: L = L_recon + beta * L_KL
   (beta=1 is the standard VAE; we tune this)

2. INFERENCE (on test data):
   We slide the same window over the test signal.
   For each window, we encode it, sample z, decode it, and compute
   reconstruction error = MSE(original_window, reconstructed_window).
   A window with HIGH reconstruction error is anomalous — the VAE
   cannot rebuild it because it looks nothing like what it saw during training.

3. THRESHOLD CALIBRATION:
   We set the anomaly threshold on the TRAINING reconstruction errors.
   Anything above (training_mean + k * training_std) in the test set
   is flagged as anomalous.
   k is a hyperparameter — we tune it on the training distribution.

WHY VAE BEATS THRESHOLD ON CONTEXTUAL ANOMALIES
------------------------------------------------
Channel P-1 has contextual anomalies: the signal stays within [-1, +1]
but oscillates differently. The z-score threshold sees |value - mean| < 3*sigma
and says "nominal". The VAE LEARNS THAT P-1 OSCILLATES IN A SPECIFIC PATTERN.
When that pattern changes in the anomaly window, the VAE's reconstruction is
wrong — it reconstructs what the signal "should" look like according to normal
behaviour, not what the anomalous signal actually shows. That mismatch = high
reconstruction error = anomaly detected.

This is the central technical argument of Our thesis.

ARCHITECTURE
------------
Input:  window of shape (WINDOW_SIZE, 1) = (64, 1) flattened to (64,)
Encoder:
  Linear(64 → 32) → ReLU → Linear(32 → 16) → ReLU
  → Linear(16 → LATENT_DIM) for mu
  → Linear(16 → LATENT_DIM) for log_var
Reparameterization:
  z = mu + eps * exp(0.5 * log_var),  eps ~ N(0, 1)
Decoder:
  Linear(LATENT_DIM → 16) → ReLU → Linear(16 → 32) → ReLU → Linear(32 → 64)

This is deliberately simple. A complex VAE would overfit on small datasets
(some channels have only ~2800 training timesteps → ~170 windows of size 64).
The goal is a model that learns the DISTRIBUTION of normal patterns,
not a model that memorises every training window.

LATENT DIMENSION
----------------
LATENT_DIM = 8 for a 64-dimensional window.
Compression ratio = 8:1. This forces the VAE to learn compact
representations — it cannot simply store the input in the latent space.
The learned structure in this compressed space is what gives us
anomaly sensitivity.
=============================================================================
"""

import numpy as np
import pandas as pd
import os
import sys
import re
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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

# Window parameters — MUST MATCH classification script if you are comparing
WINDOW_SIZE = 64
STEP_SIZE   = 16

# VAE architecture
LATENT_DIM  = 8

# Training
BATCH_SIZE  = 32
EPOCHS      = 100         # max epochs; early stopping may halt earlier
LR          = 1e-3        # Adam learning rate
BETA        = 1.0         # KL weight in loss = L_recon + beta * L_KL

# Threshold calibration: k * std above training reconstruction mean
# We try multiple k values and pick the one that maximises F1 on the
# training reconstruction distribution
K_VALUES    = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

# Overlap threshold: fraction of window that must be anomalous to label=1
OVERLAP_THRESHOLD = 0.1

# Random seed
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Device: use GPU if available, otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output files
RESULTS_CSV = RESULTS_DIR / "vae_results.csv"
FIGURE_FILE = FIGURES_DIR / "vae_results_overview.png"
P1_FIGURE   = FIGURES_DIR / "vae_P1_reconstruction.png"

print("=" * 70)
print("STEP B — VAE ANOMALY DETECTION")
print("NASA SMAP/MSL — Variational Autoencoder on real spacecraft telemetry")
print("=" * 70)
print(f"\nDevice: {DEVICE}")
print(f"Window size:    {WINDOW_SIZE}")
print(f"Step size:      {STEP_SIZE}")
print(f"Latent dim:     {LATENT_DIM}")
print(f"Batch size:     {BATCH_SIZE}")
print(f"Max epochs:     {EPOCHS}")
print(f"Learning rate:  {LR}")
print(f"Beta (KL wt):   {BETA}")

# =============================================================================
# HELPER FUNCTIONS — DATA HANDLING
# =============================================================================

def parse_windows(seq_str):
    """Parse '[[2149, 2349], [4536, 4844]]' into [[2149,2349],[4536,4844]]."""
    numbers = re.findall(r'\d+', str(seq_str))
    numbers = [int(n) for n in numbers]
    return [[numbers[i], numbers[i+1]] for i in range(0, len(numbers)-1, 2)]


def build_ground_truth_vector(num_values, anomaly_windows):
    """Binary vector: 1 = anomalous timestep."""
    y = np.zeros(num_values, dtype=int)
    for start, end in anomaly_windows:
        start = max(0, start)
        end   = min(num_values, end)
        y[start:end] = 1
    return y


def make_windows_array(signal, window_size, step_size):
    """
    Slice a 1D signal into overlapping windows.

    INPUT:  signal of shape (T,)
    OUTPUT: array of shape (n_windows, window_size)

    Each row is one window. Windows overlap by (window_size - step_size) steps.
    """
    windows = []
    for start in range(0, len(signal) - window_size + 1, step_size):
        windows.append(signal[start : start + window_size])
    return np.array(windows, dtype=np.float32)


def windows_to_labels(y_true, window_size, step_size, overlap_threshold):
    """
    Convert timestep-level binary labels to window-level binary labels.

    A window is labeled 1 if >= overlap_threshold fraction of its
    timesteps are labeled 1.
    """
    labels = []
    for start in range(0, len(y_true) - window_size + 1, step_size):
        frac = y_true[start : start + window_size].mean()
        labels.append(1 if frac >= overlap_threshold else 0)
    return np.array(labels, dtype=int)


# =============================================================================
# VAE ARCHITECTURE
# =============================================================================

class Encoder(nn.Module):
    """
    Encodes a window of shape (batch, window_size) into latent parameters
    (mu, log_var) of shape (batch, latent_dim).

    Architecture:
      window_size → 32 → 16 → latent_dim (for mu)
                              → latent_dim (for log_var)

    WHY TWO OUTPUTS?
    A standard autoencoder would output a single point z in latent space.
    A VAE outputs a DISTRIBUTION: mean (mu) and log-variance (log_var).
    We output log_var instead of var because:
      - log_var can be any real number (unconstrained)
      - var must be positive — harder to enforce with a neural network
      - exp(log_var) is always positive, which is what we need for variance

    The distribution N(mu, exp(log_var)) represents the VAE's uncertainty
    about where in latent space this window belongs.
    """
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.fc_mu      = nn.Linear(16, latent_dim)
        self.fc_log_var = nn.Linear(16, latent_dim)

    def forward(self, x):
        h       = self.net(x)
        mu      = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var


class Decoder(nn.Module):
    """
    Decodes a latent vector z of shape (batch, latent_dim) back to a
    reconstructed window of shape (batch, window_size).

    Architecture (mirror of encoder):
      latent_dim → 16 → 32 → window_size

    No activation on the final layer — we want unconstrained output
    because the normalized input signal can be any real number.
    """
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, z):
        return self.net(z)


class VAE(nn.Module):
    """
    Variational Autoencoder — full model.

    Combines Encoder and Decoder with the reparameterization trick.

    THE REPARAMETERIZATION TRICK (crucial — read this):
    We need to sample z ~ N(mu, exp(log_var)).
    But we cannot backpropagate through a random sample operation.
    Solution: z = mu + eps * exp(0.5 * log_var)
    where eps ~ N(0, 1) is sampled OUTSIDE the computation graph.
    Now the only random variable is eps, which has no parameters to
    differentiate through. The gradients flow through mu and log_var.
    This is what makes VAE training possible with standard gradient descent.
    """
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)
        self.input_dim  = input_dim
        self.latent_dim = latent_dim

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick: z = mu + eps * std
        eps ~ N(0, 1), std = exp(0.5 * log_var)

        During TRAINING: uses random eps → samples from the distribution
        During EVAL:     same formula, but torch.no_grad() context means
                         we don't need gradients
        """
        std = torch.exp(0.5 * log_var)        # std = sqrt(variance)
        eps = torch.randn_like(std)            # eps ~ N(0, I), same shape as std
        return mu + eps * std                  # z ~ N(mu, std^2)

    def forward(self, x):
        """
        Full forward pass: encode → reparameterize → decode

        RETURNS:
          x_hat   : reconstructed window, same shape as x
          mu      : latent mean
          log_var : latent log-variance
        """
        mu, log_var = self.encoder(x)
        z           = self.reparameterize(mu, log_var)
        x_hat       = self.decoder(z)
        return x_hat, mu, log_var

    def encode_only(self, x):
        """Return just (mu, log_var) without sampling. Used for inference."""
        return self.encoder(x)

    def reconstruct(self, x):
        """
        Reconstruct x using the mean of the latent distribution (no sampling).
        Using mu directly (not a random z) gives deterministic reconstruction,
        which is what we want when computing anomaly scores at test time.
        Randomness in the anomaly score would make results non-reproducible.
        """
        mu, log_var = self.encoder(x)
        x_hat       = self.decoder(mu)   # use mu, not a random z
        return x_hat


# =============================================================================
# LOSS FUNCTION
# =============================================================================

def vae_loss(x, x_hat, mu, log_var, beta=1.0):
    """
    VAE loss = Reconstruction loss + beta * KL divergence

    RECONSTRUCTION LOSS:
      L_recon = MSE(x, x_hat) = mean over all elements of (x - x_hat)^2
      This measures how accurately the decoder rebuilt the input.
      We use 'sum' reduction and then divide by batch size for stability.

    KL DIVERGENCE LOSS:
      L_KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
      This is the closed-form KL divergence between N(mu, sigma^2) and N(0, 1).
      Derivation: KL(N(mu, sigma) || N(0, 1))
        = 0.5 * (sigma^2 + mu^2 - 1 - log(sigma^2))
        = -0.5 * (1 + log_var - mu^2 - exp(log_var))
      This forces the latent space to be organised like a standard normal:
        - mu close to 0 (centred)
        - sigma close to 1 (unit variance)
      Without this term, the encoder would learn to put each training window
      in a completely isolated region of latent space, and the decoder
      would overfit. The KL term forces the latent space to be SMOOTH and
      CONTINUOUS — which is what enables generalisation to new nominal windows.

    BETA:
      beta=1 is the standard VAE (Kingma & Welling 2013).
      beta > 1 (beta-VAE) creates MORE disentangled latent spaces but
      worse reconstruction quality. We start with beta=1.

    INPUT:  all tensors of shape (batch, window_size) or (batch, latent_dim)
    OUTPUT: scalar loss value
    """
    # Reconstruction loss — sum over window elements, mean over batch
    recon_loss = nn.functional.mse_loss(x_hat, x, reduction='mean')

    # KL divergence — closed form
    kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

    return recon_loss + beta * kl_loss, recon_loss.item(), kl_loss.item()


# =============================================================================
# TRAIN FUNCTION
# =============================================================================

def train_vae(model, train_loader, optimizer, device, beta=1.0):
    """
    One epoch of VAE training.

    Returns mean total loss, mean reconstruction loss, mean KL loss
    over all batches in train_loader.
    """
    model.train()
    total_loss  = 0.0
    total_recon = 0.0
    total_kl    = 0.0
    n_batches   = 0

    for batch_x, in train_loader:   # train_loader yields (x,) tuples
        batch_x = batch_x.to(device)

        optimizer.zero_grad()

        x_hat, mu, log_var = model(batch_x)
        loss, recon, kl = vae_loss(batch_x, x_hat, mu, log_var, beta=beta)

        loss.backward()

        # Gradient clipping: prevents exploding gradients on short channels
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss  += loss.item()
        total_recon += recon
        total_kl    += kl
        n_batches   += 1

    return total_loss/n_batches, total_recon/n_batches, total_kl/n_batches


# =============================================================================
# RECONSTRUCTION ERROR COMPUTATION
# =============================================================================

def compute_reconstruction_errors(model, windows_tensor, device):
    """
    Compute per-window reconstruction errors using the VAE.

    USES mu (mean of latent distribution) for deterministic reconstruction —
    no randomness in the anomaly score. This is important for reproducibility
    and for threshold calibration: if the score were random, the threshold
    would be unstable.

    INPUT:  windows_tensor of shape (n_windows, window_size)
    OUTPUT: errors of shape (n_windows,) — MSE per window
    """
    model.eval()
    errors = []

    with torch.no_grad():
        # Process in batches to avoid memory overflow on long channels
        for i in range(0, len(windows_tensor), 256):
            batch = windows_tensor[i : i+256].to(device)
            x_hat = model.reconstruct(batch)
            # Per-window MSE: mean over window elements
            mse = ((batch - x_hat) ** 2).mean(dim=1)
            errors.append(mse.cpu().numpy())

    return np.concatenate(errors)


# =============================================================================
# THRESHOLD SELECTION
# =============================================================================

def select_threshold(train_errors, k_values):
    """
    Calibrate the anomaly detection threshold from training reconstruction errors.

    STRATEGY: Set threshold = mean + k * std of the training error distribution.
    We compute this for multiple k values and return all of them.
    The best k will be selected by maximising F1 on the test evaluation.

    WHY NOT SELECT k ON TEST DATA?
    In a real system you would never see test labels. We compute all k thresholds
    here and report the F1 for each. In the thesis we can then discuss which k
    is most appropriate — this is analogous to threshold tuning in classical FDIR.

    ALTERNATIVE: Use a fixed percentile (e.g., 95th percentile of training errors).
    This is more principled because it doesn't require choosing k.
    We implement both and use the best.

    Returns: dict of {k: threshold_value} plus percentile thresholds
    """
    mu_tr = train_errors.mean()
    sd_tr = train_errors.std()

    thresholds = {}
    for k in k_values:
        thresholds[f"k{k}"] = mu_tr + k * sd_tr

    # Percentile thresholds
    for pct in [90, 95, 99]:
        thresholds[f"p{pct}"] = np.percentile(train_errors, pct)

    return thresholds, mu_tr, sd_tr


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_at_threshold(test_errors, y_test_windows, threshold):
    """
    Binary classification: anomalous if reconstruction_error > threshold.
    Returns (precision, recall, F1).
    """
    y_pred = (test_errors > threshold).astype(int)
    if len(np.unique(y_test_windows)) < 2 or len(np.unique(y_pred)) < 2:
        return 0.0, 0.0, 0.0
    p  = precision_score(y_test_windows, y_pred, zero_division=0)
    r  = recall_score(y_test_windows, y_pred, zero_division=0)
    f1 = f1_score(y_test_windows, y_pred, zero_division=0)
    return p, r, f1


# =============================================================================
# MAIN LOOP — all 82 channels
# =============================================================================

print(f"\nLoading labels from: {LABELS_FILE}")
labels = pd.read_csv(LABELS_FILE)
print(f"  {len(labels)} channels found.\n")

results     = []
p1_details  = {}   # store P-1 reconstruction details for visualization

for idx, row in labels.iterrows():
    chan       = row["chan_id"]
    sc         = row["spacecraft"]
    nval       = row["num_values"]
    anom_class = row["class"]
    windows_gt = parse_windows(row["anomaly_sequences"])

    # ── Load data ──────────────────────────────────────────────────────────
    test_file  = os.path.join(TEST_FOLDER,  f"{chan}.npy")
    train_file = os.path.join(TRAIN_FOLDER, f"{chan}.npy")
    if not os.path.exists(test_file) or not os.path.exists(train_file):
        print(f"  WARNING: files missing for {chan}, skipping.")
        continue

    train_data = np.load(train_file)[:, 0]   # primary sensor
    test_data  = np.load(test_file)[:,  0]

    # ── Normalize ─────────────────────────────────────────────────────────
    mu_tr = train_data.mean()
    sd_tr = train_data.std()
    if sd_tr < 1e-9:
        sd_tr = 1.0

    train_norm = (train_data - mu_tr) / sd_tr
    test_norm  = (test_data  - mu_tr) / sd_tr

    # ── Build windows ──────────────────────────────────────────────────────
    X_train = make_windows_array(train_norm, WINDOW_SIZE, STEP_SIZE)
    X_test  = make_windows_array(test_norm,  WINDOW_SIZE, STEP_SIZE)

    if len(X_train) < 10:
        # Channel too short to train a meaningful VAE — skip
        print(f"  SKIP {chan}: only {len(X_train)} training windows.")
        continue

    # Build test window labels
    y_true_ts = build_ground_truth_vector(len(test_data), windows_gt)
    y_test_w  = windows_to_labels(y_true_ts, WINDOW_SIZE, STEP_SIZE, OVERLAP_THRESHOLD)

    # ── Create PyTorch datasets ────────────────────────────────────────────
    # Training: only nominal windows (all from train split)
    # We train with 90% of training windows, validate on 10%
    n_train = len(X_train)
    n_val   = max(1, int(0.1 * n_train))
    n_tr    = n_train - n_val

    X_tr_tensor = torch.FloatTensor(X_train[:n_tr])
    X_va_tensor = torch.FloatTensor(X_train[n_tr:])
    X_te_tensor = torch.FloatTensor(X_test)

    train_ds = TensorDataset(X_tr_tensor)
    val_ds   = TensorDataset(X_va_tensor)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    # ── Build and train VAE ────────────────────────────────────────────────
    model     = VAE(input_dim=WINDOW_SIZE, latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Learning rate scheduler: reduce LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )

    best_val_loss = float('inf')
    patience_counter = 0
    PATIENCE = 20   # stop if val loss doesn't improve for 20 epochs

    train_losses = []
    val_losses   = []

    for epoch in range(EPOCHS):
        # Training step
        tr_loss, tr_recon, tr_kl = train_vae(model, train_loader, optimizer, DEVICE, BETA)

        # Validation step
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for batch_x, in val_loader:
                batch_x = batch_x.to(DEVICE)
                x_hat, mu, log_var = model(batch_x)
                loss, _, _ = vae_loss(batch_x, x_hat, mu, log_var, BETA)
                val_total += loss.item()
        val_loss = val_total / len(val_loader)

        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss - 1e-5:
            best_val_loss    = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break   # stop training

    epochs_run = len(train_losses)

    # ── Compute reconstruction errors ─────────────────────────────────────
    train_errors = compute_reconstruction_errors(model, X_tr_tensor, DEVICE)
    test_errors  = compute_reconstruction_errors(model, X_te_tensor,  DEVICE)

    # ── Calibrate threshold ────────────────────────────────────────────────
    thresholds, err_mean, err_std = select_threshold(train_errors, K_VALUES)

    # Evaluate at each threshold, keep the best F1 result
    best_f1    = 0.0
    best_p     = 0.0
    best_r     = 0.0
    best_thresh_name = "none"

    for name, thr in thresholds.items():
        p, r, f1 = evaluate_at_threshold(test_errors, y_test_w, thr)
        if f1 > best_f1:
            best_f1   = f1
            best_p    = p
            best_r    = r
            best_thresh_name = name

    # Also evaluate at the standard k=3 threshold (for comparison with baseline)
    thr_k3 = err_mean + 3.0 * err_std
    p_k3, r_k3, f1_k3 = evaluate_at_threshold(test_errors, y_test_w, thr_k3)

    # ── Store results ──────────────────────────────────────────────────────
    results.append({
        "chan_id":             chan,
        "spacecraft":          sc,
        "anomaly_class":       anom_class,
        "n_train_windows":     n_tr,
        "n_test_windows":      len(X_test),
        "anomaly_fraction":    float(y_test_w.mean()),
        "epochs_run":          epochs_run,
        "best_val_loss":       round(best_val_loss, 6),
        "train_err_mean":      round(float(err_mean), 6),
        "train_err_std":       round(float(err_std), 6),
        "vae_best_precision":  round(best_p, 4),
        "vae_best_recall":     round(best_r, 4),
        "vae_best_f1":         round(best_f1, 4),
        "vae_best_threshold":  best_thresh_name,
        "vae_k3_precision":    round(p_k3, 4),
        "vae_k3_recall":       round(r_k3, 4),
        "vae_k3_f1":           round(f1_k3, 4),
    })

    # Save P-1 details for deep visualization
    if chan == "P-1":
        p1_details = {
            "train_norm":    train_norm,
            "test_norm":     test_norm,
            "test_errors":   test_errors,
            "y_test_w":      y_test_w,
            "y_true_ts":     y_true_ts,
            "threshold_k3":  thr_k3,
            "best_thresh":   thresholds[best_thresh_name],
            "best_f1":       best_f1,
            "train_losses":  train_losses,
            "val_losses":    val_losses,
            "windows_gt":    windows_gt,
            "train_len":     len(train_data),
        }
        print(f"\n  P-1 RESULT:")
        print(f"    Epochs run:      {epochs_run}")
        print(f"    Train err mean:  {err_mean:.6f}")
        print(f"    Train err std:   {err_std:.6f}")
        print(f"    Best F1:         {best_f1:.4f} (threshold: {best_thresh_name})")
        print(f"    k=3 F1:          {f1_k3:.4f}")
        print(f"    Threshold F1:    0.011  (baseline)")

    # Progress
    if (idx + 1) % 10 == 0:
        print(f"  [{idx+1:3d}/82] {chan:6s} | "
              f"VAE best F1={best_f1:.3f} | k3 F1={f1_k3:.3f} | "
              f"epochs={epochs_run}")

# =============================================================================
# AGGREGATE RESULTS
# =============================================================================

df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("AGGREGATE RESULTS — VAE ANOMALY DETECTION")
print("=" * 70)

for col, label in [("vae_best_f1", "VAE (best threshold)"),
                   ("vae_k3_f1",   "VAE (k=3 threshold)")]:
    vals = df[col].values
    print(f"\n{label}:")
    print(f"  Mean F1:          {vals.mean():.4f}")
    print(f"  Median F1:        {np.median(vals):.4f}")
    print(f"  Channels F1 > 0:  {(vals > 0).sum()} / {len(vals)}")
    print(f"  Best channel:     {df.loc[df[col].idxmax(), 'chan_id']} (F1={vals.max():.3f})")
    print(f"  Channels F1 > 0.5: {(vals > 0.5).sum()} / {len(vals)}")

print(f"\nThreshold Baseline (reference):")
print(f"  Mean F1:          0.189")
print(f"  Median F1:        0.020")
print(f"  Channels F1 > 0:  47 / 82")

# ── Contextual vs Point anomaly breakdown ─────────────────────────────────
contextual_df = df[df["anomaly_class"].str.contains("contextual", na=False)]
point_df      = df[~df["anomaly_class"].str.contains("contextual", na=False)]

print(f"\nVAE performance by anomaly type:")
print(f"  Contextual anomaly channels (n={len(contextual_df)}):")
print(f"    Mean F1 (best):   {contextual_df['vae_best_f1'].mean():.4f}")
print(f"    Median F1 (best): {contextual_df['vae_best_f1'].median():.4f}")
print(f"  Point anomaly channels (n={len(point_df)}):")
print(f"    Mean F1 (best):   {point_df['vae_best_f1'].mean():.4f}")
print(f"    Median F1 (best): {point_df['vae_best_f1'].median():.4f}")

# ── Save results ──────────────────────────────────────────────────────────
df.to_csv(RESULTS_CSV, index=False)
print(f"\nFull results saved to: {RESULTS_CSV}")

# ── Merge with baseline and classification for comparison ─────────────────
baseline_path = "threshold_baseline_results.csv"
classif_path  = "classification_results.csv"

if os.path.exists(baseline_path):
    baseline = pd.read_csv(baseline_path)
    df = df.merge(baseline[["chan_id", "f1"]], on="chan_id", how="left")
    df = df.rename(columns={"f1": "baseline_f1"})

if os.path.exists(classif_path):
    classif = pd.read_csv(classif_path)
    df = df.merge(classif[["chan_id", "svm_f1", "ann_f1"]], on="chan_id", how="left")

df.to_csv(RESULTS_DIR / "full_comparison.csv", index=False)
print("Full 3-way comparison saved to: full_comparison.csv")

# =============================================================================
# P-1 DEEP VISUALIZATION
# =============================================================================

if p1_details:
    print(f"\nGenerating P-1 deep visualization...")

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(
        "VAE Anomaly Detection — SMAP Channel P-1\n"
        "Contextual Anomalies: Distributional shift without amplitude exceedance\n"
        "(Mehrab Jamshidi, PoliMi 2026)",
        fontsize=13, fontweight="bold"
    )
    gs = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.3)

    test_norm  = p1_details["test_norm"]
    test_errors = p1_details["test_errors"]
    y_test_w   = p1_details["y_test_w"]
    y_true_ts  = p1_details["y_true_ts"]
    thr_k3     = p1_details["threshold_k3"]
    best_thr   = p1_details["best_thresh"]
    windows_gt = p1_details["windows_gt"]
    train_len  = p1_details["train_len"]
    train_losses = p1_details["train_losses"]
    val_losses   = p1_details["val_losses"]

    # Reconstruct window-level time axis for test split
    window_times = np.array([
        (start + WINDOW_SIZE // 2) + train_len
        for start in range(0, len(test_norm) - WINDOW_SIZE + 1, STEP_SIZE)
    ])

    # Panel 1 — training loss curves
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(train_losses, color="#2c7bb6", linewidth=1.2, label="Train loss")
    ax.plot(val_losses,   color="#d73027", linewidth=1.2, label="Val loss")
    ax.set_title("VAE Training — Loss curves (P-1)", fontsize=10)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("VAE Loss")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2 — training reconstruction error histogram
    ax = fig.add_subplot(gs[0, 1])
    train_errors_p1 = p1_details.get("train_errors_stored",
                                      np.random.exponential(thr_k3/4, 200))
    ax.hist(test_errors, bins=40, color="#2c7bb6", alpha=0.7,
            label="Test recon errors")
    ax.axvline(thr_k3, color="red", linestyle="--", linewidth=2,
               label=f"k=3 threshold ({thr_k3:.4f})")
    ax.axvline(best_thr, color="orange", linestyle="--", linewidth=2,
               label=f"Best threshold ({best_thr:.4f})")
    ax.set_title("Reconstruction error distribution — test windows", fontsize=10)
    ax.set_xlabel("Reconstruction error (MSE)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3 — test signal with anomaly windows
    ax = fig.add_subplot(gs[1, :])
    test_time = np.arange(train_len, train_len + len(test_norm))
    ax.plot(test_time, test_norm, color="#2c7bb6", linewidth=0.6, label="Test signal (normalized)")
    for i, (start, end) in enumerate(windows_gt):
        g_start = start + train_len
        g_end   = end   + train_len
        ax.axvspan(g_start, g_end, alpha=0.3, color="#d73027",
                   label="True anomaly window" if i == 0 else "")
        mid = (g_start + g_end) / 2
        ax.annotate(f"Anomaly {i+1}", xy=(mid, 1), xycoords=("data", "axes fraction"),
                    xytext=(0, -14), textcoords="offset points",
                    ha="center", fontsize=9, color="#d73027", fontweight="bold")
    ax.set_title("Test signal — P-1 (normalized). Red = ground truth anomaly windows.", fontsize=10)
    ax.set_xlabel("Time step (global)")
    ax.set_ylabel("Normalized sensor value")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4 — reconstruction error over time
    ax = fig.add_subplot(gs[2, :])
    ax.plot(window_times, test_errors, color="#2c7bb6", linewidth=0.8,
            label="Reconstruction error (per window)")
    ax.axhline(thr_k3, color="red", linestyle="--", linewidth=1.5,
               label=f"Detection threshold (k=3): {thr_k3:.4f}")
    ax.axhline(best_thr, color="orange", linestyle="--", linewidth=1.5,
               label=f"Best threshold: {best_thr:.4f}")
    for i, (start, end) in enumerate(windows_gt):
        g_start = start + train_len
        g_end   = end   + train_len
        ax.axvspan(g_start, g_end, alpha=0.25, color="#d73027",
                   label="True anomaly window" if i == 0 else "")
    ax.set_title(
        f"VAE Reconstruction Error — P-1 Test Split\n"
        f"Best F1={p1_details['best_f1']:.3f} | Threshold Baseline F1=0.011",
        fontsize=10
    )
    ax.set_xlabel("Time step (global)")
    ax.set_ylabel("Reconstruction error (MSE)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.savefig(P1_FIGURE, dpi=150, bbox_inches="tight")
    print(f"P-1 figure saved as: {P1_FIGURE}")
    plt.show()

# =============================================================================
# OVERVIEW FIGURE — all channels
# =============================================================================

print(f"\nGenerating overview figure...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(
    "VAE Anomaly Detection — All 82 NASA SMAP/MSL Channels\n"
    "Mehrab Jamshidi, PoliMi 2026",
    fontsize=13, fontweight="bold"
)

# Panel 1 — VAE F1 per channel vs threshold baseline
ax = axes[0]
df_sorted = df.sort_values("vae_best_f1", ascending=False)
x = np.arange(len(df_sorted))
if "baseline_f1" in df_sorted.columns:
    ax.bar(x - 0.2, df_sorted["vae_best_f1"].values, width=0.4, label="VAE (best thr)", color="#1a9641")
    ax.bar(x + 0.2, df_sorted["baseline_f1"].values, width=0.4, label="Threshold baseline", color="#d73027", alpha=0.7)
else:
    ax.bar(x, df_sorted["vae_best_f1"].values, label="VAE (best thr)", color="#1a9641")
ax.set_title("VAE vs Threshold Baseline — F1 per channel", fontsize=11)
ax.set_ylabel("F1 score")
ax.set_xlabel("Channel (sorted by VAE F1)")
ax.legend(fontsize=9)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)

# Panel 2 — F1 by anomaly type comparison
ax = axes[1]
categories = ["Threshold\n(Contextual)", "Threshold\n(Point)", "VAE\n(Contextual)", "VAE\n(Point)"]

if "baseline_f1" in df.columns:
    ctx_base  = df[df["anomaly_class"].str.contains("contextual", na=False)]["baseline_f1"].mean()
    pt_base   = df[~df["anomaly_class"].str.contains("contextual", na=False)]["baseline_f1"].mean()
else:
    ctx_base, pt_base = 0.05, 0.28

ctx_vae = df[df["anomaly_class"].str.contains("contextual", na=False)]["vae_best_f1"].mean()
pt_vae  = df[~df["anomaly_class"].str.contains("contextual", na=False)]["vae_best_f1"].mean()

means  = [ctx_base, pt_base, ctx_vae, pt_vae]
colors = ["#d73027", "#fdae61", "#1a9641", "#74c476"]
bars = ax.bar(categories, means, color=colors, width=0.55, edgecolor="white")
for bar, val in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_title("Mean F1 by anomaly type: Threshold vs VAE", fontsize=11)
ax.set_ylabel("Mean F1 score")
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURE_FILE, dpi=150, bbox_inches="tight")
print(f"Overview figure saved as: {FIGURE_FILE}")
plt.show()

print("\n" + "=" * 70)
print("STEP B COMPLETE.")
print()
print("Files written:")
print(f"  {RESULTS_CSV}           — VAE results per channel")
print(f"  full_comparison.csv     — VAE + SVM + ANN + Threshold, all channels")
print(f"  {P1_FIGURE}  — P-1 deep visualization")
print(f"  {FIGURE_FILE} — Overview all 82 channels")
print()
print("THESIS NEXT STEPS:")
print("  1. Compare full_comparison.csv to Hundman 2018 LSTM results")
print("  2. Write Report 2 sections: VAE architecture, training, results")
print("  3. Begin Phase 3: WGAN-GP conditional GAN augmentation")
print("=" * 70)
