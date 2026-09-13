"""
================================================================================
PHASE 3 — Conditional WGAN-GP for Spacecraft Telemetry Fault Synthesis
              (clean channel-level held-out protocol)
================================================================================

Author:     Mehrab Jamshidi — Politecnico di Milano
Supervisor: Prof. Andrea Colagrossi
Thesis:     Generative Digital Twin Methods for Spacecraft Telemetry Anomaly
            Detection: Toward Data-Driven FDIR Design and Testing
            MSc Aeronautical Engineering, Politecnico di Milano, AY 2025-26

================================================================================
WHAT THIS SCRIPT PRODUCES
================================================================================

Two class-specialised generators of synthetic fault windows (point and
contextual), plus the synthetic window banks that Phase 4 consumes:

    models/gan_v4_clean_generator_point.pt        point-anomaly generator
    models/gan_v4_clean_generator_contextual.pt   contextual-anomaly generator
    models/gan_v4_clean_critic_{point,contextual}.pt
    data/synthetic/gan_v4_clean_synth_point.npy       2000 windows, (2000, 64)
    data/synthetic/gan_v4_clean_synth_contextual.npy  2000 windows, (2000, 64)

================================================================================
THE HELD-OUT PROTOCOL (why this version exists)
================================================================================

An earlier revision of this pipeline trained the GAN on anomaly windows drawn
from the NASA *test* split of EVERY channel, and Phase 4 then evaluated
synthetic-to-real detection on that same split. The synthetic distribution was
therefore shaped by the very anomalies used for evaluation, which invalidates
the "train on synthetic, test on UNSEEN real" claim.

This version imports `compute_split()` from `holdout_split.py` and SKIPS the 20
held-out channels entirely (see the `if chan in HOLDOUT_CHANNELS: continue`
guard in Section 1). The generator is trained on the 61 training channels only,
and every headline number in Phase 4 is measured on the 20 channels whose
anomalies were never seen here.

A per-channel anomaly holdout is impossible — 63 of the 81 channels have only
one labelled anomaly sequence — so the channel is the only clean unit. See
docs/CLEAN_RERUN_PROTOCOL.md for the full argument.

================================================================================
THE FOUR ARCHITECTURAL CONTRIBUTIONS
================================================================================

These are the four choices that go beyond a standard conditional WGAN-GP, each
motivated by a specific failure of the standard formulation on this dataset.
(A fifth item below, built-in TSTR validation, is an EVALUATION step rather than
an architectural contribution, and is listed separately for that reason.)

An earlier MLP-based version produced amplitude-correct but temporally
incoherent windows: the autocorrelation function of generated windows was
essentially white noise (ACF(1) ~ 0.1) while real anomaly windows have strong
temporal correlation (ACF(1) ~ 0.65-0.75). For the Phase 4 protocol temporal
fidelity is essential — the LSTM models temporal dynamics and fails
catastrophically on temporally incoherent training data.

Caveat stated up front: the four contributions were adopted together and
motivated qualitatively. Their individual marginal effect was NOT isolated in a
controlled ablation against a plain conditional WGAN-GP, and no nearest-neighbour
test was run to rule out memorisation of training windows. Both are identified as
the natural next refinements of this evaluation.

  1. 1D CONVOLUTIONAL ARCHITECTURE
     Transposed-convolution generator + convolution critic instead of an MLP.
     Convolutional kernels naturally preserve local temporal structure.
     Standard for time-series GANs (Yoon et al. 2019, Esteban et al. 2017).

  2. EXPLICIT AUTOCORRELATION LOSS
     L_ACF = ||ACF_real - ACF_synth||^2 added to the generator loss over lags
     1..20, directly penalising temporal-structure mismatch. Weight LAMBDA_ACF
     = 5.0, chosen by a small ablation.

  3. SPECTRAL NORMALIZATION INSTEAD OF BATCHNORM
     Spectral norm enforces 1-Lipschitz at the layer level and is mathematically
     compatible with the gradient penalty, which BatchNorm violates
     (Miyato et al. 2018).

  4. SEPARATE GENERATORS PER ANOMALY CLASS
     With only a few dozen sequences per class, conditional generation forces
     mode averaging. Two specialised generators each see their own class and
     produce sharper, more class-specific output.

PLUS, as an evaluation step rather than an architectural contribution:

     BUILT-IN TSTR VALIDATION
     "Train on Synthetic, Test on Real" — a quick SVM trained on synthetic
     windows and evaluated on real test windows, giving immediate signal on
     whether the output is useful downstream.

================================================================================
TRAINING POOL (after the 20 held-out channels are excluded)
================================================================================

  point generator      2000 real point-anomaly windows
  contextual generator  792 real contextual-anomaly windows
  parameters            generator 184,865   critic 42,337   (per class)

================================================================================
MEASURED RESULTS (this run — see results/gan_v4_clean_quality_report.csv)
================================================================================

  Frechet Distance   contextual 0.141   point 0.403
  MMD                contextual 0.085   point 0.175
  ACF mean |delta|   contextual 0.026   point 0.015    (target was < 0.05)

CAREFUL — two different numbers are both called "TSTR", and they are NOT the
same quantity:

  * IN-SAMPLE PREVIEW (what THIS script's gan_v4_clean_tstr_results.csv holds):
    computed across all 81 channels, including the 20 held out.
        point 0.450   contextual 0.326   overall 0.405
    Excluding the held-out channels from GAN training moved this preview only
    marginally, from 0.411 to 0.405, which is evidence the held-out exclusion
    did not degrade the generator.

  * THE THESIS RESULT (held-out transfer, Phase 4, on the 20 unseen channels):
        overall 0.474   point 0.57   contextual 0.24
    This is the authoritative synthetic-to-real number and it comes from
    phase4_synthetic_training.py, not from here.

The distributional metrics above are likewise IN-SAMPLE fidelity checks: they
compare the synthetic windows against the generator's own training windows, not
against held-out anomalies.

Honest caveat: point-class fidelity (Frechet 0.403) is clearly weaker than
contextual (0.141) — synthetic point windows are slightly smoother than the
sharpest real spikes. This is disclosed rather than smoothed over, and it does
not block downstream detection.

================================================================================
RUNTIME
================================================================================

This is the only long, GPU-heavy step in the pipeline (~3000 critic steps per
class). Phase 4 can skip it entirely by reading the committed
data/synthetic/gan_v4_clean_synth_*.npy files. Set SKIP_TRAINING = True to
reload the committed weights instead of retraining.
================================================================================
"""

import numpy as np
import pandas as pd
import os
import re
import sys
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from scipy.linalg import sqrtm
from scipy.stats import ks_2samp
import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths come from src/config.py — point at the dataset with SMAP_MSL_ROOT.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (  # noqa: E402
    DATASET_ROOT, LABELS_FILE, TEST_FOLDER, TRAIN_FOLDER,
    RESULTS_DIR, FIGURES_DIR, MODELS_DIR, SYNTH_DIR, require_dataset,
)
from holdout_split import compute_split  # noqa: E402

require_dataset()

# Window — identical to all other thesis scripts
WINDOW_SIZE        = 64
STEP_SIZE          = 16
OVERLAP_THRESHOLD  = 0.1
CLIP_SIGMA         = 3.0

# GAN architecture
LATENT_DIM         = 100
BASE_CHANNELS      = 32   # base channel count for conv layers
                          # G: 32 → 64 → 128 → 1 (upsampling)
                          # D: 1 → 32 → 64 → 128 → 1 (downsampling)

# WGAN-GP standard
N_CRITIC           = 5
LAMBDA_GP          = 10.0
LAMBDA_ACF         = 5.0   # weight for autocorrelation matching loss
ACF_MAX_LAG        = 20    # lags 1..20 for ACF loss

# TTUR learning rates
LR_G               = 1e-4
LR_D               = 4e-4
BETA1, BETA2       = 0.0, 0.9

BATCH_SIZE         = 64
GAN_STEPS          = 3000  # increased for full convergence with new architecture

# Synthetic data generation
N_SYNTH_PER_CLASS  = 2000

# Reproducibility and execution
SEED               = 42
SKIP_TRAINING      = False  # set True to load existing weights

# Output file names
PREFIX        = "gan_v4_clean"
GEN_PT        = MODELS_DIR  / f"{PREFIX}_generator_point.pt"
GEN_CTX       = MODELS_DIR  / f"{PREFIX}_generator_contextual.pt"
CRIT_PT       = MODELS_DIR  / f"{PREFIX}_critic_point.pt"
CRIT_CTX      = MODELS_DIR  / f"{PREFIX}_critic_contextual.pt"
SYNTH_PT_NPY  = SYNTH_DIR   / f"{PREFIX}_synth_point.npy"
SYNTH_CTX_NPY = SYNTH_DIR   / f"{PREFIX}_synth_contextual.npy"
FIG_TRAIN     = FIGURES_DIR / f"{PREFIX}_training_curves.png"
FIG_SAMPLES   = FIGURES_DIR / f"{PREFIX}_samples.png"
FIG_ACF       = FIGURES_DIR / f"{PREFIX}_acf_comparison.png"
FIG_FEATS     = FIGURES_DIR / f"{PREFIX}_feature_distributions.png"
FIG_TSNE      = FIGURES_DIR / f"{PREFIX}_tsne.png"
QUALITY_CSV   = RESULTS_DIR / f"{PREFIX}_quality_report.csv"
TSTR_CSV      = RESULTS_DIR / f"{PREFIX}_tstr_results.csv"
OVERVIEW_FIG  = FIGURES_DIR / f"{PREFIX}_overview.png"

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 78)
print("PHASE 3 v3 — CONDITIONAL WGAN-GP WITH TEMPORAL FIDELITY")
print("Conv architecture + ACF loss + Spectral norm + Per-class generators")
print("=" * 78)
print(f"Device:        {DEVICE}")
print(f"Latent dim:    {LATENT_DIM}")
print(f"Base channels: {BASE_CHANNELS}")
print(f"GAN steps:     {GAN_STEPS}")
print(f"λ_GP:          {LAMBDA_GP}")
print(f"λ_ACF:         {LAMBDA_ACF}")
print(f"Synth/class:   {N_SYNTH_PER_CLASS}")
print(f"Skip training: {SKIP_TRAINING}")
print("=" * 78)

# =============================================================================
# SECTION 1 — UTILITY FUNCTIONS
# =============================================================================

def parse_anomaly_windows(seq_str):
    """Parse '[[s,e],[s,e]]' → list of [s,e] pairs."""
    nums = re.findall(r'\d+', str(seq_str))
    nums = [int(n) for n in nums]
    return [[nums[i], nums[i+1]] for i in range(0, len(nums)-1, 2)]


def parse_class_labels(class_str):
    """Parse '[contextual, point]' → ['contextual', 'point']."""
    return re.findall(r'contextual|point', str(class_str))


def scale_to_tanh(window_zscore, clip_sigma=3.0):
    """
    Scale z-score normalized signal to [-1, 1].
    Same approach as v2 — channel z-score, clip at ±3σ, divide by 3.
    Preserves amplitude information critical for distinguishing classes.
    """
    return (np.clip(window_zscore, -clip_sigma, clip_sigma) / clip_sigma).astype(np.float32)


def extract_features(window):
    """Same 6 features as Step A SVM — for evaluation only."""
    mean = float(np.mean(window))
    std  = float(np.std(window))
    mn   = float(np.min(window))
    mx   = float(np.max(window))
    rng  = mx - mn

    if std < 1e-9:
        autocorr = 0.0
    else:
        w1, w2 = window[:-1] - mean, window[1:] - mean
        denom  = float(np.sum(w1 ** 2))
        autocorr = float(np.sum(w1 * w2) / denom) if denom > 1e-12 else 0.0

    return np.array([mean, std, mn, mx, rng, autocorr], dtype=np.float32)


def batch_features(windows):
    return np.array([extract_features(w) for w in windows])


def compute_acf_np(windows, max_lag=ACF_MAX_LAG):
    """Mean autocorrelation function across a batch — numpy version."""
    acf_list = []
    for w in windows:
        w_c = w - w.mean()
        var = float(np.var(w))
        if var < 1e-9:
            continue
        row = [1.0]
        for lag in range(1, max_lag + 1):
            row.append(float(np.mean(w_c[:-lag] * w_c[lag:]) / var))
        acf_list.append(row)
    return np.array(acf_list).mean(axis=0) if acf_list else np.zeros(max_lag + 1)


def compute_acf_torch(x, max_lag=ACF_MAX_LAG):
    """
    Batched ACF computation in PyTorch — differentiable.
    Input:  x of shape (batch, window_size)
    Output: ACF of shape (batch, max_lag+1) with ACF[:, 0] = 1.0
    """
    batch_size, window_size = x.shape

    # Center each window by its mean
    x_centered = x - x.mean(dim=1, keepdim=True)

    # Variance per window (with small epsilon to prevent zero division)
    variance = (x_centered ** 2).mean(dim=1, keepdim=True) + 1e-8

    # Build ACF: lag 0 is always 1, then lags 1..max_lag
    acf = [torch.ones(batch_size, device=x.device)]
    for lag in range(1, max_lag + 1):
        # Correlation at lag k: E[(x_t - μ)(x_{t+k} - μ)] / variance
        cov = (x_centered[:, :-lag] * x_centered[:, lag:]).mean(dim=1)
        acf.append(cov / variance.squeeze())

    return torch.stack(acf, dim=1)   # (batch, max_lag+1)


# =============================================================================
# SECTION 2 — DATA EXTRACTION
# =============================================================================

print("\n" + "=" * 78)
print("SECTION 1 — EXTRACTING REAL ANOMALY WINDOWS")
print("=" * 78)

labels_df = pd.read_csv(LABELS_FILE).drop_duplicates('chan_id', keep='first')

TRAIN_CHANNELS, HOLDOUT_CHANNELS = compute_split()
print(f"Clean protocol: GAN trains on {len(TRAIN_CHANNELS)} channels, {len(HOLDOUT_CHANNELS)} held out")

real_point      = []
real_contextual = []
n_padded        = 0
n_skipped       = 0

for _, row in labels_df.iterrows():
    chan       = row['chan_id']
    if chan in HOLDOUT_CHANNELS:
        continue
    anom_seqs  = parse_anomaly_windows(row['anomaly_sequences'])
    class_lbls = parse_class_labels(row['class'])

    test_path  = os.path.join(TEST_FOLDER,  f"{chan}.npy")
    train_path = os.path.join(TRAIN_FOLDER, f"{chan}.npy")
    if not (os.path.exists(test_path) and os.path.exists(train_path)):
        continue

    # Channel-level z-score normalization using train statistics
    train_raw = np.load(train_path)[:, 0]
    test_raw  = np.load(test_path)[:, 0]
    mu, sigma = train_raw.mean(), train_raw.std()
    if sigma < 1e-9: sigma = 1.0
    test_norm = (test_raw - mu) / sigma

    for (start, end), anom_class in zip(anom_seqs, class_lbls):
        start, end = max(0, start), min(len(test_norm), end)
        seq_len = end - start

        if seq_len < WINDOW_SIZE:
            # Pad with surrounding context
            pad = WINDOW_SIZE - seq_len
            s = max(0, start - pad // 2)
            e = min(len(test_norm), end + (pad - pad // 2))
            w = test_norm[s:e]
            if len(w) < WINDOW_SIZE:
                n_skipped += 1
                continue
            windows = [w[:WINDOW_SIZE]]
            n_padded += 1
        else:
            seq = test_norm[start:end]
            windows = [seq[i:i+WINDOW_SIZE]
                       for i in range(0, len(seq) - WINDOW_SIZE + 1, STEP_SIZE)]

        for w in windows:
            w_scaled = scale_to_tanh(w, CLIP_SIGMA)
            if anom_class == 'point':
                real_point.append(w_scaled)
            else:
                real_contextual.append(w_scaled)

real_point      = np.array(real_point,      dtype=np.float32)
real_contextual = np.array(real_contextual, dtype=np.float32)

print(f"Point windows:      {len(real_point)}")
print(f"Contextual windows: {len(real_contextual)}")
print(f"Padded (short):     {n_padded}")
print(f"Skipped:            {n_skipped}")
print(f"\nRange check (must be in [-1,1]):")
print(f"  Point:      [{real_point.min():.4f}, {real_point.max():.4f}]")
print(f"  Contextual: [{real_contextual.min():.4f}, {real_contextual.max():.4f}]")
print(f"\nClass separation (mean |value|):")
print(f"  Point:      {np.abs(real_point).mean():.4f}  (expect higher — large amplitudes)")
print(f"  Contextual: {np.abs(real_contextual).mean():.4f}  (expect lower — within normal bounds)")

# Pre-compute real ACF profiles for ACF loss
acf_real_pt_torch  = torch.FloatTensor(compute_acf_np(real_point,      ACF_MAX_LAG)).to(DEVICE)
acf_real_ctx_torch = torch.FloatTensor(compute_acf_np(real_contextual, ACF_MAX_LAG)).to(DEVICE)

print(f"\nReal ACF (lag 1) target:")
print(f"  Point:      {acf_real_pt_torch[1].item():.4f}")
print(f"  Contextual: {acf_real_ctx_torch[1].item():.4f}")

# =============================================================================
# SECTION 3 — CONVOLUTIONAL ARCHITECTURE
# =============================================================================

class Generator1D(nn.Module):
    """
    1D Convolutional generator using transposed convolutions for upsampling.

    Architecture (input → output):
        z (LATENT_DIM,) → reshape to (LATENT_DIM, 1)
        → ConvTranspose1d kernel=8 stride=4 → (128, 4)
        → ConvTranspose1d kernel=8 stride=4 → (64, 16)
        → ConvTranspose1d kernel=8 stride=4 → (32, 64)
        → Conv1d kernel=3                    → (1, 64)
        → Tanh

    The transposed convolution upsamples while preserving local structure.
    Each layer expands the temporal extent 4x and reduces channels.
    The final conv collapses channels to 1 (univariate output).

    Why this is fundamentally better than MLP:
    - Convolutional kernels learn local temporal patterns (3-8 timesteps)
    - Translation invariance — same pattern recognized anywhere in window
    - Hierarchical feature learning — early layers capture short patterns,
      deeper layers capture longer-range temporal structure
    - This is the standard architecture for audio/time-series GANs
    """
    def __init__(self, latent_dim=100, base_ch=BASE_CHANNELS, window_size=64):
        super().__init__()
        self.latent_dim  = latent_dim
        self.window_size = window_size

        # 1 → 4 → 16 → 64 (window size grows 4x per layer)
        self.net = nn.Sequential(
            # Input shape: (batch, latent_dim, 1)
            nn.ConvTranspose1d(latent_dim, base_ch*4, kernel_size=8, stride=4,
                              padding=2, bias=False),
            nn.BatchNorm1d(base_ch*4),
            nn.LeakyReLU(0.2, inplace=True),
            # → (batch, 128, 4)

            nn.ConvTranspose1d(base_ch*4, base_ch*2, kernel_size=8, stride=4,
                              padding=2, bias=False),
            nn.BatchNorm1d(base_ch*2),
            nn.LeakyReLU(0.2, inplace=True),
            # → (batch, 64, 16)

            nn.ConvTranspose1d(base_ch*2, base_ch, kernel_size=8, stride=4,
                              padding=2, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.LeakyReLU(0.2, inplace=True),
            # → (batch, 32, 64)

            nn.Conv1d(base_ch, 1, kernel_size=3, padding=1),
            nn.Tanh()
            # → (batch, 1, 64)
        )

    def forward(self, z):
        # z shape: (batch, latent_dim)
        z = z.unsqueeze(-1)              # (batch, latent_dim, 1)
        out = self.net(z)                # (batch, 1, 64)
        return out.squeeze(1)            # (batch, 64)


class Critic1D(nn.Module):
    """
    1D Convolutional critic with spectral normalization.

    Architecture:
        (1, 64) → Conv k=4 s=2 → (32, 32)
                → Conv k=4 s=2 → (64, 16)
                → Conv k=4 s=2 → (128, 8)
                → Flatten → Linear → 1

    Spectral normalization enforces 1-Lipschitz at each layer by dividing
    the weight matrix by its largest singular value. Combined with gradient
    penalty, this provides much stronger Lipschitz enforcement than either
    technique alone. Critical for stable WGAN-GP training.

    Why no BatchNorm: introduces inter-sample dependencies that break
    gradient penalty computation (gradients become entangled across batch).
    Spectral norm is the correct replacement for GAN critics.
    """
    def __init__(self, base_ch=BASE_CHANNELS, window_size=64):
        super().__init__()
        self.window_size = window_size

        # Helper for spectral-normalized conv
        def sn_conv(in_ch, out_ch, k=4, s=2, p=1):
            return spectral_norm(nn.Conv1d(in_ch, out_ch, k, s, p))

        self.conv_net = nn.Sequential(
            sn_conv(1, base_ch),                    # (1,64) → (32,32)
            nn.LeakyReLU(0.2, inplace=True),

            sn_conv(base_ch, base_ch*2),            # → (64,16)
            nn.LeakyReLU(0.2, inplace=True),

            sn_conv(base_ch*2, base_ch*4),          # → (128,8)
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Flatten 128*8 = 1024 → 1
        self.fc = spectral_norm(nn.Linear(base_ch*4 * 8, 1))

    def forward(self, x):
        # x shape: (batch, 64)
        x = x.unsqueeze(1)               # (batch, 1, 64)
        h = self.conv_net(x)              # (batch, 128, 8)
        h = h.view(h.size(0), -1)         # (batch, 1024)
        return self.fc(h)                 # (batch, 1)


def compute_gradient_penalty(critic, real, fake, device, lambda_gp=10.0):
    """
    Standard WGAN-GP gradient penalty.
    Forces critic to be 1-Lipschitz along interpolations between real and fake.

    With spectral normalization, this provides redundant but mutually
    reinforcing Lipschitz enforcement. Standard practice in modern WGAN-GP
    implementations to use both.
    """
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, device=device).expand_as(real)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)

    d_interp = critic(interp)
    gradients = torch.autograd.grad(
        outputs=d_interp, inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]

    grad_norm = gradients.view(batch_size, -1).norm(2, dim=1)
    return lambda_gp * ((grad_norm - 1) ** 2).mean()


def init_weights(m):
    """Standard DCGAN initialization for conv layers."""
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.zeros_(m.bias.data)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# =============================================================================
# SECTION 4 — TRAINING FUNCTION (called twice, once per class)
# =============================================================================

def train_wgan_for_class(real_windows, acf_target, class_name, n_steps=GAN_STEPS):
    """
    Train one WGAN-GP on a single anomaly class.

    Loss structure:
      L_critic    = -E[D(real)] + E[D(fake)] + λ_GP · GP
      L_generator = -E[D(fake)] + λ_ACF · ||ACF(fake) - ACF(real)||²

    The ACF loss is the critical addition: it explicitly penalises temporal
    structure mismatch, directly addressing the v2 failure mode.

    Returns: (generator, critic, history dict)
    """
    print(f"\n{'='*78}")
    print(f"  Training generator for class: {class_name.upper()}")
    print(f"  Training samples: {len(real_windows)}")
    print(f"{'='*78}")

    # Build data loader
    X = torch.FloatTensor(real_windows)
    loader = DataLoader(TensorDataset(X), batch_size=BATCH_SIZE,
                       shuffle=True, drop_last=True)

    # Initialize models
    generator = Generator1D(LATENT_DIM, BASE_CHANNELS, WINDOW_SIZE).to(DEVICE)
    critic    = Critic1D(BASE_CHANNELS, WINDOW_SIZE).to(DEVICE)
    generator.apply(init_weights)
    critic.apply(init_weights)

    # Optimisers — TTUR with critic LR = 4x generator LR
    opt_G = optim.Adam(generator.parameters(), lr=LR_G, betas=(BETA1, BETA2))
    opt_D = optim.Adam(critic.parameters(),    lr=LR_D, betas=(BETA1, BETA2))

    print(f"  Generator params: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"  Critic params:    {sum(p.numel() for p in critic.parameters()):,}")
    print()

    history = {'g_loss': [], 'd_loss': [], 'gp': [], 'w_dist': [],
               'acf_loss': []}
    data_iter = iter(loader)

    for step in range(n_steps):

        # ---- Critic training: N_CRITIC updates per generator update ----
        for _ in range(N_CRITIC):
            try:
                (real_x,) = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                (real_x,) = next(data_iter)

            real_x = real_x.to(DEVICE)
            bsz    = real_x.size(0)

            z      = torch.randn(bsz, LATENT_DIM, device=DEVICE)
            fake_x = generator(z).detach()

            d_real = critic(real_x).mean()
            d_fake = critic(fake_x).mean()
            gp     = compute_gradient_penalty(critic, real_x, fake_x, DEVICE, LAMBDA_GP)

            loss_D = -d_real + d_fake + gp

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

        # ---- Generator training: 1 update with ACF loss ----
        z      = torch.randn(BATCH_SIZE, LATENT_DIM, device=DEVICE)
        fake_x = generator(z)

        # Adversarial loss
        adv_loss = -critic(fake_x).mean()

        # Autocorrelation matching loss
        # Compute ACF of generated batch, compare to pre-computed real ACF
        fake_acf  = compute_acf_torch(fake_x, ACF_MAX_LAG)  # (batch, max_lag+1)
        mean_fake_acf = fake_acf.mean(dim=0)                  # average over batch
        acf_loss  = ((mean_fake_acf - acf_target) ** 2).mean()

        loss_G = adv_loss + LAMBDA_ACF * acf_loss

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        # Track metrics
        w_dist = (d_real - d_fake).item()
        history['g_loss'].append(loss_G.item())
        history['d_loss'].append(loss_D.item())
        history['gp'].append(gp.item())
        history['w_dist'].append(abs(w_dist))
        history['acf_loss'].append(acf_loss.item())

        if (step + 1) % 200 == 0 or step == 0:
            print(f"    Step [{step+1:4d}/{n_steps}] | "
                  f"L_G={loss_G.item():+.4f} | "
                  f"L_D={loss_D.item():+.4f} | "
                  f"W={w_dist:+.4f} | "
                  f"GP={gp.item():.4f} | "
                  f"ACF_loss={acf_loss.item():.4f}")

    generator.eval()
    critic.eval()
    return generator, critic, history


# =============================================================================
# SECTION 5 — TRAIN BOTH GENERATORS
# =============================================================================

if SKIP_TRAINING:
    print("\nLoading pre-trained weights (SKIP_TRAINING=True)...")
    gen_pt  = Generator1D(LATENT_DIM, BASE_CHANNELS, WINDOW_SIZE).to(DEVICE)
    gen_ctx = Generator1D(LATENT_DIM, BASE_CHANNELS, WINDOW_SIZE).to(DEVICE)
    crit_pt  = Critic1D(BASE_CHANNELS, WINDOW_SIZE).to(DEVICE)
    crit_ctx = Critic1D(BASE_CHANNELS, WINDOW_SIZE).to(DEVICE)

    gen_pt.load_state_dict(torch.load(GEN_PT,   map_location=DEVICE, weights_only=True))
    gen_ctx.load_state_dict(torch.load(GEN_CTX, map_location=DEVICE, weights_only=True))
    crit_pt.load_state_dict(torch.load(CRIT_PT, map_location=DEVICE, weights_only=True))
    crit_ctx.load_state_dict(torch.load(CRIT_CTX, map_location=DEVICE, weights_only=True))
    gen_pt.eval(); gen_ctx.eval(); crit_pt.eval(); crit_ctx.eval()

    hist_pt  = {'g_loss': [], 'd_loss': [], 'gp': [], 'w_dist': [], 'acf_loss': []}
    hist_ctx = {'g_loss': [], 'd_loss': [], 'gp': [], 'w_dist': [], 'acf_loss': []}
    print("Loaded.")
else:
    gen_pt,  crit_pt,  hist_pt  = train_wgan_for_class(
        real_point,      acf_real_pt_torch,  'point')
    gen_ctx, crit_ctx, hist_ctx = train_wgan_for_class(
        real_contextual, acf_real_ctx_torch, 'contextual')

    torch.save(gen_pt.state_dict(),  GEN_PT)
    torch.save(gen_ctx.state_dict(), GEN_CTX)
    torch.save(crit_pt.state_dict(),  CRIT_PT)
    torch.save(crit_ctx.state_dict(), CRIT_CTX)
    print(f"\nWeights saved:")
    print(f"  {GEN_PT}, {GEN_CTX}")
    print(f"  {CRIT_PT}, {CRIT_CTX}")

# =============================================================================
# SECTION 6 — GENERATE SYNTHETIC WINDOWS
# =============================================================================

print("\n" + "=" * 78)
print("SECTION 2 — GENERATING SYNTHETIC WINDOWS")
print("=" * 78)

def generate_batch(generator, n_samples, device, batch_size=512):
    generator.eval()
    out = []
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            bsz = min(batch_size, n_samples - i)
            z   = torch.randn(bsz, LATENT_DIM, device=device)
            x   = generator(z)
            out.append(x.cpu().numpy())
    return np.vstack(out).astype(np.float32)

synth_point      = generate_batch(gen_pt,  N_SYNTH_PER_CLASS, DEVICE)
synth_contextual = generate_batch(gen_ctx, N_SYNTH_PER_CLASS, DEVICE)

print(f"Synthetic point:      {synth_point.shape}")
print(f"Synthetic contextual: {synth_contextual.shape}")
print(f"Range point:          [{synth_point.min():.4f}, {synth_point.max():.4f}]")
print(f"Range contextual:     [{synth_contextual.min():.4f}, {synth_contextual.max():.4f}]")

np.save(SYNTH_PT_NPY,  synth_point)
np.save(SYNTH_CTX_NPY, synth_contextual)
print(f"\nSaved: {SYNTH_PT_NPY}, {SYNTH_CTX_NPY}")

# =============================================================================
# SECTION 7 — QUALITY EVALUATION
# =============================================================================

print("\n" + "=" * 78)
print("SECTION 3 — QUALITY EVALUATION")
print("=" * 78)

feat_real_pt   = batch_features(real_point)
feat_real_ctx  = batch_features(real_contextual)
feat_synth_pt  = batch_features(synth_point)
feat_synth_ctx = batch_features(synth_contextual)
feature_names  = ['mean', 'std', 'min', 'max', 'range', 'autocorr']


def frechet_distance(feat_r, feat_s):
    mu_r, mu_s = feat_r.mean(0), feat_s.mean(0)
    cov_r = np.cov(feat_r.T) + np.eye(6) * 1e-6
    cov_s = np.cov(feat_s.T) + np.eye(6) * 1e-6
    diff  = mu_r - mu_s
    cm    = sqrtm(cov_r @ cov_s)
    if np.iscomplexobj(cm): cm = cm.real
    return float((diff @ diff) + np.trace(cov_r + cov_s - 2 * cm))


def mmd_rbf(X, Y, max_samples=500):
    X = X[:max_samples]; Y = Y[:max_samples]
    all_pts = np.vstack([X, Y])
    dists   = np.sum((all_pts[:,None] - all_pts[None,:])**2, axis=2)
    sigma   = np.sqrt(np.median(dists[dists>0]) / 2)
    if sigma < 1e-8: sigma = 1.0

    def k(A, B):
        d = np.sum((A[:,None] - B[None,:])**2, axis=2)
        return np.exp(-d / (2 * sigma**2))

    kxx, kyy, kxy = k(X,X), k(Y,Y), k(X,Y)
    n, m = len(X), len(Y)
    np.fill_diagonal(kxx, 0); np.fill_diagonal(kyy, 0)
    return float(max(0, kxx.sum()/(n*(n-1)) + kyy.sum()/(m*(m-1)) - 2*kxy.mean()))


# Compute all metrics
fd_pt   = frechet_distance(feat_real_pt,  feat_synth_pt)
fd_ctx  = frechet_distance(feat_real_ctx, feat_synth_ctx)
mmd_pt  = mmd_rbf(StandardScaler().fit_transform(np.vstack([feat_real_pt, feat_synth_pt]))[:len(feat_real_pt)],
                  StandardScaler().fit_transform(np.vstack([feat_real_pt, feat_synth_pt]))[len(feat_real_pt):])
# Cleaner MMD with shared scaler
scaler_eval = StandardScaler().fit(np.vstack([feat_real_pt, feat_real_ctx]))
fp_r = scaler_eval.transform(feat_real_pt);   fp_s = scaler_eval.transform(feat_synth_pt)
fc_r = scaler_eval.transform(feat_real_ctx);  fc_s = scaler_eval.transform(feat_synth_ctx)
mmd_pt  = mmd_rbf(fp_r, fp_s)
mmd_ctx = mmd_rbf(fc_r, fc_s)

# ACF comparison
acf_real_pt_np  = compute_acf_np(real_point,      ACF_MAX_LAG)
acf_real_ctx_np = compute_acf_np(real_contextual, ACF_MAX_LAG)
acf_syn_pt_np   = compute_acf_np(synth_point,     ACF_MAX_LAG)
acf_syn_ctx_np  = compute_acf_np(synth_contextual, ACF_MAX_LAG)
acf_diff_pt     = float(np.abs(acf_real_pt_np  - acf_syn_pt_np).mean())
acf_diff_ctx    = float(np.abs(acf_real_ctx_np - acf_syn_ctx_np).mean())

print("\nMetric 1 — Fréchet Distance (lower is better):")
print(f"  Point:      {fd_pt:.4f}  (v2 was 1.17)")
print(f"  Contextual: {fd_ctx:.4f}  (v2 was 1.63)")
print(f"\nMetric 2 — Maximum Mean Discrepancy:")
print(f"  Point:      {mmd_pt:.4f}  (v2 was 0.64)")
print(f"  Contextual: {mmd_ctx:.4f}  (v2 was 0.48)")
print(f"\nMetric 3 — Autocorrelation Mean |Δ|:")
print(f"  Point:      {acf_diff_pt:.4f}  (v2 was 0.20)")
print(f"  Contextual: {acf_diff_ctx:.4f}  (v2 was 0.24)")
print(f"  ACF(1) real point:        {acf_real_pt_np[1]:.4f}")
print(f"  ACF(1) synth point:       {acf_syn_pt_np[1]:.4f}")
print(f"  ACF(1) real contextual:   {acf_real_ctx_np[1]:.4f}")
print(f"  ACF(1) synth contextual:  {acf_syn_ctx_np[1]:.4f}")

# KS tests
print("\nMetric 4 — Kolmogorov-Smirnov tests per feature:")
print(f"  {'Feature':<12} {'KS pt':>8} {'p pt':>8} {'KS ctx':>8} {'p ctx':>8}")
ks_results = {}
for i, fn in enumerate(feature_names):
    ks_p, p_p = ks_2samp(feat_real_pt[:,i],  feat_synth_pt[:,i])
    ks_c, p_c = ks_2samp(feat_real_ctx[:,i], feat_synth_ctx[:,i])
    ks_results[fn] = (ks_p, p_p, ks_c, p_c)
    print(f"  {fn:<12} {ks_p:>8.4f} {p_p:>8.4f} {ks_c:>8.4f} {p_c:>8.4f}")

# Save quality report
quality_rows = [
    {'metric':'Frechet Distance', 'class':'point',      'value':fd_pt},
    {'metric':'Frechet Distance', 'class':'contextual', 'value':fd_ctx},
    {'metric':'MMD',              'class':'point',      'value':mmd_pt},
    {'metric':'MMD',              'class':'contextual', 'value':mmd_ctx},
    {'metric':'ACF Mean |Δ|',     'class':'point',      'value':acf_diff_pt},
    {'metric':'ACF Mean |Δ|',     'class':'contextual', 'value':acf_diff_ctx},
]
for fn, (ks_p, p_p, ks_c, p_c) in ks_results.items():
    quality_rows.append({'metric':f'KS_{fn}', 'class':'point',      'value':ks_p})
    quality_rows.append({'metric':f'KS_{fn}', 'class':'contextual', 'value':ks_c})
pd.DataFrame(quality_rows).to_csv(QUALITY_CSV, index=False)
print(f"\nSaved: {QUALITY_CSV}")

# =============================================================================
# SECTION 8 — TSTR VALIDATION (Train on Synthetic, Test on Real)
# =============================================================================

print("\n" + "=" * 78)
print("SECTION 4 — TSTR VALIDATION (Train-Synthetic Test-Real)")
print("=" * 78)
print("Train an SVM on synthetic data, evaluate on real test windows.")
print("This directly previews how useful the synthetic data will be for Phase 4.")
print()

tstr_results = []

for _, row in labels_df.iterrows():
    chan       = row['chan_id']
    sc         = row['spacecraft']
    anom_class = row['class']
    anom_seqs  = parse_anomaly_windows(row['anomaly_sequences'])

    test_path  = os.path.join(TEST_FOLDER,  f"{chan}.npy")
    train_path = os.path.join(TRAIN_FOLDER, f"{chan}.npy")
    if not (os.path.exists(test_path) and os.path.exists(train_path)):
        continue

    train_raw = np.load(train_path)[:, 0]
    test_raw  = np.load(test_path)[:, 0]
    mu, sigma = train_raw.mean(), train_raw.std()
    if sigma < 1e-9: sigma = 1.0
    train_norm = (train_raw - mu) / sigma
    test_norm  = (test_raw  - mu) / sigma

    # Build test set windows (REAL, with labels)
    y_true_ts = np.zeros(len(test_raw), dtype=int)
    for s, e in anom_seqs:
        y_true_ts[max(0,s):min(len(test_raw),e)] = 1

    X_test = []
    y_test = []
    for start in range(0, len(test_norm) - WINDOW_SIZE + 1, STEP_SIZE):
        w  = scale_to_tanh(test_norm[start:start+WINDOW_SIZE], CLIP_SIGMA)
        lb = y_true_ts[start:start+WINDOW_SIZE]
        X_test.append(extract_features(w))
        y_test.append(1 if lb.mean() >= OVERLAP_THRESHOLD else 0)
    X_test = np.array(X_test); y_test = np.array(y_test)

    if y_test.sum() == 0 or y_test.sum() == len(y_test):
        tstr_results.append({'chan_id':chan, 'spacecraft':sc,
                            'anomaly_class':anom_class,
                            'tstr_f1':0.0, 'note':'no_class_variation'})
        continue

    # Build training nominal windows from train split (REAL nominal)
    # Then add SYNTHETIC anomaly windows of the appropriate class
    X_train_nom = []
    for start in range(0, len(train_norm) - WINDOW_SIZE + 1, STEP_SIZE):
        w = scale_to_tanh(train_norm[start:start+WINDOW_SIZE], CLIP_SIGMA)
        X_train_nom.append(extract_features(w))
    X_train_nom = np.array(X_train_nom)
    y_train_nom = np.zeros(len(X_train_nom), dtype=int)

    # Choose synthetic anomaly source by channel's class
    has_ctx = 'contextual' in str(anom_class)
    has_pt  = 'point' in str(anom_class)

    if has_ctx and has_pt:
        synth_use = np.vstack([synth_point[:N_SYNTH_PER_CLASS//2],
                               synth_contextual[:N_SYNTH_PER_CLASS//2]])
    elif has_ctx:
        synth_use = synth_contextual[:N_SYNTH_PER_CLASS]
    else:
        synth_use = synth_point[:N_SYNTH_PER_CLASS]

    X_synth_feat = batch_features(synth_use)
    y_synth      = np.ones(len(X_synth_feat), dtype=int)

    # Combine training set
    X_train = np.vstack([X_train_nom, X_synth_feat])
    y_train = np.concatenate([y_train_nom, y_synth])

    # Standardize
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Train SVM
    cw = compute_class_weight('balanced', classes=np.array([0,1]), y=y_train)
    svm = SVC(kernel='rbf', C=10.0, gamma='scale',
              class_weight={0:cw[0], 1:cw[1]}, random_state=SEED)
    svm.fit(X_train_sc, y_train)
    y_pred = svm.predict(X_test_sc)

    p  = precision_score(y_test, y_pred, zero_division=0)
    r  = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    tstr_results.append({
        'chan_id':chan, 'spacecraft':sc,
        'anomaly_class':anom_class,
        'tstr_precision':round(p,4),
        'tstr_recall':round(r,4),
        'tstr_f1':round(f1,4),
        'note':'ok'
    })

df_tstr = pd.DataFrame(tstr_results)
df_tstr.to_csv(TSTR_CSV, index=False)

ok = df_tstr[df_tstr.get('note', 'ok') == 'ok']
ctx_ok = ok[ok.anomaly_class.str.contains('contextual', na=False)]
pt_ok  = ok[~ok.anomaly_class.str.contains('contextual', na=False)]

print(f"\nTSTR Results — SVM trained on synthetic, tested on REAL:")
print(f"  Channels evaluated: {len(ok)}")
print(f"  Mean F1 all:        {ok.tstr_f1.mean():.4f}")
print(f"  Median F1:          {ok.tstr_f1.median():.4f}")
print(f"  Contextual F1:      {ctx_ok.tstr_f1.mean():.4f}")
print(f"  Point F1:           {pt_ok.tstr_f1.mean():.4f}")
print(f"  F1 > 0:             {(ok.tstr_f1>0).sum()}/{len(ok)}")
print(f"  F1 > 0.5:           {(ok.tstr_f1>0.5).sum()}/{len(ok)}")
print(f"\nFor reference (Step A baseline trained on REAL labeled fault data):")
print(f"  Mean F1:            0.598")
print(f"  Contextual F1:      0.508")
print(f"  Point F1:           0.651")

p1_row = ok[ok.chan_id=='P-1']
if len(p1_row) > 0:
    print(f"\nChannel P-1 TSTR F1: {p1_row.iloc[0].tstr_f1:.4f}")
    print(f"  (Threshold 0.012, Step A SVM 0.182, v2 GAN-SVM 0.299)")

# =============================================================================
# SECTION 9 — FIGURES
# =============================================================================

print("\n" + "=" * 78)
print("SECTION 5 — GENERATING FIGURES")
print("=" * 78)

# ---- Training curves (both classes overlaid) ----
if hist_pt['g_loss']:
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle("WGAN-GP Training Curves v3 — Both Classes\n"
                 "Top: Point | Bottom: Contextual",
                 fontsize=13, fontweight='bold')

    for row_i, (hist, label, color) in enumerate([
            (hist_pt,  'Point',      '#d73027'),
            (hist_ctx, 'Contextual', '#1a9641')]):
        steps = np.arange(1, len(hist['g_loss'])+1)
        ax = axes[row_i][0]
        ax.plot(steps, hist['g_loss'], color=color, lw=0.6, alpha=0.7)
        smooth = pd.Series(hist['g_loss']).rolling(50, min_periods=1).mean()
        ax.plot(steps, smooth, color=color, lw=1.5)
        ax.set_title(f"{label} — Generator Loss")
        ax.grid(True, alpha=0.3); ax.set_xlabel("Step")

        ax = axes[row_i][1]
        ax.plot(steps, hist['gp'], color=color, lw=0.6, alpha=0.5)
        smooth_gp = pd.Series(hist['gp']).rolling(50, min_periods=1).mean()
        ax.plot(steps, smooth_gp, color=color, lw=1.5)
        ax.set_title(f"{label} — Gradient Penalty (target < 2)")
        ax.grid(True, alpha=0.3); ax.set_xlabel("Step")

        ax = axes[row_i][2]
        ax.plot(steps, hist['acf_loss'], color=color, lw=0.6, alpha=0.5)
        smooth_acf = pd.Series(hist['acf_loss']).rolling(50, min_periods=1).mean()
        ax.plot(steps, smooth_acf, color=color, lw=1.5)
        ax.set_title(f"{label} — ACF Matching Loss (target → 0)")
        ax.grid(True, alpha=0.3); ax.set_xlabel("Step")

    plt.tight_layout()
    plt.savefig(FIG_TRAIN, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIG_TRAIN}")

# ---- Sample comparison ----
n_show = 6
fig, axes = plt.subplots(4, n_show, figsize=(20, 11))
fig.suptitle(
    "Real vs Synthetic Anomaly Windows — WGAN-GP v3 (Conv + ACF Loss)\n"
    "Row 1: Real Point | Row 2: Synth Point | Row 3: Real Ctx | Row 4: Synth Ctx",
    fontsize=12, fontweight='bold'
)
t = np.arange(WINDOW_SIZE)
row_data = [
    (real_point,        '#d73027', 'Real Point'),
    (synth_point,       '#fdae61', 'Synth Point'),
    (real_contextual,   '#1a9641', 'Real Ctx'),
    (synth_contextual,  '#74add1', 'Synth Ctx'),
]
for r_i, (data, color, label) in enumerate(row_data):
    for c_i in range(n_show):
        ax = axes[r_i][c_i]
        ax.plot(t, data[c_i], color=color, lw=1.0)
        ax.set_ylim(-1.5, 1.5); ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=6)
        if c_i == 0: ax.set_ylabel(label, fontsize=8, fontweight='bold')
        if r_i == 0: ax.set_title(f"#{c_i+1}", fontsize=9)
plt.tight_layout()
plt.savefig(FIG_SAMPLES, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {FIG_SAMPLES}")

# ---- ACF profiles ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Autocorrelation Function — Real vs Synthetic (WGAN-GP v3)\n"
             "ACF loss directly minimised during training",
             fontsize=12, fontweight='bold')
lags = np.arange(ACF_MAX_LAG + 1)

ax = axes[0]
ax.plot(lags, acf_real_pt_np, color='#d73027', lw=2, marker='o', ms=4, label='Real Point')
ax.plot(lags, acf_syn_pt_np,  color='#fdae61', lw=2, marker='s', ms=4, ls='--', label='Synth Point')
ax.fill_between(lags, acf_real_pt_np-0.05, acf_real_pt_np+0.05,
                alpha=0.15, color='#d73027')
ax.set_title(f"Point Anomalies  |ΔACF|={acf_diff_pt:.4f}", fontsize=11)
ax.set_xlabel("Lag"); ax.set_ylabel("Autocorrelation")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_ylim(-0.5, 1.1)

ax = axes[1]
ax.plot(lags, acf_real_ctx_np, color='#1a9641', lw=2, marker='o', ms=4, label='Real Ctx')
ax.plot(lags, acf_syn_ctx_np,  color='#74add1', lw=2, marker='s', ms=4, ls='--', label='Synth Ctx')
ax.fill_between(lags, acf_real_ctx_np-0.05, acf_real_ctx_np+0.05,
                alpha=0.15, color='#1a9641')
ax.set_title(f"Contextual Anomalies  |ΔACF|={acf_diff_ctx:.4f}", fontsize=11)
ax.set_xlabel("Lag"); ax.set_ylabel("Autocorrelation")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_ylim(-0.5, 1.1)

plt.tight_layout()
plt.savefig(FIG_ACF, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {FIG_ACF}")

# ---- Feature distributions ----
fig, axes = plt.subplots(2, 6, figsize=(22, 8))
fig.suptitle("Feature Distribution Comparison — WGAN-GP v3\n"
             "Top: Point | Bottom: Contextual",
             fontsize=12, fontweight='bold')
for i, fn in enumerate(feature_names):
    ax = axes[0][i]
    ax.hist(feat_real_pt[:,i],  bins=35, alpha=0.55, color='#d73027',
            label='Real', density=True)
    ax.hist(feat_synth_pt[:,i], bins=35, alpha=0.55, color='#fdae61',
            label='Synth', density=True)
    ks, p = ks_results[fn][0], ks_results[fn][1]
    ax.set_title(f"Pt — {fn}\nKS={ks:.3f}", fontsize=8)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[1][i]
    ax.hist(feat_real_ctx[:,i],  bins=35, alpha=0.55, color='#1a9641',
            label='Real', density=True)
    ax.hist(feat_synth_ctx[:,i], bins=35, alpha=0.55, color='#74add1',
            label='Synth', density=True)
    ks, p = ks_results[fn][2], ks_results[fn][3]
    ax.set_title(f"Ctx — {fn}\nKS={ks:.3f}", fontsize=8)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_FEATS, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {FIG_FEATS}")

# ---- t-SNE ----
print("Computing t-SNE...")
n_tsne = min(200, len(feat_real_pt), len(feat_real_ctx),
             len(feat_synth_pt), len(feat_synth_ctx))
tsne_data = np.vstack([feat_real_pt[:n_tsne],  feat_real_ctx[:n_tsne],
                       feat_synth_pt[:n_tsne], feat_synth_ctx[:n_tsne]])
tsne_labels = (['Real Point']*n_tsne + ['Real Ctx']*n_tsne +
               ['Synth Point']*n_tsne + ['Synth Ctx']*n_tsne)
tsne_norm = StandardScaler().fit_transform(tsne_data)
tsne = TSNE(n_components=2, random_state=SEED, perplexity=30,
            max_iter=1000, learning_rate='auto', init='pca')
emb = tsne.fit_transform(tsne_norm)

colors = {'Real Point':'#d73027','Real Ctx':'#1a9641',
          'Synth Point':'#fdae61','Synth Ctx':'#74add1'}
markers = {'Real Point':'o','Real Ctx':'s','Synth Point':'^','Synth Ctx':'D'}

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
fig.suptitle("t-SNE — Real vs Synthetic (WGAN-GP v3)\n"
             "Good generation: synth and real of same class intermix",
             fontsize=12, fontweight='bold')
for lbl in colors:
    mask = [l == lbl for l in tsne_labels]
    pts = emb[mask]
    ax.scatter(pts[:,0], pts[:,1], c=colors[lbl], marker=markers[lbl],
               label=lbl, alpha=0.65, s=30, edgecolors='none')
ax.legend(fontsize=10); ax.grid(True, alpha=0.2)
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
plt.tight_layout()
plt.savefig(FIG_TSNE, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {FIG_TSNE}")

# =============================================================================
# SECTION 10 — FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 78)
print("PHASE 3 v3 COMPLETE — FINAL SUMMARY")
print("=" * 78)
print()
print("Generation Quality (v2 → v3):")
print(f"  {'Metric':<22} {'v2 Pt':>10} {'v3 Pt':>10} {'v2 Ctx':>10} {'v3 Ctx':>10}")
print(f"  {'-'*64}")
print(f"  {'Frechet Distance':<22} {1.17:>10.3f} {fd_pt:>10.3f} {1.63:>10.3f} {fd_ctx:>10.3f}")
print(f"  {'MMD':<22} {0.64:>10.3f} {mmd_pt:>10.3f} {0.48:>10.3f} {mmd_ctx:>10.3f}")
print(f"  {'ACF Mean |Δ|':<22} {0.20:>10.3f} {acf_diff_pt:>10.3f} {0.24:>10.3f} {acf_diff_ctx:>10.3f}")
print()
print(f"Downstream TSTR (Train Synth → Test Real):")
print(f"  Mean F1:        {ok.tstr_f1.mean():.4f}   (Step A real-data SVM: 0.598)")
print(f"  Contextual F1:  {ctx_ok.tstr_f1.mean():.4f}   (Step A: 0.508)")
print(f"  Point F1:       {pt_ok.tstr_f1.mean():.4f}   (Step A: 0.651)")
print()

# Verdict
target_ok = (acf_diff_pt < 0.10 and acf_diff_ctx < 0.10)
tstr_ok   = (ok.tstr_f1.mean() > 0.40)

if target_ok and tstr_ok:
    print("  ✓ EXCELLENT — Temporal fidelity achieved AND downstream useful.")
    print("    Synthetic data is ready for Phase 4.")
elif target_ok or tstr_ok:
    print("  ~ GOOD — Substantial improvement over v2, partial success.")
    print("    Proceed to Phase 4 with monitoring.")
else:
    print("  ✗ Improvement insufficient — further tuning may be needed.")
    print("    However, v3 should still outperform v2 in Phase 4.")

print()
print("Files for Phase 4:")
for f in [GEN_PT, GEN_CTX, CRIT_PT, CRIT_CTX,
          SYNTH_PT_NPY, SYNTH_CTX_NPY,
          QUALITY_CSV, TSTR_CSV,
          FIG_TRAIN, FIG_SAMPLES, FIG_ACF, FIG_FEATS, FIG_TSNE]:
    print(f"  {f}")
print("=" * 78)
