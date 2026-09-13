
"""
================================================================================
PHASE 4 — SENSITIVITY ANALYSIS, using the GAN as a CONTROLLABLE fault source
              (clean channel-level held-out protocol)
================================================================================
Author:     Mehrab Jamshidi — Politecnico di Milano
Supervisor: Prof. Andrea Colagrossi
Thesis:     Generative Digital Twin Methods for Spacecraft Telemetry Anomaly
            Detection: Toward Data-Driven FDIR Design and Testing
            MSc Aeronautical Engineering, Politecnico di Milano, AY 2025-26

PURPOSE — WHY THIS IS THE CORE CONTRIBUTION
   Every other Phase 4 experiment is TSTR (train-on-synthetic, test-on-real):
   it asks whether synthetic data can REPLACE real data, and the honest answer
   there is "it substitutes, with no detectable loss, but it does not beat it."

   This script answers the other, more useful half. What makes a generative
   digital twin valuable is not substitution — it is that you can generate
   faults YOU CONTROL: any magnitude, any duration, any class, any density.
   The real NASA data cannot do this. It gives one anomaly per channel at one
   fixed severity. The GAN produces thousands of controlled fault scenarios.

THE FDIR DESIGN QUESTION THIS ANSWERS
       "How big does a fault need to be, and how long does it need to last,
        before our detector catches it — and how fast?"
   This is unanswerable from real labelled data alone, and it is exactly the
   question an FDIR designer must answer to certify a detector for flight.

   Note what makes this result immune to the contamination problem that forced
   the held-out re-run: it is not a train-on-synthetic claim at all. The
   detector is trained only on REAL NOMINAL data, and the synthetic windows are
   used solely as the injected stimulus whose parameters we vary.

PROTOCOL
   For each of the 20 HELD-OUT channels:
     1. Train an LSTM on real nominal data (same architecture as Phase 2 / 4).
     2. Set threshold = mean + 3*sigma of training errors — the parameter-free
        k3 rule that Phase 4 found to be the best DEPLOYABLE choice.
     3. Identify the longest contiguous nominal segment of the REAL test split
        using NASA's labels: data the LSTM has never seen and that the dataset
        annotations guarantee is anomaly-free.
     4. Sweep a 2D grid: severity alpha in {0.1, 0.25, 0.5, 0.75, 1.0, 1.5}
                       x duration d   in {16, 32, 64, 128, 256} timesteps.
        For each (alpha, d):
            a) Inject 3 non-overlapping synthetic anomalies into the clean
               stream at random positions:
                   injected = nominal*(1-alpha) + synthetic*alpha
               (synthetic rescaled from tanh [-1,1] back to z-score units x3).
            b) Run the LSTM, apply the threshold, and compute point-adjust F1,
               precision, recall, and DETECTION LATENCY.
            c) Repeat 5x with different random positions and synthetic windows.
            d) Average over the 5 replicas.
     5. Aggregate the per-channel (alpha, d) cells across the held-out channels.

   The alpha = 0 row is a no-injection control: it measures the false-positive
   rate of the k3 threshold on untouched nominal data, which is what makes the
   rest of the surface interpretable. The MEDIAN rate is low, 0.011; the MEAN is
   0.123, inflated by two channels (D-5 and R-1) that flag their clean stream
   almost in its entirety. Both are reported so the floor is not understated —
   see the per-channel rates written to phase4sens_per_channel.csv.

WHAT THE SURFACE SHOWS (results/phase4sens_aggregated.csv)

     mean F1        d=16     d=32     d=64    d=128    d=256
     alpha=0.10    0.255    0.336    0.480    0.613    0.694
     alpha=0.25    0.277    0.369    0.522    0.654    0.742
     alpha=0.50    0.328    0.415    0.558    0.672    0.791
     alpha=0.75    0.346    0.457    0.563    0.691    0.793
     alpha=1.00    0.375    0.494    0.591    0.708    0.802
     alpha=1.50    0.441    0.561    0.667    0.772    0.845

   1. F1 rises monotonically in BOTH parameters, from 0.255 to 0.845.
   2. DURATION DOMINATES SEVERITY. Sweeping duration at fixed alpha=0.1 moves
      F1 by 0.44 (0.255 -> 0.694); sweeping severity at fixed d=16 moves it by
      only 0.19 (0.255 -> 0.441). A long faint fault is far more detectable
      than a short loud one — the practical design lesson.
   3. Detection latency falls from ~30 timesteps in the faint/short corner to
      well under 1 timestep for large alpha, and is near-instant once alpha
      >= 0.75.

   Caveat kept in view: the per-cell standard deviation is large (0.22-0.35),
   because channels differ a lot in how detectable their dynamics are. The
   SHAPE of the surface is the robust finding, not any single cell value.

INPUTS / OUTPUTS
   Reads  : data/synthetic/gan_v4_clean_synth_{point,contextual}.npy
            the NASA SMAP/MSL dataset  (see data/README.md, SMAP_MSL_ROOT)
   Writes : results/phase4sens_per_channel.csv  per (channel, alpha, d) row
            results/phase4sens_aggregated.csv   the grid above
            figures/phase4sens_heatmap.png      F1 surface (headline figure)
            figures/phase4sens_latency.png      detection-latency surface
            figures/phase4sens_curves_by_class.png  point vs contextual

   Trains one LSTM per held-out channel, then sweeps 30 grid cells x 5 replicas
   against it. Budget roughly an hour on a single GPU.
================================================================================
"""

import numpy as np, pandas as pd, os, re, sys, time, warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG — identical seeds to every other thesis script; paths via config.py
# ============================================================================
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (  # noqa: E402
    DATASET_ROOT, LABELS_FILE, TEST_FOLDER, TRAIN_FOLDER,
    RESULTS_DIR, FIGURES_DIR, SYNTH_POINT_NPY, SYNTH_CTX_NPY,
    require_dataset,
)
from holdout_split import compute_split  # noqa: E402

require_dataset()

WINDOW_SIZE = 64; STEP_SIZE = 16; CLIP_SIGMA = 3.0
LSTM_LOOKBACK = 64; LSTM_FORECAST = 1; LSTM_HIDDEN = 64; LSTM_LAYERS = 2
LSTM_DROPOUT = 0.1; LSTM_BATCH = 64; LSTM_EPOCHS = 100; LSTM_LR = 1e-3
LSTM_PATIENCE = 20

# Sensitivity sweep
ALPHAS    = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5]
DURATIONS = [16, 32, 64, 128, 256]
N_REPLICAS = 5
N_INJ_PER_REPLICA = 3

# Minimum requirements
MIN_CLEAN_LEN = 600     # need at least 600 timesteps of clean stream to inject into

SEED = 42; torch.manual_seed(SEED); np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 78)
print("PHASE 4 SENSITIVITY ANALYSIS — using GAN controllability")
print("Question: how big / long must a fault be before the detector catches it?")
print("=" * 78)
print(f"Device: {DEVICE}   Seed: {SEED}")
print(f"Sweep: alphas={ALPHAS}  durations={DURATIONS}  replicas={N_REPLICAS}")
print("=" * 78)


# ============================================================================
# UTILITIES — copied verbatim from v3 for consistency
# ============================================================================
def parse_windows(s):
    n = [int(x) for x in re.findall(r'\d+', str(s))]
    return [[n[i], n[i+1]] for i in range(0, len(n)-1, 2)]

def make_sequences(signal, lookback=LSTM_LOOKBACK, forecast=LSTM_FORECAST):
    X, y = [], []
    for t in range(len(signal) - lookback - forecast + 1):
        X.append(signal[t:t+lookback])
        y.append(signal[t+lookback:t+lookback+forecast])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def point_adjust(yt, yp):
    ya = yp.copy(); ins = False; s = 0
    for t in range(len(yt)):
        if yt[t] == 1 and not ins: ins, s = True, t
        elif yt[t] == 0 and ins:
            ins = False
            if yp[s:t].any(): ya[s:t] = 1
    if ins and yp[s:].any(): ya[s:] = 1
    return ya

def evaluate_pa(errors, y_true, threshold, offset=LSTM_LOOKBACK):
    """Point-adjusted F1, precision, recall, AND detection latency."""
    if len(np.unique(y_true)) < 2: return 0., 0., 0., np.nan
    T = len(y_true); pl = min(len(errors), T - offset)
    y_pred = np.zeros(T, dtype=int)
    y_pred[offset:offset+pl] = (errors[:pl] > threshold).astype(int)

    # Detection latency: for each ground-truth segment, find timesteps from
    # segment start to FIRST prediction inside (or before-adjustment) it.
    latencies = []
    in_seg = False; seg_start = 0
    for t in range(T):
        if y_true[t] == 1 and not in_seg:
            in_seg, seg_start = True, t
        elif y_true[t] == 0 and in_seg:
            in_seg = False
            detected = np.where(y_pred[seg_start:t] == 1)[0]
            if len(detected) > 0: latencies.append(int(detected[0]))
    if in_seg:
        detected = np.where(y_pred[seg_start:] == 1)[0]
        if len(detected) > 0: latencies.append(int(detected[0]))

    y_adj = point_adjust(y_true, y_pred)
    p = precision_score(y_true, y_adj, zero_division=0)
    r = recall_score(y_true, y_adj, zero_division=0)
    f1 = f1_score(y_true, y_adj, zero_division=0)
    lat = float(np.mean(latencies)) if latencies else np.nan
    return p, r, f1, lat


# ============================================================================
# DATA LOADING
# ============================================================================
print("\nLoading synthetic + labels ...")
SYNTH_POINT = np.load(SYNTH_POINT_NPY).astype(np.float32)
SYNTH_CTX   = np.load(SYNTH_CTX_NPY).astype(np.float32)
print(f"  Synth point: {SYNTH_POINT.shape}   contextual: {SYNTH_CTX.shape}")
labels_df = pd.read_csv(LABELS_FILE).drop_duplicates('chan_id', keep='first')

# ---- CLEAN PROTOCOL: evaluate sensitivity on held-out channels only ----
TRAIN_CHANNELS, HOLDOUT_CHANNELS = compute_split()
print(f"Clean protocol: sensitivity on {len(HOLDOUT_CHANNELS)} held-out channels")
print(f"  Channels: {len(labels_df)}")

def load_channel(chan):
    tp = os.path.join(TEST_FOLDER, f"{chan}.npy")
    rp = os.path.join(TRAIN_FOLDER, f"{chan}.npy")
    if not (os.path.exists(tp) and os.path.exists(rp)): return None
    train_raw = np.load(rp)[:, 0]; test_raw = np.load(tp)[:, 0]
    mu, sd = train_raw.mean(), train_raw.std()
    if sd < 1e-9: sd = 1.0
    return (train_raw - mu) / sd, (test_raw - mu) / sd, len(test_raw)

def synth_pool_for_channel(ac):
    """Return synthetic windows matching the channel's class.
    These are in tanh space [-1,1]. The injection routine scales them ×3 to
    bring them to z-score units (matching the LSTM's training distribution)."""
    has_ctx = 'contextual' in str(ac); has_pt = 'point' in str(ac)
    if has_ctx and has_pt:
        return np.vstack([SYNTH_POINT, SYNTH_CTX])
    if has_ctx: return SYNTH_CTX
    return SYNTH_POINT

def longest_nominal_segment(test_norm, anom_seqs):
    """Return the longest contiguous nominal slice of test_norm (slice indices)."""
    T = len(test_norm)
    mask = np.ones(T, dtype=bool)
    for s, e in anom_seqs:
        mask[max(0, s):min(T, e)] = False
    best_a, best_b = 0, 0
    a = None
    for i in range(T):
        if mask[i]:
            if a is None: a = i
        else:
            if a is not None:
                if i - a > best_b - best_a: best_a, best_b = a, i
                a = None
    if a is not None and T - a > best_b - best_a:
        best_a, best_b = a, T
    return best_a, best_b


# ============================================================================
# LSTM (identical architecture and training loop to v3)
# ============================================================================
class LSTMPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, LSTM_HIDDEN, LSTM_LAYERS,
                            dropout=LSTM_DROPOUT, batch_first=True)
        self.fc = nn.Linear(LSTM_HIDDEN, LSTM_FORECAST)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def train_lstm(train_norm):
    Xs, ys = make_sequences(train_norm)
    if len(Xs) < 20: return None, None, None
    n_val = max(1, int(0.1 * len(Xs)))
    X_tr = torch.FloatTensor(Xs[:-n_val]); y_tr = torch.FloatTensor(ys[:-n_val])
    X_va = torch.FloatTensor(Xs[-n_val:]); y_va = torch.FloatTensor(ys[-n_val:])
    tr_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=LSTM_BATCH, shuffle=True)
    va_loader = DataLoader(TensorDataset(X_va, y_va), batch_size=LSTM_BATCH)

    model = LSTMPredictor().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LSTM_LR)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
    best_val, pc = float('inf'), 0
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
        vl /= max(1, len(va_loader)); sch.step(vl)
        if vl < best_val - 1e-6: best_val, pc = vl, 0
        else:
            pc += 1
            if pc >= LSTM_PATIENCE: break

    # threshold = k3 of training errors (the v3 deployable rule)
    model.eval()
    with torch.no_grad():
        Xb = X_tr.unsqueeze(-1).to(DEVICE)
        errs = []
        for i in range(0, len(Xb), 256):
            errs.append(((model(Xb[i:i+256]) - y_tr[i:i+256].to(DEVICE)) ** 2).mean(1).cpu().numpy())
        train_err = np.concatenate(errs)
    threshold = train_err.mean() + 3.0 * train_err.std()
    return model, threshold, train_err

def lstm_errors(model, X, y):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            Xb = X[i:i+256].unsqueeze(-1).to(DEVICE)
            yb = y[i:i+256].to(DEVICE)
            out.append(((model(Xb) - yb) ** 2).mean(1).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


# ============================================================================
# INJECTION
# ============================================================================
def pick_positions(stream_len, duration, n, rng,
                   buffer=LSTM_LOOKBACK + 16, min_gap=64):
    """Pick n non-overlapping positions for injection within a clean stream.
    Buffer at the start protects against the LSTM's lookback window starting
    on injected data; min_gap ensures injections are statistically independent."""
    valid_start = buffer
    valid_end = stream_len - duration - buffer
    if valid_end <= valid_start: return []
    positions = []
    for _ in range(n * 30):
        if len(positions) >= n: break
        c = int(rng.integers(valid_start, valid_end))
        if all(abs(c - p) >= duration + min_gap for p in positions):
            positions.append(c)
    return sorted(positions)

def inject(clean_stream, synth_pool, alpha, duration, n_inj, rng):
    """Build an injected stream by blending synthetic anomaly windows into a
    clean nominal stream at controllable severity α and duration d.

    Mechanism:  injected_segment = nominal_segment·(1-α) + synth_segment·α
    The synthetic is scaled ×3 to convert tanh [-1,1] -> z-score units so the
    severity α has the same interpretation across channels.
        α=0   -> no change (TN; not tested in this experiment)
        α=1   -> the synthetic fault at the GAN's intended amplitude
        α<1   -> attenuated fault (partial blend)
        α>1   -> amplified fault (extrapolation)
    Returns: (injected_stream, y_true_binary_vector)"""
    inj = clean_stream.copy().astype(np.float32)
    y_true = np.zeros(len(inj), dtype=int)
    synth_scaled = synth_pool * 3.0   # tanh [-1,1] -> z-score scale

    positions = pick_positions(len(inj), duration, n_inj, rng)
    for loc in positions:
        # build synthetic segment of requested duration
        if duration <= synth_scaled.shape[1]:
            idx = int(rng.integers(len(synth_scaled)))
            synth_seg = synth_scaled[idx, :duration].copy()
        else:
            n_chunks = (duration + synth_scaled.shape[1] - 1) // synth_scaled.shape[1]
            idxs = rng.integers(0, len(synth_scaled), size=n_chunks)
            synth_seg = np.concatenate([synth_scaled[i] for i in idxs])[:duration]
        nom_seg = inj[loc:loc+duration]
        inj[loc:loc+duration] = nom_seg * (1.0 - alpha) + synth_seg * alpha
        y_true[loc:loc+duration] = 1
    return inj, y_true, positions


# ============================================================================
# SENSITIVITY ANALYSIS PER CHANNEL
# ============================================================================
def run_sensitivity():
    print("\n" + "=" * 78)
    print("RUNNING SENSITIVITY SWEEP")
    print("=" * 78)
    per_chan_rows = []
    t_start = time.time()
    n_done = 0

    for idx, row in labels_df.iterrows():
        chan, sc, ac = row['chan_id'], row['spacecraft'], row['class']
        if chan not in HOLDOUT_CHANNELS:
            continue
        anom_seqs = parse_windows(row['anomaly_sequences'])
        ch = load_channel(chan)
        if ch is None: continue
        train_norm, test_norm, test_len = ch

        # Train LSTM (identical to Phase 2C / v3)
        model, threshold, train_err = train_lstm(train_norm)
        if model is None: continue

        # Get the longest contiguous nominal segment of the TEST split
        a, b = longest_nominal_segment(test_norm, anom_seqs)
        clean_stream = test_norm[a:b]
        if len(clean_stream) < MIN_CLEAN_LEN: continue

        synth_pool = synth_pool_for_channel(ac)
        rng = np.random.default_rng(SEED + idx)

        # ---- No-injection baseline (alpha = 0): clean stream, no fault ----
        # Measures the detector's false-positive rate on nominal data with no
        # injected anomaly. y_true is all-zero, so any flag is a false positive.
        Xs0, ys0 = make_sequences(clean_stream)
        if len(Xs0) > 0:
            errs0 = lstm_errors(model, torch.FloatTensor(Xs0), torch.FloatTensor(ys0))
            y_true0 = np.zeros(len(clean_stream), dtype=int)
            p0, r0, f10, _ = evaluate_pa(errs0, y_true0, threshold)
            n_flags = int((errs0 > threshold).sum())
            fp_rate = float(n_flags) / max(1, len(errs0))
            per_chan_rows.append({
                'chan_id': chan, 'spacecraft': sc, 'anomaly_class': ac,
                'alpha': 0.0, 'duration': 0,
                'f1': 0.0, 'precision': round(float(p0), 4),
                'recall': 0.0,
                'latency_steps': np.nan,
                'n_replicas_valid': 1,
                'mean_injections_per_replica': 0.0,
                'fp_rate_no_injection': round(fp_rate, 5),
            })

        for alpha in ALPHAS:
            for duration in DURATIONS:
                # buffer + duration + buffer + 2 more injection slots
                if len(clean_stream) < 2 * (LSTM_LOOKBACK + 16) + duration * N_INJ_PER_REPLICA + 64 * N_INJ_PER_REPLICA:
                    continue
                f1s, precs, recs, lats, n_pos = [], [], [], [], []
                for replica in range(N_REPLICAS):
                    inj_stream, y_true_seg, positions = inject(
                        clean_stream, synth_pool, alpha, duration,
                        N_INJ_PER_REPLICA, rng)
                    if len(positions) == 0: continue
                    Xs, ys = make_sequences(inj_stream)
                    if len(Xs) == 0: continue
                    errs = lstm_errors(model, torch.FloatTensor(Xs), torch.FloatTensor(ys))
                    p, r, f1, lat = evaluate_pa(errs, y_true_seg, threshold)
                    f1s.append(f1); precs.append(p); recs.append(r)
                    if not np.isnan(lat): lats.append(lat)
                    n_pos.append(len(positions))
                if len(f1s) == 0: continue
                per_chan_rows.append({
                    'chan_id': chan, 'spacecraft': sc, 'anomaly_class': ac,
                    'alpha': alpha, 'duration': duration,
                    'f1': round(float(np.mean(f1s)), 4),
                    'precision': round(float(np.mean(precs)), 4),
                    'recall': round(float(np.mean(recs)), 4),
                    'latency_steps': round(float(np.mean(lats)), 2) if lats else np.nan,
                    'n_replicas_valid': len(f1s),
                    'mean_injections_per_replica': round(float(np.mean(n_pos)), 2),
                })

        n_done += 1
        if n_done % 10 == 0:
            elapsed = (time.time() - t_start) / 60
            print(f"  [{n_done:3d}] processed channels (elapsed {elapsed:.1f} min)")

    df = pd.DataFrame(per_chan_rows)
    df.to_csv(RESULTS_DIR / "phase4sens_per_channel.csv", index=False)
    print(f"\n  Wrote phase4sens_per_channel.csv  ({len(df)} rows)")
    return df


def aggregate(df):
    """Aggregate per-channel rows into (alpha, duration) grids."""
    agg = df.groupby(['alpha', 'duration']).agg(
        f1_mean=('f1', 'mean'),
        f1_median=('f1', 'median'),
        f1_std=('f1', 'std'),
        precision_mean=('precision', 'mean'),
        recall_mean=('recall', 'mean'),
        latency_mean=('latency_steps', 'mean'),
        latency_median=('latency_steps', 'median'),
        n_channels=('chan_id', 'nunique'),
        fp_rate_no_injection=('fp_rate_no_injection', 'median'),
    ).reset_index().round(4)
    agg.to_csv(RESULTS_DIR / "phase4sens_aggregated.csv", index=False)
    print("  Wrote phase4sens_aggregated.csv")

    # Class-specific aggregates
    ctx_mask = df.anomaly_class.str.contains('contextual', na=False)
    df_ctx = df[ctx_mask]; df_pt = df[~ctx_mask]
    agg_ctx = df_ctx.groupby(['alpha', 'duration']).f1.mean().reset_index().rename(columns={'f1': 'f1_ctx'})
    agg_pt  = df_pt.groupby(['alpha', 'duration']).f1.mean().reset_index().rename(columns={'f1': 'f1_pt'})
    return agg, agg_ctx, agg_pt


# ============================================================================
# FIGURES
# ============================================================================
def figure_heatmap(agg, value_col='f1_mean', title_suffix='F1', filename='phase4sens_heatmap.png',
                   cmap='RdYlGn', vmin=0.0, vmax=1.0, fmt='{:.3f}'):
    piv_mean = agg.pivot(index='alpha', columns='duration', values=value_col)
    piv_med  = agg.pivot(index='alpha', columns='duration', values=value_col.replace('mean', 'median'))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"Phase 4 Sensitivity — LSTM detector {title_suffix} as a function of\n"
                 f"fault severity α and duration (timesteps), aggregated over 20 held-out channels",
                 fontsize=12, fontweight='bold')
    for ax, piv, sub in [(axes[0], piv_mean, 'Mean'), (axes[1], piv_med, 'Median')]:
        im = ax.imshow(piv.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto', origin='lower')
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if np.isnan(v): continue
                col = 'black' if (vmin + 0.3*(vmax-vmin)) < v < (vmin + 0.7*(vmax-vmin)) else 'white'
                ax.text(j, i, fmt.format(v), ha='center', va='center',
                        color=col, fontsize=10, fontweight='bold')
        ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns)
        ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels([f"{a:.2f}" for a in piv.index])
        ax.set_xlabel("Fault duration (timesteps)")
        ax.set_ylabel("Fault severity α")
        ax.set_title(f"{sub} {title_suffix}", fontsize=11)
        plt.colorbar(im, ax=ax, label=title_suffix)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close(); print(f"  Saved: {FIGURES_DIR / filename}")


def figure_latency(agg):
    piv = agg.pivot(index='alpha', columns='duration', values='latency_mean')
    fig, ax = plt.subplots(figsize=(9, 6.5))
    fig.suptitle("Phase 4 Sensitivity — LSTM detection latency vs fault parameters\n"
                 "(timesteps from fault onset to first detection, lower is better)",
                 fontsize=12, fontweight='bold')
    im = ax.imshow(piv.values, cmap='RdYlGn_r', aspect='auto', origin='lower')
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isnan(v):
                ax.text(j, i, "n/a", ha='center', va='center', color='black', fontsize=9)
            else:
                ax.text(j, i, f"{v:.1f}", ha='center', va='center',
                        color='white' if v > np.nanmean(piv.values) else 'black',
                        fontsize=10, fontweight='bold')
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels([f"{a:.2f}" for a in piv.index])
    ax.set_xlabel("Fault duration (timesteps)"); ax.set_ylabel("Fault severity α")
    plt.colorbar(im, ax=ax, label="Mean detection latency (timesteps)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase4sens_latency.png", dpi=150, bbox_inches='tight')
    plt.close(); print("  Saved: phase4sens_latency.png")


def figure_curves_by_class(df):
    """One panel per class: F1 vs alpha with separate lines per duration."""
    ctx_mask = df.anomaly_class.str.contains('contextual', na=False)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Phase 4 Sensitivity — F1 vs fault severity, by anomaly class\n"
                 "Detection improves with severity; lines = different durations",
                 fontsize=12, fontweight='bold')
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(DURATIONS)))
    for ax, (mask, title) in zip(axes, [(ctx_mask, "Contextual"), (~ctx_mask, "Point")]):
        sub = df[mask]
        for d, col in zip(DURATIONS, colors):
            sub_d = sub[sub.duration == d].groupby('alpha').f1.mean().reset_index()
            ax.plot(sub_d.alpha, sub_d.f1, marker='o', lw=2, color=col, label=f'd={d}')
        ax.set_xlabel("Fault severity α"); ax.set_ylabel("Mean F1")
        ax.set_title(title); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.0)
        ax.legend(title="Duration", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase4sens_curves_by_class.png", dpi=150, bbox_inches='tight')
    plt.close(); print("  Saved: phase4sens_curves_by_class.png")


# ============================================================================
# MAIN
# ============================================================================
def main():
    t0 = time.time()
    df = run_sensitivity()
    if len(df) == 0:
        print("\nNo rows produced — check data paths and channel availability.")
        return
    agg, agg_ctx, agg_pt = aggregate(df)
    figure_heatmap(agg, value_col='f1_mean', title_suffix='F1',
                   filename='phase4sens_heatmap.png', cmap='RdYlGn',
                   vmin=0.0, vmax=1.0, fmt='{:.3f}')
    figure_latency(agg)
    figure_curves_by_class(df)

    print("\n" + "=" * 78)
    print("PHASE 4 SENSITIVITY — FINAL SUMMARY")
    print("=" * 78)
    print("\nAggregated F1 grid (mean over channels):")
    print(agg.pivot(index='alpha', columns='duration', values='f1_mean').to_string())
    print("\nAggregated F1 grid (median over channels):")
    print(agg.pivot(index='alpha', columns='duration', values='f1_median').to_string())
    print("\nDetection latency grid (mean timesteps from onset):")
    print(agg.pivot(index='alpha', columns='duration', values='latency_mean').to_string())

    print(f"\nDONE in {(time.time()-t0)/60:.1f} min. Output files:")
    for f in [RESULTS_DIR / "phase4sens_per_channel.csv", RESULTS_DIR / "phase4sens_aggregated.csv",
              "phase4sens_heatmap.png", FIGURES_DIR / "phase4sens_latency.png",
              FIGURES_DIR / "phase4sens_curves_by_class.png"]:
        print(f"  {f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
