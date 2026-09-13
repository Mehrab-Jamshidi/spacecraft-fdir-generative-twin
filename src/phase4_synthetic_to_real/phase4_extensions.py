"""
================================================================================
PHASE 4 — EXTENSIONS:  SVM AUGMENTATION GRID  (+ per-channel calibration code)
              (clean channel-level held-out protocol)
================================================================================
Author:     Mehrab Jamshidi — Politecnico di Milano
Supervisor: Prof. Andrea Colagrossi
Thesis:     Generative Digital Twin Methods to Support FDIR Design and Testing

All results here are measured on the 20 HELD-OUT channels, whose anomalies the
Phase 3 generator never saw. The 61/20 partition comes from holdout_split.py.

--------------------------------------------------------------------------------
EXPERIMENT B — SVM AUGMENTATION GRID   (this is what main() runs)
--------------------------------------------------------------------------------
Phase 4 measured two end points: an SVM trained on synthetic positives only
(0.474) against one trained on real positives (0.631). This experiment fills in
everything between them — a full grid of n_real x n_synth positives — and asks:

  - Does adding synthetic to a SMALL real set exceed the real-only ceiling?
  - How does the augmentation benefit change with the real-data budget?
  - At what point does synthetic stop helping?

IMPORTANT — what "real" means in this grid. The real positives added here are
anomaly windows drawn from the 61 TRAIN-pool channels, never from the held-out
channels being evaluated. So this grid measures CROSS-CHANNEL transfer of real
fault labels, which is a different and harder thing than the 0.631 figure above
(that one is a per-channel real-trained SVM). The two ceilings are not
interchangeable and are deliberately not compared as if they were.

THE ANSWER (results/phase4ext_clean_svm_augmentation_means.csv):

      n_real \\ n_synth      0        500      1000     2000
      0                    0.000    0.454    0.474    0.450
      5                    0.384    0.470    0.455    0.446
      10                   0.443    0.449    0.455    0.454
      20                   0.461    0.467    0.466    0.471
      all                  0.469    0.467    0.465    0.463

  Every cell with any data in it lands at roughly 0.45-0.47. Synthetic-only
  (0.474) and cross-channel-real-only (0.469) are indistinguishable, and no
  mixed cell meaningfully beats either.

  The honest reading: synthetic positives SUBSTITUTE for cross-channel real
  positives, but they do not AUGMENT them. There is no low-real/high-synth cell
  that breaks the ceiling. The one clear signal is at the bottom-left corner —
  with only 5 real positives and no synthetic, the SVM collapses to 0.384 (and a
  median of 0.132), and adding synthetic repairs it to ~0.47. Synthetic data
  rescues the tiny-real-budget regime; it does not push past the ceiling.

--------------------------------------------------------------------------------
EXPERIMENT C — PER-CHANNEL CALIBRATION (defined below, NOT run by main())
--------------------------------------------------------------------------------
Diagnosis: the synthetic-informed threshold rules in Phase 4 all failed to beat
the parameter-free k3 rule because they calibrate a GLOBAL synthetic error
distribution against a PER-CHANNEL error distribution — a scale mismatch.
Per-CLASS calibration would not fix it, because the mismatch is per-channel, not
per-class. Per-CHANNEL calibration uses each channel's own held-out nominal
windows plus synthetic anomaly windows, making the calibration set
representative of that channel's error scale.

The implementation is kept here because the diagnosis is part of the thesis
argument, but `main()` deliberately calls only the augmentation grid, so this
experiment produces no committed CSV. Call `run_per_channel_calibration()`
yourself to execute it; expect a long GPU run (a VAE and an LSTM per channel).

--------------------------------------------------------------------------------
INPUTS / OUTPUTS
--------------------------------------------------------------------------------
Reads  : data/synthetic/gan_v4_clean_synth_{point,contextual}.npy
         the NASA SMAP/MSL dataset  (see data/README.md, SMAP_MSL_ROOT)
Writes : results/phase4ext_clean_svm_augmentation.csv        (per-channel cells)
         results/phase4ext_clean_svm_augmentation_means.csv  (the grid above)
         figures/phase4ext_clean_augmentation.png
================================================================================
"""

import numpy as np, pandas as pd, os, re, sys, time, warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
warnings.filterwarnings('ignore')

# ----------- paths: shared with every other phase via src/config.py -----------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (  # noqa: E402
    DATASET_ROOT, LABELS_FILE, TEST_FOLDER, TRAIN_FOLDER,
    RESULTS_DIR, FIGURES_DIR, SYNTH_POINT_NPY, SYNTH_CTX_NPY,
    require_dataset,
)
from holdout_split import compute_split  # noqa: E402

require_dataset()

WINDOW_SIZE = 64; STEP_SIZE = 16; OVERLAP_THRESHOLD = 0.1; CLIP_SIGMA = 3.0
VAE_LATENT_DIM = 8; VAE_BATCH = 32; VAE_EPOCHS = 100; VAE_LR = 1e-3
VAE_BETA = 1.0; VAE_PATIENCE = 20
LSTM_LOOKBACK = 64; LSTM_FORECAST = 1; LSTM_HIDDEN = 64; LSTM_LAYERS = 2
LSTM_DROPOUT = 0.1; LSTM_BATCH = 64; LSTM_EPOCHS = 100; LSTM_LR = 1e-3
LSTM_PATIENCE = 20
SEED = 42; torch.manual_seed(SEED); np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Augmentation grid: rows = real, cols = synthetic
N_REAL_GRID  = [0, 5, 10, 20, 9999]    # 9999 = all available
N_SYNTH_GRID = [0, 500, 1000, 2000]

# Per-channel calibration: held-out fraction of TRAIN nominal -> negatives
CALIB_NOMINAL_FRACTION = 0.20
N_SYNTH_CALIB = 500    # synthetic windows used as positives in calibration

print("=" * 78)
print("PHASE 4 EXTENSIONS — AUGMENTATION GRID + PER-CHANNEL CALIBRATION")
print("=" * 78)
print(f"Device: {DEVICE}   Seed: {SEED}")
print("=" * 78)

# ============================================================================
# UTILITIES (subset of v3, kept identical for consistency)
# ============================================================================
def parse_windows(s):
    n = [int(x) for x in re.findall(r'\d+', str(s))]
    return [[n[i], n[i+1]] for i in range(0, len(n)-1, 2)]

def build_ground_truth_vector(num_values, anomaly_windows):
    y = np.zeros(num_values, dtype=int)
    for s, e in anomaly_windows:
        y[max(0, s):min(num_values, e)] = 1
    return y

def scale_to_tanh(w, clip_sigma=CLIP_SIGMA):
    return (np.clip(w, -clip_sigma, clip_sigma) / clip_sigma).astype(np.float32)

def make_windows_array(signal, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    return np.array([signal[i:i+window_size]
                     for i in range(0, len(signal)-window_size+1, step_size)],
                    dtype=np.float32)

def extract_features(w):
    mean = float(np.mean(w)); std = float(np.std(w))
    mn = float(np.min(w)); mx = float(np.max(w)); rng = mx - mn
    if std < 1e-9:
        ac = 0.0
    else:
        w1, w2 = w[:-1] - mean, w[1:] - mean
        d = float(np.sum(w1**2))
        ac = float(np.sum(w1*w2)/d) if d > 1e-12 else 0.0
    return np.array([mean, std, mn, mx, rng, ac], dtype=np.float32)

def batch_features(W): return np.array([extract_features(w) for w in W])

def make_sequences(signal, lookback=LSTM_LOOKBACK, forecast=LSTM_FORECAST):
    X, y = [], []
    for t in range(len(signal) - lookback - forecast + 1):
        X.append(signal[t:t+lookback])
        y.append(signal[t+lookback:t+lookback+forecast])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def point_adjust(yt, yp):
    ya = yp.copy(); ins = False; s = 0
    for t in range(len(yt)):
        if yt[t] == 1 and not ins:
            ins, s = True, t
        elif yt[t] == 0 and ins:
            ins = False
            if yp[s:t].any(): ya[s:t] = 1
    if ins and yp[s:].any(): ya[s:] = 1
    return ya

def windows_to_timestep_scores(window_errors, signal_len,
                               window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    ts = np.zeros(signal_len, dtype=np.float32)
    for w_idx, start in enumerate(range(0, signal_len-window_size+1, step_size)):
        end = start + window_size
        ts[start:end] = np.maximum(ts[start:end], window_errors[w_idx])
    return ts

def evaluate_pa_timestep(scores, yt, thr):
    if len(np.unique(yt)) < 2: return 0., 0., 0.
    yp = (scores > thr).astype(int); ya = point_adjust(yt, yp)
    return (precision_score(yt, ya, zero_division=0),
            recall_score(yt, ya, zero_division=0),
            f1_score(yt, ya, zero_division=0))

def evaluate_pa_errors(errors, yt, thr, offset=LSTM_LOOKBACK):
    if len(np.unique(yt)) < 2: return 0., 0., 0.
    T = len(yt); pl = min(len(errors), T - offset)
    yp = np.zeros(T, dtype=int)
    yp[offset:offset+pl] = (errors[:pl] > thr).astype(int)
    ya = point_adjust(yt, yp)
    return (precision_score(yt, ya, zero_division=0),
            recall_score(yt, ya, zero_division=0),
            f1_score(yt, ya, zero_division=0))

# ============================================================================
# DATA
# ============================================================================
print("\nLoading synthetic + labels ...")
SYNTH_POINT = np.load(SYNTH_POINT_NPY).astype(np.float32)
SYNTH_CTX   = np.load(SYNTH_CTX_NPY).astype(np.float32)
SYNTH_POINT_FEAT = batch_features(SYNTH_POINT)
SYNTH_CTX_FEAT   = batch_features(SYNTH_CTX)
print(f"  Synth point: {SYNTH_POINT.shape}  contextual: {SYNTH_CTX.shape}")
labels_df = pd.read_csv(LABELS_FILE).drop_duplicates('chan_id', keep='first')
print(f"  Channels: {len(labels_df)}")

# ---- CLEAN PROTOCOL: held-out split shared with Phase 3 / Phase 4 ----
TRAIN_CHANNELS, HOLDOUT_CHANNELS = compute_split()
print(f"  Clean protocol: augmentation evaluated on {len(HOLDOUT_CHANNELS)} held-out "
      f"channels; added real windows drawn only from {len(TRAIN_CHANNELS)} train channels")

def load_channel(chan):
    tp = os.path.join(TEST_FOLDER, f"{chan}.npy")
    rp = os.path.join(TRAIN_FOLDER, f"{chan}.npy")
    if not (os.path.exists(tp) and os.path.exists(rp)): return None
    train_raw = np.load(rp)[:, 0]; test_raw = np.load(tp)[:, 0]
    mu, sd = train_raw.mean(), train_raw.std()
    if sd < 1e-9: sd = 1.0
    return (train_raw - mu) / sd, (test_raw - mu) / sd, len(test_raw)

def synth_anomaly_for_channel(ac, feat=True, n=None):
    has_ctx = 'contextual' in str(ac); has_pt = 'point' in str(ac)
    src_f, src_w = ([], [])
    if has_ctx and has_pt:
        half = (n or 1000) // 2
        if feat: return np.vstack([SYNTH_POINT_FEAT[:half], SYNTH_CTX_FEAT[:half]])
        return np.vstack([SYNTH_POINT[:half], SYNTH_CTX[:half]])
    if has_ctx:
        return (SYNTH_CTX_FEAT if feat else SYNTH_CTX)[:n or 1000]
    return (SYNTH_POINT_FEAT if feat else SYNTH_POINT)[:n or 1000]

def build_real_test_windows(test_norm, test_len, anom_seqs):
    y_ts = build_ground_truth_vector(test_len, anom_seqs)
    X, y = [], []
    for start in range(0, len(test_norm)-WINDOW_SIZE+1, STEP_SIZE):
        w = scale_to_tanh(test_norm[start:start+WINDOW_SIZE])
        lb = y_ts[start:start+WINDOW_SIZE]
        X.append(extract_features(w))
        y.append(1 if lb.mean() >= OVERLAP_THRESHOLD else 0)
    return np.array(X), np.array(y)

def build_real_nominal_features(train_norm):
    feats = []
    for start in range(0, len(train_norm)-WINDOW_SIZE+1, STEP_SIZE):
        feats.append(extract_features(scale_to_tanh(train_norm[start:start+WINDOW_SIZE])))
    return np.array(feats)

def build_real_anomaly_features(test_norm, test_len, anom_seqs):
    feats = []
    for s, e in anom_seqs:
        s, e = max(0, s), min(len(test_norm), e)
        if e - s < WINDOW_SIZE:
            pad = WINDOW_SIZE - (e - s)
            s2 = max(0, s - pad//2); e2 = min(len(test_norm), e + (pad - pad//2))
            seg = test_norm[s2:e2]
            if len(seg) >= WINDOW_SIZE:
                feats.append(extract_features(scale_to_tanh(seg[:WINDOW_SIZE])))
        else:
            seg = test_norm[s:e]
            for i in range(0, len(seg)-WINDOW_SIZE+1, STEP_SIZE):
                feats.append(extract_features(scale_to_tanh(seg[i:i+WINDOW_SIZE])))
    return np.array(feats) if feats else np.empty((0, 6), dtype=np.float32)

def fit_eval_svm(Xtr, ytr, Xte, yte):
    scaler = StandardScaler().fit(Xtr)
    cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=ytr)
    svm = SVC(kernel='rbf', C=10.0, gamma='scale',
              class_weight={0: cw[0], 1: cw[1]}, random_state=SEED)
    svm.fit(scaler.transform(Xtr), ytr)
    yp = svm.predict(scaler.transform(Xte))
    return (precision_score(yte, yp, zero_division=0),
            recall_score(yte, yp, zero_division=0),
            f1_score(yte, yp, zero_division=0))

# ---- CLEAN PROTOCOL: shared TRAIN-channel real-anomaly pool (built once) ----
def build_train_anomaly_pool():
    pool = []
    for _, row in labels_df.iterrows():
        chan = row['chan_id']
        if chan not in TRAIN_CHANNELS:
            continue
        ch = load_channel(chan)
        if ch is None:
            continue
        _, test_norm, test_len = ch
        anom_seqs = parse_windows(row['anomaly_sequences'])
        feats = build_real_anomaly_features(test_norm, test_len, anom_seqs)
        if len(feats):
            pool.append(feats)
    return np.vstack(pool) if pool else np.empty((0, 6), dtype=np.float32)

TRAIN_ANOM_POOL = build_train_anomaly_pool()
print(f"  Clean protocol: TRAIN-channel real-anomaly pool = {len(TRAIN_ANOM_POOL)} windows")


# ============================================================================
# EXPERIMENT B — SVM AUGMENTATION GRID
# ============================================================================
def run_augmentation_grid():
    print("\n" + "=" * 78)
    print("EXPERIMENT B — SVM AUGMENTATION GRID  (real × synthetic)")
    print("=" * 78)
    # Cache per-channel pieces
    cache = []
    for _, row in labels_df.iterrows():
        chan, sc, ac = row['chan_id'], row['spacecraft'], row['class']
        if chan not in HOLDOUT_CHANNELS:
            continue   # clean: evaluate augmentation only on held-out channels
        anom_seqs = parse_windows(row['anomaly_sequences'])
        ch = load_channel(chan)
        if ch is None: continue
        train_norm, test_norm, test_len = ch
        Xte, yte = build_real_test_windows(test_norm, test_len, anom_seqs)
        if yte.sum() == 0 or yte.sum() == len(yte): continue
        Xnom = build_real_nominal_features(train_norm)
        Xreal = build_real_anomaly_features(test_norm, test_len, anom_seqs)
        cache.append((chan, sc, ac, Xte, yte, Xnom, Xreal))
    print(f"  Channels usable: {len(cache)}/{len(labels_df)}")

    rng = np.random.default_rng(SEED)
    per_chan_rows = []
    for (chan, sc, ac, Xte, yte, Xnom, Xreal) in cache:
        row = {'chan_id': chan, 'spacecraft': sc, 'anomaly_class': ac}
        for n_real in N_REAL_GRID:
            for n_syn in N_SYNTH_GRID:
                parts_X = [Xnom]; parts_y = [np.zeros(len(Xnom), dtype=int)]
                if n_syn > 0:
                    Xs = synth_anomaly_for_channel(ac, feat=True, n=n_syn)
                    parts_X.append(Xs); parts_y.append(np.ones(len(Xs), dtype=int))
                if n_real > 0 and len(TRAIN_ANOM_POOL) > 0:
                    take = min(n_real, len(TRAIN_ANOM_POOL))
                    sel = rng.choice(len(TRAIN_ANOM_POOL), size=take, replace=False)
                    parts_X.append(TRAIN_ANOM_POOL[sel]); parts_y.append(np.ones(take, dtype=int))
                Xtr = np.vstack(parts_X); ytr = np.concatenate(parts_y)
                if ytr.sum() == 0:
                    f1 = 0.0
                else:
                    _, _, f1 = fit_eval_svm(Xtr, ytr, Xte, yte)
                row[f"r{n_real}_s{n_syn}"] = round(f1, 4)
        per_chan_rows.append(row)
    df = pd.DataFrame(per_chan_rows)
    df.to_csv(RESULTS_DIR / "phase4ext_clean_svm_augmentation.csv", index=False)

    # Grid means / medians
    grid = []
    for n_real in N_REAL_GRID:
        for n_syn in N_SYNTH_GRID:
            col = f"r{n_real}_s{n_syn}"
            grid.append({'n_real': 'all' if n_real == 9999 else n_real,
                         'n_synth': n_syn,
                         'mean_f1': round(df[col].mean(), 4),
                         'median_f1': round(df[col].median(), 4)})
    gdf = pd.DataFrame(grid)
    gdf.to_csv(RESULTS_DIR / "phase4ext_clean_svm_augmentation_means.csv", index=False)
    print("\n  Augmentation grid — MEAN F1:")
    pivot_mean = gdf.pivot(index='n_real', columns='n_synth', values='mean_f1')
    # reorder rows
    pivot_mean = pivot_mean.reindex([0, 5, 10, 20, 'all'])
    print(pivot_mean.to_string())
    print("\n  Augmentation grid — MEDIAN F1:")
    pivot_med = gdf.pivot(index='n_real', columns='n_synth', values='median_f1')
    pivot_med = pivot_med.reindex([0, 5, 10, 20, 'all'])
    print(pivot_med.to_string())
    return df, gdf, pivot_mean, pivot_med


# ============================================================================
# EXPERIMENT C — PER-CHANNEL CALIBRATION (VAE + LSTM)
# ============================================================================
class _Enc(nn.Module):
    def __init__(s, d, l):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU())
        s.fm = nn.Linear(16, l); s.fl = nn.Linear(16, l)
    def forward(s, x):
        h = s.net(x); return s.fm(h), s.fl(h)

class _Dec(nn.Module):
    def __init__(s, l, d):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(l, 16), nn.ReLU(),
                              nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, d))
    def forward(s, z): return s.net(z)

class VAE(nn.Module):
    def __init__(s, d, l):
        super().__init__(); s.enc = _Enc(d, l); s.dec = _Dec(l, d)
    def reparam(s, m, lv):
        std = torch.exp(0.5*lv); return m + torch.randn_like(std)*std
    def forward(s, x):
        m, lv = s.enc(x); z = s.reparam(m, lv); return s.dec(z), m, lv
    def reconstruct(s, x):
        m, _ = s.enc(x); return s.dec(m)

def _vae_loss(x, xh, m, lv, beta=VAE_BETA):
    return nn.functional.mse_loss(xh, x) + beta*(-0.5*torch.mean(1+lv-m.pow(2)-lv.exp()))

def _vae_errs(model, Wt):
    model.eval(); errs = []
    with torch.no_grad():
        for i in range(0, len(Wt), 256):
            b = Wt[i:i+256].to(DEVICE); errs.append(((b - model.reconstruct(b))**2).mean(1).cpu().numpy())
    return np.concatenate(errs)

class LSTMPredictor(nn.Module):
    def __init__(s):
        super().__init__()
        s.lstm = nn.LSTM(1, LSTM_HIDDEN, LSTM_LAYERS, dropout=LSTM_DROPOUT, batch_first=True)
        s.fc = nn.Linear(LSTM_HIDDEN, LSTM_FORECAST)
    def forward(s, x):
        out, _ = s.lstm(x); return s.fc(out[:, -1, :])

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
        if vl < best_val - 1e-6: best_val, pc = vl, 0
        else:
            pc += 1
            if pc >= LSTM_PATIENCE: break
    return ep

def _lstm_errs(model, X, y):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            Xb = X[i:i+256].unsqueeze(-1).to(DEVICE); yb = y[i:i+256].to(DEVICE)
            out.append(((model(Xb) - yb)**2).mean(1).cpu().numpy())
    return np.concatenate(out) if out else np.array([])

def per_channel_calibrate(nom_errs, calib_nom_errs, synth_errs):
    """Per-channel threshold = argmax F1 on labeled mixture of
    THIS CHANNEL'S held-out nominal (label 0) + synthetic anomaly (label 1).
    Dense grid over the pooled error range. No real labels used.

    Difference vs v3 mixF1: nominals come from the SAME channel's distribution
    (held-out portion of train_err), so the calibration set is scale-matched
    to the test, addressing the per-channel scale mismatch identified in v3.
    """
    if len(calib_nom_errs) == 0 or len(synth_errs) == 0:
        mu = nom_errs.mean() if len(nom_errs) else 0.0
        sd = nom_errs.std() if len(nom_errs) else 1.0
        return mu + 3.0 * sd
    ce = np.concatenate([calib_nom_errs, synth_errs])
    cy = np.concatenate([np.zeros(len(calib_nom_errs)),
                         np.ones(len(synth_errs))]).astype(int)
    grid = np.unique(np.quantile(ce, np.linspace(0.50, 0.999, 200)))
    best_f1, best_thr = -1.0, grid[0]
    for thr in grid:
        pred = (ce > thr).astype(int)
        tp = int(((pred==1)&(cy==1)).sum()); fp = int(((pred==1)&(cy==0)).sum())
        fn = int(((pred==0)&(cy==1)).sum())
        p = tp/(tp+fp) if (tp+fp) else 0.0
        r = tp/(tp+fn) if (tp+fn) else 0.0
        f1 = 2*p*r/(p+r) if (p+r) else 0.0
        if f1 > best_f1: best_f1, best_thr = f1, thr
    return float(best_thr)


def run_per_channel_calibration():
    print("\n" + "=" * 78)
    print("EXPERIMENT C — PER-CHANNEL CALIBRATION  (VAE + LSTM)")
    print("=" * 78)
    vae_rows, lstm_rows = [], []
    for idx, row in labels_df.iterrows():
        chan, sc, ac = row['chan_id'], row['spacecraft'], row['class']
        anom_seqs = parse_windows(row['anomaly_sequences'])
        ch = load_channel(chan)
        if ch is None: continue
        train_norm, test_norm, test_len = ch
        y_true_ts = build_ground_truth_vector(test_len, anom_seqs)

        # ---------------- VAE ----------------
        Xt = make_windows_array(train_norm); Xte = make_windows_array(test_norm)
        if len(Xt) < 10: continue
        # split: train (60%) + calib (20%) + val (20%) of original train
        n = len(Xt); n_val = max(1, int(0.10 * n)); n_calib = max(1, int(CALIB_NOMINAL_FRACTION * n))
        Xtr_ = Xt[:n - n_val - n_calib]
        Xca_ = Xt[n - n_val - n_calib : n - n_val]
        Xv_  = Xt[n - n_val :]
        X_tr = torch.FloatTensor(Xtr_); X_ca = torch.FloatTensor(Xca_)
        X_va = torch.FloatTensor(Xv_);  X_te = torch.FloatTensor(Xte)

        tr_loader = DataLoader(TensorDataset(X_tr), batch_size=VAE_BATCH, shuffle=True)
        va_loader = DataLoader(TensorDataset(X_va), batch_size=VAE_BATCH)
        model = VAE(WINDOW_SIZE, VAE_LATENT_DIM).to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=VAE_LR)
        sch = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
        bv, pc = float('inf'), 0
        for _ in range(VAE_EPOCHS):
            model.train()
            for (b,) in tr_loader:
                b = b.to(DEVICE); opt.zero_grad(); xh, m, lv = model(b)
                loss = _vae_loss(b, xh, m, lv); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            model.eval(); vl = 0.0
            with torch.no_grad():
                for (b,) in va_loader:
                    b = b.to(DEVICE); xh, m, lv = model(b); vl += _vae_loss(b, xh, m, lv).item()
            vl /= len(va_loader); sch.step(vl)
            if vl < bv - 1e-5: bv, pc = vl, 0
            else:
                pc += 1
                if pc >= VAE_PATIENCE: break

        tr_err = _vae_errs(model, X_tr)
        ca_err = _vae_errs(model, X_ca)        # held-out per-channel nominal
        te_err_w = _vae_errs(model, X_te)
        ts = windows_to_timestep_scores(te_err_w, test_len)
        synth_w = synth_anomaly_for_channel(ac, feat=False, n=N_SYNTH_CALIB)
        syn_err = _vae_errs(model, torch.FloatTensor(synth_w))

        mu_e, sd_e = tr_err.mean(), tr_err.std()
        thr_k3 = mu_e + 3.0 * sd_e
        _, _, f1_k3 = evaluate_pa_timestep(ts, y_true_ts, thr_k3)

        thr_pc = per_channel_calibrate(tr_err, ca_err, syn_err)
        p_pc, r_pc, f1_pc = evaluate_pa_timestep(ts, y_true_ts, thr_pc)

        # oracle grid for sanity (same as v3)
        grid = {f"k{k}": mu_e + k * sd_e for k in [1.0,1.5,2.0,2.5,3.0,3.5,4.0]}
        for pct in [90, 95, 99]: grid[f"p{pct}"] = np.percentile(tr_err, pct)
        oracle = max(evaluate_pa_timestep(ts, y_true_ts, t)[2] for t in grid.values())

        vae_rows.append({'chan_id': chan, 'spacecraft': sc, 'anomaly_class': ac,
                         'vae_k3_f1': round(f1_k3, 4),
                         'vae_pc_f1': round(f1_pc, 4),
                         'vae_pc_precision': round(p_pc, 4),
                         'vae_pc_recall': round(r_pc, 4),
                         'vae_oracle_f1': round(oracle, 4)})

        # ---------------- LSTM ----------------
        Xtr_s, ytr_s = make_sequences(train_norm)
        Xte_s, yte_s = make_sequences(test_norm)
        if len(Xtr_s) < 20: continue
        n = len(Xtr_s); n_val = max(1, int(0.10 * n)); n_calib = max(1, int(CALIB_NOMINAL_FRACTION * n))
        X_tr = torch.FloatTensor(Xtr_s[:n - n_val - n_calib])
        y_tr = torch.FloatTensor(ytr_s[:n - n_val - n_calib])
        X_ca = torch.FloatTensor(Xtr_s[n - n_val - n_calib : n - n_val])
        y_ca = torch.FloatTensor(ytr_s[n - n_val - n_calib : n - n_val])
        X_va = torch.FloatTensor(Xtr_s[n - n_val :])
        y_va = torch.FloatTensor(ytr_s[n - n_val :])
        X_te = torch.FloatTensor(Xte_s); y_te = torch.FloatTensor(yte_s)
        tr_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=LSTM_BATCH, shuffle=True)
        va_loader = DataLoader(TensorDataset(X_va, y_va), batch_size=LSTM_BATCH)
        modelL = LSTMPredictor().to(DEVICE)
        _lstm_train(modelL, tr_loader, va_loader)

        tr_err_l = _lstm_errs(modelL, X_tr, y_tr)
        ca_err_l = _lstm_errs(modelL, X_ca, y_ca)
        te_err_l = _lstm_errs(modelL, X_te, y_te)
        synth_stream = synth_w.reshape(-1)
        sx, sy = make_sequences(synth_stream)
        syn_err_l = _lstm_errs(modelL, torch.FloatTensor(sx), torch.FloatTensor(sy)) if len(sx) else te_err_l

        muL, sdL = tr_err_l.mean(), tr_err_l.std()
        thr_k3L = muL + 3.0 * sdL
        _, _, f1_k3L = evaluate_pa_errors(te_err_l, y_true_ts, thr_k3L)
        thr_pcL = per_channel_calibrate(tr_err_l, ca_err_l, syn_err_l)
        p_pcL, r_pcL, f1_pcL = evaluate_pa_errors(te_err_l, y_true_ts, thr_pcL)

        gridL = {f"k{k}": muL + k * sdL for k in [1.0,1.5,2.0,2.5,3.0,3.5,4.0]}
        for pct in [90, 95, 99]: gridL[f"p{pct}"] = np.percentile(tr_err_l, pct)
        oracleL = max(evaluate_pa_errors(te_err_l, y_true_ts, t)[2] for t in gridL.values())

        lstm_rows.append({'chan_id': chan, 'spacecraft': sc, 'anomaly_class': ac,
                          'lstm_k3_f1': round(f1_k3L, 4),
                          'lstm_pc_f1': round(f1_pcL, 4),
                          'lstm_pc_precision': round(p_pcL, 4),
                          'lstm_pc_recall': round(r_pcL, 4),
                          'lstm_oracle_f1': round(oracleL, 4)})

        if (idx + 1) % 20 == 0:
            print(f"  [{idx+1:3d}] {chan:6s}  VAE pc={f1_pc:.3f} k3={f1_k3:.3f} | "
                  f"LSTM pc={f1_pcL:.3f} k3={f1_k3L:.3f}")

    v = pd.DataFrame(vae_rows); l = pd.DataFrame(lstm_rows)
    v.to_csv(RESULTS_DIR / "phase4ext_vae_per_channel_calib.csv", index=False)
    l.to_csv(RESULTS_DIR / "phase4ext_lstm_per_channel_calib.csv", index=False)
    print(f"\n  VAE  per-channel calib:  mean={v.vae_pc_f1.mean():.4f}  median={v.vae_pc_f1.median():.4f}")
    print(f"  VAE  k3 (within-run):    mean={v.vae_k3_f1.mean():.4f}  median={v.vae_k3_f1.median():.4f}")
    print(f"  VAE  oracle (ceiling):   mean={v.vae_oracle_f1.mean():.4f}  median={v.vae_oracle_f1.median():.4f}")
    print(f"  LSTM per-channel calib:  mean={l.lstm_pc_f1.mean():.4f}  median={l.lstm_pc_f1.median():.4f}")
    print(f"  LSTM k3 (within-run):    mean={l.lstm_k3_f1.mean():.4f}  median={l.lstm_k3_f1.median():.4f}")
    print(f"  LSTM oracle (ceiling):   mean={l.lstm_oracle_f1.mean():.4f}  median={l.lstm_oracle_f1.median():.4f}")
    return v, l


# ============================================================================
# FIGURES
# ============================================================================
def figure_augmentation_heatmap(pivot_mean, pivot_med):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Phase 4 Extension — SVM Augmentation Grid  (synthetic + real)\n"
                 "All-real ceiling = 0.598 (Phase 2A); cell colour = mean F1 across 81 channels",
                 fontsize=12, fontweight='bold')
    for ax, piv, title in [(axes[0], pivot_mean, "Mean F1"),
                            (axes[1], pivot_med, "Median F1")]:
        im = ax.imshow(piv.values, cmap='RdYlGn', vmin=0, vmax=0.7, aspect='auto')
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                ax.text(j, i, f"{v:.3f}", ha='center', va='center',
                        color='black' if 0.2 < v < 0.55 else 'white', fontsize=10, fontweight='bold')
        ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns)
        ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels([str(x) for x in piv.index])
        ax.set_xlabel("# synthetic anomaly windows")
        ax.set_ylabel("# real anomaly windows")
        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax, label='F1')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase4ext_augmentation_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close(); print("  Saved: phase4ext_augmentation_heatmap.png")


def figure_calibration_comparison(v, l, v3_vae_path="phase4_vae_results.csv",
                                  v3_lstm_path="phase4_lstm_results.csv"):
    """Side-by-side: v3 rules (mixF1, synthRecall, synthMedian, bestK, k3) vs
    the new per-channel calibration vs the oracle. Demonstrates whether
    per-channel calibration recovers the synthetic-calibration story."""
    v3v = pd.read_csv(v3_vae_path) if os.path.exists(v3_vae_path) else None
    v3l = pd.read_csv(v3_lstm_path) if os.path.exists(v3_lstm_path) else None
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Phase 4 Extension — Per-Channel Calibration vs Global Synthetic Rules vs k=3\n"
                 "Does scale-matched calibration finally beat the 3σ default?",
                 fontsize=12, fontweight='bold')

    for ax, (det, df_new, v3) in zip(axes, [("VAE", v, v3v), ("LSTM", l, v3l)]):
        names, vals = ['k=3\n(classical)'], [df_new[f"{det.lower()}_k3_f1"].mean()]
        colors = ['#999999']
        if v3 is not None:
            for rn, col in [('mixF1', f"{det.lower()}_mixF1_f1"),
                            ('synthRecall', f"{det.lower()}_synthRecall_f1"),
                            ('synthMedian', f"{det.lower()}_synthMedian_f1"),
                            ('bestK', f"{det.lower()}_bestK_f1")]:
                if col in v3.columns:
                    names.append(rn); vals.append(v3[col].mean()); colors.append('#4575b4')
        names.append('per-channel\n(NEW)'); vals.append(df_new[f"{det.lower()}_pc_f1"].mean()); colors.append('#fc8d59')
        names.append('oracle\n(ceiling)'); vals.append(df_new[f"{det.lower()}_oracle_f1"].mean()); colors.append('#1a9850')
        bars = ax.bar(names, vals, color=colors, width=0.65)
        for b, v_ in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.008,
                    f"{v_:.3f}", ha='center', fontsize=9, fontweight='bold')
        ax.axhline(vals[0], color='#999999', ls=':', lw=1)
        ax.set_ylabel("Mean F1"); ax.set_ylim(0, max(vals)*1.18)
        ax.set_title(f"{det} deployable threshold rules", fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase4ext_calibration_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(); print("  Saved: phase4ext_calibration_comparison.png")


# ============================================================================
# ORCHESTRATION
# ============================================================================
def figure_augmentation(pivot_mean, pivot_med):
    """Line-plot version of the augmentation grid (the figure used in the report):
    one line per #real windows, x-axis = #synthetic windows added; left = mean F1,
    right = median F1, over the 20 held-out channels. Shows synthetic substitutes
    for cross-channel real (plateaus together) but does not augment it."""
    labels = {0: '0 real (synthetic only)', 5: '5 real', 10: '10 real',
              20: '20 real', 'all': 'all real'}
    # the 0-real / 0-synth cell has no training data at all (F1 = 0); drop it so
    # the synthetic-only line starts where it is defined.
    for piv in (pivot_mean, pivot_med):
        if 0 in getattr(piv, 'index', []) and 0 in getattr(piv, 'columns', []):
            piv.loc[0, 0] = np.nan
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Augmentation on 20 held-out channels (clean protocol)\n"
                 "Synthetic substitutes for cross-channel real but does not augment it",
                 fontsize=12, fontweight='bold')
    for ax, piv, title in [(axes[0], pivot_mean, "MEAN F1"),
                            (axes[1], pivot_med, "MEDIAN F1")]:
        xs = list(piv.columns)
        for idx in piv.index:
            ax.plot(xs, piv.loc[idx].values, marker='o', label=labels.get(idx, str(idx)))
        if 'all' in piv.index:
            plat = float(np.nanmean(piv.loc['all'].values))
            ax.axhline(plat, color='gray', ls='--', lw=1, label=f'all-real plateau ({plat:.3f})')
        ax.set_xlabel("Synthetic windows added")
        ax.set_ylabel("F1 (20 held-out channels)")
        ax.set_title(title, fontsize=11); ax.set_ylim(0, 0.7)
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase4ext_clean_augmentation.png", dpi=150, bbox_inches='tight')
    plt.close(); print("  Saved: phase4ext_clean_augmentation.png")


def main():
    t0 = time.time()
    df_aug, gdf_aug, pmean, pmed = run_augmentation_grid()
    print(f"\nClean augmentation grid done in {(time.time()-t0)/60:.1f} min")
    print("\nMEAN F1 grid (rows = real windows from TRAIN pool, cols = synthetic):")
    print(pmean.to_string())
    print("\nMEDIAN F1 grid:")
    print(pmed.to_string())
    figure_augmentation(pmean, pmed)
    print("\nOutput files:")
    for f in [RESULTS_DIR / "phase4ext_clean_svm_augmentation.csv",
              RESULTS_DIR / "phase4ext_clean_svm_augmentation_means.csv"]:
        print(f"  {f}")


if __name__ == "__main__":
    main()
