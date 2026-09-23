"""
================================================================================
WP3 F/G/H — Extended envelope campaign:
            pipeline reproduction + second detector + point-wise protocol
================================================================================
Paper:  Detectability of learned spacecraft anomaly detectors: minimum-
        detectable-fault envelopes by controlled fault injection
Author: Mehrab Jamshidi — Politecnico di Milano

WHAT THIS RUN DELIVERS, AND WHY IT IS ONE RUN
   Three separate questions are answered by one pass, because the expensive part
   is training a detector per channel and building the injected streams, not
   scoring them.

   F  REPRODUCTION. The LSTM / point-adjust columns must reproduce the committed
      phase4sens_per_channel.csv. Both detectors are scored on the SAME injected
      streams and the VAE consumes no random numbers, so the RNG sequence is
      identical to the original run and the reproduction is exact, not
      approximate. If it does not reproduce, nothing else here is trustworthy.

   G  SECOND DETECTOR. A VAE reconstruction-error detector is characterised on
      the identical grid. The envelope is a property of the DETECTOR, and the
      paper currently characterises one. Showing that a second detector has a
      systematically different envelope under an identical campaign is what
      turns the method from a description of one model into a comparison
      instrument.

   H  POINT-WISE PROTOCOL. Every cell is scored a second time WITHOUT
      point-adjust. Section 5.4 of the paper decomposes the duration axis and
      finds that 63-83% of its effect is carried by precision, which
      point-adjust inflates by construction, against 31-46% for the severity
      axis. That yields a falsifiable prediction: under point-wise scoring the
      SEVERITY axis should survive largely intact while the DURATION axis
      compresses. This run tests it. The result is reported whichever way it
      falls.

FIDELITY TO THE ORIGINAL CAMPAIGN
   Channel loading, z-scoring, nominal-segment extraction, the injection model,
   the grid, the replica count, the RNG seeding (SEED + label-row index) and the
   LSTM architecture, training loop and k3 threshold are reproduced exactly from
   src/phase4_synthetic_to_real/phase4_sensitivity.py. The VAE follows the
   thesis Phase-2 specification (Appendix A.2).

USAGE
   set SMAP_MSL_ROOT, then
   python analysis/wp3_envelope_extended.py
================================================================================
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
from _paths import REPO  # noqa: E402
OUT = HERE.parent / "results"
OUT.mkdir(parents=True, exist_ok=True)
from _paths import DATA_ROOT, LABELS  # noqa: E402

WINDOW, STEP, CLIP = 64, 16, 3.0
LOOKBACK, HIDDEN, LAYERS, DROPOUT = 64, 64, 2, 0.1
BATCH, EPOCHS, LR, PATIENCE = 64, 100, 1e-3, 20
LATENT = 8
ALPHAS = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5]
DURATIONS = [16, 32, 64, 128, 256]
N_REPLICAS, N_INJ = 5, 3
MIN_CLEAN_LEN = 600
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- utilities
def parse_windows(s):
    n = [int(x) for x in re.findall(r"\d+", str(s))]
    return [[n[i], n[i + 1]] for i in range(0, len(n) - 1, 2)]


def make_sequences(sig, lookback=LOOKBACK):
    X, y = [], []
    for t in range(len(sig) - lookback):
        X.append(sig[t:t + lookback])
        y.append(sig[t + lookback:t + lookback + 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def point_adjust(yt, yp):
    ya = yp.copy()
    ins, s = False, 0
    for t in range(len(yt)):
        if yt[t] == 1 and not ins:
            ins, s = True, t
        elif yt[t] == 0 and ins:
            ins = False
            if yp[s:t].any():
                ya[s:t] = 1
    if ins and yp[s:].any():
        ya[s:] = 1
    return ya


def evaluate(errors, y_true, threshold, offset):
    """Return point-adjust and point-wise metrics plus latency, from one set of
    per-step predictions. Latency is identical under both protocols because it
    depends on the raw alarm train, not on the adjustment."""
    if len(np.unique(y_true)) < 2:
        return dict(f1_pa=0., p_pa=0., r_pa=0., f1_pw=0., p_pw=0., r_pw=0.,
                    lat=np.nan)
    T = len(y_true)
    pl = min(len(errors), T - offset)
    y_pred = np.zeros(T, dtype=int)
    y_pred[offset:offset + pl] = (errors[:pl] > threshold).astype(int)

    lats, in_seg, s = [], False, 0
    for t in range(T):
        if y_true[t] == 1 and not in_seg:
            in_seg, s = True, t
        elif y_true[t] == 0 and in_seg:
            in_seg = False
            det = np.where(y_pred[s:t] == 1)[0]
            if len(det):
                lats.append(int(det[0]))
    if in_seg:
        det = np.where(y_pred[s:] == 1)[0]
        if len(det):
            lats.append(int(det[0]))

    ya = point_adjust(y_true, y_pred)
    return dict(
        f1_pa=f1_score(y_true, ya, zero_division=0),
        p_pa=precision_score(y_true, ya, zero_division=0),
        r_pa=recall_score(y_true, ya, zero_division=0),
        f1_pw=f1_score(y_true, y_pred, zero_division=0),
        p_pw=precision_score(y_true, y_pred, zero_division=0),
        r_pw=recall_score(y_true, y_pred, zero_division=0),
        lat=float(np.mean(lats)) if lats else np.nan)


def load_channel(chan):
    tp, rp = DATA_ROOT / "test" / f"{chan}.npy", DATA_ROOT / "train" / f"{chan}.npy"
    if not (tp.exists() and rp.exists()):
        return None
    tr = np.load(rp)[:, 0]
    te = np.load(tp)[:, 0]
    mu, sd = tr.mean(), tr.std()
    if sd < 1e-9:
        sd = 1.0
    return (tr - mu) / sd, (te - mu) / sd


def longest_nominal_segment(test_norm, seqs):
    """Transcribed verbatim from phase4_sensitivity.py — the injected stream
    must be the same slice as the original campaign or nothing reproduces."""
    T = len(test_norm)
    mask = np.ones(T, bool)
    for s, e in seqs:
        mask[max(0, s):min(T, e)] = False
    best_a, best_b = 0, 0
    a = None
    for i in range(T):
        if mask[i]:
            if a is None:
                a = i
        else:
            if a is not None:
                if i - a > best_b - best_a:
                    best_a, best_b = a, i
                a = None
    if a is not None and T - a > best_b - best_a:
        best_a, best_b = a, T
    return best_a, best_b


def pick_positions(stream_len, duration, n, rng, buffer=LOOKBACK + 16, min_gap=64):
    lo, hi = buffer, stream_len - duration - buffer
    if hi <= lo:
        return []
    pos = []
    for _ in range(n * 30):
        if len(pos) >= n:
            break
        c = int(rng.integers(lo, hi))
        if all(abs(c - p) >= duration + min_gap for p in pos):
            pos.append(c)
    return sorted(pos)


def inject(clean, pool, alpha, duration, n_inj, rng):
    inj = clean.copy().astype(np.float32)
    y = np.zeros(len(inj), dtype=int)
    scaled = pool * 3.0
    for loc in pick_positions(len(inj), duration, n_inj, rng):
        if duration <= scaled.shape[1]:
            seg = scaled[int(rng.integers(len(scaled))), :duration].copy()
        else:
            k = (duration + scaled.shape[1] - 1) // scaled.shape[1]
            idxs = rng.integers(0, len(scaled), size=k)
            seg = np.concatenate([scaled[i] for i in idxs])[:duration]
        inj[loc:loc + duration] = inj[loc:loc + duration] * (1.0 - alpha) + seg * alpha
        y[loc:loc + duration] = 1
    return inj, y


# ---------------------------------------------------------------- detectors
class LSTMPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, HIDDEN, LAYERS, batch_first=True, dropout=DROPOUT)
        self.fc = nn.Linear(HIDDEN, 1)

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.fc(o[:, -1, :])


class VAE(nn.Module):
    """Thesis Phase-2 specification (Appendix A.2): symmetric fully connected,
    8-dimensional latent, beta = 1."""

    def __init__(self):
        super().__init__()
        self.e1, self.e2 = nn.Linear(WINDOW, 32), nn.Linear(32, 16)
        self.mu, self.lv = nn.Linear(16, LATENT), nn.Linear(16, LATENT)
        self.d1, self.d2, self.d3 = (nn.Linear(LATENT, 16), nn.Linear(16, 32),
                                     nn.Linear(32, WINDOW))

    def encode(self, x):
        h = torch.relu(self.e2(torch.relu(self.e1(x))))
        return self.mu(h), self.lv(h)

    def decode(self, z):
        return self.d3(torch.relu(self.d2(torch.relu(self.d1(z)))))

    def forward(self, x):
        mu, lv = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        return self.decode(z), mu, lv


def train_lstm(train_norm):
    Xs, ys = make_sequences(train_norm)
    if len(Xs) < 20:
        return None, None
    nv = max(1, int(0.1 * len(Xs)))
    X_tr, y_tr = torch.FloatTensor(Xs[:-nv]), torch.FloatTensor(ys[:-nv])
    X_va, y_va = torch.FloatTensor(Xs[-nv:]), torch.FloatTensor(ys[-nv:])
    trl = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH, shuffle=True)
    val = DataLoader(TensorDataset(X_va, y_va), batch_size=BATCH)

    m = LSTMPredictor().to(DEVICE)
    opt = optim.Adam(m.parameters(), lr=LR)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
    best, pc = float("inf"), 0
    for _ in range(EPOCHS):
        m.train()
        for Xb, yb in trl:
            Xb, yb = Xb.unsqueeze(-1).to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            nn.functional.mse_loss(m(Xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        m.eval()
        vl = 0.0
        with torch.no_grad():
            for Xb, yb in val:
                vl += nn.functional.mse_loss(
                    m(Xb.unsqueeze(-1).to(DEVICE)), yb.to(DEVICE)).item()
        vl /= max(1, len(val))
        sch.step(vl)
        if vl < best - 1e-6:
            best, pc = vl, 0
        else:
            pc += 1
            if pc >= PATIENCE:
                break

    m.eval()
    with torch.no_grad():
        Xb = X_tr.unsqueeze(-1).to(DEVICE)
        errs = [((m(Xb[i:i + 256]) - y_tr[i:i + 256].to(DEVICE)) ** 2).mean(1).cpu().numpy()
                for i in range(0, len(Xb), 256)]
    te = np.concatenate(errs)
    return m, te.mean() + 3.0 * te.std()


def windowise(sig):
    if len(sig) < WINDOW:
        return np.empty((0, WINDOW), np.float32), []
    starts = list(range(0, len(sig) - WINDOW + 1, STEP))
    return np.stack([sig[s:s + WINDOW] for s in starts]).astype(np.float32), starts


def train_vae(train_norm):
    W, _ = windowise(train_norm)
    if len(W) < 20:
        return None, None
    nv = max(1, int(0.1 * len(W)))
    X_tr, X_va = torch.FloatTensor(W[:-nv]), torch.FloatTensor(W[-nv:])
    trl = DataLoader(TensorDataset(X_tr), batch_size=BATCH, shuffle=True)

    m = VAE().to(DEVICE)
    opt = optim.Adam(m.parameters(), lr=LR)
    best, pc = float("inf"), 0
    for _ in range(EPOCHS):
        m.train()
        for (xb,) in trl:
            xb = xb.to(DEVICE)
            opt.zero_grad()
            xh, mu, lv = m(xb)
            loss = (nn.functional.mse_loss(xh, xb)
                    - 0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp()))
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            xv = X_va.to(DEVICE)
            xh, mu, lv = m(xv)
            vl = (nn.functional.mse_loss(xh, xv)
                  - 0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())).item()
        if vl < best - 1e-6:
            best, pc = vl, 0
        else:
            pc += 1
            if pc >= PATIENCE:
                break

    with torch.no_grad():
        xt = X_tr.to(DEVICE)
        mu, _ = m.encode(xt)
        err = ((m.decode(mu) - xt) ** 2).mean(1).cpu().numpy()
    return m, err.mean() + 3.0 * err.std()


def vae_scores(m, sig):
    """Per-time-step anomaly score: per-window reconstruction error max-pooled
    onto the time steps the window covers (thesis Phase-2 convention)."""
    W, starts = windowise(sig)
    if len(W) == 0:
        return np.zeros(len(sig))
    with torch.no_grad():
        x = torch.FloatTensor(W).to(DEVICE)
        mu, _ = m.encode(x)
        e = ((m.decode(mu) - x) ** 2).mean(1).cpu().numpy()
    out = np.zeros(len(sig))
    for s, v in zip(starts, e):
        np.maximum(out[s:s + WINDOW], v, out=out[s:s + WINDOW])
    return out


def lstm_errors(m, X, y):
    m.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            out.append((((m(X[i:i + 256].unsqueeze(-1).to(DEVICE))
                          - y[i:i + 256].to(DEVICE)) ** 2).mean(1).cpu().numpy()))
    return np.concatenate(out) if out else np.array([])


# ---------------------------------------------------------------- campaign
def main():
    print("=" * 74)
    print("WP3 F/G/H — extended envelope campaign")
    print(f"device: {DEVICE}  seed: {SEED}")
    print("=" * 74)

    labels = pd.read_csv(LABELS).drop_duplicates("chan_id", keep="first")
    split = pd.read_csv(REPO / "data" / "clean_split_assignment.csv")
    holdout = set(split[split.split == "holdout"].chan_id)
    pool_pt = np.load(REPO / "data" / "synthetic" /
                      "gan_v4_clean_synth_point.npy").astype(np.float32)
    pool_cx = np.load(REPO / "data" / "synthetic" /
                      "gan_v4_clean_synth_contextual.npy").astype(np.float32)

    rows = []
    t0 = time.time()
    done = 0

    for idx, row in labels.iterrows():
        chan, sc, ac = row["chan_id"], row["spacecraft"], row["class"]
        if chan not in holdout:
            continue
        ch = load_channel(chan)
        if ch is None:
            continue
        train_norm, test_norm = ch
        seqs = parse_windows(row["anomaly_sequences"])
        a, b = longest_nominal_segment(test_norm, seqs)
        clean = test_norm[a:b]
        if len(clean) < MIN_CLEAN_LEN:
            print(f"  {chan}: clean stream too short ({len(clean)}), skipped")
            continue

        lstm, thr_l = train_lstm(train_norm)
        if lstm is None:
            continue
        vae, thr_v = train_vae(train_norm)

        has_cx, has_pt = "contextual" in str(ac), "point" in str(ac)
        pool = (np.vstack([pool_pt, pool_cx]) if (has_cx and has_pt)
                else pool_cx if has_cx else pool_pt)

        # RNG seeded exactly as the original campaign, and the VAE consumes no
        # random numbers, so the injected streams are bit-identical to that run.
        rng = np.random.default_rng(SEED + idx)

        # ---- alpha = 0 no-injection control ----
        Xs0, ys0 = make_sequences(clean)
        if len(Xs0):
            e0 = lstm_errors(lstm, torch.FloatTensor(Xs0), torch.FloatTensor(ys0))
            fp_l = float((e0 > thr_l).sum()) / max(1, len(e0))
            v0 = vae_scores(vae, clean) if vae is not None else np.zeros(len(clean))
            fp_v = float((v0 > thr_v).sum()) / max(1, len(v0)) if vae is not None else np.nan
            for det, fp in (("lstm", fp_l), ("vae", fp_v)):
                rows.append(dict(chan_id=chan, spacecraft=sc, anomaly_class=ac,
                                 detector=det, alpha=0.0, duration=0,
                                 f1_pa=0.0, f1_pw=0.0, p_pa=0.0, p_pw=0.0,
                                 r_pa=0.0, r_pw=0.0, latency=np.nan,
                                 n_replicas=1, fp_rate_no_injection=round(fp, 5)))

        for alpha in ALPHAS:
            for duration in DURATIONS:
                if len(clean) < 2 * (LOOKBACK + 16) + duration * N_INJ + 64 * N_INJ:
                    continue
                acc = {d: [] for d in ("lstm", "vae")}
                for _ in range(N_REPLICAS):
                    inj_stream, y_true = inject(clean, pool, alpha, duration, N_INJ, rng)
                    if y_true.sum() == 0:
                        continue
                    Xs, ys = make_sequences(inj_stream)
                    if len(Xs) == 0:
                        continue
                    el = lstm_errors(lstm, torch.FloatTensor(Xs), torch.FloatTensor(ys))
                    acc["lstm"].append(evaluate(el, y_true, thr_l, LOOKBACK))
                    if vae is not None:
                        ev = vae_scores(vae, inj_stream)
                        acc["vae"].append(evaluate(ev, y_true, thr_v, 0))
                for det, lst in acc.items():
                    if not lst:
                        continue
                    lats = [r["lat"] for r in lst if not np.isnan(r["lat"])]
                    rows.append(dict(
                        chan_id=chan, spacecraft=sc, anomaly_class=ac, detector=det,
                        alpha=alpha, duration=duration,
                        f1_pa=round(float(np.mean([r["f1_pa"] for r in lst])), 4),
                        f1_pw=round(float(np.mean([r["f1_pw"] for r in lst])), 4),
                        p_pa=round(float(np.mean([r["p_pa"] for r in lst])), 4),
                        p_pw=round(float(np.mean([r["p_pw"] for r in lst])), 4),
                        r_pa=round(float(np.mean([r["r_pa"] for r in lst])), 4),
                        r_pw=round(float(np.mean([r["r_pw"] for r in lst])), 4),
                        latency=round(float(np.mean(lats)), 2) if lats else np.nan,
                        n_replicas=len(lst), fp_rate_no_injection=np.nan))

        done += 1
        print(f"  [{done:2d}/20] {chan:<5} done  ({(time.time()-t0)/60:.1f} min elapsed)")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "wp3_envelope_extended.csv", index=False)
    print(f"\nwrote {len(df)} rows -> {OUT / 'wp3_envelope_extended.csv'}")
    print(f"total time {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
