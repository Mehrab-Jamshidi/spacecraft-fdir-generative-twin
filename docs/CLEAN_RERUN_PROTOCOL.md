# Clean Re-Run Protocol — Addressing the Train/Test Contamination

**For:** Mehrab Jamshidi
**Purpose:** Fix the methodological issue Prof. Colagrossi raised, so the Phase 3–4 results become a clean "train-on-synthetic, test-on-unseen-real" study.

---

## 1. The problem, in one paragraph

The Phase 3 WGAN-GP was trained on anomaly windows extracted from the NASA **test** split (every channel's `test_norm`). Phase 4 then evaluated synthetic-to-real detection on that **same** test split, and the data-efficiency curve added real anomaly windows that also came from the test split. So the synthetic data was shaped by the very anomalies used for evaluation, and the "11–13 real windows" figure used test anomalies as the added real data. The fix is to hold out a set of channels whose anomalies the GAN never sees, and evaluate only on those.

---

## 2. Why the split must be at the channel level

A per-channel anomaly holdout is **impossible**: 63 of the 81 channels have only **one** labelled anomaly sequence. Holding out that one sequence leaves the channel with nothing to train or test on. The only clean unit is the **channel**.

The split (deterministic, seed 42, stratified by class) is:

- **61 train channels** — their anomaly windows form the GAN training pool and the data-efficiency real-anomaly pool.
- **20 held-out channels** — anomalies never seen by the GAN or any training set; used only for final evaluation.

Held-out channels: A-8, B-1, C-1, D-14, D-4, D-5, D-9, E-10, E-8, F-3, G-1, G-2, G-3, G-4, M-2, P-1, P-2, R-1, T-2, T-5.

Held-out class balance: 13 point, 6 contextual, 1 mixed. Note **P-1** (your thesis's showcase contextual channel) is in the held-out set — so its result becomes a genuine unseen-generalisation result, which is a strength.

---

## 3. Files I have prepared

All in the folder I am giving you:

| File | What it is |
|---|---|
| `holdout_split.py` | The single source of truth for the split. Both Phase 3 and Phase 4 import it. Run it alone first to print and confirm the split. |
| `phase3_clean_patch.py` | Exact change to `wgan_gp_phase3_v3.py` (skip held-out channels, new output prefix `gan_v4_clean`). |
| `phase4_clean_patch.py` | Exact changes to `phase4_synthetic_training_v3.py` and `phase4_sensitivity.py` (evaluate on held-out channels only; data-efficiency real windows from train pool only; sensitivity additions). |

---

## 4. Execution order on your machine

1. **Copy `holdout_split.py`** into your code folder next to the Phase 3/4 scripts. Edit its `LABELS_FILE` path to your machine. Run `python holdout_split.py` — confirm it prints 61/20.
2. **Patch `wgan_gp_phase3_v3.py`** per `phase3_clean_patch.py`. Set `PREFIX = "gan_v4_clean"`. Run it. This retrains both generators on the 61 train channels (the long step — budget the GPU time). Produces `gan_v4_clean_synth_point.npy` and `gan_v4_clean_synth_contextual.npy`.
3. **Patch `phase4_synthetic_training_v3.py`** per `phase4_clean_patch.py`. Point it at the clean synth files, set `OUT_PREFIX = "phase4_clean"`. Run it. Produces clean SVM/VAE/LSTM substitution numbers and the clean data-efficiency curve, all on the 20 held-out channels.
4. **Patch `phase4_sensitivity.py`** for the held-out evaluation plus the three additions (no-injection baseline, per-cell channel count, latency-conditional note). Run it.
5. **Record every new number** in the comparison table in Section 6 below.

---

## 5. What each clean experiment now answers

- **SVM TSTR (clean):** Can a GAN that never saw these 20 channels produce synthetic faults that train a detector to catch their real faults? Headline = mean F1 over held-out channels.
- **Data-efficiency (clean):** When detecting faults on unseen channels, how much does adding *k* real labelled anomaly windows *from known channels* help on top of synthetic? Recompute the equivalent-window figure from this curve.
- **VAE / LSTM (clean):** Unchanged training (nominal only). Threshold calibrated with clean synthetic; evaluated on held-out channels. Expect k3 to remain the winner — which is the honest story anyway.
- **Sensitivity (clean):** Inject clean synthetic faults into held-out channels' nominal streams. Now includes a true zero-severity baseline and per-cell channel counts.

---

## 6. Comparison table to fill in (this goes in the thesis)

| Quantity | Original Report 4 (contaminated) | Clean held-out re-run | 
|---|---|---|
| GAN Fréchet (contextual) | 0.101 | _fill in_ |
| GAN ACF mean \|Δ\| (contextual) | 0.009 | _fill in_ |
| SVM TSTR mean F1 | 0.409 | _fill in_ |
| VAE deployable mean F1 | 0.389 | _fill in_ |
| LSTM deployable mean F1 | 0.660 | _fill in_ |
| Data-efficiency: real-window equivalent | 11–13 | _recompute_ |
| Sensitivity F1 range | 0.21 → 0.86 | _fill in_ |

State clearly in the thesis: the original numbers were computed before the held-out protocol was adopted; the clean numbers are the ones the conclusions rest on; the gap (if any) between them quantifies how much the contamination had inflated the original result.

---

## 7. Honest expectations

- The clean numbers will likely be **somewhat lower**, because evaluation is now on unseen channels and the GAN trained on less data. That is correct and expected.
- The 20-channel evaluation is noisier than 81 channels — **report median alongside mean**, and widen the data-efficiency range honestly.
- If the clean SVM TSTR holds up reasonably (say within ~0.05–0.10 of the original), the substitution claim survives in a much stronger form. If it drops sharply, the honest framing becomes "synthetic substitution works in-distribution but degrades on unseen channels" — still a valid, defensible thesis finding.

The whole point of this exercise is that **whatever the clean numbers are, they are defensible**, which is exactly what the professor asked for.
