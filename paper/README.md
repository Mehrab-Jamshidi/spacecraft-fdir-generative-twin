# Journal article — code and data

This folder holds everything needed to reproduce the results of the journal article that
develops this thesis:

> **Detectability of learned spacecraft anomaly detectors: minimum-detectable-fault
> envelopes by controlled fault injection.**
> Mehrab Jamshidi, Andrea Colagrossi — Politecnico di Milano. *Manuscript in preparation.*

The manuscript text will be linked here once it is published. Until then this folder
contains the campaign code, the raw campaign outputs, every derived result, and a
scripted audit that recomputes each number the article quotes.

The thesis code in `../src/` is unchanged and still reproduces the thesis. Where the
article and the thesis disagree, the article is the corrected account; the differences
are listed in the top-level README under *Corrections since the thesis*.

---

## What the article does

Faults of controlled severity α and duration d are injected into nominal telemetry of the
20 held-out SMAP/MSL channels, using five fault models (step, ramp, spike train, noise
burst, and the thesis generator). A black-box detector's response is mapped over (α, d) as
a detection envelope — detection quality (F1) and the probability of detection within a
deadline, read against the false-call rate on nominal data — and inverted into a
minimum-detectable-fault contour with a channel-bootstrap interval. The envelope is then
validated against the 25 real labelled anomalies, scored by the same trained detector.

## One deterministic campaign

Every result comes from **one execution** of the campaign
(`analysis/wp9_same_instance_campaign.py`). Each channel's LSTM and VAE are trained once,
with channel-specific seeds and deterministic cuDNN kernels, and that one instance of each
is scored against all five fault models **and** against the channel's real test split.

The campaign is bit-reproducible: `results/wp9_canonical_envelope.csv` is a full
re-execution of `results/wp8_canonical_envelope.csv`, and the two files are identical.
Two earlier executions — the thesis campaign
(`../results/phase4sens_per_channel.csv`) and `results/wp3_envelope_extended.csv` — share
the injected fault streams but not the detector training, and are used only to measure how
much of the result is a property of one training draw.

---

## Reproducing

Requirements are those of the thesis (`../requirements.txt`, Python 3.12). Point the code at
the NASA SMAP/MSL dataset exactly as for the thesis (see `../data/README.md`):

```bash
export SMAP_MSL_ROOT="/path/to/archive/data/data"
```

**1. The campaign (optional; GPU, about 21 min on an RTX 3050 Ti).** Its outputs are
already committed in `results/`, so this step only re-creates them.

```bash
python paper/analysis/wp9_same_instance_campaign.py
```

**2. Every analysis, figure and the audit (about 10 min, CPU).**

```bash
bash paper/analysis/run_canonical_pipeline.sh          # PYTHON=... to choose an interpreter
```

The last line printed is the claims audit, which must report every registered claim as
verified. Figures are written to `paper/figures/` (not committed; they are regenerated).

| script | produces |
|---|---|
| `wp9_same_instance_campaign.py` | the campaign: envelope, same-instance real-fault detection, per-anomaly detection, false-call windows |
| `wp8_analyse.py` | envelope bootstrap, contours, axis decomposition, protocol comparison, detector and fault-model comparisons, replicates |
| `wp10_consolidated.py` | deadline probability of detection, deadline-constrained contours, false-call baseline, replicate spread → `wp10_numbers.json` |
| `wp2_validity.py`, `wp2_severity_metric.py` | severity calibration and validity under the distributional and predictive statistics |
| `wp2_metric_robustness.py`, `wp7_circularity.py`, `wp8_ablations.py` | AR-order sensitivity, leave-one-out, bootstrap of the ρ difference; ablations and partial correlations |
| `wp7_analyse_parametric.py`, `wp7_parametric_robustness.py` | which fault model predicts real-fault detection; calibration confound; AR-order stability |
| `wp11_uncertainty_checks.py` | paired bootstrap of the fault-model differences; admissible subset |
| `wp3_memorisation.py` | nearest-neighbour memorisation test and saturation statistics of the generator |
| `wp4_figures.py`, `fig_schematic.py`, `fig_graphical_abstract.py` | the article's figures; each is checked by `fig_qa.py` for overlapping text and text outside its box before it is saved |
| `wp12_claims_audit.py` | recomputes every number the article quotes from the files in `results/` |

Supporting modules: `_paths.py` (locates the thesis repository and the dataset),
`wp2_envelope.py`, `wp3_envelope_extended.py`, `wp7_parametric.py`,
`wp8_canonical_campaign.py` (shared campaign code).

The scripts carry long header docstrings that record why each analysis exists, including
checks that were added after earlier versions of the analysis were found wanting. They are
kept as a record of how the results were reached.

---

## Results

| file | contents |
|---|---|
| `wp9_canonical_envelope.csv` = `wp8_canonical_envelope.csv` | per channel × detector × fault model × (α, d): F1, precision and recall (point-adjust and point-wise), latency, faults presented / flagged / flagged within 5, 10, 20 steps; α = 0 rows give the nominal false-alarm rate |
| `wp9_real_detection.csv` | real-fault detection per channel and detector, same trained instance |
| `wp9_real_segments.csv` | per labelled anomaly: detected or not, latency |
| `wp9_chance_windows.csv` | probability that a nominal window of w steps holds a false alarm |
| `wp3_envelope_extended.csv` | replicate execution B (learned fault model, LSTM and VAE) |
| everything else | derived by the pipeline above |
