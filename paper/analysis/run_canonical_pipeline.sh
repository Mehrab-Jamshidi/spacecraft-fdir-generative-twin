#!/usr/bin/env bash
# Regenerates every table, figure and number of the paper from the campaign outputs
# in ../results/ (the GPU campaign itself is wp9_same_instance_campaign.py).
# Order matters: each validity script reads the previous one's output.
#
#   usage:  SMAP_MSL_ROOT=/path/to/archive/data/data bash analysis/run_canonical_pipeline.sh
#           optional: PYTHON=/path/to/python (default: python)
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
export ENVELOPE_SOURCE=canonical
export PYTHONIOENCODING=utf-8
: "${SMAP_MSL_ROOT:?set SMAP_MSL_ROOT to the NASA SMAP/MSL data folder (see data/README.md)}"
LOG=../results/pipeline_logs
mkdir -p "$LOG"

run () { echo ">>> $1"; "$PY" "$1" > "$LOG/${1%.py}.log" 2>&1; }

run wp8_analyse.py               # envelope, contours, axes, protocol, detectors, fault models
run wp10_consolidated.py         # POD surfaces, deadline contours, false calls, replicates
run wp2_validity.py              # distributional severity, same-instance observation
run wp2_severity_metric.py       # predictive severity, head to head
run wp2_metric_robustness.py     # AR order, monotone subset, bootstrap of the rho difference
run wp7_circularity.py           # channel-difficulty and envelope-shape controls
run wp8_ablations.py             # severity-only ablation, contextual LOO, duration alone, censoring
run wp7_analyse_parametric.py    # which stimulus predicts real faults
run wp7_parametric_robustness.py # calibration confound, AR-order stability
run wp11_uncertainty_checks.py   # paired bootstrap of the fault-model and severity-statistic differences
run wp3_memorisation.py          # nearest-neighbour memorisation and saturation statistics
run wp4_figures.py               # data figures
run fig_schematic.py             # Figure 1
run fig_graphical_abstract.py    # graphical abstract
echo ">>> wp12_claims_audit.py"
"$PY" wp12_claims_audit.py | tail -3
echo "done; logs in results/pipeline_logs/"
