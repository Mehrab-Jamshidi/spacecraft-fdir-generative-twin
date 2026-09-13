"""
================================================================================
config.py — single source of truth for every path used in this repository
================================================================================
Author:     Mehrab Jamshidi — Politecnico di Milano
Supervisor: Prof. Andrea Colagrossi
Thesis:     Generative Digital Twin Methods to Support FDIR Design and Testing

WHY THIS FILE EXISTS
--------------------
During the research the scripts carried absolute paths to the machine they were
developed on (E:\\study\\thesis\\...) and wrote every output into whatever the
current working directory happened to be. That is fine for one machine and fatal
for anybody else. This module centralises all of it:

  - INPUT  paths point at the NASA SMAP/MSL dataset, which is NOT committed
           (size + licensing). Tell the scripts where it lives with the
           SMAP_MSL_ROOT environment variable, or drop it in data/raw/.
  - OUTPUT paths are repo-relative, so results and figures always land in
           results/ and figures/ no matter where the script is invoked from.

USAGE
-----
    export SMAP_MSL_ROOT="/path/to/archive/data/data"      # bash
    $env:SMAP_MSL_ROOT = "E:\\...\\archive\\data\\data"    # PowerShell
    python src/phase4_synthetic_to_real/phase4_sensitivity.py

The expected dataset layout (see data/README.md) is:

    <SMAP_MSL_ROOT>/
    ├── train/<chan_id>.npy      # nominal only
    └── test/<chan_id>.npy       # nominal + anomalous

with labeled_anomalies.csv in the PARENT of SMAP_MSL_ROOT (that is where the
Kaggle archive puts it). Override it directly with SMAP_MSL_LABELS if your copy
is arranged differently.
================================================================================
"""

import os
from pathlib import Path

# Repository root — this file lives in src/, so one level up.
REPO_ROOT = Path(__file__).resolve().parents[1]

# ----------------------------------------------------------------------------
# INPUTS — the NASA SMAP/MSL benchmark (not committed; see data/README.md)
# ----------------------------------------------------------------------------
DATASET_ROOT = Path(os.environ.get("SMAP_MSL_ROOT", REPO_ROOT / "data" / "raw"))
TRAIN_FOLDER = DATASET_ROOT / "train"
TEST_FOLDER = DATASET_ROOT / "test"

# The Kaggle archive stores labeled_anomalies.csv one level above data/data/.
# Fall back to DATASET_ROOT itself if it is not there.
_default_labels = DATASET_ROOT.parent.parent / "labeled_anomalies.csv"
if not _default_labels.exists():
    _alt = DATASET_ROOT / "labeled_anomalies.csv"
    if _alt.exists():
        _default_labels = _alt
LABELS_FILE = Path(os.environ.get("SMAP_MSL_LABELS", _default_labels))

# ----------------------------------------------------------------------------
# OUTPUTS — always repo-relative
# ----------------------------------------------------------------------------
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"
MODELS_DIR = REPO_ROOT / "models"
DATA_DIR = REPO_ROOT / "data"
SYNTH_DIR = DATA_DIR / "synthetic"

for _d in (RESULTS_DIR, FIGURES_DIR, MODELS_DIR, SYNTH_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Named artefacts shared between phases (Phase 3 writes them, Phase 4 reads them)
# ----------------------------------------------------------------------------
SYNTH_POINT_NPY = SYNTH_DIR / "gan_v4_clean_synth_point.npy"
SYNTH_CTX_NPY = SYNTH_DIR / "gan_v4_clean_synth_contextual.npy"
SPLIT_ASSIGNMENT_CSV = DATA_DIR / "clean_split_assignment.csv"


def require_dataset() -> None:
    """Fail loudly and helpfully if the NASA dataset has not been set up."""
    missing = [p for p in (TRAIN_FOLDER, TEST_FOLDER, LABELS_FILE) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "NASA SMAP/MSL dataset not found.\n"
            + "".join(f"  missing: {p}\n" for p in missing)
            + "\nDownload it (see data/README.md) and either place it in "
            f"{DATA_DIR / 'raw'} or set SMAP_MSL_ROOT to its data/data folder."
        )
