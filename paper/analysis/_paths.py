"""
Paths shared by every analysis script.

Works in both layouts without configuration:

  * public repository   <repo>/paper/analysis/<script>.py   ->  REPO = <repo>
  * author's workspace  <ws>/manuscript/analysis/<script>.py, with the thesis
                        repository checked out beside it at
                        <ws>/spacecraft-fdir-generative-twin

The NASA SMAP/MSL dataset is located exactly as the thesis code locates it
(src/config.py): the SMAP_MSL_ROOT environment variable, otherwise
<repo>/data/raw. labeled_anomalies.csv is taken from SMAP_MSL_LABELS if set,
otherwise two levels above SMAP_MSL_ROOT (the Kaggle archive layout), otherwise
inside it.
"""
import os
from pathlib import Path

_up = Path(__file__).resolve().parents[2]     # two levels above analysis/
REPO = (_up if (_up / "src" / "config.py").exists()
        else _up / "spacecraft-fdir-generative-twin")

DATA_ROOT = Path(os.environ.get("SMAP_MSL_ROOT", REPO / "data" / "raw"))

_candidates = ([Path(os.environ["SMAP_MSL_LABELS"])] if "SMAP_MSL_LABELS" in os.environ
               else []) + [DATA_ROOT.parent.parent / "labeled_anomalies.csv",
                           DATA_ROOT / "labeled_anomalies.csv"]
LABELS = next((p for p in _candidates if p.exists()), _candidates[0])
