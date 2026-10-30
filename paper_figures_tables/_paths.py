"""Locations read and written by the builders in paper_figures_tables/ (override with environment variables).

  FQGNN_FIGDATA       plot-ready result tables (CSV); written by the task_alpha assemblers and the
                      figure-data builders of this folder
                      default: paper_figures_tables/figdata
  FQGNN_RESULTS_ROOT  archived run outputs, laid out like task_alpha/ (runs/<run>/...,
                      baselines/<model>/runs_5seed_gaps/..., baselines/<model>/hp_*/...)
                      default: task_alpha
  FQGNN_FIG_OUT       where figures (PDF/PNG) and workbooks (XLSX) are written
                      default: paper_figures_tables/output

The result tables used for the article are provided with the article, not with this repository.
"""
from __future__ import annotations

import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)


def _env_dir(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return os.path.abspath(os.path.expanduser(value)) if value else os.path.abspath(default)


FIGDATA = _env_dir("FQGNN_FIGDATA", os.path.join(HERE, "figdata"))
RESULTS = _env_dir("FQGNN_RESULTS_ROOT", os.path.join(REPO_ROOT, "task_alpha"))
OUT = _env_dir("FQGNN_FIG_OUT", os.path.join(HERE, "output"))
os.makedirs(OUT, exist_ok=True)
