"""task_alpha.paths -- filesystem locations of the pipeline, configurable through environment variables.

Every location defaults to a folder inside this repository, so nothing has to be set when the
inputs are placed there.  A variable that is set (and non-empty) overrides its default.

  FQGNN_DATA_ROOT    input AIS data (licensed, not distributed with this repository)
                     default: <repo>/external_data
  FQGNN_RAW_DIR      raw position reports, partitioned by message type, year and month
                     default: <FQGNN_DATA_ROOT>/parquet
  FQGNN_STATIC_DIR   validated-MMSI lists (clean/static_clean_<YEAR>.parquet) and the 10-min
                     regular product (SAIS_<YEAR>_CLASS_A_B_interptime10m.parquet)
                     default: <FQGNN_DATA_ROOT>/static_voyage
  FQGNN_CACHE_ROOT   per-partition graph cache written by task_alpha.data_raw
                     default: <repo>/task_alpha/cache
  FQGNN_FIGDATA      plot-ready tables written by the task_alpha assemblers and read by the
                     builders in paper_figures_tables/
                     default: <repo>/paper_figures_tables/figdata

The expected file layout of the input data is documented in README.md ("Input data").
"""
from __future__ import annotations

import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def env_dir(name: str, default: str) -> str:
    """Directory from environment variable `name`, else `default` (both returned as absolute paths)."""
    value = os.environ.get(name, "").strip()
    return os.path.abspath(os.path.expanduser(value)) if value else os.path.abspath(default)


DATA_ROOT = env_dir("FQGNN_DATA_ROOT", os.path.join(REPO_ROOT, "external_data"))
RAW_DIR = env_dir("FQGNN_RAW_DIR", os.path.join(DATA_ROOT, "parquet"))
STATIC_DIR = env_dir("FQGNN_STATIC_DIR", os.path.join(DATA_ROOT, "static_voyage"))
CACHE_ROOT = env_dir("FQGNN_CACHE_ROOT", os.path.join(REPO_ROOT, "task_alpha", "cache"))
FIGDATA_DIR = env_dir("FQGNN_FIGDATA", os.path.join(REPO_ROOT, "paper_figures_tables", "figdata"))
