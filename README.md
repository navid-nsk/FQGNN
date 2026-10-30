# FQGNN: Fourier–Quasicrystal operators for two-sided trajectory reconstruction

This repository contains the code of FQGNN, a graph neural network whose transforms are built from
Fourier-quasicrystal (FQ) operators, applied to two-sided reconstruction of vessel trajectories
across gaps in AIS position reports. It includes:

- the FQ mathematics and operator library (`fqmath/`, `fqops/`) and optional CUDA kernels (`fqcuda/`);
- the data pipeline, the models, the training and evaluation runners, the gap-length,
  anchor-sparsity, ablation and control experiments, and the diagnostics (`task_alpha/`);
- the shared-gaps evaluation protocol and the adapters for every baseline (`task_alpha/baselines/`);
- the builders of the figures, the Extended Data tables and the Source Data workbook
  (`paper_figures_tables/`);
- the Lean 4 + Mathlib formalization of the mathematical statements (`lean/`).

The AIS data used in the study are licensed and are not included. Result tables and trained
checkpoints are provided with the article.

## Repository layout

| Path | Contents |
|---|---|
| `fqmath/` | FQ mathematics: totally positive Grassmannian generator (`grassmann.py`), Lee–Yang secular unitary and Möbius variety (`lee_yang.py`), the spectrum Λ′ (`spectrum.py`), variety amplitudes (`akkv_amplitudes.py`), diffraction diagnostics, in-family certificate (`genericity.py`), membership checks, a variety solver and self-tests |
| `fqops/` | FQ operator library: spectral, coupled-spectral, secular, secular-pullback, variety, curve, eigenphase, velocity, zero-field, QFE, diffraction and convolution-aggregation blocks |
| `fqcuda/` | Optional CUDA extensions with autograd wrappers: Op-B eigenphases + Op-C secular determinant (`cuda_eig`), Op-D twisted-Eberlein convolution (`cuda_opd`), fused QFE + Dμ phase (`cuda_phase`); `cuda_eig/test_gates.py` checks them against the PyTorch reference |
| `task_alpha/` | Data pipeline (`data.py`, `data_raw.py`, `data_standard.py`, `paths.py`), gap datasets (`build_gap_datasets.py`, `build_anchor_datasets.py`), models (`fqdeep_faithful.py`, `fqdeep_native.py` with the Op-K/Op-H/Op-M readouts in `fq_opk.py`, `fq_oph.py`, `fq_opm.py`), training and evaluation runners (`run_*.py`), exact re-evaluation (`_clean_reeval_*.py`), assemblers of the plot-ready tables (`assemble_*.py`) and diagnostics (`_diag_*.py`, `_amd_*.py`, `_conform_*.py`, `_holdout_eval_2019.py`) |
| `task_alpha/analysis/` | Vessel-clustered bootstrap statistics |
| `task_alpha/baselines/` | Shared-gaps protocol (`common_gaps.py`), adapters and runners for RFF-GNN, Bi-GRU, CSDI, SAITS, BF-BiLSTM, HABIT, MH-GIN, MultitaskAIS, cubic Hermite, quintic Hermite, constrained GP and IMM-CTRV Kalman; `THIRD_PARTY.md` |
| `paper_figures_tables/` | Figure, Extended Data and Source Data builders, the shared figure style, and the figure-data builders; see its `README.md` |
| `lean/` | Lean 4 + Mathlib development (`README.md`, `FORMALIZATION_SCOPE.md`, `MainTheorems.lean`, `AxiomAudit.lean`) |

Run every command below from the repository root.

## Environment and installation

The study used Python 3.12 with PyTorch 2.11.0 (CUDA 12.8 build) on an NVIDIA GPU. The training
and evaluation runners place the model on CUDA.

```
conda create -n fqgnn python=3.12
conda activate fqgnn
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

`requirements.txt` pins the versions used for the study and lists only packages imported by this
repository. Two baselines run in separate environments because they need older TensorFlow releases:

| Environment | Used by | Versions used |
|---|---|---|
| `bfbilstm` | BF-BiLSTM stage B (`predict_gaps_bf.py`, `run_bfbilstm_h2h.py`) | Python 3.9, tensorflow-cpu 2.11.0, keras 2.11.0, numpy 1.26.4, scipy 1.13.1, pandas 2.3.3, scikit-learn 1.6.1, h5py 3.14.0, pyarrow 21.0.0 |
| `mtais_tf1` | MultitaskAIS stage 2 (`mtais_tf1_run.py`) | Python 3.7, tensorflow 1.15.0 (CPU), dm-sonnet 1.36, tensorflow-probability 0.8.0, numpy 1.16.4, scipy 1.5.4, matplotlib 3.4.3, h5py 3.8.0 |

Docstrings refer to the main environment as `gpubench`. The upstream baseline repositories are not
included; `task_alpha/baselines/THIRD_PARTY.md` lists them with the commits used and the folders
where the adapters expect them.

## Configuration

### Locations (environment variables)

All locations default to folders inside the repository; set a variable only to point elsewhere.

| Variable | Meaning | Default |
|---|---|---|
| `FQGNN_DATA_ROOT` | Input AIS data (not distributed) | `external_data/` |
| `FQGNN_RAW_DIR` | Raw position reports | `$FQGNN_DATA_ROOT/parquet/` |
| `FQGNN_STATIC_DIR` | Validated-MMSI lists and the 10-min regular product | `$FQGNN_DATA_ROOT/static_voyage/` |
| `FQGNN_CACHE_ROOT` | Per-partition graph cache written by `task_alpha/data_raw.py` | `task_alpha/cache/` |
| `FQGNN_FIGDATA` | Plot-ready tables (written by the assemblers and figure-data builders, read by the figure and table builders) | `paper_figures_tables/figdata/` |
| `FQGNN_RESULTS_ROOT` | Run outputs read by the figure and table builders, laid out like `task_alpha/` | `task_alpha/` |
| `FQGNN_FIG_OUT` | Figures and workbooks written by `paper_figures_tables/` | `paper_figures_tables/output/` |
| `FQGNN_BFBILSTM_PYTHON` | Python interpreter of the `bfbilstm` environment (used by `paper_figures_tables/params_audit.py`) | `python` |
| `MTAIS_CONDA` | conda executable used to reach the `mtais_tf1` environment | `$CONDA_EXE`, else `conda` on `PATH` |
| `FQCUDA_ARCH` | CUDA compute capability compiled by `fqcuda/*/setup_*.py` | `120` |

Gap datasets are written to `task_alpha/data/`, run outputs to `task_alpha/runs/` and to
`task_alpha/baselines/<model>/runs_*/`, all relative to the repository root; these folders may be
symbolic links to other disks.

### Model configuration (environment variables)

The model variant is selected by environment variables read when the model is constructed. The
configuration reported in the article (Op-M readout, referred to as the "Op-M stack" in the
docstrings) is:

```
export FQ_READOUT_MODE=opm FQ_OPM_BATCHED=1 FQ_OPM_GAMMA=hatp \
       FQ_OPM_VBAR=0.58298 FQ_OPM_MBAR=2.68022 FQ_OPM_BETA2=0.2148 FQ_OPM_LAM=c2 \
       FQ_AMP_MODE=akkv_torus FQ_AMP_NMC=4194304
```

Training runs additionally set `FQ_AMP_REFRESH=84` (amplitude table refreshed every 84 optimizer
steps); the official test numbers are recomputed by the `_clean_reeval_*.py` scripts with
`FQ_AMP_REFRESH` unset, i.e. with the exact amplitude tables at the loaded weights. The sweep,
ablation, one-sided and anchor runners select the corresponding variant with `FQ_SWEEP_SOBOLEV=1`,
`FQ_LADDER_SOBOLEV=1`, `FQ_ONESIDE_SOBOLEV=1` and `FQ_ANCHOR_SOBOLEV=1`, and the boundary-velocity
source with `FQ_SWEEP_DMU` (`blend` on the raw-AIS track, `fd` on the standardized track). Each
runner's docstring states its exact launch environment.

## Input data

The loaders read Apache Parquet files.

**Raw position reports** (`task_alpha/data_raw.py`), Hive-style partitions by message type, year and
month:

```
$FQGNN_RAW_DIR/
  m123/year=<YYYY>/month=<MM>/*.parquet       Class A position reports (message types 1, 2, 3)
  m18/year=<YYYY>/month=<MM>/*.parquet        Class B position reports (message type 18)
  m27/year=<YYYY>/month=<MM>/*.parquet        long-range position reports (message type 27)
  combined/year=<YYYY>/month=<MM>/*.parquet   years after 2019: all position reports in one partition
```

Columns read: `mmsi` (integer), `latitude` and `longitude` (decimal degrees), `sog` (knots), `cog`
(degrees; optional) and `ts` (timestamp). Rows south of 60° N (the study area; parameter
`min_lat`), with longitude outside [−180°, 180°], speed outside [0, 60) knots or MMSI 0 are
dropped, courses outside [0°, 360°] are set to missing, and duplicate (MMSI, time) rows are
removed.

**Validated vessel lists** (optional): `$FQGNN_STATIC_DIR/clean/static_clean_<YYYY>.parquet` with an
`mmsi` column. When present, positions are restricted to these vessels.

**Standardized track** (`task_alpha/data_standard.py`):
`$FQGNN_STATIC_DIR/SAIS_<YYYY>_CLASS_A_B_interptime10m.parquet`, positions on a regular 10-minute
grid with columns `mmsi` (int64), `time` (timestamp), `latitude`, `longitude`, `sog`, `cog`.

The study uses July and August of 2016–2018 for training, validation and test (vessel-disjoint
80/10/10 split by the MD5 hash of the MMSI; MMSIs outside the nine-digit range are dropped), July
and August 2019 for the temporal hold-out, and July and August 2012 for the CUDA gate test.

## Workflow

### 1. Gap datasets

```
python -m task_alpha.build_gap_datasets             # both tracks, all gap lengths, then verify
python -m task_alpha.build_anchor_datasets          # anchor-sparsity variants of the 6-h raw-AIS set
python -m task_alpha.baselines.common_gaps          # self-test of the shared gaps and the GCD reference
```

`build_gap_datasets` writes `task_alpha/data/<track>/gap_<G>h/{train,val,test}.pkl` (pickled lists
of `task_alpha.data.RunGraph`) and `manifest.json` for the raw-AIS track `our_processed` (gaps of
3, 6, 12 and 24 h) and the standardized track `standard_process` (2, 6, 12 and 24 h); an argument
`our`, `std` or `verify` restricts it to one step. Existing leaves are skipped, so rerunning resumes.
The `--smoke` options of the sweep runners read `task_alpha/data/_smoke/gap_6h/`, which holds the
first 120/40/40 graphs of the 6-h raw-AIS dataset in the same format.

### 2. FQGNN, five seeds

```
# with the model configuration above
FQ_AMP_REFRESH=84 python -m task_alpha.run_fqnative_opmstack_5seed
python -m task_alpha._clean_reeval_opmstack           # FQ_AMP_REFRESH unset
```

`run_opmstack_partA.py`, `run_opmstack_partB.py` and `run_opmstack_partC.py` split the five seeds
into three parts (seeds 0–1, 2–3 and 4) that can run in parallel, for example one per GPU with
`CUDA_VISIBLE_DEVICES`, and write to the same run folder (part aggregates `summary_pA.csv`,
`summary_pB.csv`, `summary_pC.csv`). Outputs go to
`task_alpha/runs/fqnative_opmstack_5seed/`: per seed `epochs.csv`, `per_gap.csv`,
`test_buckets.csv` and `best.pt`, the aggregates `all_seeds_*.csv` and `summary.csv`, and after the
re-evaluation `per_gap_clean.csv`, `test_buckets_clean.csv` and `summary_clean.csv`.

### 3. Shared-gaps protocol and baselines

`task_alpha/baselines/common_gaps.py` defines the evaluation shared by all models: the same test
gaps, the great-circle-distance (GCD) reference, per-gap rows, sinuosity buckets and the CSV
formats. A runner only produces predicted positions at the held nodes.

| Baseline | Command |
|---|---|
| RFF-GNN (matched random-feature control) | `python task_alpha/baselines/RFFGNN/run_rffgnn_5seed_gaps.py` |
| Bi-GRU | `python task_alpha/baselines/BiGRU/run_bigru_5seed_gaps.py` |
| SAITS | `python task_alpha/baselines/SAITS/run_saits_5seed_gaps.py [--track standard_process --gap 2]`; the fixed hyperparameters come from `hp_saits_optuna.py` |
| CSDI | `python task_alpha/baselines/CSDI/run_csdi_5seed_gaps.py [--track standard_process --gap 2]`; the fixed hyperparameters come from `hp_csdi_optuna.py` |
| BF-BiLSTM (pretrained, zero-shot) | `export_gaps_npz.py` (main environment) → `predict_gaps_bf.py` (`bfbilstm` environment) → `eval_gaps_bf.py` (main environment), all in `task_alpha/baselines/BF-BiLSTM/` |
| HABIT | `python -m task_alpha.baselines.HABIT.run_habit_gaps --all --resolution 6` |
| MH-GIN | `python -m task_alpha.baselines.MHGIN.run_mhgin_gaps --all` |
| MultitaskAIS (one-sided reference) | `python -m task_alpha.baselines.MTAIS.run_mtais_gaps --queue` |
| Cubic Hermite (Nguyen, Im & Lee, 2015) | `python -m task_alpha.baselines.Hermite.run_hermite_gaps --all` |
| Quintic Hermite | `python -m task_alpha.baselines.Quintic.run_quintic_gaps --all` |
| Constrained GP | `python -m task_alpha.baselines.ConstrainedGP.run_gp_gaps --all` |
| IMM-CTRV Kalman | `python -m task_alpha.baselines.CTRVKalman.run_ctrvkalman_gaps --tune`, then `--all` |

`task_alpha/baselines/run_remaining_baselines.py` (HABIT and MH-GIN, all cells) and
`task_alpha/baselines/run_d7_std_track.py` (SAITS and CSDI on the standardized cells) record the
exact arguments used. `test_new_baseline_adapters.py` and `MTAIS/test_mtais_adapter.py` check the
adapters for leakage of held positions.

### 4. Sweeps, ablations, controls and hold-out

| Experiment | Runner |
|---|---|
| Gap length × track | `python task_alpha/run_gap_sweep.py --queue` (or one cell with `--track/--gap/--model`), then `python -m task_alpha._clean_reeval_sweep --track <track> --gaps <list>` |
| Anchor sparsity | `python task_alpha/run_anchor_sweep.py --queue` |
| One-sided ablation | `python -m task_alpha.run_oneside_native_5seed` |
| Ablation ladder | `python -m task_alpha.run_ablation_ladder_native --queue`, then `python -m task_alpha._clean_reeval_ladder` |
| Frequency-set and encoder controls | `python -m task_alpha.run_fqnative_controls --variant {grid,gauss,uniform,shuffle,learned,mlp}` |
| In-family telemetry | `python -m task_alpha.run_telemetry_infamily_native` |
| 2019 temporal hold-out | `python -m task_alpha._holdout_eval_2019` |
| Reference certificate over the eight cells | `python -m task_alpha._diag_ref_cert_8cells` |
| Readout conditioning audit | `python -m task_alpha._diag_oph_conditioning` |

`python -m task_alpha._clean_reeval_oneanchor` re-evaluates the one-sided and anchor-sparsity
checkpoints with exact amplitude tables. With the article's configuration (`FQ_SWEEP_SOBOLEV=1` for
the gap sweep, `FQ_ANCHOR_SOBOLEV=1` for the anchor sweep), the two sweeps run the FQGNN cells only;
their RFF-GNN and Bi-GRU cells run without these variables. The sweep runners skip seeds whose outputs already exist, so an
interrupted sweep resumes when relaunched; every runner states its output folder under
`task_alpha/runs/` in its docstring.

### 5. Plot-ready tables and statistics

```
python -m task_alpha.assemble_paper_table --opm         # same-gaps comparison of all models
python -m task_alpha.assemble_gap_sweep --opm
python -m task_alpha.assemble_anchor_sweep --opm
python -m task_alpha.assemble_ablation_ladder --opm
python task_alpha/assemble_ed5_horizon_buckets.py --opm
python task_alpha/assemble_std2h_table.py
python task_alpha/analysis/clustered_stats.py
python task_alpha/analysis/clustered_stats_cells.py
```

The `--opm` flag selects the FQGNN runs of the article's configuration (and their exact
re-evaluation files); the other flags of each assembler select other model variants. The
assemblers write to `FQGNN_FIGDATA` and print the sources they read. The figure-data builders
in `paper_figures_tables/` (`dump_spectra.py`, `extract_hero_gaps.py`, `make_map_data.py`,
`params_audit.py`) add the tables derived from the trained checkpoints.

### 6. Figures and tables

```
python paper_figures_tables/fig1_build.py              # likewise fig2_build.py ... fig6_build.py
python paper_figures_tables/ED1_build.py               # likewise ED2_build.py ... ED5_build.py
python paper_figures_tables/build_extended_data_table.py --opm
python paper_figures_tables/split_extended_data_tables.py
python paper_figures_tables/build_source_data.py
```

The builders read the result tables provided with the article: point `FQGNN_FIGDATA` at the
plot-ready tables and `FQGNN_RESULTS_ROOT` at the run outputs. See `paper_figures_tables/README.md`.

## CUDA extensions

The three extensions in `fqcuda/` are standalone PyTorch C++/CUDA extensions built in place; the
wrappers `fqcuda/fq_eig_cuda.py`, `fq_opd_cuda.py` and `fq_phase_cuda.py` import them from their
folders. The training runners enable them (`use_cuda = True`); the PyTorch reference path is used
when `use_cuda` is `False`.

```
cd fqcuda/cuda_eig   && python setup_fq_eig.py   build_ext --inplace && cd ../..
cd fqcuda/cuda_opd   && python setup_fq_opd.py   build_ext --inplace && cd ../..
cd fqcuda/cuda_phase && python setup_fq_phase.py build_ext --inplace && cd ../..
python fqcuda/cuda_eig/test_gates.py
```

- The kernels were built with the CUDA 12.8 toolkit for compute capability 12.0. Set `FQCUDA_ARCH`
  (for example `89` or `90`) to compile for another GPU.
- On Windows, build from a shell initialised for the MSVC 14.44 toolset, which is compatible with
  CUDA 12.8, with `DISTUTILS_USE_SDK=1` and `TORCH_CUDA_ARCH_LIST=12.0`, for example
  `"<Visual Studio>\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.44`. The setup scripts pass
  `/std:c++17 /Zc:__cplusplus` to MSVC and `-allow-unsupported-compiler` to nvcc, and the kernels
  include `<torch/types.h>` rather than the full extension header. On other platforms the setup
  scripts pass `-O2 -std=c++17` to the host compiler.
- The extensions are compiled without `--use_fast_math` (fast-math `sin`/`cos` corrupt the fp32
  gradients) and are run in fp32 (fp64 for the gradient checks).
- `test_gates.py` compares forward values and gradients with the PyTorch reference, runs fp64
  `gradcheck`, checks the eigenphase–determinant identity, compares the full model on the July and
  August 2012 partitions, and times the kernels.

## Lean formalization

The toolchain is pinned in `lean/lean-toolchain` (`leanprover/lean4:v4.35.0-rc3`) and Mathlib in
`lean/lakefile.toml` and `lean/lake-manifest.json` (tag `v4.35.0-rc3`). With
[elan](https://github.com/leanprover/elan) installed:

```
cd lean
lake exe cache get            # prebuilt Mathlib build files
lake build                    # the FQGNN library and MainTheorems
lake env lean AxiomAudit.lean # #print axioms for every public theorem
```

`lean/README.md` describes the modules and `lean/FORMALIZATION_SCOPE.md` states what is formalized
and under which conventions.

## Results

Result tables and trained checkpoints are provided with the article.

## License

MIT (see `LICENSE`). The upstream baseline repositories are not part of this repository and keep
their own licenses; see `task_alpha/baselines/THIRD_PARTY.md`.
