# Figure and table builders

These scripts draw the figures and assemble the Extended Data tables and the Source Data workbook
of the article. They do not recompute results: they read the result tables provided with the
article. Locations are set in `_paths.py`:

| Variable | Read or written | Default |
|---|---|---|
| `FQGNN_FIGDATA` | plot-ready result tables (CSV) | `paper_figures_tables/figdata/` |
| `FQGNN_RESULTS_ROOT` | run outputs, laid out like `task_alpha/` (`runs/<run>/...`, `baselines/<model>/runs_5seed_gaps/...`, `baselines/<model>/hp_*/...`) | `task_alpha/` |
| `FQGNN_FIG_OUT` | figures (PDF, 600-dpi PNG, preview PNG) and workbooks (XLSX) | `paper_figures_tables/output/` |

Run each script with Python from any working directory, for example
`python paper_figures_tables/fig1_build.py`.

| Script | Output | Plot-ready tables read from `FQGNN_FIGDATA` |
|---|---|---|
| `fig1_build.py` | Figure 1 | `fq_spectrum_trained.csv`, `rff_spectrum.csv`, `fq_U_trained.csv`, `fq_L_trained.csv`, `hero_gap_*.csv` |
| `fig2_build.py` | Figure 2 | `map_tracks.csv`, `map_gaps_index.csv`, `hero_gap_*.csv` |
| `fig3_build.py` | Figure 3 | `combined_buckets_std2h.csv`, `pivot_ratio_std2h.csv` |
| `fig4_build.py` | Figure 4 | `ablation_ladder.csv`, `fq_spectrum_trained.csv`, `rff_spectrum.csv`, `spectrum_stats.csv`, `sk_fq.csv`, `sk_rff.csv` |
| `fig5_build.py` | Figure 5 | `hero_gaps_index.csv`, `hero_gap_*.csv` |
| `fig6_build.py` | Figure 6 | `gap_sweep.csv`, `anchor_sweep.csv`, `params_audit.csv` |
| `ED1_build.py` ... `ED5_build.py` | Extended Data Figures 1–5 | `map_tracks.csv`, `hero_gap_*.csv`, `pivot_ratio_mean.csv`, `combined_buckets_allmodels.csv`, `fq_spectrum_trained.csv`, `rff_spectrum.csv`, `spectrum_stats.csv`, `ed5_horizon_buckets.csv` |
| `build_extended_data_table.py --opm` | `Extended_Data_Table.xlsx` | `ed_table_literature.csv`, `combined_buckets_allmodels.csv`, `pivot_ratio_mean.csv`, `params_audit.csv`, `clustered_stats_cells.csv` |
| `split_extended_data_tables.py` | `Extended_Data_Table_1.xlsx`, `Extended_Data_Table_2.xlsx` (from `Extended_Data_Table.xlsx` in `FQGNN_FIG_OUT`) | none |
| `build_source_data.py` | `Source_Data.xlsx` | all of the above |

Besides these tables, the builders read per-run files (`summary.csv`, `summary_clean.csv`,
`epochs.csv`, `per_gap.csv`, `per_gap_clean.csv`, `test_buckets_clean.csv`, the telemetry CSVs and
the hyperparameter-search records of CSDI and SAITS) under `FQGNN_RESULTS_ROOT`; each script lists
its exact inputs in its header.

`fig_style.py` holds the shared style (colours, fonts, panel letters, model styles); the environment
variable `FQ_PALETTE` selects one of its palettes. `_palette_options.py` renders three figures
under every palette for comparison, and `_whitespace_audit.py` reports ink coverage and empty bands
of rendered PNGs.

## Figure-data builders

The following scripts produce plot-ready tables from trained checkpoints and the gap datasets. They
read from `task_alpha/` of this repository (`task_alpha/runs/...`, `task_alpha/data/...`), need a
CUDA GPU, and write to `FQGNN_FIGDATA`.

| Script | Tables written |
|---|---|
| `dump_spectra.py` | learned and initial FQ spectra, generator `L`, unitary `U`, RFF spectrum, spectrum statistics, structure factors of the frequency sets |
| `extract_hero_gaps.py` | the example gaps (`hero_gap_*.csv`, `hero_gaps_index.csv`), selected by a fixed rule stated in the script |
| `make_map_data.py` | study-area tracks and gap index (`map_tracks.csv`, `map_gaps_index.csv`) |
| `params_audit.py` | parameter counts and compute times of every model (`params_audit.csv`); the BF-BiLSTM and MultitaskAIS rows run in their own environments (`FQGNN_BFBILSTM_PYTHON`, `MTAIS_CONDA`) |

The remaining plot-ready tables are written by the assemblers in `task_alpha/` (see the main
`README.md`, workflow step 5).
