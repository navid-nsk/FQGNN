# Third-party baseline code

Six baselines are run from their published reference implementations. Those repositories are
**not included** in this repository and are **not** covered by its MIT license: each one is used
under its own license, unmodified, and has to be placed locally as described below before its
adapter can run. The adapter and runner files in this folder are part of this repository.

| Baseline | Upstream repository | Version used | Upstream license | Expected location |
|---|---|---|---|---|
| CSDI | <https://github.com/ermongroup/CSDI> | commit `7f24a43` | MIT | `task_alpha/baselines/CSDI/` |
| SAITS | <https://github.com/WenjieDu/SAITS> | commit `660b87f` | MIT | `task_alpha/baselines/SAITS/` |
| BF-BiLSTM | <https://github.com/vrai-group/BF-BiLSTM-imputation-trajectory> | commit `e5c8389` | no license file in the repository | `task_alpha/baselines/BF-BiLSTM/` |
| HABIT | <https://github.com/M3-Archimedes/HABIT> | default branch, retrieved July 2026 | GPL-3.0 | `task_alpha/baselines/HABIT-main/` |
| MH-GIN | <https://github.com/hyLiu1994/MH-GIN> | default branch, retrieved July 2026 | Apache-2.0 | `task_alpha/baselines/MH-GIN-main/` |
| MultitaskAIS | <https://github.com/dnguyengithub/MultitaskAIS> | default branch (`master`), retrieved July 2026 | MIT | `task_alpha/baselines/MultitaskAIS-master/` |

The adapters use the upstream code unmodified. Where an upstream behaviour has to change to run on
the two-sided gap task, the change lives in an adapter in this folder (subclass, wrapper or
separate runner) and is documented in that adapter's docstring.

## Placement

**CSDI, SAITS, BF-BiLSTM.** The scripts of this repository live *inside* the upstream folder and
import the upstream modules from the same directory (`main_model`, `diff_models`, `utils` for
CSDI; `modeling.saits` for SAITS; the pretrained `pretrained_models_ais/model_dblstm.h5` for
BF-BiLSTM). Clone the repository, check out the commit listed above, and copy its files into the
folder; no upstream file name coincides with a file of this repository. For example, for CSDI:

```
git clone https://github.com/ermongroup/CSDI.git _upstream_csdi
git -C _upstream_csdi checkout 7f24a43
cp -r _upstream_csdi/* task_alpha/baselines/CSDI/
```

Files of this repository in these folders:

| Folder | Files of this repository |
|---|---|
| `CSDI/` | `run_csdi_h2h.py` (data windows, contiguous-gap masks, `CSDI_AIS` subclass, training loop), `run_csdi_5seed_gaps.py` (5 seeds on the shared gaps), `hp_csdi_optuna.py` (hyperparameter search) |
| `SAITS/` | `build_ais_h5.py` (dataset export in the format SAITS reads), `run_saits_h2h.py`, `run_saits_5seed_gaps.py`, `hp_saits_optuna.py` |
| `BF-BiLSTM/` | `export_gaps_npz.py` (stage A, main environment), `predict_gaps_bf.py` (stage B, `bfbilstm` environment), `eval_gaps_bf.py` (stage C, main environment), `run_bfbilstm_h2h.py` |

The `.gitignore` of this repository ignores every other file in these three folders, so the
upstream code is not committed by accident.

**HABIT, MH-GIN, MultitaskAIS.** Place each repository as a sibling folder of its adapter, under
the folder name listed in the table (the names of the GitHub archive downloads), for example
`git clone https://github.com/M3-Archimedes/HABIT.git task_alpha/baselines/HABIT-main`. The
adapters add these folders to the import path themselves:

| Adapter | Imports from the upstream folder |
|---|---|
| `HABIT/habit_model.py` | `HABIT-main/habit/src` → `habit.imputer.HabitImputer` (subclassed by `HabitAdapter`) |
| `MHGIN/mhgin_model.py` | `MH-GIN-main` → `src.models.MTSHGNN.MtsHGnn` |
| `MTAIS/mtais_tf1_run.py` | `MultitaskAIS-master` → `bounds`, `nested_utils`, `distribution_utils`, `models.vrnn` |

HABIT's aggregation query uses the DuckDB `h3` community extension, which DuckDB installs on first
use.

## Packages needed by the upstream code

Besides `requirements.txt`, the upstream modules imported by the adapters need the following
packages in the main environment (versions used for the study): `linear-attention-transformer`
0.19.1 and `tqdm` 4.67.3 (CSDI), `einops` 0.8.2 (MH-GIN), `shapely` 2.1.2 (HABIT). MultitaskAIS
runs in its own environment (Python 3.7, TensorFlow 1.15, dm-sonnet 1.36, tensorflow-probability
0.8; see README.md).

## Adapted portions of upstream code

Two files of this repository contain portions adapted from upstream code. Those portions remain
under their original licenses, whose notices are reproduced here.

**`task_alpha/baselines/MTAIS/mtais_tf1_run.py`** adapts, from MultitaskAIS, the dataset pipeline
`create_AIS_dataset` (`data/datasets.py`), the model construction and training graph of
`create_dataset_and_model` / `run_train` (`runners.py`), and the evaluation graph
`create_eval_graph` (`runners.py`). The changes are listed in the module docstring (in-memory
voyage input, native cadence without the `[::2]` halving, the gap window read from the exported
test index, numeric output instead of plots).

`data/datasets.py` of MultitaskAIS carries the following notice:

```
MIT License

Copyright (c) 2018 Duong Nguyen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

`runners.py` of MultitaskAIS carries the following notice:

```
Copyright 2017 The TensorFlow Authors All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

**`task_alpha/baselines/CSDI/run_csdi_h2h.py`**: `CSDI_AIS.process_data` follows
`CSDI_Physio.process_data` of CSDI line for line, and `CSDI_AIS.forward` follows
`CSDI_base.forward`. CSDI is distributed under the MIT License (Copyright (c) 2021 Yusuke Tashiro),
with the same permission notice as reproduced above.
