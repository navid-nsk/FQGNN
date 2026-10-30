"""Split Extended_Data_Table.xlsx into one Excel file per manuscript-numbered table.

The combined workbook holds both manuscript display items: Extended Data Table 1 (the
Literature overview sheet, first mentioned in the Introduction) and Extended Data Table 2
(every result sheet, first mentioned in Results).  For submission each numbered table
becomes its own file, numbered by first appearance in the manuscript:

  Extended_Data_Table_1.xlsx  = README + Literature overview
  Extended_Data_Table_2.xlsx  = README + all result sheets

Source_Data.xlsx (per-figure sheets) is intentionally NOT split.

Run (after build_extended_data_table.py --opm):  python paper_figures_tables/split_extended_data_tables.py
Reads and writes the workbooks in FQGNN_FIG_OUT (see _paths.py).
"""
import os
import sys

from openpyxl import load_workbook

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _paths import OUT as OUT_DIR  # noqa: E402  (env FQGNN_FIG_OUT)
SRC = os.path.join(OUT_DIR, "Extended_Data_Table.xlsx")

T1_SHEETS = {"README", "Literature overview"}
README_ROW0 = 5          # first sheet-description row in the README (see builder sheet_desc)


def _prune_readme(ws, keep_sheets, subtitle):
    ws.cell(row=2, column=1, value=subtitle)
    # walk the sheet-description rows (name in col A, bold) and delete rows for absent sheets
    r = README_ROW0
    while True:
        name = ws.cell(row=r, column=1).value
        if name is None or name not in {
            "README", "Literature overview", "Overall test results", "Bucket test results",
            "Validation results", "Training summary", "Parameters and compute", "Classical family",
            "Frequency controls", "Encoder control", "Clustered statistics by cell",
            "Native-cadence imputers", "2019 holdout", "Reference certificate",
        }:
            break
        if name not in keep_sheets:
            ws.delete_rows(r, 1)
        else:
            r += 1


def make(out_name, keep, subtitle):
    wb = load_workbook(SRC)
    for sheet in list(wb.sheetnames):
        if sheet not in keep:
            del wb[sheet]
    _prune_readme(wb["README"], keep, subtitle)
    out = os.path.join(OUT_DIR, out_name)
    wb.save(out)
    print(f"wrote {out_name}: sheets = {wb.sheetnames}")


def main():
    all_sheets = load_workbook(SRC, read_only=True).sheetnames
    t2 = ["README"] + [s for s in all_sheets if s not in T1_SHEETS]
    make("Extended_Data_Table_1.xlsx", T1_SHEETS,
         "Extended Data Table 1: the trajectory-reconstruction literature by family "
         "(first referenced in the Introduction)")
    make("Extended_Data_Table_2.xlsx", set(t2),
         "Extended Data Table 2: consolidated train/validation/test results for FQ-GNN and "
         "every baseline across all seeds, with the control, classical-family, clustered-"
         "statistics and reference-certificate sheets (first referenced in Results)")


if __name__ == "__main__":
    main()
