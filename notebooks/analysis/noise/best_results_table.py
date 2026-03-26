"""
best_results_table.py
=====================
Tìm kết quả tốt nhất của LOC-NFST cho từng dataset:
- Lọc noise = 0
- Chọn row có AUCROC cao nhất trên toàn bộ scaler + nCluster
- Xuất bảng tổng hợp theo format: Dataset | AUC-ROC | AUC-PR | MCC | ACC | F1 | Precision | Recall
- Lưu thành CSV và in LaTeX ra terminal

Usage:
    python best_results_table.py                  # tự tìm trong subfolder 'exp/'
    python best_results_table.py <path_to_dir>    # chỉ định thủ công
"""

import os
import sys
import glob
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
_script_dir  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR   = os.path.join(_script_dir, 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

KNOWN_DATASETS = ['BoTIoT', 'ToNIoT', 'N_BaIoT', 'CICIoT']

DISPLAY_NAMES = {
    'BoTIoT':    'BoTIoT',
    'CICIoT':    'CICIoT2023',
    'N_BaIoT':   'N-BaIoT',
    'ToNIoT':    'ToN-IoT',
}

METRIC_COLS = ['aucroc', 'aucpr', 'mcc', 'accuracy', 'f1 score', 'precision', 'recall']
DISPLAY_HEADERS = ['AUC-ROC (%)', 'AUC-PR (%)', 'MCC', 'ACC (%)', 'F1', 'Precision', 'Recall']

# Metrics that are stored as 0-100 (percentage) vs 0-1 scale
PERCENT_COLS = ['aucroc', 'aucpr', 'accuracy']

# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
def normalize_dataset_name(name):
    if not isinstance(name, str): return str(name)
    name_lower = name.lower()
    for ds in KNOWN_DATASETS:
        if ds.lower() in name_lower: return ds
    base = os.path.basename(name)
    if base.endswith('.csv'): base = base[:-4]
    return base

def find_result_csvs(root_dir):
    """Find CSVs that have aucroc column (result files)."""
    files = glob.glob(os.path.join(root_dir, '**', '*.csv'), recursive=True)
    valid = []
    for f in files:
        try:
            if os.path.getsize(f) < 100: continue
            df_tmp = pd.read_csv(f, nrows=2)
            cols = [c.lower() for c in df_tmp.columns]
            if 'aucroc' in cols and ('dataset' in cols or 'ncluster' in cols):
                valid.append(f)
        except Exception:
            pass
    return valid

def load_all(search_dir):
    files = find_result_csvs(search_dir)
    if not files:
        raise FileNotFoundError(f"No result CSVs found under: {search_dir}")
    print(f"Found {len(files)} result file(s):")
    for f in files: print(f"  {f}")

    frames = []
    for f in files:
        df = pd.read_csv(f)
        df.columns = [c.lower().replace('_', ' ').strip() for c in df.columns]

        # Unify column names
        rename = {}
        for c in df.columns:
            if 'noise percentage' in c or c == 'noise percentage': rename[c] = 'noise'
            if c == 'scaled': rename[c] = 'scaler'
            if 'ncluster' in c and 'requested' not in c and c != 'ncluster': rename[c] = 'ncluster'
            if 'f1 score' not in df.columns and 'f1' in c: rename[c] = 'f1 score'
        df.rename(columns=rename, inplace=True)

        if 'dataset' in df.columns:
            df['dataset'] = df['dataset'].apply(normalize_dataset_name)

        # If this is a LOC-NFST file (no 'model' column), add one
        if 'model' not in df.columns:
            df['model'] = 'LOC-NFST'

        df['aucroc'] = pd.to_numeric(df.get('aucroc', np.nan), errors='coerce')
        df['noise']  = pd.to_numeric(df.get('noise', 0), errors='coerce').fillna(0)

        for col in METRIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        frames.append(df)

    return pd.concat(frames, ignore_index=True)

# ---------------------------------------------------------------------------
# ANALYSIS
# ---------------------------------------------------------------------------
def get_best_per_dataset(df, model_filter='LOC-NFST', noise_filter=0.0):
    """
    For each dataset:
    1. Filter by model and noise level
    2. Find the single row with highest AUCROC
    3. Return its full metric row
    """
    subset = df.copy()
    if model_filter:
        subset = subset[subset.get('model', pd.Series(['LOC-NFST']*len(subset))).str.contains(model_filter, case=False, na=False)]
    subset = subset[subset['noise'] == noise_filter]
    subset = subset.dropna(subset=['aucroc'])

    rows = []
    for ds in KNOWN_DATASETS:
        ds_rows = subset[subset['dataset'] == ds]
        if ds_rows.empty:
            print(f"  [WARNING] No data found for dataset: {ds}")
            continue

        best_row = ds_rows.loc[ds_rows['aucroc'].idxmax()]
        entry = {'Dataset': DISPLAY_NAMES.get(ds, ds)}
        for col, hdr in zip(METRIC_COLS, DISPLAY_HEADERS):
            val = best_row.get(col, np.nan)
            if pd.isna(val):
                entry[hdr] = 'N/A'
            elif col in PERCENT_COLS:
                # Already in %, round to 2 decimal
                entry[hdr] = round(float(val), 2)
            else:
                entry[hdr] = round(float(val), 4)
        rows.append(entry)

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# OUTPUT FORMATTERS
# ---------------------------------------------------------------------------
def print_pretty_table(result_df):
    print("\n" + "=" * 90)
    print("  BEST LOC-NFST RESULTS PER DATASET (noise=0)")
    print("=" * 90)
    print(result_df.to_string(index=False))
    print("=" * 90)

def to_latex(result_df):
    header_map = {
        'AUC-ROC (%)': '\\textbf{AUC-ROC} (\\%)',
        'AUC-PR (%)':  '\\textbf{AUC-PR} (\\%)',
        'MCC':         '\\textbf{MCC}',
        'ACC (%)':     '\\textbf{ACC} (\\%)',
        'F1':          '\\textbf{F1}',
        'Precision':   '\\textbf{Precision}',
        'Recall':      '\\textbf{Recall}',
    }

    n_cols = len(result_df.columns)
    col_fmt = 'l' + 'c' * (n_cols - 1)

    lines = [
        "\\begin{table}[!ht]",
        "\\centering",
        "\\caption{Best LOC-NFST performance per dataset (noise = 0\\%).}",
        f"\\begin{{tabular}}{{{col_fmt}}}",
        "\\hline",
    ]

    headers = ['\\textbf{Dataset}'] + [header_map.get(h, f'\\textbf{{{h}}}') for h in result_df.columns[1:]]
    lines.append(" & ".join(headers) + " \\\\")
    lines.append("\\hline")

    for _, row in result_df.iterrows():
        cells = [str(row['Dataset'])] + [str(row[h]) for h in DISPLAY_HEADERS]
        lines.append(" & ".join(cells) + " \\\\")

    lines += ["\\hline", "\\end{tabular}", "\\label{tab:best_results}", "\\end{table}"]
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    if len(sys.argv) > 1:
        search_dir = sys.argv[1]
    else:
        default = os.path.join(_script_dir, 'exp')
        if os.path.exists(default):
            search_dir = default
        else:
            search_dir = input("Enter path to experiment output folder:\n> ").strip()

    print(f"\nSearching in: {search_dir}")
    df = load_all(search_dir)

    print(f"\nTotal rows loaded: {len(df):,}")
    print(f"Datasets found   : {sorted(df['dataset'].unique())}")
    print(f"Noise levels     : {sorted(df['noise'].unique())}")

    result_df = get_best_per_dataset(df, model_filter='LOC-NFST', noise_filter=0.0)

    if result_df.empty:
        print("\n[ERROR] No results found! Check that your CSV files have noise=0 rows.")
        sys.exit(1)

    print_pretty_table(result_df)

    # Save CSV
    csv_out = os.path.join(OUTPUT_DIR, 'best_results_per_dataset.csv')
    result_df.to_csv(csv_out, index=False)
    print(f"\n[+] CSV saved to: {csv_out}")

    # Print LaTeX
    latex_str = to_latex(result_df)
    tex_out = os.path.join(OUTPUT_DIR, 'best_results_per_dataset.tex')
    with open(tex_out, 'w') as f:
        f.write(latex_str)
    print(f"[+] LaTeX saved to: {tex_out}")
    print("\n--- LaTeX Table ---\n")
    print(latex_str)
