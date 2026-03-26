"""
Sensitivity Analysis: AUCROC vs n_cluster
==========================================
Shows how the AUCROC score of LOC-NFST fluctuates as the n_cluster
hyperparameter changes. Filters only noise=0 and selects the best
scaler per dataset (highest max AUCROC across all n_clusters).

Output: A line-plot saved to pictures/Sensitivity_nCluster.png
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.normpath(os.path.join(_script_dir, '..', '..', '..'))
OUTPUT_DIR = os.path.normpath(os.path.join(_project_root, 'pictures'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# DATA LOADING  (reuses same convention as generate_cd_plots.py)
# ---------------------------------------------------------------------------
KNOWN_DATASETS = ['BoTIoT', 'ToNIoT', 'N_BaIoT', 'CICIoT']

def normalize_dataset_name(name):
    if not isinstance(name, str): return str(name)
    name_lower = name.lower()
    for ds in KNOWN_DATASETS:
        if ds.lower() in name_lower: return ds
    base = os.path.basename(name)
    if base.endswith('.csv'): base = base[:-4]
    return base

def find_nfst_csv_files(root_dir):
    """Search for LOC-NFST result CSVs that contain nCluster column."""
    csv_files = glob.glob(os.path.join(root_dir, "**", "*.csv"), recursive=True)
    results = []
    for f in csv_files:
        try:
            if os.path.getsize(f) < 100: continue
            df = pd.read_csv(f)
            cols = [c.lower() for c in df.columns]
            # Must have nCluster (or ncluster) AND aucroc
            if 'aucroc' in cols and any('ncluster' in c or 'n_cluster' in c for c in cols):
                results.append(f)
        except Exception:
            pass
    return results

def load_nfst_data(search_root):
    files = find_nfst_csv_files(search_root)
    if not files:
        raise FileNotFoundError(
            f"No NFST result CSVs with nCluster column found under:\n  {search_root}\n"
            "Please run OC_NFST_memory_optimized.py first."
        )
    print(f"Found {len(files)} result file(s):")
    for f in files: print(f"  {f}")
    
    frames = []
    for f in files:
        df = pd.read_csv(f)
        # Standardize column names
        df.columns = [c.lower().replace(' ', '_') for c in df.columns]
        
        # Normalize column names
        rename_map = {}
        for c in df.columns:
            if 'ncluster' in c and 'requested' not in c and c != 'ncluster': rename_map[c] = 'ncluster'
            if 'noise_percentage' == c: rename_map[c] = 'noise'
            if 'scaled' == c: rename_map[c] = 'scaler'
        df.rename(columns=rename_map, inplace=True)
        
        if 'ncluster' not in df.columns:
            for col in df.columns:
                if 'ncluster' in col or 'cluster' in col:
                    df.rename(columns={col: 'ncluster'}, inplace=True)
                    break
        
        if 'dataset' in df.columns:
            df['dataset'] = df['dataset'].apply(normalize_dataset_name)
        
        df['aucroc'] = pd.to_numeric(df.get('aucroc', np.nan), errors='coerce')
        df['noise'] = pd.to_numeric(df.get('noise', 0), errors='coerce').fillna(0)
        df['ncluster'] = pd.to_numeric(df.get('ncluster', np.nan), errors='coerce')
        df['scaler'] = df.get('scaler', 'Unknown').astype(str)
        
        frames.append(df)
    
    combined = pd.concat(frames, ignore_index=True).dropna(subset=['aucroc', 'ncluster'])
    return combined

def select_best_scaler(df):
    """
    For each (dataset, scaler) pair filtered at noise=0,
    compute the max AUCROC across all n_cluster values.
    Then pick the scaler with the highest max per dataset.
    """
    df_zero = df[df['noise'] == 0.0].copy()
    
    # Max AUCROC per (dataset, scaler)
    best = (df_zero.groupby(['dataset', 'scaler'])['aucroc']
                   .max()
                   .reset_index()
                   .rename(columns={'aucroc': 'max_aucroc'}))
    
    # Which scaler gives highest max per dataset?
    idx = best.groupby('dataset')['max_aucroc'].idxmax()
    best_scalers = best.loc[idx].set_index('dataset')['scaler'].to_dict()
    
    print("\nBest Scaler per Dataset:")
    for ds, sc in best_scalers.items():
        print(f"  {ds:<12} ->  {sc}")
    
    # Filter original data keeping only the best scaler rows at noise=0
    rows = []
    for ds, sc in best_scalers.items():
        mask = (df_zero['dataset'] == ds) & (df_zero['scaler'] == sc)
        rows.append(df_zero[mask])
    
    return pd.concat(rows, ignore_index=True), best_scalers

def smooth_curve(x, y, window=5):
    """Running average smoothing for cleaner visualization."""
    if len(y) < window: return x, y
    kernel = np.ones(window) / window
    y_smooth = np.convolve(y, kernel, mode='valid')
    trim = (window - 1) // 2
    x_smooth = x[trim: trim + len(y_smooth)]
    return x_smooth, y_smooth

def plot_sensitivity(df_filtered, best_scalers):
    datasets = sorted(df_filtered['dataset'].unique())
    n_ds = len(datasets)
    
    # Choose color palette
    palette = sns.color_palette("tab10", n_ds)
    
    fig, ax = plt.subplots(figsize=(11, 5))
    
    for i, ds in enumerate(datasets):
        sub = df_filtered[df_filtered['dataset'] == ds].copy()
        # Average AUCROC per n_cluster (multiple seeds/params may exist)
        per_cluster = (sub.groupby('ncluster')['aucroc']
                          .mean()
                          .reset_index()
                          .sort_values('ncluster'))
        
        x = per_cluster['ncluster'].values
        y = per_cluster['aucroc'].values
        
        # Raw band (std)
        std_per_cluster = (sub.groupby('ncluster')['aucroc']
                              .std()
                              .reset_index()
                              .sort_values('ncluster')['aucroc']
                              .fillna(0)
                              .values)
        
        # Smooth mean line
        x_s, y_s = smooth_curve(x, y, window=7)
        
        color = palette[i]
        sc_label = best_scalers.get(ds, '')
        
        # Shaded std band
        ax.fill_between(x, y - std_per_cluster, y + std_per_cluster,
                        alpha=0.12, color=color)
        
        # Raw dotted line
        ax.plot(x, y, linestyle=':', linewidth=0.8, color=color, alpha=0.5)
        
        # Smooth mean line (bold)
        ax.plot(x_s, y_s, linewidth=2.0, color=color,
                label=f"{ds} ({sc_label})")
    
    # --- Styling ---
    ax.set_xlabel("Number of Clusters (k)", fontsize=13)
    ax.set_ylabel("AUCROC (%)", fontsize=13)
    ax.set_title("Sensitivity of LOC-NFST AUCROC to Number of Clusters\n(noise = 0%, best scaler per dataset)", fontsize=13, pad=12)
    
    ax.legend(loc='lower right', fontsize=10, framealpha=0.85)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    ax.grid(axis='y', linestyle='--', alpha=0.45)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    out_path = os.path.join(OUTPUT_DIR, 'Sensitivity_nCluster.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[+] Plot saved to: {out_path}")
    return out_path

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # Search the outputs folder of OC_NFST experiments (adjust if needed)
    experiments_dir = os.path.normpath(os.path.join(_script_dir, '..', '..', 'experiments', 'outputs'))
    
    if not os.path.exists(experiments_dir):
        # Fallback: ask user to point us to the right folder
        import sys
        if len(sys.argv) > 1:
            experiments_dir = sys.argv[1]
        else:
            experiments_dir = input("Enter path to OC-NFST experiment output folder:\n> ").strip()

    print(f"Searching in: {experiments_dir}")
    
    df = load_nfst_data(experiments_dir)
    print(f"\nLoaded {len(df):,} rows | Noise values: {sorted(df['noise'].unique())}")
    
    df_filtered, best_scalers = select_best_scaler(df)
    print(f"Rows after filtering (noise=0, best scaler): {len(df_filtered):,}")
    
    plot_sensitivity(df_filtered, best_scalers)
