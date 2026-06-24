"""
Sensitivity Analysis: Metric vs n_cluster
==========================================
Shows how the selected metric (AUCROC or AUCPR) of LOC-NFST fluctuates
as the n_cluster hyperparameter changes. Filters only noise=0 and
selects the best scaler per dataset.

Output: A line-plot saved to results/Sensitivity_nCluster_{METRIC}.png
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
OUTPUT_DIR = os.path.normpath(os.path.join(_script_dir, 'results'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# >>> CONFIGURATION: change METRIC to 'AUCPR' to switch metric <<<
METRIC = 'AUCROC'      # Options: 'AUCROC'  |  'AUCPR'
START_K = 2            # Plot from this k value onwards
END_K = 175            # Plot up to this k value (None = no upper limit)
DEFAULT_K = 75        # Selected default k to highlight
SHOW_RAW_LINE      = True   # Show raw (pre-smoothing) dotted line
SHOW_SMOOTHED_LINE = False   # Show smoothed (rolling-average) bold line
SHOW_STD_BAND      = False  # Show shaded ±std band around the mean (can be noisy)

# High-contrast qualitative palette ─ distinct across colour-vision types
QUAL_COLORS = [
    '#E63946',  # Vivid red
    '#1D6FA4',  # Strong blue
    '#2A9D5C',  # Emerald green
    '#F4A300',  # Deep amber
    '#7B2D8B',  # Purple
    '#E08C00',  # Dark orange
    '#17BECF',  # Teal
    '#8C564B',  # Earthy brown
]
# Only use clearly distinct line styles ─ NO dots to avoid confusion with raw dotted line
LINESTYLES = ['-', '--', '-.']
MARKERS    = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# DATA LOADING  (reuses same convention as generate_cd_plots.py)
# ---------------------------------------------------------------------------
KNOWN_DATASETS = ['BoTIoT', 'ToNIoT', 'N_BaIoT', 'CICIoT', 'EdgeIIoTset', 'IoTID20']

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
        
        metric_col = METRIC.lower()   # 'aucroc' or 'aucpr'
        if metric_col not in df.columns:
            # Try loading from the original col name (e.g. 'aucpr' might be 'auc_pr')
            for c in df.columns:
                if 'aucpr' in c or 'auc_pr' in c:
                    df.rename(columns={c: 'aucpr'}, inplace=True)
                    break
        df['aucroc'] = pd.to_numeric(df.get('aucroc', np.nan), errors='coerce')
        if 'aucpr' in df.columns:
            df['aucpr'] = pd.to_numeric(df['aucpr'], errors='coerce')
        else:
            df['aucpr'] = np.nan
        df['noise'] = pd.to_numeric(df.get('noise', 0), errors='coerce').fillna(0)
        df['ncluster'] = pd.to_numeric(df.get('ncluster', np.nan), errors='coerce')
        df['scaler'] = df.get('scaler', 'Unknown').astype(str)
        
        frames.append(df)
    
    combined = pd.concat(frames, ignore_index=True).dropna(subset=['ncluster'])
    # Drop rows where the selected metric is NaN
    metric_col = METRIC.lower()
    if metric_col in combined.columns:
        combined = combined.dropna(subset=[metric_col])
    else:
        raise ValueError(f"Metric '{METRIC}' column not found in results CSVs.")
    return combined

def select_best_scaler(df):
    """
    For each (dataset, scaler) pair filtered at noise=0,
    compute the max of the selected METRIC across all n_cluster values.
    Then pick the scaler with the highest max per dataset.
    """
    metric_col = METRIC.lower()
    df_zero = df[df['noise'] == 0.0].copy()
    
    best = (df_zero.groupby(['dataset', 'scaler'])[metric_col]
                   .max()
                   .reset_index()
                   .rename(columns={metric_col: 'max_metric'}))
    
    idx = best.groupby('dataset')['max_metric'].idxmax()
    best_scalers = best.loc[idx].set_index('dataset')['scaler'].to_dict()
    
    print(f"\nBest Scaler per Dataset (by {METRIC}):")
    for ds, sc in best_scalers.items():
        print(f"  {ds:<12} ->  {sc}")
    
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
    
    # High-contrast qualitative palette is defined at module level (QUAL_COLORS etc.)

    # Filter data by START_K and END_K
    df_filtered = df_filtered[df_filtered['ncluster'] >= START_K].copy()
    if END_K is not None:
        df_filtered = df_filtered[df_filtered['ncluster'] <= END_K].copy()
    
    fig, ax = plt.subplots(figsize=(11, 5))
    
    for i, ds in enumerate(datasets):
        metric_col = METRIC.lower()
        sub = df_filtered[df_filtered['dataset'] == ds].copy()
        # Average metric per n_cluster
        per_cluster = (sub.groupby('ncluster')[metric_col]
                          .mean()
                          .reset_index()
                          .sort_values('ncluster'))
        
        x = per_cluster['ncluster'].values
        y = per_cluster[metric_col].values
        
        # Std band
        std_per_cluster = (sub.groupby('ncluster')[metric_col]
                              .std()
                              .reset_index()
                              .sort_values('ncluster')[metric_col]
                              .fillna(0)
                              .values)
        
        # Smooth mean line
        x_s, y_s = smooth_curve(x, y, window=7)
        
        color = QUAL_COLORS[i % len(QUAL_COLORS)]
        ls    = LINESTYLES[i % len(LINESTYLES)]
        mk    = MARKERS[i % len(MARKERS)]
        sc_label = best_scalers.get(ds, '')
        
        # Shaded std band (optional)
        if SHOW_STD_BAND:
            ax.fill_between(x, y - std_per_cluster, y + std_per_cluster,
                            alpha=0.10, color=color)
        
        # Raw line — primary when SHOW_SMOOTHED_LINE is off, secondary otherwise
        if SHOW_RAW_LINE:
            raw_alpha = 1.0 if not SHOW_SMOOTHED_LINE else 0.3
            raw_lw    = 2.2 if not SHOW_SMOOTHED_LINE else 1.0
            raw_ms    = 6   if not SHOW_SMOOTHED_LINE else 4
            raw_label = ds  if not SHOW_SMOOTHED_LINE else None
            ax.plot(x, y, linestyle=ls, linewidth=raw_lw, color=color, alpha=raw_alpha,
                    marker=mk, markevery=max(1, len(x)//12), markersize=raw_ms,
                    label=raw_label, zorder=1 if SHOW_SMOOTHED_LINE else 5)
        
        # Smooth mean line — primary when enabled (always gets the label)
        if SHOW_SMOOTHED_LINE:
            ax.plot(x_s, y_s, linewidth=2.2, color=color, linestyle=ls,
                    marker=mk, markevery=max(1, len(x_s)//12), markersize=6,
                    label=ds, zorder=5)
        elif not SHOW_RAW_LINE:
            # Neither line shown — still register in legend with invisible line
            ax.plot([], [], color=color, linestyle=ls, label=ds)

    # --- Highlight Default K ---
    ax.axvline(x=DEFAULT_K, color='gray', linestyle='--', alpha=0.7, linewidth=1.5,
               label=f"Default k={DEFAULT_K}")
    
    # --- Styling ---
    ax.set_xlabel("Number of Clusters (k)", fontsize=13)
    ax.set_ylabel(f"{METRIC} (%)", fontsize=13)
    # ax.set_title(f"Sensitivity of LOC-NFST {METRIC} to Number of Clusters\n(noise = 0%, best scaler per dataset)", fontsize=13, pad=12)
    ax.set_title(f"", fontsize=13, pad=12)
    
    ax.legend(loc='lower right', fontsize=10, framealpha=0.85)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    ax.grid(axis='y', linestyle='--', alpha=0.45)
    
    # Dynamic Y-min to avoid large white space at the bottom
    # Get the overall minimum value including shaded regions
    all_mins = []
    for i, ds in enumerate(datasets):
        metric_col = METRIC.lower()
        sub = df_filtered[df_filtered['dataset'] == ds]
        per_cluster = sub.groupby('ncluster')[metric_col].mean()
        std_per_cluster = sub.groupby('ncluster')[metric_col].std().fillna(0)
        all_mins.append((per_cluster - std_per_cluster).min())
    
    lowest_metric = min(all_mins) if all_mins else 50
    # User suggestion: if high, start at 70. Otherwise 50 or lower if data drops.
    if lowest_metric > 80:
        y_min = 70
    elif lowest_metric > 60:
        y_min = 50
    else:
        y_min = int(np.floor(lowest_metric / 10) * 10 - 10)
        y_min = max(0, y_min)
        
    ax.set_ylim(y_min, 108)  # Set limit first
    ax.set_yticks(range(int(y_min), 101, 10)) # Then set explicit ticks
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    ax.grid(axis='y', linestyle='--', alpha=0.45)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    out_path = os.path.join(OUTPUT_DIR, f'Sensitivity_nCluster_{METRIC}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[+] Plot saved to: {out_path}")
    return out_path

def save_sensitivity_data(df_filtered, best_scalers):
    """Save the filtered and aggregated data to CSV and LaTeX for further analysis."""
    metric_col = METRIC.lower()
    
    # 1. Save summary data (Aggregated by Dataset and nCluster)
    summary_list = []
    datasets = sorted(df_filtered['dataset'].unique())
    
    for ds in datasets:
        sub = df_filtered[df_filtered['dataset'] == ds].copy()
        per_cluster = (sub.groupby('ncluster')[metric_col]
                          .agg(['mean', 'std', 'count'])
                          .reset_index()
                          .sort_values('ncluster'))
        per_cluster['dataset'] = ds
        per_cluster['best_scaler'] = best_scalers.get(ds, 'Unknown')
        summary_list.append(per_cluster)
    
    df_summary = pd.concat(summary_list, ignore_index=True)
    
    # Reorder columns for readability
    cols = ['dataset', 'best_scaler', 'ncluster', 'mean', 'std', 'count']
    df_summary = df_summary[cols].rename(columns={'mean': f'{METRIC}_mean', 'std': f'{METRIC}_std'})
    
    csv_out = os.path.join(OUTPUT_DIR, f'Sensitivity_Numerical_Summary_{METRIC}.csv')
    df_summary.to_csv(csv_out, index=False)
    print(f"[+] Summary data saved to: {csv_out}")
    
    # 2. Save a LaTeX table of the same data (pivot format for the paper)
    # Let's pick some representative n_cluster values for the LaTeX table to keep it readable
    pivot_ds = df_summary.pivot(index='dataset', columns='ncluster', values=f'{METRIC}_mean')
    
    # Select a subset of clusters for the TeX table if there are too many
    if len(pivot_ds.columns) > 10:
        # Pick 10 evenly spaced ones
        cols_to_keep = pivot_ds.columns[np.linspace(0, len(pivot_ds.columns)-1, 10).astype(int)]
        pivot_ds = pivot_ds[cols_to_keep]
    
    tex_out = os.path.join(OUTPUT_DIR, f'Sensitivity_Numerical_Summary_{METRIC}.tex')
    pivot_ds.to_latex(tex_out, float_format="%.2f")
    print(f"[+] LaTeX table saved to: {tex_out}")
    
    return csv_out, tex_out

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # Search the outputs folder of OC_NFST experiments (adjust if needed)
    experiments_dir = os.path.normpath(os.path.join(_script_dir, 'exp'))
    
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
    save_sensitivity_data(df_filtered, best_scalers)
