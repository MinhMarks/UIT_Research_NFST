import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# CONFIGURATION & STYLING
# ============================================================================
COLOR_PALETTE = {
    'LOC-NFST': '#333333',  # Charcoal gray
    'Baseline': '#4A90E2',   # Professional Blue
}
FONT_SIZE_AXES = 15
FONT_SIZE_TICKS = 14
DPI = 300

ANOMALY_MODES = ['local', 'cluster', 'global']

# Directories to scan for results
_script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(_script_dir, "..", "..", ".."))
SCAN_DIRS = [
    os.path.join(PROJECT_ROOT, "notebooks", "experiments", "outputs"),
    os.path.join(PROJECT_ROOT, "notebooks", "baselines", "outputs"),
    # Fallback to Structure_Result and Results/final if needed
    os.path.join(PROJECT_ROOT, "Structure_Result"),
    os.path.join(PROJECT_ROOT, "Results", "final")
]

def normalize_model_name(name):
    if not isinstance(name, str): return str(name)
    name_clean = name.lower().strip()
    if any(alias in name_clean for alias in ['ourmodel', 'nhatauto', 'loc-nfst']):
        return 'LOC-NFST'
    # Standardize baseline names (e.g., iforest -> IForest)
    if name_clean == 'iforest': return 'IForest'
    if name_clean == 'ocsvm': return 'OCSVM'
    if name_clean == 'deepsvdd': return 'DeepSVDD'
    return name.upper()

def find_all_results():
    """Recursively discover and normalize all relevant CSV results."""
    dfs = []
    print(f"Scanning for results in {len(SCAN_DIRS)} directories...")
    
    for root_dir in SCAN_DIRS:
        if not os.path.exists(root_dir): continue
        
        csv_files = glob.glob(os.path.join(root_dir, "**", "*.csv"), recursive=True)
        for f in csv_files:
            try:
                # Basic validation: must contain one of our modes or be a report
                fname = os.path.basename(f).lower()
                is_relevant = any(m in fname for m in ANOMALY_MODES) or 'report' in fname or 'outlier' in fname
                if not is_relevant: continue
                
                df = pd.read_csv(f)
                df.columns = [c.lower().strip() for c in df.columns]
                
                # Check for essential columns (aucroc + model)
                if 'aucroc' not in df.columns: continue
                
                # Identify Mode
                mode_val = None
                mode_col = next((c for c in df.columns if 'mode' in c), None)
                if mode_col:
                    # If column exists, use it (handles multi-mode files like Structure_Result)
                    pass 
                else:
                    # Infer from filename
                    for m in ANOMALY_MODES:
                        if m in fname:
                            mode_val = m
                            break
                
                if mode_val:
                    df['inferred_mode'] = mode_val
                
                # Standardize model names
                if 'model' in df.columns:
                    df['clean_model'] = df['model'].apply(normalize_model_name)
                elif 'classifier_name' in df.columns:
                    df['clean_model'] = df['classifier_name'].apply(normalize_model_name)
                else:
                    # If no model column, check if it's an NFST-only result
                    if 'nfst' in fname or 'ourmodel' in fname:
                        df['clean_model'] = 'LOC-NFST'
                    else:
                        continue
                
                dfs.append(df)
            except Exception as e:
                print(f"Error reading {f}: {e}")
                
    return dfs

def aggregate_metrics(dfs):
    """Aggregate results into a single clean performance table."""
    merged = pd.concat(dfs, ignore_index=True)
    
    # Consolidate Mode
    mode_col = next((c for c in merged.columns if 'mode' in c and c != 'inferred_mode'), 'inferred_mode')
    merged['final_mode'] = merged[mode_col].str.lower()
    
    # Consolidate Dataset
    ds_col = next((c for c in merged.columns if 'dataset' in c), 'dataset')
    if ds_col in merged.columns:
        merged['clean_dataset'] = merged[ds_col].astype(str).apply(lambda x: os.path.basename(x).split('_')[1] if '_data_' in x else x)
    else:
        merged['clean_dataset'] = 'Unknown'
        
    # Group by (Mode, Model, Dataset) and take MAX AUCROC (best scaler/n_clusters)
    best_results = merged.groupby(['final_mode', 'clean_dataset', 'clean_model'])['aucroc'].max().reset_index()
    
    # Calculate Mean across all datasets for each (Mode, Model)
    summary = best_results.groupby(['final_mode', 'clean_model'])['aucroc'].mean().reset_index()
    
    return summary

def plot_anomaly_type(summary, mode):
    """Generate professional scientific bar chart for a specific anomaly mode."""
    data = summary[summary['final_mode'] == mode].copy()
    if data.empty:
        print(f"No data for mode: {mode}")
        return
        
    # Pick Top 5
    data = data.sort_values('aucroc', ascending=False).head(5)
    
    plt.figure(figsize=(10, 7))
    palette = [COLOR_PALETTE['LOC-NFST'] if m == 'LOC-NFST' else COLOR_PALETTE['Baseline'] for m in data['clean_model']]
    
    ax = sns.barplot(
        data=data,
        x='clean_model',
        y='aucroc',
        palette=palette,
        edgecolor='black',
        alpha=0.9
    )
    
    # Scientific Polish
    plt.ylabel('Average AUC-ROC (%)', fontsize=FONT_SIZE_AXES)
    plt.xlabel('Algorithm', fontsize=FONT_SIZE_AXES)
    plt.xticks(fontsize=FONT_SIZE_TICKS)
    plt.yticks(fontsize=FONT_SIZE_TICKS)
    
    # Bold LOC-NFST in ticks
    for lbl in ax.get_xticklabels():
        if lbl.get_text() == 'LOC-NFST':
            lbl.set_fontweight('bold')
            
    # Dynamic zoom for clarity
    min_auc = data['aucroc'].min()
    plt.ylim(max(0, min_auc - 2), 100)
    
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(_script_dir, "plots", f"Anomaly_Type_{mode.capitalize()}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=DPI)
    print(f"Generated: {save_path}")
    plt.close()

if __name__ == "__main__":
    print("=== Robust Anomaly Type Aggregator & Plotter ===")
    all_dfs = find_all_results()
    if not all_dfs:
        print("Fatal: No results found in search paths.")
    else:
        summary = aggregate_metrics(all_dfs)
        print("\nAggregated Summary (Top Performer per Mode):")
        print(summary.sort_values(['final_mode', 'aucroc'], ascending=[True, False]).groupby('final_mode').head(1))
        
        for mode in ANOMALY_MODES:
            plot_anomaly_type(summary, mode)
    print("\nProcessing complete.")
