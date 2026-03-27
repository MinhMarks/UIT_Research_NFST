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

# NEW CONFIGURATION (Request 27/03)
N_TOP = 5                 # Number of top baseline models to show
N_BOTTOM = 3              # Number of bottom baseline models to show
TARGET_DATASET = 'ToNIoT' # Dataset to filter for (Set to None for all)
METRIC_NAME = 'aucpr'    # Primary metric to use ('aucroc' or 'aucpr')
EXCLUDE_MODELS = ['DEVNET'] # Models to remove from all results
BASELINE_SCALER = 'MinMaxScaler' # Scaler for baselines (None for best, or 'MinMaxScaler', etc.)

ANOMALY_MODES = ['local', 'cluster', 'global']

# Directories to scan for results
_script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(_script_dir, "..", "..", ".."))
SCAN_DIRS = [
    os.path.join(PROJECT_ROOT, "notebooks", "experiments", "outputs/Anomaly_Type_NFST_MemOpt_20260325_134907"),
    os.path.join(PROJECT_ROOT, "notebooks", "baselines", "outputs/oke"),
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
                    if 'nfst' in fname or 'memopt' in fname:
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
        merged['clean_dataset'] = merged[ds_col].astype(str)
        # Attempt to extract dataset name if it's a long path (e.g. data_CICIoT2023.csv)
        def simplify_ds(x):
            lower_x = x.lower()
            if 'ciciot' in lower_x: return 'CICIoT'
            if 'botiot' in lower_x: return 'BoTIoT'
            if 'n_baiot' in lower_x: return 'N_BaIoT'
            if 'toniot' in lower_x: return 'ToNIoT'
            return os.path.basename(x)
        merged['simple_dataset'] = merged['clean_dataset'].apply(simplify_ds)
    else:
        merged['simple_dataset'] = 'Unknown'

    # FILTER FOR TARGET DATASET
    if TARGET_DATASET:
        print(f"Filtering for dataset: {TARGET_DATASET}...")
        merged = merged[merged['simple_dataset'].str.contains(TARGET_DATASET, case=False, na=False)]
        
    if merged.empty:
        print(f"Warning: No data remaining after filtering for {TARGET_DATASET}")
        return pd.DataFrame()

    # FILTER FOR EXCLUDED MODELS
    if EXCLUDE_MODELS:
        print(f"Excluding models: {EXCLUDE_MODELS}...")
        merged = merged[~merged['clean_model'].isin(EXCLUDE_MODELS)]
        
    # SCALER FILTER FOR BASELINES (Request 27/03)
    if BASELINE_SCALER:
        print(f"Filtering baselines for scaler: {BASELINE_SCALER}...")
        is_ours = merged['clean_model'] == 'LOC-NFST'
        merged = merged[is_ours | (merged['scaler'] == BASELINE_SCALER)]
        
    # Group by (Mode, Model, Dataset) and take MAX (best scaler/n_clusters)
    metric = METRIC_NAME.lower()
    if metric not in merged.columns:
        print(f"Warning: {metric} not found in columns. Defaulting to 'aucroc'")
        metric = 'aucroc'
        
    best_results = merged.groupby(['final_mode', 'simple_dataset', 'clean_model'])[metric].max().reset_index()
    # Calculate Mean across all datasets for each (Mode, Model)
    summary = best_results.groupby(['final_mode', 'clean_model'])[metric].mean().reset_index()
    return summary

def create_composite_table(summary):
    """Create and save a summary table averaged across all anomaly types."""
    metric = METRIC_NAME.lower()
    if summary.empty: return
    
    pivot_df = summary.pivot(index='clean_model', columns='final_mode', values=metric)
    
    # Calculate Cross-Type Mean
    pivot_df['Average'] = pivot_df.mean(axis=1)
    # Sort by Average
    pivot_df = pivot_df.sort_values('Average', ascending=False)
    
    # Round for clarity
    pivot_df = pivot_df.round(4)
    
    save_path = os.path.join(_script_dir, "plots", "Anomaly_Type_Summary_Table.csv")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pivot_df.to_csv(save_path)
    print(f"Aggregated summary table saved to: {save_path}")
    return pivot_df

def plot_composite_chart(pivot_df):
    """Generate professional scientific bar chart for average performance across all types."""
    if pivot_df is None or pivot_df.empty: return
    
    # Use the 'Average' column we just created
    metric_label = 'Average'
    data = pivot_df.reset_index()
    
    # Selection Logic: Always include LOC-NFST + Top N + Bottom M Baselines
    our_model = data[data['clean_model'] == 'LOC-NFST']
    baselines = data[data['clean_model'] != 'LOC-NFST'].sort_values(metric_label, ascending=False)
    
    # Pick Top and Bottom
    top_baselines = baselines.head(N_TOP)
    bottom_baselines = baselines.tail(N_BOTTOM)
    
    # Recombine and Sort by Average for plotting
    final_data = pd.concat([our_model, top_baselines, bottom_baselines]).drop_duplicates().sort_values(metric_label, ascending=False)
    
    plt.figure(figsize=(10, 7))
    palette = [COLOR_PALETTE['LOC-NFST'] if m == 'LOC-NFST' else COLOR_PALETTE['Baseline'] for m in final_data['clean_model']]
    
    ax = sns.barplot(
        data=final_data,
        x='clean_model',
        y=metric_label,
        hue='clean_model',
        palette=palette,
        edgecolor='black',
        alpha=0.9,
        legend=False
    )
    
    # Scientific Polish
    plt.ylabel('Overall Average AUC (%)', fontsize=FONT_SIZE_AXES)
    plt.xlabel('Algorithm', fontsize=FONT_SIZE_AXES)
    plt.xticks(fontsize=FONT_SIZE_TICKS, rotation=45, ha='right')
    plt.yticks(fontsize=FONT_SIZE_TICKS)
    
    for lbl in ax.get_xticklabels():
        if lbl.get_text() == 'LOC-NFST':
            lbl.set_fontweight('bold')
            
    # Dynamic zoom
    min_val = final_data[metric_label].min()
    plt.ylim(max(0, min_val - 2), 100)
    
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(_script_dir, "plots", "Anomaly_Type_Overall_Average.png")
    plt.savefig(save_path, dpi=DPI)
    print(f"Generated Composite Chart: {save_path}")
    plt.close()

def plot_anomaly_type(summary, mode):
    """Generate professional scientific bar chart for a specific anomaly mode."""
    metric = METRIC_NAME.lower()
    data = summary[summary['final_mode'] == mode].copy()
    if data.empty:
        print(f"No data for mode: {mode}")
        return
        
    # Selection Logic: Always include LOC-NFST + Top N + Bottom M Baselines
    our_model = data[data['clean_model'] == 'LOC-NFST']
    baselines = data[data['clean_model'] != 'LOC-NFST'].sort_values(metric, ascending=False)
    
    # Pick Top and Bottom
    top_baselines = baselines.head(N_TOP)
    bottom_baselines = baselines.tail(N_BOTTOM)
    
    # Recombine and Sort by Metric for plotting
    final_data = pd.concat([our_model, top_baselines, bottom_baselines]).drop_duplicates().sort_values(metric, ascending=False)
    
    if final_data.empty:
        print(f"No models found for {mode} in {TARGET_DATASET}")
        return

    plt.figure(figsize=(10, 7))
    palette = [COLOR_PALETTE['LOC-NFST'] if m == 'LOC-NFST' else COLOR_PALETTE['Baseline'] for m in final_data['clean_model']]
    
    ax = sns.barplot(
        data=final_data,
        x='clean_model',
        y=metric,
        hue='clean_model',
        palette=palette,
        edgecolor='black',
        alpha=0.9,
        legend=False
    )
    
    # Scientific Polish
    ylabel = 'Average AUC-ROC (%)' if metric == 'aucroc' else 'Average AUC-PR (%)'
    plt.ylabel(ylabel, fontsize=FONT_SIZE_AXES)
    plt.xlabel('Algorithm', fontsize=FONT_SIZE_AXES)
    # ROTATE LABELS TO PREVENT OVERLAP
    plt.xticks(fontsize=FONT_SIZE_TICKS, rotation=45, ha='right')
    plt.yticks(fontsize=FONT_SIZE_TICKS)
    
    # Bold LOC-NFST in ticks
    for lbl in ax.get_xticklabels():
        if lbl.get_text() == 'LOC-NFST':
            lbl.set_fontweight('bold')
            
    # Dynamic zoom for clarity
    min_val = final_data[metric].min()
    plt.ylim(max(0, min_val - 2), 100)
    
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
        
        # 1. Generate Individual Plots
        for mode in ANOMALY_MODES:
            plot_anomaly_type(summary, mode)
            
        # 2. Generate Composite Summary Table
        print("\nCreating Composite Anomaly Type Table...")
        composite_table = create_composite_table(summary)
        print("\nPreview of Top Performers (Mean across all types):")
        print(composite_table.head(10))
        
        # 3. Generate Composite Average Chart
        print("\nGenerating Composite Anomaly Chart...")
        plot_composite_chart(composite_table)
        
    print("\nProcessing complete.")
