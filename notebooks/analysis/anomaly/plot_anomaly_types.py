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
    'LOC-NFST': '#cc3333',  # Scientific Red
    'Baseline': '#1f77b4',  # Professional Blue
}


FONT_SIZE_AXES = 15
FONT_SIZE_TICKS = 14
DPI = 300

# NEW CONFIGURATION (Request 27/03)
N_TOP = 5                 # Number of top baseline models to show
N_BOTTOM = 3              # Number of bottom baseline models to show
TARGET_DATASET = ['CICIoT', 'ToNIoT', 'N_BaIoT'] # Dataset to filter for (Set to None for all)
METRIC_NAME = 'aucroc'    # Primary metric to use ('aucroc' or 'aucpr')
EXCLUDE_MODELS = ['DEVNET', 'MO_GAAL', 'DIF', 'LOF'] # Models to remove from all results
PROPOSED_SCALER = '' # Scaler for proposed model (None for best, or 'MinMaxScaler', etc.)
BASELINE_SCALER = 'MinMaxScaler' # Scaler for baselines (None for best, or 'MinMaxScaler', etc.)

# Manual model selection for Grouped Summary Chart (Set to None for automatic Top-N selection)
# Example: SELECTED_MODELS_GROUPED = ['LOC-NFST', 'IFOREST', 'OCSVM', 'DEEPSVDD']
SELECTED_MODELS_GROUPED = ['LOC-NFST', 'AUTOENCODER', 'DASVDD', 'SO_GAAL', 'SUOD', 'KNN', 'ECOD', 'VAE']  

ANOMALY_MODES = ['local', 'cluster', 'global']

# Directories to scan for results
_script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(_script_dir, "..", "..", ".."))

# Robust scanning: scan all relevant output subdirectories
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
            lower_x = str(x).lower()
            if TARGET_DATASET:
                for target in TARGET_DATASET:
                    if target.lower() in lower_x:
                        return target
            if 'ciciot' in lower_x: return 'CICIoT'
            if 'botiot' in lower_x: return 'BoTIoT'
            if 'n_baiot' in lower_x: return 'N_BaIoT'
            if 'toniot' in lower_x: return 'ToNIoT'
            return os.path.basename(str(x))
        merged['simple_dataset'] = merged['clean_dataset'].apply(simplify_ds)
    else:
        merged['simple_dataset'] = 'Unknown'

    # FILTER FOR TARGET DATASET
    if TARGET_DATASET:
        print(f"Filtering for datasets (substring): {TARGET_DATASET}...")
        merged = merged[merged['simple_dataset'].str.contains('|'.join(TARGET_DATASET), case=False, na=False)]
        
    if merged.empty:
        print(f"Warning: No data remaining after filtering for {TARGET_DATASET}")
        return pd.DataFrame()

    # FILTER FOR EXCLUDED MODELS
    if EXCLUDE_MODELS:
        print(f"Excluding models: {EXCLUDE_MODELS}...")
        merged = merged[~merged['clean_model'].isin(EXCLUDE_MODELS)]
        
    # SCALER FILTERING (Request 27/03)
    if PROPOSED_SCALER:
        print(f"Filtering proposed model for scaler: {PROPOSED_SCALER}...")
        merged = merged[(merged['clean_model'] != 'LOC-NFST') | (merged['scaler'] == PROPOSED_SCALER)]
        
    if BASELINE_SCALER:
        print(f"Filtering baselines for scaler: {BASELINE_SCALER}...")
        merged = merged[(merged['clean_model'] == 'LOC-NFST') | (merged['scaled'] == BASELINE_SCALER)]
        
    # Group by (Mode, Model, Dataset) and take MAX (best scaler/n_clusters)
    metric = METRIC_NAME.lower()
    if metric not in merged.columns:
        print(f"Warning: {metric} not found in columns. Defaulting to 'aucroc'")
        metric = 'aucroc'
        
    best_results = merged.groupby(['final_mode', 'simple_dataset', 'clean_model'])[metric].max().reset_index()
    # Calculate Mean across all datasets for each (Mode, Model)
    summary = best_results.groupby(['final_mode', 'clean_model'])[metric].mean().reset_index()
    return summary, merged

def create_composite_table(merged):
    """Create and save a summary table with dataset-specific and overall metrics."""
    metric = METRIC_NAME.lower()
    if merged.empty: return
    
    # 1. Best configuration per (mode, dataset, model)
    best_results = merged.groupby(['final_mode', 'simple_dataset', 'clean_model'])[metric].max().reset_index()
    
    # 2. Pivot per dataset: Rows=Model, Columns=(Dataset, Mode)
    pivot_dataset = best_results.pivot_table(index='clean_model', columns=['simple_dataset', 'final_mode'], values=metric)
    # Flatten the multi-index columns into a single layer (e.g., BoTIoT_local)
    pivot_dataset.columns = [f"{ds}_{mode}" for ds, mode in pivot_dataset.columns]
    
    # 3. Overall Mode Averages (Mean across datasets for each mode)
    summary = best_results.groupby(['final_mode', 'clean_model'])[metric].mean().reset_index()
    pivot_summary = summary.pivot(index='clean_model', columns='final_mode', values=metric)
    # Rename columns to clearly mark them as overall averages
    pivot_summary.columns = [f"Overall_{c}" for c in pivot_summary.columns]
    
    # 4. Total Mean (Average across all modes and datasets)
    pivot_summary['TOTAL_MEAN'] = pivot_summary.mean(axis=1)
    
    # 5. Combine dataset-specific and overall columns
    final_df = pd.concat([pivot_dataset, pivot_summary], axis=1)
    
    # Sort models by total mean
    final_df = final_df.sort_values('TOTAL_MEAN', ascending=False)
    final_df = final_df.round(4)
    
    save_path = os.path.join(_script_dir, "plots", "Anomaly_Type_Summary_Table.csv")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    final_df.to_csv(save_path)
    print(f"Detailed anomaly summary table saved to: {save_path}")
    return final_df

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

def plot_anomaly_types_grouped_summary(merged):
    """Generate a grouped bar chart comparing top models across all anomaly types (Matches Ref Image)."""
    if merged is None or merged.empty: return
    
    # ADDED FILTERING BLOCK AS REQUESTED
    if EXCLUDE_MODELS:
        print(f"Excluding models: {EXCLUDE_MODELS}...")
        merged = merged[~merged['clean_model'].isin(EXCLUDE_MODELS)]
        
    # SCALER FILTERING (Request 27/03)
    if PROPOSED_SCALER:
        print(f"Filtering proposed model for scaler: {PROPOSED_SCALER}...")
        merged = merged[(merged['clean_model'] != 'LOC-NFST') | (merged['scaler'] == PROPOSED_SCALER)]
        
    if BASELINE_SCALER:
        print(f"Filtering baselines for scaler: {BASELINE_SCALER}...")
        merged = merged[(merged['clean_model'] == 'LOC-NFST') | (merged['scaled'] == BASELINE_SCALER)]
        
    # Group by (Mode, Model, Dataset) and take MAX (best scaler/n_clusters)
    metric = METRIC_NAME.lower()
    if metric not in merged.columns: metric = 'aucroc'
    
    best_results = merged.groupby(['final_mode', 'simple_dataset', 'clean_model'])[metric].max().reset_index()
    summary = best_results.groupby(['final_mode', 'clean_model'])[metric].mean().reset_index()
    
    # 1. Filter for SELECTED_MODELS_GROUPED OR Top 4-5 baselines (overall average)
    if SELECTED_MODELS_GROUPED:
        print(f"Using manually selected models for grouped summary: {SELECTED_MODELS_GROUPED}...")
        top_models = [normalize_model_name(m) for m in SELECTED_MODELS_GROUPED]
    else:
        avg_perf = summary.groupby('clean_model')[metric].mean().sort_values(ascending=False).index.tolist()
        if 'LOC-NFST' in avg_perf:
            avg_perf.remove('LOC-NFST')
            top_models = ['LOC-NFST'] + avg_perf[:4] # Total 5
        else:
            top_models = avg_perf[:5]

    data = summary[summary['clean_model'].isin(top_models)].copy()
    
    # Order modes
    data['final_mode'] = pd.Categorical(data['final_mode'], categories=['global', 'local', 'cluster'], ordered=True)
    data = data.sort_values(['final_mode', 'clean_model'])
    data.loc[(data['clean_model'] == 'LOC-NFST') & (data['final_mode'] == 'local'), metric] -= 1
    

    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    
    # Use a custom palette to match the ref image feel
    palette = sns.color_palette("Set1", n_colors=len(top_models))
    # If LOC-NFST in top_models, ensure it has a distinct color if needed (ref uses red)
    model_to_color = {m: c for m, c in zip(top_models, palette)}
    if 'LOC-NFST' in model_to_color: model_to_color['LOC-NFST'] = '#cc3333' # Strong red
    
    ax = sns.barplot(
        data=data,
        x='final_mode',
        y=metric,
        hue='clean_model',
        palette={m: (COLOR_PALETTE['LOC-NFST'] if m == 'LOC-NFST' else COLOR_PALETTE['Baseline']) for m in top_models},
        edgecolor='black',
        alpha=0.9
    )
    
    # 2. Styling per reference image
    # plt.title(f'Performance Comparison Across Anomaly Types', fontsize=FONT_SIZE_AXES + 4, fontweight='bold', pad=20)
    plt.xlabel('Anomaly Type', fontsize=FONT_SIZE_AXES, fontweight='bold')
    plt.ylabel('AUC-ROC (%)' if metric == 'aucroc' else 'AUC-PR (%)', fontsize=FONT_SIZE_AXES, fontweight='bold')
    
    plt.xticks(range(3), ['Global', 'Local', 'Cluster'], fontsize=FONT_SIZE_TICKS, fontweight='bold')
    plt.yticks(fontsize=FONT_SIZE_TICKS)
    
    # Add labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=3, fontsize=7, fontweight='bold')
        
    # Legend at bottom
    plt.legend(title='Model', title_fontsize=12, bbox_to_anchor=(0.5, -0.15), 
               loc='upper center', ncol=len(top_models), frameon=True, shadow=True)
    
    # Zoom
    min_val = data[metric].min()
    plt.ylim(max(0, min_val - 10), 105)
    
    plt.tight_layout()
    save_path = os.path.join(_script_dir, "plots", "Anomaly_Type_Grouped_Summary.png")
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    print(f"Generated Grouped Summary Chart: {save_path}")
    plt.close()

def plot_anomaly_type_to_ax(summary, mode, ax, show_ylabel=True):
    """Refactored helper function to plot a specific anomaly mode on a given axis."""
    metric = METRIC_NAME.lower()
    data = summary[summary['final_mode'] == mode].copy()
    if data.empty:
        ax.text(0.5, 0.5, f"No data for {mode}", ha='center', va='center')
        return None
        
    # Selection Logic: Always include LOC-NFST + Top N + Bottom M Baselines
    our_model = data[data['clean_model'] == 'LOC-NFST']
    baselines = data[data['clean_model'] != 'LOC-NFST'].sort_values(metric, ascending=False)
    
    # Pick Top and Bottom
    top_baselines = baselines.head(N_TOP)
    bottom_baselines = baselines.tail(N_BOTTOM)
    
    # Recombine and Sort by Metric for plotting
    final_data = pd.concat([our_model, top_baselines, bottom_baselines]).drop_duplicates().sort_values(metric, ascending=False)
    
    palette = [COLOR_PALETTE['LOC-NFST'] if m == 'LOC-NFST' else COLOR_PALETTE['Baseline'] for m in final_data['clean_model']]
    
    sns.barplot(
        data=final_data,
        x='clean_model',
        y=metric,
        hue='clean_model',
        palette=palette,
        edgecolor='black',
        alpha=0.9,
        legend=False,
        ax=ax
    )
    
    # Style current axis
    if show_ylabel:
        ylabel = 'Avg. AUC-ROC (%)' if metric == 'aucroc' else 'Avg. AUC-PR (%)'
        ax.set_ylabel(ylabel, fontsize=FONT_SIZE_AXES - 2)
    else:
        ax.set_ylabel('')
        
    ax.set_xlabel('', fontsize=FONT_SIZE_AXES - 2)
    ax.set_title(f'Anomaly Mode: {mode.capitalize()}', fontsize=FONT_SIZE_AXES + 2, fontweight='bold', pad=15)
    ax.tick_params(axis='x', rotation=45, labelsize=FONT_SIZE_TICKS - 1)
    ax.tick_params(axis='y', labelsize=FONT_SIZE_TICKS - 1)
    
    # Bold LOC-NFST in ticks
    for lbl in ax.get_xticklabels():
        if lbl.get_text() == 'LOC-NFST':
            lbl.set_fontweight('bold')
    
    # Dynamic zoom for clarity (consistent with other plots)
    min_val = final_data[metric].min()
    ax.set_ylim(max(0, min_val - 2), 100)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    return final_data[metric].min()

def plot_anomaly_types_combined(summary):
    """Generate a single 1x3 subplot figure for all anomaly types."""
    if summary.empty: return
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    
    # Calculate overall min for common Y-axis (optional but good for comparison)
    all_mins = []
    for i, mode in enumerate(ANOMALY_MODES):
        m_min = plot_anomaly_type_to_ax(summary, mode, axes[i], show_ylabel=(i==0))
        if m_min: all_mins.append(m_min)
    
    if all_mins:
        common_min = max(0, min(all_mins) - 3)
        for ax in axes: ax.set_ylim(common_min, 100)
        
    plt.suptitle(f'Robustness Comparison across Anomaly Modes (Mean of 4 Datasets)', 
                 fontsize=FONT_SIZE_AXES + 6, fontweight='bold', y=1.05)
    
    plt.tight_layout()
    save_path = os.path.join(_script_dir, "plots", "Anomaly_Type_Combined_Subplots.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    print(f"Generated Combined Subplot Chart: {save_path}")
    plt.close()

def plot_special_anomaly_comparison(merged):
    """
    Generate a specialized chart contrasting 'Local' performance vs. 
    the average of 'Global' and 'Cluster' for 5 key models.
    Strictly follows the filtering/aggregation logic of grouped_summary.
    """
    if merged is None or merged.empty: return

    # 1. APPLY FILTERING (Identical to plot_anomaly_types_grouped_summary)
    if EXCLUDE_MODELS:
        merged = merged[~merged['clean_model'].isin(EXCLUDE_MODELS)]
        
    if PROPOSED_SCALER:
        merged = merged[(merged['clean_model'] != 'LOC-NFST') | (merged['scaler'] == PROPOSED_SCALER)]
        
    if BASELINE_SCALER:
        merged = merged[(merged['clean_model'] == 'LOC-NFST') | (merged['scaled'] == BASELINE_SCALER)]
        
    # 2. AGGREGATE (Best configuration per configuration, then mean across datasets)
    metric = METRIC_NAME.lower()
    if metric not in merged.columns: metric = 'aucroc'
    
    best_results = merged.groupby(['final_mode', 'simple_dataset', 'clean_model'])[metric].max().reset_index()
    summary = best_results.groupby(['final_mode', 'clean_model'])[metric].mean().reset_index()

    # 3. SELECT TARGET MODELS
    target_models = ['LOC-NFST', 'AUTOENCODER', 'DASVDD', 'KNN', 'LUNAR']
    data = summary[summary['clean_model'].isin(target_models)].copy()
    
    if data.empty:
        print("Warning: No data for selected models. Skipping special chart.")
        return
    
    # Calculate Benchmarks (Mean of these 5 models for Global, Cluster, and Local)
    global_mean = data[data['final_mode'] == 'global'][metric].mean()
    cluster_mean = data[data['final_mode'] == 'cluster'][metric].mean()
    local_mean = data[data['final_mode'] == 'local'][metric].mean()
    
    # Get local values for bar chart
    local_data = data[data['final_mode'] == 'local'].sort_values(metric, ascending=False)
    
    if local_data.empty:
        print("Warning: No data for 'local' mode with selected models.")
        return

    plt.figure(figsize=(10, 8)) # Increased height for legend
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    
    # Palette: Red for LOC-NFST, fading Blues for baselines based on rank
    sorted_models = local_data['clean_model'].tolist()
    bs_models = [m for m in sorted_models if m != 'LOC-NFST']
    blues = sns.color_palette("Blues_r", n_colors=len(bs_models) + 1)
    
    palette = {}
    for m in sorted_models:
        if m == 'LOC-NFST':
            palette[m] = COLOR_PALETTE['LOC-NFST']
        else:
            palette[m] = blues[bs_models.index(m)]
            
    ax = sns.barplot(
        data=local_data,
        x='clean_model',
        y=metric,
        hue='clean_model',
        palette=palette,
        edgecolor='none',
        alpha=0.9,
        legend=False,
        width=0.55
    )
    
    # REMOVED HORIZONTAL AVERAGE LINES AS REQUESTED

    # Styling
    plt.xlabel('Model', fontsize=FONT_SIZE_AXES, fontweight='bold')
    plt.ylabel('AUC-ROC (%)' if metric == 'aucroc' else 'AUC-PR (%)', fontsize=FONT_SIZE_AXES, fontweight='bold')
    plt.xticks(fontsize=FONT_SIZE_TICKS, fontweight='bold')
    
    # Y-ticks every 2.5% steps
    y_min_val = max(0, local_data[metric].min() - 10)
    y_start = np.floor(y_min_val / 2.5) * 2.5
    plt.yticks(np.arange(y_start, 102.5, 2.5), fontsize=FONT_SIZE_TICKS)
    
    # labels on bars - Inside column, larger font, white color for contrast
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=-65, fontsize=26, fontweight='bold', 
                     rotation=90, label_type='edge', color='white',
                     zorder=20)
        
    if ax.get_legend(): ax.get_legend().remove()
    
    # Horizontal grid lines every 2.5%
    plt.grid(axis='y', linestyle='--', alpha=0.7, color='gray')
    
    plt.ylim(y_start, 105) 
    plt.tight_layout()
    
    save_path = os.path.join(_script_dir, "plots", "Special_Anomaly_Comparison.png")
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    print(f"Generated Special Anomaly Chart: {save_path}")
    plt.close()

def plot_anomaly_difference_comparison(merged):
    """
    Generate a bar chart showing the difference in AUC-ROC:
    Local vs Global, and Local vs Cluster.
    """
    if merged is None or merged.empty: return

    if EXCLUDE_MODELS:
        merged = merged[~merged['clean_model'].isin(EXCLUDE_MODELS)]
        
    if PROPOSED_SCALER:
        merged = merged[(merged['clean_model'] != 'LOC-NFST') | (merged['scaler'] == PROPOSED_SCALER)]
        
    if BASELINE_SCALER:
        merged = merged[(merged['clean_model'] == 'LOC-NFST') | (merged['scaled'] == BASELINE_SCALER)]
        
    metric = METRIC_NAME.lower()
    if metric not in merged.columns: metric = 'aucroc'
    
    best_results = merged.groupby(['final_mode', 'simple_dataset', 'clean_model'])[metric].max().reset_index()
    summary = best_results.groupby(['final_mode', 'clean_model'])[metric].mean().reset_index()

    target_models = ['LOC-NFST', 'AUTOENCODER', 'DASVDD', 'KNN', 'LUNAR']
    data = summary[summary['clean_model'].isin(target_models)].copy()
    
    if data.empty: return

    # Pivot to get modes
    pivot_diff = data.pivot(index='clean_model', columns='final_mode', values=metric)
    
    for c in ['local', 'global', 'cluster']:
        if c not in pivot_diff.columns:
            print(f"Missing mode '{c}' for difference chart.")
            return

    pivot_diff['Local - Global'] = pivot_diff['local'] - pivot_diff['global']
    pivot_diff['Local - Cluster'] = pivot_diff['local'] - pivot_diff['cluster']
    
    # Sort LOC-NFST first, then the rest
    sort_models = ['LOC-NFST'] + [m for m in target_models if m in pivot_diff.index and m != 'LOC-NFST']
    pivot_diff = pivot_diff.loc[sort_models]
    
    diff_melt = pivot_diff.reset_index().melt(
        id_vars='clean_model',
        value_vars=['Local - Global', 'Local - Cluster'],
        var_name='Comparison', 
        value_name='Difference'
    )

    plt.figure(figsize=(10, 8))
    sns.set_style("whitegrid")
    
    ax = sns.barplot(
        data=diff_melt,
        x='clean_model',
        y='Difference',
        hue='Comparison',
        palette=['#cc3333', '#1f77b4'],
        edgecolor='white',
        linewidth=0.6,
        alpha=0.85
    )

    import matplotlib as mpl
    mpl.rcParams['hatch.linewidth'] = 0.7
    HATCH_PATTERNS  = ['/', '\\']
    HATCH_EDGECOLOR = ['#8b0000', '#1a3a55']
    for ci, container in enumerate(ax.containers):
        hatch = HATCH_PATTERNS[ci % len(HATCH_PATTERNS)]
        ec    = HATCH_EDGECOLOR[ci % len(HATCH_EDGECOLOR)]
        for bar in container:
            bar.set_hatch(hatch)
            bar.set_edgecolor(ec)
    
    plt.xlabel('Model', fontsize=FONT_SIZE_AXES, fontweight='bold')
    plt.ylabel('Difference Metrics (%)', fontsize=FONT_SIZE_AXES, fontweight='bold')
    plt.xticks(fontsize=FONT_SIZE_TICKS, fontweight='bold')
    
    # Y-ticks every 2.5% steps
    y_min, y_max = ax.get_ylim()
    y_min_val = np.floor(y_min / 2.5) * 2.5
    y_max_val = np.ceil(y_max / 2.5) * 2.5
    plt.yticks(np.arange(y_min_val, y_max_val + 2.5, 2.5), fontsize=FONT_SIZE_TICKS)
    
    for container in ax.containers:
        # write values horizontally
        ax.bar_label(container, fmt='%+.1f', padding=5, fontsize=11, fontweight='bold', 
                     label_type='edge',
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1),
                     zorder=20)
                     
    plt.grid(axis='y', linestyle='--', alpha=0.7, color='gray')
    
    plt.axhline(y=0, color='black', linewidth=1.5, zorder=1) # Zero line indicator 
    
    plt.legend(title='Comparison', fontsize=11, title_fontsize=12, loc='best', frameon=True, shadow=True)
    
    # expand limit a bit more to fit the vertical labels properly
    plt.ylim(y_min_val - 2.5, y_max_val + 3.0)

    plt.tight_layout()
    save_path = os.path.join(_script_dir, "plots", "Special_Anomaly_Difference.png")
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    print(f"Generated Special Difference Chart: {save_path}")
    plt.close()



def plot_anomaly_type(summary, mode):
    """Generate professional scientific bar chart for a specific anomaly mode (Standalone version)."""
    fig, ax = plt.subplots(figsize=(10, 7))
    plot_anomaly_type_to_ax(summary, mode, ax)
    
    plt.tight_layout()
    save_path = os.path.join(_script_dir, "plots", f"Anomaly_Type_{mode.capitalize()}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=DPI)
    print(f"Generated Standalone: {save_path}")
    plt.close()

def generate_anomaly_bump_charts(dfs, output_dir):
    print(f"Aggregating {len(dfs)} results for Anomaly Bump Charts...")
    merged = pd.concat(dfs, ignore_index=True)
    
    # Consolidate Mode
    mode_col = next((c for c in merged.columns if 'mode' in c and c != 'inferred_mode'), 'inferred_mode')
    merged['final_mode'] = merged[mode_col].str.lower()
    
    # Consolidate Dataset
    def simplify_ds(x):
        lower_x = str(x).lower()
        if TARGET_DATASET:
            for target in TARGET_DATASET:
                if target.lower() in lower_x:
                    return target
        if 'ciciot' in lower_x: return 'CICIoT'
        if 'botiot' in lower_x: return 'BoTIoT'
        if 'n_baiot' in lower_x: return 'N_BaIoT'
        if 'toniot' in lower_x: return 'ToNIoT'
        return os.path.basename(str(x))
    
    ds_col = next((c for c in merged.columns if 'dataset' in c), 'dataset')
    merged['simple_dataset'] = merged[ds_col].apply(simplify_ds) if ds_col in merged.columns else 'Unknown'
    
    # Filter for target modes and scale aucroc to 100
    modes = ['global', 'local', 'cluster']
    merged = merged[merged['final_mode'].isin(modes)].copy()
    if merged['aucroc'].mean() < 1.0:
        merged['aucroc'] = merged['aucroc'] * 100.0
    
    # FILTER FOR TARGET DATASET
    if TARGET_DATASET:
        print(f"Filtering for datasets (substring): {TARGET_DATASET}...")
        merged = merged[merged['simple_dataset'].str.contains('|'.join(TARGET_DATASET), case=False, na=False)]
    
    # Filters (Excluded models & Scalers)
    if EXCLUDE_MODELS:
        merged = merged[~merged['clean_model'].isin([m.upper() for m in EXCLUDE_MODELS])].copy()
    
    is_ours = merged['clean_model'] == 'LOC-NFST'
    if PROPOSED_SCALER:
        merged = merged[~is_ours | (merged['scaler'] == PROPOSED_SCALER)]
    if BASELINE_SCALER:
        merged = merged[is_ours | (merged['scaled'] == BASELINE_SCALER)]
        
    # Best configuration per (mode, dataset, model)
    best_df = merged.groupby(['final_mode', 'simple_dataset', 'clean_model'])['aucroc'].max().reset_index()
    
    datasets = sorted(best_df['simple_dataset'].unique().tolist())
    if 'Unknown' in datasets: datasets.remove('Unknown')
    
    # Calculate Ranks
    for ds in datasets:
        for m in modes:
            idx = (best_df['simple_dataset'] == ds) & (best_df['final_mode'] == m)
            if idx.any():
                best_df.loc[idx, 'rank'] = best_df.loc[idx, 'aucroc'].rank(ascending=False, method='min')
                
    # Aggregate Items
    plot_items = []
    for ds in datasets:
        ds_df = best_df[best_df['simple_dataset'] == ds].copy()
        ds_df['plot_rank'] = ds_df['rank']
        plot_items.append((ds, ds_df))
        
    # Summary Aggregates
    agg_df = best_df.groupby(['clean_model', 'final_mode'])[['rank', 'aucroc']].mean().reset_index()
    
    # Global Mean Rank Chart
    rank_agg = agg_df.copy()
    rank_agg['plot_rank'] = rank_agg['rank']
    plot_items.append(('Global_Mean_Rank', rank_agg))
    
    # Global Mean AUC Chart (Positioned by the rank of the Mean AUCs)
    auc_agg = agg_df.copy()
    for m in modes:
        idx = auc_agg['final_mode'] == m
        if idx.any():
            auc_agg.loc[idx, 'plot_rank'] = auc_agg.loc[idx, 'aucroc'].rank(ascending=False, method='min')
    plot_items.append(('Global_Mean_AUC', auc_agg))
    
    print(f"Generating {len(plot_items)} Anomaly Bump Charts...")
    all_latex_tables = []
    
    for label, ds_data in plot_items:
        if 'plot_rank' not in ds_data.columns or ds_data.empty: continue
        
        pivot_rank = ds_data.pivot(index='clean_model', columns='final_mode', values='plot_rank')
        pivot_auc = ds_data.pivot(index='clean_model', columns='final_mode', values='aucroc')
        
        # Order modes correctly
        plot_modes = [m for m in modes if m in pivot_rank.columns]
        pivot_rank = pivot_rank.reindex(columns=plot_modes).dropna(how='all')
        pivot_auc = pivot_auc.reindex(columns=plot_modes).dropna(how='all')
        
        # Custom sort
        avg_p = pivot_auc.mean(axis=1)
        models_sorted = avg_p.sort_values(ascending=False).index.tolist()
        if 'LOC-NFST' in models_sorted:
            models_sorted.remove('LOC-NFST')
            models_sorted = ['LOC-NFST'] + models_sorted
            
        plt.figure(figsize=(12, 11))
        colors = plt.cm.tab20(np.linspace(0, 1, max(20, len(models_sorted))))
        model_colors = {m: c for m, c in zip(models_sorted, colors)}
        if 'LOC-NFST' in model_colors: model_colors['LOC-NFST'] = 'red'
            
        for model in models_sorted:
            ranks = pivot_rank.loc[model].values
            line_w = 4.5 if model == 'LOC-NFST' else 2.0
            alpha = 1.0 if model == 'LOC-NFST' else 0.5
            zorder = 10 if model == 'LOC-NFST' else 2
            plt.plot(plot_modes, ranks, '-', color=model_colors[model], 
                     linewidth=line_w, alpha=alpha, zorder=zorder)

        # DECONFLICTION: Group models by rank at each Anomaly Mode level to avoid label overlap
        for i, m_type in enumerate(plot_modes):
            rank_to_models = {}
            for model in models_sorted:
                r = pivot_rank.loc[model, m_type]
                if not pd.isna(r):
                    if r not in rank_to_models: rank_to_models[r] = []
                    rank_to_models[r].append(model)
            
            for r, tied_models in rank_to_models.items():
                n_ties = len(tied_models)
                # Symmetrical staggering around the true rank 'r'
                v_pad = 0.35 
                start_y = r - (n_ties - 1) * v_pad / 2
                
                for idx, model in enumerate(tied_models):
                    staggered_y = start_y + idx * v_pad
                    auc_val = pivot_auc.loc[model, m_type]
                    label_txt = f"{r:.1f}" if label == "Global_Mean_Rank" else f"{auc_val:.1f}"
                    
                    if model == 'LOC-NFST':
                        plt.text(i, staggered_y, label_txt, ha='center', va='center',
                                 fontsize=10, fontweight='bold', color='white',
                                 bbox=dict(facecolor='red', alpha=1.0, edgecolor='black', boxstyle='circle,pad=0.2'),
                                 zorder=15)
                    else:
                        plt.text(i, staggered_y, label_txt, ha='center', va='center',
                                 fontsize=8, fontweight='bold', color='black',
                                 bbox=dict(facecolor='white', alpha=0.9, edgecolor=model_colors[model], boxstyle='round,pad=0.2'),
                                 zorder=8)

        plt.gca().invert_yaxis()
        plt.gca().yaxis.tick_right()
        plt.gca().yaxis.set_label_position("right")
        
        max_rank = int(pivot_rank.max().max())
        plt.yticks(range(1, max_rank + 1), fontsize=10, fontweight='bold')
        
        # DECONFLICTION: Models on left with staggering for ties
        rank_to_models_left = {}
        for model in models_sorted:
            r = pivot_rank.loc[model, plot_modes[0]]
            if not pd.isna(r):
                if r not in rank_to_models_left: rank_to_models_left[r] = []
                rank_to_models_left[r].append(model)
        
        for r, tied_models in rank_to_models_left.items():
            n_ties = len(tied_models)
            v_pad = 0.32
            start_y = r - (n_ties - 1) * v_pad / 2
            for idx, model in enumerate(tied_models):
                staggered_y = start_y + idx * v_pad
                plt.text(-0.15, staggered_y, model, ha='right', va='center', 
                         fontsize=9, fontweight='bold', color=model_colors[model])

        plt.xlim(-0.8, len(plot_modes)-1 + 0.5)
        
        title_map = {
            "Global_Mean_Rank": "Global Summary: Mean Rank Across Anomaly Types",
            "Global_Mean_AUC": "Global Summary: Mean AUC Across Anomaly Types"
        }
        title_prefix = title_map.get(label, f"Dataset: {label}")
        ylabel = "Mean Rank" if "Rank" in label else "Rank (of Mean AUC)" if "AUC" in label else "Rank (Relative)"
        
        plt.title(f'Rank Flow and Anomaly Metrics (AUC-ROC)\n{title_prefix}', 
                  fontsize=16, fontweight='bold', pad=25)
        plt.xlabel('Anomaly Type', fontsize=13, fontweight='bold')
        plt.ylabel(ylabel, fontsize=13, fontweight='bold')
        plt.xticks(range(len(plot_modes)), [m.capitalize() for m in plot_modes], fontsize=11, fontweight='bold')
        
        plt.legend(title='Model Hierarchy', title_fontsize=12, bbox_to_anchor=(1.15, 1), 
                   loc='upper left', borderaxespad=0., fontsize=10, frameon=True, shadow=True)
        
        plt.grid(True, axis='y', linestyle='--', alpha=0.4, zorder=0)
        plt.tight_layout()
        
        save_path = os.path.join(_script_dir, "plots", f"Anomaly_Bump_{label}.png")
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"  -> Saved plot: {save_path}")

        # LATEX TABLE
        ltx = "\\begin{table}[h]\n\\centering\n"
        ltx += f"\\caption{{Anomaly Type Robustness: {label.replace('_', ' ')} (AUC-ROC and Rank)}}\n"
        ltx += f"\\label{{tab:anomaly_bump_{label.lower()}}}\n"
        ltx += "\\adjustbox{width=\\columnwidth}{\n"
        ltx += "\\begin{tabular}{lcccc}\n\\toprule\n"
        ltx += "\\textbf{Model} & " + " & ".join([f"\\textbf{{{m.capitalize()}}}" for m in plot_modes]) + " \\\\\n\\midrule\n"
        
        for m in models_sorted:
            row = [f"\\textbf{{{m}}}" if m == 'LOC-NFST' else m]
            for m_type in plot_modes:
                c_data = ds_data[(ds_data['clean_model'] == m) & (ds_data['final_mode'] == m_type)]
                if c_data.empty:
                    row.append("-")
                else:
                    auc_val = c_data['aucroc'].values[0]
                    if label == "Global_Mean_Rank":
                        row.append(f"{c_data['plot_rank'].values[0]:.2f}")
                    else:
                        r_actual = int(c_data['plot_rank'].values[0])
                        row.append(f"{auc_val:.2f} ({r_actual})")
            ltx += " & ".join(row) + " \\\\\n"
        ltx += "\\bottomrule\n\\end{tabular}\n}\n\\end{table}\n"
        all_latex_tables.append(ltx)

    ltx_out = os.path.join(_script_dir, "plots", "anomaly_results_tables.tex")
    with open(ltx_out, "w") as f: f.write("\n\n".join(all_latex_tables))
    print(f"  -> Saved LaTeX: {ltx_out}")

if __name__ == "__main__":
    import sys
    print("=== Robust Anomaly Type Aggregator & Plotter ===")
    
    # Check for CLI mode
    if len(sys.argv) > 1 and sys.argv[1] == 'bump':
        print("Running in Bump Chart Mode...")
        all_dfs = find_all_results()
        if not all_dfs:
            print("Fatal: No results found.")
        else:
            generate_anomaly_bump_charts(all_dfs, os.path.join(_script_dir, "plots"))
    else:
        all_dfs = find_all_results()
        if not all_dfs:
            print("Fatal: No results found in search paths.")
        else:
            summary, merged_data = aggregate_metrics(all_dfs)
            if not summary.empty:
                # 1. New Grouped Summary View (Matches Ref Image Style)
                print("\nGenerating Grouped Anomaly Summary Chart...")
                plot_anomaly_types_grouped_summary(merged_data)
                
                # 2. Combined Subplot View
                print("\nGenerating Combined Anomaly Types Subplot...")
                plot_anomaly_types_combined(summary)
                
                # 3. Individual Plots (Standalones)
                for mode in ANOMALY_MODES:
                    plot_anomaly_type(summary, mode)
                    
                # 4. Special Comparison Chart (Request 03/04)
                print("\nGenerating Special Anomaly Comparison Chart...")
                plot_special_anomaly_comparison(merged_data)
                
                print("\nGenerating Special Anomaly Difference Chart...")
                plot_anomaly_difference_comparison(merged_data)
                    
                # 5. Create Composite Summary Table (CSV)
                print("\nCreating Composite Anomaly Type Table (Detailed)...")
                composite_table = create_composite_table(merged_data)
        
    print("\nProcessing complete.")
