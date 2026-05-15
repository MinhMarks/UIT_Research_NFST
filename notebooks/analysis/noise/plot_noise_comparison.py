import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# Standard font settings
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Helvetica', 'sans-serif']

KNOWN_DATASETS = ['BoTIoT', 'ToNIoT', 'N_BaIoT', 'CICIoT']

# CONFIGURATION
EXCLUDE_MODELS_LIST = ['DEVNET'] # Models to globally exclude
BASELINE_SCALER = 'QuantileTransformer' # Force baselines to use this scaler. Set to None to use all.

# List of specific models to plot. If empty, it defaults to top 4 baselines.
TARGET_MODELS_LIST = ['LOC-NFST', 'AUTOENCODER', 'DASVDD', 'SUOD', 'PCA', 'KNN', 'ECOD', 'LUNAR', 'LOF']



def normalize_dataset_name(name):
    if not isinstance(name, str): return str(name)
    # Remove common noise/data artifacts for cleaner matching
    name_lower = name.lower().replace('_', '').replace('-', '')
    for ds in KNOWN_DATASETS:
        ds_clean = ds.lower().replace('_', '').replace('-', '')
        if ds_clean in name_lower: return ds
    base = os.path.basename(name)
    if base.endswith('.csv'): base = base[:-4]
    return base

def find_csv_files(root_dir):
    csv_files = glob.glob(os.path.join(root_dir, "**", "*.csv"), recursive=True)
    results = []
    for f in csv_files:
        try:
            if os.path.getsize(f) < 100: continue
            df = pd.read_csv(f)
            cols = [c.lower() for c in df.columns]
            if 'aucroc' in cols and ('model' in cols or 'dataset' in cols):
                results.append(('unified', f))
        except Exception: pass
    return results

def load_and_normalize(file_info):
    _, file_path = file_info
    df = pd.read_csv(file_path)
    df.columns = [c.lower() for c in df.columns]
    
    norm_df = pd.DataFrame()
    if 'dataset' in df.columns: norm_df['dataset'] = df['dataset'].apply(normalize_dataset_name)
    elif 'data' in df.columns: norm_df['dataset'] = df['data'].apply(normalize_dataset_name)
    else: norm_df['dataset'] = normalize_dataset_name(file_path)
    
    df_models = df['model'] if 'model' in df.columns else pd.Series(['LOC-NFST']*len(df))
    norm_df['model'] = df_models.apply(lambda x: str(x).upper())
    norm_df['model'] = norm_df['model'].replace({'OURMODEL': 'LOC-NFST', 'XXX': 'LOC-NFST'})
    
    # Extract Scaler if available
    norm_df['scaler'] = df['scaler'] if 'scaler' in df.columns else (df['scaled'] if 'scaled' in df.columns else 'Unknown')
    
    # Global Model Exclusion
    norm_df = norm_df[~norm_df['model'].str.upper().isin([m.upper() for m in EXCLUDE_MODELS_LIST])].copy()
    
    if 'noise_percentage' in df.columns: norm_df['noise'] = df['noise_percentage'].astype(float)
    elif 'noise' in df.columns: norm_df['noise'] = df['noise'].astype(float)
    else: norm_df['noise'] = 0.0

    # Fallback to extracting from filename since some files are named report_noise_1.csv without a noise column
    if norm_df['noise'].mean() == 0.0:
        if 'noise_1' in file_path.lower() or 'noise1' in file_path.lower(): norm_df['noise'] = 1.0
        elif 'noise_3' in file_path.lower() or 'noise3' in file_path.lower(): norm_df['noise'] = 3.0
        elif 'noise_5' in file_path.lower() or 'noise5' in file_path.lower(): norm_df['noise'] = 5.0

    norm_df['aucroc'] = pd.to_numeric(df['aucroc'], errors='coerce')
    if 'aucroc' in norm_df.columns:
        # Scale to 100 if it's 0-1
        if norm_df['aucroc'].mean() < 1.0:
            norm_df['aucroc'] = norm_df['aucroc'] * 100.0
            
    return norm_df.dropna(subset=['aucroc'])


def generate_noise_comparison_chart(files, output_path):
    print(f"Found {len(files)} CSV files")
    
    all_data = []
    for info in files:
        df = load_and_normalize(info)
        all_data.append(df)
        
    if not all_data:
        print("No valid data found.")
        return
        
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # APPLY BASELINE SCALER FILTERING (Relaxed to allow Unknown)
    if BASELINE_SCALER:
        is_ours = combined_df['model'] == 'LOC-NFST'
        valid_baselines = combined_df['scaler'].isin([BASELINE_SCALER, 'Unknown'])
        combined_df = combined_df[is_ours | valid_baselines]

    print(f"Models passing scaler filter: {combined_df['model'].unique()}")

    combined_df = combined_df[combined_df['noise'].isin([1.0, 3.0, 5.0])]
    print(f"Loaded {len(combined_df)} rows for noise 1%, 3%, 5%.")
    
    if len(combined_df) == 0:
        print("No noise data available for 1%, 3%, or 5%. Exiting plot generation.")
        return

    # 1. First find the BEST configuration (Max AUCROC) for each model on each dataset at each noise level
    best_per_dataset = combined_df.groupby(['model', 'dataset', 'noise'])['aucroc'].max().reset_index()
    
    # 2. Average AUCROC for each model / noise level across the datasets
    avg_df = best_per_dataset.groupby(['model', 'noise'])['aucroc'].mean().reset_index()
    overall_avg = avg_df.groupby('model')['aucroc'].mean().reset_index()
    existing_models = overall_avg['model'].unique()
    
    if TARGET_MODELS_LIST:
        target_models = [m.upper() for m in TARGET_MODELS_LIST]
        if 'LOC-NFST' in target_models: target_models.remove('LOC-NFST')
        target_models = ['LOC-NFST'] + target_models
        target_models = [m for m in target_models if m in existing_models]
    else:
        baselines = overall_avg[overall_avg['model'] != 'LOC-NFST']
        top_4_baselines = baselines.sort_values(by='aucroc', ascending=False).head(4)['model'].tolist()
        target_models = ['LOC-NFST'] + top_4_baselines
        
    print(f"Plotting for Target Models: {target_models}")
    
    # =========================================================================
    # NEW FEATURE: Generate full LaTeX Table before dataset-wide averaging
    # =========================================================================
    print("Generating comprehensive LaTeX table for noise performance...")
    ds_order = ['BoTIoT', 'CICIoT', 'N_BaIoT', 'ToNIoT']
    noise_levels = [1.0, 3.0, 5.0]
    
    # Prepare filtered dataframe
    table_df = best_per_dataset[best_per_dataset['dataset'].isin(ds_order) & 
                                best_per_dataset['noise'].isin(noise_levels)].copy()
    
    # Identify top 2 models per dataset+noise for coloring
    color_map = {} 
    for ds in ds_order:
        for n in noise_levels:
            slice_df = table_df[(table_df['dataset'] == ds) & (table_df['noise'] == n)]
            if len(slice_df) > 0:
                ranked = slice_df.sort_values(by='aucroc', ascending=False)['model'].tolist()
                if len(ranked) >= 1: color_map[(ranked[0], ds, n)] = 'red'
                if len(ranked) >= 2: color_map[(ranked[1], ds, n)] = 'blue'

    # Build LaTeX string
    latex_str = "\\begin{table*}[!ht]\n"
    latex_str += "\\caption{Model performance across different noise levels (1\\%, 3\\%, 5\\%) on various IoT datasets.}\n"
    latex_str += "\\centering\n"
    latex_str += "\\resizebox{\\textwidth}{!}{%\n"
    latex_str += "\\begin{tabular}{|c|ccc|ccc|ccc|ccc|}\n\\hline\n"
    latex_str += "\\textbf{Model} & \\multicolumn{3}{c|}{\\textbf{BIoT}} & \\multicolumn{3}{c|}{\\textbf{CICIoT2023}} & \\multicolumn{3}{c|}{\\textbf{NBaIoT}} & \\multicolumn{3}{c|}{\\textbf{ToNIoT}} \\\\\n"
    latex_str += "\\cline{2-13}\n"
    latex_str += " & 1\\% & 3\\% & 5\\% & 1\\% & 3\\% & 5\\% & 1\\% & 3\\% & 5\\% & 1\\% & 3\\% & 5\\% \\\\\n\\hline\n"
    
    # Do not limit output table to target_models, print all available
    # Optional: order logic putting LOC-NFST first then by overall avg
    all_table_models = table_df.groupby('model')['aucroc'].mean().sort_values(ascending=False).index.tolist()
    if 'LOC-NFST' in all_table_models:
        all_table_models.remove('LOC-NFST')
        all_table_models = ['LOC-NFST'] + all_table_models
        
    for m in all_table_models:
        # Custom mapping for naming
        if str(m).upper() == "LOC-NFST": row_str = "OurModel"
        elif str(m).upper() == "SO_GAAL": row_str = "SO\\_GAAL"
        elif str(m).upper() == "MO_GAAL": row_str = "MO\\_GAAL"
        else: row_str = str(m)
        
        for ds in ds_order:
            for n in noise_levels:
                val_series = table_df[(table_df['model'] == m) & (table_df['dataset'] == ds) & (table_df['noise'] == n)]['aucroc']
                if len(val_series) > 0:
                    val_str = f"{val_series.values[0]:.2f}"
                    col = color_map.get((m, ds, n))
                    if col == 'red': val_str = f"\\textcolor{{red}}{{{val_str}}}"
                    elif col == 'blue': val_str = f"\\textcolor{{blue}}{{{val_str}}}"
                    row_str += f" & {val_str}"
                else:
                    row_str += " & -"
        latex_str += row_str + " \\\\\n"
    
    latex_str += "\\hline\n\\end{tabular}\n}\n"
    latex_str += "\\label{tab:model_noise_perf}\n\\end{table*}\n"
    
    # Save the file
    out_dir = os.path.dirname(output_path)
    if not out_dir: out_dir = "."
    table_path = os.path.join(out_dir, "Generated_Noise_CD_Table.tex")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(latex_str)
    print(f"--> Saved LaTeX raw table to: {table_path}\n")
    # =========================================================================
    
    plot_df = avg_df[avg_df['model'].isin(target_models)].copy()
    num_models = len(target_models)
    
    if len(plot_df) == 0:
        print("No data available for the target models.")
        return

    # Dynamic figure calculation to prevent columns from becoming too small
    fig_width = max(10, 3 * num_models * 0.35 + 2)
    plt.figure(figsize=(fig_width, 6))
    
    # Sort models so LOC-NFST is first
    plot_df['model'] = pd.Categorical(plot_df['model'], categories=target_models, ordered=True)
    plot_df = plot_df.sort_values(['noise', 'model'])
    
    # Palette configuration: Red for Proposed, Blue for all Baselines
    palette = {m: ('#cc3333' if m == 'LOC-NFST' else '#1f77b4') for m in plot_df['model'].unique()}
    
    ax = sns.barplot(
        data=plot_df,
        x='noise',
        y='aucroc',
        hue='model',
        palette=palette,
        edgecolor='black',
        linewidth=1.2
    )

    y_min = plot_df['aucroc'].min()
    y_max = plot_df['aucroc'].max()
    
    # Add a buffer
    ymin_buf = max(0, y_min - (y_max - y_min)*0.3)
    ymax_buf = min(100, y_max + (y_max - y_min)*0.2)
    plt.ylim(ymin_buf, ymax_buf)

    plt.xlabel('Noise Level (%)', fontsize=14, fontweight='bold')
    plt.ylabel('AUC-ROC (%)', fontsize=14, fontweight='bold')
    plt.title('Performance Comparison Across Noise Levels', fontsize=16, fontweight='bold', pad=20)
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12)
    
    # Annotate bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=3, fontsize=10, fontweight='bold')

    # Legend adjustment to prevent clipping
    legend_ncol = min(6, num_models)
    legend_rows = (num_models + legend_ncol - 1) // legend_ncol
    
    plt.legend(title='Model', title_fontsize='13', fontsize='12', loc='upper center', 
               bbox_to_anchor=(0.5, -0.15), ncol=legend_ncol, frameon=True, borderaxespad=0.)

    import matplotlib.ticker as ticker
    ax = plt.gca()
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5.0))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(2.5))
    plt.grid(True, which='major', axis='y', linestyle='--', linewidth=0.8, alpha=0.7)
    plt.grid(True, which='minor', axis='y', linestyle=':', linewidth=0.5, alpha=0.4)

    # Adjust layout to make room for legend
    plt.subplots_adjust(bottom=0.2 + (legend_rows * 0.05))

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved grouped noise comparison chart to {output_path}")

def generate_separate_noise_charts(files, output_dir):
    print(f"Found {len(files)} CSV files for separate mode")
    
    all_data = []
    for info in files:
        df = load_and_normalize(info)
        all_data.append(df)
        
    if not all_data:
        print("No valid data found.")
        return
        
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # APPLY BASELINE SCALER FILTERING
    if BASELINE_SCALER:
        print(f"Filtering baselines for scaler: {BASELINE_SCALER}...")
        is_ours = combined_df['model'] == 'LOC-NFST'
        combined_df = combined_df[is_ours | (combined_df['scaler'] == BASELINE_SCALER)]

    combined_df = combined_df[combined_df['noise'].isin([1.0, 3.0, 5.0])]
    print(f"Loaded {len(combined_df)} rows for noise 1%, 3%, 5%.")
    
    if len(combined_df) == 0:
        print("No noise data available for 1%, 3%, or 5%. Exiting plot generation.")
        return

    # 1. BEST configuration (Max AUCROC) for each model on each dataset at each noise level
    best_per_dataset = combined_df.groupby(['model', 'dataset', 'noise'])['aucroc'].max().reset_index()
    
    datasets = best_per_dataset['dataset'].unique()
    print(f"Generating charts for {len(datasets)} datasets: {datasets}")
    
    for ds in datasets:
        ds_df = best_per_dataset[best_per_dataset['dataset'] == ds].copy()
        
        # Rank models specifically for THIS dataset by averaging across the noise levels
        avg_df = ds_df.groupby('model')['aucroc'].mean().reset_index()
        existing_models_ds = avg_df['model'].unique()

        if TARGET_MODELS_LIST:
            target_models = [m.upper() for m in TARGET_MODELS_LIST]
            if 'LOC-NFST' in target_models: target_models.remove('LOC-NFST')
            target_models = ['LOC-NFST'] + target_models
            target_models = [m for m in target_models if m in existing_models_ds]
        else:
            baselines = avg_df[avg_df['model'] != 'LOC-NFST']
            top_4_baselines = baselines.sort_values(by='aucroc', ascending=False).head(4)['model'].tolist()
            target_models = ['LOC-NFST'] + top_4_baselines
            
        print(f"Dataset {ds} Top Models: {target_models}")
        
        plot_df = ds_df[ds_df['model'].isin(target_models)].copy()
        num_models = len(target_models)
        if len(plot_df) == 0: continue
            
        fig_width = max(9, 3 * num_models * 0.35 + 2)
        plt.figure(figsize=(fig_width, 5.5))
        plot_df['model'] = pd.Categorical(plot_df['model'], categories=target_models, ordered=True)
        plot_df = plot_df.sort_values(['noise', 'model'])
        
        # Unified palette for separate charts
        palette = {m: ('#cc3333' if m == 'LOC-NFST' else '#1f77b4') for m in plot_df['model'].unique()}
        
        ax = sns.barplot(
            data=plot_df, x='noise', y='aucroc', hue='model',
            palette=palette, edgecolor='black', linewidth=1.2
        )

        y_min, y_max = plot_df['aucroc'].min(), plot_df['aucroc'].max()
        plt.ylim(max(0, y_min - (y_max - y_min)*0.3), min(100, y_max + (y_max - y_min)*0.2))

        plt.xlabel('Noise Level (%)', fontsize=12, fontweight='bold')
        plt.ylabel('AUC-ROC (%)', fontsize=12, fontweight='bold')
        plt.title(f'Performance Comparison Across Noise Levels - {ds}', fontsize=14, fontweight='bold', pad=15)
        plt.xticks(fontsize=11, fontweight='bold')
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f', padding=3, fontsize=9, fontweight='bold')

        legend_ncol = min(6, num_models)
        legend_rows = (num_models + legend_ncol - 1) // legend_ncol
        plt.legend(title='Model', title_fontsize='11', fontsize='10', loc='upper center', 
                   bbox_to_anchor=(0.5, -0.15), ncol=legend_ncol, frameon=True, borderaxespad=0.)

        import matplotlib.ticker as ticker
        ax = plt.gca()
        ax.yaxis.set_major_locator(ticker.MultipleLocator(5.0))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(2.5))
        plt.grid(True, which='major', axis='y', linestyle='--', linewidth=0.8, alpha=0.7)
        plt.grid(True, which='minor', axis='y', linestyle=':', linewidth=0.5, alpha=0.4)

        plt.subplots_adjust(bottom=0.2 + (legend_rows * 0.05))

        out_path = os.path.join(output_dir, f"noise_levels_{ds}.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  -> Saved {out_path}")

def generate_noise_bump_charts(files, output_dir):
    print(f"Found {len(files)} CSV files for bump chart mode")
    all_data = []
    for info in files:
        df = load_and_normalize(info)
        all_data.append(df)
        
    if not all_data:
        print("No valid data found.")
        return
        
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # APPLY BASELINE SCALER FILTERING
    if BASELINE_SCALER:
        print(f"Filtering baselines for scaler: {BASELINE_SCALER}...")
        is_ours = combined_df['model'] == 'LOC-NFST'
        combined_df = combined_df[is_ours | (combined_df['scaler'] == BASELINE_SCALER)]

    if TARGET_MODELS_LIST:
        target_upper = [m.upper() for m in TARGET_MODELS_LIST]
        if 'LOC-NFST' not in target_upper: target_upper.append('LOC-NFST')
        combined_df = combined_df[combined_df['model'].isin(target_upper)]
        
    # Filter for target noise levels
    noises = [0.0, 1.0, 3.0, 5.0]
    combined_df['noise'] = combined_df['noise'].round(0)
    combined_df = combined_df[combined_df['noise'].isin(noises)]
    
    # 1. BEST configuration (Max AUCROC) for each model on each dataset at each noise level
    best_df = combined_df.groupby(['model', 'dataset', 'noise'])['aucroc'].max().reset_index()
    
    # Pre-calculate ranks within each dataset for cross-dataset aggregation
    datasets = best_df['dataset'].unique()
    noises = [0.0, 1.0, 3.0, 5.0]
    
    for ds in datasets:
        for n in noises:
            idx = (best_df['dataset'] == ds) & (best_df['noise'] == n)
            if idx.any():
                best_df.loc[idx, 'rank'] = best_df.loc[idx, 'aucroc'].rank(ascending=False, method='min')

    # Add "Global Mean" items to the list of things to plot
    plot_items = []
    for ds in datasets:
        ds_df = best_df[best_df['dataset'] == ds].copy()
        ds_df['plot_rank'] = ds_df['rank']
        plot_items.append((ds, ds_df))
    
    # Calculate aggregates across datasets
    agg_df = best_df.groupby(['model', 'noise'])[['rank', 'aucroc']].mean().reset_index()
    
    # Global Mean Rank Chart
    rank_agg = agg_df.copy()
    rank_agg['plot_rank'] = rank_agg['rank']
    plot_items.append(('Global_Mean_Rank', rank_agg))
    
    # Global Mean AUC Chart (Positioned by the rank of the Mean AUCs)
    auc_agg = agg_df.copy()
    for n in noises:
        idx = auc_agg['noise'] == n
        if idx.any():
            auc_agg.loc[idx, 'plot_rank'] = auc_agg.loc[idx, 'aucroc'].rank(ascending=False, method='min')
    plot_items.append(('Global_Mean_AUC', auc_agg))
    
    print(f"Generating bump charts & LaTeX for {len(plot_items)} groups...")
    all_latex_tables = []
    
    for label, ds_data in plot_items:
        # Pivot for plotting
        if 'plot_rank' not in ds_data.columns or ds_data.empty: continue
        
        pivot_rank = ds_data.pivot(index='model', columns='noise', values='plot_rank')
        pivot_auc = ds_data.pivot(index='model', columns='noise', values='aucroc')
        
        # Ensure all noises are present
        pivot_rank = pivot_rank.reindex(columns=noises)
        pivot_auc = pivot_auc.reindex(columns=noises)
        
        # Filter models that have data in at least one noise level
        pivot_rank = pivot_rank.dropna(how='all')
        if pivot_rank.empty: continue
        
        models_sorted = pivot_rank.index.tolist()
        # Custom sort: ROC-AUC descending on average, but LOC-NFST first
        avg_p = pivot_auc.mean(axis=1)
        models_sorted = avg_p.sort_values(ascending=False).index.tolist()
        if 'LOC-NFST' in models_sorted:
            models_sorted.remove('LOC-NFST')
            models_sorted = ['LOC-NFST'] + models_sorted
        
        # Adjusting aspect ratio to make it decisively much wider than it is tall (e.g., landscape layout for papers)
        fig_height = max(6, len(models_sorted) * 0.45)
        fig_width = 15
        plt.figure(figsize=(fig_width, fig_height))
        
        # Highlight configuration for bump charts
        highlight_cfg = {
            'LOC-NFST':    {'color': (0.6, 0.0, 0.0), 'alpha': 1.0, 'lw': 4.5, 'z': 20},
            'LUNAR':       {'color': (0.0, 0.0, 0.5), 'alpha': 1.0, 'lw': 3.5, 'z': 15},
            'LOF':         {'color': (0.4, 0.7, 1.0), 'alpha': 1.0, 'lw': 3.5, 'z': 15},
            'SUOD':        {'color': (0.0, 0.4, 0.0), 'alpha': 1.0, 'lw': 3.5, 'z': 15},
            'AUTOENCODER': {'color': (0.6, 0.9, 0.6), 'alpha': 1.0, 'lw': 3.5, 'z': 15}
        }
        
        # Base colors for non-highlighted models in bump chart
        # Using a muted HUSL to keep them distinct but subtle
        num_models = len(models_sorted)
        muted_palette = sns.husl_palette(num_models, l=0.7, s=0.4)
        
        # Pre-map colors for label consistency
        model_colors = {}
        for i, m in enumerate(models_sorted):
            m_upper = m.upper()
            if m_upper in highlight_cfg:
                model_colors[m] = highlight_cfg[m_upper]['color']
            else:
                model_colors[m] = muted_palette[i]
            
        for i, model in enumerate(models_sorted):
            m_upper = model.upper()
            is_highlight = m_upper in highlight_cfg
            
            cfg = highlight_cfg.get(m_upper, {'color': model_colors[model], 'alpha': 0.3, 'lw': 1.2, 'z': 1})
            
            ranks = pivot_rank.loc[model].values
            
            # All highlighted models use solid lines now as requested
            ls = '-' if is_highlight else '-'
            
            plt.plot(noises, ranks, ls, color=cfg['color'], 
                     linewidth=cfg['lw'], alpha=cfg['alpha'], zorder=cfg['z'])

        # DECONFLICTION: Group models by rank at each Noise level to avoid label overlap
        for noise in noises:
            rank_to_models = {}
            for model in models_sorted:
                r = pivot_rank.loc[model, noise]
                if not pd.isna(r):
                    if r not in rank_to_models: rank_to_models[r] = []
                    rank_to_models[r].append(model)
            
            for r, tied_models in rank_to_models.items():
                n_ties = len(tied_models)
                # Symmetrical staggering around the true rank 'r'
                v_pad = 0.28 
                start_y = r - (n_ties - 1) * v_pad / 2
                
                for idx, model in enumerate(tied_models):
                    staggered_y = start_y + idx * v_pad
                    auc_val = pivot_auc.loc[model, noise]
                    label_txt = f"{r:.1f}" if label == "Global_Mean_Rank" else f"{auc_val:.1f}"
                    
                    if model == 'LOC-NFST':
                        # Highlighted circle for proposed model (Higher zorder to be on top)
                        plt.text(noise, staggered_y, label_txt, ha='center', va='center',
                                 fontsize=11, fontweight='bold', color='white',
                                 bbox=dict(facecolor='red', alpha=1.0, edgecolor='black', boxstyle='circle,pad=0.15'),
                                 zorder=100)
                    else:
                        # Muted box for baselines (Higher zorder to prevent lines from cutting through)
                        f_weight = 'bold' if model.upper() in highlight_cfg else 'normal'
                        plt.text(noise, staggered_y, label_txt, ha='center', va='center',
                                 fontsize=9, fontweight=f_weight, color='black',
                                 bbox=dict(facecolor='white', alpha=1.0, edgecolor=model_colors[model], boxstyle='round,pad=0.15'),
                                 zorder=90)

        plt.gca().invert_yaxis()
        # Move Rank scale to the right side
        plt.gca().yaxis.tick_right()
        plt.gca().yaxis.set_label_position("right")

        # Set Y-ticks to integers only (Ranks)
        max_rank = int(pivot_rank.max().max())
        plt.yticks(range(1, max_rank + 1), fontsize=11, fontweight='bold')
        
        # DECONFLICTION: Label each model name on the far left margin with staggering for ties
        rank_to_models_left = {}
        for model in models_sorted:
            r = pivot_rank.loc[model, 0.0]
            if not pd.isna(r):
                if r not in rank_to_models_left: rank_to_models_left[r] = []
                rank_to_models_left[r].append(model)
        
        for r, tied_models in rank_to_models_left.items():
            n_ties = len(tied_models)
            v_pad = 0.26 # Slightly tighter for names
            start_y = r - (n_ties - 1) * v_pad / 2
            for idx, model in enumerate(tied_models):
                staggered_y = start_y + idx * v_pad
                f_weight = 'bold' if model.upper() in highlight_cfg else 'normal'
                plt.text(-0.25, staggered_y, model, ha='right', va='center', 
                         fontsize=10, fontweight=f_weight, color='black')

        # Adjust X-axis to accommodate left labels and right legend
        plt.xlim(-1.0, 5.5) 

        if label == "Global_Mean_Rank":
            title_prefix = "Global Summary: Mean Rank Across All Datasets"
            ylabel = "Mean Rank"
        elif label == "Global_Mean_AUC":
            title_prefix = "Global Summary: Mean AUC-ROC Across All Datasets"
            ylabel = "Rank (of Mean AUC)"
        else:
            title_prefix = f"Dataset: {label}"
            ylabel = "Rank (1 = Best Performance)"

        # plt.title(f'Rank Flow and Robustness Metrics\n{title_prefix}',                   fontsize=16, fontweight='bold', pad=25)
        plt.xlabel('Noise Level (%)', fontsize=13, fontweight='bold')
        plt.ylabel(ylabel, fontsize=13, fontweight='bold')
        plt.xticks(noises, [f"{int(n)}%" for n in noises], fontsize=11, fontweight='bold')
        
        # Legend outside to the right
        # plt.legend(title='Model Hierarchy', title_fontsize=12, bbox_to_anchor=(1.15, 1), 
        #            loc='upper left', borderaxespad=0., fontsize=10, frameon=True, shadow=True)
        
        plt.grid(True, axis='y', linestyle='--', alpha=0.4, zorder=0)
        plt.grid(True, axis='x', linestyle=':', alpha=0.2, zorder=0)
        plt.tight_layout()
        
        out_path = os.path.join(output_dir, f"noise_bump_{label}.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  -> Saved plot: {out_path}")

        # GENERATE LATEX TABLE FOR THIS CHART
        ltx = "\\begin{table}[h]\n\\centering\n"
        ltx_title = label.replace('_', ' ')
        ltx += f"\\caption{{Noise Robustness Metrics: {ltx_title} (AUC-ROC and Rank fluctuation)}}\n"
        ltx += f"\\label{{tab:noise_bump_{label.lower()}}}\n"
        ltx += "\\adjustbox{width=\\columnwidth}{\n"
        ltx += "\\begin{tabular}{lcccc}\n\\toprule\n"
        ltx += "\\textbf{Model} & \\textbf{0\\% Noise} & \\textbf{1\\% Noise} & \\textbf{3\\% Noise} & \\textbf{5\\% Noise} \\\\\n\\midrule\n"
        
        for m in models_sorted:
            row_cells = [f"\\textbf{{{m}}}" if m == 'LOC-NFST' else m]
            for n in noises:
                c_data = ds_data[(ds_data['model'] == m) & (ds_data['noise'] == n)]
                if c_data.empty:
                    row_cells.append("-")
                else:
                    auc_val = c_data['aucroc'].values[0]
                    if label == "Global_Mean_Rank":
                        rank_val = c_data['plot_rank'].values[0]
                        row_cells.append(f"{rank_val:.2f}")
                    else:
                        # For others, show AUC (Rank)
                        # CRITICAL: Use plot_rank for Global_Mean_AUC to ensure consistent 1-N numbering
                        rank_actual = int(c_data['plot_rank'].values[0])
                        row_cells.append(f"{auc_val:.2f} ({rank_actual})")
            ltx += " & ".join(row_cells) + " \\\\\n"
            
        ltx += "\\bottomrule\n\\end{tabular}\n}\n\\end{table}\n"
        all_latex_tables.append(ltx)

    # Save all LaTeX tables to one file
    ltx_out = os.path.join(output_dir, "noise_results_tables.tex")
    with open(ltx_out, "w", encoding='utf-8') as f:
        f.write("\n\n".join(all_latex_tables))
    print(f"  -> Saved LaTeX tables: {ltx_out}")

def generate_noise_line_chart(files, output_dir):
    print(f"Generating Noise Line Chart for all levels (0%, 1%, 3%, 5%)...")
    all_data = []
    for info in files:
        df = load_and_normalize(info)
        all_data.append(df)
        
    if not all_data: return
    combined_df = pd.concat(all_data, ignore_index=True)
    
    if BASELINE_SCALER:
        is_ours = combined_df['model'] == 'LOC-NFST'
        valid_baselines = combined_df['scaler'].isin([BASELINE_SCALER, 'Unknown'])
        combined_df = combined_df[is_ours | valid_baselines]
        
    print(f"Models passing scaler filter: {combined_df['model'].unique()}")

    noises = [0.0, 1.0, 3.0, 5.0]
    combined_df['noise'] = combined_df['noise'].round(0)
    combined_df = combined_df[combined_df['noise'].isin(noises)]

    best_per_dataset = combined_df.groupby(['model', 'dataset', 'noise'])['aucroc'].max().reset_index()
    avg_df = best_per_dataset.groupby(['model', 'noise'])['aucroc'].mean().reset_index()
    overall_avg = avg_df.groupby('model')['aucroc'].mean().reset_index()
    existing_models = overall_avg['model'].unique()

    if TARGET_MODELS_LIST:
        target_models = [m.upper() for m in TARGET_MODELS_LIST]
        if 'LOC-NFST' in target_models: target_models.remove('LOC-NFST')
        target_models = ['LOC-NFST'] + target_models
        target_models = [m for m in target_models if m in existing_models]
    else:
        baselines = overall_avg[overall_avg['model'] != 'LOC-NFST']
        top_4_baselines = baselines.sort_values(by='aucroc', ascending=False).head(4)['model'].tolist()
        target_models = ['LOC-NFST'] + top_4_baselines

    plot_df = avg_df[avg_df['model'].isin(target_models)].copy()
    pivot_auc = plot_df.pivot(index='model', columns='noise', values='aucroc')
    
    pivot_auc = pivot_auc.reindex(columns=noises)
    pivot_auc = pivot_auc.dropna(how='all')
    
    avg_p = pivot_auc.mean(axis=1)
    models_sorted = avg_p.sort_values(ascending=False).index.tolist()
    if 'LOC-NFST' in models_sorted:
        models_sorted.remove('LOC-NFST')
        models_sorted = ['LOC-NFST'] + models_sorted

    plt.figure(figsize=(12, 6.5))
    sns.set_style("whitegrid")

    markers = ['o', 's', '^', 'v', 'p', '*', 'h', 'X', 'd', '<', '>']
    linestyles = ['-', '--', '-.', ':']
    muted_palette = sns.husl_palette(max(len(models_sorted), 8), l=0.55, s=0.8)

    for i, model in enumerate(models_sorted):
        model_data = pivot_auc.loc[model].dropna()
        if model_data.empty: continue
        
        x_vals = model_data.index.values
        y_vals = model_data.values
        
        is_ours = model == 'LOC-NFST'
        color = '#cc3333' if is_ours else muted_palette[i % len(muted_palette)]
        lw = 4.0 if is_ours else 2.5
        zorder = 20 if is_ours else 10
        ms = 12 if is_ours else 9
        
        marker = 'D' if is_ours else markers[i % len(markers)]
        ls = '-' if is_ours else linestyles[i % len(linestyles)]
        
        lbl = 'OUR' if is_ours else model
        plt.plot(x_vals, y_vals, color=color, linewidth=lw, linestyle=ls, 
                 marker=marker, markersize=ms, zorder=zorder, label=lbl,
                 markeredgecolor='black', markeredgewidth=1.0 if not is_ours else 1.5)
                 
    plt.xlabel('Noise Contamination Level (%)', fontsize=14, fontweight='bold')
    plt.ylabel('AUC-ROC (%)', fontsize=14, fontweight='bold')
    plt.title('Performance Degradation Under Noise Scenarios', fontsize=16, fontweight='bold', pad=15)
    plt.xticks(noises, [f"{int(n)}%" for n in noises], fontsize=13, fontweight='bold')
    plt.yticks(fontsize=13)
    
    plt.legend(title='Algorithm', loc='lower left', ncol=3, 
               fontsize=11, title_fontsize=12, frameon=True, framealpha=0.9)
    
    import matplotlib.ticker as ticker
    ax = plt.gca()
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5.0))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(2.5))
    plt.grid(True, which='major', axis='y', linestyle='--', linewidth=1.2, alpha=0.9)
    plt.grid(True, which='minor', axis='y', linestyle=':', linewidth=0.75, alpha=0.6)
    plt.grid(True, which='major', axis='x', linestyle=':', linewidth=0.5, alpha=0.3)

    y_min_val = pivot_auc.min().min()
    y_max_val = pivot_auc.max().max()
    if not np.isnan(y_min_val) and not np.isnan(y_max_val):
        plt.ylim(max(0, y_min_val - 3), min(105, y_max_val + 3))
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, "noise_levels_line_chart.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Noise Line Chart to {out_path}")

def main():

    import sys
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(_script_dir, "results")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Noise Robustness Visualization ===")
    
    # Auto-detect if running in non-interactive environment
    if len(sys.argv) > 1:
        baseline_input = r"d:\UIT\Research\Duongcpmputer\LOC-NFST\UIT_Research_NFST\notebooks"
        model_input = r"d:\UIT\Research\Duongcpmputer\LOC-NFST\UIT_Research_NFST\notebooks"
        mode = sys.argv[1]
    else:
        baseline_input = input("Enter path to Baseline Results (File or Dir) [default: notebooks]: ").strip()
        model_input = input("Enter path to Model Results (File or Dir) [default: notebooks]: ").strip()
        if not baseline_input: baseline_input = r"d:\UIT\Research\Duongcpmputer\LOC-NFST\UIT_Research_NFST\notebooks"
        if not model_input: model_input = r"d:\UIT\Research\Duongcpmputer\LOC-NFST\UIT_Research_NFST\notebooks"
        mode = input("Choose mode [1] Mean across datasets, [2] Separate bar charts, [3] Noise Rank Bump Charts (Default: 1): ").strip()
        if not mode: mode = '1'

    files = []
    for inp in [baseline_input, model_input]:
        if not inp: continue
        if os.path.isfile(inp) and inp.endswith('.csv'): files.append(('unified', inp))
        elif os.path.isdir(inp): files.extend(find_csv_files(inp))

    if not files:
        print("No results found.")
        return

    if mode == '2':
        generate_separate_noise_charts(files, output_dir)
    elif mode == '3':
        generate_noise_bump_charts(files, output_dir)
    else:
        output_path = os.path.join(output_dir, "noise_levels_grouped.png")
        generate_noise_comparison_chart(files, output_path)
        generate_noise_line_chart(files, output_dir)

if __name__ == '__main__':
    main()
