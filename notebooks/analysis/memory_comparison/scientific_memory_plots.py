import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import argparse
import os
from datetime import datetime
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================
# Manual model selection for the comparison charts and table.
# Set to None for automatic Top-N selection.
# Example: SELECTED_MODELS = ['OUR', 'IForest', 'KNN', 'LOF', 'AutoEncoder', 'LUNAR', 'DIF', 'DASVDD']
SELECTED_MODELS = ['OUR', 'AUTOENCODER', 'LUNAR', 'KNN', 'DIF', 'HBOS', 'DASVDD', 'NeuTraLAD', 'VAE' ] 
DEFAULT_TOP_N = 15

def normalize_model_name(name):
    """Normalize model names for consistent comparison (matches plot_anomaly_types.py)"""
    if not isinstance(name, str): return str(name)
    name_clean = name.lower().strip()
    if any(alias in name_clean for alias in ['ourmodel', 'nhatauto', 'loc-nfst', 'our']):
        return 'OUR'
    # Standardize baseline names
    if name_clean == 'iforest': return 'IForest'
    if name_clean == 'ocsvm': return 'OCSVM'
    if name_clean == 'deepsvdd': return 'DeepSVDD'
    if name_clean == 'autoencoder': return 'AutoEncoder'
    return name.upper().strip()

def load_and_merge_data(baseline_csv, model_csv):
    """
    Reads baseline and proposed model results, standardizes their columns,
    and merges them into a single DataFrame for plotting.
    """
    df_list = []
    
    # 1. Load Baselines
    if os.path.exists(baseline_csv):
        print(f"Loading Baseline file: {baseline_csv}")
        df_base = pd.read_csv(baseline_csv)
        df_base['Type'] = 'Baseline'
        df_list.append(df_base)
    else:
        print(f"Warning: {baseline_csv} not found.")

    # 2. Load Proposed Model
    if os.path.exists(model_csv):
        print(f"Loading Model file: {model_csv}")
        df_model = pd.read_csv(model_csv)
        df_model['Type'] = 'LOC-NFST (Ours)'
        
        # Ensure column alignment. If your model exports 'Method' instead of 'Model', rename it.
        if 'Method' in df_model.columns and 'Model' not in df_model.columns:
            df_model = df_model.rename(columns={'Method': 'Model'})
        
        # If neither 'Method' nor 'Model' was present, assign the proposed method's name
        if 'Model' not in df_model.columns:
            df_model['Model'] = 'OUR'
            
        # If 'Dataset' is missing, add a fallback to prevent dropping during groupby
        if 'Dataset' not in df_model.columns:
            df_model['Dataset'] = 'Unknown_Dataset'
            
        df_list.append(df_model)
    else:
        print(f"Warning: {model_csv} not found.")

    if not df_list:
        raise FileNotFoundError("Both CSV files are missing! Please place them in this folder.")
        
    df_all = pd.concat(df_list, ignore_index=True)
    
    if 'Model' in df_all.columns:
        # Standardize all model names immediately
        df_all['Model'] = df_all['Model'].apply(normalize_model_name)
        df_all = df_all[df_all['Model'] != 'DEVNET']
        
        # Absolute safety override: Anything named 'OUR' MUST be 'LOC-NFST (Ours)'
        df_all.loc[df_all['Model'] == 'OUR', 'Type'] = 'LOC-NFST (Ours)'
        
    return df_all

def plot_memory_comparison(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure columns exist before plotting
    if 'Peak RAM Train (MB)' not in df.columns or 'AUCPR' not in df.columns:
        print("Required columns ('Peak RAM Train (MB)', 'AUCPR') not found in the merged data.")
        print(f"Available columns: {df.columns.tolist()}")
        return

    # To be fair, select the BEST AUCPR run for each model/method for plotting
    idx = df.groupby(['Dataset', 'Model'])['AUCPR'].idxmax()
    best_runs = df.loc[idx].copy()

    # Manual or Automatic Selection
    if SELECTED_MODELS:
        print(f"Applying manual selection: {SELECTED_MODELS}")
        selected_normalized = [normalize_model_name(m) for m in SELECTED_MODELS]
        best_runs = best_runs[best_runs['Model'].isin(selected_normalized)]
    else:
        # Fallback to Top N
        best_runs = best_runs.nlargest(DEFAULT_TOP_N, 'AUCPR')

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    # 1. Bar Chart: Memory Consumption by Model
    plt.figure(figsize=(12, 6))
    
    # ── AGGREGATE BEFORE PLOTTING TO PREVENT SEABORN MISMATCH ──
    # If there are multiple datasets, taking the mean aligns exactly with sns.barplot logic
    plot_df = best_runs.groupby(['Model', 'Type'])['Peak RAM Train (MB)'].mean().reset_index()
    # Sort for consistent patch index mapping
    plot_df = plot_df.sort_values('Peak RAM Train (MB)', ascending=False).reset_index(drop=True)
    
    ax = sns.barplot(
        data=plot_df, 
        x='Model', 
        y='Peak RAM Train (MB)', 
        color='gray', # We will overwrite this manually
        errorbar=None
    )
    
    # Apply explicit textures and colors by row index
    for i, bar in enumerate(ax.patches):
        m_type = plot_df.iloc[i]['Type']
        if m_type == 'LOC-NFST (Ours)':
            bar.set_facecolor('#cc3333')
            bar.set_hatch('x')
        else:
            bar.set_facecolor('#1f77b4')
            bar.set_hatch('/')
        bar.set_edgecolor('black')
        bar.set_linewidth(0.5)

    # Add numeric labels upward (Black text)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=3, fontsize=13, fontweight='bold', color='black')

    # Custom Legend: explicitly draw patches with Hatches
    mem_handles = [
       mpatches.Patch(facecolor='#1f77b4', edgecolor='black', hatch='/', label='Baseline'),
       mpatches.Patch(facecolor='#cc3333', edgecolor='black', hatch='x', label='LOC-NFST (Ours)')
    ]
    ax.legend(handles=mem_handles, title="Type", loc='upper right', fontsize=14, title_fontsize=15)

    # Expand Y-axis to prevent label overflow
    ax.set_ylim(0, plot_df['Peak RAM Train (MB)'].max() * 1.2)

    plt.title('Training Memory Footprint Comparison', fontweight='bold', fontsize=18)
    plt.ylabel('Peak RAM Usage (MB)', fontsize=16)
    plt.xlabel('Algorithm', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=15)
    plt.yticks(fontsize=14)
    # Tăng cường đường lưới ngang vạch chuẩn
    ax.yaxis.grid(True, linestyle='-', linewidth=1.2, alpha=0.25, color='gray')
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Bar_Memory_Footprint.png"), dpi=300)
    plt.close()

    # 2. Scatter Plot: Trade-off between Detection Performance vs Memory Resource
    plt.figure(figsize=(10, 8))
    # Build a palette for the scatter plot that maps every baseline model to Blue
    unique_models = best_runs['Model'].unique()
    scatter_palette = {m: '#cc3333' if m == 'OUR' else '#1f77b4' for m in unique_models}

    sns.scatterplot(
        data=best_runs, 
        x='Peak RAM Train (MB)', 
        y='AUCPR', 
        hue='Model', 
        style='Type',
        s=150, 
        palette=scatter_palette,
        markers={'Baseline': 'o', 'LOC-NFST (Ours)': '*'}
    )
    plt.title('Trade-off Analysis: Resource Efficiency vs Performance', fontweight='bold')
    plt.xlabel('Peak RAM Training Allocation (MB) -> (Lower is better)')
    plt.ylabel('Area Under Precision-Recall Curve (%) -> (Higher is better)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Scatter_Efficiency_Tradeoff.png"), dpi=300)
    plt.close()

    # 3. Bar Chart: Training Time vs Test Time — vertical, top-15 models
    if 'Time Train' in best_runs.columns and 'Time Test' in best_runs.columns:
        top_runs = best_runs.copy()

        time_cols = ['Model', 'Time Train', 'Time Test', 'Type']
        df_time = top_runs[time_cols].melt(
            id_vars=['Model', 'Type'], var_name='Phase', value_name='Time'
        )
        df_time['Phase'] = df_time['Phase'].replace({
            'Time Train': 'Train Time', 'Time Test': 'Inference Time'
        })

        # Group by Phase instead of massive 4-group to ensure exactly 2 FAT bars per model!
        n_models = top_runs['Model'].nunique()
        fig_w = max(14, n_models * 1.5) # Extra width
        plt.figure(figsize=(fig_w, 8))

        # ── AGGREGATE BEFORE PLOTTING TO PREVENT SEABORN MISMATCH ──
        df_time_agg = df_time.groupby(['Model', 'Type', 'Phase'])['Time'].mean().reset_index()
        
        # Calculate strict model order 
        order_df = df_time_agg[df_time_agg['Phase'] == 'Train Time'].sort_values('Time', ascending=True)
        model_order = order_df['Model'].tolist()
        model_types = {row['Model']: row['Type'] for _, row in order_df.iterrows()}

        # Force exactly 2 phases
        ax = sns.barplot(
            data=df_time_agg,
            x='Model',
            y='Time',
            hue='Phase', 
            order=model_order,
            hue_order=['Train Time', 'Inference Time'],
            palette={'Train Time': 'gray', 'Inference Time': 'lightgray'}, # override below
            width=0.8,
            dodge=True,
            errorbar=None
        )

        # Apply precise textures and colors manually per bar based on Model Type
        for i, container in enumerate(ax.containers):
            phase_name = 'Train Time' if i == 0 else 'Inference Time'
            for j, bar in enumerate(container):
                model_name = model_order[j]
                m_type = model_types.get(model_name, 'Baseline')
                
                if m_type == 'LOC-NFST (Ours)':
                    c = '#cc3333' if phase_name == 'Train Time' else '#ff9999'
                    h = 'x' if phase_name == 'Train Time' else '.'
                else:
                    c = '#1f77b4' if phase_name == 'Train Time' else '#a1c9ed'
                    h = '/' if phase_name == 'Train Time' else '\\'
                
                bar.set_facecolor(c)
                bar.set_hatch(h)
                bar.set_edgecolor('black')
                bar.set_linewidth(0.5)

        # Bold OUR tick label
        for lbl in ax.get_xticklabels():
            if 'OUR' in lbl.get_text():
                lbl.set_fontweight('bold')
                
        # Add labels vertically inside
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3, fontsize=14, fontweight='bold',
                         rotation=90, label_type='edge', color='black')
            
        # Custom Legend manually specifying colors and hatches
        time_handles = [
            mpatches.Patch(facecolor='#cc3333', edgecolor='black', hatch='x', label='Our Train Time'),
            mpatches.Patch(facecolor='#ff9999', edgecolor='black', hatch='.', label='Our Inference Time'),
            mpatches.Patch(facecolor='#1f77b4', edgecolor='black', hatch='/', label='Baseline Train Time'),
            mpatches.Patch(facecolor='#a1c9ed', edgecolor='black', hatch='\\', label='Baseline Inference Time')
        ]
        ax.legend(handles=time_handles, title="Phase & Model", loc='upper left', fontsize=14, title_fontsize=15, frameon=True)

        plt.ylabel('Seconds (Log Scale)', fontsize=17)
        plt.xlabel('Algorithm', fontsize=17)
        plt.yscale('log')
        
        # Expand Y-axis significantly (Log scale) to prevent label overflow at top
        y_min = df_time['Time'].min()
        y_max = df_time['Time'].max()
        plt.ylim(bottom=max(1e-5, y_min / 5), top=y_max * 50)  # Tăng khoảng trống trên để chứa text lớn

        plt.xticks(rotation=40, ha='right', fontsize=15)
        plt.yticks(fontsize=15)
        
        # Lưới chuẩn cho biểu đồ vạch log
        ax.yaxis.grid(True, which='major', linestyle='-', linewidth=1.2, alpha=0.6, color='gray')
        ax.yaxis.grid(True, which='minor', linestyle='--', linewidth=0.8, alpha=0.1, color='gray')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "Bar_Time_Complexity.png"), dpi=300)
        plt.close()

    # === ADDED LOGGING FOR USER ===
    log_file = os.path.join(output_dir, "Bar_Chart_Data_Log.csv")
    cols_to_keep = ['Model', 'Peak RAM Train (MB)', 'AUCROC', 'AUCPR']
    if 'Time Train' in best_runs.columns and 'Time Test' in best_runs.columns:
        cols_to_keep += ['Time Train', 'Time Test']
        
    log_df = best_runs[cols_to_keep].copy()
    # Seaborn barplot automatically averages across Datasets. We do the same here to match the charts.
    avg_log_df = log_df.groupby('Model').mean(numeric_only=True).reset_index()
    avg_log_df.to_csv(log_file, index=False)
    print("\n" + "="*70)
    print("BAR CHART DATA METRICS (Averaged across Datasets):")
    print("="*70)
    print(avg_log_df.to_string(index=False))
    print("="*70 + "\n")

    print(f"Standard Visualizations saved to: {output_dir}/")

def plot_radar_summary(df, output_dir):
    """
    Creates a Radar (Spider) Chart comparing the proposed model vs the average of top 10 baselines.
    Metrics: AUCROC, AUCPR, RAM Efficiency, Inference Speed Efficiency.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Prepare metrics
    # Get best results per model
    idx = df.groupby(['Dataset', 'Model'])['AUCPR'].idxmax()
    best_runs = df.loc[idx].copy()
    
    # Convert metrics to 0-100 where higher is always better
    # RAM Efficiency: (Max - Actual) / Range -> Normalized to 0-100
    r_min, r_max = best_runs['Peak RAM Train (MB)'].min(), best_runs['Peak RAM Train (MB)'].max()
    if r_max > r_min:
        best_runs['RAM_Eff'] = 100 * (r_max - best_runs['Peak RAM Train (MB)']) / (r_max - r_min)
    else:
        best_runs['RAM_Eff'] = 100
        
    # Inference Speed Efficiency (MaxTime - ActualTime) / Range -> Normalized to 0-100
    t_min, t_max = best_runs['Time Test'].min(), best_runs['Time Test'].max()
    if t_max > t_min:
        best_runs['Speed_Eff'] = 100 * (t_max - best_runs['Time Test']) / (t_max - t_min)
    else:
        best_runs['Speed_Eff'] = 100
        
    # Training Speed Efficiency
    tr_min, tr_max = best_runs['Time Train'].min(), best_runs['Time Train'].max()
    if tr_max > tr_min:
        best_runs['Train_Eff'] = 100 * (tr_max - best_runs['Time Train']) / (tr_max - tr_min)
    else:
        best_runs['Train_Eff'] = 100

    metrics_to_plot = ['AUCROC', 'AUCPR', 'RAM_Eff', 'Speed_Eff', 'Train_Eff']
    labels = ['AUC-ROC', 'AUC-PR', 'RAM Efficiency', 'Inference Speed', 'Training Speed']
    
    # ── 1. Proposed model: single globally best AUCROC row ────────────────────
    proposed_all = best_runs[best_runs['Type'] == 'LOC-NFST (Ours)'].copy()
    baselines     = best_runs[best_runs['Type'] == 'Baseline'].copy()

    if proposed_all.empty:
        print("No proposed model data found — skipping radar chart.")
        return

    # Pick one row: the one with the highest AUCROC across all datasets/scalers
    proposed = proposed_all.loc[[proposed_all['AUCROC'].idxmax()]]

    # ── 2. Average of selected baselines per metric ─────────────────────────────
    # Standardize baseline selection for the "Average" line
    if SELECTED_MODELS:
        selected_norms = [normalize_model_name(m) for m in SELECTED_MODELS if normalize_model_name(m) != 'OUR']
        target_baselines = baselines[baselines['Model'].isin(selected_norms)]
        n_count = len(target_baselines)
    else:
        n_count = 15
        target_baselines = baselines.nlargest(n_count, 'AUCROC')

    top_avg = {}
    for metric in metrics_to_plot:
        top_avg[metric] = target_baselines[metric].mean()

    top_avg_df = pd.DataFrame([top_avg])
    top_avg_df['Model'] = f'Selected Baselines (Avg)' if SELECTED_MODELS else f'Top {n_count} (Avg)'
    top_avg_df['Type']  = 'Top N Average'

    plot_df = pd.concat([proposed, top_avg_df], ignore_index=True)

    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(12, 12))
    ax = plt.subplot(111, polar=True)

    # ── Draw axis labels manually at fixed radius outside all data ────────────
    # Hide the default xtick labels (they sit too close to outer ring)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])   # clear default labels
    ax.tick_params(axis='x', pad=0)  # no padding needed

    label_radius = 145   # place labels well outside ylim=115
    for angle, lbl in zip(angles[:-1], labels):
        # Convert polar to unit direction vector
        x_unit = np.cos(angle - np.pi / 2)
        y_unit = np.sin(angle - np.pi / 2)   # matplotlib polar: 0=top, CCW+

        # Horizontal alignment based on x direction
        if x_unit > 0.25:    ha = 'left'
        elif x_unit < -0.25: ha = 'right'
        else:                ha = 'center'

        # Vertical alignment based on y direction  
        if y_unit > 0.25:    va = 'bottom'
        elif y_unit < -0.25: va = 'top'
        else:                va = 'center'

        ax.text(angle, label_radius, lbl,
                ha=ha, va=va, size=19, color='#444444', fontweight='bold',
                multialignment='center')

    ax.set_rlabel_position(30)
    plt.yticks([25, 50, 75, 100], ["25", "50", "75", "100"], color="grey", size=9)
    plt.ylim(0, 130)

    # ── Collect all model values before annotating ────────────────────────────
    all_values = []
    all_colors = []
    for _, row in plot_df.iterrows():
        vals = row[metrics_to_plot].values.flatten().tolist()
        color = '#cc3333' if row['Type'] == 'LOC-NFST (Ours)' else '#1f77b4'
        all_values.append(vals)
        all_colors.append(color)

    # ── Draw lines and fills ─────────────────────────────────────────────────
    styles = [
        dict(linewidth=4, linestyle='solid',  alpha=0.4, label=plot_df.iloc[0]['Model']),
        dict(linewidth=3, linestyle='dashed', alpha=0.2, label=plot_df.iloc[1]['Model'] if len(plot_df) > 1 else ''),
    ]
    for i, (vals, color) in enumerate(zip(all_values, all_colors)):
        v = vals + vals[:1]
        st = styles[i] if i < len(styles) else styles[-1]
        ax.plot(angles, v, linewidth=st['linewidth'], linestyle=st['linestyle'],
                label=st['label'], color=color)
        ax.fill(angles, v, alpha=st['alpha'], color=color)


    # ── Draw one combined annotation per vertex (no overlap possible) ─────────
    # Each vertex gets a stacked text: line 0 = proposed, line 1 = baseline avg
    # Placed at a fixed r just outside the outermost data point
    for vi, angle in enumerate(angles[:-1]):
        x_unit = np.cos(angle - np.pi / 2)
        y_unit = np.sin(angle - np.pi / 2)

        # ha/va based on direction
        ha = 'center'
        if x_unit > 0.25:    ha = 'left'
        elif x_unit < -0.25: ha = 'right'
        va = 'center'
        if y_unit > 0.25:    va = 'bottom'
        elif y_unit < -0.25: va = 'top'

        # Radial position: just beyond the maximum value at this vertex
        max_val = max(v[vi] for v in all_values)
        r_base = max_val + 6

        # Draw each model's value stacked vertically (offset in points)
        # Model 0 just above center, model 1 just below
        v_offsets = [+9, -9]  # points offset in y
        for mi, (vals, color) in enumerate(zip(all_values, all_colors)):
            val = vals[vi]
            ax.annotate(
                f"{val:.1f}",
                xy=(angle, r_base),
                xycoords='data',
                xytext=(0, v_offsets[mi]),
                textcoords='offset points',
                ha=ha, va='center',
                fontsize=11, fontweight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85),
            )

    # No title, place legend ABOVE the radar circle
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.12), ncol=2, fontsize=12, frameon=True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # leave space at top for legend
    plt.savefig(os.path.join(output_dir, "Radar_Comparison.png"), dpi=300)
    plt.close()
    print(f"Radar Comparison Chart saved to: {output_dir}/Radar_Comparison.png")

def export_results_to_latex(df, output_dir):
    """
    Exports summary tables for Memory and Computational Complexity to a LaTeX file.
    """
    os.makedirs(output_dir, exist_ok=True)
    tex_path = os.path.join(output_dir, "Memory_Efficiency_Results.tex")
    
    # 1. Main Performance/Efficiency Table (Top 15 + LOC-NFST)
    idx = df.groupby(['Dataset', 'Model'])['AUCPR'].idxmax()
    best_runs = df.loc[idx].copy()
    
    # Manual or Automatic Selection for LaTeX output
    if SELECTED_MODELS:
        print(f"Applying manual selection for LaTeX table: {SELECTED_MODELS}")
        selected_normalized = [normalize_model_name(m) for m in SELECTED_MODELS]
        top_runs = best_runs[best_runs['Model'].isin(selected_normalized)].sort_values('Peak RAM Train (MB)')
    else:
        # Fallback to Top N
        top_runs = best_runs.nlargest(DEFAULT_TOP_N, 'AUCPR').sort_values('Peak RAM Train (MB)')

    with open(tex_path, "w") as f:
        # Table 1: Memory & Complexity
        f.write("% --- TABLE: Memory and Computational Complexity ---\n")
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Computational Resource Efficiency: Peak RAM and Time Complexity Comparison}\n")
        f.write("\\label{tab:memory_complexity}\n")
        f.write("\\begin{tabular}{lrrr}\n\\toprule\n")
        f.write("\\textbf{Algorithm} & \\textbf{Peak RAM (MB)} & \\textbf{Train Time (s)} & \\textbf{Test Time (s)} \\\\\n\\midrule\n")
        
        for _, row in top_runs.iterrows():
            model_name = f"\\textbf{{{row['Model']}}}" if row['Type'] == 'LOC-NFST (Ours)' else row['Model']
            f.write(f"{model_name} & {row['Peak RAM Train (MB)']:.2f} & {row['Time Train']:.4f} & {row['Time Test']:.4f} \\\\\n")
            
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

        # Table 2: Radar Metric Summary (Proposed vs Top N Avg)
        f.write("% --- TABLE: Radar Metric Summary (Overall Effectiveness) ---\n")
        
        # Prepare radar data (copy-pasted logic from radar function for consistency)
        # RAM Efficiency
        r_min, r_max = best_runs['Peak RAM Train (MB)'].min(), best_runs['Peak RAM Train (MB)'].max()
        best_runs['RAM_Eff'] = 100 * (r_max - best_runs['Peak RAM Train (MB)']) / (r_max - r_min) if r_max > r_min else 100
        # Speed Efficiency
        t_max = best_runs['Time Test'].max(); t_min = best_runs['Time Test'].min()
        best_runs['Speed_Eff'] = 100 * (t_max - best_runs['Time Test']) / (t_max - t_min) if t_max > t_min else 100
        # Train Efficiency
        tr_max = best_runs['Time Train'].max(); tr_min = best_runs['Time Train'].min()
        best_runs['Train_Eff'] = 100 * (tr_max - best_runs['Time Train']) / (tr_max - tr_min) if tr_max > tr_min else 100

        metrics = ['AUCROC', 'AUCPR', 'RAM_Eff', 'Speed_Eff', 'Train_Eff']
        p_row = best_runs[best_runs['Type'] == 'LOC-NFST (Ours)'].iloc[0]
        b_avg = best_runs[best_runs['Type'] == 'Baseline'].nlargest(22, 'AUCROC')[metrics].mean()

        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Radar Summary: Comparison of Macro-Efficiency Scores}\n")
        f.write("\\label{tab:radar_summary}\n")
        f.write("\\begin{tabular}{lccccc}\n\\toprule\n")
        f.write("\\textbf{Metric} & \\textbf{AUC-ROC} & \\textbf{AUC-PR} & \\textbf{RAM Eff.} & \\textbf{Inf. Speed} & \\textbf{Train Speed} \\\\\n\\midrule\n")
        
        f.write(f"\\textbf{{{p_row['Model']}}} & {p_row['AUCROC']:.2f} & {p_row['AUCPR']:.2f} & {p_row['RAM_Eff']:.1f} & {p_row['Speed_Eff']:.1f} & {p_row['Train_Eff']:.1f} \\\\\n")
        f.write(f"Baseline (Avg) & {b_avg['AUCROC']:.2f} & {b_avg['AUCPR']:.2f} & {b_avg['RAM_Eff']:.1f} & {b_avg['Speed_Eff']:.1f} & {b_avg['Train_Eff']:.1f} \\\\\n")
        
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    print(f"LaTeX Results Tables saved to: {tex_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Memory and Performance Charts for Paper")
    parser.add_argument("--baseline", type=str, default="baseline_results.csv", help="CSV containing tuned baseline results")
    parser.add_argument("--model", type=str, default="model_results.csv", help="CSV containing proposed model results")
    parser.add_argument("--output", type=str, default="plots", help="Output directory for images")
    
    args = parser.parse_args()
    
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.output == "plots":
        # Default case: create a timestamped folder
        experiment_name = f"Scientific_Plots_{RUN_TIMESTAMP}"
        final_output_dir = os.path.join(_script_dir, 'plots', experiment_name)
    else:
        final_output_dir = args.output
    
    df_merged = load_and_merge_data(args.baseline, args.model)
    plot_memory_comparison(df_merged, final_output_dir)
    plot_radar_summary(df_merged, final_output_dir)
    export_results_to_latex(df_merged, final_output_dir)
