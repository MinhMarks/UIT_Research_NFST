import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from datetime import datetime
import numpy as np

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
        df_model['Type'] = 'Proposed Model'
        
        # Ensure column alignment. If your model exports 'Method' instead of 'Model', rename it.
        if 'Method' in df_model.columns and 'Model' not in df_model.columns:
            df_model = df_model.rename(columns={'Method': 'Model'})
        
        # If neither 'Method' nor 'Model' was present, assign the proposed method's name
        if 'Model' not in df_model.columns:
            df_model['Model'] = 'LOC-NFST'
            
        # If 'Dataset' is missing, add a fallback to prevent dropping during groupby
        if 'Dataset' not in df_model.columns:
            df_model['Dataset'] = 'Unknown_Dataset'
            
        df_list.append(df_model)
    else:
        print(f"Warning: {model_csv} not found.")

    if not df_list:
        raise FileNotFoundError("Both CSV files are missing! Please place them in this folder.")
        
    df_all = pd.concat(df_list, ignore_index=True)
    
    # --- Filter out DevNet ---
    if 'Model' in df_all.columns:
        df_all = df_all[df_all['Model'] != 'DevNet']
        
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
    best_runs = df.loc[idx]

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    # 1. Bar Chart: Memory Consumption by Model
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=best_runs, 
        x='Model', 
        y='Peak RAM Train (MB)', 
        hue='Type',
        palette={'Baseline': 'salmon', 'Proposed Model': 'dodgerblue'},
        dodge=False
    )
    plt.title('Training Memory Footprint Comparison', fontweight='bold')
    plt.ylabel('Peak RAM Usage (MB)')
    plt.xlabel('Algorithm')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Bar_Memory_Footprint.png"), dpi=300)
    plt.close()

    # 2. Scatter Plot: Trade-off between Detection Performance vs Memory Resource
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=best_runs, 
        x='Peak RAM Train (MB)', 
        y='AUCPR', 
        hue='Model', 
        style='Type',
        s=150, 
        palette='tab20',
        markers={'Baseline': 'o', 'Proposed Model': '*'}
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

        # Keep proposed model + top-(N-1) baselines by AUCPR
        TOP_N = 15
        proposed_rows = best_runs[best_runs['Type'] == 'Proposed Model']
        baseline_rows = best_runs[best_runs['Type'] == 'Baseline']
        top_baselines = baseline_rows.nlargest(TOP_N - len(proposed_rows), 'AUCPR')
        top_runs = pd.concat([proposed_rows, top_baselines]).drop_duplicates('Model')

        time_cols = ['Model', 'Time Train', 'Time Test', 'Type']
        df_time = top_runs[time_cols].melt(
            id_vars=['Model', 'Type'], var_name='Phase', value_name='Time'
        )
        df_time['Phase'] = df_time['Phase'].replace({
            'Time Train': 'Train Time', 'Time Test': 'Inference Time'
        })

        # Tag each row: LOC-NFST gets distinct orange tones, baselines get blue tones
        df_time['Group'] = df_time.apply(
            lambda r: f"LOC-NFST {r['Phase']}" if r['Type'] == 'Proposed Model'
                      else f"Baseline {r['Phase']}",
            axis=1
        )

        COLOR_PALETTE = {
            'LOC-NFST Train Time':      '#5C5C5C',   # dark gray   (proposed, train)
            'LOC-NFST Inference Time':  '#A8A8A8',   # light gray  (proposed, infer)
            'Baseline Train Time':      '#2E6DA4',   # muted blue  (baseline, train)
            'Baseline Inference Time':  '#A8C4E0',   # pale blue   (baseline, infer)
        }

        n_models = top_runs['Model'].nunique()
        fig_w = max(9, n_models * 0.6)   # tighter horizontal space
        plt.figure(figsize=(fig_w, 7))

        model_order = top_runs.sort_values('Time Train', ascending=True)['Model'].tolist()
        ax = sns.barplot(
            data=df_time,
            x='Model',
            y='Time',
            hue='Group',
            palette=COLOR_PALETTE,
            order=model_order,
            hue_order=list(COLOR_PALETTE.keys()),
            width=0.85,   # wider bars within their category bin
        )

        # Value labels on top of each bar (currently disabled as requested)
        # for p in ax.patches:
        #     h = p.get_height()
        #     ...

        # Bold LOC-NFST tick label
        for lbl in ax.get_xticklabels():
            if 'LOC-NFST' in lbl.get_text():
                lbl.set_fontweight('bold')

        # plt.title('Computational Time Complexity (Top 15 Models)', fontweight='bold')
        plt.ylabel('Seconds (Log Scale)', fontsize=15)
        plt.xlabel('Algorithm', fontsize=15)
        plt.yscale('log')
        plt.xticks(rotation=40, ha='right', fontsize=14)
        plt.yticks(fontsize=13)
        plt.legend(title='Model & Phase', loc='upper left', fontsize=11, frameon=True)
        plt.grid(True, which='both', ls='--', alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "Bar_Time_Complexity.png"), dpi=300)
        plt.close()

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
    proposed_all = best_runs[best_runs['Type'] == 'Proposed Model'].copy()
    baselines     = best_runs[best_runs['Type'] == 'Baseline'].copy()

    if proposed_all.empty:
        print("No proposed model data found — skipping radar chart.")
        return

    # Pick one row: the one with the highest AUCROC across all datasets/scalers
    proposed = proposed_all.loc[[proposed_all['AUCROC'].idxmax()]]

    # ── 2. Average of top-15 baselines per metric ─────────────────────────────
    top_n = 22
    top15_avg = {}
    for metric in metrics_to_plot:
        top15_avg[metric] = baselines[metric].nlargest(top_n).mean()

    top15_df = pd.DataFrame([top15_avg])
    top15_df['Model'] = f'Top {top_n} Baselines (Avg)'
    top15_df['Type']  = 'Top N Average'

    plot_df = pd.concat([proposed, top15_df], ignore_index=True)

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
        color = 'dodgerblue' if row['Type'] == 'Proposed Model' else 'salmon'
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
