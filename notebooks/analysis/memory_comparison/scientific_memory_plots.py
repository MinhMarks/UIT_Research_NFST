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
            df_model['Model'] = 'XXX'
            
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

    # 3. Bar Chart: Training Time vs Test Time (Grouped)
    if 'Time Train' in best_runs.columns and 'Time Test' in best_runs.columns:
        plt.figure(figsize=(12, 6))
        
        # Melt the dataframe into long-form for side-by-side bars
        time_cols = ['Model', 'Time Train', 'Time Test']
        df_time = best_runs[time_cols].melt(id_vars='Model', var_name='Phase', value_name='Time')
        
        # Cleaner Phase names for legend
        df_time['Phase'] = df_time['Phase'].replace({'Time Train': 'Train Time', 'Time Test': 'Inference Time'})

        # Grouped bar plot (Horizontal)
        ax = sns.barplot(
            data=df_time, 
            y='Model', 
            x='Time', 
            hue='Phase',
            palette={'Train Time': 'skyblue', 'Inference Time': 'navy'}
        )
        
        # Add numeric labels on the end of each bar
        for p in ax.patches:
            if p.get_width() > 0:
                ax.annotate(f'{p.get_width():.2f}s', 
                            (p.get_width(), p.get_y() + p.get_height() / 2.), 
                            ha = 'left', va = 'center', 
                            xytext = (5, 0), 
                            textcoords = 'offset points',
                            fontsize=9, fontweight='bold')

        plt.title('Computational Time Complexity (Horizontal Grouped)', fontweight='bold')
        plt.xlabel('Seconds (Log Scale)')
        
        # Refine log scale limits
        plt.xscale('log')
        curr_xmin, curr_xmax = plt.xlim()
        plt.xlim(curr_xmin, curr_xmax * 10) 
        
        plt.ylabel('Algorithm')
        plt.legend(title='Execution Phase', loc='lower right')
        plt.grid(True, which="both", ls="-", alpha=0.2)
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
    
    # Separate proposed model and baselines
    proposed = best_runs[best_runs['Type'] == 'Proposed Model'].copy()
    baselines = best_runs[best_runs['Type'] == 'Baseline'].copy()
    
    # Calculate average of top 10 baselines for each metric
    top10_avg = {}
    for metric in metrics_to_plot:
        top10_avg[metric] = baselines[metric].nlargest(10).mean()
        
    top10_df = pd.DataFrame([top10_avg])
    top10_df['Model'] = 'Top 10 Baselines (Avg)'
    top10_df['Type'] = 'Top 10 Average'
    
    # Combine proposed model and top 10 average
    plot_df = pd.concat([proposed, top10_df], ignore_index=True)

    num_vars = len(labels)
    # Compute angle of each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] # Close the circle

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    
    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], labels, color='grey', size=11)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([25, 50, 75, 100], ["25", "50", "75", "100"], color="grey", size=8)
    plt.ylim(0, 110)
    
    # Plot each model
    for i, row in plot_df.iterrows():
        values = row[metrics_to_plot].values.flatten().tolist()
        values += values[:1] # Close the circle
        
        # Differentiate Proposed Model vs Top 10 Average
        if row['Type'] == 'Proposed Model':
            linewidth, linestyle, alpha = 4, 'solid', 0.4
            color = 'dodgerblue'
        else: # Top 10 Average
            linewidth, linestyle, alpha = 3, 'dashed', 0.2
            color = 'salmon'

        ax.plot(angles, values, linewidth=linewidth, linestyle=linestyle, label=row['Model'], color=color)
        ax.fill(angles, values, alpha=alpha, color=color)

        # ── Annotate values at each vertex ───────────────────────────────────
        annotation_color = color
        for angle, val, label_txt in zip(angles, values[:-1], labels):
            # Offset slightly outward from the data point for readability
            offset = 8  # in data units (0-110 scale)
            r_label = val + offset
            # Clamp so label stays within ylim
            r_label = min(r_label, 108)

            # Determine horizontal alignment based on angle quadrant
            x_cart = np.cos(angle - np.pi / 2)
            ha = 'center'
            if x_cart > 0.3:  ha = 'left'
            elif x_cart < -0.3: ha = 'right'

            ax.annotate(
                f"{val:.1f}",
                xy=(angle, val),
                xytext=(angle, r_label),
                ha=ha, va='center',
                fontsize=8,
                fontweight='bold',
                color=annotation_color,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.7),
            )

    plt.title('Holistic Model Comparison: Proposed vs Top 10 Baselines Average', size=15, color='black', y=1.1, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
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
