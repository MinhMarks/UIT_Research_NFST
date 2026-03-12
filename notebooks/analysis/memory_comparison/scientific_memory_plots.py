import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

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
            
        df_list.append(df_model)
    else:
        print(f"Warning: {model_csv} not found.")

    if not df_list:
        raise FileNotFoundError("Both CSV files are missing! Please place them in this folder.")
        
    df_all = pd.concat(df_list, ignore_index=True)
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

    # 3. Bar Chart: Training Time vs Test Time (if columns exist)
    if 'Time Train' in df.columns and 'Time Test' in df.columns:
        plt.figure(figsize=(12, 6))
        
        # Sort by train time
        best_runs_sorted = best_runs.sort_values(by='Time Train', ascending=False)
        sns.barplot(
            data=best_runs_sorted, 
            x='Model', 
            y='Time Train', 
            color='lightblue', 
            label='Train Time'
        )
        sns.barplot(
            data=best_runs_sorted, 
            x='Model', 
            y='Time Test', 
            color='darkblue', 
            label='Test Time'
        )
        plt.title('Computational Time Constraints (Log Scale)', fontweight='bold')
        plt.ylabel('Time (Seconds) - Log Scale')
        plt.yscale('log') # Use log scale because deep models take hours while KNN takes seconds
        plt.xlabel('Algorithm')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "Bar_Time_Complexity.png"), dpi=300)
        plt.close()

    print(f"Scientific Paper Visualizations saved to: {output_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Memory and Performance Charts for Paper")
    parser.add_argument("--baseline", type=str, default="baseline_results.csv", help="CSV containing tuned baseline results")
    parser.add_argument("--model", type=str, default="model_results.csv", help="CSV containing proposed model results")
    parser.add_argument("--output", type=str, default="plots", help="Output directory for images")
    
    args = parser.parse_args()
    
    df_merged = load_and_merge_data(args.baseline, args.model)
    plot_memory_comparison(df_merged, args.output)
