import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os
from datetime import datetime

def analyze_tuning_impact(input_csv):
    """
    Analyzes the 'Tuned_Baseline_Results_All.csv' structure to show the variance 
    and importance of hyperparameter tuning for baseline models.
    """
    if not os.path.exists(input_csv):
        print(f"File not found: {input_csv}. Please provide a valid results CSV.")
        return

    print(f"Reading {input_csv} ...")
    df = pd.read_csv(input_csv)
    
    # Ensure AUCPR is numeric
    df['AUCPR'] = pd.to_numeric(df['AUCPR'], errors='coerce')
    df = df.dropna(subset=['AUCPR'])
    
    if df.empty:
        print("No valid AUCPR data found to analyze.")
        return

    # Create a timestamped output directory
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(_script_dir, "tuning_analysis_output", f"Analysis_{RUN_TIMESTAMP}")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Boxplot showing the variance of AUCPR for each model across all parameter configurations
    plt.figure(figsize=(14, 7))
    sns.boxplot(data=df, x='Model', y='AUCPR', showfliers=False, color='lightblue')
    sns.stripplot(data=df, x='Model', y='AUCPR', color='darkblue', alpha=0.5, jitter=True)
    plt.title('Distribution of AUCPR Across Different Hyperparameters (per Model)')
    plt.ylabel('AUCPR (%)')
    plt.xlabel('Baseline Model')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    boxplot_path = os.path.join(output_dir, "tuning_variance_boxplot.png")
    plt.savefig(boxplot_path)
    print(f"Saved variance boxplot to {boxplot_path}")
    plt.close()

    # 2. Max vs Min comparison chart
    # To be fair, calculate the max and min *per dataset* then average them
    grouped = df.groupby(['Model', 'Dataset'])['AUCPR'].agg(['max', 'min']).reset_index()
    grouped['difference'] = grouped['max'] - grouped['min']
    
    # Average across all datasets
    avg_impact = grouped.groupby('Model')[['max', 'min', 'difference']].mean().reset_index()
    avg_impact = avg_impact.sort_values(by='difference', ascending=False)
    
    # Plotting Max vs Min
    plt.figure(figsize=(14, 7))
    x_indices = np.arange(len(avg_impact['Model']))
    width = 0.35
    
    plt.bar(x_indices - width/2, avg_impact['max'], width, label='Best Params (Max AUCPR)', color='forestgreen')
    plt.bar(x_indices + width/2, avg_impact['min'], width, label='Worst Params (Min AUCPR)', color='firebrick')
    
    plt.ylabel('Average AUCPR (%)')
    plt.title('Impact of Tuning: Best vs Worst Parameter Configuration (Average across datasets)')
    plt.xticks(x_indices, avg_impact['Model'], rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    diff_chart_path = os.path.join(output_dir, "max_vs_min_tuning_impact.png")
    plt.savefig(diff_chart_path)
    print(f"Saved Best vs Worst comparison chart to {diff_chart_path}")
    plt.close()

    # Output analytical summary to a markdown text file
    summary_path = os.path.join(output_dir, "tuning_sensitivity_report.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=== HYPERPARAMETER TUNING SENSITIVITY REPORT ===\n\n")
        f.write("This report shows how much a model's performance (AUCPR) fluctuates depending on the hyperparameters chosen.\n")
        f.write("Models with a high 'Avg Difference' are HIGHLY SENSITIVE to tuning and will fail without the right parameters.\n\n")
        f.write(avg_impact.to_string(index=False))
        f.write("\n\nConclusion:\n")
        f.write("- Models at the top of this list require rigorous GridSearch/Hyperopt.\n")
        f.write("- Models at the bottom are more robust or have fewer parameters affecting variance.\n")
    print(f"Saved textual report to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze tuning impact of baseline models.")
    parser.add_argument("--input", "-i", type=str, default="Tuned_Baseline_Results_All.csv", help="Path to the All Results CSV file")
    args = parser.parse_args()
    
    analyze_tuning_impact(args.input)
