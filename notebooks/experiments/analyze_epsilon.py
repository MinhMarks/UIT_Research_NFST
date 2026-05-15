import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def analyze_and_plot():
    results_path = "outputs/ablation_epsilon_results.csv"
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found.")
        return
        
    df = pd.read_csv(results_path)
    
    # Handle epsilon = 0.0 for log scale plotting
    # We replace 0.0 with 1e-12 to plot it on log scale
    df['Plot_Epsilon'] = df['Epsilon'].replace(0.0, 1e-12)
    
    os.makedirs("plots", exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    datasets = df['Dataset'].unique()
    
    # ==========================================
    # 1. Performance Sensitivity Plot
    # ==========================================
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Custom ticks to show 1e-12 as "0.0"
    base_ticks = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
    base_labels = ['1e-2', '1e-4', '1e-6', '1e-8', '1e-10', '0.0']

    for ds in datasets:
        subset = df[df['Dataset'] == ds]
        axes[0].plot(subset['Plot_Epsilon'], subset['AUC-PR'], marker='o', label=ds, linewidth=2.5)
        axes[1].plot(subset['Plot_Epsilon'], subset['AUC-ROC'], marker='s', label=ds, linewidth=2.5)
        if 'F1 Score' in subset.columns:
            axes[2].plot(subset['Plot_Epsilon'], subset['F1 Score'], marker='^', label=ds, linewidth=2.5)
        
    axes[0].set_xscale('log')
    axes[0].set_title('AUC-PR Sensitivity to Epsilon', fontweight='bold')
    axes[0].set_xlabel('Epsilon (Thresholding Rank)', fontweight='bold')
    axes[0].set_ylabel('AUC-PR (%)', fontweight='bold')
    axes[0].set_xticks(base_ticks)
    axes[0].set_xticklabels(base_labels)
    axes[0].invert_xaxis()
    axes[0].legend()
    
    axes[1].set_xscale('log')
    axes[1].set_title('AUC-ROC Sensitivity to Epsilon', fontweight='bold')
    axes[1].set_xlabel('Epsilon (Thresholding Rank)', fontweight='bold')
    axes[1].set_ylabel('AUC-ROC (%)', fontweight='bold')
    axes[1].set_xticks(base_ticks)
    axes[1].set_xticklabels(base_labels)
    axes[1].invert_xaxis()
    axes[1].legend()

    axes[2].set_xscale('log')
    axes[2].set_title('F1 Score Sensitivity to Epsilon', fontweight='bold')
    axes[2].set_xlabel('Epsilon (Thresholding Rank)', fontweight='bold')
    axes[2].set_ylabel('F1 Score (%)', fontweight='bold')
    axes[2].set_xticks(base_ticks)
    axes[2].set_xticklabels(base_labels)
    axes[2].invert_xaxis()
    axes[2].legend()

    plt.suptitle("Impact of Epsilon Threshold on Detection Performance", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/Epsilon_Sensitivity.png', dpi=300)
    plt.close()
    
    # ==========================================
    # 2. Rank & Cost Trade-off (Dual Axis)
    # ==========================================
    # To keep it clear, we do a plot per dataset or average them
    # Let's do an average line for clarity across the 2 datasets
    df_avg = df.groupby('Plot_Epsilon', as_index=False).mean(numeric_only=True)
    df_avg.sort_values(by='Plot_Epsilon', ascending=False, inplace=True)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Epsilon (Log Scale) -> Lower values mean more dimensions kept', fontweight='bold')
    ax1.set_ylabel('Estimated Subspace Dimension (Rank_Pt)', color=color, fontweight='bold')
    ax1.plot(df_avg['Plot_Epsilon'], df_avg['Rank_Pt'], color=color, marker='o', linewidth=2.5, label='Subspace Rank')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xscale('log')
    ax1.set_xticks(base_ticks)
    ax1.set_xticklabels(base_labels)
    ax1.invert_xaxis() # High epsilon on left, low epsilon (0.0) on right

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Train Time (seconds)', color=color, fontweight='bold')
    ax2.plot(df_avg['Plot_Epsilon'], df_avg['Train Time (s)'], color=color, marker='s', linestyle='--', linewidth=2.5, label='Train Time')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Ablation Study: Subspace Rank vs. Computational Cost", fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig('plots/Epsilon_Cost_Tradeoff.png', dpi=300)
    plt.close()

    print("Successfully generated Epsilon Sensitivity and Cost Tradeoff plots in the 'plots/' directory.")

if __name__ == "__main__":
    analyze_and_plot()
