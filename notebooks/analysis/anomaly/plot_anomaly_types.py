import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set academic style
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 15,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 13,
    'font.family': 'serif',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

def clean_value(val):
    if pd.isna(val) or val == '':
        return 0.0
    if isinstance(val, str):
        # Remove quotes and whitespace
        val = val.replace('"', '').strip()
        # Replace comma with dot
        val = val.replace(',', '.')
    try:
        return float(val)
    except ValueError:
        return 0.0

def plot_anomaly_type(csv_path, output_path, type_label):
    # Load data, skipping the first two header rows (Anomaly Type title and empty row)
    # The actual header is on the 3rd row (index 2)
    df = pd.read_csv(csv_path, skiprows=2)
    
    # Clean the column names (remove empty ones if any)
    df = df.dropna(axis=1, how='all')
    
    # Map "OurModel" to "LOC-NFST"
    df['Model'] = df['Model'].replace('OurModel', 'LOC-NFST')
    
    # Clean numeric columns
    metric_cols = ['BoTIoT', 'CICIoT2023', 'NBaIoT', 'ToNIoT', 'Average']
    for col in metric_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_value)
            
    # Sort by Average AUCROC and take Top 5
    # Ensure LOC-NFST is included even if not in Top 5 (though it usually is)
    df_sorted = df.sort_values('Average', ascending=False)
    top_5 = df_sorted.head(5).copy()
    
    if 'LOC-NFST' not in top_5['Model'].values:
        our_model_row = df[df['Model'] == 'LOC-NFST']
        if not our_model_row.empty:
            top_5 = pd.concat([top_5.iloc[:4], our_model_row])
    
    # Re-sort Top 5 for aesthetic bar ranking (Descending)
    top_5 = top_5.sort_values('Average', ascending=False)
    
    # Set up the plot
    plt.figure(figsize=(10, 6))
    
    # Define colors: LOC-NFST is charcoal/gray, baselines are muted blue
    colors = ['#5C5C5C' if m == 'LOC-NFST' else '#2E6DA4' for m in top_5['Model']]
    
    ax = sns.barplot(
        data=top_5,
        x='Model',
        y='Average',
        palette=colors,
        hue='Model',
        legend=False
    )
    
    # Add value labels on top of bars
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(
                f'{h:.2f}%',
                (p.get_x() + p.get_width() / 2., h),
                ha='center', va='bottom',
                xytext=(0, 5), textcoords='offset points',
                fontsize=13, fontweight='bold'
            )
            
    # Refine axes
    plt.ylabel('Average AUC-ROC (%)', fontsize=15)
    plt.xlabel('Algorithm', fontsize=15)
    plt.ylim(0, 115) # Leave space for labels
    
    # Draw a line at 100%
    plt.axhline(y=100, color='gray', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved {type_label} plot to {output_path}")

if __name__ == "__main__":
    base_dir = r"d:\UIT\Research\Duongcpmputer\LOC-NFST\UIT_Research_NFST\notebooks\analysis\anomaly\draft"
    out_dir = r"d:\UIT\Research\Duongcpmputer\LOC-NFST\UIT_Research_NFST\notebooks\analysis\anomaly\plots"
    os.makedirs(out_dir, exist_ok=True)
    
    configs = [
        ('local.csv', 'Anomaly_Type_Local.png', 'Local Anomaly'),
        ('cluster.csv', 'Anomaly_Type_Cluster.png', 'Cluster Anomaly'),
        ('global.csv', 'Anomaly_Type_Global.png', 'Global Anomaly')
    ]
    
    for csv_file, out_file, label in configs:
        csv_path = os.path.join(base_dir, csv_file)
        out_path = os.path.join(out_dir, out_file)
        if os.path.exists(csv_path):
            plot_anomaly_type(csv_path, out_path, label)
        else:
            print(f"Warning: {csv_path} not found.")
