import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def clean_value(val):
    if pd.isna(val):
        return 0.0
    if isinstance(val, str):
        val = val.replace('"', '').replace(',', '.')
    return float(val)

def process_anomaly_file(filepath, label):
    if not os.path.exists(filepath):
        print(f"Warning: File {filepath} not found.")
        return pd.DataFrame()

    # The actual header is on row index 2 (line 3)
    df = pd.read_csv(filepath, skiprows=2)
    
    # Drop empty rows
    df = df.dropna(subset=['Model'])
    
    # Clean the Average column
    if 'Average' not in df.columns:
        print(f"Error: 'Average' column not found in {filepath}")
        return pd.DataFrame()
        
    df['Average_Num'] = df['Average'].apply(clean_value)
    
    # Rename OurModel to LOC-NFST
    df.loc[df['Model'] == 'OurModel', 'Model'] = 'LOC-NFST'
    
    # Isolate Proposed Model
    proposed_df = df[df['Model'] == 'LOC-NFST']
    
    # Identify Top 4 Baselines (exclude proposed)
    baselines_df = df[df['Model'] != 'LOC-NFST']
    top4_baselines = baselines_df.nlargest(4, 'Average_Num')
    
    # Combine
    combined = pd.concat([proposed_df, top4_baselines])
    combined['Anomaly Type'] = label
    
    return combined

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    files = {
        'Cluster': os.path.join(script_dir, 'cluster.csv'),
        'Global': os.path.join(script_dir, 'global.csv'),
        'Local': os.path.join(script_dir, 'local.csv')
    }
    
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    for label, filepath in files.items():
        df = process_anomaly_file(filepath, label)
        if df.empty:
            continue
            
        # Ensure it's sorted by Average_Num descending
        df = df.sort_values(by='Average_Num', ascending=False)
        
        plt.figure(figsize=(10, 6))
        
        # Color proposed model distinctly
        colors = ['dodgerblue' if m == 'LOC-NFST' else 'salmon' for m in df['Model']]
        
        ax = sns.barplot(
            data=df,
            x='Model',
            y='Average_Num',
            palette=colors,
            edgecolor='black'
        )
        
        # Add exact percentages on top
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(f'{p.get_height():.2f}%', 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha = 'center', va = 'center', 
                            xytext = (0, 10), 
                            textcoords = 'offset points',
                            fontsize=11, fontweight='bold')
        
        # Dynamic Y-Axis Scale
        y_min = df['Average_Num'].min()
        y_max = df['Average_Num'].max()
        val_range = y_max - y_min
        
        # Give enough padding (e.g., 5-10% of range, but min 1.0 unit) for top labels
        padding = max(val_range * 0.2, 1.0)
        
        y_lower = max(0, y_min - padding * 0.5)
        y_upper = min(105, y_max + padding)
        
        plt.ylim(y_lower, y_upper)
        
        plt.title(f'Top Performing Algorithms: {label} Anomalies', fontweight='bold', fontsize=16)
        plt.ylabel('Average Detection Score (%)', fontweight='bold', fontsize=13)
        plt.xlabel('Algorithm', fontweight='bold', fontsize=13)
        plt.xticks(rotation=15)
        
        plt.tight_layout()
        
        output_path = os.path.join(script_dir, f"Anomaly_Type_{label}.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"Saved independent chart for {label} to: {output_path}")

if __name__ == "__main__":
    main()
