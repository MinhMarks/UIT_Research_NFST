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

def normalize_dataset_name(name):
    if not isinstance(name, str): return str(name)
    name_lower = name.lower()
    for ds in KNOWN_DATASETS:
        if ds.lower() in name_lower: return ds
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
    norm_df['dataset'] = df['dataset'].apply(normalize_dataset_name) if 'dataset' in df.columns else 'Unknown'
    
    df_models = df['model'] if 'model' in df.columns else pd.Series(['LOC-NFST']*len(df))
    norm_df['model'] = df_models
    norm_df['model'] = norm_df['model'].replace({'ourmodel': 'LOC-NFST'})
    
    norm_df = norm_df[~norm_df['model'].str.lower().str.contains('devnet')].copy()
    
    if 'noise_percentage' in df.columns: norm_df['noise'] = df['noise_percentage'].astype(float)
    elif 'noise' in df.columns: norm_df['noise'] = df['noise'].astype(float)
    else: norm_df['noise'] = 0.0

    # Fallback to extracting from filename since some files are named report_noise_1.csv without a noise column
    if norm_df['noise'].mean() == 0.0:
        if 'noise_1' in file_path.lower() or 'noise1' in file_path.lower(): norm_df['noise'] = 1.0
        elif 'noise_3' in file_path.lower() or 'noise3' in file_path.lower(): norm_df['noise'] = 3.0
        elif 'noise_5' in file_path.lower() or 'noise5' in file_path.lower(): norm_df['noise'] = 5.0

    norm_df['aucroc'] = pd.to_numeric(df['aucroc'], errors='coerce')
    
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
    
    combined_df = combined_df[combined_df['noise'].isin([1.0, 3.0, 5.0])]
    print(f"Loaded {len(combined_df)} rows for noise 1%, 3%, 5%.")
    
    if len(combined_df) == 0:
        print("No noise data available for 1%, 3%, or 5%. Exiting plot generation.")
        return

    # 1. First find the BEST configuration (Max AUCROC) for each model on each dataset at each noise level
    best_per_dataset = combined_df.groupby(['model', 'dataset', 'noise'])['aucroc'].max().reset_index()
    
    # 2. Average AUCROC for each model / noise level across the datasets
    avg_df = best_per_dataset.groupby(['model', 'noise'])['aucroc'].mean().reset_index()
    
    # 2. To find the top 4 baselines, robust approach: rank by average auc across all 3 noise levels
    overall_avg = avg_df.groupby('model')['aucroc'].mean().reset_index()
    
    baselines = overall_avg[overall_avg['model'] != 'LOC-NFST']
    top_4_baselines = baselines.sort_values(by='aucroc', ascending=False).head(4)['model'].tolist()
    
    # Target models
    target_models = ['LOC-NFST'] + top_4_baselines
    print(f"Plotting for Target Models: {target_models}")
    
    # Filter dataset
    plot_df = avg_df[avg_df['model'].isin(target_models)].copy()
    
    # Ensure there are actually target models to plot
    if len(plot_df) == 0:
        print("No data available for the target models.")
        return

    # Set up the plot
    plt.figure(figsize=(10, 6))
    
    # Sort models so LOC-NFST is first
    plot_df['model'] = pd.Categorical(plot_df['model'], categories=target_models, ordered=True)
    plot_df = plot_df.sort_values(['noise', 'model'])
    
    # Palette configuration
    # Create enough colors
    num_models = len(plot_df['model'].unique())
    palette = sns.color_palette("deep", num_models)
    
    # Ensure LOC-NFST is red if it exists in the data
    if 'LOC-NFST' in plot_df['model'].values:
        palette = [(0.85, 0.15, 0.15)] + list(palette[1:num_models])
    
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

    # Legend
    plt.legend(title='Model', title_fontsize='13', fontsize='12', loc='lower center', 
               bbox_to_anchor=(0.5, -0.25), ncol=num_models, frameon=True, borderaxespad=0.)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

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
        baselines = avg_df[avg_df['model'] != 'LOC-NFST']
        top_4_baselines = baselines.sort_values(by='aucroc', ascending=False).head(4)['model'].tolist()
        
        target_models = ['LOC-NFST'] + top_4_baselines
        print(f"Dataset {ds} Top Models: {target_models}")
        
        plot_df = ds_df[ds_df['model'].isin(target_models)].copy()
        if len(plot_df) == 0: continue
            
        plt.figure(figsize=(9, 5.5))
        plot_df['model'] = pd.Categorical(plot_df['model'], categories=target_models, ordered=True)
        plot_df = plot_df.sort_values(['noise', 'model'])
        
        num_models = len(plot_df['model'].unique())
        palette = sns.color_palette("deep", num_models)
        if 'LOC-NFST' in plot_df['model'].values:
            palette = [(0.85, 0.15, 0.15)] + list(palette[1:num_models])
        
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

        plt.legend(title='Model', title_fontsize='11', fontsize='10', loc='lower center', 
                   bbox_to_anchor=(0.5, -0.2), ncol=num_models, frameon=True, borderaxespad=0.)

        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"noise_levels_{ds}.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  -> Saved {out_path}")

def main():

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(_script_dir, "results")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Noise Comparison Chart Generator ===")
    baseline_input = input("Enter path to Baseline Results (File or Dir) [default: notebooks]: ").strip()
    model_input = input("Enter path to Model Results (File or Dir) [default: notebooks]: ").strip()

    if not baseline_input: baseline_input = r"d:\UIT\Research\Duongcpmputer\LOC-NFST\UIT_Research_NFST\notebooks"
    if not model_input: model_input = r"d:\UIT\Research\Duongcpmputer\LOC-NFST\UIT_Research_NFST\notebooks"

    files = []
    for inp in [baseline_input, model_input]:
        if not inp: continue
        if os.path.isfile(inp) and inp.endswith('.csv'): files.append(('unified', inp))
        elif os.path.isdir(inp): files.extend(find_csv_files(inp))

    if not files:
        print("No results found.")
        return

    mode = input("Choose mode [1] Mean across datasets, [2] Separate chart for each dataset (Default: 1): ").strip()
    if not mode: mode = '1'

    if mode == '2':
        generate_separate_noise_charts(files, output_dir)
    else:
        output_path = os.path.join(output_dir, "noise_levels_grouped.png")
        generate_noise_comparison_chart(files, output_path)

if __name__ == '__main__':
    main()
