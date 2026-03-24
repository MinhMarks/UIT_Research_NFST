import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare
import glob

# ============================================================================
# SETTINGS & PATHS
# ============================================================================
KNOWN_DATASETS = ['BoTIoT', 'ToNIoT', 'N_BaIoT', 'CICIoT']

def normalize_dataset_name(name):
    if not isinstance(name, str): return str(name)
    name_lower = name.lower()
    for ds in KNOWN_DATASETS:
        if ds.lower() in name_lower: return ds
    # Fallback to base name if no match
    base = os.path.basename(name)
    if base.endswith('.csv'): base = base[:-4]
    return base

def load_all_results(root_dir):
    """Find and load all result CSVs in the project."""
    csv_files = glob.glob(os.path.join(root_dir, "**", "*.csv"), recursive=True)
    all_dfs = []
    
    for f in csv_files:
        try:
            if os.path.getsize(f) < 100: continue
            df = pd.read_csv(f)
            cols = [c.lower() for c in df.columns]
            
            if 'aucroc' in cols and ('model' in cols or 'dataset' in cols):
                df.columns = [c.lower() for c in df.columns]
                
                # Standardize columns
                temp_df = pd.DataFrame()
                temp_df['dataset'] = df['dataset'].apply(normalize_dataset_name) if 'dataset' in df.columns else 'Unknown'
                temp_df['model'] = df['model'] if 'model' in df.columns else 'LOC-NFST'
                
                if 'noise_percentage' in df.columns: temp_df['noise'] = df['noise_percentage'].astype(float)
                elif 'noise' in df.columns: temp_df['noise'] = df['noise'].astype(float)
                else: temp_df['noise'] = 0.0

                if 'scaler' in df.columns: temp_df['scaler'] = df['scaler']
                elif 'scaled' in df.columns: temp_df['scaler'] = df['scaled']
                else: temp_df['scaler'] = 'Unknown'

                temp_df['aucroc'] = pd.to_numeric(df['aucroc'], errors='coerce')
                temp_df = temp_df.dropna(subset=['aucroc'])
                all_dfs.append(temp_df)
        except Exception:
            pass
            
    if not all_dfs: return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)

# ============================================================================
# STATISTICAL PLOTTING (CD DIAGRAM)
# ============================================================================
def draw_cd_diagram(df_pivot, alpha=0.05, output_path='cd_diagram.png'):
    """
    Draw a Critical Difference (CD) Diagram based on Nemenyi post-hoc test.
    df_pivot: Rows = Datasets, Columns = Models, Values = AUCROC
    """
    n_datasets, n_models = df_pivot.shape
    if n_datasets < 2 or n_models < 2:
        print("Not enough data to draw CD diagram.")
        return

    # 1. Ranks (1 is best)
    ranks = df_pivot.rank(axis=1, ascending=False)
    avg_ranks = ranks.mean(axis=0).sort_values()
    
    # 2. Friedman Test
    try:
        stat, p = friedmanchisquare(*[df_pivot[col] for col in df_pivot.columns])
        print(f"Friedman test: stat={stat:.3f}, p={p:.4e}")
    except Exception as e:
        print(f"Friedman test failed: {e}")
        p = 1.0

    # 3. Nemenyi Critical Difference
    # Studentized range statistic q_alpha for alpha=0.05
    q_alphas = {
        2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 
        9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391,
        16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544
    }
    q = q_alphas.get(n_models, 3.6) # Fallback
    cd = q * np.sqrt(n_models * (n_models + 1) / (6 * n_datasets))
    print(f"CD value (alpha={alpha}): {cd:.3f}")

    # 4. Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Range of ranks
    low, high = 1, n_models
    ax.set_xlim(low - 0.5, high + 0.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xticks(np.arange(low, high + 1))
    ax.set_xlabel('Average Rank (Lower is Better)')
    ax.invert_xaxis() # Better models (rank 1) on the right usually, but we can do A-B style
    
    # Rank line
    ax.axhline(0, color='black', linewidth=1.5)
    
    # Model markers and labels
    model_ranks = avg_ranks.to_dict()
    sorted_models = list(avg_ranks.index)
    
    for i, model in enumerate(sorted_models):
        rank = model_ranks[model]
        # Alternate sides for labels
        side = 1 if i % 2 == 0 else -1
        y_tick = 0.1 * side
        y_text = 0.5 * side
        
        ax.plot([rank, rank], [0, y_tick], color='black', linewidth=1)
        ax.text(rank, y_text, f"{model}\n({rank:.2f})", 
                ha='center', va='center', fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        ax.plot([rank, rank], [y_tick, y_text*0.8], color='gray', linestyle=':', linewidth=0.8)

    # CD bar
    ax.plot([low, low + cd], [1.2, 1.2], color='red', linewidth=4)
    ax.text(low + cd/2, 1.3, f"CD = {cd:.2f}", color='red', ha='center', fontweight='bold')

    # Draw Cliques (Groups not significantly different)
    # Find all groups where the max difference is < CD
    def find_cliques(ranks_dict, cd_val):
        sorted_items = sorted(ranks_dict.items(), key=lambda x: x[1])
        cliques = []
        for i in range(len(sorted_items)):
            clique = [sorted_items[i]]
            for j in range(i + 1, len(sorted_items)):
                if abs(sorted_items[i][1] - sorted_items[j][1]) <= cd_val:
                    clique.append(sorted_items[j])
                else:
                    break
            if len(clique) > 1:
                cliques.append(clique)
        
        # Filter: only keep maximal cliques
        maximal = []
        for c in cliques:
            is_sub = False
            for m in cliques:
                if c == m: continue
                # Check if c is a subset of m
                if all(item in m for item in c):
                    is_sub = True
                    break
            if not is_sub:
                maximal.append(c)
        return maximal

    cliques = find_cliques(model_ranks, cd)
    y_bar = -0.3
    for clique in cliques:
        c_ranks = [item[1] for item in clique]
        ax.plot([min(c_ranks), max(c_ranks)], [y_bar, y_bar], color='blue', linewidth=5, alpha=0.6)
        y_bar -= 0.15

    ax.axis('off')
    plt.title(f"Critical Difference Diagram (alpha={alpha})\nFriedman p-value: {p:.4e}", pad=30)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"CD Diagram saved to {os.path.abspath(output_path)}")
    plt.close()

# ============================================================================
# MAIN
# ============================================================================
def main():
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(_script_dir, "..", "..", ".."))
    
    print(">>> Loading all results...")
    df = load_all_results(project_root)
    if df.empty:
        print("No results found. Please run experiments first.")
        return

    # Unify Model names (important for LOC-NFST)
    df['model'] = df['model'].replace({'ourmodel': 'LOC-NFST'})

    # Filter for Noise=0 (Clean comparison)
    df_clean = df[df['noise'] == 0.0].copy()
    print(f">>> Found {len(df_clean)} rows with Noise=0.0")
    print("Dataset counts (Normalized):")
    print(df_clean['dataset'].value_counts())
    print("Model counts (Total rows):")
    print(df_clean['model'].value_counts().head(5))
    print(">>> Generating Best Scaler table...")
    # Group by Model and Dataset, pick best scaler
    best_scaler_idx = df_clean.groupby(['model', 'dataset'])['aucroc'].idxmax()
    df_best_scalers = df_clean.loc[best_scaler_idx]
    
    # Pivot to show Scaler per (Model, Dataset)
    scaler_pivot = df_best_scalers.pivot(index='model', columns='dataset', values='scaler')
    scaler_pivot.to_csv(os.path.join(_script_dir, "best_scaler_per_model.csv"))
    print(f"Best scaler table saved to {os.path.join(_script_dir, 'best_scaler_per_model.csv')}")

    # 2. CD DIAGRAM
    print(">>> Generating CD Diagram...")
    # Matrix of (Dataset x Model) using the best AUCROC found across scalers
    aucroc_pivot = df_best_scalers.pivot(index='dataset', columns='model', values='aucroc')
    
    # Drop models with too many NaNs if any (Friedman needs complete rows)
    aucroc_pivot = aucroc_pivot.dropna(axis=1) # Drop models not present in all datasets
    
    if aucroc_pivot.shape[1] < 2:
        print("Error: Need at least 2 models present in all 4 datasets to compute ranks.")
        print(f"Available models after filtering: {aucroc_pivot.columns.tolist()}")
        return

    draw_cd_diagram(aucroc_pivot, output_path=os.path.join(_script_dir, "cd_diagram.png"))

if __name__ == "__main__":
    main()
