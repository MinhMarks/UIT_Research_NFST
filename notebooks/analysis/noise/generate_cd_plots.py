import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import operator
import math
import glob
import networkx
import seaborn as sns
from scipy.stats import wilcoxon, friedmanchisquare

# Standard font settings (with fallbacks for Linux/Notebook environments)
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Helvetica', 'sans-serif']

# ============================================================================
# DATA NORMALIZATION (Consistent with generate_best_results_report.py)
# ============================================================================
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
    
    # Filter out DevNet as requested
    df_models = df['model'] if 'model' in df.columns else pd.Series(['LOC-NFST']*len(df))
    norm_df['model'] = df_models
    norm_df['model'] = norm_df['model'].replace({'ourmodel': 'LOC-NFST'})
    
    # Exclude DevNet (case-insensitive)
    norm_df = norm_df[~norm_df['model'].str.lower().str.contains('devnet')].copy()
    
    if 'noise_percentage' in df.columns: norm_df['noise'] = df['noise_percentage'].astype(float)
    elif 'noise' in df.columns: norm_df['noise'] = df['noise'].astype(float)
    else: norm_df['noise'] = 0.0

    if 'scaler' in df.columns: norm_df['scaler'] = df['scaler']
    elif 'scaled' in df.columns: norm_df['scaler'] = df['scaled']
    else: norm_df['scaler'] = 'Unknown'

    norm_df['aucroc'] = pd.to_numeric(df['aucroc'], errors='coerce')
    
    # Load Time metrics for Pareto plot
    if 'time_test' in df.columns: norm_df['time_test'] = pd.to_numeric(df['time_test'], errors='coerce')
    elif 'time test' in df.columns: norm_df['time_test'] = pd.to_numeric(df['time test'], errors='coerce')
    else: norm_df['time_test'] = 0.0

    if 'time_train' in df.columns: norm_df['time_train'] = pd.to_numeric(df['time_train'], errors='coerce')
    elif 'time train' in df.columns: norm_df['time_train'] = pd.to_numeric(df['time train'], errors='coerce')
    else: norm_df['time_train'] = 0.0

    return norm_df.dropna(subset=['aucroc'])

# ============================================================================
# CD DIAGRAM LOGIC (Adopted from promt/main.py)
# ============================================================================

def graph_ranks(avranks, names, avg_value, p_values, cd=None, cdmethod=None, lowv=None, highv=None,
                width=24, textspace=5, reverse=True, filename=None, labels=True, **kwargs):
    """
    Fixed logic: Best models go to the RIGHT (near 1), Worst models go to the LEFT (near 22).
    This prevents lines from crossing the entire chart.
    Matching 'main.py' style but with better side selection and scaling.
    """
    k = len(avranks)
    width = float(width)
    textspace = float(textspace)

    def nth(l, n):
        n = lloc(l, n)
        return [a[n] for a in l]

    def lloc(l, n):
        if n < 0: return len(l[0]) + n
        else: return n

    if lowv is None:
        lowv = min(1, int(math.floor(min(avranks))))
    if highv is None:
        highv = max(len(avranks), int(math.ceil(max(avranks))))

    cline = 0.4
    scalewidth = width - 2 * textspace

    def rankpos(rank):
        if not reverse: a = rank - lowv
        else: a = highv - rank
        return textspace + scalewidth / (highv - lowv) * a

    distanceh = 0.25
    cline += distanceh
    
    # Higher spacing to prevent vertical overlap
    space_between_names = 0.6 
    label_size = 14
    metric_size = 12
    
    minnotsignificant = 0.8
    height = cline + (math.ceil(k / 2) + 1) * space_between_names + minnotsignificant + 1.0
    
    fig = plt.figure(figsize=(width, height))
    fig.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    hf = 1. / height
    wf = 1. / width
    def hfl(l): return [a * hf for a in l]
    def wfl(l): return [a * wf for a in l]

    ax.plot([0, 1], [0, 1], c="w")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    def line(l, color='k', **kwargs):
        ax.plot(wfl(nth(l, 0)), hfl(nth(l, 1)), color=color, **kwargs)

    def text(x, y, s, *args, **kwargs):
        ax.text(wf * x, hf * y, s, *args, **kwargs)

    line([(textspace, cline), (width - textspace, cline)], linewidth=2)

    bigtick = 0.3
    smalltick = 0.15
    linewidth = 2.0
    linewidth_sign = 6.0 

    for a in list(np.arange(lowv, highv, 0.5)) + [highv]:
        tick = smalltick
        if a == int(a): tick = bigtick
        line([(rankpos(a), cline - tick / 2), (rankpos(a), cline)], linewidth=2)

    # Top scale font
    tick_font = 14
    for a in range(lowv, highv + 1):
        text(rankpos(a), cline - 0.2, str(a), ha="center", va="bottom", size=tick_font)

    # WORST MODELS (Ranks k/2 to k) -> LEFT Side (Near 22 in reverse mode)
    for i in range(math.ceil(k / 2), k):
        idx = i # Original rank index
        # We want to stack them from bottom UP (index k..k/2) or top DOWN.
        # Let's stack them top down on the LEFT side.
        chei = cline + minnotsignificant + (i - math.ceil(k / 2)) * space_between_names
        line([(rankpos(avranks[idx]), cline), (rankpos(avranks[idx]), chei), (textspace - 0.1, chei)], linewidth=linewidth)
        
        name = names[idx]
        is_our = "LOC-NFST" in name
        f_weight = "bold" if is_our else "normal"
        f_color = "red" if is_our else "black"
        
        if labels:
            # Added bbox (white background) to prevent line crossing and increase zorder
            text(textspace + 2.0, chei, "{0:.2f} / {1:.2f}".format(avg_value[idx], avranks[idx]), 
                 ha="right", va="center", size=metric_size, color=f_color, zorder=20,
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.9, pad=1.5))
        text(textspace - 0.3, chei, name, ha="right", va="center", size=label_size, weight=f_weight, color=f_color, zorder=20)

    # BEST MODELS (Ranks 0 to k/2) -> RIGHT Side (Near 1 in reverse mode)
    for i in range(math.ceil(k / 2)):
        idx = i
        chei = cline + minnotsignificant + i * space_between_names
        line([(rankpos(avranks[idx]), cline), (rankpos(avranks[idx]), chei), (width - textspace + 0.1, chei)], linewidth=linewidth)
        
        name = names[i]
        is_our = "LOC-NFST" in name
        f_weight = "bold" if is_our else "normal"
        f_color = "red" if is_our else "black"
        
        if labels:
            # Added bbox (white background) to hide line behind text
            text(width - textspace - 2.0, chei, "{0:.2f} / {1:.2f}".format(avg_value[idx], avranks[idx]), 
                 ha="left", va="center", size=metric_size, color=f_color, zorder=20,
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.9, pad=1.5))
        text(width - textspace + 0.3, chei, name, ha="left", va="center", size=label_size, weight=f_weight, color=f_color, zorder=20)

    # DRAW CLIQUES (Blue significance bars)
    try:
        cliques = form_cliques(p_values, names)
        start = cline + 0.2
        side = -0.02
        height_inc = 0.2
        name_list = list(names)
        for clq in cliques:
            if len(clq) == 1: continue
            valid_indices = [name_list.index(name) for name in clq if name in names]
            if not valid_indices: continue
            min_idx = min(valid_indices)
            max_idx = max(valid_indices)
            line([(rankpos(avranks[min_idx]) - side, start), (rankpos(avranks[max_idx]) + side, start)], 
                 linewidth=linewidth_sign, color='blue', alpha=0.6)
            start += height_inc
    except Exception as e:
        print(f"Warning drawing cliques: {e}")

def form_cliques(p_values, nnames):
    m = len(nnames)
    g_data = np.zeros((m, m), dtype=np.int64)
    name_list = list(nnames)
    for p in p_values:
        if p[3] == False: # Not significant
            if p[0] in name_list and p[1] in name_list:
                i = name_list.index(p[0])
                j = name_list.index(p[1])
                g_data[min(i, j), max(i, j)] = 1
    g = networkx.Graph(g_data)
    return list(networkx.find_cliques(g))

def wilcoxon_holm(alpha=0.05, df_perf=None):
    classifiers = sorted(df_perf['classifier_name'].unique())
    datasets = sorted(df_perf['dataset_name'].unique())
    m = len(classifiers)
    n = len(datasets)
    
    # Test Friedman first
    friedman_p = friedmanchisquare(*(np.array(df_perf.loc[df_perf['classifier_name'] == c]['accuracy']) for c in classifiers))[1]
    if friedman_p >= alpha:
        print('Friedman test not significant. CD Diagram might not show meaningful results.')

    p_values = []
    for i in range(m - 1):
        c1 = classifiers[i]
        perf1 = np.array(df_perf.loc[df_perf['classifier_name'] == c1]['accuracy'], dtype=np.float64)
        for j in range(i + 1, m):
            c2 = classifiers[j]
            perf2 = np.array(df_perf.loc[df_perf['classifier_name'] == c2]['accuracy'], dtype=np.float64)
            p = wilcoxon(perf1, perf2, zero_method='pratt')[1]
            p_values.append((c1, c2, p, False))
    
    k = len(p_values)
    p_values.sort(key=operator.itemgetter(2))
    for i in range(k):
        new_alpha = float(alpha / (k - i))
        if p_values[i][2] <= new_alpha:
            p_values[i] = (p_values[i][0], p_values[i][1], p_values[i][2], True)
        else: break
            
    # Compute ranks
    # Pivot to get matrix: Rows = Dataset, Cols = Classifier
    pivot = df_perf.pivot(index='dataset_name', columns='classifier_name', values='accuracy')
    ranks_df = pivot.rank(ascending=False, axis=1)
    average_ranks = ranks_df.mean(axis=0).sort_values(ascending=True) # Ascending: 1.0 is best
    
    average_value = df_perf.groupby('classifier_name').agg({'accuracy': 'mean'}).reset_index()
    # Align average_value with sorted average_ranks
    average_value.classifier_name = average_value.classifier_name.astype("category")
    average_value.classifier_name = average_value.classifier_name.cat.set_categories(average_ranks.index)
    average_value = average_value.sort_values(["classifier_name"])
    
    return p_values, average_ranks, n, average_value

# ============================================================================
# ALTERNATIVE VISUALIZATIONS
# ============================================================================

def plot_ranked_heatmap(average_ranks, df_perf, output_path):
    """Draw a professional heatmap of AUC values by Dataset and Model."""
    # Pivot to Matrix: Rows=Model, Cols=Dataset
    pivot = df_perf.pivot(index='classifier_name', columns='dataset_name', values='accuracy')
    # Sort by overall rank
    pivot = pivot.reindex(average_ranks.index)
    
    plt.figure(figsize=(10, 14))
    sns.set(style="white")
    ax = sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'AUCROC'})
    
    # Highlight LOC-NFST in Red
    for label in ax.get_yticklabels():
        if "LOC-NFST" in label.get_text():
            label.set_color("red")
            label.set_weight("bold")
            
    plt.title("Performance Matrix (AUCROC) - Ranked from Top to Bottom", size=15)
    plt.xlabel("Dataset", size=12)
    plt.ylabel("Model", size=12)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_average_rank_bar(average_ranks, average_value, output_path):
    """Draw a clean horizontal bar chart of average ranks with AUCROC labels."""
    plt.figure(figsize=(10, 14))
    
    # Merge rank and auc for labeling
    # average_value should be aligned with average_ranks index
    auc_map = average_value.set_index('classifier_name')['accuracy'].to_dict()
    
    colors = ['red' if "LOC-NFST" in m else 'skyblue' for m in average_ranks.index]
    
    ax = average_ranks.plot(kind='barh', color=colors, edgecolor='black', alpha=0.8)
    ax.invert_yaxis() # Rank 1 at top
    
    plt.title("Overall Model Comparison (Ranked)", size=18, pad=20)
    plt.xlabel("Average Rank (Lower is Better)", size=14)
    plt.ylabel("Model", size=14)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    # Add Rank + AUCROC on the bars
    for i, model in enumerate(average_ranks.index):
        rank = average_ranks[model]
        auc = auc_map.get(model, 0.0)
        label = f"Rank: {rank:.2f} | AUC: {auc:.2f}%"
        
        # Highlight color for text
        is_our = "LOC-NFST" in model
        txt_color = "red" if is_our else "black"
        txt_weight = "bold" if is_our else "normal"
        
        ax.text(rank + 0.1, i, label, va='center', size=11, color=txt_color, fontweight=txt_weight)
        
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_performance_profiles(df_perf, output_path):
    """Draw Dolan-More Performance Profiles for 20+ models."""
    # Data is Model, Dataset, AUCROC
    pivot = df_perf.pivot(index='dataset_name', columns='classifier_name', values='accuracy')
    
    # Calculate performance ratios: r = max_auc_on_dataset / auc_of_model
    # (Since AUC is "bigger is better", we use max/val)
    max_perf = pivot.max(axis=1)
    ratios = pivot.divide(max_perf, axis=0)
    # Ratios >= 1.0. 1.0 is the best on that dataset.
    
    plt.figure(figsize=(12, 8))
    
    # Plot top 15 models or highlight specific ones to avoid rainbow mess
    # But for 22 models, we can use a thin line style
    tau_vals = np.linspace(1.0, 1.2, 100) # From 100% to 120% of best
    
    for model in ratios.columns:
        # Calculate rho(tau) = fraction of datasets where ratio <= tau
        rho = [ (ratios[model] <= t).mean() for t in tau_vals ]
        
        is_our = "LOC-NFST" in model
        color = 'red' if is_our else None
        alpha = 1.0 if is_our else 0.4
        lw = 3 if is_our else 1.5
        zorder = 10 if is_our else 1
        
        plt.plot(tau_vals, rho, label=model if is_our else None, 
                 color=color, alpha=alpha, linewidth=lw, zorder=zorder)
        
    plt.title("Dolan-Moré Performance Profiles (Robustness Analysis)", size=16)
    plt.xlabel(r"Performance Ratio $\tau$ (relative to best)", size=14)
    plt.ylabel(r"Fraction of Datasets $\rho(\tau)$", size=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_pareto_efficiency(df_best, output_path):
    """Scatter plot of AUCROC vs inference time."""
    # Group by model to get averages
    summary = df_best.groupby('model').agg({
        'aucroc': 'mean',
        'time_test': 'mean'
    })
    
    plt.figure(figsize=(12, 8))
    
    # Use log scale for time if differences are huge
    plt.xscale('log')
    
    for model in summary.index:
        x = summary.loc[model, 'time_test']
        y = summary.loc[model, 'aucroc']
        
        is_our = "LOC-NFST" in model
        color = 'red' if is_our else 'blue'
        size = 200 if is_our else 80
        alpha = 0.9 if is_our else 0.5
        
        plt.scatter(x, y, c=color, s=size, alpha=alpha, edgecolors='black')
        
        # Label only our model and 5 top competitors to keep it clean
        if is_our or y > summary['aucroc'].quantile(0.8):
            plt.text(x * 1.05, y, model, size=10, weight='bold' if is_our else 'normal')
            
    plt.title("Pareto Efficiency: Accuracy vs. Inference Time", size=16)
    plt.xlabel("Average Inference Time (s) - Log Scale", size=14)
    plt.ylabel("Average AUCROC (%)", size=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

# ============================================================================
# MAIN
# ============================================================================

def main():
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(_script_dir, "results")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Advanced ML Results Plotter ===")
    baseline_input = input("Enter path to Baseline Results (File or Dir): ").strip()
    model_input = input("Enter path to Model Results (File or Dir): ").strip()

    files = []
    for inp in [baseline_input, model_input]:
        if not inp: continue
        if os.path.isfile(inp) and inp.endswith('.csv'): files.append(('unified', inp))
        elif os.path.isdir(inp): files.extend(find_csv_files(inp))

    if not files:
        print("No results found.")
        return

    all_data = [load_and_normalize(f) for f in files]
    full_df = pd.concat(all_data, ignore_index=True)
    
    # Pre-process: Filter Noise=0
    df_clean = full_df[full_df['noise'] == 0.0].copy()
    
    # Best Scaler per (Model, Dataset)
    best_idx = df_clean.groupby(['model', 'dataset'])['aucroc'].idxmax()
    df_best = df_clean.loc[best_idx]
    
    # Save best scaler table
    scaler_pivot = df_best.pivot(index='model', columns='dataset', values='scaler')
    scaler_pivot.to_csv(os.path.join(_script_dir, "best_scaler_per_model.csv"))
    print(f"Best scaler table saved to {os.path.join(_script_dir, 'best_scaler_per_model.csv')}")

    # Prepare df_perf for Wilcoxon-Holm (columns: classifier_name, dataset_name, accuracy)
    df_perf = df_best[['model', 'dataset', 'aucroc']].rename(
        columns={'model': 'classifier_name', 'dataset': 'dataset_name', 'aucroc': 'accuracy'}
    )
    
    # Keep only models present in ALL datasets for a fair ranking
    counts = df_perf.groupby('classifier_name').size()
    max_nb = counts.max()
    valid_models = counts[counts == max_nb].index
    df_perf = df_perf[df_perf['classifier_name'].isin(valid_models)]
    
    # Filter df_best for Pareto (only valid models)
    df_best_valid = df_best[df_best['model'].isin(valid_models)]

    if df_perf.empty or len(df_perf['classifier_name'].unique()) < 2:
        print("Error: Not enough data for statistical analysis. Models must be present in all datasets.")
        return

    print(f">>> Computing rankings for {len(valid_models)} models across {max_nb} datasets...")
    p_values, average_ranks, n, average_value = wilcoxon_holm(df_perf=df_perf)

    # 1. Performance Heatmap
    heatmap_path = os.path.join(output_dir, "rank_heatmap.png")
    plot_ranked_heatmap(average_ranks, df_perf, heatmap_path)
    print(f"Ranked Heatmap saved to {heatmap_path}")

    # 2. Average Rank Bar Chart (With AUCROC)
    bar_path = os.path.join(output_dir, "rank_bar_chart.png")
    plot_average_rank_bar(average_ranks, average_value, bar_path)
    print(f"Rank Bar Chart saved to {bar_path}")

    # 3. Performance Profiles (Advanced Robustness Analysis)
    prof_path = os.path.join(output_dir, "performance_profiles.png")
    plot_performance_profiles(df_perf, prof_path)
    print(f"Performance Profiles saved to {prof_path}")

    # 4. Pareto Efficiency Plot (Accuracy vs Speed)
    pareto_path = os.path.join(output_dir, "pareto_efficiency.png")
    plot_pareto_efficiency(df_best_valid, pareto_path)
    print(f"Pareto Plot saved to {pareto_path}")

    # 5. CD Diagram (Legacy)
    try:
        graph_ranks(average_ranks.values, average_ranks.index, average_value['accuracy'].values, p_values,
                    reverse=True, labels=True)
        cd_diag_path = os.path.join(output_dir, "cd_diagram_custom.png")
        # plt.title("Critical Difference Diagram (Wilcoxon-Holm)", y=1.05)
        plt.savefig(cd_diag_path, bbox_inches='tight', dpi=300)
        print(f"CD Diagram (Legacy) saved to {cd_diag_path}")
    except Exception:
        print(f"Note: Standard CD Layout is too crowded for these models. Focus on Heatmap/Bar Chart/Profiles.")

if __name__ == "__main__":
    main()
