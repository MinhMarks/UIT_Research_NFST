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
    norm_df['model'] = df['model'] if 'model' in df.columns else 'LOC-NFST'
    norm_df['model'] = norm_df['model'].replace({'ourmodel': 'LOC-NFST'})
    
    if 'noise_percentage' in df.columns: norm_df['noise'] = df['noise_percentage'].astype(float)
    elif 'noise' in df.columns: norm_df['noise'] = df['noise'].astype(float)
    else: norm_df['noise'] = 0.0

    if 'scaler' in df.columns: norm_df['scaler'] = df['scaler']
    elif 'scaled' in df.columns: norm_df['scaler'] = df['scaled']
    else: norm_df['scaler'] = 'Unknown'

    norm_df['aucroc'] = pd.to_numeric(df['aucroc'], errors='coerce')
    return norm_df.dropna(subset=['aucroc'])

# ============================================================================
# CD DIAGRAM LOGIC (Adopted from promt/main.py)
# ============================================================================

def graph_ranks(avranks, names, avg_value, p_values, cd=None, cdmethod=None, lowv=None, highv=None,
                width=8, textspace=1, reverse=False, filename=None, labels=False, **kwargs):
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
    k = len(avranks)
    linesblank = 0
    
    # DYNAMIC LAYOUT: Increase space for many models
    if k > 10:
        width = max(width, 14) # Increased from 10
        textspace = max(textspace, 3) # Increased from 2
    
    scalewidth = width - 2 * textspace
    space_between_names = 0.35 # Increased from 0.25
    
    def rankpos(rank):
        if not reverse: a = rank - lowv
        else: a = highv - rank
        return textspace + scalewidth / (highv - lowv) * a

    distanceh = 0.25
    cline += distanceh
    
    # Adjust height dynamically based on number of models and cliques
    minnotsignificant = 0.6 # Increased from max(2 * 0.2, linesblank)
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
    linewidth_sign = 4.0

    tick = None
    for a in list(np.arange(lowv, highv, 0.5)) + [highv]:
        tick = smalltick
        if a == int(a): tick = bigtick
        line([(rankpos(a), cline - tick / 2), (rankpos(a), cline)], linewidth=2)

    tick_size = 12 if k < 15 else 10 # Smaller tick labels for many models
    for a in range(lowv, highv + 1):
        text(rankpos(a), cline - tick / 2 - 0.05, str(a), ha="center", va="bottom", size=tick_size)

    def filter_names(name): return name

    # Dynamic font size
    label_size = 14 if k < 15 else 11
    
    for i in range(math.ceil(k / 2)):
        chei = cline + minnotsignificant + i * space_between_names
        line([(rankpos(avranks[i]), cline), (rankpos(avranks[i]), chei), (textspace - 0.1, chei)], linewidth=linewidth)
        
        # Highlight our model
        name = names[i]
        is_our = "LOC-NFST" in name
        f_weight = "bold" if is_our else "normal"
        f_color = "red" if is_our else "black"
        
        if labels:
            # Shift label further right to avoid overlap with name on the left
            text(textspace + 1.2, chei - 0.05, "{0:.2f} / {1:.2f}".format(avg_value[i], avranks[i]), 
                 ha="right", va="center", size=label_size-2, color=f_color, alpha=0.7)
        text(textspace - 0.2, chei, name, ha="right", va="center", size=label_size, weight=f_weight, color=f_color)

    for i in range(math.ceil(k / 2), k):
        chei = cline + minnotsignificant + (k - i - 1) * space_between_names
        line([(rankpos(avranks[i]), cline), (rankpos(avranks[i]), chei), (textspace + scalewidth + 0.1, chei)], linewidth=linewidth)
        
        name = names[i]
        is_our = "LOC-NFST" in name
        f_weight = "bold" if is_our else "normal"
        f_color = "red" if is_our else "black"
        
        if labels:
            # Shift label further left to avoid overlap with name on the right
            text(textspace + scalewidth - 1.2, chei - 0.05, "{0:.2f} / {1:.2f}".format(avg_value[i], avranks[i]), 
                 ha="left", va="center", size=label_size-2, color=f_color, alpha=0.7)
        text(textspace + scalewidth + 0.2, chei, name, ha="left", va="center", size=label_size, weight=f_weight, color=f_color)

    # draw no significant lines (cliques)
    cliques = form_cliques(p_values, names)
    start = cline + 0.2
    side = -0.02
    height_inc = 0.15 # Increased space between bars
    achieved_half = False
    for clq in cliques:
        if len(clq) == 1: continue
        name_list = list(names)
        indices = [name_list.index(name) for name in clq if name in name_list]
        if not indices: continue
        min_idx = min(indices)
        max_idx = max(indices)
        
        if min_idx >= len(names) / 2 and not achieved_half:
            start = cline + 0.25
            achieved_half = True
        
        line([(rankpos(avranks[min_idx]) - side, start), (rankpos(avranks[max_idx]) + side, start)], linewidth=linewidth_sign, color='blue', alpha=0.7)
        start += height_inc

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
# MAIN
# ============================================================================

def main():
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(_script_dir, "results")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Critical Difference Plotter (promt/main.py style) ===")
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
    
    if df_perf.empty or len(df_perf['classifier_name'].unique()) < 2:
        print("Error: Not enough data for CD Diagram. Models must be present in all datasets.")
        return

    print(f">>> Computing CD statistics for {len(valid_models)} models across {max_nb} datasets...")
    p_values, average_ranks, n, average_value = wilcoxon_holm(df_perf=df_perf)

    # Plot
    graph_ranks(average_ranks.values, average_ranks.index, average_value['accuracy'].values, p_values,
                reverse=True, width=10, textspace=2, labels=True)
    
    plt.title("Critical Difference Diagram (Wilcoxon-Holm)", y=0.9)
    cd_diag_path = os.path.join(output_dir, "cd_diagram_custom.png")
    plt.savefig(cd_diag_path, bbox_inches='tight')
    print(f"CD Diagram saved to {cd_diag_path}")

if __name__ == "__main__":
    main()
