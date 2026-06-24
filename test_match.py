import pandas as pd
import numpy as np
import sys
import os

sys.path.append("./notebooks/analysis/noise")
from generate_cd_plots import load_and_normalize, EXCLUDE_MODELS_LIST, BASELINE_SCALER

f1 = "./notebooks/analysis/noise/baseline/Baseline_Noise_Results_Fixed.csv"
f2 = "./notebooks/experiments/results_memory_optimized.csv"

all_data = []
if os.path.exists(f1): all_data.append(load_and_normalize(('unified', f1)))
if os.path.exists(f2): all_data.append(load_and_normalize(('unified', f2)))

if not all_data:
    print("No data found")
    exit()

full_df = pd.concat(all_data, ignore_index=True)

df_clean = full_df[full_df['noise'] == 0.0].copy()

if EXCLUDE_MODELS_LIST:
    df_clean = df_clean[~df_clean['model'].isin(EXCLUDE_MODELS_LIST)]
    full_df = full_df[~full_df['model'].isin(EXCLUDE_MODELS_LIST)]

if BASELINE_SCALER:
    is_ours = df_clean['model'] == 'LOC-NFST'
    df_clean = df_clean[is_ours | (df_clean['scaler'] == BASELINE_SCALER)]
    is_ours_full = full_df['model'] == 'LOC-NFST'
    full_df = full_df[is_ours_full | (full_df['scaler'] == BASELINE_SCALER)]

best_idx = df_clean.groupby(['model', 'dataset'])['aucroc'].idxmax()
df_best = df_clean.loc[best_idx]

df_perf = df_best[['model', 'dataset', 'aucroc']].rename(columns={'model': 'classifier_name', 'dataset': 'dataset_name', 'aucroc': 'accuracy'})
counts = df_perf.groupby('classifier_name').size()
max_nb = counts.max()
valid_models = counts[counts == max_nb].index
df_perf = df_perf[df_perf['classifier_name'].isin(valid_models)]

cd_avg = df_perf.groupby('classifier_name').agg({'accuracy': 'mean'}).reset_index()
print("=== CD CHART (MEAN ACROSS DATASETS AT NOISE 0) ===")
print(cd_avg.sort_values('classifier_name').to_string(index=False))

df_noise = full_df[full_df['noise'].isin([0.0])].copy()
df_noise = df_noise[df_noise['model'].isin(valid_models)]

best_scalers = df_best[['model', 'dataset', 'scaler']].drop_duplicates()
df_noise = pd.merge(df_noise, best_scalers, on=['model', 'dataset', 'scaler'], how='inner')

agg_df = df_noise.groupby(['model', 'dataset', 'noise'])['aucroc'].max().reset_index()

latex_avg = agg_df.groupby('model').agg({'aucroc': 'mean'}).reset_index()
print("\n=== LATEX TABLE (MEAN ACROSS DATASETS AT NOISE 0) ===")
print(latex_avg.sort_values('model').to_string(index=False))

diff = cd_avg.set_index('classifier_name')['accuracy'] - latex_avg.set_index('model')['aucroc']
print("\n=== DIFFERENCE ===")
print(diff)
