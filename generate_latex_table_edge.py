import pandas as pd
import numpy as np

# Read the CSV
df = pd.read_csv('/home/jupyter-iec_duongnt/New Project/NFST/GMM-nfst/notebooks/analysis/anomaly/plots/Anomaly_Type_Summary_Table.csv')

# Models to include in the latex table and their latex names
models = {
    'LOC-NFST': '\\textbf{XXX (Our)}',
    'AUTOENCODER': 'AutoEncoder',
    'DASVDD': 'DASVDD',
    'KNN': 'KNN',
    'LUNAR': 'LUNAR',
    'CBLOF': 'CBLOF',
    'OCSVM': 'OCSVM',
    'AE1SVM': 'AE1SVM',
    'VAE': 'VAE',
    'PCA': 'PCA',
    'LODA': 'LODA',
    'IForest': 'IForest',
    'SUOD': 'SUOD',
    'COPOD': 'COPOD',
    'HBOS': 'HBOS',
    'ECOD': 'ECOD',
    'NEUTRALAD': 'NEUTRALAD',
    'ALAD': 'ALAD',
    'SO_GAAL': 'SO-GAAL'
}

latex_rows = []

def format_val(val, is_best, is_second_best, is_bold=False):
    if pd.isna(val):
        return "-"
    s = f"{val:.2f}"
    if is_best:
        return f"\\textcolor{{red}}{{\\textbf{{{s}}}}}"
    if is_second_best:
        return f"\\textcolor{{blue}}{{\\textit{{{s}}}}}"
    if is_bold:
        return f"\\textbf{{{s}}}"
    return s

for metric in ['local', 'global', 'cluster']:
    df[f'Avg_{metric}'] = df[[f'CICIoT_{metric}', f'N_BaIoT_{metric}', f'ToNIoT_{metric}', f'EdgeIIoTset_{metric}']].mean(axis=1)

df['Total_Mean'] = df[['Avg_local', 'Avg_global', 'Avg_cluster']].mean(axis=1)

# Find best and second best
best_vals = {}
for col in ['CICIoT_local', 'N_BaIoT_local', 'ToNIoT_local', 'EdgeIIoTset_local', 'Avg_local',
            'CICIoT_global', 'N_BaIoT_global', 'ToNIoT_global', 'EdgeIIoTset_global', 'Avg_global',
            'CICIoT_cluster', 'N_BaIoT_cluster', 'ToNIoT_cluster', 'EdgeIIoTset_cluster', 'Avg_cluster',
            'Total_Mean']:
    vals = df[df['clean_model'].isin(models.keys())][col].dropna().values
    if len(vals) > 0:
        sorted_vals = sorted(list(set(vals)), reverse=True)
        best_vals[col] = sorted_vals[0] if len(sorted_vals) > 0 else None
        second_best_vals = sorted_vals[1] if len(sorted_vals) > 1 else None
        best_vals[col + "_2nd"] = second_best_vals

for orig_name, latex_name in models.items():
    row_data = df[df['clean_model'] == orig_name]
    if len(row_data) == 0:
        continue
    row_data = row_data.iloc[0]
    
    row_str = [latex_name]
    
    for metric in ['local', 'global', 'cluster']:
        for ds in ['CICIoT', 'N_BaIoT', 'ToNIoT', 'EdgeIIoTset', 'Avg']:
            col_name = f'{ds}_{metric}'
            val = row_data[col_name]
            
            is_best = False
            is_second_best = False
            if pd.notna(val):
                if best_vals.get(col_name) is not None and abs(val - best_vals[col_name]) < 1e-4:
                    is_best = True
                elif best_vals.get(col_name + "_2nd") is not None and abs(val - best_vals[col_name + "_2nd"]) < 1e-4:
                    is_second_best = True
            
            is_bold = (orig_name == 'LOC-NFST') and not is_best and not is_second_best
            row_str.append(format_val(val, is_best, is_second_best, is_bold))
            
    # Total mean
    val = row_data['Total_Mean']
    is_best = False
    is_second_best = False
    if pd.notna(val):
        if best_vals.get('Total_Mean') is not None and abs(val - best_vals['Total_Mean']) < 1e-4:
            is_best = True
        elif best_vals.get('Total_Mean_2nd') is not None and abs(val - best_vals['Total_Mean_2nd']) < 1e-4:
            is_second_best = True
    is_bold = (orig_name == 'LOC-NFST') and not is_best and not is_second_best
    row_str.append(format_val(val, is_best, is_second_best, is_bold))
    
    latex_rows.append(" & ".join(row_str) + " \\\\")

print("\n".join(latex_rows))
