import os
import pandas as pd
import glob
from datetime import datetime

KNOWN_DATASETS = ['data_CICIoT2023', 'data_ToNIoT', 'data_N_BaIoT', 'data_BoTIoT']

def normalize_dataset_name(name):
    """Extract known dataset name from string like 'Train_StandardScaler_data_CICIoT2023.csv'."""
    if not isinstance(name, str):
        return str(name)
    
    # Check if any known dataset name is inside the string
    for ds in KNOWN_DATASETS:
        if ds in name:
            return ds
            
    # Fallback: strip path and extension
    base = os.path.basename(name)
    if base.endswith('.csv'):
        base = base[:-4]
    return base

def find_csv_files(root_dir):
    """Recursively find all CSV files and identify if they are baseline or model results."""
    csv_files = glob.glob(os.path.join(root_dir, "**", "*.csv"), recursive=True)
    results = []
    for f in csv_files:
        try:
            # Skip empty or non-result CSVs
            if os.path.getsize(f) < 100:
                continue
            
            df = pd.read_csv(f)
            cols = [c.lower() for c in df.columns]
            
            # Identify Baseline type
            if 'model' in cols and 'parameters' in cols and 'aucroc' in cols:
                results.append(('baseline', f))
            # Identify NFST/Model type (different column names)
            elif 'ncluster' in cols and 'aucroc' in cols and ('scaler' in cols or 'noise_percentage' in cols):
                results.append(('model', f))
        except Exception as e:
            print(f"Skipping {f} due to error: {e}")
    return results

def load_and_normalize(file_info):
    type_info, file_path = file_info
    df = pd.read_csv(file_path)
    
    # Standardize column names to lower case for easier mapping
    df.columns = [c.lower() for c in df.columns]
    
    normalized_df = pd.DataFrame()
    
    # Map common columns
    normalized_df['dataset'] = df['dataset'].apply(normalize_dataset_name)
    normalized_df['source_file'] = os.path.basename(file_path)
    
    if type_info == 'baseline':
        normalized_df['model'] = df['model']
        normalized_df['noise'] = df['noise'].astype(float)
        normalized_df['scaler'] = df['scaled'] if 'scaled' in df.columns else 'Unknown'
        normalized_df['params'] = df['parameters']
    else:
        # For the proposed model, we can try to get the name from the filename or use a default
        fname = os.path.basename(file_path).lower()
        if 'anomaly_template' in fname:
            normalized_df['model'] = 'NFST-AnomalyTemplate'
        else:
            normalized_df['model'] = 'NFST-MemOpt'
            
        normalized_df['noise'] = df['noise_percentage'].astype(float)
        normalized_df['scaler'] = df['scaler'] if 'scaler' in df.columns else 'Unknown'
        normalized_df['params'] = df['ncluster'].apply(lambda x: f"n_clusters={x}")

    # Performance metrics (ensure they exist)
    metrics = ['aucroc', 'aucpr', 'accuracy', 'mcc', 'f1 score', 'precision', 'recall']
    for m in metrics:
        if m in df.columns:
            normalized_df[m] = pd.to_numeric(df[m], errors='coerce')
    
    # Time and RAM
    if 'time train' in df.columns: 
        normalized_df['time_train'] = df['time train']
    if 'time test' in df.columns: 
        normalized_df['time_test'] = df['time test']
    if 'peak ram train (mb)' in df.columns: 
        normalized_df['peak_ram_train'] = df['peak ram train (mb)']
    if 'peak ram test (mb)' in df.columns: 
        normalized_df['peak_ram_test'] = df['peak ram test (mb)']
    
    return normalized_df

def main():
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("=== Results Report Generator ===")
    baseline_input = input("Enter path to Baseline Results CSV (e.g., notebooks/baselines/outputs/Experiment_.../Tuned_Baseline_Results_All.csv): ").strip()
    model_input = input("Enter path to Model Results (File or Directory containing dataset CSVs): ").strip()

    files = []
    
    # Process Baseline Input
    if os.path.isfile(baseline_input) and baseline_input.endswith('.csv'):
        files.append(('baseline', baseline_input))
    elif os.path.isdir(baseline_input):
        files.extend(find_csv_files(baseline_input))
    
    # Process Model Input
    if os.path.isfile(model_input) and model_input.endswith('.csv'):
        files.append(('model', model_input))
    elif os.path.isdir(model_input):
        files.extend(find_csv_files(model_input))

    if not files:
        print("No valid CSV files found at specified paths.")
        return
    
    print(f"Found {len(files)} result files to process.")
    
    all_data = []
    for f_info in files:
        print(f"Processing: {f_info[1]}")
        all_data.append(load_and_normalize(f_info))
    
    if not all_data:
        print("No result data found.")
        return
        
    full_df = pd.concat(all_data, ignore_index=True)
    
    # Drop rows with NaN in key columns
    full_df = full_df.dropna(subset=['dataset', 'noise', 'aucroc'])
    
    # Group by Dataset and Noise to create separate reports
    output_base_dir = os.path.join(_script_dir, "best_results_reports")
    os.makedirs(output_base_dir, exist_ok=True)
    
    unique_combinations = full_df[['dataset', 'noise']].drop_duplicates()
    
    for _, row in unique_combinations.iterrows():
        ds = row['dataset']
        ns = row['noise']
        
        subset = full_df[(full_df['dataset'] == ds) & (full_df['noise'] == ns)]
        
        # Get best result for EACH model in this group, sorted by model name
        best_per_model = subset.sort_values(['model', 'aucroc'], ascending=[True, False]).groupby('model').first().reset_index()
        best_per_model = best_per_model.sort_values('model')
        
        # Format for output
        out_filename = f"{ds}_Noise_{int(ns)}.csv"
        out_path = os.path.join(output_base_dir, out_filename)
        best_per_model.to_csv(out_path, index=False)
        print(f"Generated report: {out_path}")

    # Generate a summary showing the best result for each model across all dataset/noise combinations, sorted by dataset and model
    overall_best = full_df.sort_values(['dataset', 'noise', 'model', 'aucroc'], ascending=[True, True, True, False]).groupby(['dataset', 'noise', 'model']).first().reset_index()
    summary_path = os.path.join(_script_dir, "all_models_summary.csv")
    overall_best.to_csv(summary_path, index=False)
    print(f"Generated overall summary: {summary_path}")

if __name__ == "__main__":
    main()
