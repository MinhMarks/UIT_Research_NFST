import os
import glob
import pandas as pd
import numpy as np
import logging
from sklearn.mixture import GaussianMixture

def setup_logger(name="Anomaly_Generator"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', "%H:%M:%S")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

log = setup_logger()

_script_dir = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.normpath(os.path.join(_script_dir, '..', 'Datascaled', 'Official_OC_Data'))
OUTPUT_DIR = os.path.normpath(os.path.join(_script_dir, '..', 'Datascaled', 'Official_Anomaly_Data'))

OUTLIER_MODES = {
    'local': 5,
    'cluster': 5,
    'global': 1.1
}

def generate_realistic_synthetic(X, y, realistic_synthetic_mode, alpha:float, percentage:float, seed:int=42):
    """
    Generates synthetic Normal and Anomaly datasets using Gaussian Mixture Models.
    Matches logic from GenerateOutlierData+.ipynb
    """
    if realistic_synthetic_mode not in ['local', 'cluster', 'global']:
        raise NotImplementedError()

    pts_n = len(np.where(y == 0)[0])
    pts_a = len(np.where(y == 1)[0])

    X_normal = X[y == 0]

    metric_list = []
    # To save time on large datasets, limit n_components range
    n_components_list = list(np.arange(1, 6))

    for n_components in n_components_list:
        gm = GaussianMixture(n_components=n_components, random_state=seed).fit(X_normal)
        metric_list.append(gm.bic(X_normal))

    best_n_components = n_components_list[np.argmin(metric_list)]
    gm = GaussianMixture(n_components=best_n_components, random_state=seed).fit(X_normal)

    # Synthetic Normal Data 
    X_synthetic_normal = gm.sample(pts_n)[0]

    # Synthetic Anomaly Data
    if realistic_synthetic_mode == 'local':
        gm.covariances_ = alpha * gm.covariances_
        X_synthetic_anomalies = gm.sample(pts_a)[0]
    elif realistic_synthetic_mode == 'cluster':
        gm.means_ = alpha * gm.means_
        X_synthetic_anomalies = gm.sample(pts_a)[0]
    elif realistic_synthetic_mode == 'global':
        X_synthetic_anomalies = []
        for i in range(X_synthetic_normal.shape[1]):
            low = np.min(X_synthetic_normal[:, i]) * (1 + percentage)
            high = np.max(X_synthetic_normal[:, i]) * (1 + percentage)
            X_synthetic_anomalies.append(np.random.uniform(low=low, high=high, size=pts_a))
        X_synthetic_anomalies = np.array(X_synthetic_anomalies).T

    X_combined = np.concatenate((X_synthetic_normal, X_synthetic_anomalies), axis=0)
    y_combined = np.append(np.repeat(0, X_synthetic_normal.shape[0]), np.repeat(1, X_synthetic_anomalies.shape[0]))

    return X_combined, y_combined

def process_datasets():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log.info(f"Input Directory: {INPUT_DIR}")
    log.info(f"Target Output Directory: {OUTPUT_DIR}")

    # Find all train files to identify dataset & scaler combos
    train_files = glob.glob(os.path.join(INPUT_DIR, "Train_*_data_*.csv"))
    
    if not train_files:
        log.warning(f"No Train files found in {INPUT_DIR}. Please generate OC datasets first.")
        return

    for train_path in train_files:
        filename = os.path.basename(train_path)
        # File format: Train_{scaler}_data_{dataset}.csv
        parts = filename.replace("Train_", "").replace(".csv", "").split("_data_")
        if len(parts) != 2: continue
        scaler_name, dataset_name = parts[0], parts[1]
        
        test_filename = f"Test_{scaler_name}_data_{dataset_name}.csv"
        test_path = os.path.join(INPUT_DIR, test_filename)
        
        if not os.path.exists(test_path):
            log.warning(f"Missing test file for {filename}, skipping.")
            continue
            
        log.info(f"=== Processing Pre-Scaled: {dataset_name} | {scaler_name} ===")
        
        # Load pre-scaled data
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        y_train = train_df.pop('label').values
        y_test = test_df.pop('label').values
        
        X_train = train_df.to_numpy()
        X_test = test_df.to_numpy()
        
        X_all = np.vstack((X_train, X_test))
        y_all = np.concatenate((y_train, y_test))
        feature_columns = train_df.columns
        
        # Guarantee no inf/nan from previous corruptions
        X_all[np.isnan(X_all)] = 0
        X_all[np.isinf(X_all)] = 0
        
        for mode, alpha in OUTLIER_MODES.items():
            out_train_name = f"Train_{mode}_{scaler_name}_data_{dataset_name}.csv"
            out_test_name = f"Test_{mode}_{scaler_name}_data_{dataset_name}.csv"
            
            # Skip if already exists
            if os.path.exists(os.path.join(OUTPUT_DIR, out_train_name)):
                log.info(f"  [Skipping] {mode.upper()} already exists.")
                continue

            log.info(f"  -> Generating {mode.upper()} Outliers...")
            
            try:
                X_syn, y_syn = generate_realistic_synthetic(X_all, y_all, mode, alpha, percentage=0.1)
                
                syn_df = pd.DataFrame(X_syn, columns=feature_columns)
                syn_df['label'] = y_syn
                
                df_syn_normal = syn_df[syn_df['label'] == 0].sample(frac=1, random_state=42).reset_index(drop=True)
                df_syn_anomaly = syn_df[syn_df['label'] == 1].sample(frac=1, random_state=42).reset_index(drop=True)
                
                n_train_norm = len(X_train) # Mimic exact same sizes! e.g., 30000
                
                final_train = df_syn_normal.iloc[:n_train_norm].copy()
                final_test_normal = df_syn_normal.iloc[n_train_norm:].copy()
                
                final_test = pd.concat([final_test_normal, df_syn_anomaly], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
                
                final_train.to_csv(os.path.join(OUTPUT_DIR, out_train_name), index=False)
                final_test.to_csv(os.path.join(OUTPUT_DIR, out_test_name), index=False)
                
                log.info(f"     [+] Saved {out_train_name} & {out_test_name}")
            except Exception as e:
                log.error(f"     [!] Failed on {mode}: {e}")

if __name__ == "__main__":
    process_datasets()
