import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, QuantileTransformer, RobustScaler
import logging

def setup_logger(name="OC_Generator"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', "%H:%M:%S")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

log = setup_logger()

# Setup paths (Server deployment friendly)
_script_dir = os.path.dirname(os.path.abspath(__file__))
# Default DATA_DIR (where output CSVs will be saved)
DATA_DIR = os.environ.get('DATA_DIR', os.path.normpath(os.path.join(_script_dir, '..', 'Datascaled', 'Official_OC_Data')))
# RawData folder (where dataset classes are)
RAW_DATA_DIR = _script_dir
# DataLoader folder (required by RawData classes)
LOC_NFST_DIR = os.path.normpath(os.path.join(_script_dir, '..'))

sys.path.append(RAW_DATA_DIR)
sys.path.append(LOC_NFST_DIR)  # For DataLoader.utils import *

try:
    from BoTIoT import BoTIoT
    from CICIoT2023 import CICIoT2023
    from N_BaIoT import N_BaIoT
    from ToNIoT import ToNIoT
    from EdgeIIoTset import EdgeIIoTset
    from FiveGNIDD import FiveGNIDD
    from IoTID20 import IoTID20
except ImportError as e:
    log.error(f"Cannot import RawData classes: {e}. Make sure PYTHONPATH is correctly set.")
    sys.exit(1)

# Configurations
TRAIN_NORMAL_SAMPLES = 30000
TEST_NORMAL_SAMPLES = 15000
TEST_ANOMALY_SAMPLES = 15000
SCALERS = {
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'Normalizer': Normalizer(),
    'QuantileTransformer': QuantileTransformer(),
    'RobustScaler': RobustScaler()
}

DATASETS = [
    # ('ToNIoT', ToNIoT, 'label', ['normal', 'Normal', 'Benign', 0, '0']),
    # ('N_BaIoT', N_BaIoT, 'class', ['benign', 'Benign', 0, '0']),
    # ('BoTIoT', BoTIoT, 'subcategory', ['Normal', 'normal', 0, '0']), 
    # ('CICIoT2023', CICIoT2023, 'label', ['BenignTraffic', 'Benign', 'normal', 'Normal', 0, '0']),
    # ('EdgeIIoTset', EdgeIIoTset, 'label', ['Normal', 'normal', 0, '0']),
    ('FiveGNIDD', FiveGNIDD, 'label', ['Normal', 'normal', 0, '0']),
    ('IoTID20', IoTID20, 'label', ['Normal', 'normal', 0, '0'])
]

def generate_datasets():
    os.makedirs(DATA_DIR, exist_ok=True)
    log.info(f"Target Output Directory: {DATA_DIR}")
    
    for prefix, DLClass, target_col, normal_labels in DATASETS:
        log.info("=" * 60)
        log.info(f"Processing Dataset: {prefix}")
        
        # Load Raw Data
        ds = DLClass(print_able=False)
        try:
            # First, assume user wants to do full raw download
            ds.DownLoad_Data(load_type="raw")
            ds.Load_Data(load_type="raw", limit_cnt=100_000)
            
            if ds.To_dataframe().empty:
                raise ValueError("Raw Load resulted in empty dataframe, falling back to Preload")
                
            ds.Preprocess_Data() # default scaling/encoding on raw categorical
        except Exception as e:
            log.warning(f"Failed to load Raw data for {prefix}: {e}. Trying Preload CSV...")
            try:
                ds.DownLoad_Data(load_type="preload")
                ds.Load_Data(load_type="preload")
                
                if ds.To_dataframe().empty:
                    raise ValueError("Preload load resulted in empty dataframe")
                    
                try:
                    ds.Preprocess_Data()
                except Exception as preprocess_e:
                    log.warning(f"Preprocess skipped or partially failed for predefined preload: {preprocess_e}")
            except Exception as preload_e:
                log.error(f"Failed completely for {prefix}: {preload_e}")
                continue
                
        df = ds.To_dataframe()
        if df.empty:
            log.error(f"Dataframe is empty for {prefix}!")
            continue

        # Feature vs Label split
        label_col = target_col
        if 'Binary_label' in df.columns:
            label_col = 'Binary_label'
        elif 'Default_label' in df.columns:
            label_col = 'Default_label'
            
        if label_col not in df.columns:
            # Fallback if preprocessing stripped it
            # Just take the last column as label
            label_col = df.columns[-1]

        log.info(f"Using label column '{label_col}' on shape {df.shape}")
        
        # Identify Normal vs Anomaly
        is_normal = df[label_col].isin(normal_labels) | df[label_col].astype(str).str.lower().str.contains('normal|benign', na=False)
        
        df_normal = df[is_normal]
        df_anomaly = df[~is_normal]
        
        log.info(f"Found {len(df_normal)} Normal samples and {len(df_anomaly)} Anomaly samples.")
        
        if len(df_normal) < (TRAIN_NORMAL_SAMPLES + TEST_NORMAL_SAMPLES):
            log.warning(f"Not enough Normal samples! Needed {TRAIN_NORMAL_SAMPLES + TEST_NORMAL_SAMPLES}, got {len(df_normal)}")
        if len(df_anomaly) < TEST_ANOMALY_SAMPLES:
            log.warning(f"Not enough Anomaly samples! Needed {TEST_ANOMALY_SAMPLES}, got {len(df_anomaly)}")
            
        # Sample rigorously
        total_normal = len(df_normal)
        if total_normal < (TRAIN_NORMAL_SAMPLES + TEST_NORMAL_SAMPLES):
            log.warning(f"Not enough Normal samples! Splitting available {total_normal} proportionally 80% Train / 20% Test.")
            n_train_norm = int(total_normal * 0.8)
            n_test_norm = total_normal - n_train_norm
        else:
            n_train_norm = TRAIN_NORMAL_SAMPLES
            n_test_norm = TEST_NORMAL_SAMPLES
            
        n_test_anom = min(TEST_ANOMALY_SAMPLES, len(df_anomaly))
        
        # Ensure we have exactly the needed splits (or as close as possible)
        df_normal_shuffled = df_normal.sample(frac=1, random_state=42).reset_index(drop=True)
        df_anomaly_shuffled = df_anomaly.sample(frac=1, random_state=42).reset_index(drop=True)
        
        train_df = df_normal_shuffled.iloc[:n_train_norm].copy()
        test_normal_df = df_normal_shuffled.iloc[n_train_norm:n_train_norm+n_test_norm].copy()
        test_anomaly_df = df_anomaly_shuffled.iloc[:n_test_anom].copy()
        
        test_df = pd.concat([test_normal_df, test_anomaly_df], ignore_index=True)
        test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Convert labels to strictly 0 (normal) and 1 (anomaly)
        train_df[label_col] = 0
        test_df[label_col] = test_df[label_col].apply(lambda x: 0 if str(x).lower() in [str(n).lower() for n in normal_labels] or 'normal' in str(x).lower() or 'benign' in str(x).lower() else 1)
        
        log.info(f"FINAL Train shape: {train_df.shape} (Normal ONLY)")
        log.info(f"FINAL Test  shape: {test_df.shape} (Normal: {len(test_normal_df)}, Anomaly: {len(test_anomaly_df)})")
        
        # Ensure data is clean numeric before scaling
        # Drop any leftover label columns besides label_col
        label_fts_names_to_drop = [c for c in ['attack', 'category', 'subcategory', 'Binary_label', 'Category_label', 'Default_label'] if c in train_df.columns and c != label_col]
        train_df = train_df.drop(columns=label_fts_names_to_drop, errors='ignore')
        test_df  = test_df.drop(columns=label_fts_names_to_drop, errors='ignore')
        
        # Separate X and Y
        y_train = train_df.pop(label_col)
        y_test = test_df.pop(label_col)
        
        # Guarantee no inf/nan
        train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        train_df.fillna(train_df.mean(numeric_only=True), inplace=True)
        test_df.fillna(train_df.mean(numeric_only=True), inplace=True) # use train mean

        # Scale and Save
        for scaler_name, scaler in SCALERS.items():
            log.info(f"  > Applying {scaler_name}...")
            
            # Reset scaler
            scaler = SCALERS[scaler_name] 
            
            X_train_scaled = pd.DataFrame(scaler.fit_transform(train_df), columns=train_df.columns)
            X_test_scaled  = pd.DataFrame(scaler.transform(test_df), columns=test_df.columns)
            
            # Merge labels back for the final CSVs
            X_train_scaled['label'] = y_train.values
            X_test_scaled['label'] = y_test.values
            
            # Save Train
            train_filename = f"Train_{scaler_name}_data_{prefix}.csv"
            train_path = os.path.join(DATA_DIR, train_filename)
            X_train_scaled.to_csv(train_path, index=False)
            
            # Save Test
            test_filename = f"Test_{scaler_name}_data_{prefix}.csv"
            test_path = os.path.join(DATA_DIR, test_filename)
            X_test_scaled.to_csv(test_path, index=False)
            
        log.info(f"Successfully generated all scaled CSVs for {prefix}!")
        
        # Free up disk space by deleting the dataset zip file
        zip_path_1 = os.path.join(RAW_DATA_DIR, f"{prefix}.zip")
        zip_path_2 = os.path.join(os.getcwd(), f"{prefix}.zip")
        
        for z_path in [zip_path_1, zip_path_2]:
            if os.path.exists(z_path):
                try:
                    os.remove(z_path)
                    log.info(f"Deleted raw zip file to save disk space: {z_path}")
                except Exception as e:
                    log.warning(f"Could not delete {z_path}: {e}")

if __name__ == "__main__":
    generate_datasets()
