import os
import pandas as pd
import numpy as np

def test_datasets():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Datascaled', 'Official_OC_Data')
    
    if not os.path.exists(data_dir):
        print(f"❌ Error: Data directory not found at {data_dir}")
        return

    datasets = ['ToNIoT', 'N_BaIoT', 'BoTIoT', 'CICIoT2023', 'EdgeIIoTset', 'IoTID20']
    scalers = ['QuantileTransformer', 'StandardScaler', 'MinMaxScaler', 'RobustScaler', 'Normalizer']
    
    print("==========================================================")
    print(f"🔍 Starting Dataset Verification in {data_dir}")
    print("==========================================================")
    
    total_files_checked = 0
    passed_files = 0
    failed_files = 0
    
    for ds in datasets:
        print(f"\n📊 Checking Dataset: {ds}")
        for scaler in scalers:
            train_file = os.path.join(data_dir, f'Train_{scaler}_data_{ds}.csv')
            test_file = os.path.join(data_dir, f'Test_{scaler}_data_{ds}.csv')
            
            for file_path, split_name in [(train_file, "Train"), (test_file, "Test")]:
                if not os.path.exists(file_path):
                    # It's okay if not all combinations exist yet, just skip silently or note it.
                    continue
                
                total_files_checked += 1
                try:
                    df = pd.read_csv(file_path)
                    
                    # 1. NaN / Inf Check
                    has_nan = df.isna().sum().sum() > 0
                    has_inf = np.isinf(df.select_dtypes(include=[np.number])).values.sum() > 0
                    
                    # 2. Shape Check
                    if df.empty:
                        raise ValueError("Dataframe is empty!")
                        
                    # 3. Label check
                    if 'label' not in df.columns:
                        raise ValueError("Missing 'label' column!")
                        
                    label_counts = df['label'].value_counts().to_dict()
                    
                    # Validation Logic
                    if has_nan or has_inf:
                        raise ValueError(f"Contains NaNs ({has_nan}) or Infs ({has_inf})")
                        
                    if split_name == "Train" and (1 in label_counts or 1.0 in label_counts):
                         raise ValueError(f"Train set contains Anomaly (1) labels! Counts: {label_counts}")
                         
                    if split_name == "Test" and (1 not in label_counts and 1.0 not in label_counts):
                         raise ValueError(f"Test set is missing Anomaly (1) labels! Counts: {label_counts}")
                         
                    print(f"  ✅ [PASS] {split_name} ({scaler:20s}) | Shape: {df.shape} | Labels: {label_counts}")
                    passed_files += 1

                except Exception as e:
                    print(f"  ❌ [FAIL] {split_name} ({scaler:20s}) | Error: {e}")
                    failed_files += 1
                    
    print("\n==========================================================")
    print(f"🏁 Verification Summary:")
    print(f"   Files Checked : {total_files_checked}")
    print(f"   Passed        : {passed_files}")
    print(f"   Failed        : {failed_files}")
    if failed_files == 0 and total_files_checked > 0:
        print("   Status        : SUCCESS 🎉 All datasets are clean and valid.")
    elif total_files_checked == 0:
        print("   Status        : WARNING ⚠️ No generated CSVs found to test.")
    else:
        print("   Status        : FAILED 💥 Some datasets have issues.")
    print("==========================================================")

if __name__ == "__main__":
    test_datasets()
