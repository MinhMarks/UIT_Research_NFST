import os
import pandas as pd

DATA_DIR = r"d:\UIT\Research\Duongcpmputer\LOC-NFST\UIT_Research_NFST\Datascaled\Official_OC_Data"
DATASETS = ['data_CICIoT2023', 'data_ToNIoT', 'data_N_BaIoT', 'data_BoTIoT']
SCALER = 'QuantileTransformer'

print(f"Checking dataset distributions in: {DATA_DIR}")
print(f"Using Scaler: {SCALER}\n")

for ds in DATASETS:
    train_path = os.path.join(DATA_DIR, f"Train_{SCALER}_{ds}.csv")
    test_path = os.path.join(DATA_DIR, f"Test_{SCALER}_{ds}.csv")
    
    print("=" * 60)
    print(f"Dataset: {ds}")
    
    if os.path.exists(train_path):
        try:
            df_train = pd.read_csv(train_path)
            print(f"  [Train] Total samples: {df_train.shape[0]}")
            print(f"  [Train] Dimensions (Features + Label): {df_train.shape[1]}")
            # Label is usually the last column
            y_train = df_train.iloc[:, -1]
            dist = y_train.value_counts().to_dict()
            print(f"  [Train] Label distribution: {dist}")
        except Exception as e:
            print(f"  [Train] Error reading file: {e}")
    else:
        print(f"  [Train] File not found: {train_path}")
        
    if os.path.exists(test_path):
        try:
            df_test = pd.read_csv(test_path)
            print(f"  [Test] Total samples: {df_test.shape[0]}")
            print(f"  [Test] Dimensions (Features + Label): {df_test.shape[1]}")
            y_test = df_test.iloc[:, -1]
            dist = y_test.value_counts().to_dict()
            print(f"  [Test] Label distribution: {dist}")
        except Exception as e:
            print(f"  [Test] Error reading file: {e}")
    else:
        print(f"  [Test] File not found: {test_path}")

print("=" * 60)
