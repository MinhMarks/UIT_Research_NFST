import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    matthews_corrcoef, f1_score, precision_score, recall_score,
    accuracy_score, roc_auc_score, average_precision_score
)
from sklearn.impute import SimpleImputer
import traceback
import random
import torch

# Fix seeds globally
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

sys.path.append("../..")

# --- Fix for ModuleNotFoundError ---
# Get the directory where tune_baselines.py is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# The project root is two levels up from notebooks/baselines
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# -----------------------------------

# PyOD Models
from pyod.models.knn import KNN
from pyod.models.lunar import LUNAR
from pyod.models.lof import LOF
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.cblof import CBLOF
from pyod.models.hbos import HBOS
from pyod.models.pca import PCA
from pyod.models.ae1svm import AE1SVM
from pyod.models.devnet import DevNet
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.iforest import IForest
from pyod.models.ocsvm import OCSVM
from pyod.models.loda import LODA
from pyod.models.mo_gaal import MO_GAAL
from pyod.models.suod import SUOD
from pyod.models.alad import ALAD
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.vae import VAE
from pyod.models.so_gaal import SO_GAAL

from baseline_model.dasvdd_wrapper import DASVDD
from baseline_model.neutralad_wrapper import NeuTraLAD
from baseline_model.dif_wrapper import DIF

def preprocess_data_noise(train_data, test_data, noise_percentage=10):
    X_train_total = train_data.iloc[:, :-1].to_numpy()
    y_train_total = train_data.iloc[:, -1].to_numpy()

    X_train = X_train_total[y_train_total == 0]
    y_train = y_train_total[y_train_total == 0]

    n_samples = X_train.shape[0]
    noise_samples_count = int(n_samples * (noise_percentage / 100))

    X_train_noise = X_train_total[y_train_total == 1]
    
    noisy_indices = np.random.choice(X_train_noise.shape[0], size=noise_samples_count, replace=False)
    X_train_noise = X_train_noise[noisy_indices]
    
    X_train = np.vstack((X_train, X_train_noise))
    y_train = np.concatenate((y_train, np.zeros(X_train_noise.shape[0])))

    X_test = test_data.iloc[:, :-1].to_numpy()
    y_test = test_data.iloc[:, -1].to_numpy()

    return X_train, y_train, X_test, y_test


def evaluate_model(y_true, y_pred, y_scores=None, y_probabilities=None):
    # Lật nhãn dữ liệu: Do Normal (0) ít hơn Anomaly (1), ta lật nhãn để đánh giá AUCPR 
    # tập trung vào lớp thiểu số (chuyển Normal thành 1 và Anomaly thành 0)
    y_true_flipped = 1 - y_true
    y_pred_flipped = 1 - y_pred
    
    mcc = matthews_corrcoef(y_true_flipped, y_pred_flipped)
    f1 = f1_score(y_true_flipped, y_pred_flipped)
    precision = precision_score(y_true_flipped, y_pred_flipped, zero_division=0)
    recall = recall_score(y_true_flipped, y_pred_flipped, zero_division=0)
    accuracy = accuracy_score(y_true_flipped, y_pred_flipped)

    auc_roc, auc_pr = None, None
    if y_probabilities is not None:
        try:
            # Xác suất của lớp 0 nguyên bản nay trở thành xác suất của lớp 1 sau khi lật
            auc_roc = roc_auc_score(y_true_flipped, y_probabilities[:, 0])
            auc_pr = average_precision_score(y_true_flipped, y_probabilities[:, 0])
        except Exception:
            pass
            
    return {"AUCROC": auc_roc * 100 if auc_roc else None, 
            "AUCPR": auc_pr * 100 if auc_pr else None,
            "Accuracy": accuracy * 100, 
            "MCC": mcc, 
            "F1 Score": f1, 
            "Precision": precision, 
            "Recall": recall}


# Parameter Grid Definition
PARAM_GRIDS = {
    "CBLOF": [{"n_clusters": 10}, {"n_clusters": 50}, {"n_clusters": 100}, {"n_clusters": 150}],
    "KNN": [{"n_neighbors": 5}, {"n_neighbors": 20}, {"n_neighbors": 50}],
    "LOF": [{"n_neighbors": 5}, {"n_neighbors": 20}, {"n_neighbors": 50}],
    "HBOS": [{"n_bins": 10}, {"n_bins": 50}, {"n_bins": 100}],
    "IForest": [{"n_estimators": 50}, {"n_estimators": 100}, {"n_estimators": 200}],
    "PCA": [{"n_components": 0.5}, {"n_components": 0.7}, {"n_components": 0.9}],
    "OCSVM": [{"nu": 0.1}, {"nu": 0.5}], # Standard nu
    "AutoEncoder": [{"hidden_neurons": [64, 32, 32, 64], "epochs": 50}, {"hidden_neurons": [128, 64, 64, 128], "epochs": 50}],
    "DIF": [{"n_ensemble": 50, "n_estimators": 6}, {"n_ensemble": 100, "n_estimators": 10}],
    "NeuTraLAD": [{"latent_dim": 32, "enc_hdim": 32}, {"latent_dim": 64, "enc_hdim": 64}],
    "DASVDD": [{"code_size": 32}, {"code_size": 64}],
    "LODA": [{"n_bins": 10}, {"n_bins": 50}],
    
    # Models without obvious fast tuning parameters left default
    "ALAD": [{}],
    "COPOD": [{}],
    "ECOD": [{}],
    "VAE": [{"encoder_neurons": [64, 32], "decoder_neurons": [32, 64], "epochs": 50}],
    "SO_GAAL": [{}],
    "MO_GAAL": [{}],
    "SUOD": [{}],
    "DeepSVDD": [{"hidden_neurons": [64, 32]}],
    "LUNAR": [{"n_endpoints": 10}],
    "AE1SVM": [{}],
    "DevNet": [{}]
}

def get_model(model_name, params):
    model_dict = {
        "CBLOF": CBLOF,
        "KNN": KNN,
        "IForest": IForest,
        "OCSVM": OCSVM,
        "LOF": LOF,
        "DeepSVDD": DeepSVDD,
        "HBOS": HBOS,
        "LODA": LODA,
        "PCA": PCA,
        "ECOD": ECOD,
        "COPOD": COPOD,
        "AutoEncoder": AutoEncoder,
        "DevNet": DevNet,
        "LUNAR": LUNAR,
        "AE1SVM": AE1SVM,
        "ALAD": ALAD,
        "VAE": VAE,
        "SO_GAAL": SO_GAAL,
        "MO_GAAL": MO_GAAL,
        "SUOD": SUOD,
    }
    
    if model_name == 'DASVDD':
        return DASVDD(**params, num_epochs=50, verbose=0)
    elif model_name == "DIF": 
        return DIF(**params, verbose=0)
    elif model_name == "NeuTraLAD": 
        return NeuTraLAD(**params, num_epochs=50, verbose=0)
    elif model_name == "SUOD":
        base_estimators = [HBOS(n_bins=10), COPOD(), ECOD()]
        return SUOD(base_estimators=base_estimators, n_jobs=1, rp_flag_global=True, bps_flag=False, verbose=False)
    
    model_class = model_dict.get(model_name)
    if model_class is None:
        raise ValueError(f"Model {model_name} not found.")
    
    try:
        return model_class(**params)
    except Exception as e:
        print(f"Warning: Could not initialize {model_name} with {params}: {e}. Retrying with default parameters.")
        return model_class()

def run_experiment(X_train, y_train, X_test, y_test, dataset_name, noise_percentage, scaler, output_file_all, output_file_best):
    best_results = []
    
    for model_name, param_list in PARAM_GRIDS.items():
        print(f"\\n--- Tuning {model_name} on {dataset_name} ({scaler}, Noise: {noise_percentage}) ---")
        best_auc = -1
        best_result_row = None
        
        for params in param_list:
            print(f"Testing params: {params}")
            try:
                import tracemalloc
                tracemalloc.start()
                start_time = time.time()
                
                if model_name == 'DevNet':
                    model.fit(X_train, y_train)
                else:
                    model.fit(X_train)
                    
                train_time = time.time() - start_time
                current, peak_train = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                tracemalloc.start()
                start_time = time.time()
                y_pred = model.predict(X_test)
                
                if hasattr(model, "predict_proba"):
                    y_probabilities = model.predict_proba(X_test)
                else:
                    y_probabilities = None
                    
                test_time = time.time() - start_time
                current, peak_test = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                metrics = evaluate_model(y_test, y_pred, y_probabilities=y_probabilities)
                
                result_row = {
                    "Dataset": dataset_name,
                    "Model": model_name,
                    "Parameters": str(params),
                    "Scaled": scaler,
                    "Noise": noise_percentage,
                    **metrics,
                    "Time Train": train_time,
                    "Time Test": test_time,
                    "Peak RAM Train (MB)": peak_train / 10**6,
                    "Peak RAM Test (MB)": peak_test / 10**6
                }
                
                # Save to All file
                pd.DataFrame([result_row]).to_csv(output_file_all, mode='a', header=not os.path.exists(output_file_all), index=False)
                
                # Check if best
                current_auc = metrics.get("AUCROC") or 0
                if current_auc > best_auc:
                    best_auc = current_auc
                    best_result_row = result_row
                    
            except Exception as e:
                print(f"Error with {model_name} params {params}: {e}")
                traceback.print_exc()
                
        # Save best result for this model/dataset
        if best_result_row:
            print(f"Best params for {model_name}: {best_result_row['Parameters']} (AUC: {best_result_row['AUCROC']})")
            pd.DataFrame([best_result_row]).to_csv(output_file_best, mode='a', header=not os.path.exists(output_file_best), index=False)

if __name__ == "__main__": 
    # Use small subset for testing purposes first
    # Change these back to full lists for real experiment
    dataset_prefixes = ['data_ToNIoT.csv', 'data_N_BaIoT.csv', 'data_CICIoT2023.csv', 'data_BoTIoT.csv']
    scaler_names = ['QuantileTransformer', 'MinMaxScaler', 'Normalizer', 'RobustScaler']
    noise_levels = [0, 1, 3, 5]
    
    output_all = "Tuned_Baseline_Results_All1.csv"
    output_best = "Best_Baseline_Results_Per_Model1.csv"

    for prefix in dataset_prefixes:
        for scaler in scaler_names: 
            train_file = f'../../Datascaled/NoiseOCData/Train_{scaler}_{prefix}'
            test_file = f'../../Datascaled/NoiseOCData/Test_{scaler}_{prefix}'
            
            if not os.path.exists(train_file) or not os.path.exists(test_file):
                print(f"File not found for {prefix}, {scaler}. Skipping...")
                continue
                
            df_train = pd.read_csv(train_file).dropna()
            df_test = pd.read_csv(test_file).dropna()
            
            df_full = pd.concat([df_train, df_test], ignore_index=True)
            df_train_new, df_test_new = train_test_split(df_full, test_size=0.3, random_state=42)

            for noise in noise_levels: 
                print(f"\\nProcessing: {prefix}, Scaler: {scaler}, Noise: {noise}")
                X_train, y_train, X_test, y_test = preprocess_data_noise(df_train_new, df_test_new, noise)
                
                imputer = SimpleImputer(strategy="mean") 
                X_train[np.isinf(X_train)] = np.nan
                X_train = imputer.fit_transform(X_train)
                
                X_test[np.isinf(X_test)] = np.nan
                X_test = imputer.transform(X_test)
                
                run_experiment(X_train, y_train, X_test, y_test, prefix, noise, scaler, output_all, output_best) 
