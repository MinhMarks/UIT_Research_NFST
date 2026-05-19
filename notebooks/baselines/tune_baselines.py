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
from datetime import datetime
from joblib import Parallel, delayed

# --- Configuration Toggle ---
USE_TUNING = True     # Set to False to skip hyperparameter search
USE_PARALLEL = False    # Set to False if models crash (Stability Mode)
N_JOBS = 4            # Number of parallel workers when tuning/running models

# Define which models to explicitly run. If empty, runs all models in PARAM_GRIDS.
# MODELS_TO_RUN = ["OCKMeans"]

# Define which models to explicitly run. If empty, runs all models in PARAM_GRIDS.
MODELS_TO_RUN = ["LUNAR", "KNN", "LOF", "IForest", "AutoEncoder", "OCKMeans"]
# ----------------------------

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

from sklearn.cluster import KMeans as SKL_KMeans
from pyod.models.base import BaseDetector
from sklearn.utils.validation import check_is_fitted

class OCKMeans(BaseDetector):
    def __init__(self, n_clusters=8, contamination=0.1):
        super(OCKMeans, self).__init__(contamination=contamination)
        self.n_clusters = n_clusters

    def fit(self, X, y=None):
        self.model_ = SKL_KMeans(n_clusters=self.n_clusters, random_state=42)
        self.model_.fit(X)
        X_trans = self.model_.transform(X)
        self.decision_scores_ = np.min(X_trans, axis=1)
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        check_is_fitted(self, ['model_'])
        X_trans = self.model_.transform(X)
        return np.min(X_trans, axis=1)

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
        except Exception as auc_err:
            # Báo lỗi rõ ràng thay vì nuốt im lặng
            print(f"  [WARNING] AUCROC/AUCPR computation failed: {auc_err}")
    else:
        print("  [WARNING] y_probabilities is None — AUCROC/AUCPR will not be computed.")
            
    return {"AUCROC": auc_roc * 100 if auc_roc is not None else None, 
            "AUCPR": auc_pr * 100 if auc_pr is not None else None,
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
    # PyOD >= 2.x renamed: hidden_neurons → hidden_neuron_list, epochs → epoch_num
    # hidden_neuron_list chỉ cần nửa encoder; PyOD tự build decoder bằng cách reverse lại
    "AutoEncoder": [
        {"hidden_neuron_list": [32, 16], "epoch_num": 50, "contamination": 0.05},
        {"hidden_neuron_list": [64, 32], "epoch_num": 50, "contamination": 0.05},
        {"hidden_neuron_list": [128, 64], "epoch_num": 50, "contamination": 0.05}
    ],
    "DIF": [{"n_ensemble": 50, "n_estimators": 6}, {"n_ensemble": 100, "n_estimators": 10}],
    "NeuTraLAD": [{"latent_dim": 32, "enc_hdim": 32}, {"latent_dim": 64, "enc_hdim": 64}],
    "DASVDD": [{"code_size": 32}, {"code_size": 64}],
    "LODA": [{"n_bins": 10}, {"n_bins": 50}],
    "OCKMeans": [{"n_clusters": 5}, {"n_clusters": 10}, {"n_clusters": 20}, {"n_clusters": 50}],
    
    # Models without obvious fast tuning parameters left default
    "ALAD": [{}],
    "COPOD": [{}],
    "ECOD": [{}],
    "VAE": [{"encoder_neurons": [64, 32], "decoder_neurons": [32, 64], "epochs": 50}],
    "SO_GAAL": [{}],
    "MO_GAAL": [{}],
    "SUOD": [{}],
    "DeepSVDD": [{"hidden_neurons": [64, 32]}],
    "LUNAR": [{"n_endpoints": 5}, {"n_endpoints": 10}, {"n_endpoints": 20}],
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
        "OCKMeans": OCKMeans,
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

def run_experiment(X_train, y_train, X_test, y_test, dataset_name, noise_percentage, scaler, output_file_all, output_file_best, use_tuning=True):
    best_results = []
    
    # We will process models in parallel over model_name if USE_PARALLEL is True
    # or sequentially otherwise. 
    # To keep code clean, we define a helper for a single model run.
    
    def process_model(model_name, param_list):
        print(f"\n--- Processing {model_name} on {dataset_name} ({scaler}, Noise: {noise_percentage}) ---")
        best_auc = -1
        best_row = None
        
        # Determine which parameters to test
        params_to_test = param_list if use_tuning else [param_list[0]]
        
        for params in params_to_test:
            if use_tuning: print(f"  Testing params: {params}")
            try:
                model = get_model(model_name, params)
                import tracemalloc
                tracemalloc.start()
                t0 = time.time()
                
                if model_name == 'DevNet': model.fit(X_train, y_train)
                else: model.fit(X_train)
                    
                train_time = time.time() - t0
                _, peak_train = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                tracemalloc.start()
                t1 = time.time()
                y_pred = model.predict(X_test)
                y_probs = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
                test_time = time.time() - t1
                _, peak_test = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                metrics = evaluate_model(y_test, y_pred, y_probabilities=y_probs)
                
                row = {
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
                
                # Save to All file (Atomic write if parallel)
                pd.DataFrame([row]).to_csv(output_file_all, mode='a', header=not os.path.exists(output_file_all), index=False)
                
                # AUCROC là metric chính để chọn best config.
                # Nếu None (đã in warning ở evaluate_model), raise lỗi để không âm thầm bỏ qua.
                cur_auc = metrics.get("AUCROC")
                if cur_auc is None:
                    raise ValueError(
                        f"AUCROC is None for {model_name} with params {params}. "
                        "Check evaluate_model warnings above for the root cause."
                    )
                if cur_auc > best_auc:
                    best_auc = cur_auc
                    best_row = row
                    
            except Exception as e:
                print(f"  Error with {model_name}: {e}")
        
        if best_row:
            pd.DataFrame([best_row]).to_csv(output_file_best, mode='a', header=not os.path.exists(output_file_best), index=False)
            return best_row
        return None

    models_to_run = []
    for m, p in PARAM_GRIDS.items():
        if MODELS_TO_RUN and m not in MODELS_TO_RUN:
            continue
        models_to_run.append((m, p))
    
    if USE_PARALLEL and not use_tuning:
        # High speed mode: Parallelize across MODELS (since each only has 1 param set)
        Parallel(n_jobs=N_JOBS, backend="multiprocessing")(
            delayed(process_model)(m, p) for m, p in models_to_run
        )
    else:
        # Tuning mode or Stability mode: Sequential model processing
        for m, p in models_to_run:
            process_model(m, p)

def run_tuning_experiment(X_train, y_train, X_test, y_test, dataset_name, noise_percentage, scaler, output_file_all, output_file_best):
    """Legacy wrapper for backward compatibility if needed, though we update the call site."""
    return run_experiment(X_train, y_train, X_test, y_test, dataset_name, noise_percentage, scaler, output_file_all, output_file_best, use_tuning=USE_TUNING)

if __name__ == "__main__": 
    # Use small subset for testing purposes first
    # Change these back to full lists for real experiment
    # dataset_prefixes = ['data_ToNIoT.csv', 'data_N_BaIoT.csv', 'data_CICIoT2023.csv', 'data_BoTIoT.csv', 'data_EdgeIIoTset.csv', 'data_IoTID20.csv', 'data_FiveGNIDD.csv']
    # scaler_names = ['QuantileTransformer', 'MinMaxScaler', 'Normalizer', 'RobustScaler', 'StandardScaler']
    # noise_levels = [0, 1, 3, 5]

    dataset_prefixes = ['data_EdgeIIoTset.csv', 'data_IoTID20.csv']
    scaler_names = ['QuantileTransformer']
    noise_levels = [0]
    
    RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"Experiment_Baseline_Tuning_{RUN_TIMESTAMP}"
    
    # Create the central experiment directory inside notebooks/baselines/outputs
    exp_dir = os.path.join(current_dir, 'outputs', experiment_name)
    os.makedirs(exp_dir, exist_ok=True)

    output_all = os.path.join(exp_dir, "Tuned_Baseline_Results_All.csv")
    output_best = os.path.join(exp_dir, "Best_Baseline_Results_Per_Model.csv")

    for prefix in dataset_prefixes:
        for scaler in scaler_names: 
            train_file = os.path.join(current_dir, '..', '..', 'Datascaled', 'Official_OC_Data', f'Train_{scaler}_{prefix}')
            test_file = os.path.join(current_dir, '..', '..', 'Datascaled', 'Official_OC_Data', f'Test_{scaler}_{prefix}')
            
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
                
                run_experiment(X_train, y_train, X_test, y_test, prefix, noise, scaler, output_all, output_best, use_tuning=USE_TUNING) 
