import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    matthews_corrcoef, f1_score, precision_score, recall_score,
    accuracy_score, roc_auc_score, average_precision_score
)
from sklearn.impute import SimpleImputer
import random
import torch
from datetime import datetime
from joblib import Parallel, delayed

# --- Configuration Toggle ---
USE_TUNING = False     # Set to False to skip hyperparameter search
USE_PARALLEL = False    # Set to False if models crash (Stability Mode)
N_JOBS = 4            # Number of parallel workers when tuning/running models
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

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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

def evaluate_model(y_true, y_pred, y_probabilities=None):
    y_true_flipped = 1 - y_true
    y_pred_flipped = 1 - y_pred
    
    mcc = matthews_corrcoef(y_true_flipped, y_pred_flipped)
    f1 = f1_score(y_true_flipped, y_pred_flipped, zero_division=0)
    precision = precision_score(y_true_flipped, y_pred_flipped, zero_division=0)
    recall = recall_score(y_true_flipped, y_pred_flipped, zero_division=0)
    accuracy = accuracy_score(y_true_flipped, y_pred_flipped)

    auc_roc, auc_pr = None, None
    if y_probabilities is not None:
        try:
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
    "OCSVM": [{"nu": 0.1}, {"nu": 0.5}], 
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
        "CBLOF": CBLOF, "KNN": KNN, "IForest": IForest, "OCSVM": OCSVM,
        "LOF": LOF, "DeepSVDD": DeepSVDD, "HBOS": HBOS, "LODA": LODA,
        "PCA": PCA, "ECOD": ECOD, "COPOD": COPOD, "AutoEncoder": AutoEncoder,
        "DevNet": DevNet, "LUNAR": LUNAR, "AE1SVM": AE1SVM, "ALAD": ALAD,
        "VAE": VAE, "SO_GAAL": SO_GAAL, "MO_GAAL": MO_GAAL, "SUOD": SUOD,
    }
    
    if model_name == 'DASVDD': return DASVDD(**params, num_epochs=50, verbose=0)
    elif model_name == "DIF": return DIF(**params, verbose=0)
    elif model_name == "NeuTraLAD": return NeuTraLAD(**params, num_epochs=50, verbose=0)
    elif model_name == "SUOD":
        base_estimators = [HBOS(n_bins=10), COPOD(), ECOD()]
        return SUOD(base_estimators=base_estimators, n_jobs=1, rp_flag_global=True, bps_flag=False, verbose=False)
    
    model_class = model_dict.get(model_name)
    if model_class is None: raise ValueError(f"Model {model_name} not found.")
    
    try: return model_class(**params)
    except Exception as e:
        print(f"Warning: Could not initialize {model_name} with {params}: {e}. Retrying with default.")
        return model_class()

def run_experiment(X_train, y_train, X_test, y_test, dataset_name, anomaly_mode, scaler, output_file_all, output_file_best, use_tuning=True):
    best_results = []
    
    def process_model(model_name, param_list):
        print(f"\n--- Processing {model_name} on {dataset_name} ({scaler}, Mode: {anomaly_mode.upper()}) ---")
        best_auc = -1
        best_row = None
        
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
                    "Anomaly_Mode": anomaly_mode,
                    "Model": model_name,
                    "Parameters": str(params),
                    "Scaled": scaler,
                    "Noise": 0, # Pure anomaly testing
                    **metrics,
                    "Time Train": train_time,
                    "Time Test": test_time,
                    "Peak RAM Train (MB)": peak_train / 10**6,
                    "Peak RAM Test (MB)": peak_test / 10**6
                }
                
                # Atomic writes since parallel mode is an option
                pd.DataFrame([row]).to_csv(output_file_all, mode='a', header=not os.path.exists(output_file_all), index=False)
                
                cur_auc = metrics.get("AUCROC") or 0
                if cur_auc > best_auc:
                    best_auc = cur_auc
                    best_row = row
                    
            except Exception as e:
                print(f"  Error with {model_name}: {e}")
        
        if best_row:
            pd.DataFrame([best_row]).to_csv(output_file_best, mode='a', header=not os.path.exists(output_file_best), index=False)
            return best_row
        return None

    models_to_run = list(PARAM_GRIDS.items())
    
    if USE_PARALLEL and not use_tuning:
        Parallel(n_jobs=N_JOBS, backend="multiprocessing")(
            delayed(process_model)(m, p) for m, p in models_to_run
        )
    else:
        for m, p in models_to_run:
            process_model(m, p)

if __name__ == "__main__": 
    dataset_prefixes = ['BoTIoT', 'CICIoT2023', 'ToNIoT', 'N_BaIoT', 'EdgeIIoTset', 'IoTID20', 'FiveGNIDD']
    scaler_names = ['StandardScaler', 'MinMaxScaler', 'Normalizer', 'QuantileTransformer', 'RobustScaler']
    anomaly_modes = ['local', 'cluster', 'global']
    
    RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"Experiment_Anomaly_Type_Baseline_Tuning_{RUN_TIMESTAMP}"
    
    exp_dir = os.path.join(current_dir, 'outputs', experiment_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Note: Split outputs so they don't corrupt in multi-processing. They run top-level loops.
    for prefix in dataset_prefixes:
        for mode in anomaly_modes:
            # Segment CSV by Prefix and Mode to avoid data contention if multiple scripts run
            output_all = os.path.join(exp_dir, f"{prefix}_{mode}_Baselines_All.csv")
            output_best = os.path.join(exp_dir, f"{prefix}_{mode}_Baselines_Best.csv")
            
            for scaler in scaler_names: 
                train_file = os.path.join(current_dir, '..', '..', 'Datascaled', 'Official_Anomaly_Data', f'Train_{mode}_{scaler}_data_{prefix}.csv')
                test_file = os.path.join(current_dir, '..', '..', 'Datascaled', 'Official_Anomaly_Data', f'Test_{mode}_{scaler}_data_{prefix}.csv')
                
                if not os.path.exists(train_file) or not os.path.exists(test_file):
                    print(f"[{mode.upper()}] File not found for {prefix}, {scaler}. Skipping...")
                    continue
                    
                df_train = pd.read_csv(train_file).dropna()
                df_test = pd.read_csv(test_file).dropna()
                
                print(f"\n========================================================")
                print(f"Loading direct: {prefix} ({mode}) - {scaler}")
                print(f"Train Shape: {df_train.shape} | Test Shape: {df_test.shape}")
                print(f"========================================================")
                
                # Direct array conversion! Because generator script produced perfectly constructed targets.
                X_train = df_train.iloc[:, :-1].to_numpy()
                y_train = df_train.iloc[:, -1].to_numpy()
                
                X_test = df_test.iloc[:, :-1].to_numpy()
                y_test = df_test.iloc[:, -1].to_numpy()
                
                # Simple imputation for any straggler edgecases mathematically created by PCA inverse traces
                imputer = SimpleImputer(strategy="mean") 
                X_train[np.isinf(X_train)] = np.nan
                X_train = imputer.fit_transform(X_train)
                
                X_test[np.isinf(X_test)] = np.nan
                X_test = imputer.transform(X_test)
                
                run_experiment(X_train, y_train, X_test, y_test, prefix, mode, scaler, output_all, output_best, use_tuning=USE_TUNING) 
                
    print("\nAll anomaly type Baseline experiments finished successfully.")
