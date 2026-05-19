import os
import sys
import time
import numpy as np
import pandas as pd
import traceback
import random
import logging
import tracemalloc
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    matthews_corrcoef, f1_score, precision_score, recall_score,
    accuracy_score, roc_auc_score, average_precision_score
)
from sklearn.impute import SimpleImputer
from joblib import Parallel, delayed
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

# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# PyOD Models
try:
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
except ImportError:
    print("Warning: Some PyOD models could not be imported. Ensure pyod is installed.")

try:
    from baseline_model.dasvdd_wrapper import DASVDD
    from baseline_model.neutralad_wrapper import NeuTraLAD
    from baseline_model.dif_wrapper import DIF
except ImportError:
    print("Warning: Custom wrappers (DASVDD, NeuTraLAD, DIF) not found. Skipping those.")

# ============================================================================
# LOGGING
# ============================================================================
def setup_logger(log_path, name="baseline"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger

# ============================================================================
# DATA PREPROCESSING
# ============================================================================
def preprocess_data_noise(train_data, test_data, noise_percentage=10):
    X_train_total = train_data.iloc[:, :-1].to_numpy(dtype=np.float32)
    y_train_total = train_data.iloc[:, -1].to_numpy()

    X_train = X_train_total[y_train_total == 0]
    y_train = y_train_total[y_train_total == 0]

    n_samples = X_train.shape[0]
    noise_count = int(n_samples * (noise_percentage / 100))

    X_train_noise = X_train_total[y_train_total == 1]
    if noise_count > 0 and len(X_train_noise) > 0:
        idx = np.random.choice(len(X_train_noise), size=min(noise_count, len(X_train_noise)), replace=False)
        X_train = np.vstack((X_train, X_train_noise[idx]))
        y_train = np.concatenate((y_train, np.zeros(len(idx))))

    X_test = test_data.iloc[:, :-1].to_numpy(dtype=np.float32)
    y_test = test_data.iloc[:, -1].to_numpy()

    return X_train, y_train, X_test, y_test

# ============================================================================
# EVALUATION
# ============================================================================
def evaluate_model(y_true, y_pred, y_probabilities=None):
    # Flip labels: Normal (0) -> 1, Anomaly (1) -> 0 for AUCPR focus
    y_true_flipped = (1 - y_true).astype(int)
    y_pred_flipped = (1 - y_pred).astype(int)
    
    mcc = matthews_corrcoef(y_true_flipped, y_pred_flipped)
    f1 = f1_score(y_true_flipped, y_pred_flipped, zero_division=0)
    ppv = precision_score(y_true_flipped, y_pred_flipped, zero_division=0)
    recall = recall_score(y_true_flipped, y_pred_flipped, zero_division=0)
    accuracy = accuracy_score(y_true_flipped, y_pred_flipped)

    auc_roc, auc_pr = 0.0, 0.0
    if y_probabilities is not None:
        try:
            # Probability of being normal (class 0)
            auc_roc = roc_auc_score(y_true_flipped, y_probabilities[:, 0])
            auc_pr = average_precision_score(y_true_flipped, y_probabilities[:, 0])
        except Exception:
            pass
            
    return {
        "AUCROC": auc_roc * 100 if auc_roc else None, 
        "AUCPR": auc_pr * 100 if auc_pr else None,
        "Accuracy": accuracy * 100, 
        "MCC": mcc, 
        "F1 Score": f1, 
        "Precision": ppv, 
        "Recall": recall
    }

# ============================================================================
# FIXED PARAMETERS (NO TUNING FOR SPEED)
# ============================================================================
FAST_PARAMS = {
    "CBLOF": {"n_clusters": 50},
    "KNN": {"n_neighbors": 20},
    "LOF": {"n_neighbors": 20},
    "HBOS": {"n_bins": 50},
    "IForest": {"n_estimators": 100},
    "PCA": {"n_components": 0.7},
    "OCSVM": {"nu": 0.1},
    "AutoEncoder": {"hidden_neurons": [64, 32, 32, 64], "epochs": 20},
    "DIF": {"n_ensemble": 50, "n_estimators": 6},
    "NeuTraLAD": {"latent_dim": 32, "enc_hdim": 32, "num_epochs": 20},
    "DASVDD": {"code_size": 32, "num_epochs": 20},
    "LODA": {"n_bins": 50},
    "COPOD": {},
    "ECOD": {},
    "VAE": {"encoder_neurons": [64, 32], "decoder_neurons": [32, 64], "epochs": 20},
    "DeepSVDD": {"hidden_neurons": [64, 32], "epochs": 20},
    "LUNAR": {"n_neighbors": 5},
    "AE1SVM": {"epochs": 20},
    "DevNet": {"epochs": 20},
    "ALAD": {"epochs": 20},
}

def get_model(model_name, params, n_features):
    model_dict = {
        "CBLOF": CBLOF, "KNN": KNN, "IForest": IForest, "OCSVM": OCSVM,
        "LOF": LOF, "DeepSVDD": DeepSVDD, "HBOS": HBOS, "LODA": LODA,
        "PCA": PCA, "ECOD": ECOD, "COPOD": COPOD, "AutoEncoder": AutoEncoder,
        "DevNet": DevNet, "LUNAR": LUNAR, "AE1SVM": AE1SVM, "ALAD": ALAD,
        "VAE": VAE, "SO_GAAL": SO_GAAL, "MO_GAAL": MO_GAAL, "SUOD": SUOD,
    }
    
    # Common flags to remove if they cause issues or aren't supported
    params = params.copy()
    
    # Forced CPU for all torch-based baseline models to avoid CUDA fork issues
    torch_models = ['AutoEncoder', 'VAE', 'DeepSVDD', 'AE1SVM', 'DevNet', 'ALAD']
    if model_name in torch_models:
        params['device'] = 'cpu'
    
    if model_name == 'DASVDD': return DASVDD(**params, verbose=0)
    elif model_name == "DIF": return DIF(**params, verbose=0)
    elif model_name == "NeuTraLAD": return NeuTraLAD(**params, verbose=0)
    elif model_name == "DeepSVDD": return DeepSVDD(n_features=n_features, **params)
    elif model_name == "SUOD":
        base_estimators = [HBOS(n_bins=10), COPOD(), ECOD()]
        return SUOD(base_estimators=base_estimators, n_jobs=1, rp_flag_global=True, bps_flag=False, verbose=False)
    
    model_class = model_dict.get(model_name)
    if model_class is None: return None
    
    # Try initialization, remove problematic arguments if they fail
    try:
        return model_class(**params)
    except TypeError as e:
        # If 'hidden_neurons' failed for AutoEncoder, it's likely very old PyOD
        msg = str(e).lower()
        if 'hidden_neurons' in msg and 'unexpected' in msg:
            params['hidden_layers'] = params.pop('hidden_neurons')
        if 'encoder_neurons' in msg and 'unexpected' in msg:
            params['encoder_layers'] = params.pop('encoder_neurons')
            if 'decoder_neurons' in params:
                 params['decoder_layers'] = params.pop('decoder_neurons')
        if 'verbose' in msg and 'unexpected' in msg:
            params.pop('verbose', None)
        return model_class(**params)

# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================
def run_single_model(model_name, X_train, y_train, X_test, y_test, dataset_name, noise, scaler):
    params = FAST_PARAMS.get(model_name, {})
    n_features = X_train.shape[1]
    try:
        model = get_model(model_name, params, n_features)
        if model is None: return None
        
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
        
        return {
            "Dataset": dataset_name,
            "Model": model_name,
            "Parameters": str(params),
            "Scaled": scaler,
            "Noise": noise,
            **metrics,
            "Time Train": train_time,
            "Time Test": test_time,
            "Peak RAM Train (MB)": peak_train / 10**6,
            "Peak RAM Test (MB)": peak_test / 10**6
        }
    except Exception as e:
        return {"error": f"Error with {model_name} on {dataset_name}: {e}"}

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    _default_data = os.path.normpath(os.path.join(project_root, 'Datascaled', 'Official_OC_Data'))
    DATA_DIR = os.environ.get('DATA_DIR', _default_data)
    # dataset_prefixes = ['data_ToNIoT.csv', 'data_N_BaIoT.csv', 'data_CICIoT2023.csv', 'data_BoTIoT.csv', 'data_EdgeIIoTset.csv', 'data_IoTID20.csv', 'data_FiveGNIDD.csv']
    dataset_prefixes = ['data_EdgeIIoTset.csv', 'data_IoTID20.csv']
    
    # scaler_names = ['StandardScaler', 'MinMaxScaler', 'Normalizer', 'QuantileTransformer', 'RobustScaler']
    scaler_names = ['QuantileTransformer']
    
    # noise_levels = [0, 1, 3, 5]
    noise_levels = [0]
    
    # --- Acceleration Toggle ---
    USE_PARALLEL = False  # Set to False if models crash (stability mode)
    N_JOBS = 4           # Number of parallel workers
    # ---------------------------
    
    RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(current_dir, 'outputs', f"Experiment_FastBaseline_{RUN_TIMESTAMP}")
    os.makedirs(exp_dir, exist_ok=True)
    
    log_path = os.path.join(exp_dir, "fast_baseline_run.log")
    logger = setup_logger(log_path)
    
    logger.info("=" * 60)
    logger.info(f"FAST BASELINE EXPERIMENT (Parallel={USE_PARALLEL})")
    logger.info("=" * 60)
    # Single output file for the entire experiment
    out_path = os.path.join(exp_dir, "all_fast_baseline_results.csv")
    
    for prefix in dataset_prefixes:
        for scaler in scaler_names:
            train_file = os.path.join(DATA_DIR, f'Train_{scaler}_{prefix}')
            test_file = os.path.join(DATA_DIR, f'Test_{scaler}_{prefix}')
            
            if not os.path.exists(train_file):
                logger.warning(f"File not found: {train_file}. Skipping.")
                continue
                
            logger.info(f"\nProcessing Dataset: {prefix} | Scaler: {scaler}")
            df_train = pd.read_csv(train_file).dropna()
            df_test = pd.read_csv(test_file).dropna()
            
            # Use original split or merge/re-split as per tune_baselines.py
            df_full = pd.concat([df_train, df_test], ignore_index=True)
            df_train_new, df_test_new = train_test_split(df_full, test_size=0.3, random_state=42)
            
            for noise in noise_levels:
                logger.info(f"--- Noise: {noise}% ---")
                X_train, y_train, X_test, y_test = preprocess_data_noise(df_train_new, df_test_new, noise)
                
                imputer = SimpleImputer(strategy="mean")
                X_train = imputer.fit_transform(X_train)
                X_test = imputer.transform(X_test)
                
                # Run models
                models_to_run = list(FAST_PARAMS.keys())
                
                if USE_PARALLEL:
                    logger.info(f"Running {len(models_to_run)} models in PARALLEL (n_jobs={N_JOBS})...")
                    results = Parallel(n_jobs=N_JOBS, backend="multiprocessing", verbose=10)(
                        delayed(run_single_model)(m, X_train, y_train, X_test, y_test, prefix, noise, scaler) 
                        for m in models_to_run
                    )
                else:
                    logger.info(f"Running {len(models_to_run)} models SEQUENTIALLY (Stability Mode)...")
                    results = []
                    for m in models_to_run:
                        res = run_single_model(m, X_train, y_train, X_test, y_test, prefix, noise, scaler)
                        results.append(res)
                
                valid_results = []
                for res in results:
                    if res and "error" in res: logger.error(res["error"])
                    elif res:
                        valid_results.append(res)
                        logger.info(f"  {res['Model']}: AUCROC={res['AUCROC']}%  Train={res['Time Train']}s")
                
                if valid_results:
                    pd.DataFrame(valid_results).to_csv(out_path, mode='a', header=not os.path.exists(out_path), index=False)

    logger.info("\nAll results saved to: " + exp_dir)
    print("\nAll done.")
