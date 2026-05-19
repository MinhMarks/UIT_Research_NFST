import os
import pandas as pd
import numpy as np
import time
import tracemalloc
import traceback
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, average_precision_score
import sys
from sklearn.impute import SimpleImputer

# Link DRLAD from the root path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# ============================================================================
# CONFIGURATION
# ============================================================================
output_file = "Baseline_Noise_Results_Fixed.csv"
columns = ["Dataset", "Model", "Parameters", "Scaled", "Noise", "AUCROC", "AUCPR", "Accuracy", "MCC", "F1 Score", 
           "Precision", "Recall", "Time Train", "Time Test", "Peak RAM Train (MB)", "Peak RAM Test (MB)"]

# ============================================================================
# UTILITIES
# ============================================================================

def preprocess_data_noise(train_data, test_data, noise_percentage=10):
    print("..............................Data Overview................................")
    X_train_total = train_data.iloc[:, :-1].to_numpy()
    y_train_total = train_data.iloc[:, -1].to_numpy()
    X_train = X_train_total[y_train_total == 0]
    y_train = y_train_total[y_train_total == 0]
    n_samples = X_train.shape[0]
    noise_count = int(n_samples * (noise_percentage / 100))
    X_train_noise = X_train_total[y_train_total == 1]
    
    if noise_count > 0 and len(X_train_noise) > 0:
        idx = np.random.choice(len(X_train_noise), size=min(noise_count, len(X_train_noise)), replace=False)
        X_train = np.vstack((X_train, X_train_noise[idx]))
        y_train = np.concatenate((y_train, np.zeros(len(idx)))) # Noise is labeled 0 for training
        
    X_test = test_data.iloc[:, :-1].to_numpy()
    y_test = test_data.iloc[:, -1].to_numpy()
    print(f"Samples after noise: {len(X_train)} | Features: {X_train.shape[1]}")
    return X_train, y_train, X_test, y_test

def evaluate_model(y_true, y_pred, y_probabilities=None):
    # Flip labels: Normal (0) -> 1, Anomaly (1) -> 0 for AUCPR focus (consistent with existing paper logic)
    y_true_flipped = (1 - y_true).astype(int)
    y_pred_flipped = (1 - y_pred).astype(int)
    
    mcc = matthews_corrcoef(y_true_flipped, y_pred_flipped)
    f1 = f1_score(y_true_flipped, y_pred_flipped, zero_division=0)
    ppv = precision_score(y_true_flipped, y_pred_flipped, zero_division=0)
    recall = recall_score(y_true_flipped, y_pred_flipped, zero_division=0)
    accuracy = accuracy_score(y_true_flipped, y_pred_flipped)
    
    auc_roc, auc_pr = None, None
    if y_probabilities is not None:
        try:
            # Probability of being normal (class 0)
            auc_roc = roc_auc_score(y_true_flipped, y_probabilities[:, 0])
            auc_pr = average_precision_score(y_true_flipped, y_probabilities[:, 0])
        except: pass
        
    return {
        "AUCROC": auc_roc * 100 if auc_roc is not None else None, 
        "AUCPR": auc_pr * 100 if auc_pr is not None else None,
        "Accuracy": accuracy * 100, "MCC": mcc, "F1 Score": f1, "Precision": ppv, "Recall": recall
    }

def get_model(name, **kwargs):
    # Standard PyOD Imports
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

    # Optional Wrapper-based Models
    DASVDD, NeuTraLAD, DIF, DRLAD , PMKFN = None, None, None, None, None 
    try:
        from baseline_model.dasvdd_wrapper import DASVDD
        from baseline_model.neutralad_wrapper import NeuTraLAD
        from baseline_model.dif_wrapper import DIF
        from baseline_model.DRLAD import DRLAD
        from baseline_model.PMKFN import PMKFN
    except Exception: pass

    class DRLADWrapper:
        def __init__(self, **kwargs):
            self.model = DRLAD(**kwargs)
        def fit(self, X, y=None):
            self.model.fit(X, y)
            self.decision_scores_ = self.model.decision_function(X)
            return self
        def predict(self, X):
            return self.model.predict(X)
        def decision_function(self, X):
            return self.model.decision_function(X)

    class PMKFNWrapper:
        def __init__(self, **kwargs):
            self.model = PMKFN(**kwargs)
        def fit(self, X, y=None):
            self.model.fit(X, y)
            self.decision_scores_ = self.model.decision_function(X)
            return self
        def predict(self, X):
            scores = self.model.decision_function(X)
            thr = np.percentile(scores, 95) # Top 5% as anomaly
            return (scores > thr).astype(int)
        def decision_function(self, X):
            return self.model.decision_function(X)

    model_dict = {
        "CBLOF": CBLOF, "KNN": KNN, "IForest": IForest, "OCSVM": OCSVM, "LOF": LOF, "DeepSVDD": DeepSVDD,
        "HBOS": HBOS, "LODA": LODA, "PCA": PCA, "ECOD": ECOD, "COPOD": COPOD, "AutoEncoder": AutoEncoder,
        "DevNet": DevNet, "LUNAR": LUNAR, "AE1SVM": AE1SVM, "ALAD": ALAD, "VAE": VAE, "SO_GAAL": SO_GAAL,
        "MO_GAAL": MO_GAAL, "DASVDD": DASVDD, "SUOD": SUOD, "NeuTraLAD": NeuTraLAD, "DIF": DIF,
        "DRLAD": DRLADWrapper, "PMKFN": PMKFNWrapper
    }
    
    m_class = model_dict.get(name)
    if m_class is None: 
        raise ValueError(f"Model {name} not found or required wrapper is missing.")
    return m_class(**kwargs)

# ============================================================================
# EXECUTION ENGINE
# ============================================================================

def run_experiment(X_train, y_train, X_test, y_test, dataset_name, noise_percentage, scaler, models_list):
    # Force float64 and C-contiguous for PCA/sklearn stability
    X_train = np.ascontiguousarray(X_train, dtype=np.float64)
    X_test = np.ascontiguousarray(X_test, dtype=np.float64)

    if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
        print(f"CRITICAL WARNING: Non-finite values detected in {dataset_name} ({scaler})")

    for model_name in models_list:
        try:
            print(f"\nRunning dataset {dataset_name} with model {model_name}")
            params = { "device": "cpu" } # Force CPU for stability
            
            tracemalloc.start()
            start_time = time.time()
            
            if model_name == 'DeepSVDD':
                model = get_model(model_name, n_features=X_train.shape[1], **params)
                model.fit(X_train)
            elif model_name == 'DASVDD':
                model = get_model(model_name, code_size=32, num_epochs=100, batch_size=128, verbose=0)
                model.fit(X_train)
            elif model_name == "DIF":
                model = get_model(model_name, n_ensemble=50, n_estimators=6, verbose=0, device='cpu')
                model.fit(X_train)
            elif model_name == "NeuTraLAD":
                model = get_model(model_name, latent_dim=32, enc_hdim=32, num_epochs=100, verbose=0, device='cpu')
                model.fit(X_train)
            elif model_name == "SUOD":
                from pyod.models.hbos import HBOS
                from pyod.models.copod import COPOD
                from pyod.models.ecod import ECOD
                model = get_model(model_name, base_estimators=[HBOS(), COPOD(), ECOD()], n_jobs=1, verbose=False)
                model.fit(X_train)
            elif model_name == "DRLAD":
                import torch
                dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
                model = get_model("DRLAD", in_features=X_train.shape[1], epochs=100, device=dev)
                model.fit(X_train)
            elif model_name == "PMKFN":
                model = get_model("PMKFN", max_train_size=3000)
                model.fit(X_train)
            else:
                model = get_model(model_name)
                if model_name == 'DevNet': 
                    model.fit(X_train, y_train)
                else: 
                    model.fit(X_train)
                
            # SANITIZE INTERNAL SCORES (Fix for PyOD overflow in predict_proba)
            # Some models like PCA can produce infinite outlier scores if eigenvalues are near-zero
            if hasattr(model, 'decision_scores_'):
                model.decision_scores_ = np.nan_to_num(model.decision_scores_, posinf=1e15, neginf=-1e15)
                
            train_time = time.time() - start_time
            _, peak_train = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            tracemalloc.start()
            start_time = time.time()
            y_pred = model.predict(X_test)
            
            # Defensive probability calculation
            y_probabilities = None
            if hasattr(model, "predict_proba"):
                try:
                    y_probabilities = model.predict_proba(X_test)
                except Exception as prob_e:
                    # ROBUST FALLBACK: Manual scaling of decision scores (Fix for PCA/ToNIoT/CICIoT stability)
                    try:
                        print(f"DEBUG: predict_proba failed for {model_name}, manual scaling fallback...")
                        test_scores = model.decision_function(X_test)
                        # Clean test scores
                        test_scores = np.nan_to_num(test_scores, posinf=1e15, neginf=-1e15)
                        train_scores = model.decision_scores_
                        # Fit our own scaling safely
                        s_min, s_max = np.min(train_scores), np.max(train_scores)
                        if s_max > s_min:
                            probs = (test_scores - s_min) / (s_max - s_min)
                        else:
                            probs = np.zeros_like(test_scores)
                        probs = np.clip(probs, 0.0, 1.0)
                        # y_probabilities: [prob_normal, prob_anomaly]
                        y_probabilities = np.vstack([1 - probs, probs]).T
                    except Exception as manual_e:
                        print(f"DEBUG: Manual scaling also failed for {model_name}: {manual_e}")
                        y_probabilities = None
            
            # Final probability sanitization
            if y_probabilities is not None:
                y_probabilities = np.nan_to_num(y_probabilities, nan=0.5, posinf=1.0, neginf=0.0)
                y_probabilities = np.clip(y_probabilities, 0.0, 1.0)
                    
            test_time = time.time() - start_time
            _, peak_test = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            m = evaluate_model(y_test, y_pred, y_probabilities=y_probabilities)
            result = [dataset_name, model_name, str(params), scaler, noise_percentage, 
                      m['AUCROC'], m['AUCPR'], m['Accuracy'], m['MCC'], m['F1 Score'], 
                      m['Precision'], m['Recall'], train_time, test_time, 
                      peak_train/10**6, peak_test/10**6]
            
            pd.DataFrame([result], columns=columns).to_csv(output_file, mode='a', header=False, index=False)
            print(f"Results saved for {dataset_name} with model {model_name}")
            
        except Exception as e:
            print(f"Error with dataset {dataset_name}, model {model_name}: {e}")
            traceback.print_exc()

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    if not os.path.exists(output_file):
        pd.DataFrame(columns=columns).to_csv(output_file, index=False)
        
    models_list = ['PMKFN', 'DRLAD']
    
    dataset_prefixes = ['data_ToNIoT.csv', 'data_N_BaIoT.csv', 'data_CICIoT2023.csv', 'data_BoTIoT.csv', 'data_EdgeIIoTset.csv', 'data_IoTID20.csv', 'data_FiveGNIDD.csv']
    # dataset_prefixes = ['data_EdgeIIoTset.csv', 'data_IoTID20.csv']
    
    scaler_names = ['StandardScaler', 'MinMaxScaler', 'Normalizer', 'QuantileTransformer', 'RobustScaler']
    
    imputer = SimpleImputer(strategy="mean")
    
    for prefix in dataset_prefixes:
        for scaler in scaler_names: 
            train_file = f'../../Datascaled/Official_OC_Data/Train_{scaler}_{prefix}'
            test_file = f'../../Datascaled/Official_OC_Data/Test_{scaler}_{prefix}'
            
            if not os.path.exists(train_file):
                print(f"File not found: {train_file}")
                continue
                
            print(f"\nProcessing {prefix} with {scaler} scaler...")
            df_train = pd.read_csv(train_file).dropna()
            df_test = pd.read_csv(test_file).dropna()
            df_full = pd.concat([df_train, df_test], ignore_index=True)
            
            df_train_new, df_test_new = train_test_split(df_full, test_size=0.3, random_state=42)
            
            for noise in [0, 1, 3, 5]: 
                X_train, y_train, X_test, y_test = preprocess_data_noise(df_train_new, df_test_new, noise)
                
                # Handle NaNs/Infs/Overflows (Extreme Clipping for PCA stability)
                # Using 1e5 as a very safe limit for any scaled dataset
                X_train = np.nan_to_num(X_train, nan=np.nan, posinf=1e5, neginf=-1e5)
                X_train = np.clip(X_train, -1e5, 1e5)
                X_train = imputer.fit_transform(X_train)
                
                X_test = np.nan_to_num(X_test, nan=np.nan, posinf=1e5, neginf=-1e5)
                X_test = np.clip(X_test, -1e5, 1e5)
                X_test = imputer.transform(X_test)
                
                # Final check to ensure NO NaNs remain (e.g. if a whole column was NaNs)
                X_train = np.nan_to_num(X_train, nan=0.0)
                X_test = np.nan_to_num(X_test, nan=0.0)
                
                if not np.isfinite(X_train).all():
                    print(f"CRITICAL WARNING: X_train still contains non-finite values after cleaning ({prefix})")
                
                run_experiment(X_train, y_train, X_test, y_test, prefix, noise, scaler, models_list)
