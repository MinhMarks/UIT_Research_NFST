import os
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    matthews_corrcoef, f1_score, precision_score, recall_score,
    accuracy_score, roc_auc_score, average_precision_score
)
from sklearn.impute import SimpleImputer

from pyod.models.ae1svm import AE1SVM
from pyod.models.devnet import DevNet
from pyod.models.lunar import LUNAR
from pyod.models.cblof import CBLOF
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.ocsvm import OCSVM
from pyod.models.lof import LOF
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.hbos import HBOS
from pyod.models.pca import PCA
from pyod.models.sod import SOD
from pyod.models.cof import COF
from pyod.models.loda import LODA
from pyod.models.alad import ALAD
from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.vae import VAE
from pyod.models.so_gaal import SO_GAAL  # đúng: "sogaal" (không phải "so_gaal")
from pyod.models.mo_gaal import MO_GAAL  # đúng: "mogaal" (không phải "mo_gaal")
from pyod.models.suod import SUOD

import sys
import os
sys.path.append("../..")

from baseline_model.dasvdd_wrapper import DASVDD
from baseline_model.neutralad_wrapper import NeuTraLAD
from baseline_model.dif_wrapper import DIF

models_list = [
    "KNN",
    "LUNAR",
    "NeuTraLAD",     
    "LOF",
    "AutoEncoder",
    "CBLOF",
    "HBOS",
    "DASVDD", 
    "PCA",
    "AE1SVM",
    "DevNet",
    "DeepSVDD",
    "IForest",
    "OCSVM",
    "LODA",
    "MO_GAAL", 
    "SUOD", 
    "DIF", 
    "ALAD",
    "COPOD",
    "ECOD",
    "VAE", 
    "SO_GAAL",
]

output_file = f"additionalBaselineNoise.csv"
# output_file = f"../../Results/additionalBaselineNoise.csv"
columns = ["Dataset","Model",  "scaled", "noise_percentage", "AUCROC", "AUCPR", "Accuracy", "MCC", "F1 Score",
           "Precision", "Recall", "Time Train", "Time Test"]

def preprocess_data_noise(train_data, test_data, noise_percentage=10):
    """
    Preprocess the training and testing data by separating features and labels,
    and return the preprocessed feature sets and labels with added noise.
    
    Args:
        train_data (DataFrame): Training data containing features and labels.
        test_data (DataFrame): Testing data containing features and labels.
        noise_percentage (float): The percentage of training samples to add noise to.
        
    Returns:
        X_train (ndarray): Preprocessed training features with added noise.
        y_train (ndarray): Preprocessed training labels with added noise.
        X_test (ndarray): Preprocessed testing features.
        y_test (ndarray): Preprocessed testing labels.
    """
    print("..............................Data Overview................................")
    print("Train Data Shape:", train_data.shape)
    print("Test Data Shape:", test_data.shape)
    
    # Convert to numpy arrays for easier manipulation
    X_train_total = train_data.iloc[:, :-1].to_numpy()
    y_train_total = train_data.iloc[:, -1].to_numpy()

    # Separate the samples with label 0
    X_train = X_train_total[y_train_total == 0]
    y_train = y_train_total[y_train_total == 0]

    print("Train Data Labels [0]:", np.unique(y_train))

    # Calculate how many samples to add noise to based on the provided percentage
    n_samples = X_train.shape[0]
    noise_samples_count = int(n_samples * (noise_percentage / 100))

    # Get the samples with label 1 (for generating noise)
    X_train_noise = X_train_total[y_train_total == 1]
    
    # Randomly select noise_samples_count from X_train_noise
    noisy_indices = np.random.choice(X_train_noise.shape[0], size=noise_samples_count, replace=False)
    X_train_noise = X_train_noise[noisy_indices]
    
    # Add the noisy samples to the training set
    X_train = np.vstack((X_train, X_train_noise))
    # y_train = np.concatenate((y_train, np.ones(X_train_noise.shape[0])))
    y_train = np.concatenate((y_train, np.zeros(X_train_noise.shape[0])))

    print(y_train) 
    # Prepare test data
    X_test = test_data.iloc[:, :-1].to_numpy()
    y_test = test_data.iloc[:, -1].to_numpy()

    # Print the new size of training data
    n_samples = X_train.shape[0]
    n_features = X_train.shape[1]
    print("Number of samples after adding noise:", n_samples)
    print("Number of features:", n_features)

    return X_train, y_train, X_test, y_test


def evaluate_model(y_true, y_pred, y_scores=None, y_probabilities=None):
    """
    Evaluates the model using multiple metrics and prints the results.
    This version includes AUCROC and AUCPR using predicted probabilities.

    Parameters:
    - y_true (np.ndarray): True labels of the test data.
    - y_pred (np.ndarray): Predicted labels of the test data.
    - y_scores (np.ndarray, optional): Scores used to compute AUCROC and AUCPR.
    - y_probabilities (np.ndarray, optional): Predicted probabilities for each class.

    Returns:
    - metrics (list): A list containing AUCROC, AUCPR, accuracy, MCC, F1 score, precision, and recall.
    """
    print("..............................Evaluation Metrics...............................")

    # Calculate standard metrics
    mcc = matthews_corrcoef(y_true, y_pred)  # Matthew's correlation coefficient
    f1 = f1_score(y_true, y_pred)  # F1 score
    precision = precision_score(y_true, y_pred)  # Precision
    recall = recall_score(y_true, y_pred)  # Recall
    accuracy = accuracy_score(y_true, y_pred)  # Accuracy

    # ROC and PR curve scores using predicted probabilities
    auc_roc, auc_pr = None, None
    if y_probabilities is not None:
        auc_roc = roc_auc_score(y_true, y_probabilities[:, 1])  # Probabilities for class 1 (outliers)
        auc_pr = average_precision_score(y_true, y_probabilities[:, 1])  # AUCPR for class 1

    # Display metrics
    print(f"AUCROC: {auc_roc * 100 if auc_roc else 'N/A'}")
    print(f"AUCPR: {auc_pr * 100 if auc_pr else 'N/A'}")
    print(f"Accuracy: {accuracy * 100:.2f}")
    print(f"MCC: {mcc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    # Return the evaluation metrics as a list
    return [auc_roc * 100 if auc_roc else None, auc_pr * 100 if auc_pr else None,
            accuracy * 100, mcc, f1, precision, recall]


def get_model(name, **kwargs):
    """
    Returns the appropriate PyOD model based on the provided name.
    
    Parameters:
    - name (str): The name of the anomaly detection model (e.g., 'CBLOF', 'KNN').
    - kwargs (dict): Additional parameters to initialize the model.

    Returns:
    - model (object): An instance of the specified anomaly detection model.
    """
    model_dict = {
        "CBLOF": CBLOF,
        "KNN": KNN,
        "IForest": IForest,
        "OCSVM": OCSVM,
        "LOF": LOF,
        "DeepSVDD": DeepSVDD,
        "HBOS": HBOS,
        "SOD": SOD,
        "COF": COF,
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
        "DASVDD": DASVDD, 
        "SUOD": SUOD, 
        "NeuTraLAD": NeuTraLAD,
        "DIF": DIF, 
    }
    
    # Fetch the model class based on the provided model name
    model_class = model_dict.get(name)
    
    # Raise an error if the model name is not found
    if model_class is None:
        raise ValueError(f"Model {name} not found.")
    
    # Return the model initialized with the provided parameters
    return model_class(**kwargs)


# Initialize output CSV file
if not os.path.exists(output_file):
    pd.DataFrame(columns=columns).to_csv(output_file, index=False)

# Define a function to run the model training and evaluation
def run_experiment(X_train, y_train, X_test, y_test, name, noise_percentage, scaler ):
    for model_name in models_list:
        try:
            print(f"\nRunning dataset {name} with model {model_name}")
            
            # train_data = pd.concat([X_train, y_train], axis=1, ignore_index=True)
            # test_data = pd.concat([X_test, y_test], axis=1, ignore_index=True)

            # # Preprocess data (assuming preprocess_data is defined elsewhere)
            # X_train, y_train, X_test, y_test = preprocess_data(train_data, test_data)
            
            if model_name == 'DeepSVDD':
                start_time = time.time()
                n_features = X_train.shape[1]
                model = DeepSVDD(n_features=n_features)
                model.fit(X_train)
            elif model_name == 'DASVDD':
                start_time = time.time()
                model = DASVDD(
                    code_size=32,
                    num_epochs=100,
                    batch_size=128,
                    lr=1e-3,
                    K=0.9,
                    T=10,
                    verbose=1
                )
                model.fit(X_train)
            elif model_name == "DIF": 
                start_time = time.time()
                model = DIF(n_ensemble=50, n_estimators=6)

                # Hoặc với custom config
                model = DIF(
                    n_ensemble=50,
                    n_estimators=6,
                    max_samples=256,
                    hidden_dim=[500, 100],
                    rep_dim=20,
                    activation='tanh',
                    batch_size=64,
                    device='cuda',
                    verbose=1
                )
                model.fit(X_train)
                
            elif model_name == "NeuTraLAD": 
                start_time = time.time()
                model = NeuTraLAD(
                    latent_dim=32,      # Kích thước latent space
                    enc_hdim=32,        # Hidden dim encoder
                    num_trans=11,       # Số transformations
                    num_epochs=200,     # Số epochs
                    batch_size=128,
                    lr=0.001,
                    device='cuda',      # hoặc 'cpu'
                    verbose=1           # 1 để hiển thị progress
                )
                # Fit model (chỉ dùng normal data)
                model.fit(X_train)
            elif model_name == "SUOD": 
                start_time = time.time()
                
                base_estimators = [
                    HBOS(n_bins=10),
                    COPOD(),
                    ECOD(),
                    IForest(n_estimators=50, max_samples=128, random_state=42),
                    PCA(n_components=10, random_state=42),
                ]
                
                model = SUOD(
                    base_estimators=base_estimators,
                    n_jobs=1,                    # BẮT BUỘC = 1 để tránh hoàn toàn bug joblib
                    rp_flag_global=True,         # vẫn tăng tốc
                    bps_flag=False,              # tắt BPS → nguyên nhân chính của lỗi
                    contamination=0.1,
                    verbose=True                 # in chi tiết để bạn thấy nó chạy
                )
                
                model.fit(X_train)  # fit all models with X
            else:
                start_time = time.time()
                model = get_model(model_name)
                if model_name == 'DevNet':
                    model.fit(X_train, y_train)
                else:
                    model.fit(X_train)
                    
            train_time = time.time() - start_time

           #  columns = ["Dataset", "Model", "outlier_mode", "scaler", "AUCROC", "AUCPR", "Accuracy", "MCC", "F1 Score",
           # "Precision", "Recall", "Time Train", "Time Test"]
            # Test the model
            start_time = time.time()
            y_pred = model.predict(X_test)
            # Check if the model supports predict_proba and calculate probabilities
            if hasattr(model, "predict_proba"):
                y_probabilities = model.predict_proba(X_test)  # Class probabilities for each instance
            else:
                y_probabilities = None  # Some models do not have predict_proba method

            test_time = time.time() - start_time

           #  columns = ["Dataset","Model",  "scaled", "noise_percentage", "AUCROC", "AUCPR", "Accuracy", "MCC", "F1 Score",
           # "Precision", "Recall", "Time Train", "Time Test"]
            
            # Evaluate the model using the appropriate probabilities
            metrics = evaluate_model(y_test, y_pred, y_scores=None, y_probabilities=y_probabilities)
            result = [name, model_name] + [scaler, noise_percentage]+ metrics + [train_time, test_time]
            result_df = pd.DataFrame([result], columns=columns)
            result_df.to_csv(output_file, mode='a', header=False, index=False)

            print(f"Results saved for {name} with model {model_name}")

        except Exception as e:
            print(f"Error with dataset {name}, model {model_name}: {e}")



if __name__ == "__main__": 
    
    dataset_prefixes =  ['data_ToNIoT.csv', 'data_N_BaIoT.csv' , 'data_CICIoT2023.csv', 'data_BoTIoT.csv', 'data_EdgeIIoTset.csv', 'data_IoTID20.csv', 'data_FiveGNIDD.csv']
    
    # scaler_names = ['MinMaxScaler'] 'StandardScaler', 
    scaler_names = ['QuantileTransformer', 'MinMaxScaler','Normalizer','RobustScaler']
    
    # scaler_names = ['QuantileTransformer']
    
    for prefix in dataset_prefixes:
        
        for scaler in scaler_names: 
         
            # Construct file paths for train and test datasets with 'Train_' and 'Test_' prefixes
            train_file = f'../../Datascaled/NoiseOCData/Train_{scaler}_{prefix}'
            test_file = f'../../Datascaled/NoiseOCData/Test_{scaler}_{prefix}'
            
            # Load the CSV files
            df_train = pd.read_csv(train_file)
            df_test = pd.read_csv(test_file)

            
            df_train = df_train.dropna() 
            df_test = df_test.dropna() 
            
            # Nối lại thành 1 DataFrame
            df_full = pd.concat([df_train, df_test], ignore_index=True)
            
            # Chia theo tỉ lệ 70% train, 30% test
            df_train_new, df_test_new = train_test_split(df_full, test_size=0.3, random_state=42)

            # print(f"Số lượng hàng có NaN: {num_rows_with_nan}")

            for noise in [0, 1, 3, 5]: 
                
                # Step 1: Preprocess data
                X_train, y_train, X_test, y_test = preprocess_data_noise(df_train_new, df_test_new, noise)
                imputer = SimpleImputer(strategy="mean") 
                X_train[np.isinf(X_train)] = np.nan  # Đổi vô hạn thành NaN
                X_train = imputer.fit_transform(X_train)
                
                # Run the experiments for each dataset type
                run_experiment(X_train, y_train, X_test, y_test, prefix, noise, scaler ) 
               

