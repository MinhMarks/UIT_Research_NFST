import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from fed_lunar.federated.fed_lunar import FedLUNAR
from fed_lunar.baselines.naive_lunar import NaiveFedLunar
from fed_lunar.benchmark.data_loader import partition_and_prepare_dataset

data_dir = "/home/jupyter-iec_duongnt/New Project/NFST/GMM-nfst/Datascaled/Official_OC_Data"

for ds in ["CICIoT2023", "N_BaIoT"]:
    client_data, X_te, y_te, meta = partition_and_prepare_dataset(
        dataset_name=ds,
        data_dir=data_dir,
        num_clients=3,
        alpha=0.5,
        max_train_samples=6000,
        max_test_samples=4000,
        seed=42,
    )
    print(f"\n==================== Testing {ds} (Dim: {meta['input_dim']}) with 10 rounds ====================")
    model = FedLUNAR(
        k=10,
        rank=10,
        lr=0.002,
        local_epochs=4,
        mode="CAGrad",
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=42,
    )
    model.fit(client_data, rounds=10, verbose=True)
    scores = model.decision_function(X_te)
    auc = roc_auc_score(y_te, scores) * 100.0
    print(f"[RESULT] {ds} 10-round AUC-ROC: {auc:.2f}% | Normal mean: {scores[y_te==0].mean():.4f} | Anom mean: {scores[y_te==1].mean():.4f}")
