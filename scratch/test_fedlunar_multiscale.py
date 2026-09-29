import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, f1_score

from fed_lunar.federated.fed_lunar import FedLUNAR
from fed_lunar.baselines.naive_lunar import NaiveFedLunar
from fed_lunar.benchmark.data_loader import partition_and_prepare_dataset

data_dir = "/home/jupyter-iec_duongnt/New Project/NFST/GMM-nfst/Datascaled/Official_OC_Data"
datasets = ["BoTIoT", "EdgeIIoTset", "CICIoT2023", "N_BaIoT"]

print("=" * 70)
print("TESTING FEDLUNAR WITH MULTI-SCALE NEGATIVES ACROSS ALL 4 DATASETS")
print("=" * 70)

for ds in datasets:
    client_data, X_te, y_te, meta = partition_and_prepare_dataset(
        dataset_name=ds,
        data_dir=data_dir,
        num_clients=3,
        alpha=0.5,
        max_train_samples=4000,
        max_test_samples=4000,
        seed=42,
    )
    print(f"\n--- Dataset: {ds} (Dim: {meta['input_dim']}, Test Total: {len(y_te)}, Attack%: {np.mean(y_te):.1%}) ---")

    # FedLUNAR
    model = FedLUNAR(
        k=10,
        rank=10,
        mode="CAGrad",
        multi_scale=True,
        scales=[0.2, 0.5, 1.5, 3.0, 6.0],
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=42,
    )
    model.fit(client_data, rounds=5)
    scores = model.decision_function(X_te)
    auc = roc_auc_score(y_te, scores) * 100.0
    print(f"FedLUNAR AUC-ROC: {auc:.2f}% | Norm Score Mean: {scores[y_te==0].mean():.4f} | Anom Score Mean: {scores[y_te==1].mean():.4f}")

    # Naive FedLUNAR
    naive = NaiveFedLunar(
        k=10,
        multi_scale=True,
        scales=[0.2, 0.5, 1.5, 3.0, 6.0],
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=42,
    )
    naive.fit(client_data, rounds=5)
    naive_scores = naive.decision_function(X_te)
    naive_auc = roc_auc_score(y_te, naive_scores) * 100.0
    print(f"Naive AUC-ROC:    {naive_auc:.2f}% | Norm Score Mean: {naive_scores[y_te==0].mean():.4f} | Anom Score Mean: {naive_scores[y_te==1].mean():.4f}")
