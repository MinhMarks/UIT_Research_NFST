#!/usr/bin/env python3
"""
Dirichlet Non-IID Sensitivity Sweep for Federated LUNAR.
Evaluates alpha in {0.1, 0.5, 1.0, 5.0} across Proposed Fed-LUNAR, Naive Fed-LUNAR,
FedProx, and PCGrad to demonstrate DROGA and CMNP resilience under extreme non-IID conditions.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score

from fed_lunar.federated.fed_lunar import FedLUNAR
from fed_lunar.baselines.naive_lunar import NaiveFedLunar
from fed_lunar.baselines.fedprox_lunar import FedProxLunar, PCGradFedLunar
from fed_lunar.benchmark.data_loader import partition_and_prepare_dataset
from fed_lunar.benchmark.metrics import calculate_detection_metrics

data_dir = "/home/jupyter-iec_duongnt/New Project/NFST/GMM-nfst/Datascaled/Official_OC_Data"
output_dir = "outputs/lunar_results/sensitivity_sweep"
os.makedirs(output_dir, exist_ok=True)

alphas = [0.1, 0.5, 1.0, 5.0]
datasets = ["BoTIoT", "CICIoT2023"]
methods = ["Proposed_FedLUNAR", "Naive_FedLUNAR", "FedProx_LUNAR", "PCGrad_FedLUNAR"]

results = []

for ds in datasets:
    for alpha in alphas:
        print(f"\n==================== Sweep: {ds} | Alpha = {alpha} ====================")
        client_data, X_te, y_te, meta = partition_and_prepare_dataset(
            dataset_name=ds,
            data_dir=data_dir,
            num_clients=3,
            alpha=alpha,
            max_train_samples=6000,
            max_test_samples=4000,
            seed=42,
        )

        for m_name in methods:
            if m_name == "Proposed_FedLUNAR":
                model = FedLUNAR(k=10, rank=10, mode="CAGrad", enable_cmnp=True, lr=0.002, local_epochs=4, device="cuda", seed=42)
            elif m_name == "Naive_FedLUNAR":
                model = NaiveFedLunar(k=10, lr=0.002, local_epochs=4, device="cuda", seed=42)
            elif m_name == "FedProx_LUNAR":
                model = FedProxLunar(k=10, mu=0.01, lr=0.002, local_epochs=4, device="cuda", seed=42)
            elif m_name == "PCGrad_FedLUNAR":
                model = PCGradFedLunar(k=10, lr=0.002, local_epochs=4, device="cuda", seed=42)

            model.fit(client_data, rounds=10)
            scores = model.decision_function(X_te)
            metrics = calculate_detection_metrics(y_te, scores)
            
            # GCR
            history = getattr(model, "history", [])
            conflicts = sum(1 for h in history if h.get("pre_gcr", 0.0) > 0)
            gcr = (conflicts / len(history)) * 100.0 if history else 0.0

            res = {
                "dataset": ds,
                "alpha": alpha,
                "method": m_name,
                "auc_roc": metrics["auc_roc"],
                "f1_score": metrics["f1_score"],
                "f1_optimal": metrics["f1_optimal"],
                "far": metrics["far"],
                "detection_rate": metrics["detection_rate"],
                "round_conflict_ratio": gcr,
            }
            results.append(res)
            print(f"[{m_name}] Alpha={alpha} -> AUC-ROC: {metrics['auc_roc']}% | F1: {metrics['f1_optimal']}% | Conflict Rds: {gcr:.1f}%")

df_sweep = pd.DataFrame(results)
sweep_csv = os.path.join(output_dir, "alpha_sensitivity_summary.csv")
df_sweep.to_csv(sweep_csv, index=False)
print(f"\n[SAVED] Sweep saved to: {sweep_csv}")
print("\n" + df_sweep.to_string(index=False))
