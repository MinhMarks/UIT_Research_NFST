#!/usr/bin/env python3
"""
Master Benchmark Runner for Federated LUNAR against 3-Tier Baseline Hierarchy.

Evaluates:
- Proposed Fed-LUNAR: Cross-Manifold Negative Purging (CMNP) + DROGA (DR-CAGrad).
- Tier 1: NaiveFedLunar (Standard FedAvg on LUNAR with uncoordinated perturbation).
- Tier 2: FedAutoEncoder (Deep Autoencoder with MSE loss), FedProxLunar, PCGradFedLunar.
- Tier 3: LOC_NFST_Bound (Closed-form spectral null-space projection).
- Ablations: Fed-LUNAR without DROGA (CMNP only), Fed-LUNAR without CMNP (DROGA only).

Evaluated across canonical IoT Intrusion Detection Datasets:
- BoTIoT
- EdgeIIoTset
- CICIoT2023
- N_BaIoT

Usage:
    python -m fed_lunar.benchmark.run_benchmark --dataset all --clients 3 --alpha 0.5 --rounds 10
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import torch

from fed_lunar.federated.fed_lunar import FedLUNAR
from fed_lunar.baselines.naive_lunar import NaiveFedLunar
from fed_lunar.baselines.fed_ae import FedAutoEncoder
from fed_lunar.baselines.fedprox_lunar import FedProxLunar, PCGradFedLunar
from fed_lunar.baselines.loc_nfst_bound import LOC_NFST_Bound
from fed_lunar.benchmark.data_loader import partition_and_prepare_dataset
from fed_lunar.benchmark.metrics import calculate_detection_metrics, measure_inference_latency


def get_default_data_dir() -> str:
    """Find default directory for Official_OC_Data across local and server paths."""
    candidates = [
        "/home/jupyter-iec_duongnt/New Project/NFST/GMM-nfst/Datascaled/Official_OC_Data",
        "d:/UIT/Research/IEC2023/LOC-NFST/UIT_Research_NFST/Datascaled/Official_OC_Data",
        "./Datascaled/Official_OC_Data",
        "../Datascaled/Official_OC_Data",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


def instantiate_model(model_name: str, device: str, seed: int = 42) -> Any:
    """Instantiate model by canonical benchmark name."""
    if model_name == "Proposed_FedLUNAR":
        return FedLUNAR(
            k=10,
            rank=10,
            c_param=0.4,
            mode="CAGrad",
            tau_null=1.0,
            alpha=0.01,
            lr=0.001,
            batch_size=128,
            local_epochs=3,
            device=device,
            seed=seed,
        )
    elif model_name == "Ablation_FedLUNAR_NoDROGA":
        # CMNP enabled, but standard FedAvg instead of DROGA
        return FedLUNAR(
            k=10,
            rank=10,
            mode="FedAvg",
            tau_null=1.0,
            alpha=0.01,
            lr=0.001,
            batch_size=128,
            local_epochs=3,
            device=device,
            seed=seed,
        )
    elif model_name == "Ablation_FedLUNAR_NoCMNP":
        # DROGA enabled, but NO CMNP filtering (tau_null=0 -> no negatives purged)
        return FedLUNAR(
            k=10,
            rank=10,
            c_param=0.4,
            mode="CAGrad",
            tau_null=0.0,
            alpha=0.0,
            lr=0.001,
            batch_size=128,
            local_epochs=3,
            device=device,
            seed=seed,
        )
    elif model_name == "Naive_FedLUNAR":
        return NaiveFedLunar(
            k=10,
            lr=0.001,
            batch_size=128,
            local_epochs=3,
            device=device,
            seed=seed,
        )
    elif model_name == "FedAutoEncoder":
        return FedAutoEncoder(
            code_size=32,
            lr=0.001,
            batch_size=128,
            local_epochs=3,
            device=device,
            seed=seed,
        )
    elif model_name == "FedProx_LUNAR":
        return FedProxLunar(
            k=10,
            mu=0.01,
            lr=0.001,
            batch_size=128,
            local_epochs=3,
            device=device,
            seed=seed,
        )
    elif model_name == "PCGrad_FedLUNAR":
        return PCGradFedLunar(
            k=10,
            lr=0.001,
            batch_size=128,
            local_epochs=3,
            device=device,
            seed=seed,
        )
    elif model_name == "LOC_NFST_Bound":
        return LOC_NFST_Bound(
            n_components=10,
            tol=1e-5,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def run_benchmark(args):
    """Main benchmark execution loop."""
    print("=" * 80)
    print("FED-LUNAR BENCHMARK SUITE: NON-IID IOT ANOMALY DETECTION")
    print("=" * 80)
    print(f"Data Directory:     {args.data_dir}")
    print(f"Output Directory:   {args.output_dir}")
    print(f"Number of Clients:  {args.clients}")
    print(f"Dirichlet Alpha:    {args.alpha} (Non-IID skew)")
    print(f"Communication Rds:  {args.rounds}")
    print(f"Device:             {args.device}")
    print(f"Seed:               {args.seed}")
    print("=" * 80)

    os.makedirs(args.output_dir, exist_ok=True)

    # Determine datasets to evaluate
    if args.dataset.lower() == "all":
        datasets = ["BoTIoT", "EdgeIIoTset", "CICIoT2023", "N_BaIoT"]
    else:
        datasets = [d.strip() for d in args.dataset.split(",") if d.strip()]

    # Determine models to evaluate
    if args.models.lower() == "all":
        model_names = [
            "Proposed_FedLUNAR",
            "Naive_FedLUNAR",
            "FedAutoEncoder",
            "FedProx_LUNAR",
            "PCGrad_FedLUNAR",
            "LOC_NFST_Bound",
            "Ablation_FedLUNAR_NoDROGA",
            "Ablation_FedLUNAR_NoCMNP",
        ]
    elif args.models.lower() == "core":
        model_names = [
            "Proposed_FedLUNAR",
            "Naive_FedLUNAR",
            "FedAutoEncoder",
            "FedProx_LUNAR",
            "PCGrad_FedLUNAR",
            "LOC_NFST_Bound",
        ]
    else:
        model_names = [m.strip() for m in args.models.split(",") if m.strip()]

    all_results: List[Dict[str, Any]] = []

    for ds_idx, dataset in enumerate(datasets, 1):
        print(f"\n[{ds_idx}/{len(datasets)}] Loading and Partitioning: {dataset} (Dirichlet alpha={args.alpha})...")

        try:
            client_train_data, X_test, y_test, meta = partition_and_prepare_dataset(
                dataset_name=dataset,
                data_dir=args.data_dir,
                num_clients=args.clients,
                alpha=args.alpha,
                scaler="StandardScaler",
                max_train_samples=args.max_train_samples,
                max_test_samples=args.max_test_samples,
                seed=args.seed,
            )
            print(f"    Features: {meta['input_dim']} | Train Normal: {meta['total_train_normal']} across {meta['num_clients']} clients: {meta['client_sizes']}")
            print(f"    Test Total: {meta['test_total']} (Normal: {meta['test_normal_count']}, Attack: {meta['test_attack_count']}, Attack Ratio: {meta['test_attack_ratio']:.1%})")
        except Exception as e:
            print(f"    [ERROR] Failed to load dataset {dataset}: {e}")
            continue

        for m_idx, m_name in enumerate(model_names, 1):
            print(f"    --> [{m_idx}/{len(model_names)}] Evaluating {m_name}...")
            try:
                model = instantiate_model(m_name, device=args.device, seed=args.seed)

                # Track training wall-clock time
                t_start = time.perf_counter()
                if hasattr(model, "fit"):
                    # LOC-NFST is one-shot spectral (rounds not applicable or ignored)
                    if m_name == "LOC_NFST_Bound":
                        model.fit(client_train_data)
                    else:
                        model.fit(client_train_data, rounds=args.rounds, verbose=False)
                t_train = time.perf_counter() - t_start

                # Test scoring
                if hasattr(model, "decision_function"):
                    y_scores = model.decision_function(X_test)
                elif hasattr(model, "score"):
                    y_scores = model.score(X_test)
                else:
                    raise AttributeError(f"Model {m_name} has no decision_function or score method")

                # Metrics calculation
                metrics = calculate_detection_metrics(y_true=y_test, y_scores=y_scores)

                # Inference latency (ms / sample)
                latency_ms = measure_inference_latency(model, X_test, n_runs=2)

                # Gradient conflict ratio if available
                final_gcr = None
                mean_cosine = None
                if hasattr(model, "history") and len(model.history) > 0:
                    last_h = model.history[-1]
                    final_gcr = last_h.get("pre_gcr", last_h.get("gcr", None))
                    mean_cosine = last_h.get("pre_mean_cosine", last_h.get("mean_cosine", None))

                result_row = {
                    "dataset": dataset,
                    "model": m_name,
                    "auc_roc": metrics["auc_roc"],
                    "f1_macro": metrics["f1_macro"],
                    "f1_binary": metrics["f1_binary"],
                    "precision": metrics["precision"],
                    "detection_rate": metrics["detection_rate"],
                    "far": metrics["far"],
                    "latency_ms": latency_ms,
                    "train_time_sec": round(t_train, 2),
                    "gcr": round(final_gcr * 100.0, 1) if final_gcr is not None else "N/A",
                    "mean_cosine": round(mean_cosine, 3) if mean_cosine is not None else "N/A",
                    "num_clients": args.clients,
                    "dirichlet_alpha": args.alpha,
                    "rounds": args.rounds if m_name != "LOC_NFST_Bound" else 1,
                    "input_dim": meta["input_dim"],
                }
                all_results.append(result_row)

                print(
                    f"        [RESULT] AUC-ROC: {metrics['auc_roc']}% | "
                    f"F1-Macro: {metrics['f1_macro']}% | "
                    f"FAR: {metrics['far']}% | "
                    f"DR: {metrics['detection_rate']}% | "
                    f"Latency: {latency_ms:.3f} ms | "
                    f"Train: {t_train:.1f}s"
                )

            except Exception as e:
                print(f"        [ERROR] Failed evaluating {m_name} on {dataset}: {e}")
                import traceback
                traceback.print_exc()

    # Save to CSV and JSON
    if all_results:
        df_results = pd.DataFrame(all_results)
        csv_path = os.path.join(args.output_dir, "benchmark_summary.csv")
        df_results.to_csv(csv_path, index=False)
        print(f"\n[SAVED] Benchmark summary saved to: {csv_path}")

        json_path = os.path.join(args.output_dir, "benchmark_summary.json")
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"[SAVED] JSON metadata saved to: {json_path}")

        # Print formatted markdown table
        print("\n" + "=" * 80)
        print("AGGREGATED BENCHMARK SUMMARY TABLE")
        print("=" * 80)
        pivot_cols = ["dataset", "model", "auc_roc", "f1_macro", "detection_rate", "far", "latency_ms", "train_time_sec"]
        print(df_results[pivot_cols].to_markdown(index=False))
        print("=" * 80)

    else:
        print("[WARNING] No benchmark results collected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated LUNAR Benchmark Suite")
    parser.add_argument("--data_dir", type=str, default=get_default_data_dir(), help="Path to Official_OC_Data CSVs")
    parser.add_argument("--output_dir", type=str, default="outputs/lunar_results", help="Directory for benchmark outputs")
    parser.add_argument("--dataset", type=str, default="all", help="Dataset name ('all' or comma-separated)")
    parser.add_argument("--models", type=str, default="all", help="Models to benchmark ('all', 'core', or comma-separated)")
    parser.add_argument("--clients", type=int, default=3, help="Number of federated edge clients")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet concentration parameter")
    parser.add_argument("--rounds", type=int, default=10, help="Federated communication rounds")
    parser.add_argument("--max_train_samples", type=int, default=20000, help="Max training samples per dataset")
    parser.add_argument("--max_test_samples", type=int, default=10000, help="Max test samples per dataset")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device ('cuda' or 'cpu')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    run_benchmark(args)
