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
- BoTIoT (D=35)
- EdgeIIoTset (D=42)
- CICIoT2023 (D=46)
- N_BaIoT (D=115)

Features:
- Dual CSV logging:
  1. Detailed per-run round-by-round trajectory: outputs/lunar_results/{dataset_name}_{method}.csv
  2. Master 20-column summary: outputs/lunar_results/benchmark_summary.csv
- Automated peak RAM/VRAM memory tracking and latency profiling.
- Formatted Markdown summary table output.

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
from fed_lunar.benchmark.metrics import (
    calculate_detection_metrics,
    calculate_optimization_dynamics,
    measure_inference_latency,
    MemoryTracker,
)


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


ALIAS_MAP = {
    "proposed": "Proposed_FedLUNAR",
    "proposed_fedlunar": "Proposed_FedLUNAR",
    "fedlunar": "Proposed_FedLUNAR",
    "fed-lunar-novel": "Proposed_FedLUNAR",
    "ablation_nodroga": "Ablation_FedLUNAR_NoDROGA",
    "ablation_fedlunar_nodroga": "Ablation_FedLUNAR_NoDROGA",
    "ablation_nocmnp": "Ablation_FedLUNAR_NoCMNP",
    "ablation_fedlunar_nocmnp": "Ablation_FedLUNAR_NoCMNP",
    "naive": "Naive_FedLUNAR",
    "naive_fedlunar": "Naive_FedLUNAR",
    "naivefedlunar": "Naive_FedLUNAR",
    "fedautoencoder": "FedAutoEncoder",
    "fed_ae": "FedAutoEncoder",
    "fedae": "FedAutoEncoder",
    "fedprox": "FedProx_LUNAR",
    "fedprox_lunar": "FedProx_LUNAR",
    "pcgrad": "PCGrad_FedLUNAR",
    "pcgrad_fedlunar": "PCGrad_FedLUNAR",
    "loc_nfst": "LOC_NFST_Bound",
    "loc_nfst_bound": "LOC_NFST_Bound",
    "loc-nfst": "LOC_NFST_Bound",
}


def resolve_model_name(name: str) -> str:
    cleaned = name.strip()
    return ALIAS_MAP.get(cleaned.lower(), cleaned)


def instantiate_model(model_name: str, device: str, seed: int = 42) -> Any:
    """Instantiate model by canonical benchmark name."""
    resolved = resolve_model_name(model_name)
    if resolved == "Proposed_FedLUNAR":
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
    elif resolved == "Ablation_FedLUNAR_NoDROGA":
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
    elif resolved == "Ablation_FedLUNAR_NoCMNP":
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
    elif resolved == "Naive_FedLUNAR":
        return NaiveFedLunar(
            k=10,
            lr=0.001,
            batch_size=128,
            local_epochs=3,
            device=device,
            seed=seed,
        )
    elif resolved == "FedAutoEncoder":
        return FedAutoEncoder(
            code_size=32,
            lr=0.001,
            batch_size=128,
            local_epochs=3,
            device=device,
            seed=seed,
        )
    elif resolved == "FedProx_LUNAR":
        return FedProxLunar(
            k=10,
            mu=0.01,
            lr=0.001,
            batch_size=128,
            local_epochs=3,
            device=device,
            seed=seed,
        )
    elif resolved == "PCGrad_FedLUNAR":
        return PCGradFedLunar(
            k=10,
            lr=0.001,
            batch_size=128,
            local_epochs=3,
            device=device,
            seed=seed,
        )
    elif resolved == "LOC_NFST_Bound":
        return LOC_NFST_Bound(
            n_components=10,
            tol=1e-5,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name} (resolved to {resolved})")


def write_per_run_trajectory_csv(
    filepath: str,
    dataset: str,
    method: str,
    clients: int,
    alpha: float,
    total_rounds: int,
    model: Any,
    t_train: float,
    metrics: Dict[str, Any],
    latency_ms_per_sample: float,
    peak_memory_mb: float,
) -> None:
    """Writes detailed round-by-round trajectory for an evaluated model to CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    history = getattr(model, "history", [])

    trajectory_cols = [
        "dataset", "method", "clients", "alpha", "round", "total_rounds",
        "round_train_loss", "gcr_pre", "mean_cosine_pre", "min_cosine_pre",
        "conflicting_pairs", "total_pairs", "aligned_norm", "cmnp_rejection_rate",
        "round_time_sec", "final_auc_roc", "final_f1_optimal", "final_f1_calibrated",
        "final_far", "latency_ms_per_sample", "peak_memory_mb"
    ]

    rows: List[Dict[str, Any]] = []

    if history and len(history) > 0:
        for idx, h in enumerate(history, 1):
            r_num = h.get("round", idx)
            r_loss = h.get("mean_loss", h.get("mean_mse_loss", 0.0))
            gcr_val = h.get("pre_gcr", h.get("gcr", 0.0))
            if gcr_val is not None and gcr_val <= 1.0:
                gcr_val = gcr_val * 100.0

            rows.append({
                "dataset": dataset,
                "method": method,
                "clients": clients,
                "alpha": alpha,
                "round": r_num,
                "total_rounds": total_rounds,
                "round_train_loss": round(float(r_loss) if r_loss is not None else 0.0, 5),
                "gcr_pre": round(float(gcr_val) if gcr_val is not None else 0.0, 2),
                "mean_cosine_pre": round(float(h.get("pre_mean_cosine", h.get("mean_cosine", 1.0))), 4),
                "min_cosine_pre": round(float(h.get("pre_min_cosine", h.get("min_cosine", 1.0))), 4),
                "conflicting_pairs": int(h.get("conflicting_pairs", 0)),
                "total_pairs": int(h.get("total_pairs", 0)),
                "aligned_norm": round(float(h.get("aligned_gradient_norm", h.get("aligned_norm", 0.0))), 5),
                "cmnp_rejection_rate": round(float(h.get("mean_cmnp_rejection_rate", 0.0)), 4),
                "round_time_sec": round(float(h.get("round_time_sec", t_train / max(1, len(history)))), 3),
                "final_auc_roc": metrics["auc_roc"],
                "final_f1_optimal": metrics["f1_optimal"],
                "final_f1_calibrated": metrics["f1_calibrated"],
                "final_far": metrics["far"],
                "latency_ms_per_sample": latency_ms_per_sample,
                "peak_memory_mb": peak_memory_mb,
            })
    else:
        # One-shot model (e.g. LOC_NFST_Bound)
        rows.append({
            "dataset": dataset,
            "method": method,
            "clients": clients,
            "alpha": alpha,
            "round": 1,
            "total_rounds": 1,
            "round_train_loss": 0.0,
            "gcr_pre": 0.0,
            "mean_cosine_pre": 1.0,
            "min_cosine_pre": 1.0,
            "conflicting_pairs": 0,
            "total_pairs": 0,
            "aligned_norm": 0.0,
            "cmnp_rejection_rate": 0.0,
            "round_time_sec": round(t_train, 3),
            "final_auc_roc": metrics["auc_roc"],
            "final_f1_optimal": metrics["f1_optimal"],
            "final_f1_calibrated": metrics["f1_calibrated"],
            "final_far": metrics["far"],
            "latency_ms_per_sample": latency_ms_per_sample,
            "peak_memory_mb": peak_memory_mb,
        })

    df_traj = pd.DataFrame(rows, columns=trajectory_cols)
    df_traj.to_csv(filepath, index=False)


def run_benchmark(args) -> List[Dict[str, Any]]:
    """Main benchmark execution loop."""
    print("=" * 80)
    print("FED-LUNAR BENCHMARK SUITE: NON-IID IOT ANOMALY DETECTION")
    print("=" * 80)
    print(f"Data Directory:     {args.data_dir}")
    print(f"Output Directory:   {args.output_dir}")
    print(f"Number of Clients:  {args.clients}")
    print(f"Dirichlet Alpha:    {args.alpha} (Non-IID skew)")
    print(f"Communication Rds:  {args.rounds}")
    print(f"Scaler:             {args.scaler}")
    print(f"Synthetic Fallback: {args.synthetic_fallback}")
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
    selected_method_arg = args.method if hasattr(args, "method") and args.method else args.models
    if selected_method_arg.lower() == "all":
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
    elif selected_method_arg.lower() == "core":
        model_names = [
            "Proposed_FedLUNAR",
            "Naive_FedLUNAR",
            "FedAutoEncoder",
            "FedProx_LUNAR",
            "PCGrad_FedLUNAR",
            "LOC_NFST_Bound",
        ]
    else:
        model_names = [resolve_model_name(m.strip()) for m in selected_method_arg.split(",") if m.strip()]

    master_summary_cols = [
        "dataset", "method", "clients", "alpha", "rounds",
        "auc_roc", "f1_score", "f1_optimal", "f1_calibrated",
        "far", "detection_rate", "precision",
        "gradient_conflict_ratio", "round_conflict_ratio",
        "convergence_rounds", "latency_ms_per_sample", "latency_single_ms",
        "peak_memory_mb", "train_time_sec", "input_dim"
    ]

    all_results: List[Dict[str, Any]] = []

    for ds_idx, dataset in enumerate(datasets, 1):
        print(f"\n[{ds_idx}/{len(datasets)}] Loading and Partitioning: {dataset} (Dirichlet alpha={args.alpha})...")

        try:
            client_train_data, X_test, y_test, meta = partition_and_prepare_dataset(
                dataset_name=dataset,
                data_dir=args.data_dir,
                num_clients=args.clients,
                alpha=args.alpha,
                scaler=args.scaler,
                max_train_samples=args.max_train_samples,
                max_test_samples=args.max_test_samples,
                test_contamination=0.05,
                allow_synthetic_fallback=args.synthetic_fallback,
                seed=args.seed,
            )
            print(
                f"    Features: {meta['input_dim']} | Train Normal: {meta['total_train_normal']} "
                f"across {meta['num_clients']} clients: {meta['client_sizes']}"
            )
            print(
                f"    Test Total: {meta['test_total']} (Normal: {meta['test_normal_count']}, "
                f"Attack: {meta['test_attack_count']}, Attack Ratio: {meta['test_attack_ratio']:.1%})"
            )
        except Exception as e:
            print(f"    [ERROR] Failed to load dataset {dataset}: {e}")
            import traceback
            traceback.print_exc()
            continue

        for m_idx, m_name in enumerate(model_names, 1):
            print(f"    --> [{m_idx}/{len(model_names)}] Evaluating {m_name}...")
            try:
                # Track memory and execution time
                with MemoryTracker(device=args.device) as mem_tracker:
                    model = instantiate_model(m_name, device=args.device, seed=args.seed)

                    t_start = time.perf_counter()
                    if hasattr(model, "fit"):
                        if m_name == "LOC_NFST_Bound":
                            model.fit(client_train_data)
                        else:
                            model.fit(client_train_data, rounds=args.rounds, verbose=args.verbose)
                    t_train = time.perf_counter() - t_start

                    # Test scoring
                    if hasattr(model, "decision_function"):
                        y_scores = model.decision_function(X_test)
                    elif hasattr(model, "score"):
                        y_scores = model.score(X_test)
                    else:
                        raise AttributeError(f"Model {m_name} has no decision_function or score method")

                peak_mem = mem_tracker.peak_memory_mb

                # Metrics calculation
                metrics = calculate_detection_metrics(y_true=y_test, y_scores=y_scores)

                # Inference latency profiling (both batched and single-sample)
                lat_dict = measure_inference_latency(model, X_test, n_runs=2, measure_single=True)
                lat_ms = lat_dict["latency_ms_per_sample"]
                lat_single_ms = lat_dict["latency_single_ms"]

                # Optimization dynamics
                history = getattr(model, "history", [])
                actual_rounds = args.rounds if m_name != "LOC_NFST_Bound" else 1
                dynamics = calculate_optimization_dynamics(
                    history=history,
                    total_rounds=actual_rounds,
                )

                # Per-run trajectory CSV
                per_run_path = os.path.join(args.output_dir, f"{dataset}_{m_name}.csv")
                write_per_run_trajectory_csv(
                    filepath=per_run_path,
                    dataset=dataset,
                    method=m_name,
                    clients=args.clients,
                    alpha=args.alpha,
                    total_rounds=actual_rounds,
                    model=model,
                    t_train=t_train,
                    metrics=metrics,
                    latency_ms_per_sample=lat_ms,
                    peak_memory_mb=peak_mem,
                )

                # Master summary row (strict 20-column schema)
                summary_row = {
                    "dataset": dataset,
                    "method": m_name,
                    "clients": args.clients,
                    "alpha": args.alpha,
                    "rounds": actual_rounds,
                    "auc_roc": metrics["auc_roc"],
                    "f1_score": metrics["f1_score"],
                    "f1_optimal": metrics["f1_optimal"],
                    "f1_calibrated": metrics["f1_calibrated"],
                    "far": metrics["far"],
                    "detection_rate": metrics["detection_rate"],
                    "precision": metrics["precision"],
                    "gradient_conflict_ratio": dynamics["gradient_conflict_ratio"],
                    "round_conflict_ratio": dynamics["round_conflict_ratio"],
                    "convergence_rounds": dynamics["convergence_rounds"],
                    "latency_ms_per_sample": lat_ms,
                    "latency_single_ms": lat_single_ms,
                    "peak_memory_mb": peak_mem,
                    "train_time_sec": round(t_train, 2),
                    "input_dim": meta["input_dim"],
                }
                all_results.append(summary_row)

                print(
                    f"        [RESULT] AUC-ROC: {metrics['auc_roc']}% | "
                    f"F1-Opt: {metrics['f1_optimal']}% | "
                    f"FAR: {metrics['far']}% | "
                    f"GCR: {dynamics['gradient_conflict_ratio']}% | "
                    f"Latency: {lat_ms:.3f} ms | "
                    f"RAM: {peak_mem:.1f} MB | "
                    f"Train: {t_train:.1f}s"
                )

            except Exception as e:
                print(f"        [ERROR] Failed evaluating {m_name} on {dataset}: {e}")
                import traceback
                traceback.print_exc()

    # Save Master Summary CSV and JSON
    if all_results:
        df_results = pd.DataFrame(all_results, columns=master_summary_cols)
        summary_csv_path = os.path.join(args.output_dir, "benchmark_summary.csv")
        df_results.to_csv(summary_csv_path, index=False)
        print(f"\n[SAVED] Master benchmark summary saved to: {summary_csv_path}")

        json_path = os.path.join(args.output_dir, "benchmark_summary.json")
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"[SAVED] JSON metadata saved to: {json_path}")

        # Print formatted Markdown summary table
        print("\n" + "=" * 80)
        print("AGGREGATED BENCHMARK SUMMARY TABLE")
        print("=" * 80)
        display_cols = [
            "dataset", "method", "auc_roc", "f1_score", "far",
            "gradient_conflict_ratio", "latency_ms_per_sample", "peak_memory_mb", "train_time_sec"
        ]
        try:
            print(df_results[display_cols].to_markdown(index=False))
        except Exception:
            print(df_results[display_cols].to_string(index=False))
        print("=" * 80)

    else:
        print("[WARNING] No benchmark results collected.")

    return all_results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Federated LUNAR Benchmark Suite")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=get_default_data_dir(),
        help="Path to Official_OC_Data CSV directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/lunar_results",
        help="Directory for benchmark outputs",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="Dataset name ('all' or comma-separated list: BoTIoT, EdgeIIoTset, CICIoT2023, N_BaIoT)",
    )
    parser.add_argument(
        "--method",
        "--models",
        dest="method",
        type=str,
        default="all",
        help="Models to benchmark ('all', 'core', or comma-separated names)",
    )
    parser.add_argument(
        "--clients",
        type=int,
        default=3,
        help="Number of federated edge clients (default: 3)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Dirichlet concentration parameter (default: 0.5)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Federated communication rounds (default: 10)",
    )
    parser.add_argument(
        "--synthetic_fallback",
        action="store_true",
        default=True,
        help="Whether to fall back to synthetic data if CSV files are absent (default: True)",
    )
    parser.add_argument(
        "--no_synthetic_fallback",
        dest="synthetic_fallback",
        action="store_false",
        help="Disable synthetic fallback",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=20000,
        help="Max training samples per dataset",
    )
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=10000,
        help="Max test samples per dataset",
    )
    parser.add_argument(
        "--scaler",
        type=str,
        default="StandardScaler",
        help="Feature scaler prefix ('StandardScaler', 'QuantileTransformer')",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose client training progress",
    )
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    # Support backward compatibility if called with models attribute
    if not hasattr(args, "models"):
        args.models = args.method
    run_benchmark(args)
