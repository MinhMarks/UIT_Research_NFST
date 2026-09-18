"""
FL-LOC-NFST Simulation Runner
==============================
Main entry point for Federated LOC-NFST experiments using Flower simulation.

Usage (local test):
    python run_fl_simulation.py --dataset data_CICIoT2023 --scaler StandardScaler

Usage (server — override dataset path):
    DATA_DIR=/path/to/data python run_fl_simulation.py \
        --dataset data_CICIoT2023 --scaler StandardScaler --clients 3 --clusters 5

Output:
    notebooks/experiments/outputs/federated_results/
        fl_<dataset>_<scaler>_<timestamp>.csv    (per-dataset results)
        fl_comparison_<timestamp>.csv             (FL vs Centralized summary)
        fl_<dataset>_<scaler>_<timestamp>.log
"""
import os
import sys
import time
import json
import logging
import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── Ensure parent experiments folder is on path ──────────────────────────────
_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR.parent))

import flwr as fl

from fed_loc_nfst.config import (
    DATA_DIR, OUTPUT_DIR, NUM_CLIENTS, NUM_ROUNDS, K_CLUSTERS,
    DIRICHLET_ALPHA, NOISE_PCT, SEED
)
from fed_loc_nfst.data_utils import (
    load_dataset, partition_iid, partition_dirichlet, shard_test_data
)
from fed_loc_nfst.client import create_client_fn
from fed_loc_nfst.strategy import FedLOCStrategy

warnings.filterwarnings('ignore')


# ============================================================
# Logging Setup
# ============================================================

def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("fl_loc_nfst")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


# ============================================================
# Communication Cost Tracking
# ============================================================

def compute_payload_stats(d: int, K: int, M: int) -> dict:
    """
    Compute expected communication cost (client → server only, Protocol A).
    Broadcast (server → client) is counted separately.

    Per client upload:
      S_w: d×d×4B, centroids: K×d×4B, counts: K×4B
    Server broadcast:
      W: d×L×4B, null_centers: K×L×4B, [max_train]: 4B
    """
    upload_per_client_bytes = (d * d + K * d + K) * 4
    upload_total_bytes = upload_per_client_bytes * M
    # L estimated as min(d - rank_Sw, rank_Pt) — typically 1-10
    L_estimate = max(1, int(d * 0.1))
    broadcast_bytes = (d * L_estimate + K * L_estimate + 1) * 4
    return {
        "upload_per_client_KB": round(upload_per_client_bytes / 1024, 2),
        "upload_total_KB": round(upload_total_bytes / 1024, 2),
        "broadcast_server_KB": round(broadcast_bytes / 1024, 2),
        "total_comm_KB": round((upload_total_bytes + broadcast_bytes) / 1024, 2),
    }


# ============================================================
# FL Experiment Runner
# ============================================================

def run_fl_experiment(
    dataset_prefix: str,
    scaler: str,
    num_clients: int,
    num_rounds: int,
    K: int,
    partition_type: str = 'iid',
    noise_pct: float = NOISE_PCT,
    logger: logging.Logger = None,
) -> dict:
    """
    Run one FL experiment configuration.

    Returns result dict with FL metrics and communication cost.
    """
    log = logger or logging.getLogger("fl_loc_nfst")

    train_path = os.path.join(DATA_DIR, f"Train_{scaler}_{dataset_prefix}.csv")
    test_path  = os.path.join(DATA_DIR, f"Test_{scaler}_{dataset_prefix}.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        log.warning(f"[SKIP] Files not found: {train_path}")
        return {"error": f"Files not found: {train_path}"}

    log.info("=" * 60)
    log.info(f"FL Experiment: {dataset_prefix} | scaler={scaler}")
    log.info(f"Clients={num_clients}, Rounds={num_rounds}, K={K}, "
             f"Partition={partition_type}, Noise={noise_pct}%")
    log.info("=" * 60)

    # --- Load data ---
    np.random.seed(SEED)
    X_train, y_train, X_test, y_test = load_dataset(train_path, test_path, noise_pct)
    d = X_train.shape[1]
    N_total = len(X_train)

    log.info(f"Dataset: N_train={N_total}, N_test={len(X_test)}, d={d}")

    # --- Partition training data ---
    if partition_type == 'iid':
        partitions = partition_iid(X_train, num_clients, seed=SEED)
    else:
        partitions = partition_dirichlet(X_train, X_test, y_test, num_clients,
                                          alpha=DIRICHLET_ALPHA, seed=SEED)

    # --- Shard test data for per-client evaluation ---
    test_shards = shard_test_data(X_test, y_test, num_clients, seed=SEED)

    # --- Create Flower strategy ---
    strategy = FedLOCStrategy(
        num_clusters=K,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
    )

    # --- Create client factory ---
    client_fn = create_client_fn(partitions, test_shards, K=K)

    # --- Run Flower simulation ---
    log.info(f"Starting Flower simulation ({num_rounds} round(s))...")
    t_fl_start = time.time()

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )

    fl_time = time.time() - t_fl_start
    log.info(f"FL simulation done in {fl_time:.2f}s")

    # --- Extract aggregated metrics ---
    fl_metrics = {}
    if history.metrics_distributed_fit:
        last_round_fit = dict(history.metrics_distributed_fit)
        for key, vals in last_round_fit.items():
            if vals:
                fl_metrics[f"fit_{key}"] = vals[-1][1]

    if history.metrics_distributed:
        last_round_eval = dict(history.metrics_distributed)
        for key, vals in last_round_eval.items():
            if vals:
                fl_metrics[key] = vals[-1][1]

    # Communication cost
    comm_stats = compute_payload_stats(d, K, num_clients)
    log.info(f"Communication: {comm_stats}")

    result = {
        "mode": "federated",
        "dataset": dataset_prefix,
        "scaler": scaler,
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "K_clusters": K,
        "partition_type": partition_type,
        "noise_pct": noise_pct,
        "N_train_total": N_total,
        "N_test": len(X_test),
        "d_features": d,
        "fl_time_s": round(fl_time, 3),
        "aggregation_time_s": round(strategy.aggregation_time, 3),
        "L_null_dim": strategy.L,
        "K_global": strategy.K_global,
        **{k: v for k, v in fl_metrics.items()},
        **{f"comm_{k}": v for k, v in comm_stats.items()},
    }

    # Extract per-client metrics from strategy.fit_metrics
    for i, cm in enumerate(strategy.fit_metrics):
        result[f"client_{i}_train_time_s"] = cm.get("train_time_s", None)
        result[f"client_{i}_payload_kb"] = cm.get("payload_kb", None)

    log.info(f"Result: FL_F1={fl_metrics.get('FL_F1_weighted', 'N/A')}, "
             f"FL_AUC={fl_metrics.get('FL_AUCROC_weighted', 'N/A')}")

    return result


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="FL-LOC-NFST Simulation Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--dataset', type=str, default='data_CICIoT2023',
                        help='Dataset prefix (e.g. data_CICIoT2023, data_ToNIoT)')
    parser.add_argument('--scaler', type=str, default='StandardScaler',
                        help='Scaler name used in filename')
    parser.add_argument('--clients', type=int, default=NUM_CLIENTS,
                        help='Number of simulated FL clients')
    parser.add_argument('--rounds', type=int, default=NUM_ROUNDS,
                        help='Number of FL rounds (Protocol A = 1)')
    parser.add_argument('--clusters', type=int, default=K_CLUSTERS,
                        help='K pseudo-classes per client')
    parser.add_argument('--partition', type=str, default='iid',
                        choices=['iid', 'dirichlet'],
                        help='Data partition strategy')
    parser.add_argument('--noise', type=float, default=NOISE_PCT,
                        help='Noise injection percentage')
    parser.add_argument('--all-datasets', action='store_true',
                        help='Run all configured datasets and scalers')
    args = parser.parse_args()

    # Setup output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.all_datasets:
        from fed_loc_nfst.config import DATASETS, SCALERS
        datasets = DATASETS
        scalers = SCALERS
    else:
        datasets = [args.dataset]
        scalers = [args.scaler]

    all_results = []

    for dataset in datasets:
        log_path = os.path.join(OUTPUT_DIR, f"fl_{dataset}_{timestamp}.log")
        logger = setup_logger(log_path)

        for scaler in scalers:
            result = run_fl_experiment(
                dataset_prefix=dataset,
                scaler=scaler,
                num_clients=args.clients,
                num_rounds=args.rounds,
                K=args.clusters,
                partition_type=args.partition,
                noise_pct=args.noise,
                logger=logger,
            )
            if "error" not in result:
                all_results.append(result)

                # Save incrementally
                out_csv = os.path.join(OUTPUT_DIR, f"fl_{dataset}_{scaler}_{timestamp}.csv")
                pd.DataFrame([result]).to_csv(out_csv, index=False)
                logger.info(f"Result saved: {out_csv}")

    # Save combined results
    if all_results:
        combined_csv = os.path.join(OUTPUT_DIR, f"fl_all_results_{timestamp}.csv")
        pd.DataFrame(all_results).to_csv(combined_csv, index=False)
        print(f"\n✓ All FL results saved to: {combined_csv}")

    print("\nDone.")


if __name__ == "__main__":
    main()
