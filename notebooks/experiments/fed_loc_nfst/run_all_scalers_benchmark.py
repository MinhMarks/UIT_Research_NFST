"""
Comprehensive Scaler & Dataset Benchmark for FL-LOC-NFST vs Centralized LOC-NFST
==================================================================================
Runs across all 5 scalers:
  - MinMaxScaler
  - StandardScaler
  - RobustScaler
  - QuantileTransformer
  - Normalizer

Across all major IoT benchmark datasets:
  - data_CICIoT2023
  - data_ToNIoT
  - data_BoTIoT
  - data_N_BaIoT

Compares Centralized vs One-Shot Federated (M=3 clients).
Saves results to CSV and generates comparison summary.
"""
import os
import sys
import time
import argparse
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR.parent))

from fed_loc_nfst.data_utils import load_dataset, partition_iid
from fed_loc_nfst.client import compute_local_scatter
from fed_loc_nfst.strategy import aggregate_scatter_matrices, adaptive_spectral_solve, compute_null_centers
from fed_loc_nfst.evaluate import compute_fl_scores, evaluate_scores
from fed_loc_nfst.run_centralized import compute_centralized_NPD

warnings.filterwarnings('ignore')


def run_benchmark(data_dir: str, output_csv: str, datasets=None, scalers=None, K: int = 15, M: int = 3, noise_pct: float = 0.0):
    if datasets is None:
        datasets = ['data_CICIoT2023', 'data_ToNIoT', 'data_BoTIoT', 'data_N_BaIoT']
    if scalers is None:
        scalers = ['MinMaxScaler', 'StandardScaler', 'RobustScaler', 'QuantileTransformer', 'Normalizer']

    results = []
    print("=" * 80)
    print(f"Starting Benchmark: Datasets={datasets}")
    print(f"Scalers={scalers} | K={K} | Clients={M} | Noise={noise_pct}%")
    print("=" * 80)

    for ds in datasets:
        for sc in scalers:
            train_file = os.path.join(data_dir, f"Train_{sc}_{ds}.csv")
            test_file  = os.path.join(data_dir, f"Test_{sc}_{ds}.csv")

            if not os.path.exists(train_file) or not os.path.exists(test_file):
                print(f"[SKIP] Missing: {train_file}")
                continue

            print(f"\n>>> Running: {ds} | {sc} ...")
            try:
                t0 = time.time()
                X_train, y_train, X_test, y_test = load_dataset(train_file, test_file, noise_pct=noise_pct)
                load_time = time.time() - t0

                # 1. Centralized
                t_c_train = time.time()
                W_c, centers_c, max_c, L_c, _ = compute_centralized_NPD(X_train, K=K)
                c_train_time = time.time() - t_c_train

                t_c_test = time.time()
                y_proba_c = compute_fl_scores(X_test, W_c, centers_c, max_c)
                c_test_time = time.time() - t_c_test
                metrics_c = evaluate_scores(y_test, y_proba_c)

                # 2. Federated
                t_fl_start = time.time()
                partitions = partition_iid(X_train, num_clients=M, seed=42)
                S_list, cent_list, cnt_list = [], [], []
                comm_bytes = 0

                for i, X_m in enumerate(partitions):
                    S_w_m, c_m, cnt_m, _, _ = compute_local_scatter(X_m, K=K)
                    S_list.append(S_w_m)
                    cent_list.append(c_m)
                    cnt_list.append(cnt_m)
                    comm_bytes += (S_w_m.nbytes + c_m.nbytes + cnt_m.nbytes)

                S_w_g, S_t_g, anchors, N_per = aggregate_scatter_matrices(S_list, cent_list, cnt_list, K_global=K)
                W_fl, L_fl = adaptive_spectral_solve(S_w_g, S_t_g)
                null_c, max_t = compute_null_centers(anchors, W_fl)
                fl_train_time = time.time() - t_fl_start

                t_fl_test = time.time()
                y_proba_fl = compute_fl_scores(X_test, W_fl, null_c, max_t)
                fl_test_time = time.time() - t_fl_test
                metrics_fl = evaluate_scores(y_test, y_proba_fl)

                # Delta metrics
                delta_auc = round(metrics_fl["AUCROC"] - metrics_c["AUCROC"], 4)
                delta_f1  = round(metrics_fl["F1 Score"] - metrics_c["F1 Score"], 4)

                row = {
                    "Dataset": ds.replace("data_", ""),
                    "Scaler": sc,
                    "N_Train": len(X_train),
                    "N_Test": len(X_test),
                    "d_Features": X_train.shape[1],
                    "K": K,
                    "Cent_AUCROC": metrics_c["AUCROC"],
                    "Cent_F1": metrics_c["F1 Score"],
                    "Cent_Acc": metrics_c["Accuracy"],
                    "Cent_L": L_c,
                    "FL_AUCROC": metrics_fl["AUCROC"],
                    "FL_F1": metrics_fl["F1 Score"],
                    "FL_Acc": metrics_fl["Accuracy"],
                    "FL_L": L_fl,
                    "Delta_AUCROC": delta_auc,
                    "Delta_F1": delta_f1,
                    "Cent_Train_s": round(c_train_time, 3),
                    "FL_Train_s": round(fl_train_time, 3),
                    "Comm_Upload_KB": round(comm_bytes / 1024, 2),
                }
                results.append(row)

                print(f"    [Cent] AUC={metrics_c['AUCROC']:6.2f}% | F1={metrics_c['F1 Score']:6.4f} | L={L_c}")
                print(f"    [FL  ] AUC={metrics_fl['AUCROC']:6.2f}% | F1={metrics_fl['F1 Score']:6.4f} | L={L_fl}")
                print(f"    [Diff] ΔAUC={delta_auc:+6.2f}% | ΔF1={delta_f1:+6.4f}")

                # Save intermediate
                pd.DataFrame(results).to_csv(output_csv, index=False)

            except Exception as e:
                print(f"    [ERROR] {ds} - {sc}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETED. Summary table:")
    print("=" * 80)
    print(df[["Dataset", "Scaler", "Cent_AUCROC", "FL_AUCROC", "Cent_F1", "FL_F1", "Delta_AUCROC", "Delta_F1"]].to_string(index=False))
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="benchmark_scalers_results.csv")
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--noise", type=float, default=0.0)
    args = parser.parse_args()

    run_benchmark(data_dir=args.data_dir, output_csv=args.output, K=args.k, M=args.clients, noise_pct=args.noise)
