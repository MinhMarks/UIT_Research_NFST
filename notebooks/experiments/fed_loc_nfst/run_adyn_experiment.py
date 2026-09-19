"""
ADYN-LOC-NFST: Full Benchmark Runner
======================================
Runs Static-K vs ADYN-LOC-NFST across ALL datasets × ALL scalers.
Designed to run on server: postmaster.iec

Usage (on server):
    cd '/home/jupyter-iec_duongnt/New Project/NFST/GMM-nfst/notebooks/experiments'
    PYTHONPATH='.' python3 fed_loc_nfst/run_adyn_experiment.py

    # Single dataset test:
    PYTHONPATH='.' python3 fed_loc_nfst/run_adyn_experiment.py \\
        --dataset data_CICIoT2023 --scaler Normalizer --K-init 20

    # Full benchmark (all 6 datasets × 5 scalers):
    PYTHONPATH='.' python3 fed_loc_nfst/run_adyn_experiment.py --all

Output:
    outputs/adyn_results/adyn_benchmark_YYYYMMDD_HHMMSS.csv
    outputs/adyn_results/adyn_benchmark_YYYYMMDD_HHMMSS.log
"""
import os
import sys
import time
import logging
import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ── Path setup ────────────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).parent
_EXPERIMENTS_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_EXPERIMENTS_DIR))

from fed_loc_nfst.config import (
    DATA_DIR, OUTPUT_DIR, NOISE_PCT, SEED
)
from fed_loc_nfst.data_utils import load_dataset
from fed_loc_nfst.adyn_loc_nfst import run_adyn_vs_static

warnings.filterwarnings('ignore')

# ── All datasets and scalers ─────────────────────────────────────────────────
ALL_DATASETS = [
    'data_CICIoT2023',
    'data_ToNIoT',
    'data_BoTIoT',
    'data_N_BaIoT',
    'data_EdgeIIoTset',
    'data_IoTID20',
]

ALL_SCALERS = [
    'MinMaxScaler',
    'StandardScaler',
    'RobustScaler',
    'QuantileTransformer',
    'Normalizer',
]

# Default K values to compare (Static-K baselines)
K_STATIC_LIST = [20, 50, 100]
K_INIT_ADYN = 20
N_CHUNKS = 5   # Temporal concept drift simulation chunks


# ============================================================
# Setup Logging
# ============================================================

def setup_logging(out_dir: str, timestamp: str) -> logging.Logger:
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f"adyn_benchmark_{timestamp}.log")

    logger = logging.getLogger("adyn_experiment")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# ============================================================
# Main Experiment Loop
# ============================================================

def run_experiment(
    datasets: list,
    scalers: list,
    K_static_list: list,
    K_init_adyn: int,
    n_chunks: int,
    out_dir: str,
    timestamp: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Run full experiment matrix and save results to CSV.
    """
    all_results = []
    total = len(datasets) * len(scalers)
    done = 0

    for dataset in datasets:
        for scaler in scalers:
            done += 1
            logger.info(f"\n{'='*70}")
            logger.info(f"[{done}/{total}] Dataset={dataset} | Scaler={scaler}")
            logger.info(f"{'='*70}")

            train_path = os.path.join(DATA_DIR, f"Train_{scaler}_{dataset}.csv")
            test_path  = os.path.join(DATA_DIR, f"Test_{scaler}_{dataset}.csv")

            if not os.path.exists(train_path) or not os.path.exists(test_path):
                logger.warning(f"[SKIP] Files not found: {train_path}")
                continue

            try:
                X_train, y_train, X_test, y_test = load_dataset(
                    train_path, test_path, noise_pct=NOISE_PCT
                )
                logger.info(
                    f"Loaded: N_train={len(X_train)}, N_test={len(X_test)}, "
                    f"d={X_train.shape[1]}, "
                    f"anomaly_rate={y_test.mean()*100:.1f}%"
                )

            except Exception as e:
                logger.error(f"[ERROR] Load failed for {dataset}/{scaler}: {e}")
                continue

            t_start = time.time()
            try:
                results = run_adyn_vs_static(
                    X_train=X_train,
                    X_test=X_test,
                    y_test=y_test,
                    K_static_list=K_static_list,
                    K_init_adyn=K_init_adyn,
                    n_chunks=n_chunks,
                    seed=SEED,
                    dataset_name=dataset,
                    scaler_name=scaler,
                )
                all_results.extend(results)

            except Exception as e:
                logger.error(f"[ERROR] Experiment failed for {dataset}/{scaler}: {e}",
                             exc_info=True)
                continue

            elapsed = time.time() - t_start
            logger.info(f"[{done}/{total}] Done: {dataset}/{scaler} in {elapsed:.1f}s")

            # Save intermediate results after each dataset/scaler pair
            if all_results:
                interim_df = pd.DataFrame(all_results)
                interim_path = os.path.join(
                    out_dir, f"adyn_benchmark_{timestamp}_interim.csv"
                )
                interim_df.to_csv(interim_path, index=False)

    return pd.DataFrame(all_results)


# ============================================================
# Results Summary
# ============================================================

def print_summary(df: pd.DataFrame, logger: logging.Logger) -> None:
    """Print comparison table: ADYN vs best Static-K."""
    if df.empty:
        logger.info("No results to summarize.")
        return

    logger.info("\n" + "="*80)
    logger.info("SUMMARY: ADYN-LOC-NFST vs Static-K LOC-NFST")
    logger.info("="*80)

    # Pivot: for each (dataset, scaler), compare methods
    for (ds, sc), grp in df.groupby(["dataset", "scaler"]):
        static_rows = grp[grp["method"].str.startswith("Static")]
        adyn_rows   = grp[grp["method"].str.startswith("ADYN")]

        if static_rows.empty or adyn_rows.empty:
            continue

        best_static_auc = static_rows["AUCROC"].max()
        best_static_row = static_rows.loc[static_rows["AUCROC"].idxmax()]
        adyn_row = adyn_rows.iloc[0]

        delta = adyn_row["AUCROC"] - best_static_auc
        sign = "▲" if delta >= 0 else "▼"

        logger.info(
            f"{ds[:20]:<22} | {sc:<20} | "
            f"Static-best({best_static_row['method'].split('(')[1][:-1]})={best_static_auc:.2f}% | "
            f"ADYN={adyn_row['AUCROC']:.2f}% | "
            f"{sign}{abs(delta):.2f}% | "
            f"K: {adyn_row['K_init']}→{adyn_row['K_final']} | "
            f"L={adyn_row['L']}"
        )

    # Overall average
    if "AUCROC" in df.columns:
        logger.info("\n--- Overall AUC by Method ---")
        for method, grp in df.groupby("method"):
            logger.info(
                f"  {method:<40}: AUC={grp['AUCROC'].mean():.2f}% ± "
                f"{grp['AUCROC'].std():.2f}%"
            )


# ============================================================
# Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="ADYN-LOC-NFST vs Static-K Benchmark"
    )
    parser.add_argument('--dataset', type=str, default=None,
                        help='Single dataset prefix (default: all)')
    parser.add_argument('--scaler', type=str, default=None,
                        help='Single scaler (default: all)')
    parser.add_argument('--all', action='store_true',
                        help='Run all datasets and scalers')
    parser.add_argument('--K-init', type=int, default=K_INIT_ADYN,
                        help='Initial K for ADYN (default: 20)')
    parser.add_argument('--n-chunks', type=int, default=N_CHUNKS,
                        help='Number of temporal chunks (default: 5)')
    parser.add_argument('--K-static', type=int, nargs='+', default=K_STATIC_LIST,
                        help='Static K values to compare (default: 20 50 100)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: outputs/adyn_results)')
    args = parser.parse_args()

    # Setup output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or os.path.join(
        str(_EXPERIMENTS_DIR), "outputs", "adyn_results"
    )
    logger = setup_logging(out_dir, timestamp)

    # Dataset/scaler selection
    if args.all or (args.dataset is None and args.scaler is None):
        datasets = ALL_DATASETS
        scalers = ALL_SCALERS
    else:
        datasets = [args.dataset] if args.dataset else ALL_DATASETS
        scalers  = [args.scaler]  if args.scaler  else ALL_SCALERS

    logger.info(f"ADYN-LOC-NFST Experiment")
    logger.info(f"Datasets : {datasets}")
    logger.info(f"Scalers  : {scalers}")
    logger.info(f"K_static : {args.K_static}")
    logger.info(f"K_init   : {args.K_init}")
    logger.info(f"n_chunks : {args.n_chunks}")
    logger.info(f"DATA_DIR : {DATA_DIR}")
    logger.info(f"OUTPUT   : {out_dir}")

    total_start = time.time()
    df = run_experiment(
        datasets=datasets,
        scalers=scalers,
        K_static_list=args.K_static,
        K_init_adyn=args.K_init,
        n_chunks=args.n_chunks,
        out_dir=out_dir,
        timestamp=timestamp,
        logger=logger,
    )

    total_elapsed = time.time() - total_start

    if not df.empty:
        # Save final results
        out_csv = os.path.join(out_dir, f"adyn_benchmark_{timestamp}.csv")
        df.to_csv(out_csv, index=False)
        logger.info(f"\n✓ Final results saved to: {out_csv}")
        logger.info(f"✓ Total experiments: {len(df)}")
        logger.info(f"✓ Total time: {total_elapsed:.1f}s")

        print_summary(df, logger)

        # Print final table to stdout
        print("\n" + df.to_string(index=False))
    else:
        logger.warning("No results produced!")

    logger.info("Done.")


if __name__ == "__main__":
    main()
