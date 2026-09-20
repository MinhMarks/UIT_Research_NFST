"""
run_drift_injection.py
======================
Canonical Drift Injection Protocol for ADYN-LOC-NFST (Rule 21.3)
Implements D1-Abrupt, D2-Gradual, D3-Recurring scenarios.

Usage:
    python run_drift_injection.py --dataset N_BaIoT --scenario D1

Output: outputs/drift_results/drift_{dataset}_{scenario}_{scaler}.csv
"""
import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parent.parent
SRC_DIR     = SCRIPT_DIR / "fed_loc_nfst"

_DEFAULT_DATA = Path("/home/jupyter-iec_duongnt/New Project/NFST/GMM-nfst/Datascaled/Official_OC_Data")
if not _DEFAULT_DATA.exists():
    _DEFAULT_DATA = REPO_ROOT / "Datascaled" / "Official_OC_Data"
DATA_DIR = Path(os.environ.get("DATA_DIR", _DEFAULT_DATA))

OUTPUT_DIR  = SCRIPT_DIR / "outputs" / "drift_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from fed_loc_nfst.adyn_loc_nfst import AdynLOCNFST
from fed_loc_nfst.data_utils import load_dataset as fl_load_dataset
from sklearn.metrics import roc_auc_score

# ─── Constants (Rule 21.2 & 21.3) ─────────────────────────────────────────────
RANDOM_SEED       = 42
CONTAMINATION_MAX = 0.05   # ≤ 5% contamination in test set
N_SEEDS           = 10     # 10 independent runs (Rule 21.5.1)

DATASET_FILES = {
    "N_BaIoT":     "data_N_BaIoT",
    "BoTIoT":      "data_BoTIoT",
    "EdgeIIoTset": "data_EdgeIIoTset",
    "CICIoT2023":  "data_CICIoT2023",
    "IoTID20":     "data_IoTID20",
    "ToNIoT":      "data_ToNIoT",
}

SCALERS = [
    "StandardScaler",
    "MinMaxScaler",
    "Normalizer",
    "RobustScaler",
    "QuantileTransformer",
]


# ─── Data Loading ─────────────────────────────────────────────────────────────
def load_dataset(dataset_name: str, scaler_name: str):
    """Load pre-scaled train (normal) and test (normal+attack) splits."""
    prefix = DATASET_FILES.get(dataset_name, dataset_name)
    train_path = DATA_DIR / f"Train_{scaler_name}_{prefix}.csv"
    test_path  = DATA_DIR / f"Test_{scaler_name}_{prefix}.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    X_train, y_train, X_test, y_test = fl_load_dataset(
        str(train_path), str(test_path), noise_pct=0.0
    )
    return X_train, X_test, y_test



def enforce_contamination(X_test, y_test, max_rate=CONTAMINATION_MAX, seed=RANDOM_SEED):
    """Subsample attack samples to enforce contamination ≤ max_rate (Rule 21.2)."""
    rng = np.random.default_rng(seed)
    n_normal = (y_test == 0).sum()
    n_attack_max = int(n_normal * max_rate / (1 - max_rate))
    attack_idx  = np.where(y_test == 1)[0]
    normal_idx  = np.where(y_test == 0)[0]

    if len(attack_idx) > n_attack_max:
        attack_idx = rng.choice(attack_idx, size=n_attack_max, replace=False)

    idx = np.sort(np.concatenate([normal_idx, attack_idx]))
    return X_test[idx], y_test[idx]


# ─── Drift Injection ──────────────────────────────────────────────────────────
def inject_abrupt_drift(X_test: np.ndarray, drift_fraction: float = 0.5, seed: int = RANDOM_SEED):
    """
    D1 — Abrupt Drift (Rule 21.3):
    At t_drift = drift_fraction * N_test, swap the distribution by
    reversing the feature sign pattern (equivalent to Device A → Device B switch).
    transition_width = 1 sample.
    """
    rng = np.random.default_rng(seed)
    N = len(X_test)
    drift_point = int(N * drift_fraction)
    X_drifted = X_test.copy()
    # Simulate Device B: permute features in post-drift region
    perm = rng.permutation(X_test.shape[1])
    X_drifted[drift_point:] = X_test[drift_point:][:, perm]
    drift_info = {"type": "abrupt", "drift_point": drift_point, "transition_width": 1}
    return X_drifted, drift_info


def inject_gradual_drift(X_test: np.ndarray, t_start: float = 0.30, t_end: float = 0.70, seed: int = RANDOM_SEED):
    """
    D2 — Gradual Drift (Rule 21.3):
    x_blend(t) = (1 - alpha(t)) * x_old + alpha(t) * x_new
    alpha(t) = (t - t_start) / (t_end - t_start)  in [t_start, t_end]
    """
    rng = np.random.default_rng(seed)
    N = len(X_test)
    i_start = int(N * t_start)
    i_end   = int(N * t_end)
    perm = rng.permutation(X_test.shape[1])
    X_new = X_test[:, perm]   # "Device B" distribution

    X_drifted = X_test.copy()
    for i in range(i_start, i_end):
        alpha = (i - i_start) / (i_end - i_start)
        X_drifted[i] = (1 - alpha) * X_test[i] + alpha * X_new[i]
    X_drifted[i_end:] = X_new[i_end:]

    drift_info = {
        "type": "gradual",
        "drift_point": i_start,
        "transition_width": i_end - i_start,
    }
    return X_drifted, drift_info


def inject_recurring_drift(X_test: np.ndarray, n_cycles: int = 5, T_cycle: int = 1000, seed: int = RANDOM_SEED):
    """
    D3 — Recurring Drift (Rule 21.3):
    Alternates between Device A and Device B every T_cycle samples.
    n_cycles = 5 full cycles.
    """
    rng = np.random.default_rng(seed)
    N = len(X_test)
    perm = rng.permutation(X_test.shape[1])
    X_new = X_test[:, perm]

    X_drifted = X_test.copy()
    for i in range(N):
        cycle_idx = i // T_cycle
        if cycle_idx % 2 == 1:          # odd cycles → Device B
            X_drifted[i] = X_new[i]

    drift_info = {
        "type": "recurring",
        "n_cycles": n_cycles,
        "T_cycle": T_cycle,
    }
    return X_drifted, drift_info


DRIFT_INJECTORS = {
    "D1": inject_abrupt_drift,
    "D2": inject_gradual_drift,
    "D3": inject_recurring_drift,
}


# ─── Metric Helpers ───────────────────────────────────────────────────────────
def compute_recovery_time(scores, y_true, drift_point: int, auc_threshold: float = 0.90, window: int = 200):
    """
    D1 metric: Recovery Time (RT) — number of samples after drift_point
    until AUC-ROC ≥ auc_threshold over a rolling window.
    """
    for offset in range(0, len(scores) - drift_point - window):
        start = drift_point + offset
        end   = start + window
        if end > len(scores):
            break
        try:
            auc = roc_auc_score(y_true[start:end], scores[start:end])
            if auc >= auc_threshold:
                return offset
        except ValueError:
            continue
    return len(scores) - drift_point   # never recovered


def compute_audc(scores, y_true, i_start: int, i_end: int):
    """
    D2 metric: Area Under the Degradation Curve (AUDC).
    Integral of AUC-ROC over the drift window [i_start, i_end].
    """
    window = 100
    auc_vals = []
    for i in range(i_start, i_end - window, window // 2):
        try:
            auc = roc_auc_score(y_true[i:i + window], scores[i:i + window])
            auc_vals.append(auc)
        except ValueError:
            continue
    return float(np.mean(auc_vals)) if auc_vals else 0.0


def compute_stability_score(scores, y_true, T_cycle: int):
    """
    D3 metric: Stability Score = std of AUC-ROC per cycle.
    """
    N = len(scores)
    auc_per_cycle = []
    for start in range(0, N - T_cycle, T_cycle):
        try:
            auc = roc_auc_score(y_true[start:start + T_cycle], scores[start:start + T_cycle])
            auc_per_cycle.append(auc)
        except ValueError:
            continue
    return float(np.std(auc_per_cycle)) if len(auc_per_cycle) > 1 else 0.0


# ─── Single Run ───────────────────────────────────────────────────────────────
def run_single(dataset_name: str, scenario: str, scaler_name: str, seed: int) -> dict:
    """Run one (dataset, scenario, scaler, seed) experiment."""
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # 1. Load pre-scaled data
    X_train, X_test, y_test = load_dataset(dataset_name, scaler_name)

    # 2. Enforce contamination
    X_test, y_test = enforce_contamination(X_test, y_test, seed=seed)

    # 3. Inject drift into test stream
    injector = DRIFT_INJECTORS[scenario]
    X_test_drifted, drift_info = injector(X_test, seed=seed)

    # 4. Train ADYN model on clean train data
    model = AdynLOCNFST(
        n_clusters=20,
        max_clusters=128,
        n_chunks=5,
        random_state=seed,
    )
    model.fit(X_train)

    # 5. Score drifted test stream
    t0 = time.perf_counter()
    scores = model.score_samples(X_test_drifted)
    latency_ms = (time.perf_counter() - t0) / len(X_test_drifted) * 1000.0

    # Invert: higher score = more anomalous
    if scores.mean() > 0:
        scores = -scores

    # 6. Overall AUC-ROC
    try:
        auc_overall = roc_auc_score(y_test, scores)
    except ValueError:
        auc_overall = float("nan")

    # 7. Scenario-specific metrics
    extra = {}
    if scenario == "D1":
        dp = drift_info["drift_point"]
        extra["recovery_time"] = compute_recovery_time(scores, y_test, dp)
    elif scenario == "D2":
        i_start = drift_info["drift_point"]
        i_end   = i_start + drift_info["transition_width"]
        extra["audc"] = compute_audc(scores, y_test, i_start, i_end)
    elif scenario == "D3":
        extra["stability_score_std"] = compute_stability_score(scores, y_test, drift_info["T_cycle"])

    return {
        "dataset": dataset_name,
        "scenario": scenario,
        "scaler": scaler_name,
        "seed": seed,
        "auc_roc": auc_overall,
        "latency_ms_per_sample": latency_ms,
        **drift_info,
        **extra,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Drift Injection Benchmark — Rule 21.3")
    parser.add_argument("--dataset", choices=list(DATASET_FILES.keys()) + ["all"], default="all")
    parser.add_argument("--scenario", choices=["D1", "D2", "D3", "all"], default="all")
    parser.add_argument("--scaler", choices=SCALERS, default="StandardScaler")
    args = parser.parse_args()


    datasets  = list(DATASET_FILES.keys()) if args.dataset == "all" else [args.dataset]
    scenarios = ["D1", "D2", "D3"] if args.scenario == "all" else [args.scenario]

    all_results = []
    total = len(datasets) * len(scenarios) * N_SEEDS
    done  = 0

    for dataset in datasets:
        for scenario in scenarios:
            seed_results = []
            for seed in range(N_SEEDS):
                done += 1
                print(f"[{done}/{total}] {dataset} | {scenario} | {args.scaler} | seed={seed}")
                try:
                    r = run_single(dataset, scenario, args.scaler, seed)
                    seed_results.append(r)
                except Exception as e:
                    print(f"  ERROR: {e}")

            # Aggregate mean ± std across seeds (Rule 21.5.1)
            if seed_results:
                df_seed = pd.DataFrame(seed_results)
                agg = {
                    "dataset": dataset,
                    "scenario": scenario,
                    "scaler": args.scaler,
                    "n_seeds": len(seed_results),
                    "auc_roc_mean": df_seed["auc_roc"].mean(),
                    "auc_roc_std":  df_seed["auc_roc"].std(),
                    "latency_ms_mean": df_seed["latency_ms_per_sample"].mean(),
                }
                # scenario-specific
                for col in ["recovery_time", "audc", "stability_score_std"]:
                    if col in df_seed.columns:
                        agg[f"{col}_mean"] = df_seed[col].mean()
                        agg[f"{col}_std"]  = df_seed[col].std()

                all_results.append(agg)
                print(f"  → AUC = {agg['auc_roc_mean']:.4f} ± {agg['auc_roc_std']:.4f}")

    # Save
    if all_results:
        out_df = pd.DataFrame(all_results)
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        out_path = OUTPUT_DIR / f"drift_{args.scaler}_{ts}.csv"
        out_df.to_csv(out_path, index=False)
        print(f"\n✓ Saved: {out_path}")
        print(out_df[["dataset", "scenario", "auc_roc_mean", "auc_roc_std"]].to_string(index=False))


if __name__ == "__main__":
    main()
