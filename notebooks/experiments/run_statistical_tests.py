"""
run_statistical_tests.py
========================
Statistical Testing Protocol for ADYN-LOC-NFST (Rule 21.5)
Implements:
  - Wilcoxon Signed-Rank Test (pairwise, Rule 21.5.2)
  - Friedman + Iman-Davenport Test (omnibus, Rule 21.5.3)
  - Nemenyi / Bonferroni-Dunn Post-hoc (Rule 21.5.3)
  - Critical Difference (CD) Diagram generation

Usage:
    python run_statistical_tests.py --results_csv outputs/adyn_results/adyn_benchmark_*.csv
    python run_statistical_tests.py --results_csv outputs/drift_results/drift_StandardScaler_*.csv

Output:
    outputs/stats/wilcoxon_pairwise.csv
    outputs/stats/friedman_nemenyi.csv
    outputs/stats/cd_diagram.pdf
"""
import argparse
import os
import sys
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations
from scipy.stats import wilcoxon, friedmanchisquare, rankdata, norm

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "stats"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Publication typography (Rule 5 — Typography)
plt.rcParams["font.family"]  = "serif"
plt.rcParams["font.serif"]   = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["axes.spines.top"]   = False
plt.rcParams["axes.spines.right"] = False

# ─── Constants ────────────────────────────────────────────────────────────────
ALPHA_SIG = 0.05     # significance threshold (Rule 21.5.2)

# Baseline tiers (Rule 21.1)
METHODS_ORDER = [
    "Static-K (T1)",
    "ADYN only (T2)",
    "FL only (T3)",
    "FLAD (T4)",
    "Ours (V5)",
]


# ─── Wilcoxon Pairwise ────────────────────────────────────────────────────────
def wilcoxon_pairwise(df: pd.DataFrame, metric_col: str = "auc_roc_mean",
                      method_col: str = "method", group_col: str = "dataset") -> pd.DataFrame:
    """
    Perform Wilcoxon Signed-Rank Test for all pairs of methods.
    Paired across datasets (one score per dataset per method).

    Returns DataFrame with columns:
        method_a, method_b, W, N, p_value, effect_size_r, significant
    """
    methods = df[method_col].unique().tolist()
    records = []

    for m_a, m_b in combinations(methods, 2):
        scores_a = df[df[method_col] == m_a].set_index(group_col)[metric_col]
        scores_b = df[df[method_col] == m_b].set_index(group_col)[metric_col]
        # Align on common datasets
        common = scores_a.index.intersection(scores_b.index)
        if len(common) < 5:
            print(f"  SKIP {m_a} vs {m_b}: only {len(common)} common datasets (need ≥ 5)")
            continue
        x = scores_a.loc[common].values
        y = scores_b.loc[common].values
        diff = x - y
        if np.all(diff == 0):
            print(f"  SKIP {m_a} vs {m_b}: zero difference")
            continue
        try:
            W_stat, p_val = wilcoxon(x, y, alternative="two-sided")
        except ValueError as e:
            print(f"  ERROR {m_a} vs {m_b}: {e}")
            continue

        N = len(common)
        # Effect size r = Z / sqrt(N)  where Z derived from p-value (two-sided)
        Z = norm.ppf(1 - p_val / 2)
        r = abs(Z) / np.sqrt(N)

        records.append({
            "method_a": m_a,
            "method_b": m_b,
            "W": W_stat,
            "N": N,
            "p_value": p_val,
            "effect_size_r": round(r, 3),
            "significant": p_val < ALPHA_SIG,
        })

    return pd.DataFrame(records)


# ─── Friedman + Iman-Davenport ────────────────────────────────────────────────
def friedman_iman_davenport(df: pd.DataFrame, metric_col: str = "auc_roc_mean",
                             method_col: str = "method", group_col: str = "dataset"):
    """
    Friedman + Iman-Davenport F-statistic omnibus test.
    More powerful than chi-squared Friedman (Iman & Davenport 1980).
    """
    methods = df[method_col].unique().tolist()
    datasets = df[group_col].unique().tolist()
    k = len(methods)
    N = len(datasets)

    # Build rank matrix (N datasets × k methods)
    rank_matrix = np.zeros((N, k))
    for i, ds in enumerate(datasets):
        row = df[df[group_col] == ds].set_index(method_col)[metric_col]
        scores = np.array([row.get(m, np.nan) for m in methods])
        # Rank ascending (lower rank = worse)
        valid = ~np.isnan(scores)
        ranks = np.zeros(k)
        ranks[valid] = rankdata(scores[valid])   # ties handled by average
        rank_matrix[i] = ranks

    avg_ranks = rank_matrix.mean(axis=0)

    # Friedman chi-squared
    chi_F = (12 * N) / (k * (k + 1)) * (np.sum(avg_ranks**2) - k * (k + 1)**2 / 4)

    # Iman-Davenport F-statistic
    F_F = ((N - 1) * chi_F) / (N * (k - 1) - chi_F)

    from scipy.stats import f as f_dist
    p_friedman = f_dist.sf(F_F, dfn=k - 1, dfd=(k - 1) * (N - 1))

    print(f"\n── Friedman + Iman-Davenport Test ──")
    print(f"  k={k} methods, N={N} datasets")
    print(f"  χ²_F = {chi_F:.4f},  F_F = {F_F:.4f},  p = {p_friedman:.6f}")
    print(f"  H0 rejected: {p_friedman < ALPHA_SIG}")
    print(f"  Average ranks: " + ", ".join(f"{m}={r:.2f}" for m, r in zip(methods, avg_ranks)))

    return {
        "methods": methods,
        "avg_ranks": avg_ranks,
        "chi_F": chi_F,
        "F_F": F_F,
        "p_friedman": p_friedman,
        "h0_rejected": p_friedman < ALPHA_SIG,
        "k": k,
        "N": N,
    }


# ─── Nemenyi CD ───────────────────────────────────────────────────────────────
def nemenyi_cd(k: int, N: int, alpha: float = ALPHA_SIG) -> float:
    """
    Critical Difference for Nemenyi test (Demšar 2006).
    CD = q_alpha * sqrt(k(k+1) / 6N)
    q_alpha from Studentized range distribution / Nemenyi table.
    """
    # Approximate q_alpha values (Demšar 2006 Table 5, two-tailed alpha=0.05)
    q_table = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
               7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
    q = q_table.get(k, 2.728)  # default k=5
    cd = q * np.sqrt(k * (k + 1) / (6 * N))
    return cd


def bonferroni_dunn_cd(k: int, N: int, alpha: float = ALPHA_SIG) -> float:
    """
    Critical Difference for Bonferroni-Dunn test (1 control vs k-1 competitors).
    More powerful than Nemenyi when comparing proposed vs. all baselines.
    """
    alpha_adj = alpha / (k - 1)
    q = norm.ppf(1 - alpha_adj / 2)
    cd = q * np.sqrt(k * (k + 1) / (6 * N))
    return cd


# ─── CD Diagram ───────────────────────────────────────────────────────────────
def plot_cd_diagram(friedman_result: dict, title: str = "Critical Difference Diagram",
                    save_path: Path = None, test: str = "nemenyi"):
    """
    Plot Critical Difference Diagram (Demšar 2006 style).
    Methods are sorted by average rank (lower = better on right side).
    Connected by thick bar if difference < CD.
    """
    methods    = friedman_result["methods"]
    avg_ranks  = friedman_result["avg_ranks"]
    k          = friedman_result["k"]
    N          = friedman_result["N"]

    if test == "nemenyi":
        cd = nemenyi_cd(k, N)
        cd_label = f"Nemenyi CD = {cd:.3f}"
    else:
        cd = bonferroni_dunn_cd(k, N)
        cd_label = f"Bonferroni-Dunn CD = {cd:.3f}"

    # Sort by rank ascending
    order = np.argsort(avg_ranks)
    sorted_methods = [methods[i] for i in order]
    sorted_ranks   = avg_ranks[order]

    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.5 * k + 1.5)))
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(-0.5, k + 0.5)
    ax.invert_xaxis()   # rank 1 (best) on right
    ax.set_yticks([])
    ax.set_xlabel("Average Rank", fontsize=12)
    ax.set_title(title + f"\n({cd_label})", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Draw method labels
    for i, (name, rank) in enumerate(zip(sorted_methods, sorted_ranks)):
        ax.plot(rank, 0, "o", color="black", markersize=7, zorder=5)
        ax.text(rank, 0.3 + 0.5 * i, f"{name}\n({rank:.2f})",
                ha="center", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", lw=0.5))
        ax.plot([rank, rank], [0, 0.3 + 0.5 * i], "gray", lw=0.8, ls="--")

    # Draw CD groups (connected bar)
    def within_cd(i, j):
        return abs(sorted_ranks[i] - sorted_ranks[j]) < cd

    drawn = set()
    bar_y = -0.3
    for i in range(len(sorted_ranks)):
        group = [j for j in range(i, len(sorted_ranks)) if within_cd(i, j)]
        if len(group) > 1:
            key = tuple(group)
            if key not in drawn:
                drawn.add(key)
                r_start = sorted_ranks[group[0]]
                r_end   = sorted_ranks[group[-1]]
                ax.plot([r_start, r_end], [bar_y, bar_y], "k-", lw=3)
                bar_y -= 0.15

    ax.set_aspect("auto")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"  CD diagram saved: {save_path}")
    else:
        plt.savefig(OUTPUT_DIR / "cd_diagram.pdf", bbox_inches="tight")
        print(f"  CD diagram saved: {OUTPUT_DIR / 'cd_diagram.pdf'}")
    plt.close()


# ─── Win/Tie/Loss Summary ─────────────────────────────────────────────────────
def win_tie_loss(wilcoxon_df: pd.DataFrame, proposed: str = "Ours (V5)") -> pd.DataFrame:
    """
    Compute W/T/L table: proposed method vs. each baseline.
    """
    records = []
    rows_ab = wilcoxon_df[wilcoxon_df["method_a"] == proposed]
    rows_ba = wilcoxon_df[wilcoxon_df["method_b"] == proposed]

    for _, row in rows_ab.iterrows():
        opponent = row["method_b"]
        if row["significant"]:
            outcome = "Win" if row["method_a"] == proposed else "Loss"
        else:
            outcome = "Tie"
        records.append({"opponent": opponent, "outcome": outcome, "p": row["p_value"]})

    for _, row in rows_ba.iterrows():
        opponent = row["method_a"]
        if row["significant"]:
            outcome = "Win" if row["method_b"] == proposed else "Loss"
        else:
            outcome = "Tie"
        records.append({"opponent": opponent, "outcome": outcome, "p": row["p_value"]})

    df_wtl = pd.DataFrame(records)
    if df_wtl.empty:
        return df_wtl
    summary = df_wtl.groupby("opponent")["outcome"].value_counts().unstack(fill_value=0)
    return summary


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Statistical Tests — Rule 21.5")
    parser.add_argument("--results_csv", nargs="+", required=True,
                        help="One or more result CSV files (supports glob patterns)")
    parser.add_argument("--metric", default="auc_roc_mean",
                        help="Metric column name for comparison")
    parser.add_argument("--method_col", default="method",
                        help="Column for method names")
    parser.add_argument("--dataset_col", default="dataset",
                        help="Column for dataset names")
    parser.add_argument("--proposed", default="Ours (V5)",
                        help="Name of proposed method for W/T/L and Bonferroni-Dunn")
    parser.add_argument("--cd_test", choices=["nemenyi", "bonferroni_dunn"],
                        default="bonferroni_dunn",
                        help="Post-hoc CD test to draw (bonferroni_dunn preferred when comparing 1 vs all)")
    args = parser.parse_args()

    # Load CSV files
    files = []
    for pattern in args.results_csv:
        files.extend(glob.glob(pattern))
    if not files:
        print("ERROR: No CSV files found.")
        sys.exit(1)

    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"  Skip {f}: {e}")

    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} rows from {len(files)} file(s).")
    print(f"Columns: {df.columns.tolist()}")

    if args.method_col not in df.columns:
        print(f"ERROR: Column '{args.method_col}' not found. Available: {df.columns.tolist()}")
        sys.exit(1)

    if args.metric not in df.columns:
        print(f"WARNING: Metric '{args.metric}' not found. Using first numeric column.")
        args.metric = df.select_dtypes(include=np.number).columns[0]

    # ── 1. Wilcoxon Pairwise ──────────────────────────────────────────────────
    print(f"\n── Wilcoxon Signed-Rank Test (α={ALPHA_SIG}) ──")
    wilcoxon_df = wilcoxon_pairwise(df, args.metric, args.method_col, args.dataset_col)
    if not wilcoxon_df.empty:
        print(wilcoxon_df.to_string(index=False))
        wilcoxon_df.to_csv(OUTPUT_DIR / "wilcoxon_pairwise.csv", index=False)
        print(f"  Saved: {OUTPUT_DIR / 'wilcoxon_pairwise.csv'}")

    # ── 2. Friedman + Iman-Davenport ─────────────────────────────────────────
    friedman_result = friedman_iman_davenport(df, args.metric, args.method_col, args.dataset_col)

    friedman_summary = pd.DataFrame({
        "method": friedman_result["methods"],
        "avg_rank": friedman_result["avg_ranks"],
    }).sort_values("avg_rank")
    friedman_summary["chi_F"]      = friedman_result["chi_F"]
    friedman_summary["F_F"]        = friedman_result["F_F"]
    friedman_summary["p_friedman"] = friedman_result["p_friedman"]
    friedman_summary.to_csv(OUTPUT_DIR / "friedman_nemenyi.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR / 'friedman_nemenyi.csv'}")

    # ── 3. CD Diagram ─────────────────────────────────────────────────────────
    if friedman_result["h0_rejected"]:
        print("\n── Generating Critical Difference Diagram ──")
        plot_cd_diagram(
            friedman_result,
            title="ADYN+FL vs Baselines — CD Diagram",
            save_path=OUTPUT_DIR / "cd_diagram.pdf",
            test=args.cd_test,
        )
    else:
        print("\n  H0 NOT rejected by Friedman — CD diagram skipped (no significant overall difference).")

    # ── 4. Win/Tie/Loss ───────────────────────────────────────────────────────
    if not wilcoxon_df.empty:
        print(f"\n── Win/Tie/Loss Table ({args.proposed} vs. Baselines) ──")
        wtl = win_tie_loss(wilcoxon_df, proposed=args.proposed)
        if not wtl.empty:
            print(wtl)
            wtl.to_csv(OUTPUT_DIR / "win_tie_loss.csv")

    print("\n✓ Statistical tests complete. Results in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
