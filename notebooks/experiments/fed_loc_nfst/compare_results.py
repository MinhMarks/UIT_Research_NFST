"""
FL vs Centralized Comparison & Plot Generator
==============================================
Loads FL results and centralized baseline, computes delta metrics,
and generates comparison plots for thesis.

Usage:
    python compare_results.py \
        --fl-csv outputs/federated_results/fl_all_results_<ts>.csv \
        --central-csv outputs/federated_results/centralized_results_<ts>.csv

Output:
    outputs/federated_results/plots/
        f1_comparison.png
        auc_comparison.png
        communication_cost.png
        comparison_table.csv
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server

_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR.parent))
from fed_loc_nfst.config import OUTPUT_DIR


PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)


def load_results(fl_csv: str, central_csv: str):
    fl_df = pd.read_csv(fl_csv)
    central_df = pd.read_csv(central_csv)
    return fl_df, central_df


def merge_results(fl_df: pd.DataFrame, central_df: pd.DataFrame) -> pd.DataFrame:
    """Merge on dataset + scaler, compute delta metrics."""
    fl_cols = {
        'dataset': 'dataset',
        'scaler': 'scaler',
        'FL_F1_weighted': 'FL_F1',
        'FL_AUCROC_weighted': 'FL_AUCROC',
        'FL_MCC_weighted': 'FL_MCC',
        'comm_total_comm_KB': 'Total_Comm_KB',
        'fl_time_s': 'FL_Time_s',
        'num_clients': 'Clients',
        'K_clusters': 'K',
        'L_null_dim': 'L',
    }
    central_cols = {
        'dataset': 'dataset',
        'scaler': 'scaler',
        'F1 Score': 'Central_F1',
        'AUCROC': 'Central_AUCROC',
        'MCC': 'Central_MCC',
        'train_time_s': 'Central_Time_s',
    }

    fl_sub = fl_df.rename(columns=fl_cols)[[c for c in fl_cols.values() if c in fl_df.rename(columns=fl_cols).columns]]
    central_sub = central_df.rename(columns=central_cols)[[c for c in central_cols.values() if c in central_df.rename(columns=central_cols).columns]]

    merged = pd.merge(fl_sub, central_sub, on=['dataset', 'scaler'], how='inner')

    # Delta metrics
    if 'FL_F1' in merged.columns and 'Central_F1' in merged.columns:
        merged['Delta_F1_pct'] = ((merged['FL_F1'] - merged['Central_F1']) * 100).round(4)
    if 'FL_AUCROC' in merged.columns and 'Central_AUCROC' in merged.columns:
        merged['Delta_AUCROC'] = (merged['FL_AUCROC'] - merged['Central_AUCROC']).round(4)

    return merged


def plot_f1_comparison(merged: pd.DataFrame, out_path: str):
    """Bar chart: FL F1 vs Centralized F1 per dataset."""
    fig, ax = plt.subplots(figsize=(10, 5))

    datasets = merged['dataset'].unique()
    x = np.arange(len(datasets))
    w = 0.35

    fl_f1s = [merged[merged['dataset'] == d]['FL_F1'].mean() for d in datasets]
    c_f1s  = [merged[merged['dataset'] == d]['Central_F1'].mean() for d in datasets]

    bars_fl = ax.bar(x - w/2, fl_f1s, w, label='FL-LOC-NFST (Federated)', color='steelblue', alpha=0.85)
    bars_c  = ax.bar(x + w/2, c_f1s,  w, label='LOC-NFST (Centralized)',  color='darkorange', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([d.replace('data_', '') for d in datasets], rotation=15, ha='right')
    ax.set_ylabel('F1 Score')
    ax.set_title('FL-LOC-NFST vs Centralized LOC-NFST: F1 Score Comparison')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    # Annotate delta
    for i, (fl, c) in enumerate(zip(fl_f1s, c_f1s)):
        delta = (fl - c) * 100
        ax.annotate(f'Δ={delta:+.2f}%', xy=(x[i], max(fl, c) + 0.01),
                    ha='center', fontsize=8, color='gray')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"F1 comparison plot saved: {out_path}")


def plot_communication_cost(merged: pd.DataFrame, out_path: str):
    """Communication cost per experiment configuration."""
    if 'Total_Comm_KB' not in merged.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    datasets = merged['dataset'].unique()
    comm_kbs = [merged[merged['dataset'] == d]['Total_Comm_KB'].mean() for d in datasets]

    bars = ax.bar([d.replace('data_', '') for d in datasets], comm_kbs,
                   color='teal', alpha=0.8)
    ax.set_ylabel('Total Communication (KB)')
    ax.set_title('FL-LOC-NFST: Total Communication Cost (Upload + Broadcast)')
    ax.set_xlabel('Dataset')
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, comm_kbs):
        ax.annotate(f'{val:.1f} KB', xy=(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2),
                    ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Communication cost plot saved: {out_path}")


def plot_auc_comparison(merged: pd.DataFrame, out_path: str):
    """AUC-ROC comparison."""
    if 'FL_AUCROC' not in merged.columns or 'Central_AUCROC' not in merged.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    datasets = merged['dataset'].unique()
    x = np.arange(len(datasets))
    w = 0.35

    fl_aucs = [merged[merged['dataset'] == d]['FL_AUCROC'].mean() for d in datasets]
    c_aucs  = [merged[merged['dataset'] == d]['Central_AUCROC'].mean() for d in datasets]

    ax.bar(x - w/2, fl_aucs, w, label='FL-LOC-NFST', color='steelblue', alpha=0.85)
    ax.bar(x + w/2, c_aucs,  w, label='Centralized',  color='darkorange', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace('data_', '') for d in datasets], rotation=15, ha='right')
    ax.set_ylabel('AUC-ROC (%)')
    ax.set_title('FL-LOC-NFST vs Centralized: AUC-ROC Comparison')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"AUC comparison plot saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="FL vs Centralized Comparison")
    parser.add_argument('--fl-csv', type=str, required=True)
    parser.add_argument('--central-csv', type=str, required=True)
    args = parser.parse_args()

    fl_df, central_df = load_results(args.fl_csv, args.central_csv)
    merged = merge_results(fl_df, central_df)

    if merged.empty:
        print("⚠ No matching rows to compare. Check dataset/scaler names.")
        return

    # Save comparison table
    table_path = os.path.join(PLOT_DIR, 'comparison_table.csv')
    merged.to_csv(table_path, index=False)
    print(f"Comparison table saved: {table_path}")
    print(merged.to_string())

    # Plots
    plot_f1_comparison(merged, os.path.join(PLOT_DIR, 'f1_comparison.png'))
    plot_auc_comparison(merged, os.path.join(PLOT_DIR, 'auc_comparison.png'))
    plot_communication_cost(merged, os.path.join(PLOT_DIR, 'communication_cost.png'))

    # Summary
    if 'Delta_F1_pct' in merged.columns:
        max_delta = merged['Delta_F1_pct'].abs().max()
        print(f"\n{'='*50}")
        print(f"Max F1 deviation (FL vs Centralized): {max_delta:.4f}%")
        if max_delta <= 0.5:
            print("✓ PASS: F1 deviation ≤ 0.5% (Protocol A One-Shot target met)")
        else:
            print(f"⚠ F1 deviation {max_delta:.4f}% > 0.5% — investigate scatter correction")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
