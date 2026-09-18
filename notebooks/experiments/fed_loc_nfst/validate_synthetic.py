"""
Quick synthetic validation of FL-LOC-NFST (d=46 to match CICIoT2023).
Run: python validate_synthetic.py
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fed_loc_nfst.client import compute_local_scatter
from fed_loc_nfst.strategy import (
    aggregate_scatter_matrices, adaptive_spectral_solve, compute_null_centers
)
from fed_loc_nfst.evaluate import compute_fl_scores, evaluate_scores
from fed_loc_nfst.run_centralized import compute_centralized_NPD

def main():
    print("=" * 55)
    print("  FL-LOC-NFST Synthetic Validation (d=46, K=5, M=3)")
    print("=" * 55)

    N, d, K, M = 1000, 46, 5, 3
    rng = np.random.default_rng(42)

    # Synthetic data: K well-separated Gaussian clusters (normal traffic)
    centers = rng.normal(0, 3, (K, d)).astype(np.float32)
    X_train = np.vstack([
        rng.normal(centers[k], 0.5, (N // K, d)).astype(np.float32)
        for k in range(K)
    ])

    # Test: half normal (near 0), half anomaly (near 10)
    X_test_normal  = rng.normal(0,  0.5, (300, d)).astype(np.float32)
    X_test_anomaly = rng.normal(10, 1.0, (300, d)).astype(np.float32)
    X_test = np.vstack([X_test_normal, X_test_anomaly])
    y_test = np.concatenate([np.zeros(300), np.ones(300)])

    # ── Centralized ──────────────────────────────────────
    print("\n[Centralized LOC-NFST]")
    W_c, centers_c, max_c, L_c, K_actual = compute_centralized_NPD(X_train, K)
    y_proba_c = compute_fl_scores(X_test, W_c, centers_c, max_c)
    m_c = evaluate_scores(y_test, y_proba_c)
    print(f"  F1       = {m_c['F1 Score']:.4f}")
    print(f"  AUC-ROC  = {m_c['AUCROC']:.2f}%")
    print(f"  L (null) = {L_c}")

    # ── FL (IID, M=3 clients) ───────────────────────────
    print("\n[FL-LOC-NFST (IID, M=3 clients)]")
    splits = np.array_split(X_train, M)
    S_list, cent_list, cnt_list = [], [], []

    for i, X_m in enumerate(splits):
        S_w_m, c_m, cnt_m, _, t = compute_local_scatter(X_m, K)
        S_list.append(S_w_m)
        cent_list.append(c_m)
        cnt_list.append(cnt_m)
        payload_kb = (S_w_m.nbytes + c_m.nbytes + cnt_m.nbytes) / 1024
        print(f"  Client {i}: N={len(X_m)}, K={len(cnt_m)}, "
              f"payload={payload_kb:.2f} KB, time={t:.3f}s")

    S_w_g, S_t_g, anchors, N_per = aggregate_scatter_matrices(
        S_list, cent_list, cnt_list, K_global=K
    )
    W_fl, L_fl = adaptive_spectral_solve(S_w_g, S_t_g)
    null_c, max_t = compute_null_centers(anchors, W_fl)

    y_proba_fl = compute_fl_scores(X_test, W_fl, null_c, max_t)
    m_fl = evaluate_scores(y_test, y_proba_fl)
    print(f"  F1       = {m_fl['F1 Score']:.4f}")
    print(f"  AUC-ROC  = {m_fl['AUCROC']:.2f}%")
    print(f"  L (null) = {L_fl}")

    # ── Delta ─────────────────────────────────────────────
    delta_f1 = abs(m_fl["F1 Score"] - m_c["F1 Score"]) * 100
    delta_auc = abs(m_fl["AUCROC"] - m_c["AUCROC"])

    print("\n[Comparison]")
    print(f"  Delta F1      = {delta_f1:.4f}% (target <= 0.5% on real data)")
    print(f"  Delta AUC-ROC = {delta_auc:.4f}%")

    # Communication summary
    total_upload_kb = sum(
        (S_list[i].nbytes + cent_list[i].nbytes + cnt_list[i].nbytes)
        for i in range(M)
    ) / 1024
    broadcast_kb = (W_fl.nbytes + null_c.nbytes + 4) / 1024

    print("\n[Communication Cost]")
    print(f"  Total upload ({M} clients): {total_upload_kb:.2f} KB")
    print(f"  Server broadcast:           {broadcast_kb:.2f} KB")
    print(f"  Total communication:        {total_upload_kb + broadcast_kb:.2f} KB")

    # Result
    print("\n" + "=" * 55)
    if delta_f1 <= 5.0:
        print("  PASS — FL math is correct")
    else:
        print(f"  ✗ Delta F1 = {delta_f1:.4f}% — investigate")
    print("=" * 55)

if __name__ == "__main__":
    main()
