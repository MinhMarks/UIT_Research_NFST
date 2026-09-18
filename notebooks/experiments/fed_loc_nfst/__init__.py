"""
Federated LOC-NFST (FL-LOC-NFST)
==================================
One-Shot Lossless Scatter Matrix Aggregation for Federated Anomaly Detection.

Architecture: Protocol A — Single communication round (T=1).
- Clients compute local scatter matrices S_w_m via Welford accumulation.
- Server aggregates S_w_global with scatter-shift correction.
- Server solves global NPD → broadcasts W, null_centers, max_train.
- Clients perform local inference.

Reference: FEDERATED_EDGE_LOC_NFST_COMPREHENSIVE_REPORT.md
"""
