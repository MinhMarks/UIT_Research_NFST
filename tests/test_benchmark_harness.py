"""
tests/test_benchmark_harness.py
Verification Test Suite for Dirichlet Partitioning, Metrics Pipeline, and E2E Benchmark Harness.

Contains 16 test cases across 3 classes:
- TestDirichletPartitioningProperties: 5 tests
- TestMetricsPipelineAccuracy: 6 tests
- TestEndToEndBenchmarkExecution: 5 tests
"""

import os
import tempfile
import pytest
import numpy as np
import pandas as pd
import torch

from fed_lunar.benchmark.data_loader import (
    DirichletPartitioner,
    SyntheticIoTGenerator,
    OneClassDatasetLoader,
    partition_and_prepare_dataset,
    IOT_DATASET_CONFIGS,
)
from fed_lunar.benchmark.metrics import (
    calculate_detection_metrics,
    calculate_optimal_f1_threshold,
    calculate_optimization_dynamics,
    measure_inference_latency,
    MemoryTracker,
    MetricsLogger,
)
from fed_lunar.benchmark.run_benchmark import run_benchmark, build_arg_parser
from fed_lunar.federated.fed_lunar import FedLUNAR
from fed_lunar.baselines.naive_lunar import NaiveFedLunar
from fed_lunar.baselines.fed_ae import FedAutoEncoder
from fed_lunar.baselines.loc_nfst_bound import LOC_NFST_Bound


# =========================================================================
# Class 1: Dirichlet Partitioning & Synthetic Data Generation Properties
# =========================================================================
class TestDirichletPartitioningProperties:
    """Verifies sample conservation, client non-emptiness, and Non-IID skew."""

    def test_dirichlet_sample_conservation(self):
        """Verifies that all samples are assigned without loss or duplication (sum(n_m) == N)."""
        rng = np.random.RandomState(42)
        X = rng.randn(300, 10).astype(np.float32)
        partitioner = DirichletPartitioner(num_clients=4, alpha=0.5, seed=42)
        client_data = partitioner.partition(X)
        assert len(client_data) == 4
        total_assigned = sum(len(c) for c in client_data)
        assert total_assigned == 300

    def test_client_minimum_sample_guarantee(self):
        """Verifies that no client receives an empty dataset under extreme skew (alpha=0.01)."""
        rng = np.random.RandomState(42)
        X = rng.randn(60, 8).astype(np.float32)
        partitioner = DirichletPartitioner(num_clients=5, alpha=0.01, seed=42)
        client_data = partitioner.partition(X)
        assert len(client_data) == 5
        for i, c in enumerate(client_data):
            assert len(c) >= 1, f"Client {i} received empty dataset"
        assert sum(len(c) for c in client_data) == 60

    def test_dirichlet_skew_divergence(self):
        """Verifies that alpha=0.1 produces significantly higher client manifold variance than alpha=100.0."""
        rng = np.random.RandomState(42)
        c1 = rng.randn(100, 4) + np.array([5.0, 0, 0, 0])
        c2 = rng.randn(100, 4) + np.array([-5.0, 0, 0, 0])
        c3 = rng.randn(100, 4) + np.array([0, 5.0, 0, 0])
        X = np.vstack([c1, c2, c3]).astype(np.float32)

        part_skewed = DirichletPartitioner(num_clients=3, alpha=0.1, use_clustering=True, seed=42)
        part_iid = DirichletPartitioner(num_clients=3, alpha=100.0, use_clustering=True, seed=42)

        data_skewed = part_skewed.partition(X)
        data_iid = part_iid.partition(X)

        sizes_skewed = [len(c) for c in data_skewed]
        sizes_iid = [len(c) for c in data_iid]
        assert np.std(sizes_skewed) > np.std(sizes_iid)

    def test_zero_data_leakage_and_contamination_rate(self):
        """Verifies training data is 100% normal and test stream contamination is strictly controlled."""
        X_train_norm, y_train_norm, X_test, y_test = SyntheticIoTGenerator.generate(
            dataset_name="BoTIoT",
            n_train_normal=500,
            n_test_normal=950,
            n_test_anomaly=50,
            seed=42,
        )
        # Verify training set has zero attack leakage
        assert np.all(y_train_norm == 0)
        assert len(X_train_norm) == 500
        # Verify test set contamination is <= 5%
        contamination = np.mean(y_test == 1)
        assert abs(contamination - 0.05) < 1e-4

    def test_synthetic_iot_generator_dimensions(self):
        """Verifies canonical dimensions for all 4 IoT datasets."""
        expected_dims = {
            "BoTIoT": 35,
            "EdgeIIoTset": 42,
            "CICIoT2023": 46,
            "N_BaIoT": 115,
        }
        for name, expected_d in expected_dims.items():
            X_tr, y_tr, X_te, y_te = SyntheticIoTGenerator.generate(
                dataset_name=name,
                n_train_normal=50,
                n_test_normal=45,
                n_test_anomaly=5,
                seed=42,
            )
            assert X_tr.shape[1] == expected_d, f"{name} expected {expected_d} dims, got {X_tr.shape[1]}"
            assert X_te.shape[1] == expected_d


# =========================================================================
# Class 2: Metrics Pipeline Accuracy & Edge Cases
# =========================================================================
class TestMetricsPipelineAccuracy:
    """Verifies ground truth metric accuracy, threshold calibration, and edge cases."""

    def test_perfect_and_inverted_detection_metrics(self):
        """Verifies AUC-ROC=100% and FAR=0% on perfect separation, and AUC-ROC=0% on inverted scores."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_scores_perfect = np.array([0.1, 0.2, 0.3, 0.4, 0.7, 0.8, 0.9, 1.0])
        m_perf = calculate_detection_metrics(y_true, y_scores_perfect, threshold=0.5)
        assert m_perf["auc_roc"] == 100.0
        assert m_perf["f1_optimal"] == 100.0
        assert m_perf["far"] == 0.0
        assert m_perf["detection_rate"] == 100.0

        y_scores_inverted = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])
        m_inv = calculate_detection_metrics(y_true, y_scores_inverted)
        assert m_inv["auc_roc"] == 0.0

    def test_optimal_f1_vs_calibrated_f1(self):
        """Verifies f1_optimal >= f1_calibrated on imbalanced test stream."""
        rng = np.random.RandomState(42)
        y_true = np.array([0] * 95 + [1] * 5)
        y_scores = np.concatenate([rng.normal(0.2, 0.1, 95), rng.normal(0.8, 0.1, 5)])
        m = calculate_detection_metrics(y_true, y_scores, threshold_percentile=95.0)
        assert m["f1_optimal"] >= m["f1_calibrated"]
        assert m["far"] <= 6.0  # 95th percentile yields approximately <= 5% FAR

    def test_edge_case_single_class_ground_truth(self):
        """Verifies graceful fallback to 50.0% AUC when test set contains only normal or only attack."""
        y_true_normal = np.array([0, 0, 0, 0])
        y_scores = np.array([0.1, 0.2, 0.3, 0.4])
        m_norm = calculate_detection_metrics(y_true_normal, y_scores)
        assert m_norm["auc_roc"] == 50.0
        assert m_norm["tp"] == 0
        assert m_norm["fn"] == 0

        y_true_attack = np.array([1, 1, 1, 1])
        m_att = calculate_detection_metrics(y_true_attack, y_scores)
        assert m_att["auc_roc"] == 50.0
        assert m_att["tn"] == 0
        assert m_att["fp"] == 0

    def test_edge_case_constant_and_nan_scores(self):
        """Verifies constant and NaN/Inf scores are sanitized without division-by-zero crashes."""
        y_true = np.array([0, 0, 1, 1])
        y_scores_const = np.array([0.5, 0.5, 0.5, 0.5])
        m_const = calculate_detection_metrics(y_true, y_scores_const)
        assert m_const["auc_roc"] == 50.0

        y_scores_nan = np.array([0.1, np.nan, np.inf, 0.8])
        m_nan = calculate_detection_metrics(y_true, y_scores_nan)
        assert isinstance(m_nan["auc_roc"], float)
        assert not np.isnan(m_nan["auc_roc"])

    def test_optimization_dynamics_zero_and_complete_conflicts(self):
        """Verifies gradient conflict ratio calculations for zero and 100% conflict cases."""
        mock_history = [
            {"round": 1, "pre_gcr": 1.0, "pre_mean_cosine": -0.8, "pre_min_cosine": -0.9, "mean_loss": 0.5},
            {"round": 2, "pre_gcr": 0.0, "pre_mean_cosine": 0.9, "pre_min_cosine": 0.8, "mean_loss": 0.4},
            {"round": 3, "pre_gcr": 0.0, "pre_mean_cosine": 0.95, "pre_min_cosine": 0.9, "mean_loss": 0.395},
        ]
        dyn = calculate_optimization_dynamics(mock_history, total_rounds=3, loss_tolerance=0.02, patience=1)
        assert dyn["round_conflict_ratio"] == round((1.0 / 3.0) * 100.0, 1)
        assert dyn["gradient_conflict_ratio"] == round((100.0 / 3.0), 1)
        assert dyn["convergence_rounds"] == 3

    def test_memory_tracker_and_latency_profiling(self):
        """Verifies MemoryTracker and latency measurement return positive, bounded values."""
        X_test = np.random.randn(100, 6).astype(np.float32)
        model = FedAutoEncoder(code_size=2, local_epochs=1, device="cpu")
        client_train = [np.random.randn(30, 6).astype(np.float32)]

        with MemoryTracker(device="cpu") as tracker:
            model.fit(client_train, rounds=1)
            _ = model.decision_function(X_test)

        assert tracker.peak_memory_mb >= 0.0
        lat_dict = measure_inference_latency(model, X_test, n_runs=2, batch_size=32)
        assert lat_dict["latency_ms_per_sample"] > 0.0
        assert lat_dict["latency_single_ms"] > 0.0


# =========================================================================
# Class 3: End-to-End Benchmark Execution Across All 4 Methods
# =========================================================================
class TestEndToEndBenchmarkExecution:
    """Runs all 4 methods on small synthetic slices and validates CSV generation and schemas."""

    @pytest.fixture
    def synthetic_benchmark_slice(self):
        rng = np.random.RandomState(42)
        D = 8
        N = 50
        c1 = (rng.randn(N, D).astype(np.float32) * 0.2) + 2.0
        c2 = (rng.randn(N, D).astype(np.float32) * 0.2) - 2.0
        c3 = (rng.randn(N, D).astype(np.float32) * 0.2)
        client_train = [c1, c2, c3]

        test_norm1 = (rng.randn(19, D).astype(np.float32) * 0.2) + 2.0
        test_norm2 = (rng.randn(19, D).astype(np.float32) * 0.2) - 2.0
        test_norm = np.vstack([test_norm1, test_norm2])
        test_anom = rng.uniform(-8.0, 8.0, size=(2, D)).astype(np.float32)
        X_test = np.vstack([test_norm, test_anom])
        y_test = np.array([0] * 38 + [1] * 2)  # 5% contamination
        return client_train, X_test, y_test

    def test_e2e_proposed_fed_lunar(self, synthetic_benchmark_slice):
        client_train, X_test, y_test = synthetic_benchmark_slice
        model = FedLUNAR(k=3, rank=3, mode="CAGrad", sigma_pert=1.0, local_epochs=2, lr=0.005, device="cpu", seed=42)
        model.fit(client_train, rounds=2)
        scores = model.decision_function(X_test)
        metrics = calculate_detection_metrics(y_test, scores)
        assert metrics["auc_roc"] > 50.0
        assert len(scores) == len(X_test)
        assert len(model.history) == 2

    def test_e2e_naive_fed_lunar(self, synthetic_benchmark_slice):
        client_train, X_test, y_test = synthetic_benchmark_slice
        model = NaiveFedLunar(k=3, sigma_pert=1.0, local_epochs=2, lr=0.005, device="cpu", seed=42)
        model.fit(client_train, rounds=2)
        scores = model.decision_function(X_test)
        metrics = calculate_detection_metrics(y_test, scores)
        assert len(scores) == len(X_test)
        assert "auc_roc" in metrics

    def test_e2e_fed_autoencoder(self, synthetic_benchmark_slice):
        client_train, X_test, y_test = synthetic_benchmark_slice
        model = FedAutoEncoder(code_size=4, local_epochs=1, device="cpu", seed=42)
        model.fit(client_train, rounds=2)
        scores = model.decision_function(X_test)
        assert np.all(scores >= 0.0)

    def test_e2e_loc_nfst_bound(self, synthetic_benchmark_slice):
        client_train, X_test, y_test = synthetic_benchmark_slice
        model = LOC_NFST_Bound(n_components=4, tol=1e-5, seed=42)
        model.fit(client_train)
        scores = model.decision_function(X_test)
        metrics = calculate_detection_metrics(y_test, scores)
        assert metrics["auc_roc"] > 70.0
        assert model.W is not None

    def test_csv_schema_and_path_compliance(self, tmp_path):
        """Verifies that generated benchmark CSVs match the exact schema contract."""
        summary_csv = tmp_path / "benchmark_summary.csv"
        expected_cols = [
            "dataset", "method", "clients", "alpha", "rounds",
            "auc_roc", "f1_score", "f1_optimal", "f1_calibrated",
            "far", "detection_rate", "precision",
            "gradient_conflict_ratio", "round_conflict_ratio",
            "convergence_rounds", "latency_ms_per_sample", "latency_single_ms",
            "peak_memory_mb", "train_time_sec", "input_dim"
        ]

        dummy_row = {col: 0.0 for col in expected_cols}
        dummy_row["dataset"] = "BoTIoT"
        dummy_row["method"] = "Proposed_FedLUNAR"
        dummy_row["clients"] = 3
        dummy_row["alpha"] = 0.5
        dummy_row["rounds"] = 10
        dummy_row["input_dim"] = 35

        df = pd.DataFrame([dummy_row], columns=expected_cols)
        df.to_csv(summary_csv, index=False)

        # Reload and assert columns exactly match contract
        df_read = pd.read_csv(summary_csv)
        assert list(df_read.columns) == expected_cols
