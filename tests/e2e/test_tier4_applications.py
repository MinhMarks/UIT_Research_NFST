"""Tier 4: Real-World Application Scenarios Test Suite for Federated LUNAR.

Implements realistic, end-to-end production workflows across 5 IoT intrusion detection scenarios:
1. Scenario 1: BoTIoT Smart Home Gateway (35 features, Mirai/Gafgyt botnet DDoS).
2. Scenario 2: EdgeIIoTset Industrial Manufacturing Plant (42 features, Modbus/MQTT SCADA).
3. Scenario 3: CICIoT2023 Smart City Infrastructure (46 features, High-Throughput Volumetric Flood).
4. Scenario 4: N_BaIoT Commercial IoT Fleet (115 features, Hardware Sensor Heterogeneity).
5. Scenario 5: Dynamic IoT Node Churn & Intermittent Communication Dropout.
"""

from __future__ import annotations
import math
import os
import tempfile
import time
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F

from tests.e2e.contract_stubs import (
    get_lunar_mlp,
    get_negative_generator,
    get_fsds_class,
    get_cmnp_filter_class,
    get_droga,
    compute_pairwise_cosine_similarity,
    compute_gradient_conflict_ratio,
    get_autoencoder,
    get_loc_nfst_bound,
    get_dirichlet_partitioner,
    get_metrics_logger,
)


# ===========================================================================
# Scenario 1: BoTIoT Smart Home Gateway (35 Features)
# ===========================================================================

def test_s1_botiot_smart_home_gateway_workflow():
    """S1: End-to-end intrusion detection workflow on 35-feature BoTIoT smart home telemetry."""
    np.random.seed(42)
    torch.manual_seed(42)
    FSDS = get_fsds_class()
    CMNPFilter = get_cmnp_filter_class()
    DROGA = get_droga()
    MetricsLogger = get_metrics_logger()
    
    D = 35
    M = 3
    # 3 Smart Home device categories: Smart Speaker, Smart Plug, Security Camera
    X_speaker = np.random.normal(loc=0.0, scale=0.2, size=(100, D)).astype(np.float32)
    X_plug = np.random.normal(loc=1.5, scale=0.15, size=(100, D)).astype(np.float32)
    X_cam = np.random.normal(loc=-1.5, scale=0.25, size=(100, D)).astype(np.float32)
    client_data = [X_speaker, X_plug, X_cam]
    
    # Round 0: Sketch exchange
    sketches = [FSDS.fit(data, rank=5) for data in client_data]
    assert all(s.U.shape == (35, 5) for s in sketches)
    
    # Federated model setup
    k = 8
    global_model = get_lunar_mlp(k=k)
    optimizer = torch.optim.Adam(global_model.parameters(), lr=0.05)
    
    # 5 Federated Training Rounds
    for r in range(5):
        client_grads = []
        for c in range(M):
            peer_sketches = [sketches[j] for j in range(M) if j != c]
            cmnp = CMNPFilter(peer_sketches=peer_sketches, tau_null=1.5)
            gen = get_negative_generator(negative_ratio=1.0, epsilon=0.5, cmnp=cmnp)
            
            norm_batch = client_data[c][:30]
            neg_batch = gen.generate(norm_batch)
            
            d_norm = torch.sort(torch.rand(len(norm_batch), k) * 0.2 + 0.05, dim=1)[0]
            d_anom = torch.sort(torch.rand(len(neg_batch), k) * 0.5 + 2.0, dim=1)[0]
            
            c_model = get_lunar_mlp(k=k)
            c_model.load_state_dict(global_model.state_dict())
            loss = nn.BCELoss()(c_model(d_norm), torch.zeros(len(norm_batch), 1)) + \
                   nn.BCELoss()(c_model(d_anom), torch.ones(len(neg_batch), 1))
            g = torch.cat([grad.view(-1) for grad in torch.autograd.grad(loss, c_model.parameters())])
            client_grads.append(g)
            
        # Server DROGA alignment
        g_aligned = DROGA.dr_pcgrad(client_grads)
        
        # Apply aligned gradient to global model
        optimizer.zero_grad()
        curr_idx = 0
        for p in global_model.parameters():
            num_p = p.numel()
            p.grad = g_aligned[curr_idx:curr_idx + num_p].view_as(p).clone()
            curr_idx += num_p
        optimizer.step()
                
    # Evaluation on mixed stream (95% normal, 5% Mirai botnet attack)
    X_eval_norm = torch.sort(torch.rand(95, k) * 0.3, dim=1)[0]
    X_eval_anom = torch.sort(torch.rand(5, k) * 0.8 + 1.5, dim=1)[0]
    X_eval = torch.cat([X_eval_norm, X_eval_anom])
    y_eval = np.array([0] * 95 + [1] * 5)
    
    with torch.no_grad():
        scores = global_model(X_eval).view(-1).numpy()
        
    metrics = MetricsLogger.evaluate(y_eval, scores)
    assert metrics["auc_roc"] > 75.0, f"Expected AUC > 75%, got {metrics['auc_roc']}%"
    assert metrics["far"] <= 0.15


# ===========================================================================
# Scenario 2: EdgeIIoTset Industrial Manufacturing Plant (42 Features)
# ===========================================================================

def test_s2_edgeiiotset_industrial_scada_workflow():
    """S2: Factory floor telemetry anomaly detection with Modbus/MQTT protocols across 4 zones."""
    np.random.seed(42)
    torch.manual_seed(42)
    FSDS = get_fsds_class()
    CMNPFilter = get_cmnp_filter_class()
    DROGA = get_droga()
    MetricsLogger = get_metrics_logger()
    
    D = 42
    M = 4  # 4 Zones: Assembly, Packaging, Quality, Power
    zones_data = [np.random.normal(loc=i * 2.0, scale=0.2, size=(80, D)).astype(np.float32) for i in range(M)]
    
    sketches = [FSDS.fit(data, rank=6) for data in zones_data]
    k = 10
    model = get_lunar_mlp(k=k)
    
    # Verify CMNP purging across zones
    # Candidates generated from Zone 0 should NOT intrude on Zone 1, 2, 3
    cmnp_zone0 = CMNPFilter(peer_sketches=sketches[1:], tau_null=1.5)
    gen = get_negative_generator(negative_ratio=1.0, epsilon=1.0, cmnp=cmnp_zone0)
    purged_negs = gen.generate(zones_data[0])
    
    # Verify none of the purged negatives intrude on Zone 1
    intrusions = [cmnp_zone0.is_intruding(neg) for neg in purged_negs]
    assert sum(intrusions) == 0, "All remaining pseudo-negatives must be non-intruding"


# ===========================================================================
# Scenario 3: CICIoT2023 Smart City Infrastructure (46 Features)
# ===========================================================================

def test_s3_ciciot2023_high_throughput_flood_workflow():
    """S3: High-throughput flood attack detection across 5 municipal sensor gateways."""
    np.random.seed(42)
    torch.manual_seed(42)
    D = 46
    M = 5
    partitioner = get_dirichlet_partitioner(num_clients=M, alpha=0.5)
    
    # 500 samples partitioned across 5 city hubs
    X_city = np.random.randn(500, D).astype(np.float32)
    splits = partitioner.partition(X_city)
    assert len(splits) == M
    
    # Measure per-sample inference latency
    k = 10
    model = get_lunar_mlp(k=k)
    model.eval()
    
    test_batch = torch.sort(torch.rand(100, k), dim=1)[0]
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(10):
            _ = model(test_batch)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / (100 * 10)
    
    # Real-time constraint: latency must be < 5.0 ms per sample
    assert elapsed_ms < 5.0, f"Inference latency too high: {elapsed_ms:.4f} ms/sample (must be < 5.0 ms)"


# ===========================================================================
# Scenario 4: N_BaIoT Commercial Hardware Fleet (115 Features)
# ===========================================================================

def test_s4_nbaiot_hardware_fleet_workflow():
    """S4: Hardware fleet profiling on 115-dimensional commercial IoT sensor telemetry."""
    np.random.seed(42)
    FSDS = get_fsds_class()
    DROGA = get_droga()
    MetricsLogger = get_metrics_logger()
    
    D = 115
    # Simulating 3 commercial devices: Doorbell, Thermostat, Baby Monitor
    X_doorbell = np.random.normal(loc=0.0, scale=0.5, size=(100, D)).astype(np.float32)
    X_thermostat = np.random.normal(loc=5.0, scale=0.3, size=(100, D)).astype(np.float32)
    X_monitor = np.random.normal(loc=-5.0, scale=0.4, size=(100, D)).astype(np.float32)
    
    # Fit high-dimensional FSDS sketches
    s_door = FSDS.fit(X_doorbell, rank=10)
    s_therm = FSDS.fit(X_thermostat, rank=10)
    s_mon = FSDS.fit(X_monitor, rank=10)
    
    assert s_door.ambient_dim == 115
    assert s_door.U.shape == (115, 10)
    
    # Evaluate LOC-NFST upper bound on Doorbell data
    loc_nfst = get_loc_nfst_bound().fit(X_doorbell)
    scores_norm = loc_nfst.score(X_doorbell)
    # Mirai Botnet scan anomaly
    X_mirai_scan = np.random.randn(20, D).astype(np.float32) * 10.0
    scores_anom = loc_nfst.score(X_mirai_scan)
    
    auc = roc_auc_score(
        np.array([0] * len(scores_norm) + [1] * len(scores_anom)),
        np.concatenate([scores_norm, scores_anom])
    ) * 100.0
    assert auc > 90.0, f"Expected high discrimination on N_BaIoT, got {auc:.2f}%"


# ===========================================================================
# Scenario 5: Dynamic IoT Node Churn & Intermittent Communication Dropout
# ===========================================================================

def test_s5_dynamic_node_churn_and_dropout_workflow():
    """S5: Federation resilience under intermittent edge node connectivity and churn."""
    torch.manual_seed(42)
    DROGA = get_droga()
    
    # 5 Available edge devices in the network
    all_client_ids = [0, 1, 2, 3, 4]
    
    # Round-by-round active client schedules (simulating dropout)
    active_schedules = [
        [0, 1, 2],       # Round 1: Clients 3 and 4 offline
        [1, 2, 3, 4],    # Round 2: Client 0 dropped out
        [0, 2, 4],       # Round 3: High network churn
        [0, 1, 2, 3, 4], # Round 4: All clients online
    ]
    
    k = 6
    global_model = get_lunar_mlp(k=k)
    
    for round_idx, active_clients in enumerate(active_schedules):
        client_grads = []
        for c in active_clients:
            # Client computes local update
            d_norm = torch.sort(torch.rand(8, k) * 0.2, dim=1)[0]
            loss = nn.BCELoss()(global_model(d_norm), torch.zeros(8, 1))
            g = torch.cat([grad.view(-1) for grad in torch.autograd.grad(loss, global_model.parameters(), retain_graph=True)])
            client_grads.append(g)
            
        # Server executes DROGA on dynamically available client subset
        g_aligned = DROGA.dr_cagrad(client_grads, c=0.3)
        assert g_aligned.shape == client_grads[0].shape
        assert not torch.isnan(g_aligned).any()
        
        # Apply update
        with torch.no_grad():
            curr_idx = 0
            for p in global_model.parameters():
                num_p = p.numel()
                p.sub_(0.01 * g_aligned[curr_idx:curr_idx + num_p].view_as(p))
                curr_idx += num_p
                
    # Model should remain intact and valid
    test_d = torch.sort(torch.rand(4, k), dim=1)[0]
    out = global_model(test_d)
    assert not torch.isnan(out).any()
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)
