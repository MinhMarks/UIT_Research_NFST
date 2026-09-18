"""
FL-LOC-NFST Configuration
==========================
Central configuration for all hyperparameters and paths.
Override via environment variables for server deployment.
"""
import os

# ============================================================
# Dataset & Path Configuration
# ============================================================
# Override with DATA_DIR env var on server:
#   DATA_DIR=/path/to/data python run_fl_simulation.py
_SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_DATA = os.path.normpath(
    os.path.join(_SCRIPT_DIR, '..', '..', 'Datascaled', 'Official_OC_Data')
)
DATA_DIR = os.environ.get('DATA_DIR', _DEFAULT_DATA)

# Output directory for FL experiment results
_DEFAULT_OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'federated_results'
)
OUTPUT_DIR = os.environ.get('FL_OUTPUT_DIR', _DEFAULT_OUT)

# ============================================================
# FL Experiment Configuration
# ============================================================

# Datasets to experiment on (prefix names matching Train_<SCALER>_<PREFIX>.csv)
DATASETS = [
    'data_CICIoT2023',
    'data_ToNIoT',
]

# Scalers to try
SCALERS = ['StandardScaler', 'MinMaxScaler']

# Federated Learning config
NUM_CLIENTS = 3          # M — number of simulated clients
NUM_ROUNDS = 1           # T=1 (Protocol A: One-Shot)
K_CLUSTERS = 5           # K — number of pseudo-classes per client

# Non-IID Dirichlet concentration parameter (lower = more heterogeneous)
DIRICHLET_ALPHA = 0.5

# Noise injection percentage (same as centralized baseline)
NOISE_PCT = 1.0

# ============================================================
# Algorithm Hyperparameters
# ============================================================
EPSILON_SVD = 1e-6           # Threshold for near-zero singular values
EPSILON_NEAR_NULL = 1e-4     # Near-null relaxation threshold τ (for S_w null detection)
ALPHA_ORTHO = 0.5            # Weight for orthogonal residual distance
L_MIN = 1                    # Minimum null-space dimensions (fallback protection)

# ============================================================
# Communication Budget Tracking
# ============================================================
# Expected payload per client (d=46, K=5):
#   S_w:       d×d float32 = 46×46×4B = 8464 B
#   centroids: K×d float32 = 5×46×4B  = 920 B
#   counts:    K int32      = 5×4B     = 20 B
#   Total per client:       ≈ 9.4 KB
EXPECTED_PAYLOAD_KB = 9.4

# Random seed for reproducibility
SEED = 42
