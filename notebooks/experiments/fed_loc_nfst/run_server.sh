#!/usr/bin/env bash
# ============================================================
# FL-LOC-NFST Server Setup & Run Script
# Usage (on postmaster.iec):
#   1. Clone/pull the repo (feature/federated-loc-nfst branch)
#   2. chmod +x run_server.sh
#   3. DATA_DIR=/path/to/your/data ./run_server.sh
# ============================================================

set -e

# ── Configuration ──────────────────────────────────────────
REPO_DIR="${REPO_DIR:-$HOME/UIT_Research_NFST}"
DATA_DIR="${DATA_DIR:-/path/to/Datascaled/Official_OC_Data}"  # OVERRIDE THIS!
BRANCH="feature/federated-loc-nfst"
EXPERIMENTS_DIR="$REPO_DIR/notebooks/experiments"
FL_DIR="$EXPERIMENTS_DIR/fed_loc_nfst"

# ── Colors ─────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN} FL-LOC-NFST Server Experiment Runner   ${NC}"
echo -e "${GREEN}========================================${NC}"

# ── Step 1: Git setup ──────────────────────────────────────
if [ -d "$REPO_DIR/.git" ]; then
    echo -e "${YELLOW}Updating existing repo...${NC}"
    cd "$REPO_DIR"
    git fetch origin
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
else
    echo -e "${YELLOW}Cloning repo...${NC}"
    git clone -b "$BRANCH" https://github.com/MinhMarks/UIT_Research_NFST.git "$REPO_DIR"
    cd "$REPO_DIR"
fi

echo -e "${GREEN}✓ Repo ready at: $REPO_DIR (branch: $BRANCH)${NC}"

# ── Step 2: Install dependencies ───────────────────────────
echo -e "${YELLOW}Installing Python dependencies...${NC}"
pip install -q flwr>=1.5.0 scikit-learn scipy numpy pandas matplotlib seaborn pytest

echo -e "${GREEN}✓ Dependencies installed${NC}"

# ── Step 3: Verify DATA_DIR ────────────────────────────────
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}ERROR: DATA_DIR=$DATA_DIR does not exist!${NC}"
    echo "Please set DATA_DIR to the path containing Train_*.csv files"
    echo "Example: DATA_DIR=/data/OC_Data ./run_server.sh"
    exit 1
fi

echo -e "${GREEN}✓ DATA_DIR: $DATA_DIR${NC}"
ls "$DATA_DIR" | head -5
echo "..."

# ── Step 4: Run unit tests ─────────────────────────────────
echo -e "${YELLOW}Running unit tests...${NC}"
cd "$EXPERIMENTS_DIR"
python -m pytest fed_loc_nfst/tests/ -v --tb=short
echo -e "${GREEN}✓ Tests passed${NC}"

# ── Step 5: Run centralized baseline ──────────────────────
echo -e "${YELLOW}Running centralized baseline (all datasets)...${NC}"
DATA_DIR="$DATA_DIR" FL_OUTPUT_DIR="$REPO_DIR/notebooks/experiments/outputs/federated_results" \
    python fed_loc_nfst/run_centralized.py \
        --all-datasets \
        --clusters 5 \
        --noise 1.0

echo -e "${GREEN}✓ Centralized baseline done${NC}"

# ── Step 6: Run FL simulation (IID) ───────────────────────
echo -e "${YELLOW}Running FL simulation (IID partition)...${NC}"
DATA_DIR="$DATA_DIR" FL_OUTPUT_DIR="$REPO_DIR/notebooks/experiments/outputs/federated_results" \
    python fed_loc_nfst/run_fl_simulation.py \
        --all-datasets \
        --clients 3 \
        --rounds 1 \
        --clusters 5 \
        --partition iid \
        --noise 1.0

echo -e "${GREEN}✓ FL simulation (IID) done${NC}"

# ── Step 7: Run FL simulation (Non-IID Dirichlet) ─────────
echo -e "${YELLOW}Running FL simulation (Non-IID Dirichlet α=0.5)...${NC}"
DATA_DIR="$DATA_DIR" FL_OUTPUT_DIR="$REPO_DIR/notebooks/experiments/outputs/federated_results" \
    python fed_loc_nfst/run_fl_simulation.py \
        --all-datasets \
        --clients 3 \
        --rounds 1 \
        --clusters 5 \
        --partition dirichlet \
        --noise 1.0

echo -e "${GREEN}✓ FL simulation (Non-IID) done${NC}"

# ── Step 8: Compare and plot ───────────────────────────────
OUT_DIR="$REPO_DIR/notebooks/experiments/outputs/federated_results"
FL_CSV=$(ls "$OUT_DIR"/fl_all_results_*.csv 2>/dev/null | sort | tail -1)
CENTRAL_CSV=$(ls "$OUT_DIR"/centralized_results_*.csv 2>/dev/null | sort | tail -1)

if [ -n "$FL_CSV" ] && [ -n "$CENTRAL_CSV" ]; then
    echo -e "${YELLOW}Generating comparison plots...${NC}"
    DATA_DIR="$DATA_DIR" FL_OUTPUT_DIR="$OUT_DIR" \
        python fed_loc_nfst/compare_results.py \
            --fl-csv "$FL_CSV" \
            --central-csv "$CENTRAL_CSV"
    echo -e "${GREEN}✓ Plots generated in $OUT_DIR/plots/${NC}"
else
    echo -e "${YELLOW}No result files found for comparison (check if experiments completed)${NC}"
fi

# ── Step 9: Commit results ─────────────────────────────────
echo -e "${YELLOW}Committing results to git...${NC}"
cd "$REPO_DIR"
git add notebooks/experiments/outputs/federated_results/ 2>/dev/null || true
git commit -m "results: FL-LOC-NFST experiments on server $(date +%Y-%m-%d)" 2>/dev/null || \
    echo "Nothing to commit (results may already be committed)"

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  FL-LOC-NFST Experiments COMPLETE!         ${NC}"
echo -e "${GREEN}  Results: $OUT_DIR                          ${NC}"
echo -e "${GREEN}============================================${NC}"
