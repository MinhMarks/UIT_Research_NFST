#!/usr/bin/env python3
"""Master E2E Test Suite Runner for Federated LUNAR.

Executes all 4 test tiers (Tier 1: Features, Tier 2: Boundaries, Tier 3: Pairwise, Tier 4: Applications),
produces an aggregated execution summary table, and exits with code 0 on complete success.

Usage:
    python tests/run_e2e_tests.py
    python tests/run_e2e_tests.py --tier 1
    python tests/run_e2e_tests.py --verbose
"""

from __future__ import annotations
import argparse
import os
import sys
import time
from typing import List, Dict, Any
import pytest


TIER_CONFIG = [
    {
        "tier_id": 1,
        "name": "Tier 1: Feature Coverage (F1 - F14)",
        "path": os.path.join("tests", "e2e", "test_tier1_features.py"),
        "expected_min": 70,
        "description": "Exhaustive functional verification of Features F1 through F14 (>=5 per feature)"
    },
    {
        "tier_id": 2,
        "name": "Tier 2: Boundary & Corner Cases",
        "path": os.path.join("tests", "e2e", "test_tier2_boundaries.py"),
        "expected_min": 25,
        "description": "Stress testing under empty inputs, zero-variance, extreme k-NN, and 1D-500D dimensions"
    },
    {
        "tier_id": 3,
        "name": "Tier 3: Cross-Feature Interactions",
        "path": os.path.join("tests", "e2e", "test_tier3_pairwise.py"),
        "expected_min": 8,
        "description": "Pairwise integration testing (CMNP + DROGA, Dirichlet non-IID + PCGrad, LOC-NFST bound)"
    },
    {
        "tier_id": 4,
        "name": "Tier 4: Real-World Applications",
        "path": os.path.join("tests", "e2e", "test_tier4_applications.py"),
        "expected_min": 5,
        "description": "Realistic end-to-end IoT intrusion workflows across BoTIoT, EdgeIIoTset, CICIoT, N_BaIoT"
    },
]


class PytestSummaryCollector:
    """Custom pytest plugin to collect test counts and statuses."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.errors = 0

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            if report.passed:
                self.passed += 1
            elif report.failed:
                self.failed += 1
            elif report.skipped:
                self.skipped += 1
        elif report.when in ("setup", "teardown") and report.failed:
            self.errors += 1


def run_tier(tier: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    """Runs a single test tier via pytest and captures statistics."""
    collector = PytestSummaryCollector()
    args = [tier["path"], "-q"]
    if verbose:
        args.append("-v")
    else:
        args.append("--tb=short")
        
    t0 = time.perf_counter()
    exit_code = pytest.main(args, plugins=[collector])
    duration = time.perf_counter() - t0
    
    total = collector.passed + collector.failed + collector.skipped + collector.errors
    is_success = (exit_code == 0) and (collector.failed == 0) and (collector.errors == 0)
    
    return {
        "tier_id": tier["tier_id"],
        "name": tier["name"],
        "total": total,
        "passed": collector.passed,
        "failed": collector.failed + collector.errors,
        "skipped": collector.skipped,
        "duration": duration,
        "success": is_success,
        "exit_code": exit_code
    }


def main():
    parser = argparse.ArgumentParser(description="Federated LUNAR E2E Master Test Runner")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3, 4], default=None,
                        help="Run only a specific test tier (1, 2, 3, or 4)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose pytest output")
    args = parser.parse_args()

    # Ensure root is in sys.path
    project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    print("=" * 80)
    print("FEDERATED LUNAR: 4-TIER E2E TEST SUITE RUNNER")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print(f"Python: {sys.version.split()[0]} | Platform: {sys.platform}")
    print("=" * 80)

    selected_tiers = TIER_CONFIG
    if args.tier is not None:
        selected_tiers = [t for t in TIER_CONFIG if t["tier_id"] == args.tier]

    results = []
    all_success = True

    for tier in selected_tiers:
        print(f"\n>> Executing {tier['name']}...")
        print(f"   Target: {tier['path']}")
        res = run_tier(tier, verbose=args.verbose)
        results.append(res)
        if not res["success"]:
            all_success = False

    # Print Summary Table
    print("\n" + "=" * 80)
    print("TEST SUITE EXECUTION SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Tier':<35} | {'Total':<6} | {'Passed':<6} | {'Failed':<6} | {'Time (s)':<8} | {'Status'}")
    print("-" * 80)
    
    total_tests = 0
    total_passed = 0
    total_failed = 0
    total_duration = 0.0

    for r in results:
        status_str = "PASS" if r["success"] else "FAIL"
        print(f"{r['name']:<35} | {r['total']:<6} | {r['passed']:<6} | {r['failed']:<6} | {r['duration']:<8.2f} | {status_str}")
        total_tests += r["total"]
        total_passed += r["passed"]
        total_failed += r["failed"]
        total_duration += r["duration"]

    print("-" * 80)
    final_status = "PASSED" if all_success else "FAILED"
    print(f"{'TOTAL / AGGREGATE':<35} | {total_tests:<6} | {total_passed:<6} | {total_failed:<6} | {total_duration:<8.2f} | {final_status}")
    print("=" * 80)

    if all_success:
        print(f"\n[SUCCESS] All {total_tests} tests in selected tier(s) passed successfully with exit code 0.")
        sys.exit(0)
    else:
        print(f"\n[FAILURE] {total_failed} test(s) failed. Exiting with code 1.")
        sys.exit(1)


if __name__ == "__main__":
    main()
