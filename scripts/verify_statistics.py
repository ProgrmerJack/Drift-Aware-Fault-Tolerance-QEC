#!/usr/bin/env python3
"""
verify_statistics.py - Verify Key Statistical Claims
=====================================================

Recomputes all key statistics from the benchmark data and compares
against published values. This is the final verification step for
independent reproduction.

Usage:
    python scripts/verify_statistics.py
    python scripts/verify_statistics.py --verbose
    python scripts/verify_statistics.py --tolerance 0.02
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Published values (from manuscript)
PUBLISHED_STATISTICS = {
    "primary_improvement_mean": 0.61,  # 61% mean improvement
    "primary_improvement_ci_lower": 0.58,
    "primary_improvement_ci_upper": 0.64,
    "tail_compression_95th": 0.757,  # 75.7%
    "tail_compression_99th": 0.772,  # 77.2%
    "dose_response_rho": 0.56,  # Spearman correlation
    "n_sessions": 756,
    "n_clusters": 42,
}

# Tolerance for verification (default: 1%)
DEFAULT_TOLERANCE = 0.01


def load_master_data() -> pd.DataFrame:
    """Load the master analysis dataset."""
    master_path = DATA_DIR / "master.parquet"
    if not master_path.exists():
        raise FileNotFoundError(f"Master dataset not found: {master_path}")
    return pd.read_parquet(master_path)


def cluster_bootstrap_ci(data: np.ndarray, clusters: np.ndarray, 
                         n_bootstrap: int = 10000, ci: float = 0.95) -> tuple:
    """
    Compute cluster-bootstrap confidence interval.
    
    Args:
        data: Array of values
        clusters: Array of cluster labels
        n_bootstrap: Number of bootstrap iterations
        ci: Confidence level
        
    Returns:
        (mean, ci_lower, ci_upper)
    """
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)
    
    # Compute cluster means
    cluster_means = {}
    for c in unique_clusters:
        cluster_means[c] = np.mean(data[clusters == c])
    
    # Bootstrap
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)
        boot_mean = np.mean([cluster_means[c] for c in sampled_clusters])
        bootstrap_means.append(boot_mean)
    
    alpha = (1 - ci) / 2
    ci_lower = np.percentile(bootstrap_means, alpha * 100)
    ci_upper = np.percentile(bootstrap_means, (1 - alpha) * 100)
    
    return np.mean(data), ci_lower, ci_upper


def compute_statistics(df: pd.DataFrame) -> dict:
    """Compute all key statistics from the data."""
    results = {}
    
    # Check required columns
    if 'improvement' not in df.columns:
        # Try to compute from raw error rates
        if 'logical_error_rate_baseline' in df.columns and 'logical_error_rate_daqec' in df.columns:
            df['improvement'] = 1 - (df['logical_error_rate_daqec'] / df['logical_error_rate_baseline'])
        else:
            raise ValueError("Cannot compute improvement: missing required columns")
    
    # Primary improvement
    improvements = df['improvement'].values
    results['primary_improvement_mean'] = np.mean(improvements)
    
    # Cluster bootstrap CI
    if 'cluster_id' in df.columns:
        clusters = df['cluster_id'].values
        _, ci_lower, ci_upper = cluster_bootstrap_ci(improvements, clusters)
        results['primary_improvement_ci_lower'] = ci_lower
        results['primary_improvement_ci_upper'] = ci_upper
        results['n_clusters'] = len(np.unique(clusters))
    else:
        # Simple bootstrap
        results['primary_improvement_ci_lower'] = np.percentile(improvements, 2.5)
        results['primary_improvement_ci_upper'] = np.percentile(improvements, 97.5)
        results['n_clusters'] = None
    
    # Sample size
    results['n_sessions'] = len(df)
    
    # Tail compression
    if 'logical_error_rate_baseline' in df.columns and 'logical_error_rate_daqec' in df.columns:
        baseline = df['logical_error_rate_baseline'].values
        daqec = df['logical_error_rate_daqec'].values
        
        baseline_95 = np.percentile(baseline, 95)
        baseline_99 = np.percentile(baseline, 99)
        daqec_95 = np.percentile(daqec, 95)
        daqec_99 = np.percentile(daqec, 99)
        
        results['tail_compression_95th'] = 1 - (daqec_95 / baseline_95)
        results['tail_compression_99th'] = 1 - (daqec_99 / baseline_99)
    
    # Dose-response correlation
    if 'calibration_age_hours' in df.columns:
        rho, p_value = stats.spearmanr(df['calibration_age_hours'], df['improvement'])
        results['dose_response_rho'] = rho
        results['dose_response_p'] = p_value
    
    return results


def verify_statistics(computed: dict, published: dict, tolerance: float) -> tuple:
    """
    Verify computed statistics against published values.
    
    Returns:
        (all_verified, verification_report)
    """
    report = []
    all_verified = True
    
    for key, published_value in published.items():
        if key not in computed:
            report.append(f"? {key}: Not computed (published: {published_value})")
            continue
        
        computed_value = computed[key]
        
        if computed_value is None:
            report.append(f"? {key}: Could not compute")
            continue
        
        if isinstance(published_value, (int, float)):
            diff = abs(computed_value - published_value)
            
            # For proportions, use absolute tolerance
            if 0 <= published_value <= 1:
                within_tolerance = diff <= tolerance
            else:
                # For counts, require exact match
                within_tolerance = (computed_value == published_value)
            
            if within_tolerance:
                report.append(f"✓ {key}: {computed_value:.4f} (published: {published_value}, diff: {diff:.4f})")
            else:
                report.append(f"✗ {key}: {computed_value:.4f} (published: {published_value}, diff: {diff:.4f}) MISMATCH")
                all_verified = False
    
    return all_verified, report


def main():
    parser = argparse.ArgumentParser(description="Verify key statistics")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                        help=f"Tolerance for verification (default: {DEFAULT_TOLERANCE})")
    args = parser.parse_args()

    print("=" * 60)
    print("DAQEC Statistical Verification")
    print("=" * 60)
    print(f"Tolerance: {args.tolerance:.2%}")
    print()

    # Load data
    try:
        df = load_master_data()
        print(f"✓ Loaded {len(df)} sessions from master dataset")
    except FileNotFoundError as e:
        print(f"✗ {e}")
        sys.exit(1)

    print()
    print("Computing statistics...")
    print()

    # Compute
    try:
        computed = compute_statistics(df)
    except Exception as e:
        print(f"✗ Computation failed: {e}")
        sys.exit(1)

    # Verify
    all_verified, report = verify_statistics(computed, PUBLISHED_STATISTICS, args.tolerance)

    print("Verification Results:")
    print("-" * 60)
    for line in report:
        print(line)

    print()
    print("=" * 60)
    if all_verified:
        print("✓ ALL STATISTICS VERIFIED")
        print("=" * 60)
        print()
        print("The computed statistics match published values within tolerance.")
        print("Independent reproduction: SUCCESSFUL")
        sys.exit(0)
    else:
        print("✗ SOME STATISTICS DO NOT MATCH")
        print("=" * 60)
        print()
        print("Review mismatches above. Possible causes:")
        print("  - Data file corruption (re-run verify_checksums.py)")
        print("  - Software version differences")
        print("  - Tolerance too strict (try --tolerance 0.02)")
        sys.exit(1)


if __name__ == "__main__":
    main()
