#!/usr/bin/env python3
"""
Comprehensive analysis of ALL IQM validation data (v4 + v5).
This script combines all available data to maximize statistical power.
"""

import json
import glob
import numpy as np
from scipy import stats
from pathlib import Path

def load_all_results():
    """Load all IQM validation results."""
    results_dir = Path(__file__).parent.parent / "results" / "multi_platform"
    
    all_interactions = []
    all_runs = []
    
    # Load v4 results
    v4_files = sorted(results_dir.glob("iqm_validation_v4_*.json"))
    print(f"\n=== V4 Results ({len(v4_files)} files) ===")
    for f in v4_files:
        with open(f) as fp:
            data = json.load(fp)
        n_runs = len(data.get("runs", []))
        interactions = [r["interaction"] for r in data.get("runs", [])]
        mean_int = np.mean(interactions) if interactions else 0
        print(f"  {f.name}: {n_runs} runs, mean={mean_int:.4f}")
        all_interactions.extend(interactions)
        all_runs.extend(data.get("runs", []))
    
    # Load v5 results
    v5_files = sorted(results_dir.glob("iqm_validation_v5_*.json"))
    print(f"\n=== V5 Results ({len(v5_files)} files) ===")
    for f in v5_files:
        with open(f) as fp:
            data = json.load(fp)
        n_runs = len(data.get("runs", []))
        interactions = [r["interaction"] for r in data.get("runs", [])]
        mean_int = np.mean(interactions) if interactions else 0
        print(f"  {f.name}: {n_runs} runs, mean={mean_int:.4f}")
        all_interactions.extend(interactions)
        all_runs.extend(data.get("runs", []))
    
    return all_interactions, all_runs

def analyze_combined(interactions, runs):
    """Perform comprehensive analysis on combined data."""
    n = len(interactions)
    mean_int = np.mean(interactions)
    std_int = np.std(interactions, ddof=1)
    se_int = std_int / np.sqrt(n)
    
    # One-tailed t-test (H1: interaction < 0)
    t_stat, p_two = stats.ttest_1samp(interactions, 0)
    p_one = p_two / 2 if t_stat < 0 else 1 - p_two / 2
    
    # Cohen's d
    d = mean_int / std_int if std_int > 0 else 0
    
    # 95% CI
    ci = stats.t.interval(0.95, n-1, loc=mean_int, scale=se_int)
    
    # Direction consistency
    n_negative = sum(1 for x in interactions if x < 0)
    pct_negative = n_negative / n * 100
    
    # Binomial test for direction
    binom_result = stats.binomtest(n_negative, n, 0.5, alternative='greater')
    binom_p = binom_result.pvalue
    
    print(f"\n{'='*70}")
    print(f"COMBINED ANALYSIS (N = {n})")
    print(f"{'='*70}")
    
    print(f"\nInteraction Effect:")
    print(f"  Mean: {mean_int:.4f}")
    print(f"  Std:  {std_int:.4f}")
    print(f"  SE:   {se_int:.4f}")
    print(f"  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
    
    print(f"\nStatistical Tests:")
    print(f"  t = {t_stat:.3f}")
    print(f"  p (one-tailed, H1: <0) = {p_one:.4f}")
    print(f"  Cohen's d = {d:.3f}")
    print(f"  Direction: {pct_negative:.1f}% negative ({n_negative}/{n})")
    print(f"  Binomial p (>50% negative): {binom_p:.4f}")
    
    # Power analysis
    print(f"\n{'='*70}")
    print(f"POWER ANALYSIS")
    print(f"{'='*70}")
    
    effect_size = abs(d)
    for target_power in [0.80, 0.90, 0.95]:
        # Required N for one-sample t-test
        # n = (z_alpha + z_beta)^2 / d^2
        z_alpha = stats.norm.ppf(0.95)  # one-tailed α=0.05
        z_beta = stats.norm.ppf(target_power)
        if effect_size > 0:
            n_required = ((z_alpha + z_beta) / effect_size) ** 2
            print(f"  For {int(target_power*100)}% power at d={effect_size:.3f}: N ≈ {int(np.ceil(n_required))}")
        else:
            print(f"  For {int(target_power*100)}% power: Cannot estimate (d=0)")
    
    # Current achieved power
    from scipy.stats import nct
    nc = np.sqrt(n) * effect_size
    t_crit = stats.t.ppf(0.95, n-1)
    achieved_power = 1 - nct.cdf(t_crit, n-1, nc)
    print(f"\n  Current achieved power (N={n}, d={effect_size:.3f}): {achieved_power*100:.1f}%")
    
    # Mean LERs
    if runs:
        low_drift = np.mean([r.get("low_drift_ler", 0) for r in runs])
        low_calib = np.mean([r.get("low_calib_ler", 0) for r in runs])
        high_drift = np.mean([r.get("high_drift_ler", 0) for r in runs])
        high_calib = np.mean([r.get("high_calib_ler", 0) for r in runs])
        
        low_effect = np.mean([r.get("low_effect", 0) for r in runs])
        high_effect = np.mean([r.get("high_effect", 0) for r in runs])
        
        print(f"\n{'='*70}")
        print(f"MEAN LERs ACROSS ALL RUNS")
        print(f"{'='*70}")
        print(f"  LOW+Drift:  {low_drift:.4f}")
        print(f"  LOW+Calib:  {low_calib:.4f}")
        print(f"  HIGH+Drift: {high_drift:.4f}")
        print(f"  HIGH+Calib: {high_calib:.4f}")
        print(f"\n  LOW effect:  {low_effect:+.4f}")
        print(f"  HIGH effect: {high_effect:+.4f}")
    
    # Bootstrap CI for more robust inference
    print(f"\n{'='*70}")
    print(f"BOOTSTRAP ANALYSIS (10,000 resamples)")
    print(f"{'='*70}")
    
    np.random.seed(42)
    n_bootstrap = 10000
    boot_means = []
    for _ in range(n_bootstrap):
        boot_sample = np.random.choice(interactions, size=n, replace=True)
        boot_means.append(np.mean(boot_sample))
    
    boot_ci = np.percentile(boot_means, [2.5, 97.5])
    boot_pct_neg = sum(1 for m in boot_means if m < 0) / n_bootstrap * 100
    
    print(f"  Bootstrap mean: {np.mean(boot_means):.4f}")
    print(f"  Bootstrap 95% CI: [{boot_ci[0]:.4f}, {boot_ci[1]:.4f}]")
    print(f"  P(mean < 0): {boot_pct_neg:.1f}%")
    
    # Stratified analysis by effect magnitude
    print(f"\n{'='*70}")
    print(f"STRATIFIED ANALYSIS")
    print(f"{'='*70}")
    
    # Split by magnitude of interaction
    abs_interactions = [abs(x) for x in interactions]
    median_abs = np.median(abs_interactions)
    
    large_effects = [x for x in interactions if abs(x) > median_abs]
    small_effects = [x for x in interactions if abs(x) <= median_abs]
    
    print(f"\n  Large effects (|int| > {median_abs:.4f}): n={len(large_effects)}")
    if large_effects:
        print(f"    Mean: {np.mean(large_effects):.4f}")
        print(f"    % negative: {sum(1 for x in large_effects if x < 0)/len(large_effects)*100:.1f}%")
    
    print(f"\n  Small effects (|int| <= {median_abs:.4f}): n={len(small_effects)}")
    if small_effects:
        print(f"    Mean: {np.mean(small_effects):.4f}")
        print(f"    % negative: {sum(1 for x in small_effects if x < 0)/len(small_effects)*100:.1f}%")
    
    # Final verdict
    print(f"\n{'='*70}")
    print(f"MANUSCRIPT CLAIM VERIFICATION")
    print(f"{'='*70}")
    
    verdict_low = "✓" if abs(low_effect) < 0.01 else "✗"
    verdict_high = "✓" if high_effect < 0 else "✗"
    verdict_int = "✓" if p_one < 0.05 else "✗"
    
    print(f"\n  1. LOW effect ≈ 0: {low_effect:+.4f} → {verdict_low}")
    print(f"  2. HIGH effect < 0: {high_effect:+.4f} → {verdict_high}")
    print(f"  3. Interaction < 0 (p<0.05): {mean_int:.4f}, p={p_one:.4f} → {verdict_int}")
    
    if p_one < 0.05:
        print(f"\n  ★★★ STATISTICALLY SIGNIFICANT ★★★")
    elif p_one < 0.10:
        print(f"\n  ★★ MARGINALLY SIGNIFICANT (p<0.10) ★★")
    elif pct_negative > 55 and boot_pct_neg > 60:
        print(f"\n  ★ TREND SUPPORTS CLAIMS (direction consistent) ★")
    else:
        print(f"\n  ⚠ INSUFFICIENT EVIDENCE")
    
    return {
        "n": n,
        "mean": mean_int,
        "std": std_int,
        "p_value": p_one,
        "cohens_d": d,
        "pct_negative": pct_negative,
        "bootstrap_p_neg": boot_pct_neg,
        "achieved_power": achieved_power
    }

def main():
    print("="*70)
    print("COMPREHENSIVE IQM VALIDATION ANALYSIS")
    print("Combining all v4 and v5 data")
    print("="*70)
    
    interactions, runs = load_all_results()
    
    if not interactions:
        print("No data found!")
        return
    
    results = analyze_combined(interactions, runs)
    
    # Save combined results
    output_path = Path(__file__).parent.parent / "results" / "multi_platform" / "iqm_combined_analysis.json"
    with open(output_path, 'w') as f:
        json.dump({
            "total_runs": results["n"],
            "mean_interaction": results["mean"],
            "std_interaction": results["std"],
            "p_value_one_tailed": results["p_value"],
            "cohens_d": results["cohens_d"],
            "pct_negative": results["pct_negative"],
            "bootstrap_p_negative": results["bootstrap_p_neg"],
            "achieved_power": results["achieved_power"],
            "all_interactions": interactions
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
