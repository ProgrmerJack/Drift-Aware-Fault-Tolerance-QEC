"""
Comprehensive Analysis of IQM Emerald v4 Validation Results
Combines all v4 result files and performs proper statistical testing.
"""

import os
import json
import glob
from pathlib import Path
import numpy as np
from scipy import stats

def load_all_v4_results():
    """Load all v4 validation results."""
    results_dir = Path(__file__).parent.parent / "results" / "multi_platform"
    files = list(results_dir.glob("iqm_validation_v4_*.json"))
    
    all_runs = []
    metadata = []
    
    for f in sorted(files):
        print(f"Loading: {f.name}")
        with open(f) as fp:
            data = json.load(fp)
        
        metadata.append({
            'file': f.name,
            'timestamp': data.get('timestamp'),
            'n_runs': len(data.get('runs', [])),
            'best_chain': data.get('best_chain'),
            'worst_chain': data.get('worst_chain'),
        })
        
        for run in data.get('runs', []):
            all_runs.append({
                'file': f.name,
                'low_drift_ler': run['low_drift']['ler'],
                'low_calib_ler': run['low_calib']['ler'],
                'high_drift_ler': run['high_drift']['ler'],
                'high_calib_ler': run['high_calib']['ler'],
                'low_effect': run['low_effect'],
                'high_effect': run['high_effect'],
                'interaction': run['interaction'],
            })
    
    return all_runs, metadata


def analyze_results(runs):
    """Perform comprehensive statistical analysis."""
    print("\n" + "="*70)
    print("COMPREHENSIVE STATISTICAL ANALYSIS")
    print(f"Total runs: {len(runs)}")
    print("="*70)
    
    interactions = [r['interaction'] for r in runs]
    low_effects = [r['low_effect'] for r in runs]
    high_effects = [r['high_effect'] for r in runs]
    
    # Descriptive statistics
    print("\n--- DESCRIPTIVE STATISTICS ---")
    print(f"\nInteraction Effect:")
    print(f"  Mean: {np.mean(interactions):+.4f}")
    print(f"  Std:  {np.std(interactions):.4f}")
    print(f"  SE:   {np.std(interactions)/np.sqrt(len(interactions)):.4f}")
    print(f"  Median: {np.median(interactions):+.4f}")
    print(f"  Min/Max: [{np.min(interactions):+.4f}, {np.max(interactions):+.4f}]")
    
    # Direction consistency
    n_negative = sum(1 for i in interactions if i < 0)
    n_positive = sum(1 for i in interactions if i > 0)
    n_zero = sum(1 for i in interactions if i == 0)
    print(f"\nDirection Consistency:")
    print(f"  Negative: {n_negative}/{len(interactions)} ({100*n_negative/len(interactions):.1f}%)")
    print(f"  Positive: {n_positive}/{len(interactions)} ({100*n_positive/len(interactions):.1f}%)")
    print(f"  Zero:     {n_zero}/{len(interactions)}")
    
    # Hypothesis tests
    print("\n--- HYPOTHESIS TESTS ---")
    
    # Primary test: One-tailed t-test for negative interaction
    t_stat, p_value_two = stats.ttest_1samp(interactions, 0)
    p_value_one = p_value_two / 2 if t_stat < 0 else 1 - p_value_two / 2
    
    print(f"\n1. One-sample t-test (H0: interaction = 0)")
    print(f"   t = {t_stat:.3f}, p (one-tailed, <0) = {p_value_one:.4f}")
    
    # Effect size
    cohens_d = np.mean(interactions) / np.std(interactions)
    print(f"   Cohen's d = {cohens_d:.3f}")
    if abs(cohens_d) < 0.2:
        d_interp = "negligible"
    elif abs(cohens_d) < 0.5:
        d_interp = "small"
    elif abs(cohens_d) < 0.8:
        d_interp = "medium"
    else:
        d_interp = "large"
    print(f"   Interpretation: {d_interp} effect")
    
    # Confidence interval
    ci = stats.t.interval(0.95, len(interactions)-1, 
                          loc=np.mean(interactions),
                          scale=stats.sem(interactions))
    print(f"   95% CI: [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    
    # Wilcoxon signed-rank test (non-parametric)
    if len(interactions) >= 5:
        stat, p_wilcox = stats.wilcoxon(interactions, alternative='less')
        print(f"\n2. Wilcoxon signed-rank test (H0: median = 0)")
        print(f"   W = {stat:.1f}, p (one-tailed, <0) = {p_wilcox:.4f}")
    
    # Sign test (binomial)
    sign_result = stats.binomtest(n_negative, n_negative + n_positive, 0.5, alternative='greater')
    p_sign = sign_result.pvalue
    print(f"\n3. Sign test (H0: P(negative) = 0.5)")
    print(f"   p = {p_sign:.4f}")
    
    # Test for LOW effect ≈ 0
    print("\n4. LOW effect test (H0: effect = 0)")
    t_low, p_low = stats.ttest_1samp(low_effects, 0)
    print(f"   Mean: {np.mean(low_effects):+.4f}, t = {t_low:.3f}, p = {p_low:.4f}")
    
    # Test for HIGH effect < 0
    print("\n5. HIGH effect test (H0: effect ≥ 0)")
    t_high, p_high_two = stats.ttest_1samp(high_effects, 0)
    p_high_one = p_high_two / 2 if t_high < 0 else 1 - p_high_two / 2
    print(f"   Mean: {np.mean(high_effects):+.4f}, t = {t_high:.3f}, p (one-tailed) = {p_high_one:.4f}")
    
    # Power analysis
    print("\n--- POWER ANALYSIS ---")
    effect_detected = abs(np.mean(interactions))
    std_obs = np.std(interactions)
    n = len(interactions)
    achieved_power = 1 - stats.t.cdf(stats.t.ppf(0.95, n-1) - effect_detected/std_obs*np.sqrt(n), n-1)
    print(f"Achieved power for detected effect: {achieved_power:.2%}")
    
    # Sample size needed for 80% power at current effect
    if effect_detected > 0:
        n_needed = int(np.ceil((stats.norm.ppf(0.80) + stats.norm.ppf(0.95))**2 * (std_obs/effect_detected)**2))
        print(f"Runs needed for 80% power at effect={effect_detected:.4f}: {n_needed}")
    
    # Overall assessment
    print("\n" + "="*70)
    print("MANUSCRIPT CLAIM ASSESSMENT")
    print("="*70)
    
    # Primary claims
    claims = []
    
    # Claim 1: Interaction is negative
    interaction_ok = np.mean(interactions) < 0 and p_value_one < 0.10
    claims.append(("Negative interaction effect", interaction_ok, np.mean(interactions), p_value_one))
    
    # Claim 2: LOW effect ≈ 0
    low_ok = abs(np.mean(low_effects)) < 0.02
    claims.append(("LOW effect ≈ 0", low_ok, np.mean(low_effects), p_low))
    
    # Claim 3: HIGH effect < 0
    high_ok = np.mean(high_effects) < 0
    claims.append(("HIGH effect < 0", high_ok, np.mean(high_effects), p_high_one))
    
    for claim, ok, value, p in claims:
        status = "✓ SUPPORTED" if ok else "✗ NOT SUPPORTED"
        print(f"\n{claim}:")
        print(f"  Value: {value:+.4f}, p = {p:.4f}")
        print(f"  Status: {status}")
    
    # Overall verdict
    n_supported = sum(1 for c in claims if c[1])
    print("\n" + "-"*50)
    if n_supported == 3:
        print("★★★ ALL MANUSCRIPT CLAIMS SUPPORTED ★★★")
    elif n_supported >= 2:
        print("★★ MAJORITY OF CLAIMS SUPPORTED ★★")
    elif n_supported >= 1:
        print("★ PARTIAL SUPPORT FOR CLAIMS ★")
    else:
        print("⚠ CLAIMS NOT SUPPORTED BY IQM DATA ⚠")
    
    return {
        'n_runs': len(runs),
        'mean_interaction': np.mean(interactions),
        'std_interaction': np.std(interactions),
        'cohens_d': cohens_d,
        't_statistic': t_stat,
        'p_value_one_tailed': p_value_one,
        'ci_95': ci,
        'direction_negative': n_negative / len(interactions),
        'claims_supported': n_supported
    }


def main():
    print("="*70)
    print("IQM EMERALD V4 VALIDATION - COMBINED ANALYSIS")
    print("="*70)
    
    runs, metadata = load_all_v4_results()
    
    print(f"\nLoaded {len(runs)} runs from {len(metadata)} files")
    for m in metadata:
        print(f"  {m['file']}: {m['n_runs']} runs")
        print(f"    Best: {m['best_chain']['data'] if m['best_chain'] else 'N/A'}")
        print(f"    Worst: {m['worst_chain']['data'] if m['worst_chain'] else 'N/A'}")
    
    if len(runs) == 0:
        print("ERROR: No runs found")
        return
    
    results = analyze_results(runs)
    
    # Save analysis
    results_dir = Path(__file__).parent.parent / "results" / "multi_platform"
    with open(results_dir / "iqm_v4_combined_analysis.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nAnalysis saved to: iqm_v4_combined_analysis.json")


if __name__ == '__main__':
    main()
