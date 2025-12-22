"""
Comprehensive statistical analysis of N=69 paired experimental results.
"""

import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
with open('results/ibm_experiments/collected_results_20251222_124949.json', 'r') as f:
    data = json.load(f)

# Extract deployment results
all_results = data['results']
baseline_jobs = [j for j in all_results if j['mode'] == 'baseline']
daqec_jobs = [j for j in all_results if j['mode'] == 'daqec']

print(f"Baseline jobs: {len(baseline_jobs)}")
print(f"DAQEC jobs: {len(daqec_jobs)}")

# Extract LER values
baseline_ler = np.array([j['logical_error_rate'] for j in baseline_jobs])
daqec_ler = np.array([j['logical_error_rate'] for j in daqec_jobs])

print(f"\n{'='*70}")
print(f"DESCRIPTIVE STATISTICS")
print(f"{'='*70}")
print(f"\nBaseline:")
print(f"  Mean: {np.mean(baseline_ler):.6f}")
print(f"  SD:   {np.std(baseline_ler, ddof=1):.6f}")
print(f"  SE:   {stats.sem(baseline_ler):.6f}")
print(f"  Range: [{np.min(baseline_ler):.6f}, {np.max(baseline_ler):.6f}]")

print(f"\nDAQEC:")
print(f"  Mean: {np.mean(daqec_ler):.6f}")
print(f"  SD:   {np.std(daqec_ler, ddof=1):.6f}")
print(f"  SE:   {stats.sem(daqec_ler):.6f}")
print(f"  Range: [{np.min(daqec_ler):.6f}, {np.max(daqec_ler):.6f}]")

# Improvement
improvement = (np.mean(baseline_ler) - np.mean(daqec_ler)) / np.mean(baseline_ler) * 100
print(f"\nRelative improvement: {improvement:.2f}%")

print(f"\n{'='*70}")
print(f"STATISTICAL TESTS")
print(f"{'='*70}")

# Independent t-test
t_stat, p_value = stats.ttest_ind(baseline_ler, daqec_ler)
print(f"\nIndependent t-test:")
print(f"  t = {t_stat:.4f}")
print(f"  p = {p_value:.4f}")
print(f"  df = {len(baseline_ler) + len(daqec_ler) - 2}")

# Effect size (Cohen's d)
pooled_std = np.sqrt(((len(baseline_ler)-1)*np.var(baseline_ler, ddof=1) + 
                      (len(daqec_ler)-1)*np.var(daqec_ler, ddof=1)) / 
                     (len(baseline_ler) + len(daqec_ler) - 2))
cohens_d = (np.mean(baseline_ler) - np.mean(daqec_ler)) / pooled_std
print(f"\nEffect size (Cohen's d): {cohens_d:.4f}")

# 95% CI for mean difference
mean_diff = np.mean(baseline_ler) - np.mean(daqec_ler)
se_diff = np.sqrt(np.var(baseline_ler, ddof=1)/len(baseline_ler) + 
                  np.var(daqec_ler, ddof=1)/len(daqec_ler))
ci_lower = mean_diff - 1.96 * se_diff
ci_upper = mean_diff + 1.96 * se_diff
print(f"\n95% CI for difference: [{ci_lower:.6f}, {ci_upper:.6f}]")
print(f"CI includes zero: {ci_lower < 0 < ci_upper}")

# Mann-Whitney U test (non-parametric)
u_stat, p_mw = stats.mannwhitneyu(baseline_ler, daqec_ler, alternative='two-sided')
print(f"\nMann-Whitney U test:")
print(f"  U = {u_stat:.1f}")
print(f"  p = {p_mw:.4f}")

# Check for paired structure
print(f"\n{'='*70}")
print(f"PAIRED ANALYSIS (if applicable)")
print(f"{'='*70}")

# Try to pair by API key and session
baseline_by_session = {}
for job in baseline_jobs:
    key = (job['api_key_index'], job['session'])
    baseline_by_session[key] = job['logical_error_rate']

daqec_by_session = {}
for job in daqec_jobs:
    key = (job['api_key_index'], job['session'])
    daqec_by_session[key] = job['logical_error_rate']

# Find matching pairs
pairs = []
for key in baseline_by_session:
    if key in daqec_by_session:
        pairs.append((baseline_by_session[key], daqec_by_session[key]))

if len(pairs) == len(baseline_jobs):
    print(f"\nSuccessfully paired all {len(pairs)} sessions")
    baseline_paired = np.array([p[0] for p in pairs])
    daqec_paired = np.array([p[1] for p in pairs])
    
    # Paired t-test
    t_paired, p_paired = stats.ttest_rel(baseline_paired, daqec_paired)
    print(f"\nPaired t-test:")
    print(f"  t = {t_paired:.4f}")
    print(f"  p = {p_paired:.4f}")
    print(f"  df = {len(pairs)-1}")
    
    # Within-pair correlation
    correlation = np.corrcoef(baseline_paired, daqec_paired)[0, 1]
    print(f"\nWithin-pair correlation: r = {correlation:.4f}")
    
    # Paired effect size
    differences = baseline_paired - daqec_paired
    cohens_d_paired = np.mean(differences) / np.std(differences, ddof=1)
    print(f"Paired Cohen's d: {cohens_d_paired:.4f}")
    
    # 95% CI for paired difference
    se_paired = stats.sem(differences)
    t_crit = stats.t.ppf(0.975, len(differences)-1)
    ci_paired_lower = np.mean(differences) - t_crit * se_paired
    ci_paired_upper = np.mean(differences) + t_crit * se_paired
    print(f"95% CI for paired difference: [{ci_paired_lower:.6f}, {ci_paired_upper:.6f}]")
else:
    print(f"\nWARNING: Could only pair {len(pairs)}/{len(baseline_jobs)} sessions")

print(f"\n{'='*70}")
print(f"INTERPRETATION")
print(f"{'='*70}")

if p_value < 0.05:
    print("\n✓ Result is statistically significant (p < 0.05)")
else:
    print("\n✗ Result is NOT statistically significant (p ≥ 0.05)")

if improvement > 0:
    print(f"✓ DAQEC shows {abs(improvement):.2f}% improvement over baseline")
else:
    print(f"✗ DAQEC shows {abs(improvement):.2f}% WORSE performance than baseline")

print("\n" + "="*70)
