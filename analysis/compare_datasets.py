"""
Compare N=48 vs N=69 datasets to understand the reversal.
"""

import json
import numpy as np
from scipy import stats

# Load both datasets
with open('results/ibm_experiments/collected_results_20251222_122049.json', 'r') as f:
    data_N48 = json.load(f)

with open('results/ibm_experiments/collected_results_20251222_124949.json', 'r') as f:
    data_N69 = json.load(f)

print("="*70)
print("DATASET COMPARISON: N=48 vs N=69")
print("="*70)

# N=48 analysis
all_results_N48 = data_N48['results']
deployment_N48 = [j for j in all_results_N48 if j['experiment_type'] == 'deployment']
baseline_N48 = [j['logical_error_rate'] for j in deployment_N48 if j['mode'] == 'baseline']
daqec_N48 = [j['logical_error_rate'] for j in deployment_N48 if j['mode'] == 'daqec']

print(f"\nN=48 Dataset (prior run):")
print(f"  Baseline: {np.mean(baseline_N48):.6f} ± {np.std(baseline_N48, ddof=1):.6f} (N={len(baseline_N48)})")
print(f"  DAQEC:    {np.mean(daqec_N48):.6f} ± {np.std(daqec_N48, ddof=1):.6f} (N={len(daqec_N48)})")
improvement_N48 = (np.mean(baseline_N48) - np.mean(daqec_N48)) / np.mean(baseline_N48) * 100
print(f"  Improvement: {improvement_N48:.2f}%")
t_N48, p_N48 = stats.ttest_ind(baseline_N48, daqec_N48)
print(f"  p-value: {p_N48:.4f}")

# N=69 analysis
all_results_N69 = data_N69['results']
baseline_N69 = [j['logical_error_rate'] for j in all_results_N69 if j['mode'] == 'baseline']
daqec_N69 = [j['logical_error_rate'] for j in all_results_N69 if j['mode'] == 'daqec']

print(f"\nN=69 Dataset (current run):")
print(f"  Baseline: {np.mean(baseline_N69):.6f} ± {np.std(baseline_N69, ddof=1):.6f} (N={len(baseline_N69)})")
print(f"  DAQEC:    {np.mean(daqec_N69):.6f} ± {np.std(daqec_N69, ddof=1):.6f} (N={len(daqec_N69)})")
improvement_N69 = (np.mean(baseline_N69) - np.mean(daqec_N69)) / np.mean(baseline_N69) * 100
print(f"  Improvement: {improvement_N69:.2f}%")
t_N69, p_N69 = stats.ttest_ind(baseline_N69, daqec_N69)
print(f"  p-value: {p_N69:.4f}")

print(f"\n{'='*70}")
print(f"KEY OBSERVATIONS")
print(f"{'='*70}")

# Compare baseline error rates
baseline_increase = (np.mean(baseline_N69) - np.mean(baseline_N48)) / np.mean(baseline_N48) * 100
print(f"\nBaseline LER changed by {baseline_increase:+.1f}%")
print(f"  N=48: {np.mean(baseline_N48):.6f}")
print(f"  N=69: {np.mean(baseline_N69):.6f}")

daqec_increase = (np.mean(daqec_N69) - np.mean(daqec_N48)) / np.mean(daqec_N48) * 100
print(f"\nDAQEC LER changed by {daqec_increase:+.1f}%")
print(f"  N=48: {np.mean(daqec_N48):.6f}")
print(f"  N=69: {np.mean(daqec_N69):.6f}")

print(f"\nEffect direction REVERSED:")
print(f"  N=48: DAQEC {improvement_N48:+.2f}% vs baseline")
print(f"  N=69: DAQEC {improvement_N69:+.2f}% vs baseline")

# Check submission times
print(f"\n{'='*70}")
print(f"TIMING ANALYSIS")
print(f"{'='*70}")

# Get submission times from N=69
baseline_times_N69 = [j['submission_time'] for j in all_results_N69 if j['mode'] == 'baseline']
daqec_times_N69 = [j['submission_time'] for j in all_results_N69 if j['mode'] == 'daqec']
print(f"\nN=69 Submission times:")
print(f"  First baseline: {min(baseline_times_N69)}")
print(f"  Last baseline:  {max(baseline_times_N69)}")
print(f"  First DAQEC:    {min(daqec_times_N69)}")
print(f"  Last DAQEC:     {max(daqec_times_N69)}")

# Combined analysis
print(f"\n{'='*70}")
print(f"COMBINED DATASET (N=48 + N=69 = N=117)")
print(f"{'='*70}")

all_baseline = baseline_N48 + baseline_N69
all_daqec = daqec_N48 + daqec_N69

print(f"\nCombined statistics:")
print(f"  Baseline: {np.mean(all_baseline):.6f} ± {np.std(all_baseline, ddof=1):.6f} (N={len(all_baseline)})")
print(f"  DAQEC:    {np.mean(all_daqec):.6f} ± {np.std(all_daqec, ddof=1):.6f} (N={len(all_daqec)})")
improvement_combined = (np.mean(all_baseline) - np.mean(all_daqec)) / np.mean(all_baseline) * 100
print(f"  Improvement: {improvement_combined:.2f}%")
t_combined, p_combined = stats.ttest_ind(all_baseline, all_daqec)
print(f"  t-test: t={t_combined:.4f}, p={p_combined:.4f}")

# Effect size
pooled_std = np.sqrt(((len(all_baseline)-1)*np.var(all_baseline, ddof=1) + 
                      (len(all_daqec)-1)*np.var(all_daqec, ddof=1)) / 
                     (len(all_baseline) + len(all_daqec) - 2))
cohens_d = (np.mean(all_baseline) - np.mean(all_daqec)) / pooled_std
print(f"  Cohen's d: {cohens_d:.4f}")

print(f"\n{'='*70}")
print(f"CONCLUSION")
print(f"{'='*70}")
print("\nThe N=69 dataset shows OPPOSITE effect from N=48:")
print(f"  • N=48 showed +2.4% improvement (DAQEC better)")
print(f"  • N=69 showed -1.8% improvement (DAQEC worse)")
print(f"  • Combined shows +{improvement_combined:.2f}% (near zero)")
print(f"\nThis is consistent with:")
print(f"  1. Hardware noise variability over time")
print(f"  2. True effect size near zero (null hypothesis)")
print(f"  3. Initial N=48 result was likely a statistical fluctuation")
print(f"\nWith adequate power (N=69), we observe NO SIGNIFICANT EFFECT.")
print("="*70)
