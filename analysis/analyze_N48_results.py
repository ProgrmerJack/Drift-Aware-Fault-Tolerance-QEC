#!/usr/bin/env python3
"""
Statistical Analysis of N=48 IBM Hardware Results
==================================================

Analyzes complete dataset from scaled async job submission.
"""

import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load results
results_file = Path("results/ibm_experiments/collected_results_20251222_122049.json")
with open(results_file) as f:
    data = json.load(f)

# Extract experimental results
surface_code_results = []
baseline_results = []
daqec_results = []

for job in data["results"]:
    if job["status"] == "completed" and "logical_error_rate" in job:
        ler = job["logical_error_rate"]
        
        if job["experiment_type"] == "surface_code":
            surface_code_results.append({
                "ler": ler,
                "logical_state": job.get("logical_state", "unknown"),
                "rep": job.get("repetition", 0),
                "api_key": job.get("api_key_index", 0),
            })
        elif job["experiment_type"] == "deployment":
            result = {
                "ler": ler,
                "session": job.get("session", 0),
                "api_key": job.get("api_key_index", 0),
            }
            if job.get("mode") == "baseline":
                baseline_results.append(result)
            else:
                daqec_results.append(result)

# Convert to arrays
surface_ler = np.array([r["ler"] for r in surface_code_results])
baseline_ler = np.array([r["ler"] for r in baseline_results])
daqec_ler = np.array([r["ler"] for r in daqec_results])

print("=" * 70)
print("IBM HARDWARE RESULTS: N=48 COMPLETE DATASET")
print("=" * 70)
print()

# Surface Code Analysis
print("SURFACE CODE (Distance-3, 17 qubits, 3 syndrome rounds)")
print("-" * 70)
print(f"Total runs: N={len(surface_ler)}")
print(f"Mean LER: {surface_ler.mean():.4f} ± {surface_ler.std():.4f}")
print(f"Median LER: {np.median(surface_ler):.4f}")
print(f"Range: [{surface_ler.min():.4f}, {surface_ler.max():.4f}]")
print(f"95% CI: [{np.percentile(surface_ler, 2.5):.4f}, {np.percentile(surface_ler, 97.5):.4f}]")
print()

# Breakdown by logical state
for state in ["+", "0"]:
    state_ler = np.array([r["ler"] for r in surface_code_results if r["logical_state"] == state])
    print(f"|{state}⟩ state: N={len(state_ler)}, mean LER={state_ler.mean():.4f} ± {state_ler.std():.4f}")
print()

# Deployment Study Analysis
print("DEPLOYMENT STUDY (Distance-5 repetition code)")
print("-" * 70)
print(f"Baseline sessions: N={len(baseline_ler)}")
print(f"  Mean LER: {baseline_ler.mean():.4f} ± {baseline_ler.std():.4f}")
print(f"  Median: {np.median(baseline_ler):.4f}")
print(f"  Range: [{baseline_ler.min():.4f}, {baseline_ler.max():.4f}]")
print()

print(f"DAQEC sessions: N={len(daqec_ler)}")
print(f"  Mean LER: {daqec_ler.mean():.4f} ± {daqec_ler.std():.4f}")
print(f"  Median: {np.median(daqec_ler):.4f}")
print(f"  Range: [{daqec_ler.min():.4f}, {daqec_ler.max():.4f}]")
print()

# Statistical Tests
print("STATISTICAL TESTS")
print("-" * 70)

# Independent samples t-test
t_stat, p_value = stats.ttest_ind(baseline_ler, daqec_ler)
print(f"Independent t-test:")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.4f}")
print(f"  Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")
print()

# Effect size (Cohen's d)
pooled_std = np.sqrt(((len(baseline_ler) - 1) * baseline_ler.std()**2 + 
                       (len(daqec_ler) - 1) * daqec_ler.std()**2) / 
                      (len(baseline_ler) + len(daqec_ler) - 2))
cohens_d = (baseline_ler.mean() - daqec_ler.mean()) / pooled_std
print(f"Effect size (Cohen's d): {cohens_d:.4f}")
effect_interpretation = "negligible" if abs(cohens_d) < 0.2 else \
                       "small" if abs(cohens_d) < 0.5 else \
                       "medium" if abs(cohens_d) < 0.8 else "large"
print(f"  Interpretation: {effect_interpretation}")
print()

# Relative improvement
rel_improvement = (baseline_ler.mean() - daqec_ler.mean()) / baseline_ler.mean() * 100
print(f"Relative improvement: {rel_improvement:.2f}%")
print()

# Mann-Whitney U test (non-parametric alternative)
u_stat, p_value_mw = stats.mannwhitneyu(baseline_ler, daqec_ler, alternative='two-sided')
print(f"Mann-Whitney U test:")
print(f"  U-statistic: {u_stat:.4f}")
print(f"  p-value: {p_value_mw:.4f}")
print(f"  Significant at α=0.05: {'Yes' if p_value_mw < 0.05 else 'No'}")
print()

# Power analysis (post-hoc)
from statsmodels.stats.power import ttest_power
observed_power = ttest_power(cohens_d, len(baseline_ler), alpha=0.05, alternative='two-sided')
print(f"POST-HOC POWER ANALYSIS")
print(f"-" * 70)
print(f"Observed effect size: d={cohens_d:.4f}")
print(f"Sample size per group: N={len(baseline_ler)}")
print(f"Statistical power: {observed_power:.4f} ({observed_power*100:.1f}%)")
print()

# 95% Confidence intervals
baseline_ci = stats.t.interval(0.95, len(baseline_ler)-1, 
                                loc=baseline_ler.mean(), 
                                scale=stats.sem(baseline_ler))
daqec_ci = stats.t.interval(0.95, len(daqec_ler)-1, 
                             loc=daqec_ler.mean(), 
                             scale=stats.sem(daqec_ler))
print(f"95% Confidence Intervals:")
print(f"  Baseline: [{baseline_ci[0]:.4f}, {baseline_ci[1]:.4f}]")
print(f"  DAQEC: [{daqec_ci[0]:.4f}, {daqec_ci[1]:.4f}]")
print(f"  Intervals {'overlap' if daqec_ci[1] > baseline_ci[0] else 'do not overlap'}")
print()

# Distribution tests
print("NORMALITY TESTS (Shapiro-Wilk)")
print("-" * 70)
w_baseline, p_baseline = stats.shapiro(baseline_ler)
w_daqec, p_daqec = stats.shapiro(daqec_ler)
print(f"Baseline: W={w_baseline:.4f}, p={p_baseline:.4f} {'(normal)' if p_baseline > 0.05 else '(non-normal)'}")
print(f"DAQEC: W={w_daqec:.4f}, p={p_daqec:.4f} {'(normal)' if p_daqec > 0.05 else '(non-normal)'}")
print()

# Summary for manuscript
print("=" * 70)
print("MANUSCRIPT SUMMARY")
print("=" * 70)
print(f"""
Total hardware experiments: N={data['completed']}
- Surface code runs: N={len(surface_ler)} (distance-3, |+⟩ and |0⟩ states)
- Deployment sessions: N={len(baseline_ler) + len(daqec_ler)} (N={len(baseline_ler)} baseline, N={len(daqec_ler)} DAQEC)

Key Result:
DAQEC strategy achieved {rel_improvement:.2f}% relative reduction in logical error rate
compared to baseline (p={p_value:.3f}, Cohen's d={cohens_d:.3f}).

Baseline: {baseline_ler.mean():.4f} ± {baseline_ler.std():.4f}
DAQEC:    {daqec_ler.mean():.4f} ± {daqec_ler.std():.4f}

Statistical significance: {'Yes (p < 0.05)' if p_value < 0.05 else 'No (p ≥ 0.05)'}
Effect size: {effect_interpretation}
Statistical power: {observed_power*100:.1f}%
""")

# Save summary
summary = {
    "total_jobs": data["completed"],
    "surface_code": {
        "n": len(surface_ler),
        "mean_ler": float(surface_ler.mean()),
        "std_ler": float(surface_ler.std()),
        "median_ler": float(np.median(surface_ler)),
        "min_ler": float(surface_ler.min()),
        "max_ler": float(surface_ler.max()),
    },
    "deployment": {
        "baseline": {
            "n": len(baseline_ler),
            "mean_ler": float(baseline_ler.mean()),
            "std_ler": float(baseline_ler.std()),
            "ci_95": [float(baseline_ci[0]), float(baseline_ci[1])],
        },
        "daqec": {
            "n": len(daqec_ler),
            "mean_ler": float(daqec_ler.mean()),
            "std_ler": float(daqec_ler.std()),
            "ci_95": [float(daqec_ci[0]), float(daqec_ci[1])],
        },
        "statistics": {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "cohens_d": float(cohens_d),
            "effect_interpretation": effect_interpretation,
            "relative_improvement_percent": float(rel_improvement),
            "significant_at_0_05": bool(p_value < 0.05),
            "statistical_power": float(observed_power),
        },
    },
}

summary_file = Path("results/ibm_experiments/N48_statistical_summary.json")
with open(summary_file, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nStatistical summary saved to: {summary_file}")
print()
print("=" * 70)
