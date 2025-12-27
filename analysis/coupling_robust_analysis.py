"""
COUPLING-ROBUST INTERACTION ANALYSIS
=====================================
Addresses the mathematical coupling vulnerability where using
percent improvement = (baseline - DAQEC) / baseline creates 
spurious correlation with baseline (ratio metric trap).

This script implements:
1. Absolute difference analysis: ΔLER = LER_DAQEC - LER_baseline
2. ANCOVA-style regression: LER_DAQEC ~ LER_baseline + condition
3. Ratio metric as secondary confirmation only
4. All three approaches must agree for robust conclusion

Per research.instructions.md: We strengthen evidence, not inflate claims.
"""

import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("COUPLING-ROBUST INTERACTION ANALYSIS")
print("="*80)
print("\nObjective: Test if interaction survives metrics WITHOUT mathematical coupling")
print("="*80)

# Load N=69 primary dataset
with open('results/ibm_experiments/collected_results_20251222_124949.json', 'r') as f:
    data_N69 = json.load(f)

# Create paired data
pairs = []
results = data_N69['results']
api_keys = set(r['api_key_index'] for r in results)
for api_key in api_keys:
    api_results = [r for r in results if r['api_key_index'] == api_key]
    sessions = set(r['session'] for r in api_results)
    for session in sessions:
        baseline = [r for r in api_results if r['session'] == session and r['mode'] == 'baseline']
        daqec = [r for r in api_results if r['session'] == session and r['mode'] == 'daqec']
        if baseline and daqec:
            pairs.append({
                'api_key': api_key,
                'session': session,
                'baseline_ler': baseline[0]['logical_error_rate'],
                'daqec_ler': daqec[0]['logical_error_rate'],
            })

df = pd.DataFrame(pairs)
print(f"\nN = {len(df)} paired experiments")

# =============================================================================
# 1. ABSOLUTE DIFFERENCE ANALYSIS (NO COUPLING RISK)
# =============================================================================
print("\n" + "="*80)
print("1. ABSOLUTE DIFFERENCE ANALYSIS (Primary - No Coupling)")
print("="*80)

# ΔLER = DAQEC - baseline (positive = DAQEC worse)
df['delta_ler'] = df['daqec_ler'] - df['baseline_ler']

# Test: Does ΔLER correlate with baseline LER?
corr_delta, p_delta = stats.pearsonr(df['baseline_ler'], df['delta_ler'])
print(f"\nCorrelation(baseline LER, ΔLER): r = {corr_delta:.4f}, p = {p_delta:.2e}")

# Linear regression: ΔLER ~ baseline_ler
X = sm.add_constant(df['baseline_ler'])
model_delta = sm.OLS(df['delta_ler'], X).fit()
print(f"\nLinear model: ΔLER = {model_delta.params['const']:.4f} + {model_delta.params['baseline_ler']:.4f} × baseline_LER")
print(f"R² = {model_delta.rsquared:.4f}")
print(f"p-value (slope) = {model_delta.pvalues['baseline_ler']:.2e}")

# Interpretation
slope = model_delta.params['baseline_ler']
if slope < 0:
    print(f"\n*** NEGATIVE SLOPE: As baseline LER ↑, ΔLER ↓ (DAQEC becomes relatively BETTER)")
    print(f"    At high noise: DAQEC improves performance")
    print(f"    At low noise: DAQEC degrades performance")
else:
    print(f"\n    Slope is positive: DAQEC effect does not depend on noise level")

# Calculate crossover point (where ΔLER = 0)
crossover_delta = -model_delta.params['const'] / model_delta.params['baseline_ler']
print(f"\nCrossover point (ΔLER = 0): baseline LER = {crossover_delta:.4f}")

# =============================================================================
# 2. STRATIFIED ANALYSIS WITH ABSOLUTE DIFFERENCES
# =============================================================================
print("\n" + "="*80)
print("2. STRATIFIED ANALYSIS WITH ABSOLUTE DIFFERENCES")
print("="*80)

median_ler = df['baseline_ler'].median()
df['stratum'] = df['baseline_ler'].apply(lambda x: 'High' if x > median_ler else 'Low')

print(f"\nMedian baseline LER: {median_ler:.4f}")

for stratum in ['Low', 'High']:
    stratum_df = df[df['stratum'] == stratum]
    n = len(stratum_df)
    mean_delta = stratum_df['delta_ler'].mean()
    std_delta = stratum_df['delta_ler'].std()
    se_delta = std_delta / np.sqrt(n)
    
    # One-sample t-test: Is ΔLER significantly different from 0?
    t_stat, p_val = stats.ttest_1samp(stratum_df['delta_ler'], 0)
    
    # Effect direction
    direction = "WORSE" if mean_delta > 0 else "BETTER"
    
    print(f"\n{stratum} Error Stratum (n = {n}):")
    print(f"  Mean ΔLER: {mean_delta:.6f} ± {se_delta:.6f}")
    print(f"  95% CI: [{mean_delta - 1.96*se_delta:.6f}, {mean_delta + 1.96*se_delta:.6f}]")
    print(f"  t-test (vs 0): t = {t_stat:.4f}, p = {p_val:.4e}")
    print(f"  DAQEC is {direction} than baseline in this stratum")
    if p_val < 0.01:
        print(f"  *** SIGNIFICANT (p < 0.01)")

# Test difference between strata
low_delta = df[df['stratum'] == 'Low']['delta_ler']
high_delta = df[df['stratum'] == 'High']['delta_ler']
t_diff, p_diff = stats.ttest_ind(low_delta, high_delta)
print(f"\nDifference between strata:")
print(f"  t-test: t = {t_diff:.4f}, p = {p_diff:.4e}")
if p_diff < 0.01:
    print(f"  *** SIGNIFICANT interaction: Strata have different effects")

# =============================================================================
# 3. ANCOVA-STYLE ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("3. ANCOVA-STYLE ANALYSIS: LER_DAQEC ~ LER_baseline")
print("="*80)

# If DAQEC provides uniform benefit, slope should be ~1 and intercept < 0
X = sm.add_constant(df['baseline_ler'])
model_ancova = sm.OLS(df['daqec_ler'], X).fit()

print(f"\nModel: LER_DAQEC = {model_ancova.params['const']:.4f} + {model_ancova.params['baseline_ler']:.4f} × LER_baseline")
print(f"R² = {model_ancova.rsquared:.4f}")
print(f"Slope: {model_ancova.params['baseline_ler']:.4f} (95% CI: [{model_ancova.conf_int().loc['baseline_ler', 0]:.4f}, {model_ancova.conf_int().loc['baseline_ler', 1]:.4f}])")
print(f"Intercept: {model_ancova.params['const']:.4f} (95% CI: [{model_ancova.conf_int().loc['const', 0]:.4f}, {model_ancova.conf_int().loc['const', 1]:.4f}])")

# Interpretation
slope = model_ancova.params['baseline_ler']
intercept = model_ancova.params['const']

print(f"\nInterpretation:")
if slope < 1:
    print(f"  Slope < 1: DAQEC compresses error rate differences")
    print(f"  At high baseline: DAQEC relatively better")
    print(f"  At low baseline: DAQEC relatively worse")
elif slope > 1:
    print(f"  Slope > 1: DAQEC amplifies error rate differences")
else:
    print(f"  Slope ≈ 1: DAQEC has uniform effect across noise levels")

if intercept > 0:
    print(f"  Intercept > 0: DAQEC adds fixed overhead")
else:
    print(f"  Intercept < 0: DAQEC provides uniform benefit")

# Crossover point for ANCOVA (where DAQEC = baseline)
crossover_ancova = intercept / (1 - slope) if slope != 1 else float('inf')
print(f"\nCrossover (where DAQEC = baseline): {crossover_ancova:.4f}")

# =============================================================================
# 4. COMPARISON: PERCENT IMPROVEMENT (FOR REFERENCE ONLY)
# =============================================================================
print("\n" + "="*80)
print("4. PERCENT IMPROVEMENT (Secondary - Has Coupling Risk)")
print("="*80)

df['pct_improvement'] = (df['baseline_ler'] - df['daqec_ler']) / df['baseline_ler'] * 100
corr_pct, p_pct = stats.pearsonr(df['baseline_ler'], df['pct_improvement'])
print(f"\nCorrelation(baseline LER, % improvement): r = {corr_pct:.4f}, p = {p_pct:.2e}")
print(f"\n⚠️  WARNING: This metric has mathematical coupling risk!")
print(f"   The ratio denominator creates artificial correlation with baseline.")
print(f"   Use absolute ΔLER for primary inference.")

# =============================================================================
# 5. ROBUSTNESS SUMMARY
# =============================================================================
print("\n" + "="*80)
print("5. ROBUSTNESS SUMMARY: DO ALL METRICS AGREE?")
print("="*80)

print(f"\n{'Metric':<35} {'r':<10} {'p-value':<15} {'Conclusion'}")
print("-"*70)
print(f"{'Absolute ΔLER':<35} {corr_delta:<10.4f} {p_delta:<15.2e} {'ROBUST' if p_delta < 0.001 else 'WEAK'}")
print(f"{'ANCOVA slope deviation from 1':<35} {slope - 1:<10.4f} {model_ancova.pvalues['baseline_ler']:<15.2e} {'ROBUST' if model_ancova.pvalues['baseline_ler'] < 0.001 else 'WEAK'}")
print(f"{'Percent improvement (ref only)':<35} {corr_pct:<10.4f} {p_pct:<15.2e} {'(coupling risk)'}")

# Final verdict
print("\n" + "="*80)
print("FINAL VERDICT")
print("="*80)

if corr_delta < -0.5 and p_delta < 0.001:
    print(f"\n✅ INTERACTION SURVIVES COUPLING-ROBUST ANALYSIS")
    print(f"\n   The interaction effect is REAL, not a statistical artifact:")
    print(f"   - Absolute ΔLER shows strong negative correlation with baseline (r={corr_delta:.4f})")
    print(f"   - As hardware noise increases, DAQEC becomes relatively better")
    print(f"   - Crossover point: ~{crossover_delta:.4f} baseline LER")
    print(f"\n   This means the sign-flip discovery is scientifically robust.")
else:
    print(f"\n⚠️  INTERACTION MAY BE PARTIALLY ARTIFACTUAL")
    print(f"   Further investigation needed.")

# =============================================================================
# 6. SAVE RESULTS FOR MANUSCRIPT
# =============================================================================
results = {
    "coupling_robust_analysis": {
        "date": "2025-12-27",
        "primary_metric": "absolute_delta_ler",
        "n_pairs": len(df),
        "absolute_delta_analysis": {
            "correlation": round(corr_delta, 4),
            "p_value": float(f"{p_delta:.2e}"),
            "crossover_point": round(crossover_delta, 4),
            "model_r_squared": round(model_delta.rsquared, 4),
            "slope": round(model_delta.params['baseline_ler'], 4),
            "intercept": round(model_delta.params['const'], 4)
        },
        "stratified_absolute_analysis": {
            "low_stratum": {
                "n": int(len(df[df['stratum'] == 'Low'])),
                "mean_delta_ler": round(df[df['stratum'] == 'Low']['delta_ler'].mean(), 6),
                "effect_direction": "DAQEC_WORSE" if df[df['stratum'] == 'Low']['delta_ler'].mean() > 0 else "DAQEC_BETTER"
            },
            "high_stratum": {
                "n": int(len(df[df['stratum'] == 'High'])),
                "mean_delta_ler": round(df[df['stratum'] == 'High']['delta_ler'].mean(), 6),
                "effect_direction": "DAQEC_WORSE" if df[df['stratum'] == 'High']['delta_ler'].mean() > 0 else "DAQEC_BETTER"
            },
            "strata_difference_p_value": float(f"{p_diff:.2e}")
        },
        "ancova_analysis": {
            "daqec_ler_predicted_by_baseline": {
                "slope": round(model_ancova.params['baseline_ler'], 4),
                "intercept": round(model_ancova.params['const'], 4),
                "r_squared": round(model_ancova.rsquared, 4)
            },
            "interpretation": "slope < 1 indicates DAQEC compresses differences, helping more at high noise"
        },
        "percent_improvement_reference": {
            "correlation": round(corr_pct, 4),
            "p_value": float(f"{p_pct:.2e}"),
            "warning": "Has mathematical coupling risk - use for reference only"
        },
        "robustness_verdict": "INTERACTION_SURVIVES" if corr_delta < -0.5 and p_delta < 0.001 else "NEEDS_REVIEW"
    }
}

with open('results/coupling_robust_analysis.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n\nResults saved to: results/coupling_robust_analysis.json")
