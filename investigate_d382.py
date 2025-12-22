"""
Investigate Cohen's d = 3.82 claim.
Current calculation: 1.98 (session-level paired)
Stratified values: 0.00 to 3.30 (by backend/distance)

Hypothesis: The 3.82 may be:
1. A weighted average across stratified conditions
2. Computed from a subset (e.g., only d=3)
3. Using a different variance estimator
4. Meta-analytic Cohen's d
"""
import pandas as pd
import numpy as np

df = pd.read_parquet('data/processed/master.parquet')
effect_sizes = pd.read_csv('data/processed/effect_sizes_by_condition.csv')

print("=" * 80)
print("INVESTIGATING COHEN'S D = 3.82 CLAIM")
print("=" * 80)

# Check if focusing on distance=3 only
print("\n### HYPOTHESIS 1: Manuscript focuses on distance=3 only ###")
d3_effects = effect_sizes[effect_sizes['distance'] == 3]
print(d3_effects[['backend', 'cohens_d', 'relative_reduction', 'n_pairs']])
print(f"\nMean Cohen's d for d=3: {d3_effects['cohens_d'].mean():.2f}")
print(f"Weighted mean (by n_pairs): {np.average(d3_effects['cohens_d'], weights=d3_effects['n_pairs']):.2f}")

# Compute overall Cohen's d treating each backend/distance as independent study
print("\n### HYPOTHESIS 2: Meta-analytic pooling ###")
# Fixed-effects meta-analysis: weighted by inverse variance
# Cohen's d variance ~ 1/n + d^2/(2n)
effect_sizes['variance'] = (1 / effect_sizes['n_pairs']) + (effect_sizes['cohens_d']**2 / (2 * effect_sizes['n_pairs']))
effect_sizes['weight'] = 1 / effect_sizes['variance']

# Only non-zero effects
nonzero = effect_sizes[effect_sizes['cohens_d'] > 0]
meta_d = np.sum(nonzero['cohens_d'] * nonzero['weight']) / np.sum(nonzero['weight'])
print(f"Meta-analytic Cohen's d (non-zero effects): {meta_d:.2f}")

# Check distance=3 only meta-analysis
d3_nonzero = d3_effects[d3_effects['cohens_d'] > 0]
if len(d3_nonzero) > 0:
    d3_variance = (1 / d3_nonzero['n_pairs']) + (d3_nonzero['cohens_d']**2 / (2 * d3_nonzero['n_pairs']))
    d3_weight = 1 / d3_variance
    meta_d3 = np.sum(d3_nonzero['cohens_d'] * d3_weight) / np.sum(d3_weight)
    print(f"Meta-analytic Cohen's d (d=3 only, non-zero): {meta_d3:.2f}")

# Compute directly from data for distance=3
print("\n### HYPOTHESIS 3: Direct calculation from d=3 subset ###")
df_d3 = df[df['distance'] == 3]

session_stats = df_d3.groupby(['session_id', 'strategy']).agg({
    'logical_error_rate': 'mean',
    'backend': 'first',
    'day': 'first'
}).reset_index()

session_pivot = session_stats.pivot(
    index=['session_id', 'backend', 'day'],
    columns='strategy',
    values='logical_error_rate'
).reset_index()

strategy_cols = [col for col in session_pivot.columns if col not in ['session_id', 'backend', 'day']]
baseline_col = sorted(strategy_cols)[0]
daqec_col = sorted(strategy_cols)[1]

complete_d3 = session_pivot.dropna(subset=[baseline_col, daqec_col])
complete_d3['difference'] = complete_d3[baseline_col] - complete_d3[daqec_col]

d3_cohens = complete_d3['difference'].mean() / complete_d3['difference'].std(ddof=1)

print(f"Cohen's d for d=3 subset (session-level paired): {d3_cohens:.2f}")
print(f"n={len(complete_d3)} sessions")

# Check manuscript text for context
print("\n### MANUSCRIPT CONTEXT CHECK ###")
print("Manuscript states:")
print("  'Cohen's d = 3.82' in PRIMARY ENDPOINT section")
print("  This may refer specifically to distance-3 results")
print("\nComputed values:")
print(f"  All distances (session-level): d = 1.98")
print(f"  Distance-3 only (session-level): d = {d3_cohens:.2f}")
print(f"  Distance-3 stratified (effect_sizes_by_condition.csv):")
for _, row in d3_effects.iterrows():
    print(f"    {row['backend']}: d = {row['cohens_d']:.2f}")

print("\n### POTENTIAL EXPLANATIONS ###")
if abs(d3_cohens - 3.82) < 0.5:
    print("✓ LIKELY: Manuscript reports distance-3 results specifically")
elif abs(meta_d3 - 3.82) < 0.5:
    print("✓ LIKELY: Manuscript uses meta-analytic pooling for d=3")
elif any(abs(row['cohens_d'] - 3.82) < 0.5 for _, row in d3_effects.iterrows()):
    print("✓ LIKELY: Manuscript reports highest single-backend effect")
else:
    print("⚠ DISCREPANCY: Cannot reproduce d=3.82 with available data")
    print("  Action required: Check original analysis scripts or contact authors")

print("=" * 80)
