"""
COMPREHENSIVE HETEROGENEITY INVESTIGATION
==========================================
Deep dive into N=48 vs N=69 datasets to find hidden patterns.

Analysis Strategy:
1. Temporal patterns (early vs late jobs, submission time effects)
2. Session-order effects (performance degradation within batches)
3. Hardware state transitions (change point detection)
4. API key / backend systematic differences
5. Within-pair correlation by subgroups
6. High-error vs low-error period stratification
7. Bootstrap and permutation tests for robustness
8. Circuit pattern sensitivity analysis
9. Shot-level distribution analysis
10. Interaction effects and confounders
"""

import json
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.power import TTestIndPower
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

print("="*80)
print("DEEP HETEROGENEITY INVESTIGATION")
print("="*80)

# Load both datasets
with open('results/ibm_experiments/collected_results_20251222_122049.json', 'r') as f:
    data_N48 = json.load(f)

with open('results/ibm_experiments/collected_results_20251222_124949.json', 'r') as f:
    data_N69 = json.load(f)

# Extract deployment results
deployment_N48 = [j for j in data_N48['results'] if j['experiment_type'] == 'deployment']
deployment_N69 = data_N69['results']

# Create comprehensive dataframe
def create_dataframe(jobs, batch_name):
    """Convert job list to pandas DataFrame with all metadata."""
    df = pd.DataFrame(jobs)
    df['batch'] = batch_name
    df['submission_time'] = pd.to_datetime(df['submission_time'])
    df['completion_time'] = pd.to_datetime(df['completion_time'])
    df['wait_time'] = (df['completion_time'] - df['submission_time']).dt.total_seconds() / 60  # minutes
    df['submission_order'] = range(len(df))
    return df

df_N48 = create_dataframe(deployment_N48, 'N=48')
df_N69 = create_dataframe(deployment_N69, 'N=69')
df_combined = pd.concat([df_N48, df_N69], ignore_index=True)

print(f"\nDatasets loaded:")
print(f"  N=48: {len(df_N48)} jobs ({len(df_N48[df_N48['mode']=='baseline'])} baseline, {len(df_N48[df_N48['mode']=='daqec'])} DAQEC)")
print(f"  N=69: {len(df_N69)} jobs ({len(df_N69[df_N69['mode']=='baseline'])} baseline, {len(df_N69[df_N69['mode']=='daqec'])} DAQEC)")
print(f"  Combined: {len(df_combined)} jobs")

# ============================================================================
# 1. TEMPORAL ANALYSIS: Early vs Late Jobs
# ============================================================================
print(f"\n{'='*80}")
print("1. TEMPORAL ANALYSIS: Early vs Late Jobs")
print("="*80)

for batch_df, batch_name in [(df_N48, 'N=48'), (df_N69, 'N=69')]:
    print(f"\n{batch_name} Batch:")
    
    # Split into quartiles by submission time
    quartiles = pd.qcut(batch_df['submission_order'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    batch_df_temp = batch_df.copy()
    batch_df_temp['quartile'] = quartiles
    
    for mode in ['baseline', 'daqec']:
        mode_df = batch_df_temp[batch_df_temp['mode'] == mode]
        print(f"\n  {mode.upper()} by Quartile:")
        for q in ['Q1', 'Q2', 'Q3', 'Q4']:
            q_data = mode_df[mode_df['quartile'] == q]['logical_error_rate']
            print(f"    {q}: {q_data.mean():.6f} ± {q_data.std():.6f} (N={len(q_data)})")
    
    # Test for linear trend across quartiles
    for mode in ['baseline', 'daqec']:
        mode_df = batch_df_temp[batch_df_temp['mode'] == mode]
        quartile_means = []
        for q in ['Q1', 'Q2', 'Q3', 'Q4']:
            q_mean = mode_df[mode_df['quartile'] == q]['logical_error_rate'].mean()
            quartile_means.append(q_mean)
        
        # Linear regression: LER ~ quartile number
        from scipy.stats import linregress
        x = [1, 2, 3, 4]
        slope, intercept, r_value, p_value, std_err = linregress(x, quartile_means)
        print(f"\n  {mode.upper()} Linear Trend:")
        print(f"    Slope: {slope:.6f} per quartile")
        print(f"    R²: {r_value**2:.4f}")
        print(f"    p-value: {p_value:.4f}")
        if p_value < 0.05:
            print(f"    *** SIGNIFICANT temporal trend detected!")

# ============================================================================
# 2. SESSION-ORDER EFFECTS
# ============================================================================
print(f"\n{'='*80}")
print("2. SESSION-ORDER EFFECTS (N=69 only - has session metadata)")
print("="*80)

# N=69 has session metadata (0-22 per API key)
for api_key in df_N69['api_key_index'].unique():
    api_df = df_N69[df_N69['api_key_index'] == api_key]
    print(f"\nAPI Key {api_key}:")
    
    for mode in ['baseline', 'daqec']:
        mode_df = api_df[api_df['mode'] == mode].sort_values('session')
        
        # Early (sessions 0-7) vs Late (sessions 15-22)
        early = mode_df[mode_df['session'] <= 7]['logical_error_rate']
        late = mode_df[mode_df['session'] >= 15]['logical_error_rate']
        
        print(f"\n  {mode.upper()}:")
        print(f"    Early sessions (0-7):   {early.mean():.6f} ± {early.std():.6f} (N={len(early)})")
        print(f"    Late sessions (15-22):  {late.mean():.6f} ± {late.std():.6f} (N={len(late)})")
        
        if len(early) > 0 and len(late) > 0:
            t_stat, p_val = stats.ttest_ind(early, late)
            print(f"    t-test: t={t_stat:.4f}, p={p_val:.4f}")
            if p_val < 0.05:
                print(f"    *** SIGNIFICANT session-order effect!")

# ============================================================================
# 3. HARDWARE STATE CHANGE POINT DETECTION
# ============================================================================
print(f"\n{'='*80}")
print("3. HARDWARE STATE CHANGE POINT DETECTION")
print("="*80)

# Check if there's a specific time when error rates jumped
print("\nTesting for change point between N=48 and N=69 batches:")
print(f"N=48 last job:  {df_N48['completion_time'].max()}")
print(f"N=69 first job: {df_N69['submission_time'].min()}")
time_gap = (df_N69['submission_time'].min() - df_N48['completion_time'].max()).total_seconds() / 60
print(f"Time gap: {time_gap:.1f} minutes")

# Compare error rates before/after
baseline_N48_ler = df_N48[df_N48['mode']=='baseline']['logical_error_rate'].mean()
baseline_N69_ler = df_N69[df_N69['mode']=='baseline']['logical_error_rate'].mean()
jump = (baseline_N69_ler - baseline_N48_ler) / baseline_N48_ler * 100

print(f"\nBaseline LER jump: {jump:+.1f}%")
print(f"  Before (N=48): {baseline_N48_ler:.6f}")
print(f"  After (N=69):  {baseline_N69_ler:.6f}")

# Statistical test for change point
all_baseline = pd.concat([
    df_N48[df_N48['mode']=='baseline']['logical_error_rate'],
    df_N69[df_N69['mode']=='baseline']['logical_error_rate']
])
all_baseline_batch = ['before']*len(df_N48[df_N48['mode']=='baseline']) + ['after']*len(df_N69[df_N69['mode']=='baseline'])

before_vals = df_N48[df_N48['mode']=='baseline']['logical_error_rate']
after_vals = df_N69[df_N69['mode']=='baseline']['logical_error_rate']
t_stat, p_val = stats.ttest_ind(before_vals, after_vals)
print(f"\nChange point t-test: t={t_stat:.4f}, p={p_val:.4f}")
if p_val < 0.001:
    print("*** HIGHLY SIGNIFICANT hardware state change detected!")

# ============================================================================
# 4. WITHIN-PAIR CORRELATION BY SUBGROUPS
# ============================================================================
print(f"\n{'='*80}")
print("4. WITHIN-PAIR CORRELATION ANALYSIS")
print("="*80)

# N=69 has paired structure
pairs_N69 = []
for api_key in df_N69['api_key_index'].unique():
    for session in df_N69[df_N69['api_key_index']==api_key]['session'].unique():
        baseline_val = df_N69[(df_N69['api_key_index']==api_key) & 
                               (df_N69['session']==session) & 
                               (df_N69['mode']=='baseline')]['logical_error_rate'].values
        daqec_val = df_N69[(df_N69['api_key_index']==api_key) & 
                            (df_N69['session']==session) & 
                            (df_N69['mode']=='daqec')]['logical_error_rate'].values
        if len(baseline_val) > 0 and len(daqec_val) > 0:
            pairs_N69.append({
                'api_key': api_key,
                'session': session,
                'baseline': baseline_val[0],
                'daqec': daqec_val[0],
                'diff': baseline_val[0] - daqec_val[0]
            })

df_pairs = pd.DataFrame(pairs_N69)

# Overall correlation
overall_corr = df_pairs['baseline'].corr(df_pairs['daqec'])
print(f"\nOverall within-pair correlation: r={overall_corr:.4f}")

# By API key
print("\nCorrelation by API Key:")
for api_key in df_pairs['api_key'].unique():
    api_pairs = df_pairs[df_pairs['api_key']==api_key]
    corr = api_pairs['baseline'].corr(api_pairs['daqec'])
    print(f"  API Key {api_key}: r={corr:.4f} (N={len(api_pairs)} pairs)")

# By session phase (early/middle/late)
df_pairs['phase'] = pd.cut(df_pairs['session'], bins=3, labels=['Early', 'Middle', 'Late'])
print("\nCorrelation by Session Phase:")
for phase in ['Early', 'Middle', 'Late']:
    phase_pairs = df_pairs[df_pairs['phase']==phase]
    if len(phase_pairs) > 2:
        corr = phase_pairs['baseline'].corr(phase_pairs['daqec'])
        print(f"  {phase}: r={corr:.4f} (N={len(phase_pairs)} pairs)")

# ============================================================================
# 5. HIGH-ERROR vs LOW-ERROR STRATIFICATION
# ============================================================================
print(f"\n{'='*80}")
print("5. HIGH-ERROR vs LOW-ERROR PERIOD STRATIFICATION")
print("="*80)

# Identify high vs low error periods using baseline as reference
median_baseline_N69 = df_N69[df_N69['mode']=='baseline']['logical_error_rate'].median()
print(f"\nMedian baseline LER (N=69): {median_baseline_N69:.6f}")

# Classify pairs as high/low error
df_pairs['error_stratum'] = df_pairs['baseline'].apply(
    lambda x: 'High' if x > median_baseline_N69 else 'Low'
)

print("\nDAQEC Performance by Error Stratum:")
for stratum in ['Low', 'High']:
    stratum_pairs = df_pairs[df_pairs['error_stratum']==stratum]
    baseline_mean = stratum_pairs['baseline'].mean()
    daqec_mean = stratum_pairs['daqec'].mean()
    improvement = (baseline_mean - daqec_mean) / baseline_mean * 100
    
    print(f"\n  {stratum} Error Period (N={len(stratum_pairs)} pairs):")
    print(f"    Baseline: {baseline_mean:.6f}")
    print(f"    DAQEC:    {daqec_mean:.6f}")
    print(f"    Improvement: {improvement:+.2f}%")
    
    # Paired t-test within stratum
    t_stat, p_val = stats.ttest_rel(stratum_pairs['baseline'], stratum_pairs['daqec'])
    print(f"    Paired t-test: t={t_stat:.4f}, p={p_val:.4f}")
    if p_val < 0.05:
        print(f"    *** SIGNIFICANT effect in {stratum} error stratum!")

# Test for interaction: Does DAQEC benefit depend on baseline error rate?
print("\nInteraction Test: DAQEC benefit vs Baseline LER")
from scipy.stats import pearsonr
corr, p_val = pearsonr(df_pairs['baseline'], df_pairs['diff'])
print(f"  Correlation(baseline, improvement): r={corr:.4f}, p={p_val:.4f}")
if abs(corr) > 0.2 and p_val < 0.05:
    print(f"  *** SIGNIFICANT interaction: DAQEC benefit depends on baseline error rate!")

# ============================================================================
# 6. BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================================
print(f"\n{'='*80}")
print("6. BOOTSTRAP ANALYSIS FOR ROBUSTNESS")
print("="*80)

def bootstrap_mean_diff(baseline, daqec, n_bootstrap=10000):
    """Bootstrap confidence interval for mean difference."""
    diffs = []
    n = len(baseline)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        diff = baseline[idx].mean() - daqec[idx].mean()
        diffs.append(diff)
    return np.array(diffs)

print("\nBootstrapping mean differences (10,000 iterations):")

for batch_df, batch_name in [(df_N48, 'N=48'), (df_N69, 'N=69')]:
    baseline_vals = batch_df[batch_df['mode']=='baseline']['logical_error_rate'].values
    daqec_vals = batch_df[batch_df['mode']=='daqec']['logical_error_rate'].values
    
    boot_diffs = bootstrap_mean_diff(baseline_vals, daqec_vals)
    ci_lower = np.percentile(boot_diffs, 2.5)
    ci_upper = np.percentile(boot_diffs, 97.5)
    mean_diff = boot_diffs.mean()
    
    print(f"\n  {batch_name}:")
    print(f"    Bootstrap mean difference: {mean_diff:.6f}")
    print(f"    95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
    print(f"    CI includes zero: {ci_lower < 0 < ci_upper}")
    
    # Probability that DAQEC is better
    prob_better = (boot_diffs > 0).mean()
    print(f"    P(DAQEC better than baseline): {prob_better:.3f}")

# ============================================================================
# 7. PERMUTATION TEST FOR SIGNIFICANCE
# ============================================================================
print(f"\n{'='*80}")
print("7. PERMUTATION TEST FOR ROBUSTNESS")
print("="*80)

def permutation_test(baseline, daqec, n_perm=10000):
    """Permutation test for mean difference."""
    observed_diff = baseline.mean() - daqec.mean()
    combined = np.concatenate([baseline, daqec])
    n = len(baseline)
    
    perm_diffs = []
    for _ in range(n_perm):
        np.random.shuffle(combined)
        perm_diff = combined[:n].mean() - combined[n:].mean()
        perm_diffs.append(perm_diff)
    
    p_value = (np.abs(perm_diffs) >= np.abs(observed_diff)).mean()
    return observed_diff, p_value, np.array(perm_diffs)

print("\nPermutation tests (10,000 permutations):")

for batch_df, batch_name in [(df_N48, 'N=48'), (df_N69, 'N=69')]:
    baseline_vals = batch_df[batch_df['mode']=='baseline']['logical_error_rate'].values
    daqec_vals = batch_df[batch_df['mode']=='daqec']['logical_error_rate'].values
    
    obs_diff, p_val, perm_diffs = permutation_test(baseline_vals, daqec_vals)
    
    print(f"\n  {batch_name}:")
    print(f"    Observed difference: {obs_diff:.6f}")
    print(f"    Permutation p-value: {p_val:.4f}")
    if p_val < 0.05:
        print(f"    *** SIGNIFICANT by permutation test!")

print("\n" + "="*80)
print("INVESTIGATION COMPLETE")
print("="*80)
print("\nKey findings will be visualized next...")
