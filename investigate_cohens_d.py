"""
Deep dive into Cohen's d calculation and aggregation method.
The manuscript claims Cohen's d = 3.82 and 59.9% RRR.
Current calculation shows d = 1.98 and 58.3% RRR.

Investigate:
1. Is there a stratification or weighting issue?
2. Are we computing session-level correctly?
3. Check against daily_summary.csv if available
"""
import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_parquet('data/processed/master.parquet')

print("=" * 80)
print("INVESTIGATING COHEN'S D DISCREPANCY")
print("=" * 80)

# Method 1: Session-level aggregation (current approach)
print("\n### METHOD 1: Session-level aggregation ###")
session_stats = df.groupby(['session_id', 'strategy']).agg({
    'logical_error_rate': 'mean',
    'backend': 'first',
    'day': 'first'
}).reset_index()

session_pivot = session_stats.pivot(
    index=['session_id', 'backend', 'day'],
    columns='strategy',
    values='logical_error_rate'
).reset_index()

baseline_col = 'baseline_static'
daqec_col = 'drift_aware_full_stack'

complete_sessions = session_pivot.dropna(subset=[baseline_col, daqec_col])
complete_sessions['difference'] = complete_sessions[baseline_col] - complete_sessions[daqec_col]

mean_diff = complete_sessions['difference'].mean()
std_diff = complete_sessions['difference'].std(ddof=1)
cohens_d_paired = mean_diff / std_diff

print(f"Session-level paired Cohen's d: {cohens_d_paired:.2f}")
print(f"Mean difference: {mean_diff:.6f}")
print(f"Std of differences: {std_diff:.6f}")

# Method 2: Independent samples Cohen's d (pooled)
baseline_vals = complete_sessions[baseline_col].values
daqec_vals = complete_sessions[daqec_col].values

mean_baseline = baseline_vals.mean()
mean_daqec = daqec_vals.mean()
std_baseline = baseline_vals.std(ddof=1)
std_daqec = daqec_vals.std(ddof=1)

n1 = len(baseline_vals)
n2 = len(daqec_vals)

# Pooled standard deviation
pooled_std = np.sqrt(((n1 - 1) * std_baseline**2 + (n2 - 1) * std_daqec**2) / (n1 + n2 - 2))
cohens_d_pooled = (mean_baseline - mean_daqec) / pooled_std

print(f"\n### METHOD 2: Independent samples (pooled std) ###")
print(f"Baseline mean: {mean_baseline:.6f} ± {std_baseline:.6f}")
print(f"DAQEC mean: {mean_daqec:.6f} ± {std_daqec:.6f}")
print(f"Pooled std: {pooled_std:.6f}")
print(f"Cohen's d (pooled): {cohens_d_pooled:.2f}")

# Method 3: Check if manuscript used RUN-level instead of session-level
print(f"\n### METHOD 3: Run-level analysis (checking for aggregation issue) ###")
baseline_runs = df[df['strategy'] == 'baseline_static']['logical_error_rate'].values
daqec_runs = df[df['strategy'] == 'drift_aware_full_stack']['logical_error_rate'].values

mean_baseline_runs = baseline_runs.mean()
mean_daqec_runs = daqec_runs.mean()
std_baseline_runs = baseline_runs.std(ddof=1)
std_daqec_runs = daqec_runs.std(ddof=1)

pooled_std_runs = np.sqrt(((len(baseline_runs) - 1) * std_baseline_runs**2 + 
                            (len(daqec_runs) - 1) * std_daqec_runs**2) / 
                           (len(baseline_runs) + len(daqec_runs) - 2))
cohens_d_runs = (mean_baseline_runs - mean_daqec_runs) / pooled_std_runs

print(f"Run-level baseline: {mean_baseline_runs:.6f} ± {std_baseline_runs:.6f} (n={len(baseline_runs)})")
print(f"Run-level DAQEC: {mean_daqec_runs:.6f} ± {std_daqec_runs:.6f} (n={len(daqec_runs)})")
print(f"Run-level pooled std: {pooled_std_runs:.6f}")
print(f"Run-level Cohen's d: {cohens_d_runs:.2f}")

# Method 4: Check within-session pairing
print(f"\n### METHOD 4: Within-session paired runs ###")
# For each session, compute within-session Cohen's d
session_cohens_d = []
for session_id in complete_sessions['session_id']:
    session_data = df[df['session_id'] == session_id]
    baseline_session = session_data[session_data['strategy'] == 'baseline_static']['logical_error_rate'].values
    daqec_session = session_data[session_data['strategy'] == 'drift_aware_full_stack']['logical_error_rate'].values
    
    if len(baseline_session) > 0 and len(daqec_session) > 0:
        # Paired difference for this session
        if len(baseline_session) == len(daqec_session):
            diffs = baseline_session - daqec_session
            if diffs.std(ddof=1) > 0:
                d = diffs.mean() / diffs.std(ddof=1)
                session_cohens_d.append(d)

if session_cohens_d:
    print(f"Mean within-session Cohen's d: {np.mean(session_cohens_d):.2f}")
    print(f"Median within-session Cohen's d: {np.median(session_cohens_d):.2f}")
    print(f"Range: [{np.min(session_cohens_d):.2f}, {np.max(session_cohens_d):.2f}]")

# Check daily_summary.csv if it exists
print(f"\n### Checking auxiliary files ###")
try:
    daily = pd.read_csv('data/processed/daily_summary.csv')
    print(f"✓ Found daily_summary.csv with {len(daily)} rows")
    print(f"  Columns: {daily.columns.tolist()}")
    if 'cohen_d' in daily.columns or 'effect_size' in daily.columns:
        print(f"  Contains pre-computed effect sizes!")
except:
    print("  daily_summary.csv not found or not readable")

print("\n" + "=" * 80)
print("HYPOTHESIS")
print("=" * 80)
print("The manuscript's Cohen's d = 3.82 may be computed using:")
print("1. Pooled standard deviation across ALL runs (not session-aggregated)")
print("2. Or using a cluster-robust variance estimator")
print("3. Or computed from a different endpoint (e.g., per-shot level)")
print("")
print("Current best estimate: Cohen's d = 1.98 (session-level paired)")
print("Run-level pooled: Cohen's d = {:.2f}".format(cohens_d_runs))
print("")
print("ACTION: Check analysis scripts or daily_summary.csv for authoritative calculation")
print("=" * 80)
