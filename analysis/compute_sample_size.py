from statsmodels.stats.power import TTestIndPower, TTestPower
import math

# Observed effect size
d = 0.2176244149581216
alpha = 0.05
power = 0.80

ind = TTestIndPower()
ndef = ind.solve_power(effect_size=d, power=power, alpha=alpha, alternative='two-sided')

# Paired with r = 0.5 and r = 0.8
r1 = 0.5
rp1 = d / ((2*(1-r1))**0.5)
paired = TTestPower()
npaired1 = paired.solve_power(effect_size=rp1, power=power, alpha=alpha, alternative='two-sided')

r2 = 0.8
rp2 = d / ((2*(1-r2))**0.5)
npaired2 = paired.solve_power(effect_size=rp2, power=power, alpha=alpha, alternative='two-sided')

print(f"Observed Cohen's d: {d:.4f}")
print(f"Required n per group (independent t-test) for 80% power: {ndef:.1f} (~{math.ceil(ndef)} per group)")
print(f"Paired (r=0.5) -> d_paired={rp1:.4f} => n (pairs): {npaired1:.1f} (~{math.ceil(npaired1)})")
print(f"Paired (r=0.8) -> d_paired={rp2:.4f} => n (pairs): {npaired2:.1f} (~{math.ceil(npaired2)})")

# Translate to jobs and 10-min windows
jobs_per_key_per_window = 16  # observed throughput from previous run
keys = 3
jobs_per_window_total = jobs_per_key_per_window * keys

# independent total jobs
n_ind_group = math.ceil(ndef)
total_jobs_ind = n_ind_group * 2
windows_ind = math.ceil(total_jobs_ind / jobs_per_window_total)
minutes_ind = windows_ind * 10

# paired total jobs (pairs means both conditions per pair)
n_paired_05 = math.ceil(npaired1)
total_jobs_paired = n_paired_05 * 2
windows_paired = math.ceil(total_jobs_paired / jobs_per_window_total)
minutes_paired = windows_paired * 10

print()
print(f"Assuming {keys} keys × {jobs_per_key_per_window} jobs per key per 10-min window -> {jobs_per_window_total} jobs per 10-min window total.")
print(f"Independent test: need ~{total_jobs_ind} jobs total → {windows_ind} windows → {minutes_ind} minutes")
print(f"Paired (r=0.5): need ~{total_jobs_paired} jobs total → {windows_paired} windows → {minutes_paired} minutes")

# Also print for optimistic case: r=0.8
npaired2c = math.ceil(npaired2)
total_jobs_paired2 = npaired2c * 2
windows_paired2 = math.ceil(total_jobs_paired2 / jobs_per_window_total)
minutes_paired2 = windows_paired2 * 10
print(f"Paired (r=0.8): need ~{total_jobs_paired2} jobs total → {windows_paired2} windows → {minutes_paired2} minutes")
