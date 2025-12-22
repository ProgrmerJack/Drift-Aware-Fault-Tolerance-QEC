"""Generate Extended Data figures from simulation results."""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set publication-quality style
sns.set_context("paper", font_scale=1.2)
sns.set_style("ticks")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'

results_dir = Path("simulations/results")
figures_dir = Path("simulations/figures")
figures_dir.mkdir(parents=True, exist_ok=True)

# Load simulation data
distance_data = pd.read_csv(results_dir / "distance_scaling_ibm_v2.csv")
platform_data = pd.read_csv(results_dir / "platform_comparison_d7_v2.csv")
drift_model_data = pd.read_csv(results_dir / "drift_model_robustness_d7_v2.csv")

with open(results_dir / "summary_statistics_v2.json", 'r') as f:
    stats = json.load(f)

print("="*80)
print("GENERATING EXTENDED DATA FIGURES")
print("="*80)

# ============================================================================
# Extended Data Figure 1: Distance Scaling
# ============================================================================
print("\nGenerating Extended Data Fig 1: Distance Scaling...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: Improvement vs distance
sns.boxplot(data=distance_data, x='distance', y='improvement_pct', ax=ax1, color='skyblue')
ax1.set_xlabel('Surface code distance')
ax1.set_ylabel('Logical error reduction (%)')
ax1.set_title('A. Drift-aware benefit vs code distance')
ax1.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax1.grid(axis='y', alpha=0.3)

# Panel B: Mean improvement with error bars
mean_by_distance = distance_data.groupby('distance')['improvement_pct'].agg(['mean', 'std', 'count'])
distances = mean_by_distance.index
means = mean_by_distance['mean']
stds = mean_by_distance['std']
ns = mean_by_distance['count']
sems = stds / np.sqrt(ns)

ax2.errorbar(distances, means, yerr=1.96*sems, fmt='o-', capsize=5, capthick=2, markersize=8)
ax2.set_xlabel('Surface code distance')
ax2.set_ylabel('Mean logical error reduction (%)')
ax2.set_title('B. Benefit scales with code size')
ax2.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(figures_dir / "extended_data_fig1_distance_scaling.png", bbox_inches='tight')
plt.savefig(figures_dir / "extended_data_fig1_distance_scaling.pdf", bbox_inches='tight')
print(f"  Saved: {figures_dir / 'extended_data_fig1_distance_scaling.png'}")
plt.close()

# ============================================================================
# Extended Data Figure 2: Platform Comparison
# ============================================================================
print("\nGenerating Extended Data Fig 2: Platform Comparison...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: Distribution by platform
sns.violinplot(data=platform_data, x='platform', y='improvement_pct', ax=ax1, palette='Set2')
ax1.set_xlabel('Platform')
ax1.set_ylabel('Logical error reduction (%)')
ax1.set_title('A. Platform-dependent drift characteristics')
ax1.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax1.grid(axis='y', alpha=0.3)

# Panel B: Mean comparison with significance
platform_means = platform_data.groupby('platform')['improvement_pct'].agg(['mean', 'std', 'count'])
platforms = ['IBM', 'Google', 'Rigetti']
platform_codes = ['ibm', 'google', 'rigetti']
means = [platform_means.loc[code, 'mean'] for code in platform_codes]
stds = [platform_means.loc[code, 'std'] for code in platform_codes]
ns = [platform_means.loc[code, 'count'] for code in platform_codes]
sems = [std / np.sqrt(n) for std, n in zip(stds, ns)]

x_pos = np.arange(len(platforms))
ax2.bar(x_pos, means, yerr=[1.96*sem for sem in sems], capsize=5, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax2.set_xticks(x_pos)
ax2.set_xticklabels(platforms)
ax2.set_ylabel('Mean logical error reduction (%)')
ax2.set_title('B. Platform-specific benefit (95% CI)')
ax2.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(figures_dir / "extended_data_fig2_platform_comparison.png", bbox_inches='tight')
plt.savefig(figures_dir / "extended_data_fig2_platform_comparison.pdf", bbox_inches='tight')
print(f"  Saved: {figures_dir / 'extended_data_fig2_platform_comparison.png'}")
plt.close()

# ============================================================================
# Extended Data Figure 3: Drift Model Robustness
# ============================================================================
print("\nGenerating Extended Data Fig 3: Drift Model Robustness...")

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

sns.boxplot(data=drift_model_data, x='drift_model', y='improvement_pct', ax=ax, palette='viridis')
ax.set_xlabel('Drift model')
ax.set_ylabel('Logical error reduction (%)')
ax.set_title('Drift-aware benefit across drift mechanisms')
ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.grid(axis='y', alpha=0.3)

# Annotate sample sizes
for i, model in enumerate(['gaussian', 'power_law', 'exponential', 'correlated']):
    n = len(drift_model_data[drift_model_data['drift_model'] == model])
    ax.text(i, ax.get_ylim()[1], f'n={n}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(figures_dir / "extended_data_fig3_drift_models.png", bbox_inches='tight')
plt.savefig(figures_dir / "extended_data_fig3_drift_models.pdf", bbox_inches='tight')
print(f"  Saved: {figures_dir / 'extended_data_fig3_drift_models.png'}")
plt.close()

# ============================================================================
# Print Summary Statistics for Manuscript
# ============================================================================
print("\n" + "="*80)
print("SUMMARY STATISTICS FOR MANUSCRIPT")
print("="*80)

print("\nDistance Scaling (Study 1):")
for distance in [3, 5, 7, 9, 11, 13]:
    subset = distance_data[distance_data['distance'] == distance]
    mean_imp = subset['improvement_pct'].mean()
    std_imp = subset['improvement_pct'].std()
    print(f"  d={distance:2d}: {mean_imp:5.1f}% ± {std_imp:4.1f}% (n={len(subset)})")

print("\nPlatform Comparison (Study 2):")
for platform, label in [('ibm', 'IBM'), ('google', 'Google'), ('rigetti', 'Rigetti')]:
    subset = platform_data[platform_data['platform'] == platform]
    mean_imp = subset['improvement_pct'].mean()
    std_imp = subset['improvement_pct'].std()
    print(f"  {label:7s}: {mean_imp:5.1f}% ± {std_imp:4.1f}% (n={len(subset)})")

print("\nDrift Model Robustness (Study 3):")
for model, label in [('gaussian', 'Gaussian'), ('power_law', 'Power-law'), 
                      ('exponential', 'Exponential'), ('correlated', 'Correlated')]:
    subset = drift_model_data[drift_model_data['drift_model'] == model]
    mean_imp = subset['improvement_pct'].mean()
    std_imp = subset['improvement_pct'].std()
    print(f"  {label:11s}: {mean_imp:5.1f}% ± {std_imp:4.1f}% (n={len(subset)})")

print("\n" + "="*80)
print("ALL FIGURES GENERATED")
print("="*80)
print(f"\nFigures saved to: {figures_dir}")
print("\nReady for manuscript integration!")
