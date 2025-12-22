"""
Generate publication-quality figures from simulation results.

Creates Extended Data figures for Nature Communications submission:
- ED Fig 1: Distance scaling (d=3-13)
- ED Fig 2: Platform comparison (IBM/Google/Rigetti)
- ED Fig 3: Drift model robustness
- ED Fig 4: Time-since-calibration correlation
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# Create output directory
output_dir = Path("simulations/results/figures")
output_dir.mkdir(exist_ok=True, parents=True)

# Load data
df_distance = pd.read_csv("simulations/results/distance_scaling_ibm_v2.csv")
df_platform = pd.read_csv("simulations/results/platform_comparison_d7_v2.csv")
df_drift = pd.read_csv("simulations/results/drift_model_robustness_d7_v2.csv")

print("="*80)
print("GENERATING PUBLICATION FIGURES")
print("="*80)

# ============================================================================
# EXTENDED DATA FIGURE 1: Distance Scaling
# ============================================================================
print("\nGenerating ED Fig 1: Distance scaling...")

fig, ax = plt.subplots(figsize=(8, 6))

# Group by distance
grouped = df_distance.groupby('distance')['improvement_pct']
distances = sorted(df_distance['distance'].unique())

# Calculate statistics
means = [grouped.get_group(d).mean() for d in distances]
stds = [grouped.get_group(d).std() for d in distances]
counts = [len(grouped.get_group(d)) for d in distances]
sems = [s / np.sqrt(c) for s, c in zip(stds, counts)]

# Plot with error bars
ax.errorbar(distances, means, yerr=[1.96 * sem for sem in sems], 
            marker='o', markersize=8, linewidth=2, capsize=5,
            label='DAQEC benefit (95% CI)', color='#2E86AB')

# Add baseline
ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='No benefit')

# Formatting
ax.set_xlabel('Surface Code Distance (d)', fontsize=14, fontweight='bold')
ax.set_ylabel('Drift-Aware Benefit (%)', fontsize=14, fontweight='bold')
ax.set_title('Benefit Persists Across Fault-Tolerance Scales', 
             fontsize=15, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=12, loc='best')
ax.set_xticks(distances)

# Add qubit count annotations
for d, mean in zip(distances, means):
    n_qubits = 2 * d**2 - 1
    ax.annotate(f'{n_qubits} qubits', xy=(d, mean), 
               xytext=(0, 10), textcoords='offset points',
               ha='center', fontsize=9, color='gray')

plt.tight_layout()
plt.savefig(output_dir / "ED_Fig1_distance_scaling.png", dpi=300, bbox_inches='tight')
plt.savefig(output_dir / "ED_Fig1_distance_scaling.pdf", bbox_inches='tight')
print(f"  Saved to {output_dir}/ED_Fig1_distance_scaling.png")
plt.close()

# ============================================================================
# EXTENDED DATA FIGURE 2: Platform Comparison
# ============================================================================
print("\nGenerating ED Fig 2: Platform comparison...")

fig, ax = plt.subplots(figsize=(8, 6))

# Group by platform
platforms = ['ibm', 'google', 'rigetti']
platform_labels = ['IBM Quantum', 'Google Sycamore', 'Rigetti Aspen']
colors = ['#2E86AB', '#A23B72', '#F18F01']

grouped_platform = df_platform.groupby('platform')['improvement_pct']

positions = np.arange(len(platforms))
means_platform = [grouped_platform.get_group(p).mean() for p in platforms]
stds_platform = [grouped_platform.get_group(p).std() for p in platforms]
counts_platform = [len(grouped_platform.get_group(p)) for p in platforms]
sems_platform = [s / np.sqrt(c) for s, c in zip(stds_platform, counts_platform)]

# Bar plot
bars = ax.bar(positions, means_platform, 
              yerr=[1.96 * sem for sem in sems_platform],
              capsize=5, color=colors, edgecolor='black', linewidth=1.5,
              alpha=0.8)

# Formatting
ax.set_ylabel('Drift-Aware Benefit (%)', fontsize=14, fontweight='bold')
ax.set_title('Platform-General Benefit (d=7 Surface Code)', 
             fontsize=15, fontweight='bold', pad=15)
ax.set_xticks(positions)
ax.set_xticklabels(platform_labels, fontsize=12)
ax.grid(True, alpha=0.3, linestyle='--', axis='y')
ax.axhline(0, color='gray', linestyle='--', linewidth=1)

# Add value labels on bars
for bar, mean in zip(bars, means_platform):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 2,
            f'{mean:.1f}%', ha='center', va='bottom', 
            fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / "ED_Fig2_platform_comparison.png", dpi=300, bbox_inches='tight')
plt.savefig(output_dir / "ED_Fig2_platform_comparison.pdf", bbox_inches='tight')
print(f"  Saved to {output_dir}/ED_Fig2_platform_comparison.png")
plt.close()

# ============================================================================
# EXTENDED DATA FIGURE 3: Drift Model Robustness
# ============================================================================
print("\nGenerating ED Fig 3: Drift model robustness...")

fig, ax = plt.subplots(figsize=(8, 6))

# Group by drift model
drift_models = ['gaussian', 'power_law', 'exponential', 'correlated']
drift_labels = ['Gaussian\n(Linear)', 'Power Law\n(1/f noise)', 
                'Exponential\n(Heating)', 'Correlated\n(System-wide)']
colors_drift = ['#2E86AB', '#A23B72', '#F18F01', '#6A994E']

grouped_drift = df_drift.groupby('drift_model')['improvement_pct']

positions = np.arange(len(drift_models))
means_drift = [grouped_drift.get_group(d).mean() for d in drift_models]
stds_drift = [grouped_drift.get_group(d).std() for d in drift_models]
counts_drift = [len(grouped_drift.get_group(d)) for d in drift_models]
sems_drift = [s / np.sqrt(c) for s, c in zip(stds_drift, counts_drift)]

# Bar plot
bars = ax.bar(positions, means_drift,
              yerr=[1.96 * sem for sem in sems_drift],
              capsize=5, color=colors_drift, edgecolor='black', linewidth=1.5,
              alpha=0.8)

# Formatting
ax.set_ylabel('Drift-Aware Benefit (%)', fontsize=14, fontweight='bold')
ax.set_title('Robust to Drift Model Assumptions (d=7, IBM)', 
             fontsize=15, fontweight='bold', pad=15)
ax.set_xticks(positions)
ax.set_xticklabels(drift_labels, fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--', axis='y')
ax.axhline(0, color='gray', linestyle='--', linewidth=1)

# Add value labels
for bar, mean in zip(bars, means_drift):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 1,
            f'{mean:.1f}%', ha='center', va='bottom',
            fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / "ED_Fig3_drift_robustness.png", dpi=300, bbox_inches='tight')
plt.savefig(output_dir / "ED_Fig3_drift_robustness.pdf", bbox_inches='tight')
print(f"  Saved to {output_dir}/ED_Fig3_drift_robustness.png")
plt.close()

# ============================================================================
# EXTENDED DATA FIGURE 4: Time-Since-Calibration Correlation
# ============================================================================
print("\nGenerating ED Fig 4: Time-since-calibration correlation...")

fig, ax = plt.subplots(figsize=(8, 6))

# Use distance scaling data (largest dataset)
time_bins = np.linspace(0, 24, 13)
bin_centers = (time_bins[:-1] + time_bins[1:]) / 2

# Bin data
df_distance['time_bin'] = pd.cut(df_distance['time_since_cal'], bins=time_bins)
binned = df_distance.groupby('time_bin')['improvement_pct']

means_time = [binned.get_group(bin).mean() for bin in binned.groups.keys()]
stds_time = [binned.get_group(bin).std() for bin in binned.groups.keys()]
counts_time = [len(binned.get_group(bin)) for bin in binned.groups.keys()]

# Plot scatter with trend
ax.scatter(df_distance['time_since_cal'], df_distance['improvement_pct'],
           alpha=0.3, s=20, color='#2E86AB', label='Individual sessions')

# Add binned means with error bars
ax.errorbar(bin_centers, means_time, yerr=stds_time,
            fmt='o', markersize=10, linewidth=2, capsize=5,
            color='#A23B72', label='Binned mean ± SD', zorder=10)

# Fit trend line
z = np.polyfit(df_distance['time_since_cal'], df_distance['improvement_pct'], 1)
p = np.poly1d(z)
x_trend = np.linspace(0, 24, 100)
ax.plot(x_trend, p(x_trend), '--', linewidth=2, color='black',
        label=f'Linear fit (slope={z[0]:.2f}%/h)', alpha=0.7)

# Correlation statistics
from scipy.stats import spearmanr
rho, pval = spearmanr(df_distance['time_since_cal'], df_distance['improvement_pct'])
ax.text(0.05, 0.95, f'Spearman ρ = {rho:.3f}\np < {pval:.1e}',
        transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Formatting
ax.set_xlabel('Time Since Calibration (hours)', fontsize=14, fontweight='bold')
ax.set_ylabel('Drift-Aware Benefit (%)', fontsize=14, fontweight='bold')
ax.set_title('Benefit Correlates with Calibration Staleness', 
             fontsize=15, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=10, loc='lower right')
ax.set_xlim(-1, 25)

plt.tight_layout()
plt.savefig(output_dir / "ED_Fig4_time_correlation.png", dpi=300, bbox_inches='tight')
plt.savefig(output_dir / "ED_Fig4_time_correlation.pdf", bbox_inches='tight')
print(f"  Saved to {output_dir}/ED_Fig4_time_correlation.png")
plt.close()

# ============================================================================
# COMBINED SUMMARY FIGURE (for main text)
# ============================================================================
print("\nGenerating main text summary figure...")

fig = plt.figure(figsize=(14, 5))
gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)

# Panel A: Distance scaling (simplified)
ax1 = fig.add_subplot(gs[0, 0])
ax1.errorbar(distances, means, yerr=[1.96 * sem for sem in sems],
             marker='o', markersize=6, linewidth=2, capsize=4,
             color='#2E86AB')
ax1.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax1.set_xlabel('Code Distance', fontsize=12, fontweight='bold')
ax1.set_ylabel('Benefit (%)', fontsize=12, fontweight='bold')
ax1.set_title('A. Distance Scaling', fontsize=13, fontweight='bold', loc='left')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(distances)

# Panel B: Platform comparison
ax2 = fig.add_subplot(gs[0, 1])
ax2.bar(positions, means_platform, color=colors, edgecolor='black', 
        linewidth=1, alpha=0.8)
ax2.set_ylabel('Benefit (%)', fontsize=12, fontweight='bold')
ax2.set_title('B. Platform Generality', fontsize=13, fontweight='bold', loc='left')
ax2.set_xticks(positions)
ax2.set_xticklabels(['IBM', 'Google', 'Rigetti'], fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

# Panel C: Time correlation
ax3 = fig.add_subplot(gs[0, 2])
ax3.scatter(df_distance['time_since_cal'], df_distance['improvement_pct'],
           alpha=0.2, s=10, color='#2E86AB')
ax3.plot(x_trend, p(x_trend), '--', linewidth=2, color='black', alpha=0.7)
ax3.set_xlabel('Time Since Cal (h)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Benefit (%)', fontsize=12, fontweight='bold')
ax3.set_title('C. Staleness Correlation', fontsize=13, fontweight='bold', loc='left')
ax3.grid(True, alpha=0.3)
ax3.text(0.05, 0.95, f'ρ={rho:.2f}***', transform=ax3.transAxes,
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Simulation Validation: Surface Code Fault-Tolerance Scaling',
             fontsize=15, fontweight='bold', y=1.02)

plt.savefig(output_dir / "Main_Fig_Simulation_Summary.png", dpi=300, bbox_inches='tight')
plt.savefig(output_dir / "Main_Fig_Simulation_Summary.pdf", bbox_inches='tight')
print(f"\n  Saved to {output_dir}/Main_Fig_Simulation_Summary.png")
plt.close()

print("\n" + "="*80)
print("FIGURE GENERATION COMPLETE")
print("="*80)
print(f"\nGenerated 5 figures in {output_dir}/")
print("\nExtended Data Figures:")
print("  - ED_Fig1_distance_scaling.png (d=3-13)")
print("  - ED_Fig2_platform_comparison.png (IBM/Google/Rigetti)")
print("  - ED_Fig3_drift_robustness.png (4 drift models)")
print("  - ED_Fig4_time_correlation.png (staleness effect)")
print("\nMain Text Figure:")
print("  - Main_Fig_Simulation_Summary.png (3-panel summary)")
print("\nAll figures saved in PNG (300 DPI) and PDF formats.")
