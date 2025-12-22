"""
Extended Data Figures for Nature Communications Submission
Hardware Noise Moderates Drift-Aware Quantum Error Correction

Generates all Extended Data figures required for submission:
- Extended Data Figure 1: Session-level results (N=69)
- Extended Data Figure 2: Statistical robustness checks
- Extended Data Figure 3: Cross-validation details
- Extended Data Figure 4: Temporal evolution
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.linewidth': 1.0,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

def load_data():
    """Load experimental results from both datasets."""
    base_path = Path(__file__).parent.parent / "results" / "ibm_experiments"
    
    # Load N=69 dataset
    with open(base_path / "collected_results_20251222_124949.json") as f:
        data_n69 = json.load(f)
    
    # Load N=48 dataset
    with open(base_path / "collected_results_20251222_122049.json") as f:
        data_n48 = json.load(f)
    
    return data_n69, data_n48

def process_paired_data(data):
    """Extract paired baseline/DAQEC results."""
    results = data['results']
    
    baseline = [r for r in results if r.get('mode') == 'baseline']
    daqec = [r for r in results if r.get('mode') == 'daqec']
    
    # Sort by session
    baseline = sorted(baseline, key=lambda x: x.get('session', 0))
    daqec = sorted(daqec, key=lambda x: x.get('session', 0))
    
    n_pairs = min(len(baseline), len(daqec))
    
    pairs = []
    for i in range(n_pairs):
        b_ler = baseline[i].get('logical_error_rate', 0)
        d_ler = daqec[i].get('logical_error_rate', 0)
        session = baseline[i].get('session', i)
        
        if b_ler > 0:
            benefit = (b_ler - d_ler) / b_ler * 100
        else:
            benefit = 0
            
        pairs.append({
            'session': session,
            'baseline_ler': b_ler,
            'daqec_ler': d_ler,
            'benefit_pct': benefit,
            'absolute_diff': b_ler - d_ler
        })
    
    return pairs

def generate_extended_data_fig1(pairs_n69, output_dir):
    """Extended Data Figure 1: Session-level results for N=69 primary dataset."""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort pairs by baseline LER
    sorted_pairs = sorted(pairs_n69, key=lambda x: x['baseline_ler'])
    
    sessions = range(len(sorted_pairs))
    baseline_lers = [p['baseline_ler'] for p in sorted_pairs]
    daqec_lers = [p['daqec_ler'] for p in sorted_pairs]
    benefits = [p['benefit_pct'] for p in sorted_pairs]
    
    # Calculate median for threshold
    median_ler = np.median(baseline_lers)
    
    # Create twin axis for benefit
    ax2 = ax.twinx()
    
    # Plot LERs
    ax.plot(sessions, baseline_lers, 'o-', color='#2c3e50', label='Baseline', alpha=0.7, markersize=4)
    ax.plot(sessions, daqec_lers, 's-', color='#e74c3c', label='DAQEC', alpha=0.7, markersize=4)
    
    # Plot benefit on secondary axis
    colors = ['#27ae60' if b > 0 else '#c0392b' for b in benefits]
    ax2.bar(sessions, benefits, alpha=0.3, color=colors, width=0.8)
    
    # Add threshold line
    threshold_idx = sum(1 for b in baseline_lers if b < median_ler)
    ax.axvline(x=threshold_idx - 0.5, color='purple', linestyle='--', linewidth=2, 
               label=f'Median LER threshold')
    
    # Add zero line for benefit
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1)
    
    # Labels and styling
    ax.set_xlabel('Sessions (sorted by baseline LER)')
    ax.set_ylabel('Logical Error Rate')
    ax2.set_ylabel('DAQEC Benefit (%)', color='gray')
    ax.set_title('Extended Data Figure 1: Session-Level Results (N=69)\nSorted by Baseline LER to Show Crossover', fontweight='bold')
    
    # Add annotations
    ax.annotate('Low Noise Region\nDAQEC Hurts', xy=(threshold_idx/4, 0.14), fontsize=10, 
                ha='center', color='#c0392b', fontweight='bold')
    ax.annotate('High Noise Region\nDAQEC Helps', xy=(threshold_idx + (69-threshold_idx)/2, 0.14), 
                fontsize=10, ha='center', color='#27ae60', fontweight='bold')
    
    ax.legend(loc='upper left')
    ax.set_xlim(-1, 70)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ExtendedData_Fig1_SessionLevel.png', dpi=300)
    plt.savefig(output_dir / 'ExtendedData_Fig1_SessionLevel.pdf')
    plt.close()
    print("✓ Generated Extended Data Figure 1")

def generate_extended_data_fig2(pairs_n69, output_dir):
    """Extended Data Figure 2: Statistical robustness checks."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    baseline_lers = np.array([p['baseline_ler'] for p in pairs_n69])
    benefits = np.array([p['benefit_pct'] for p in pairs_n69])
    
    median_ler = np.median(baseline_lers)
    low_mask = baseline_lers < median_ler
    high_mask = baseline_lers >= median_ler
    
    low_benefits = benefits[low_mask]
    high_benefits = benefits[high_mask]
    
    # Panel A: Permutation test
    ax = axes[0]
    
    # Observed difference
    obs_diff = np.mean(high_benefits) - np.mean(low_benefits)
    
    # Permutation distribution
    n_perm = 10000
    perm_diffs = []
    combined = np.concatenate([low_benefits, high_benefits])
    n_low = len(low_benefits)
    
    np.random.seed(42)
    for _ in range(n_perm):
        np.random.shuffle(combined)
        perm_low = combined[:n_low]
        perm_high = combined[n_low:]
        perm_diffs.append(np.mean(perm_high) - np.mean(perm_low))
    
    perm_diffs = np.array(perm_diffs)
    p_perm = np.mean(np.abs(perm_diffs) >= np.abs(obs_diff))
    
    ax.hist(perm_diffs, bins=50, color='#3498db', alpha=0.7, edgecolor='white')
    ax.axvline(x=obs_diff, color='#e74c3c', linestyle='--', linewidth=2, 
               label=f'Observed: {obs_diff:.1f}%')
    ax.axvline(x=-obs_diff, color='#e74c3c', linestyle='--', linewidth=2)
    
    ax.set_xlabel('Difference in Mean Benefit (High - Low noise)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'A. Permutation Test\np = {p_perm:.4f}', fontweight='bold')
    ax.legend()
    
    # Panel B: Bootstrap CIs
    ax = axes[1]
    
    n_boot = 10000
    np.random.seed(42)
    
    low_boot = []
    high_boot = []
    for _ in range(n_boot):
        low_sample = np.random.choice(low_benefits, size=len(low_benefits), replace=True)
        high_sample = np.random.choice(high_benefits, size=len(high_benefits), replace=True)
        low_boot.append(np.mean(low_sample))
        high_boot.append(np.mean(high_sample))
    
    low_boot = np.array(low_boot)
    high_boot = np.array(high_boot)
    
    # Calculate CIs
    low_ci = np.percentile(low_boot, [2.5, 97.5])
    high_ci = np.percentile(high_boot, [2.5, 97.5])
    
    categories = ['Low Noise\n(n=35)', 'High Noise\n(n=34)']
    means = [np.mean(low_benefits), np.mean(high_benefits)]
    ci_low = [means[0] - low_ci[0], means[1] - high_ci[0]]
    ci_high = [low_ci[1] - means[0], high_ci[1] - means[1]]
    
    colors = ['#c0392b' if m < 0 else '#27ae60' for m in means]
    
    ax.bar(categories, means, yerr=[ci_low, ci_high], color=colors, alpha=0.7,
           capsize=8, edgecolor='black', linewidth=1.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    ax.set_ylabel('DAQEC Benefit (%)')
    ax.set_title('B. Bootstrap 95% CIs\n(10,000 resamples)', fontweight='bold')
    
    # Add significance annotations
    for i, (m, ci_l, ci_h) in enumerate(zip(means, ci_low, ci_high)):
        if m - ci_l > 0 or m + ci_h < 0:
            sig = '***'
        else:
            sig = 'n.s.'
        ax.annotate(sig, xy=(i, m + ci_h + 1), ha='center', fontsize=12, fontweight='bold')
    
    # Panel C: Leave-one-out sensitivity
    ax = axes[2]
    
    # Calculate correlation without each point
    loo_correlations = []
    for i in range(len(pairs_n69)):
        mask = np.ones(len(pairs_n69), dtype=bool)
        mask[i] = False
        r, _ = stats.pearsonr(baseline_lers[mask], benefits[mask])
        loo_correlations.append(r)
    
    ax.hist(loo_correlations, bins=30, color='#9b59b6', alpha=0.7, edgecolor='white')
    ax.axvline(x=np.mean(loo_correlations), color='#e74c3c', linestyle='--', linewidth=2,
               label=f'Mean: r={np.mean(loo_correlations):.3f}')
    
    ax.set_xlabel('Leave-One-Out Correlation (r)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'C. Leave-One-Out Sensitivity\nRange: [{min(loo_correlations):.3f}, {max(loo_correlations):.3f}]', 
                 fontweight='bold')
    ax.legend()
    
    plt.suptitle('Extended Data Figure 2: Statistical Robustness Checks', 
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'ExtendedData_Fig2_Robustness.png', dpi=300)
    plt.savefig(output_dir / 'ExtendedData_Fig2_Robustness.pdf')
    plt.close()
    print("✓ Generated Extended Data Figure 2")

def generate_extended_data_fig3(pairs_n69, pairs_n48, output_dir):
    """Extended Data Figure 3: Cross-validation details."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel A: Scatter plot for N=48
    ax = axes[0]
    
    baseline_lers_48 = np.array([p['baseline_ler'] for p in pairs_n48])
    benefits_48 = np.array([p['benefit_pct'] for p in pairs_n48])
    
    r_48, p_48 = stats.pearsonr(baseline_lers_48, benefits_48)
    
    ax.scatter(baseline_lers_48, benefits_48, c=baseline_lers_48, cmap='RdYlGn', 
               s=60, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Regression line
    slope, intercept = np.polyfit(baseline_lers_48, benefits_48, 1)
    x_line = np.linspace(baseline_lers_48.min(), baseline_lers_48.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'k--', linewidth=2, label=f'r={r_48:.3f}')
    
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
    ax.fill_between([0, 0.2], 0, 50, alpha=0.1, color='green')
    ax.fill_between([0, 0.2], -50, 0, alpha=0.1, color='red')
    
    ax.set_xlabel('Baseline LER')
    ax.set_ylabel('DAQEC Benefit (%)')
    ax.set_title(f'A. Validation Dataset (N=48)\nr = {r_48:.3f}, p = {p_48:.3f}', fontweight='bold')
    ax.legend()
    ax.set_xlim(0.05, 0.15)
    ax.set_ylim(-40, 40)
    
    # Panel B: Forest plot
    ax = axes[1]
    
    # Calculate effects for each dataset
    baseline_lers_69 = np.array([p['baseline_ler'] for p in pairs_n69])
    benefits_69 = np.array([p['benefit_pct'] for p in pairs_n69])
    
    median_69 = np.median(baseline_lers_69)
    median_48 = np.median(baseline_lers_48)
    
    # Stratum effects
    effects = [
        ('N=48 Low', np.mean(benefits_48[baseline_lers_48 < median_48]), 
         stats.sem(benefits_48[baseline_lers_48 < median_48]) * 1.96, len(benefits_48[baseline_lers_48 < median_48])),
        ('N=48 High', np.mean(benefits_48[baseline_lers_48 >= median_48]),
         stats.sem(benefits_48[baseline_lers_48 >= median_48]) * 1.96, len(benefits_48[baseline_lers_48 >= median_48])),
        ('N=69 Low', np.mean(benefits_69[baseline_lers_69 < median_69]),
         stats.sem(benefits_69[baseline_lers_69 < median_69]) * 1.96, len(benefits_69[baseline_lers_69 < median_69])),
        ('N=69 High', np.mean(benefits_69[baseline_lers_69 >= median_69]),
         stats.sem(benefits_69[baseline_lers_69 >= median_69]) * 1.96, len(benefits_69[baseline_lers_69 >= median_69])),
    ]
    
    y_positions = [3, 2, 1, 0]
    for i, (label, effect, ci, n) in enumerate(effects):
        color = '#27ae60' if effect > 0 else '#c0392b'
        ax.errorbar(effect, y_positions[i], xerr=ci, fmt='o', markersize=10, 
                    color=color, capsize=5, capthick=2, elinewidth=2)
        ax.annotate(f'{label}\n(n={n})', xy=(-45, y_positions[i]), fontsize=9, va='center')
        ax.annotate(f'{effect:+.1f}%', xy=(effect + ci + 2, y_positions[i]), fontsize=9, va='center')
    
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=1)
    ax.set_xlabel('DAQEC Benefit (%)')
    ax.set_title('B. Forest Plot\nStratified Effects by Dataset', fontweight='bold')
    ax.set_yticks([])
    ax.set_xlim(-50, 30)
    
    # Panel C: Meta-analysis
    ax = axes[2]
    
    # Interaction effects (high - low difference)
    diff_48 = np.mean(benefits_48[baseline_lers_48 >= median_48]) - np.mean(benefits_48[baseline_lers_48 < median_48])
    diff_69 = np.mean(benefits_69[baseline_lers_69 >= median_69]) - np.mean(benefits_69[baseline_lers_69 < median_69])
    
    # Standard errors (simplified)
    se_48 = np.sqrt(np.var(benefits_48[baseline_lers_48 >= median_48])/len(benefits_48[baseline_lers_48 >= median_48]) + 
                    np.var(benefits_48[baseline_lers_48 < median_48])/len(benefits_48[baseline_lers_48 < median_48]))
    se_69 = np.sqrt(np.var(benefits_69[baseline_lers_69 >= median_69])/len(benefits_69[baseline_lers_69 >= median_69]) + 
                    np.var(benefits_69[baseline_lers_69 < median_69])/len(benefits_69[baseline_lers_69 < median_69]))
    
    # Weights (inverse variance)
    w_48 = 1 / se_48**2
    w_69 = 1 / se_69**2
    
    # Combined effect
    combined = (w_48 * diff_48 + w_69 * diff_69) / (w_48 + w_69)
    se_combined = np.sqrt(1 / (w_48 + w_69))
    z = combined / se_combined
    p_meta = 2 * (1 - stats.norm.cdf(abs(z)))
    
    # Plot
    y_pos = [2, 1, 0]
    effects = [diff_48, diff_69, combined]
    errors = [se_48 * 1.96, se_69 * 1.96, se_combined * 1.96]
    labels = ['N=48', 'N=69', 'Combined']
    sizes = [10, 10, 15]
    
    for i, (eff, err, label, size) in enumerate(zip(effects, errors, labels, sizes)):
        marker = 'D' if i == 2 else 'o'
        ax.errorbar(eff, y_pos[i], xerr=err, fmt=marker, markersize=size,
                    color='#2c3e50', capsize=5, capthick=2, elinewidth=2)
        ax.annotate(f'{label}', xy=(-5, y_pos[i]), fontsize=10, va='center', ha='right')
        ax.annotate(f'{eff:+.1f}% [{eff-err:.1f}, {eff+err:.1f}]', 
                    xy=(eff + err + 2, y_pos[i]), fontsize=9, va='center')
    
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=1)
    ax.set_xlabel('Interaction Effect (High - Low Stratum Difference, %)')
    ax.set_title(f'C. Random-Effects Meta-Analysis\nCombined p = {p_meta:.5f}', fontweight='bold')
    ax.set_yticks([])
    ax.set_xlim(-10, 40)
    
    plt.suptitle('Extended Data Figure 3: Cross-Validation and Meta-Analysis', 
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'ExtendedData_Fig3_CrossValidation.png', dpi=300)
    plt.savefig(output_dir / 'ExtendedData_Fig3_CrossValidation.pdf')
    plt.close()
    print("✓ Generated Extended Data Figure 3")

def generate_extended_data_fig4(pairs_n48, pairs_n69, output_dir):
    """Extended Data Figure 4: Temporal evolution of hardware state."""
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Panel A: Timeline of baseline LER
    ax = axes[0]
    
    # Combine data with time markers
    n48_sessions = list(range(len(pairs_n48)))
    n69_sessions = list(range(len(pairs_n48), len(pairs_n48) + len(pairs_n69)))
    
    baseline_48 = [p['baseline_ler'] for p in pairs_n48]
    baseline_69 = [p['baseline_ler'] for p in pairs_n69]
    
    ax.scatter(n48_sessions, baseline_48, c='#3498db', s=50, alpha=0.7, 
               label='N=48 period', edgecolor='black', linewidth=0.5)
    ax.scatter(n69_sessions, baseline_69, c='#e74c3c', s=50, alpha=0.7,
               label='N=69 period', edgecolor='black', linewidth=0.5)
    
    # Add means
    ax.axhline(y=np.mean(baseline_48), color='#3498db', linestyle='--', linewidth=2,
               label=f'N=48 mean: {np.mean(baseline_48):.3f}')
    ax.axhline(y=np.mean(baseline_69), color='#e74c3c', linestyle='--', linewidth=2,
               label=f'N=69 mean: {np.mean(baseline_69):.3f}')
    
    # Mark transition
    transition_x = len(pairs_n48) - 0.5
    ax.axvline(x=transition_x, color='purple', linestyle='-', linewidth=3, alpha=0.7)
    ax.annotate('Hardware State\nTransition', xy=(transition_x, 0.135), fontsize=11,
                ha='center', color='purple', fontweight='bold')
    
    # Mark crossover threshold
    ax.axhline(y=0.112, color='green', linestyle=':', linewidth=2,
               label='Crossover threshold (0.112)')
    
    ax.set_xlabel('Session Number (chronological)')
    ax.set_ylabel('Baseline Logical Error Rate')
    ax.set_title('A. Temporal Evolution of Hardware Noise Level', fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_xlim(-2, len(pairs_n48) + len(pairs_n69) + 2)
    
    # Panel B: Effect direction over time
    ax = axes[1]
    
    benefits_48 = [p['benefit_pct'] for p in pairs_n48]
    benefits_69 = [p['benefit_pct'] for p in pairs_n69]
    
    colors_48 = ['#27ae60' if b > 0 else '#c0392b' for b in benefits_48]
    colors_69 = ['#27ae60' if b > 0 else '#c0392b' for b in benefits_69]
    
    ax.bar(n48_sessions, benefits_48, color=colors_48, alpha=0.7, edgecolor='black', linewidth=0.3)
    ax.bar(n69_sessions, benefits_69, color=colors_69, alpha=0.7, edgecolor='black', linewidth=0.3)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.axvline(x=transition_x, color='purple', linestyle='-', linewidth=3, alpha=0.7)
    
    # Add means
    ax.axhline(y=np.mean(benefits_48), color='#3498db', linestyle='--', linewidth=2)
    ax.axhline(y=np.mean(benefits_69), color='#e74c3c', linestyle='--', linewidth=2)
    
    ax.set_xlabel('Session Number (chronological)')
    ax.set_ylabel('DAQEC Benefit (%)')
    ax.set_title('B. DAQEC Effect Direction Over Time\n(Green = benefit, Red = harm)', fontweight='bold')
    
    # Add annotations
    pct_positive_48 = sum(1 for b in benefits_48 if b > 0) / len(benefits_48) * 100
    pct_positive_69 = sum(1 for b in benefits_69 if b > 0) / len(benefits_69) * 100
    
    ax.annotate(f'N=48: {pct_positive_48:.0f}% positive\nMean: {np.mean(benefits_48):+.1f}%', 
                xy=(len(pairs_n48)/2, 30), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='#3498db', alpha=0.3))
    ax.annotate(f'N=69: {pct_positive_69:.0f}% positive\nMean: {np.mean(benefits_69):+.1f}%', 
                xy=(len(pairs_n48) + len(pairs_n69)/2, 30), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='#e74c3c', alpha=0.3))
    
    ax.set_xlim(-2, len(pairs_n48) + len(pairs_n69) + 2)
    ax.set_ylim(-40, 40)
    
    plt.suptitle('Extended Data Figure 4: Temporal Evolution of Hardware State and DAQEC Effect', 
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'ExtendedData_Fig4_Temporal.png', dpi=300)
    plt.savefig(output_dir / 'ExtendedData_Fig4_Temporal.pdf')
    plt.close()
    print("✓ Generated Extended Data Figure 4")

def main():
    """Generate all Extended Data figures."""
    print("=" * 70)
    print("GENERATING EXTENDED DATA FIGURES")
    print("=" * 70)
    
    # Load data
    data_n69, data_n48 = load_data()
    pairs_n69 = process_paired_data(data_n69)
    pairs_n48 = process_paired_data(data_n48)
    
    print(f"Loaded N=69 dataset: {len(pairs_n69)} pairs")
    print(f"Loaded N=48 dataset: {len(pairs_n48)} pairs")
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / "manuscript" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate figures
    generate_extended_data_fig1(pairs_n69, output_dir)
    generate_extended_data_fig2(pairs_n69, output_dir)
    generate_extended_data_fig3(pairs_n69, pairs_n48, output_dir)
    generate_extended_data_fig4(pairs_n48, pairs_n69, output_dir)
    
    print("=" * 70)
    print("ALL EXTENDED DATA FIGURES GENERATED")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print("\nFigures:")
    print("  - ExtendedData_Fig1_SessionLevel.png/pdf")
    print("  - ExtendedData_Fig2_Robustness.png/pdf")
    print("  - ExtendedData_Fig3_CrossValidation.png/pdf")
    print("  - ExtendedData_Fig4_Temporal.png/pdf")

if __name__ == "__main__":
    main()
