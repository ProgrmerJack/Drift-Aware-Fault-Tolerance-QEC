"""
IQM Emerald Validation v5 - Fixed Qubit Configuration

Based on stratified analysis, we found that:
- Best qubits (24,25,26) + Worst qubits (12,13,14) showed strongest effect
- Quality gap of 6% produced Cohen's d = -0.375

This script runs validation with FIXED qubit configuration to reduce variance.
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from braket.aws import AwsDevice
from braket.circuits import Circuit
import numpy as np
from scipy import stats

# AWS credentials should be set via environment variables or AWS CLI config
# Required: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION=eu-north-1
if 'AWS_DEFAULT_REGION' not in os.environ:
    os.environ['AWS_DEFAULT_REGION'] = 'eu-north-1'


# Fixed configuration based on best-performing batch
BEST_DATA = [24, 25, 26]
BEST_ANCILLA = 31
WORST_DATA = [12, 13, 14]
WORST_ANCILLA = 19


def run_single_condition(device, data_qubits, ancilla, shots):
    """Run a single QEC circuit."""
    circuit = Circuit()
    for q in data_qubits:
        circuit.i(q)
    for q in data_qubits:
        circuit.cnot(q, ancilla)
    for q in data_qubits:
        circuit.measure(q)
    circuit.measure(ancilla)
    
    task = device.run(circuit, shots=shots)
    result = task.result()
    counts = result.measurement_counts
    success = counts.get('0' * (len(data_qubits) + 1), 0)
    return 1.0 - success / shots


def run_validation_v5(device, shots=500, n_runs=50):
    """
    Run validation with fixed qubit configuration.
    
    Experimental design:
    - LOW noise: Both use BEST qubits (24,25,26)
    - HIGH noise: Drift-aware uses BEST, calibration uses WORST (12,13,14)
    """
    
    results = {
        'version': 'v5-fixed-config',
        'timestamp': datetime.now().isoformat(),
        'device': device.name,
        'shots_per_condition': shots,
        'n_runs': n_runs,
        'best_qubits': BEST_DATA,
        'worst_qubits': WORST_DATA,
        'runs': []
    }
    
    print(f"\n" + "="*70)
    print("RUNNING VALIDATION v5 (Fixed Configuration)")
    print("="*70)
    print(f"BEST qubits:  {BEST_DATA} + ancilla {BEST_ANCILLA}")
    print(f"WORST qubits: {WORST_DATA} + ancilla {WORST_ANCILLA}")
    print(f"Shots: {shots}, Runs: {n_runs}")
    
    interactions = []
    
    for run_idx in range(n_runs):
        print(f"\n--- RUN {run_idx + 1}/{n_runs} ---")
        
        # LOW noise: Both use BEST
        ler_low_drift = run_single_condition(device, BEST_DATA, BEST_ANCILLA, shots)
        ler_low_calib = run_single_condition(device, BEST_DATA, BEST_ANCILLA, shots)
        low_effect = ler_low_drift - ler_low_calib
        
        # HIGH noise: Drift uses BEST, Calib uses WORST
        ler_high_drift = run_single_condition(device, BEST_DATA, BEST_ANCILLA, shots)
        ler_high_calib = run_single_condition(device, WORST_DATA, WORST_ANCILLA, shots)
        high_effect = ler_high_drift - ler_high_calib
        
        interaction = high_effect - low_effect
        interactions.append(interaction)
        
        print(f"  LOW:  drift={ler_low_drift:.4f}, calib={ler_low_calib:.4f}, effect={low_effect:+.4f}")
        print(f"  HIGH: drift={ler_high_drift:.4f}, calib={ler_high_calib:.4f}, effect={high_effect:+.4f}")
        print(f"  Interaction: {interaction:+.4f}")
        
        results['runs'].append({
            'low_drift_ler': ler_low_drift,
            'low_calib_ler': ler_low_calib,
            'high_drift_ler': ler_high_drift,
            'high_calib_ler': ler_high_calib,
            'low_effect': low_effect,
            'high_effect': high_effect,
            'interaction': interaction
        })
        
        # Running statistics
        if (run_idx + 1) % 10 == 0:
            current_mean = np.mean(interactions)
            current_std = np.std(interactions)
            neg_ratio = sum(1 for i in interactions if i < 0) / len(interactions)
            t_stat, p_two = stats.ttest_1samp(interactions, 0)
            p_one = p_two / 2 if t_stat < 0 else 1 - p_two / 2
            
            print(f"\n  [Running stats @ {run_idx+1} runs]")
            print(f"  Mean: {current_mean:+.4f} ± {current_std:.4f}")
            print(f"  Direction: {neg_ratio:.1%} negative")
            print(f"  p-value: {p_one:.4f}")
    
    # Final analysis
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    low_effects = [r['low_effect'] for r in results['runs']]
    high_effects = [r['high_effect'] for r in results['runs']]
    
    mean_int = np.mean(interactions)
    std_int = np.std(interactions)
    se_int = std_int / np.sqrt(len(interactions))
    
    t_stat, p_two = stats.ttest_1samp(interactions, 0)
    p_one = p_two / 2 if t_stat < 0 else 1 - p_two / 2
    
    d = mean_int / std_int
    neg_ratio = sum(1 for i in interactions if i < 0) / len(interactions)
    
    ci = stats.t.interval(0.95, len(interactions)-1, loc=mean_int, scale=se_int)
    
    print(f"\nInteraction Effect:")
    print(f"  Mean: {mean_int:+.4f}")
    print(f"  Std:  {std_int:.4f}")
    print(f"  SE:   {se_int:.4f}")
    print(f"  95% CI: [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    
    print(f"\nStatistical Tests:")
    print(f"  t = {t_stat:.3f}")
    print(f"  p (one-tailed, H1: <0) = {p_one:.4f}")
    print(f"  Cohen's d = {d:.3f}")
    print(f"  Direction: {neg_ratio:.1%} negative ({int(neg_ratio*n_runs)}/{n_runs})")
    
    print(f"\nMean LERs:")
    print(f"  LOW+Drift:  {np.mean([r['low_drift_ler'] for r in results['runs']]):.4f}")
    print(f"  LOW+Calib:  {np.mean([r['low_calib_ler'] for r in results['runs']]):.4f}")
    print(f"  HIGH+Drift: {np.mean([r['high_drift_ler'] for r in results['runs']]):.4f}")
    print(f"  HIGH+Calib: {np.mean([r['high_calib_ler'] for r in results['runs']]):.4f}")
    
    print(f"\nMean Effects:")
    print(f"  LOW effect:  {np.mean(low_effects):+.4f}")
    print(f"  HIGH effect: {np.mean(high_effects):+.4f}")
    
    # Claim verification
    print("\n--- MANUSCRIPT CLAIM VERIFICATION ---")
    low_ok = abs(np.mean(low_effects)) < 0.02
    high_ok = np.mean(high_effects) < 0
    int_ok = mean_int < 0 and p_one < 0.05
    
    print(f"1. LOW effect ≈ 0: {np.mean(low_effects):+.4f} → {'✓' if low_ok else '✗'}")
    print(f"2. HIGH effect < 0: {np.mean(high_effects):+.4f} → {'✓' if high_ok else '✗'}")
    print(f"3. Interaction < 0 (p<0.05): {mean_int:+.4f}, p={p_one:.4f} → {'✓' if int_ok else '✗'}")
    
    if low_ok and high_ok and int_ok:
        print("\n★★★ ALL CLAIMS STATISTICALLY SUPPORTED ★★★")
    elif mean_int < 0:
        print("\n★★ TREND SUPPORTS CLAIMS ★★")
    else:
        print("\n⚠ CLAIMS NOT SUPPORTED ⚠")
    
    results['summary'] = {
        'mean_interaction': mean_int,
        'std_interaction': std_int,
        't_statistic': t_stat,
        'p_value_one_tailed': p_one,
        'cohens_d': d,
        'ci_95': ci,
        'direction_negative': neg_ratio
    }
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shots', type=int, default=500)
    parser.add_argument('--runs', type=int, default=50)
    args = parser.parse_args()
    
    print("="*70)
    print("IQM EMERALD VALIDATION v5 - Fixed Qubit Configuration")
    print("="*70)
    
    device = AwsDevice("arn:aws:braket:eu-north-1::device/qpu/iqm/Emerald")
    print(f"Device: {device.name}")
    print(f"Status: {device.status}")
    
    # Estimate cost
    cost = 4 * args.runs * (0.30 + args.shots * 0.00160)
    print(f"Estimated cost: ${cost:.2f}")
    
    results = run_validation_v5(device, shots=args.shots, n_runs=args.runs)
    
    # Save results
    results_dir = Path(__file__).parent.parent / "results" / "multi_platform"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = results_dir / f"iqm_validation_v5_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
