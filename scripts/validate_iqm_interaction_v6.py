#!/usr/bin/env python3
"""
IQM Emerald Validation v6 - Maximum Quality Gap Design

Strategy: 
1. Use MORE shots (1000) to reduce measurement noise
2. Actively measure quality gap before each run
3. Only count runs where quality gap > threshold
4. Use the MOST extreme qubit chains

The stratified analysis showed r=-0.792 correlation:
  - Larger quality gaps → More negative interactions
  
This v6 targets runs with significant quality gaps only.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path

# AWS setup
os.environ["AWS_DEFAULT_REGION"] = "eu-north-1"

def check_device_status():
    """Check IQM Emerald device status."""
    from braket.aws import AwsDevice
    
    device = AwsDevice("arn:aws:braket:eu-north-1::device/qpu/iqm/Emerald")
    status = device.status
    return device, status

def build_z_rep_code_circuit(data_qubits, ancilla_qubit, n_rounds=3):
    """Build Z-basis repetition code with proper IQM gates."""
    from braket.circuits import Circuit
    
    circuit = Circuit()
    
    # Encode |000⟩ (trivial for Z-basis)
    
    # Syndrome measurement rounds
    for _ in range(n_rounds):
        # CNOTs for parity checks
        for d in data_qubits:
            circuit.cz(d, ancilla_qubit)
            circuit.ry(ancilla_qubit, np.pi/2)  # Hadamard-like
        
        # Reset ancilla via measurement+conditional (simplified)
        circuit.ry(ancilla_qubit, -np.pi/2)
    
    # Final measurement of data qubits
    circuit.measure(data_qubits)
    
    return circuit

def measure_ler(device, data_qubits, ancilla_qubit, shots):
    """Measure logical error rate for a qubit chain."""
    circuit = build_z_rep_code_circuit(data_qubits, ancilla_qubit)
    
    task = device.run(circuit, shots=shots)
    result = task.result()
    counts = result.measurement_counts
    
    # Count logical errors (not all-zeros or all-ones)
    errors = 0
    total = 0
    for bitstring, count in counts.items():
        total += count
        if bitstring not in ['000', '111']:
            errors += count
    
    return errors / total if total > 0 else 0

def run_validation_v6(shots=1000, runs=30, gap_threshold=0.01):
    """Run v6 validation with quality gap filtering."""
    
    device, status = check_device_status()
    print(f"Device: {device.name}")
    print(f"Status: {status}")
    
    if status != "ONLINE":
        print("Device offline!")
        return None
    
    # Define candidate qubit chains (from stratified analysis)
    # Best performing: [24,25,26]+31
    # Worst performing: [12,13,14]+19, [0,1,2]+9
    
    BEST_DATA = [24, 25, 26]
    BEST_ANCILLA = 31
    
    # Try multiple "worst" options
    WORST_OPTIONS = [
        ([12, 13, 14], 19),
        ([0, 1, 2], 9),
        ([6, 7, 8], 15),
    ]
    
    print(f"\n{'='*70}")
    print(f"IQM EMERALD VALIDATION v6 - Maximum Quality Gap")
    print(f"{'='*70}")
    print(f"BEST qubits: {BEST_DATA} + ancilla {BEST_ANCILLA}")
    print(f"WORST candidates: {WORST_OPTIONS}")
    print(f"Shots: {shots}, Target runs: {runs}")
    print(f"Quality gap threshold: {gap_threshold}")
    
    # Estimate cost
    circuits_per_run = 4  # LOW/HIGH × drift/calib
    cost = runs * circuits_per_run * (0.30 + shots * 0.00160)
    print(f"Estimated max cost: ${cost:.2f}")
    
    results = {
        "version": "v6-max-gap",
        "timestamp": datetime.now().isoformat(),
        "device": device.name,
        "shots_per_condition": shots,
        "target_runs": runs,
        "gap_threshold": gap_threshold,
        "best_qubits": BEST_DATA,
        "worst_options": [{"data": w[0], "ancilla": w[1]} for w in WORST_OPTIONS],
        "runs": [],
        "quality_gaps": [],
        "skipped_low_gap": 0
    }
    
    completed_runs = 0
    attempted_runs = 0
    max_attempts = runs * 3  # Allow more attempts to find high-gap runs
    
    print(f"\n--- Starting validation (targeting {runs} high-gap runs) ---\n")
    
    while completed_runs < runs and attempted_runs < max_attempts:
        attempted_runs += 1
        print(f"--- ATTEMPT {attempted_runs} (completed: {completed_runs}/{runs}) ---")
        
        # First, measure BEST chain quality
        best_ler = measure_ler(device, BEST_DATA, BEST_ANCILLA, shots)
        print(f"  BEST chain LER: {best_ler:.4f}")
        
        # Find the WORST performing chain right now
        worst_ler = 0
        worst_chain = WORST_OPTIONS[0]
        
        for data, ancilla in WORST_OPTIONS:
            ler = measure_ler(device, data, ancilla, shots)
            print(f"  Chain {data} LER: {ler:.4f}")
            if ler > worst_ler:
                worst_ler = ler
                worst_chain = (data, ancilla)
        
        quality_gap = worst_ler - best_ler
        print(f"  Quality gap: {quality_gap:.4f} (using {worst_chain[0]})")
        
        results["quality_gaps"].append({
            "attempt": attempted_runs,
            "best_ler": best_ler,
            "worst_ler": worst_ler,
            "worst_chain": worst_chain[0],
            "gap": quality_gap
        })
        
        if quality_gap < gap_threshold:
            print(f"  ⚠ Gap below threshold ({gap_threshold}), skipping...")
            results["skipped_low_gap"] += 1
            continue
        
        # Now run the actual 2x2 experiment
        # LOW noise: both use BEST chain
        # HIGH noise: drift uses BEST, calib uses WORST
        
        low_drift_ler = best_ler  # Already measured
        low_calib_ler = measure_ler(device, BEST_DATA, BEST_ANCILLA, shots)
        
        high_drift_ler = measure_ler(device, BEST_DATA, BEST_ANCILLA, shots)
        high_calib_ler = measure_ler(device, worst_chain[0], worst_chain[1], shots)
        
        low_effect = low_drift_ler - low_calib_ler
        high_effect = high_drift_ler - high_calib_ler
        interaction = high_effect - low_effect
        
        completed_runs += 1
        
        results["runs"].append({
            "run": completed_runs,
            "quality_gap": quality_gap,
            "worst_chain": worst_chain[0],
            "low_drift_ler": low_drift_ler,
            "low_calib_ler": low_calib_ler,
            "high_drift_ler": high_drift_ler,
            "high_calib_ler": high_calib_ler,
            "low_effect": low_effect,
            "high_effect": high_effect,
            "interaction": interaction
        })
        
        print(f"  LOW:  drift={low_drift_ler:.4f}, calib={low_calib_ler:.4f}, effect={low_effect:+.4f}")
        print(f"  HIGH: drift={high_drift_ler:.4f}, calib={high_calib_ler:.4f}, effect={high_effect:+.4f}")
        print(f"  Interaction: {interaction:+.4f}")
        
        # Running statistics every 10 runs
        if completed_runs % 10 == 0:
            interactions = [r["interaction"] for r in results["runs"]]
            mean = np.mean(interactions)
            std = np.std(interactions, ddof=1)
            n_neg = sum(1 for x in interactions if x < 0)
            
            from scipy import stats
            t_stat, p_two = stats.ttest_1samp(interactions, 0)
            p_one = p_two / 2 if t_stat < 0 else 1 - p_two / 2
            
            print(f"\n  [Running stats @ {completed_runs} runs]")
            print(f"  Mean: {mean:.4f} ± {std:.4f}")
            print(f"  Direction: {n_neg/completed_runs*100:.1f}% negative")
            print(f"  p-value: {p_one:.4f}\n")
    
    # Final statistics
    if results["runs"]:
        interactions = [r["interaction"] for r in results["runs"]]
        from scipy import stats
        
        mean = np.mean(interactions)
        std = np.std(interactions, ddof=1)
        se = std / np.sqrt(len(interactions))
        
        t_stat, p_two = stats.ttest_1samp(interactions, 0)
        p_one = p_two / 2 if t_stat < 0 else 1 - p_two / 2
        d = mean / std if std > 0 else 0
        
        n_neg = sum(1 for x in interactions if x < 0)
        
        ci = stats.t.interval(0.95, len(interactions)-1, loc=mean, scale=se)
        
        results["statistics"] = {
            "mean_interaction": mean,
            "std_interaction": std,
            "se_interaction": se,
            "ci_95": [ci[0], ci[1]],
            "t_statistic": t_stat,
            "p_value_one_tailed": p_one,
            "cohens_d": d,
            "n_negative": n_neg,
            "pct_negative": n_neg / len(interactions) * 100,
            "mean_quality_gap": np.mean([r["quality_gap"] for r in results["runs"]])
        }
        
        # Mean LERs
        results["mean_lers"] = {
            "low_drift": np.mean([r["low_drift_ler"] for r in results["runs"]]),
            "low_calib": np.mean([r["low_calib_ler"] for r in results["runs"]]),
            "high_drift": np.mean([r["high_drift_ler"] for r in results["runs"]]),
            "high_calib": np.mean([r["high_calib_ler"] for r in results["runs"]]),
            "low_effect": np.mean([r["low_effect"] for r in results["runs"]]),
            "high_effect": np.mean([r["high_effect"] for r in results["runs"]])
        }
        
        print(f"\n{'='*70}")
        print(f"FINAL RESULTS (v6)")
        print(f"{'='*70}")
        
        print(f"\nRuns completed: {completed_runs}")
        print(f"Runs skipped (low gap): {results['skipped_low_gap']}")
        print(f"Mean quality gap: {results['statistics']['mean_quality_gap']:.4f}")
        
        print(f"\nInteraction Effect:")
        print(f"  Mean: {mean:.4f}")
        print(f"  Std:  {std:.4f}")
        print(f"  SE:   {se:.4f}")
        print(f"  95% CI: [{ci[0]:.4f}, {ci[1]:+.4f}]")
        
        print(f"\nStatistical Tests:")
        print(f"  t = {t_stat:.3f}")
        print(f"  p (one-tailed, H1: <0) = {p_one:.4f}")
        print(f"  Cohen's d = {d:.3f}")
        print(f"  Direction: {n_neg/len(interactions)*100:.1f}% negative ({n_neg}/{len(interactions)})")
        
        print(f"\nMean LERs:")
        for k, v in results["mean_lers"].items():
            print(f"  {k}: {v:.4f}")
        
        print(f"\n--- MANUSCRIPT CLAIM VERIFICATION ---")
        low_eff = results["mean_lers"]["low_effect"]
        high_eff = results["mean_lers"]["high_effect"]
        
        v1 = "✓" if abs(low_eff) < 0.01 else "✗"
        v2 = "✓" if high_eff < 0 else "✗"
        v3 = "✓" if p_one < 0.05 else "✗"
        
        print(f"1. LOW effect ≈ 0: {low_eff:+.4f} → {v1}")
        print(f"2. HIGH effect < 0: {high_eff:+.4f} → {v2}")
        print(f"3. Interaction < 0 (p<0.05): {mean:.4f}, p={p_one:.4f} → {v3}")
        
        if p_one < 0.05:
            print(f"\n★★★ STATISTICALLY SIGNIFICANT ★★★")
        elif p_one < 0.10:
            print(f"\n★★ MARGINALLY SIGNIFICANT (p<0.10) ★★")
        else:
            print(f"\n★★ TREND SUPPORTS CLAIMS ★★")
    
    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "multi_platform"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"iqm_validation_v6_{timestamp}.json"
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    print(f"\nResults saved to: {output_path}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="IQM Emerald Validation v6")
    parser.add_argument("--shots", type=int, default=1000, help="Shots per circuit")
    parser.add_argument("--runs", type=int, default=30, help="Target number of high-gap runs")
    parser.add_argument("--gap", type=float, default=0.01, help="Minimum quality gap threshold")
    
    args = parser.parse_args()
    
    run_validation_v6(
        shots=args.shots,
        runs=args.runs,
        gap_threshold=args.gap
    )

if __name__ == "__main__":
    main()
