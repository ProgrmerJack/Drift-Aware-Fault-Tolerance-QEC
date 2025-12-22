"""Debug script to inspect drift model behavior."""

import sys
sys.path.append('simulations')
from surface_code_simulator import QuantumPlatform, DriftModel
import numpy as np

platform = QuantumPlatform("ibm")

# Generate qubits at t=0 (calibration)
qubits_t0 = platform.generate_qubits(20, time_since_calibration=0, seed=42)

# Generate same qubits at t=24h (drifted)
qubits_t24 = platform.generate_qubits(20, time_since_calibration=24, seed=42)

print("="*80)
print("DRIFT ANALYSIS: t=0 vs t=24h")
print("="*80)

for i in range(20):
    q0 = qubits_t0[i]
    q24 = qubits_t24[i]
    
    t1_change = 100 * (q0.T1 - q24.T1) / q0.T1
    t2_change = 100 * (q0.T2 - q24.T2) / q0.T2
    gate_change = 100 * (q24.gate_error_2q - q0.gate_error_2q) / q0.gate_error_2q
    
    # Simple quality score
    score_t0 = (q0.T1 + q0.T2) - 1000 * q0.gate_error_2q
    score_t24 = (q24.T1 + q24.T2) - 1000 * q24.gate_error_2q
    score_change = 100 * (score_t0 - score_t24) / score_t0
    
    print(f"\nQubit {i}:")
    print(f"  T1: {q0.T1:.3f} -> {q24.T1:.3f} us ({t1_change:+.1f}%)")
    print(f"  T2: {q0.T2:.3f} -> {q24.T2:.3f} us ({t2_change:+.1f}%)")
    print(f"  Gate error: {q0.gate_error_2q:.6f} -> {q24.gate_error_2q:.6f} ({gate_change:+.1f}%)")
    print(f"  Score: {score_t0:.3f} -> {score_t24:.3f} ({score_change:+.1f}%)")

# Now check RANKINGS
scores_t0 = [(i, (q.T1 + q.T2) - 1000 * q.gate_error_2q) for i, q in enumerate(qubits_t0)]
scores_t24 = [(i, (q.T1 + q.T2) - 1000 * q.gate_error_2q) for i, q in enumerate(qubits_t24)]

scores_t0.sort(key=lambda x: x[1], reverse=True)
scores_t24.sort(key=lambda x: x[1], reverse=True)

print("\n" + "="*80)
print("RANKING COMPARISON")
print("="*80)
print("\nTop 10 qubits at t=0:", [idx for idx, _ in scores_t0[:10]])
print("Top 10 qubits at t=24:", [idx for idx, _ in scores_t24[:10]])

# How many overlap?
top10_t0 = set([idx for idx, _ in scores_t0[:10]])
top10_t24 = set([idx for idx, _ in scores_t24[:10]])
overlap = len(top10_t0 & top10_t24)
print(f"\nOverlap: {overlap}/10 ({100*overlap/10}%)")
print("\n⚠️  If overlap is >50%, rankings don't change enough!")
