"""Diagnose the fundamental qubit quality difference on IQM Emerald."""

import os
import json
from datetime import datetime
from pathlib import Path
from braket.aws import AwsDevice
from braket.circuits import Circuit
import numpy as np

# AWS credentials should be set via environment variables or AWS CLI config
# Required: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION=eu-north-1
if 'AWS_DEFAULT_REGION' not in os.environ:
    os.environ['AWS_DEFAULT_REGION'] = 'eu-north-1'

print("="*70)
print("IQM EMERALD QUBIT QUALITY CHARACTERIZATION")
print("="*70)

# Get device
device = AwsDevice("arn:aws:braket:eu-north-1::device/qpu/iqm/Emerald")
print(f"Device: {device.name}")
print(f"Status: {device.status}")

# Define qubit sets
OPTIMAL_QUBITS = [25, 26, 27]  # Center of device
STALE_QUBITS = [0, 1, 2]       # Edge of device
MEDIUM_QUBITS = [12, 13, 14]   # Another region

SHOTS = 1000  # Per qubit

def measure_single_qubit_error(qubits: list, shots: int, label: str):
    """Measure T1/T2-like decay for given qubits."""
    print(f"\n--- Testing {label} qubits: {qubits} ---")
    
    results = {}
    for q in qubits:
        # Test 1: Initialize |0>, measure
        circuit0 = Circuit().i(q).measure(q)
        
        # Test 2: Initialize |1>, measure (X gate)
        circuit1 = Circuit().x(q).measure(q)
        
        # Run both
        task0 = device.run(circuit0, shots=shots)
        task1 = device.run(circuit1, shots=shots)
        
        r0 = task0.result()
        r1 = task1.result()
        
        # Error rates
        counts0 = r0.measurement_counts
        counts1 = r1.measurement_counts
        
        err_0 = counts0.get('1', 0) / shots  # Should be all 0
        err_1 = counts1.get('0', 0) / shots  # Should be all 1
        
        avg_err = (err_0 + err_1) / 2
        
        results[q] = {
            'err_0': err_0,
            'err_1': err_1,
            'avg_err': avg_err
        }
        
        print(f"  Qubit {q}: |0〉→1 err={err_0:.3f}, |1〉→0 err={err_1:.3f}, avg={avg_err:.3f}")
    
    return results

def run_z_basis_rep_code(data_qubits: list, ancilla: int, shots: int, label: str):
    """Run Z-basis repetition code on specified qubits."""
    print(f"\n--- {label}: data={data_qubits}, ancilla={ancilla} ---")
    
    circuit = Circuit()
    
    # Initialize logical |0〉 (all zeros)
    for q in data_qubits:
        circuit.i(q)
    
    # One stabilizer round
    for q in data_qubits:
        circuit.cnot(q, ancilla)
    
    # Measure all
    for q in data_qubits:
        circuit.measure(q)
    circuit.measure(ancilla)
    
    task = device.run(circuit, shots=shots)
    result = task.result()
    counts = result.measurement_counts
    
    # Expected outcomes: all zeros if no error
    n_qubits = len(data_qubits) + 1
    expected = '0' * n_qubits
    
    success_count = counts.get(expected, 0)
    ler = 1.0 - success_count / shots
    
    print(f"  LER = {ler:.4f} ({success_count}/{shots} correct)")
    print(f"  Top outcomes: {dict(sorted(counts.items(), key=lambda x: -x[1])[:5])}")
    
    return ler, counts

# Main characterization
print("\n" + "="*70)
print("PHASE 1: SINGLE-QUBIT ERROR RATES")
print("="*70)

optimal_err = measure_single_qubit_error(OPTIMAL_QUBITS, SHOTS, "OPTIMAL")
stale_err = measure_single_qubit_error(STALE_QUBITS, SHOTS, "STALE")

print("\n" + "="*70)
print("PHASE 2: REPETITION CODE COMPARISON")
print("="*70)

# Using qubit 30 as ancilla for optimal, qubit 7 for stale
optimal_ler, _ = run_z_basis_rep_code(OPTIMAL_QUBITS, 30, SHOTS, "OPTIMAL")
stale_ler, _ = run_z_basis_rep_code(STALE_QUBITS, 7, SHOTS, "STALE")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print("\nSingle-qubit errors:")
print(f"  OPTIMAL (center): mean={np.mean([r['avg_err'] for r in optimal_err.values()]):.4f}")
print(f"  STALE (edge):     mean={np.mean([r['avg_err'] for r in stale_err.values()]):.4f}")

print("\nRepetition code LER:")
print(f"  OPTIMAL (center): {optimal_ler:.4f}")
print(f"  STALE (edge):     {stale_ler:.4f}")

diff = stale_ler - optimal_ler
print(f"\nDifference (STALE - OPTIMAL): {diff:+.4f}")
if diff > 0:
    print("  → STALE qubits are WORSE (as expected)")
else:
    print("  → STALE qubits are BETTER or SAME (unexpected!)")

# Save results
results = {
    'timestamp': datetime.now().isoformat(),
    'device': device.name,
    'shots': SHOTS,
    'single_qubit_errors': {
        'optimal': optimal_err,
        'stale': stale_err
    },
    'rep_code_ler': {
        'optimal': optimal_ler,
        'stale': stale_ler,
        'difference': diff
    }
}

results_dir = Path(__file__).parent.parent / "results" / "multi_platform"
results_dir.mkdir(parents=True, exist_ok=True)
results_file = results_dir / f"iqm_qubit_characterization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

with open(results_file, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to: {results_file}")
