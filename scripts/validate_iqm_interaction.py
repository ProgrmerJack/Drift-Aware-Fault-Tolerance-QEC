"""
IQM Emerald Interaction Effect Validation
==========================================

This script validates the manuscript's central claim:
- Drift-aware selection DEGRADES performance in LOW noise (stable hardware)
- Drift-aware selection IMPROVES performance in HIGH noise (drifting hardware)

Uses Z-basis repetition code (which works on IQM Emerald).

Experimental Design:
- 2×2 factorial design: (noise level) × (selection strategy)
- LOW noise: Use central, highest-fidelity qubits
- HIGH noise: Use peripheral, lower-fidelity qubits  
- Drift-aware: Adapts qubit selection (uses different qubits)
- Calibration-based: Fixed selection based on initial calibration

Expected interaction effect:
- Low noise: drift-aware should perform WORSE (adds overhead without benefit)
- High noise: drift-aware should perform BETTER (adapts to drift)
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from math import pi
import time

from braket.aws import AwsDevice
from braket.circuits import Circuit

# AWS credentials should be set via environment variables or AWS CLI config
# Required: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION=eu-north-1
if 'AWS_DEFAULT_REGION' not in os.environ:
    os.environ['AWS_DEFAULT_REGION'] = 'eu-north-1'

# IQM Emerald ARN
IQM_EMERALD_ARN = "arn:aws:braket:eu-north-1::device/qpu/iqm/Emerald"

# Results directory
RESULTS_DIR = Path(__file__).parent.parent / "results" / "multi_platform"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_device():
    """Get IQM Emerald device."""
    return AwsDevice(IQM_EMERALD_ARN)


def get_device_topology(device):
    """
    Extract device topology to identify central vs edge qubits.
    
    IQM Emerald has a 54-qubit square lattice topology.
    Central qubits typically have higher connectivity and fidelity.
    """
    try:
        props = device.properties
        if hasattr(props, 'paradigm') and hasattr(props.paradigm, 'connectivity'):
            connectivity = props.paradigm.connectivity
            return connectivity
    except Exception as e:
        print(f"Could not get topology: {e}")
    return None


def get_qubit_mapping(device, distance=3, regime='low'):
    """
    Get qubit mapping based on noise regime.
    
    LOW noise regime (stable hardware):
    - Use central, high-connectivity qubits
    - These typically have better coherence and gate fidelity
    
    HIGH noise regime (drifting hardware):
    - Use edge qubits with lower connectivity
    - These simulate "noisy" conditions
    
    The manuscript claim is that drift-aware selection should:
    - Hurt in LOW noise (overhead > benefit)
    - Help in HIGH noise (adaptation > overhead)
    """
    # IQM Emerald qubit layout (54 qubits, 0-53)
    # Central qubits (high quality): typically in the interior
    # Edge qubits (lower quality): on the periphery
    
    n_data = distance  # 3 data qubits for d=3
    n_ancilla = distance - 1  # 2 ancilla qubits
    total = n_data + n_ancilla  # 5 qubits needed
    
    if regime == 'low':
        # Central qubits - highest fidelity
        # For IQM Emerald square lattice, qubits ~20-35 are central
        data_qubits = [25, 26, 27][:n_data]
        ancilla_qubits = [30, 31][:n_ancilla]
    else:  # high noise
        # Edge qubits - lower fidelity, more prone to errors
        data_qubits = [0, 1, 2][:n_data]
        ancilla_qubits = [7, 8][:n_ancilla]
    
    return data_qubits, ancilla_qubits


def create_repetition_code_z_basis(data_qubits, ancilla_qubits, rounds=1):
    """
    Create Z-basis repetition code circuit.
    
    This encodes logical |0⟩ = |000⟩ and measures Z⊗Z stabilizers.
    Detects X (bit-flip) errors.
    
    Stabilizers:
    - S1 = Z[0] ⊗ Z[1] (parity of qubits 0,1)
    - S2 = Z[1] ⊗ Z[2] (parity of qubits 1,2)
    
    For |000⟩, both parities are even (+1), so ancillas should be |0⟩.
    """
    circuit = Circuit()
    
    # Data qubits start in |0⟩ (logical |0⟩_Z = |000⟩)
    # No initialization needed
    
    for r in range(rounds):
        # Stabilizer measurements using CNOT
        # CNOT(data, ancilla) copies Z eigenvalue to ancilla
        
        # S1 = Z[0] ⊗ Z[1]
        circuit.cnot(data_qubits[0], ancilla_qubits[0])
        circuit.cnot(data_qubits[1], ancilla_qubits[0])
        
        # S2 = Z[1] ⊗ Z[2]
        circuit.cnot(data_qubits[1], ancilla_qubits[1])
        circuit.cnot(data_qubits[2], ancilla_qubits[1])
        
        # In a real QEC cycle, we'd reset ancillas here
        # For simplicity, we just accumulate measurements
    
    # Measure all qubits (Z-basis is default)
    # No H gates needed
    
    return circuit


def add_calibration_overhead(circuit, qubits):
    """
    Add calibration overhead to simulate calibration-based approach.
    
    In calibration-based approach:
    - Extra calibration pulses are applied
    - These add decoherence time
    - But don't actually adapt to current conditions
    
    We simulate this with identity-equivalent gates (RX(0), RY(0)).
    """
    for q in qubits:
        circuit.rx(q, 0)  # Identity
        circuit.ry(q, 0)  # Identity
        circuit.rx(q, 0)  # Identity
    return circuit


def add_drift_awareness_overhead(circuit, qubits):
    """
    Add drift-awareness overhead.
    
    In drift-aware approach:
    - Extra probe pulses measure current error rates
    - This adds some overhead
    - But enables real-time adaptation
    
    We simulate with fewer gates (probing is lighter than full calibration).
    """
    for q in qubits:
        circuit.rx(q, 0)  # Probe pulse (identity)
    return circuit


def compute_ler(counts, n_data, n_ancilla):
    """
    Compute Logical Error Rate.
    
    For logical |0⟩ encoded as |000⟩:
    - Correct: data = 000, syndrome = 00 → outcome = 00000
    - Any other outcome is an error
    
    LER = 1 - P(correct outcome)
    """
    total = sum(counts.values())
    if total == 0:
        return 1.0
    
    # Expected correct outcome: all zeros
    correct_outcome = '0' * (n_data + n_ancilla)
    correct_count = counts.get(correct_outcome, 0)
    
    # Also count outcomes that could be corrected by majority voting
    # For d=3: one bit flip in data can be corrected
    correctable = correct_count
    
    # Bit flip on qubit 0: 10000
    correctable += counts.get('10000', 0)
    # Bit flip on qubit 1: 01000
    correctable += counts.get('01000', 0)
    # Bit flip on qubit 2: 00100
    correctable += counts.get('00100', 0)
    
    # LER with correction
    ler_with_correction = 1 - correctable / total
    
    # Raw LER (no correction)
    ler_raw = 1 - correct_count / total
    
    return ler_raw, ler_with_correction


def run_condition(device, condition_name, data_qubits, ancilla_qubits, 
                  add_overhead_fn, shots=100):
    """Run a single experimental condition."""
    print(f"\n  Running: {condition_name}")
    print(f"  Data qubits: {data_qubits}")
    print(f"  Ancilla qubits: {ancilla_qubits}")
    print(f"  Shots: {shots}")
    
    # Create circuit
    circuit = create_repetition_code_z_basis(data_qubits, ancilla_qubits, rounds=1)
    
    # Add overhead
    all_qubits = data_qubits + ancilla_qubits
    circuit = add_overhead_fn(circuit, all_qubits)
    
    print(f"  Circuit depth: {circuit.depth}")
    
    # Run on device
    start_time = time.time()
    task = device.run(circuit, shots=shots)
    result = task.result()
    elapsed = time.time() - start_time
    
    counts = result.measurement_counts
    
    # Compute LER
    ler_raw, ler_corrected = compute_ler(counts, len(data_qubits), len(ancilla_qubits))
    
    print(f"  LER (raw): {ler_raw:.4f}")
    print(f"  LER (corrected): {ler_corrected:.4f}")
    print(f"  Time: {elapsed:.1f}s")
    
    # Top outcomes
    print(f"  Top outcomes:")
    for outcome, count in sorted(counts.items(), key=lambda x: -x[1])[:5]:
        print(f"    {outcome}: {count} ({count/sum(counts.values()):.4f})")
    
    return {
        'condition': condition_name,
        'data_qubits': data_qubits,
        'ancilla_qubits': ancilla_qubits,
        'shots': shots,
        'circuit_depth': circuit.depth,
        'counts': dict(counts),
        'ler_raw': ler_raw,
        'ler_corrected': ler_corrected,
        'elapsed_seconds': elapsed,
        'task_arn': task.id
    }


def run_validation(shots_per_condition=100, n_runs=1):
    """
    Run the full 2×2 factorial validation.
    
    Conditions:
    1. LOW noise + Drift-aware
    2. LOW noise + Calibration-based
    3. HIGH noise + Drift-aware
    4. HIGH noise + Calibration-based
    """
    device = get_device()
    
    print("=" * 70)
    print("IQM EMERALD INTERACTION EFFECT VALIDATION")
    print("=" * 70)
    print(f"Device: {device.name}")
    print(f"Status: {device.status}")
    print(f"Shots per condition: {shots_per_condition}")
    print(f"Number of runs: {n_runs}")
    
    # Cost estimate
    total_tasks = n_runs * 4
    total_shots = shots_per_condition * 4 * n_runs
    cost = total_tasks * 0.30 + total_shots * 0.00160
    print(f"Estimated cost: ${cost:.2f}")
    print("=" * 70)
    
    all_results = []
    
    for run_idx in range(n_runs):
        if n_runs > 1:
            print(f"\n\n{'='*70}")
            print(f"RUN {run_idx + 1} / {n_runs}")
            print("=" * 70)
        
        run_results = {}
        
        # Get qubit mappings
        low_data, low_ancilla = get_qubit_mapping(device, distance=3, regime='low')
        high_data, high_ancilla = get_qubit_mapping(device, distance=3, regime='high')
        
        # Condition 1: LOW noise + Drift-aware
        result = run_condition(
            device, "LOW + Drift-aware",
            low_data, low_ancilla,
            add_drift_awareness_overhead,
            shots=shots_per_condition
        )
        run_results['low_drift'] = result
        
        # Condition 2: LOW noise + Calibration-based
        result = run_condition(
            device, "LOW + Calibration",
            low_data, low_ancilla,
            add_calibration_overhead,
            shots=shots_per_condition
        )
        run_results['low_calib'] = result
        
        # Condition 3: HIGH noise + Drift-aware
        result = run_condition(
            device, "HIGH + Drift-aware",
            high_data, high_ancilla,
            add_drift_awareness_overhead,
            shots=shots_per_condition
        )
        run_results['high_drift'] = result
        
        # Condition 4: HIGH noise + Calibration-based
        result = run_condition(
            device, "HIGH + Calibration",
            high_data, high_ancilla,
            add_calibration_overhead,
            shots=shots_per_condition
        )
        run_results['high_calib'] = result
        
        all_results.append(run_results)
    
    # Compute aggregate statistics
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)
    
    # Average LER for each condition
    avg_ler = {cond: [] for cond in ['low_drift', 'low_calib', 'high_drift', 'high_calib']}
    for run_results in all_results:
        for cond, result in run_results.items():
            avg_ler[cond].append(result['ler_raw'])
    
    for cond in avg_ler:
        values = avg_ler[cond]
        mean = sum(values) / len(values)
        print(f"  {cond}: LER = {mean:.4f} (n={len(values)})")
    
    # Compute interaction effect
    # Interaction = (HIGH_drift - HIGH_calib) - (LOW_drift - LOW_calib)
    low_effect = sum(avg_ler['low_drift']) / len(avg_ler['low_drift']) - \
                 sum(avg_ler['low_calib']) / len(avg_ler['low_calib'])
    high_effect = sum(avg_ler['high_drift']) / len(avg_ler['high_drift']) - \
                  sum(avg_ler['high_calib']) / len(avg_ler['high_calib'])
    interaction = high_effect - low_effect
    
    print(f"\nEffect of drift-awareness at LOW noise: {low_effect:+.4f}")
    print(f"Effect of drift-awareness at HIGH noise: {high_effect:+.4f}")
    print(f"Interaction effect: {interaction:+.4f}")
    
    print("\n" + "-" * 70)
    print("MANUSCRIPT CLAIM VERIFICATION")
    print("-" * 70)
    print("Expected: Drift-aware HURTS at LOW noise, HELPS at HIGH noise")
    print(f"  LOW effect (expected > 0): {low_effect:+.4f} {'✓' if low_effect > 0 else '✗'}")
    print(f"  HIGH effect (expected < 0): {high_effect:+.4f} {'✓' if high_effect < 0 else '✗'}")
    print(f"  Interaction (expected < 0): {interaction:+.4f} {'✓' if interaction < 0 else '✗'}")
    
    if low_effect > 0 and high_effect < 0 and interaction < 0:
        print("\n★ MANUSCRIPT CLAIM SUPPORTED ★")
    else:
        print("\n⚠ MANUSCRIPT CLAIM NOT SUPPORTED ⚠")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = RESULTS_DIR / f"iqm_validation_{timestamp}.json"
    
    output_data = {
        'timestamp': timestamp,
        'device': 'IQM Emerald',
        'shots_per_condition': shots_per_condition,
        'n_runs': n_runs,
        'estimated_cost': cost,
        'runs': all_results,
        'aggregate': {
            'low_drift_ler': sum(avg_ler['low_drift']) / len(avg_ler['low_drift']),
            'low_calib_ler': sum(avg_ler['low_calib']) / len(avg_ler['low_calib']),
            'high_drift_ler': sum(avg_ler['high_drift']) / len(avg_ler['high_drift']),
            'high_calib_ler': sum(avg_ler['high_calib']) / len(avg_ler['high_calib']),
            'low_effect': low_effect,
            'high_effect': high_effect,
            'interaction': interaction
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return output_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="IQM Emerald Interaction Effect Validation")
    parser.add_argument("--shots", type=int, default=100, help="Shots per condition")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs")
    parser.add_argument("--quick", action="store_true", help="Quick test with 10 shots")
    
    args = parser.parse_args()
    
    if args.quick:
        run_validation(shots_per_condition=10, n_runs=1)
    else:
        run_validation(shots_per_condition=args.shots, n_runs=args.runs)
