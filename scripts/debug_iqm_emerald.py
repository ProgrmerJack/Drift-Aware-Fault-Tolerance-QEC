"""
IQM Emerald Deep Debug and Validation
=====================================

Comprehensive debugging to understand IQM Emerald's behavior and validate
the manuscript's drift-aware QEC claim.

Strategy:
1. Basic tests (single qubit, Bell pair) - 10-100 shots
2. Understand measurement conventions
3. Fix QEC circuit
4. Scale up for statistical validation (1000 shots × 10 runs)
"""

import json
import os
from datetime import datetime
from pathlib import Path
from math import pi

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


def test_single_qubit_init(device, shots=10):
    """
    Test 1: Single qubit initialization and measurement.
    
    Expected: Qubit starts in |0⟩, should measure 0 most of the time.
    """
    print("\n" + "=" * 70)
    print("TEST 1: Single Qubit |0⟩ State")
    print("=" * 70)
    
    # Simple circuit: just measure qubit 1 (no gates)
    circuit = Circuit()
    # No gates - qubit should be |0⟩
    
    print(f"Circuit: (no gates, direct measurement)")
    print(f"Shots: {shots}")
    
    task = device.run(circuit, shots=shots)
    result = task.result()
    counts = result.measurement_counts
    
    print(f"Results: {counts}")
    
    # Calculate |0⟩ probability
    total = sum(counts.values())
    zeros = counts.get('0', 0)
    p_zero = zeros / total if total > 0 else 0
    
    print(f"P(|0⟩) = {p_zero:.4f}")
    print(f"Expected: ~1.0 (initialization fidelity)")
    
    return {'test': 'single_qubit_0', 'counts': counts, 'p_zero': p_zero}


def test_single_qubit_x(device, shots=10):
    """
    Test 2: X gate (bit flip).
    
    Apply X to |0⟩ → |1⟩, should measure 1 most of the time.
    IQM native: X = RX(π)
    """
    print("\n" + "=" * 70)
    print("TEST 2: Single Qubit X Gate (|0⟩ → |1⟩)")
    print("=" * 70)
    
    circuit = Circuit()
    circuit.x(1)  # Braket should transpile to RX(π)
    
    print(f"Circuit: X(1)")
    print(f"Shots: {shots}")
    
    task = device.run(circuit, shots=shots)
    result = task.result()
    counts = result.measurement_counts
    
    print(f"Results: {counts}")
    
    total = sum(counts.values())
    ones = counts.get('1', 0)
    p_one = ones / total if total > 0 else 0
    
    print(f"P(|1⟩) = {p_one:.4f}")
    print(f"Expected: ~1.0 (X gate fidelity)")
    
    return {'test': 'single_qubit_x', 'counts': counts, 'p_one': p_one}


def test_hadamard(device, shots=10):
    """
    Test 3: Hadamard gate (superposition).
    
    Apply H to |0⟩ → |+⟩, should measure 0 and 1 with ~50% each.
    IQM native: H = RZ(π/2) · RX(π/2) · RZ(π/2) or similar decomposition
    """
    print("\n" + "=" * 70)
    print("TEST 3: Hadamard Gate (|0⟩ → |+⟩)")
    print("=" * 70)
    
    circuit = Circuit()
    circuit.h(1)
    
    print(f"Circuit: H(1)")
    print(f"Shots: {shots}")
    
    task = device.run(circuit, shots=shots)
    result = task.result()
    counts = result.measurement_counts
    
    print(f"Results: {counts}")
    
    total = sum(counts.values())
    zeros = counts.get('0', 0)
    ones = counts.get('1', 0)
    
    print(f"P(|0⟩) = {zeros/total:.4f}, P(|1⟩) = {ones/total:.4f}")
    print(f"Expected: ~0.5 each")
    
    return {'test': 'hadamard', 'counts': counts, 'p_zero': zeros/total, 'p_one': ones/total}


def test_bell_pair(device, shots=10):
    """
    Test 4: Bell pair (entanglement).
    
    H(1) · CNOT(1,2) → (|00⟩ + |11⟩)/√2
    Should measure 00 and 11 with ~50% each, never 01 or 10.
    
    IQM native: CNOT = H(target) · CZ(control, target) · H(target)
    """
    print("\n" + "=" * 70)
    print("TEST 4: Bell Pair (|00⟩ + |11⟩)/√2")
    print("=" * 70)
    
    circuit = Circuit()
    circuit.h(1)
    circuit.cnot(1, 2)
    
    print(f"Circuit: H(1) · CNOT(1,2)")
    print(f"Shots: {shots}")
    
    task = device.run(circuit, shots=shots)
    result = task.result()
    counts = result.measurement_counts
    
    print(f"Results: {counts}")
    
    total = sum(counts.values())
    p_00 = counts.get('00', 0) / total
    p_01 = counts.get('01', 0) / total
    p_10 = counts.get('10', 0) / total
    p_11 = counts.get('11', 0) / total
    
    print(f"P(|00⟩) = {p_00:.4f}, P(|01⟩) = {p_01:.4f}")
    print(f"P(|10⟩) = {p_10:.4f}, P(|11⟩) = {p_11:.4f}")
    print(f"Expected: P(00)≈P(11)≈0.5, P(01)≈P(10)≈0")
    print(f"Bell fidelity proxy: {p_00 + p_11:.4f}")
    
    return {'test': 'bell_pair', 'counts': counts, 'bell_fidelity': p_00 + p_11}


def test_ghz_state(device, n_qubits=3, shots=10):
    """
    Test 5: GHZ state (multi-qubit entanglement).
    
    H(1) · CNOT(1,2) · CNOT(2,3) · ... → (|00...0⟩ + |11...1⟩)/√2
    """
    print("\n" + "=" * 70)
    print(f"TEST 5: GHZ State ({n_qubits} qubits)")
    print("=" * 70)
    
    qubits = list(range(1, n_qubits + 1))
    
    circuit = Circuit()
    circuit.h(qubits[0])
    for i in range(len(qubits) - 1):
        circuit.cnot(qubits[i], qubits[i + 1])
    
    print(f"Circuit: H({qubits[0]}) · CNOTs")
    print(f"Qubits: {qubits}")
    print(f"Shots: {shots}")
    
    task = device.run(circuit, shots=shots)
    result = task.result()
    counts = result.measurement_counts
    
    print(f"Results: {counts}")
    
    total = sum(counts.values())
    all_zeros = '0' * n_qubits
    all_ones = '1' * n_qubits
    p_000 = counts.get(all_zeros, 0) / total
    p_111 = counts.get(all_ones, 0) / total
    
    print(f"P(|{all_zeros}⟩) = {p_000:.4f}, P(|{all_ones}⟩) = {p_111:.4f}")
    print(f"GHZ fidelity proxy: {p_000 + p_111:.4f}")
    
    return {'test': f'ghz_{n_qubits}', 'counts': counts, 'ghz_fidelity': p_000 + p_111}


def test_simple_repetition_code(device, shots=10):
    """
    Test 6: Simplest repetition code (d=3, r=0).
    
    Encode |0⟩_L = |000⟩ using CNOT fan-out.
    No syndrome measurement, just check if we can prepare |000⟩.
    """
    print("\n" + "=" * 70)
    print("TEST 6: Simple Repetition Code Encoding (d=3)")
    print("=" * 70)
    
    # Data qubits: 1, 2, 3
    circuit = Circuit()
    
    # Encode: start with |000⟩, do nothing (already encoded)
    # Just measure
    
    print(f"Circuit: (no gates - prepare |000⟩)")
    print(f"Data qubits: [1, 2, 3]")
    print(f"Shots: {shots}")
    
    task = device.run(circuit, shots=shots)
    result = task.result()
    counts = result.measurement_counts
    
    print(f"Results: {counts}")
    
    total = sum(counts.values())
    p_000 = counts.get('000', 0) / total
    
    print(f"P(|000⟩) = {p_000:.4f}")
    print(f"Expected: ~1.0")
    
    return {'test': 'rep_code_d3_encode', 'counts': counts, 'p_logical_0': p_000}


def test_rep_code_with_syndrome(device, shots=10):
    """
    Test 7: Repetition code with syndrome measurement.
    
    d=3: data qubits [1,2,3], ancilla qubits [4,5]
    
    Syndrome checks:
    - Ancilla 4 measures Z1⊗Z2 (parity of qubits 1,2)
    - Ancilla 5 measures Z2⊗Z3 (parity of qubits 2,3)
    
    For |000⟩, both parities should be +1 (even), so ancillas should be |0⟩.
    """
    print("\n" + "=" * 70)
    print("TEST 7: Repetition Code with Syndrome Measurement")
    print("=" * 70)
    
    data_qubits = [1, 2, 3]
    ancilla_qubits = [4, 5]
    
    circuit = Circuit()
    
    # Data starts in |000⟩ (already logical |0⟩)
    
    # Syndrome measurement using CNOT
    # Ancilla 4: CNOT(1,4) · CNOT(2,4) measures parity of qubits 1,2
    circuit.cnot(data_qubits[0], ancilla_qubits[0])
    circuit.cnot(data_qubits[1], ancilla_qubits[0])
    
    # Ancilla 5: CNOT(2,5) · CNOT(3,5) measures parity of qubits 2,3
    circuit.cnot(data_qubits[1], ancilla_qubits[1])
    circuit.cnot(data_qubits[2], ancilla_qubits[1])
    
    print(f"Circuit: Syndrome extraction for d=3 rep code")
    print(f"Data qubits: {data_qubits}")
    print(f"Ancilla qubits: {ancilla_qubits}")
    print(f"Shots: {shots}")
    print(f"Circuit depth: {circuit.depth}")
    
    task = device.run(circuit, shots=shots)
    result = task.result()
    counts = result.measurement_counts
    
    print(f"\nResults: {counts}")
    print(f"\nMeasurement order: qubits 1,2,3,4,5 (data + ancilla)")
    
    total = sum(counts.values())
    
    # Correct outcome: |00000⟩ (all zeros - no errors, trivial syndrome)
    p_correct = counts.get('00000', 0) / total
    
    # Analyze outcomes
    print("\nOutcome analysis:")
    for outcome, count in sorted(counts.items(), key=lambda x: -x[1])[:10]:
        data_bits = outcome[:3]
        syndrome_bits = outcome[3:]
        p = count / total
        print(f"  {outcome} (data={data_bits}, syndrome={syndrome_bits}): {p:.4f} ({count}/{total})")
    
    print(f"\nP(|00000⟩) = {p_correct:.4f}")
    print(f"Expected: ~1.0 (no errors)")
    
    return {'test': 'rep_code_syndrome', 'counts': counts, 'p_correct': p_correct}


def test_rep_code_x_basis(device, shots=10):
    """
    Test 8: Repetition code in X-basis (bit-flip code).
    
    This is what the manuscript actually uses!
    
    Logical |0⟩_X = |+++⟩ = H⊗3|000⟩
    Measure in X-basis at the end: H then measure.
    
    Z-type errors (phase flips) become detectable.
    """
    print("\n" + "=" * 70)
    print("TEST 8: Repetition Code (X-Basis / Bit-Flip Code)")
    print("=" * 70)
    
    data_qubits = [1, 2, 3]
    ancilla_qubits = [4, 5]
    
    circuit = Circuit()
    
    # Encode into X-basis: |000⟩ → |+++⟩
    for q in data_qubits:
        circuit.h(q)
    
    # X-basis stabilizer measurement (Z⊗Z parity)
    # In X-basis, this detects bit flips
    
    # Method: Use CZ gates (native to IQM)
    # CZ flips phase based on both qubits being |1⟩
    # For X-basis measurement, we need to detect X errors
    
    # Stabilizer S1 = X1⊗X2: check if both qubits have same X eigenvalue
    # Using Hadamard transform: H·Z·H = X
    # So: H(anc) · CZ(1,anc) · CZ(2,anc) · H(anc) measures X1⊗X2
    
    circuit.h(ancilla_qubits[0])
    circuit.cz(data_qubits[0], ancilla_qubits[0])
    circuit.cz(data_qubits[1], ancilla_qubits[0])
    circuit.h(ancilla_qubits[0])
    
    circuit.h(ancilla_qubits[1])
    circuit.cz(data_qubits[1], ancilla_qubits[1])
    circuit.cz(data_qubits[2], ancilla_qubits[1])
    circuit.h(ancilla_qubits[1])
    
    # Measure data qubits in X-basis
    for q in data_qubits:
        circuit.h(q)
    
    print(f"Circuit: X-basis rep code with CZ stabilizers")
    print(f"Data qubits: {data_qubits}")
    print(f"Ancilla qubits: {ancilla_qubits}")
    print(f"Shots: {shots}")
    print(f"Circuit depth: {circuit.depth}")
    
    task = device.run(circuit, shots=shots)
    result = task.result()
    counts = result.measurement_counts
    
    print(f"\nResults: {counts}")
    
    total = sum(counts.values())
    
    # Correct outcome: |00000⟩
    # Data in |000⟩ (measured after H, so |+++⟩ → |000⟩)
    # Syndrome in |00⟩ (no errors detected)
    p_correct = counts.get('00000', 0) / total
    
    print("\nOutcome analysis:")
    for outcome, count in sorted(counts.items(), key=lambda x: -x[1])[:10]:
        data_bits = outcome[:3]
        syndrome_bits = outcome[3:]
        p = count / total
        print(f"  {outcome} (data={data_bits}, syndrome={syndrome_bits}): {p:.4f} ({count}/{total})")
    
    print(f"\nP(|00000⟩) = {p_correct:.4f}")
    
    return {'test': 'rep_code_x_basis', 'counts': counts, 'p_correct': p_correct}


def test_z_basis_rep_code(device, shots=10):
    """
    Test 9: Z-basis repetition code (phase-flip code).
    
    Logical |0⟩_Z = |000⟩
    Stabilizers: Z1⊗Z2, Z2⊗Z3
    Detects X (bit-flip) errors.
    
    This is the standard repetition code!
    """
    print("\n" + "=" * 70)
    print("TEST 9: Z-Basis Repetition Code (Phase-Flip Code)")
    print("=" * 70)
    
    data_qubits = [1, 2, 3]
    ancilla_qubits = [4, 5]
    
    circuit = Circuit()
    
    # Data starts in |000⟩ (logical |0⟩_Z)
    # No initialization gates needed
    
    # Z⊗Z stabilizer measurement using CNOT
    # CNOT(data, ancilla) copies Z eigenvalue to ancilla
    
    # S1 = Z1⊗Z2
    circuit.cnot(data_qubits[0], ancilla_qubits[0])
    circuit.cnot(data_qubits[1], ancilla_qubits[0])
    
    # S2 = Z2⊗Z3  
    circuit.cnot(data_qubits[1], ancilla_qubits[1])
    circuit.cnot(data_qubits[2], ancilla_qubits[1])
    
    # Measure everything in Z-basis (default)
    # No H gates at the end
    
    print(f"Circuit: Z-basis rep code (detects X errors)")
    print(f"Data qubits: {data_qubits}")
    print(f"Ancilla qubits: {ancilla_qubits}")
    print(f"Shots: {shots}")
    print(f"Circuit depth: {circuit.depth}")
    
    task = device.run(circuit, shots=shots)
    result = task.result()
    counts = result.measurement_counts
    
    print(f"\nResults: {counts}")
    
    total = sum(counts.values())
    p_correct = counts.get('00000', 0) / total
    
    print("\nOutcome analysis:")
    for outcome, count in sorted(counts.items(), key=lambda x: -x[1])[:10]:
        data_bits = outcome[:3]
        syndrome_bits = outcome[3:]
        p = count / total
        print(f"  {outcome} (data={data_bits}, syndrome={syndrome_bits}): {p:.4f}")
    
    print(f"\nP(|00000⟩) = {p_correct:.4f}")
    
    return {'test': 'rep_code_z_basis', 'counts': counts, 'p_correct': p_correct}


def run_all_basic_tests():
    """Run all basic tests with minimal shots."""
    device = get_device()
    
    print("\n" + "=" * 70)
    print("IQM EMERALD BASIC DIAGNOSTIC TESTS")
    print("=" * 70)
    print(f"Device: {device.name}")
    print(f"Status: {device.status}")
    print("=" * 70)
    
    results = []
    
    # Run tests with 10 shots each (minimal cost)
    tests = [
        (test_single_qubit_init, {}),
        (test_single_qubit_x, {}),
        (test_hadamard, {}),
        (test_bell_pair, {}),
        (test_ghz_state, {'n_qubits': 3}),
        (test_simple_repetition_code, {}),
        (test_rep_code_with_syndrome, {}),
        (test_rep_code_x_basis, {}),
        (test_z_basis_rep_code, {}),
    ]
    
    for test_func, kwargs in tests:
        try:
            result = test_func(device, shots=10, **kwargs)
            results.append(result)
        except Exception as e:
            print(f"\nERROR in {test_func.__name__}: {e}")
            results.append({'test': test_func.__name__, 'error': str(e)})
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = RESULTS_DIR / f"iqm_debug_basic_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'device': 'IQM Emerald',
            'tests': results
        }, f, indent=2)
    
    print(f"\n\nResults saved to: {output_file}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in results:
        if 'error' in r:
            print(f"  {r['test']}: ERROR - {r['error']}")
        elif 'p_zero' in r:
            print(f"  {r['test']}: P(0) = {r['p_zero']:.4f}")
        elif 'p_one' in r:
            print(f"  {r['test']}: P(1) = {r['p_one']:.4f}")
        elif 'bell_fidelity' in r:
            print(f"  {r['test']}: Bell fidelity = {r['bell_fidelity']:.4f}")
        elif 'ghz_fidelity' in r:
            print(f"  {r['test']}: GHZ fidelity = {r['ghz_fidelity']:.4f}")
        elif 'p_logical_0' in r:
            print(f"  {r['test']}: P(logical 0) = {r['p_logical_0']:.4f}")
        elif 'p_correct' in r:
            print(f"  {r['test']}: P(correct) = {r['p_correct']:.4f}")
    
    return results


def run_scaled_validation(shots_per_run=1000, n_runs=10):
    """
    Run scaled validation once basic tests pass.
    
    This will test the interaction effect claim with proper statistics.
    """
    print("\n" + "=" * 70)
    print("SCALED VALIDATION FOR INTERACTION EFFECT")
    print("=" * 70)
    print(f"Shots per run: {shots_per_run}")
    print(f"Number of runs: {n_runs}")
    print(f"Total shots per condition: {shots_per_run * n_runs}")
    
    # Cost estimate
    total_tasks = n_runs * 4  # 4 conditions per run
    total_shots = shots_per_run * n_runs * 4
    cost = total_tasks * 0.30 + total_shots * 0.00160
    print(f"Estimated cost: ${cost:.2f}")
    
    # Implementation would go here after basic tests pass
    pass


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="IQM Emerald Debug and Validation")
    parser.add_argument("--basic", action="store_true", help="Run basic diagnostic tests")
    parser.add_argument("--scaled", action="store_true", help="Run scaled validation")
    parser.add_argument("--test", type=int, help="Run specific test number (1-9)")
    parser.add_argument("--shots", type=int, default=10, help="Shots per test")
    
    args = parser.parse_args()
    
    if args.basic:
        run_all_basic_tests()
    elif args.test:
        device = get_device()
        tests = [
            test_single_qubit_init,
            test_single_qubit_x,
            test_hadamard,
            test_bell_pair,
            lambda d, s: test_ghz_state(d, 3, s),
            test_simple_repetition_code,
            test_rep_code_with_syndrome,
            test_rep_code_x_basis,
            test_z_basis_rep_code,
        ]
        if 1 <= args.test <= len(tests):
            tests[args.test - 1](device, args.shots)
        else:
            print(f"Invalid test number. Choose 1-{len(tests)}")
    else:
        print("Usage:")
        print("  --basic    Run all basic diagnostic tests (10 shots each)")
        print("  --test N   Run specific test number (1-9)")
        print("  --shots N  Set shots per test")
        print("  --scaled   Run full validation (after basic tests pass)")
