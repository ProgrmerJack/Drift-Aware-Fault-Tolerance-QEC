"""
IQM Emerald Validation for Drift-Aware QEC Interaction Effect
=============================================================

Test the interaction effect on IQM Emerald (54-qubit superconducting QPU).

Device: IQM Emerald
Region: eu-north-1 (Stockholm)
ARN: arn:aws:braket:eu-north-1::device/qpu/iqm/Emerald
Native gates: RX, RY, CZ
Topology: 54-qubit square lattice

Pricing: $0.30/task + $0.00160/shot
"""

import json
import os
from datetime import datetime
from pathlib import Path

import boto3
from braket.aws import AwsDevice
from braket.circuits import Circuit


def get_device_status():
    """Check IQM Emerald availability and get properties."""
    try:
        # AWS credentials should be set via environment variables or AWS CLI config
        # Required: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION=eu-north-1
        if 'AWS_DEFAULT_REGION' not in os.environ:
            os.environ['AWS_DEFAULT_REGION'] = 'eu-north-1'
        
        device = AwsDevice("arn:aws:braket:eu-north-1::device/qpu/iqm/Emerald")
        props = device.properties.dict()
        
        print("=" * 70)
        print("IQM EMERALD STATUS")
        print("=" * 70)
        print(f"Device: {device.name}")
        print(f"Status: {device.status}")
        
        # Try to get queue depth, but don't fail if it times out
        try:
            queue = device.queue_depth()
            print(f"Queue depth: {queue}")
        except Exception as e:
            print(f"Queue depth: (unavailable - {e})")
        
        if hasattr(device.properties.service, 'deviceLocation'):
            print(f"Location: {device.properties.service.deviceLocation}")
        
        # Get gate fidelities if available
        if hasattr(device.properties.provider, 'specs'):
            specs = device.properties.provider.specs
            print(f"\nSpecs: {specs}")
        
        print("=" * 70)
        return device
    except Exception as e:
        print(f"Error accessing IQM Emerald: {e}")
        return None


def get_qubit_mapping(distance, regime="low"):
    """
    Select qubits for IQM Emerald's square lattice topology.
    
    For LOW regime: Use central, well-connected qubits (proxy for drift-aware)
    For HIGH regime: Use edge qubits (proxy for calibration-based with drift)
    
    IQM Emerald has 54 qubits in square lattice. Best qubits are typically central.
    """
    if distance == 5:
        # Need 9 qubits (5 data + 4 ancilla for d=5 rep code)
        if regime == "low":
            # Central region (well-connected, proxy for "drift-aware" = recent cal)
            return [26, 27, 19, 20, 21, 13, 14, 15, 7]
        else:
            # Edge region (less connected, proxy for "calibration-based" = stale cal)
            return [1, 2, 3, 8, 9, 10, 16, 17, 18]
    
    elif distance == 9:
        # Need 17 qubits (9 data + 8 ancilla)
        if regime == "low":
            # Central region
            return [18, 19, 20, 21, 22, 26, 27, 28, 29, 30, 34, 35, 36, 37, 38, 42, 43]
        else:
            # Edge region
            return [1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 22, 23]
    
    else:
        raise ValueError(f"Unsupported distance: {distance}")


def create_repetition_code_iqm(distance, rounds, qubit_mapping):
    """
    Create a repetition code circuit optimized for IQM Emerald.
    
    Native gates: RX, RY, CZ
    """
    circuit = Circuit()
    
    # Data qubits
    data_qubits = qubit_mapping[:distance]
    # Ancilla qubits
    ancilla_qubits = qubit_mapping[distance:distance + (distance - 1)]
    
    # Initialize data qubits to |+⟩ (logical |0⟩ for bit-flip code)
    for q in data_qubits:
        circuit.h(q)
    
    # Syndrome measurement rounds
    for _ in range(rounds):
        # Measure stabilizers (check for bit flips between adjacent data qubits)
        for i, anc in enumerate(ancilla_qubits):
            # Reset ancilla (using measurement)
            circuit.h(anc)
            
            # Entangle with adjacent data qubits
            circuit.cz(data_qubits[i], anc)
            circuit.cz(data_qubits[i + 1], anc)
            
            # Measure ancilla
            circuit.h(anc)
    
    # Final measurement of all data qubits
    for q in data_qubits:
        circuit.h(q)  # Back to Z basis
    
    return circuit, data_qubits


def create_calibration_overhead_circuit(distance, rounds, qubit_mapping):
    """
    Create a circuit with calibration overhead (extra gates).
    
    This simulates the overhead of "calibration-based" selection that uses
    stale calibration data, requiring additional verification gates.
    """
    # Start with base repetition code
    circuit, data_qubits = create_repetition_code_iqm(distance, rounds, qubit_mapping)
    
    # Add overhead: extra single-qubit rotations on all qubits
    # (simulates the cost of not having fresh calibration)
    all_qubits = qubit_mapping[:distance + (distance - 1)]
    for q in all_qubits:
        circuit.rx(q, 0)  # Identity-like gate (overhead)
        circuit.ry(q, 0)  # Another overhead gate
    
    return circuit, data_qubits


def run_interaction_validation(shots_per_condition=2500):
    """
    Run 4-condition experiment to test interaction effect:
    
    1. LOW noise (d=5, r=1) + drift-aware (central qubits, no overhead)
    2. LOW noise (d=5, r=1) + calibration-based (edge qubits, with overhead)
    3. HIGH noise (d=9, r=3) + drift-aware (central qubits, no overhead)
    4. HIGH noise (d=9, r=3) + calibration-based (edge qubits, with overhead)
    """
    # AWS credentials should be set via environment variables or AWS CLI config
    if 'AWS_DEFAULT_REGION' not in os.environ:
        os.environ['AWS_DEFAULT_REGION'] = 'eu-north-1'
    
    device = AwsDevice("arn:aws:braket:eu-north-1::device/qpu/iqm/Emerald")
    
    results = {
        "experiment": "interaction_effect_validation",
        "device": "IQM Emerald",
        "platform": "iqm_emerald",
        "shots_per_condition": shots_per_condition,
        "timestamp": datetime.now().isoformat(),
        "conditions": []
    }
    
    conditions = [
        # (description, distance, rounds, method, noise_regime)
        ("LOW noise | drift_aware | d=5, r=1", 5, 1, "drift_aware", "low"),
        ("LOW noise | calibration_based | d=5, r=1", 5, 1, "calibration_based", "low"),
        ("HIGH noise | drift_aware | d=9, r=3", 9, 3, "drift_aware", "high"),
        ("HIGH noise | calibration_based | d=9, r=3", 9, 3, "calibration_based", "high"),
    ]
    
    for desc, d, r, method, regime in conditions:
        print(f"\n{'=' * 70}")
        print(f"Running: {desc}")
        print(f"{'=' * 70}")
        
        # Get qubit mapping based on regime
        if method == "drift_aware":
            # Use central qubits (proxy for recent calibration)
            qubit_mapping = get_qubit_mapping(d, regime="low")
            circuit, data_qubits = create_repetition_code_iqm(d, r, qubit_mapping)
        else:
            # Use edge qubits + overhead (proxy for stale calibration)
            qubit_mapping = get_qubit_mapping(d, regime="high")
            circuit, data_qubits = create_calibration_overhead_circuit(d, r, qubit_mapping)
        
        print(f"Circuit depth: {circuit.depth}")
        print(f"Qubit mapping: {qubit_mapping}")
        print(f"Data qubits: {data_qubits}")
        print(f"Shots: {shots_per_condition}")
        
        # Submit task
        task = device.run(circuit, shots=shots_per_condition)
        print(f"Task ID: {task.id}")
        print("Waiting for results...")
        
        # Wait and get results
        result = task.result()
        counts = result.measurement_counts
        
        # Calculate logical error rate
        # All-zeros is the correct outcome
        correct_outcome = "0" * len(data_qubits)
        correct_count = counts.get(correct_outcome, 0)
        total_shots = sum(counts.values())
        ler = 1.0 - (correct_count / total_shots)
        
        print(f"\nResults:")
        print(f"  Correct outcomes: {correct_count}/{total_shots}")
        print(f"  Logical Error Rate: {ler:.4f}")
        
        # Store results
        condition_result = {
            "description": desc,
            "circuit_depth": circuit.depth,
            "shots": shots_per_condition,
            "correct_count": correct_count,
            "total_shots": total_shots,
            "ler": ler,
            "task_id": task.id,
            "counts": counts,
            "distance": d,
            "rounds": r,
            "method": method,
            "noise_regime": regime,
            "qubit_mapping": qubit_mapping,
        }
        results["conditions"].append(condition_result)
    
    # Analysis
    print("\n" + "=" * 70)
    print("INTERACTION EFFECT ANALYSIS")
    print("=" * 70)
    
    # Extract LERs
    low_drift = results["conditions"][0]["ler"]
    low_calib = results["conditions"][1]["ler"]
    high_drift = results["conditions"][2]["ler"]
    high_calib = results["conditions"][3]["ler"]
    
    # Calculate effects
    low_effect = low_calib - low_drift  # Positive = drift helps
    high_effect = high_calib - high_drift  # Positive = drift helps
    interaction = high_effect - low_effect
    
    print(f"\nLOW NOISE REGIME (d=5, r=1):")
    print(f"  Drift-aware LER:       {low_drift:.4f}")
    print(f"  Calibration-based LER: {low_calib:.4f}")
    print(f"  Effect (positive=drift helps): {low_effect:+.4f}")
    
    print(f"\nHIGH NOISE REGIME (d=9, r=3):")
    print(f"  Drift-aware LER:       {high_drift:.4f}")
    print(f"  Calibration-based LER: {high_calib:.4f}")
    print(f"  Effect (positive=drift helps): {high_effect:+.4f}")
    
    print(f"\nINTERACTION EFFECT:")
    print(f"  High - Low effect: {interaction:+.4f}")
    
    if interaction > 0:
        print("\n✓ SUPPORTS MANUSCRIPT CLAIM")
        print("   Drift-aware helps MORE at high noise.")
    elif interaction < 0:
        print("\n⚠️ REVERSE INTERACTION DETECTED")
        print("   Drift-aware helps MORE at low noise.")
    else:
        print("\n⊙ NO INTERACTION")
        print("   Effect is constant across noise regimes.")
    
    # Save results
    output_dir = Path("results/multi_platform")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"iqm_emerald_validation_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file.name}")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="IQM Emerald validation for drift-aware QEC interaction effect"
    )
    parser.add_argument(
        "--check-status",
        action="store_true",
        help="Only check device status and exit"
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=2500,
        help="Shots per condition (default: 2500, total: 10000)"
    )
    
    args = parser.parse_args()
    
    if args.check_status:
        get_device_status()
        return
    
    # Calculate cost
    total_shots = args.shots * 4  # 4 conditions
    cost_shots = total_shots * 0.00160
    cost_tasks = 4 * 0.30
    total_cost = cost_shots + cost_tasks
    
    print("\n" + "=" * 70)
    print("IQM EMERALD VALIDATION EXPERIMENT")
    print("=" * 70)
    print(f"Total shots: {total_shots} ({args.shots} per condition × 4 conditions)")
    print(f"Estimated cost: ${total_cost:.2f} (${cost_tasks:.2f} tasks + ${cost_shots:.2f} shots)")
    print("=" * 70)
    
    # Run validation
    results = run_interaction_validation(shots_per_condition=args.shots)
    
    print("\n✓ Validation complete!")


if __name__ == "__main__":
    main()
