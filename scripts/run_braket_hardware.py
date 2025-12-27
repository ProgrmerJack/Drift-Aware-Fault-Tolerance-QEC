#!/usr/bin/env python
"""
Run Amazon Braket experiments on IonQ Forte-1 and IQM Emerald.

This script validates drift-aware QEC across multiple quantum hardware platforms
to address the single-platform limitation in the manuscript.

Platform Details:
-----------------
IonQ Forte-1 (Trapped Ion):
    - 36 qubits, all-to-all connectivity
    - Native gates: GPi, GPi2, MS (Mølmer-Sørensen)
    - Error mitigation: Debiasing available
    - Cost: $0.30/task + $0.01/shot
    - ARN: arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1

IQM Emerald (Superconducting):
    - 54 qubits, square lattice topology
    - Native gates: PRX, CZ
    - Median 1Q fidelity: 99.93%, 2Q fidelity: 99.5%
    - Cost: $0.30/task + $0.00145/shot
    - ARN: arn:aws:braket:eu-north-1::device/qpu/iqm/Emerald

Usage:
------
# Test on local simulator first
python run_braket_hardware.py --simulator --distance 3 --shots 100

# Run on IonQ Forte-1
python run_braket_hardware.py --device ionq --distance 3 --shots 100

# Run on IQM Emerald  
python run_braket_hardware.py --device iqm --distance 3 --shots 100
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Device ARNs and pricing
DEVICES = {
    "ionq": {
        "arn": "arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1",
        "name": "IonQ Forte-1",
        "type": "trapped-ion",
        "qubits": 36,
        "task_cost": 0.30,
        "shot_cost": 0.01,
        "region": "us-east-1",
    },
    "iqm": {
        "arn": "arn:aws:braket:eu-north-1::device/qpu/iqm/Emerald",
        "name": "IQM Emerald",
        "type": "superconducting",
        "qubits": 54,
        "task_cost": 0.30,
        "shot_cost": 0.00145,
        "region": "eu-north-1",
    },
}


def estimate_cost(device_key: str, shots: int, n_tasks: int = 1) -> float:
    """Estimate cost for running on a device."""
    if device_key not in DEVICES:
        return 0.0
    device = DEVICES[device_key]
    return n_tasks * device["task_cost"] + n_tasks * shots * device["shot_cost"]


def create_repetition_code_braket(n_data_qubits: int, n_rounds: int = 1):
    """
    Create a repetition code circuit using Braket SDK.
    
    For trapped-ion (IonQ): Use native MS gates if available
    For superconducting (IQM): Use CZ gates
    """
    from braket.circuits import Circuit
    
    n_ancilla = n_data_qubits - 1
    total_qubits = n_data_qubits + n_ancilla
    
    circuit = Circuit()
    
    # Initialize in |0...0> state (default)
    # For a proper QEC test, we prepare |+...+> or just test syndrome extraction
    
    # Syndrome extraction rounds
    for round_idx in range(n_rounds):
        # For each ancilla, do CNOT from adjacent data qubits
        for i in range(n_ancilla):
            data1 = i
            data2 = i + 1
            ancilla = n_data_qubits + i
            
            # CNOT gates for parity check
            circuit.cnot(data1, ancilla)
            circuit.cnot(data2, ancilla)
        
        # Optional: Reset ancillas between rounds (not supported on all devices)
    
    # Measure all qubits
    for i in range(total_qubits):
        circuit.measure(i)
    
    return circuit, n_data_qubits, n_ancilla


def create_simple_qec_test(n_qubits: int = 5):
    """
    Create a simple QEC-style circuit that tests multi-qubit correlations.
    This is more robust across different hardware architectures.
    """
    from braket.circuits import Circuit
    
    circuit = Circuit()
    
    # Create a GHZ-like state for testing
    circuit.h(0)
    for i in range(1, n_qubits):
        circuit.cnot(0, i)
    
    # Measure all qubits
    for i in range(n_qubits):
        circuit.measure(i)
    
    return circuit


def run_experiment(
    device_key: str = None,
    n_data_qubits: int = 5,
    n_rounds: int = 1,
    shots: int = 100,
    use_simulator: bool = False,
    s3_bucket: str = None,
) -> dict:
    """Run QEC experiment on specified device."""
    from braket.circuits import Circuit
    from braket.devices import LocalSimulator
    
    print(f"\n{'='*70}")
    print(f"Amazon Braket QEC Experiment")
    print(f"{'='*70}")
    
    # Determine device
    if use_simulator:
        device = LocalSimulator()
        device_name = "LocalSimulator"
        device_type = "simulator"
        estimated_cost = 0.0
        print(f"Device: Local Simulator (FREE)")
    else:
        from braket.aws import AwsDevice
        
        if device_key not in DEVICES:
            raise ValueError(f"Unknown device: {device_key}. Use 'ionq' or 'iqm'")
        
        device_info = DEVICES[device_key]
        device = AwsDevice(device_info["arn"])
        device_name = device_info["name"]
        device_type = device_info["type"]
        estimated_cost = estimate_cost(device_key, shots)
        
        print(f"Device: {device_name}")
        print(f"ARN: {device_info['arn']}")
        print(f"Type: {device_type}")
        print(f"Qubits: {device_info['qubits']}")
        print(f"Estimated Cost: ${estimated_cost:.4f}")
    
    print(f"\nExperiment Parameters:")
    print(f"  Data qubits: {n_data_qubits}")
    print(f"  Syndrome rounds: {n_rounds}")
    print(f"  Shots: {shots}")
    print(f"{'='*70}\n")
    
    # Create circuit
    print("Creating repetition code circuit...")
    circuit, n_data, n_ancilla = create_repetition_code_braket(n_data_qubits, n_rounds)
    total_qubits = n_data + n_ancilla
    
    print(f"Circuit created:")
    print(f"  - Data qubits: {n_data}")
    print(f"  - Ancilla qubits: {n_ancilla}")
    print(f"  - Total qubits: {total_qubits}")
    print(f"  - Circuit depth: {circuit.depth}")
    
    # Check device availability (for hardware)
    if not use_simulator:
        try:
            status = device.status
            print(f"\nDevice status: {status}")
            if status != "ONLINE":
                print(f"WARNING: Device is {status}, task may be queued")
        except Exception as e:
            print(f"Could not check device status: {e}")
    
    # Run the circuit
    print(f"\nSubmitting task with {shots} shots...")
    start_time = time.time()
    
    try:
        if use_simulator:
            task = device.run(circuit, shots=shots)
        else:
            # For AWS devices, we need S3 bucket
            if s3_bucket:
                s3_folder = (s3_bucket, f"braket-results/{device_key}")
                task = device.run(circuit, s3_folder, shots=shots)
            else:
                # Use default bucket (requires S3 permissions)
                task = device.run(circuit, shots=shots)
        
        task_id = getattr(task, 'id', 'local')
        print(f"Task submitted: {task_id}")
        
        # Wait for results
        print("Waiting for results...")
        result = task.result()
        elapsed = time.time() - start_time
        
        counts = result.measurement_counts
        print(f"\nTask completed in {elapsed:.1f}s")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        return {
            "error": str(e),
            "device": device_name,
            "platform": "amazon_braket",
            "timestamp": datetime.now().isoformat(),
        }
    
    # Analyze results
    print(f"\n{'='*70}")
    print("RESULTS ANALYSIS")
    print(f"{'='*70}")
    
    # For repetition code: correct state is all data qubits = 0, ancillas = 0
    correct_state = '0' * total_qubits
    correct_count = counts.get(correct_state, 0)
    total_shots = sum(counts.values())
    ler = 1.0 - (correct_count / total_shots) if total_shots > 0 else 1.0
    
    # Also check for correctable errors (single bit flips)
    correctable = 0
    for state, count in counts.items():
        # Count number of 1s in data qubits
        data_errors = sum(1 for i, bit in enumerate(state[:n_data]) if bit == '1')
        if data_errors <= (n_data - 1) // 2:  # Correctable by majority vote
            correctable += count
    
    correctable_ler = 1.0 - (correctable / total_shots) if total_shots > 0 else 1.0
    
    print(f"Total shots: {total_shots}")
    print(f"Perfect state '{correct_state}': {correct_count}")
    print(f"Raw Logical Error Rate: {ler:.4f}")
    print(f"Correctable LER (majority vote): {correctable_ler:.4f}")
    
    # Show top measurement outcomes
    sorted_counts = sorted(counts.items(), key=lambda x: -x[1])[:10]
    print(f"\nTop 10 measurement outcomes:")
    for state, count in sorted_counts:
        pct = 100 * count / total_shots
        print(f"  {state}: {count} ({pct:.1f}%)")
    
    print(f"{'='*70}\n")
    
    return {
        "device": device_name,
        "device_key": device_key,
        "device_type": device_type,
        "platform": "amazon_braket",
        "device_arn": DEVICES.get(device_key, {}).get("arn"),
        "n_data_qubits": n_data,
        "n_ancilla_qubits": n_ancilla,
        "total_qubits": total_qubits,
        "n_rounds": n_rounds,
        "shots": shots,
        "circuit_depth": circuit.depth,
        "counts": dict(counts),
        "raw_ler": ler,
        "correctable_ler": correctable_ler,
        "correct_count": correct_count,
        "correctable_count": correctable,
        "task_id": task_id,
        "elapsed_seconds": elapsed,
        "estimated_cost_usd": estimated_cost,
        "timestamp": datetime.now().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run Amazon Braket QEC experiments on IonQ and IQM hardware"
    )
    parser.add_argument(
        "--device", type=str, choices=["ionq", "iqm"], default=None,
        help="Device to use: 'ionq' (Forte-1) or 'iqm' (Emerald)"
    )
    parser.add_argument(
        "--simulator", action="store_true",
        help="Use local simulator instead of hardware"
    )
    parser.add_argument(
        "--distance", type=int, default=3,
        help="Code distance (determines data qubits)"
    )
    parser.add_argument(
        "--shots", type=int, default=100,
        help="Number of shots"
    )
    parser.add_argument(
        "--rounds", type=int, default=1,
        help="Syndrome extraction rounds"
    )
    parser.add_argument(
        "--s3-bucket", type=str, default=None,
        help="S3 bucket for results (required for some operations)"
    )
    parser.add_argument(
        "--estimate-only", action="store_true",
        help="Only estimate cost, don't run"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.simulator and not args.device:
        print("ERROR: Must specify --device (ionq/iqm) or --simulator")
        sys.exit(1)
    
    n_data_qubits = args.distance
    
    # Cost estimation
    if args.device and not args.simulator:
        cost = estimate_cost(args.device, args.shots)
        print(f"\n{'#'*70}")
        print(f"# COST ESTIMATE")
        print(f"# Device: {DEVICES[args.device]['name']}")
        print(f"# Shots: {args.shots}")
        print(f"# Task cost: ${DEVICES[args.device]['task_cost']:.2f}")
        print(f"# Shot cost: ${DEVICES[args.device]['shot_cost']:.5f}/shot")
        print(f"# TOTAL: ${cost:.4f}")
        print(f"{'#'*70}\n")
        
        if args.estimate_only:
            return
    
    print(f"\n{'#'*70}")
    print(f"# AMAZON BRAKET QEC EXPERIMENT")
    print(f"# Distance: {args.distance}")
    print(f"# Mode: {'Simulator' if args.simulator else args.device.upper()}")
    print(f"{'#'*70}\n")
    
    # Run experiment
    result = run_experiment(
        device_key=args.device,
        n_data_qubits=n_data_qubits,
        n_rounds=args.rounds,
        shots=args.shots,
        use_simulator=args.simulator,
        s3_bucket=args.s3_bucket,
    )
    
    # Save results
    output_dir = project_root / "results" / "multi_platform"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.simulator:
        mode = "braket_simulator"
    else:
        mode = f"braket_{args.device}"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{mode}_d{args.distance}_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    
    # Summary
    if "error" not in result:
        print(f"\n{'='*70}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*70}")
        print(f"Device: {result['device']}")
        print(f"Raw LER: {result['raw_ler']:.4f}")
        print(f"Correctable LER: {result['correctable_ler']:.4f}")
        print(f"Circuit depth: {result['circuit_depth']}")
        if result.get('estimated_cost_usd', 0) > 0:
            print(f"Cost: ${result['estimated_cost_usd']:.4f}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
