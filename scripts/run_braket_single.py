#!/usr/bin/env python
"""
Run a single Amazon Braket quantum experiment.

Usage:
    # Local simulator (free, no AWS credentials needed)
    python run_braket_single.py --simulator --distance 3 --shots 100
    
    # AWS hardware (requires AWS credentials)
    python run_braket_single.py --device "arn:aws:braket:eu-north-1::device/qpu/iqm/Emerald" --distance 3 --shots 100

AWS Device ARNs:
    - IQM Emerald: arn:aws:braket:eu-north-1::device/qpu/iqm/Emerald ($0.30/task + $0.0016/shot)
    - IonQ Forte-1: arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1 ($0.30/task + $0.08/shot)
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from braket.circuits import Circuit
from braket.devices import LocalSimulator


def create_repetition_code_circuit(
    n_data_qubits: int,
    n_rounds: int = 3,
) -> Circuit:
    """Create a repetition code circuit using Braket SDK."""
    n_ancilla = n_data_qubits - 1
    
    circuit = Circuit()
    
    # Syndrome extraction rounds
    for _ in range(n_rounds):
        # CNOT from each pair of data qubits to ancilla
        for i in range(n_ancilla):
            data1 = i
            data2 = i + 1
            ancilla = n_data_qubits + i
            circuit.cnot(data1, ancilla)
            circuit.cnot(data2, ancilla)
    
    return circuit


def run_braket_experiment(
    n_data_qubits: int = 5,
    n_rounds: int = 3,
    shots: int = 100,
    device_arn: str = None,
    s3_folder: tuple = None,
) -> dict:
    """Run experiment on Braket local simulator or AWS hardware."""
    
    print(f"\n{'='*60}")
    print(f"Amazon Braket Experiment Configuration")
    print(f"{'='*60}")
    print(f"Data qubits: {n_data_qubits}")
    print(f"Syndrome rounds: {n_rounds}")
    print(f"Shots: {shots}")
    
    if device_arn:
        from braket.aws import AwsDevice
        print(f"Device: {device_arn}")
        device = AwsDevice(device_arn)
        device_name = device.name
    else:
        print("Device: Local Simulator (StateVectorSimulator)")
        device = LocalSimulator()
        device_name = "LocalSimulator"
    
    print(f"{'='*60}\n")
    
    # Create circuit
    print("Creating repetition code circuit...")
    circuit = create_repetition_code_circuit(n_data_qubits, n_rounds)
    
    # Add measurement on data qubits only
    for i in range(n_data_qubits):
        circuit.measure(i)
    
    print(f"Circuit created:")
    print(f"  - Total qubits: {n_data_qubits + n_data_qubits - 1}")
    print(f"  - Depth: {circuit.depth}")
    
    # Run the circuit
    print(f"\nSubmitting task with {shots} shots...")
    
    if device_arn and s3_folder:
        task = device.run(circuit, s3_folder, shots=shots)
    else:
        task = device.run(circuit, shots=shots)
    
    print(f"Task ARN/ID: {getattr(task, 'id', 'local')}")
    
    print("Waiting for results...")
    result = task.result()
    counts = result.measurement_counts
    
    print(f"\nRaw counts: {dict(counts)}")
    
    # Calculate LER
    correct_state = '0' * n_data_qubits
    correct_count = counts.get(correct_state, 0)
    total_shots = sum(counts.values())
    ler = 1.0 - (correct_count / total_shots)
    
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Correct state '{correct_state}': {correct_count}/{total_shots}")
    print(f"Logical Error Rate (LER): {ler:.4f}")
    print(f"{'='*60}\n")
    
    return {
        "backend": device_name,
        "platform": "amazon_braket",
        "device_arn": device_arn,
        "n_data_qubits": n_data_qubits,
        "n_rounds": n_rounds,
        "shots": shots,
        "circuit_depth": circuit.depth,
        "counts": dict(counts),
        "ler": ler,
        "correct_count": correct_count,
        "task_id": getattr(task, 'id', 'local'),
        "timestamp": datetime.now().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description="Run Amazon Braket quantum experiment")
    parser.add_argument("--device", type=str, default=None,
                        help="AWS Device ARN (omit for local simulator)")
    parser.add_argument("--simulator", action="store_true",
                        help="Use local simulator (default if no device specified)")
    parser.add_argument("--distance", type=int, default=3,
                        help="Code distance (n_data = distance)")
    parser.add_argument("--shots", type=int, default=100,
                        help="Number of shots")
    parser.add_argument("--rounds", type=int, default=3, 
                        help="Syndrome rounds")
    parser.add_argument("--s3-bucket", type=str, default=None,
                        help="S3 bucket for AWS device results")
    parser.add_argument("--s3-prefix", type=str, default="braket-results",
                        help="S3 prefix for results")
    
    args = parser.parse_args()
    
    # Determine device
    device_arn = None if args.simulator else args.device
    
    # S3 folder for AWS devices
    s3_folder = None
    if device_arn and args.s3_bucket:
        s3_folder = (args.s3_bucket, args.s3_prefix)
    
    n_data_qubits = args.distance
    
    print(f"\n{'#'*60}")
    print(f"# AMAZON BRAKET SINGLE EXPERIMENT")
    print(f"# Distance: {args.distance} (data qubits: {n_data_qubits})")
    print(f"# Shots: {args.shots}")
    print(f"# Mode: {'Local Simulator' if not device_arn else 'AWS Hardware'}")
    print(f"{'#'*60}\n")
    
    try:
        result = run_braket_experiment(
            n_data_qubits=n_data_qubits,
            n_rounds=args.rounds,
            shots=args.shots,
            device_arn=device_arn,
            s3_folder=s3_folder,
        )
        
        # Save result
        output_dir = project_root / "results" / "multi_platform"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        mode = "local_sim" if not device_arn else "aws"
        if device_arn:
            # Extract device name from ARN
            device_short = device_arn.split("/")[-1].lower().replace("-", "_")
            mode = f"aws_{device_short}"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"braket_{mode}_d{args.distance}_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
