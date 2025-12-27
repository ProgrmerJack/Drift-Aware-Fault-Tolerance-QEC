#!/usr/bin/env python
"""
Run a single IonQ quantum experiment via direct IonQ API.

Usage:
    python run_ionq_single.py --token "API_KEY" --drift-aware --distance 3 --shots 100

Note: IonQ charges per shot, so we use fewer shots by default (100).
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

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_ionq import IonQProvider


def create_repetition_code_circuit(
    n_data_qubits: int,
    n_rounds: int = 3,
) -> QuantumCircuit:
    """Create a simple repetition code circuit for testing.
    
    For IonQ's all-to-all connectivity, we use a simpler construction.
    """
    n_ancilla = n_data_qubits - 1
    total_qubits = n_data_qubits + n_ancilla
    
    qc = QuantumCircuit(total_qubits, n_data_qubits)
    
    # Initialize in |0...0> state
    # Syndrome extraction rounds
    for _ in range(n_rounds):
        # CNOT from each pair of data qubits to ancilla
        for i in range(n_ancilla):
            data1 = i
            data2 = i + 1
            ancilla = n_data_qubits + i
            qc.cx(data1, ancilla)
            qc.cx(data2, ancilla)
        
        # Reset ancillas (simplified - just barrier for now)
        qc.barrier()
    
    # Measure data qubits
    for i in range(n_data_qubits):
        qc.measure(i, i)
    
    return qc


def run_ionq_experiment(
    api_token: str,
    n_data_qubits: int = 5,
    n_rounds: int = 3,
    shots: int = 100,
    use_simulator: bool = False,
    noise_model: str = None,
) -> dict:
    """Run experiment on IonQ hardware or simulator.
    
    Args:
        noise_model: For simulator, can be 'ideal', 'aria-1', or 'harmony'
    """
    
    print(f"\n{'='*60}")
    print(f"IonQ Experiment Configuration")
    print(f"{'='*60}")
    print(f"Data qubits: {n_data_qubits}")
    print(f"Syndrome rounds: {n_rounds}")
    print(f"Shots: {shots}")
    sim_desc = 'simulator'
    if use_simulator and noise_model:
        sim_desc = f'simulator (noise model: {noise_model})'
    print(f"Backend: {sim_desc if use_simulator else 'ionq_qpu (Forte/Aria)'}")
    print(f"{'='*60}\n")
    
    # Initialize IonQ provider
    print("Connecting to IonQ...")
    provider = IonQProvider(token=api_token)
    
    # List available backends
    print("\nAvailable IonQ backends:")
    backends = provider.backends()
    for b in backends:
        print(f"  - {b.name}")
    
    # Select backend
    if use_simulator:
        backend = provider.get_backend("ionq_simulator")
    else:
        # Try to get a QPU backend
        try:
            backend = provider.get_backend("ionq_qpu")
        except Exception as e:
            print(f"Could not get ionq_qpu: {e}")
            print("Trying ionq_forte...")
            try:
                backend = provider.get_backend("ionq_forte")
            except Exception:
                print("Trying ionq_aria...")
                backend = provider.get_backend("ionq_aria")
    
    print(f"\nUsing backend: {backend.name}")
    
    # Create circuit
    print("\nCreating repetition code circuit...")
    qc = create_repetition_code_circuit(n_data_qubits, n_rounds)
    
    print(f"Circuit created:")
    print(f"  - Total qubits: {qc.num_qubits}")
    print(f"  - Original depth: {qc.depth()}")
    
    # Transpile for IonQ (all-to-all connectivity)
    print("\nTranspiling for IonQ...")
    transpiled = transpile(qc, backend=backend, optimization_level=1)
    print(f"  - Transpiled depth: {transpiled.depth()}")
    
    # Run the circuit
    print(f"\nSubmitting job with {shots} shots...")
    run_options = {"shots": shots}
    if noise_model and use_simulator:
        run_options["noise"] = {"model": noise_model}
        print(f"  - Using noise model: {noise_model}")
    
    job = backend.run(transpiled, **run_options)
    print(f"Job ID: {job.job_id()}")
    
    print("Waiting for results...")
    result = job.result()
    counts = result.get_counts()
    
    print(f"\nRaw counts: {counts}")
    
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
        "backend": backend.name,
        "platform": "ionq_direct",
        "noise_model": noise_model if use_simulator else None,
        "n_data_qubits": n_data_qubits,
        "n_rounds": n_rounds,
        "shots": shots,
        "circuit_depth": transpiled.depth(),
        "counts": counts,
        "ler": ler,
        "correct_count": correct_count,
        "job_id": job.job_id(),
        "timestamp": datetime.now().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description="Run IonQ quantum experiment")
    parser.add_argument("--token", required=True, help="IonQ API token")
    parser.add_argument("--distance", type=int, default=3, 
                        help="Code distance (n_data = distance, so d=3 means 3 qubits)")
    parser.add_argument("--shots", type=int, default=100, 
                        help="Number of shots (default 100 - IonQ charges per shot)")
    parser.add_argument("--rounds", type=int, default=3, help="Syndrome rounds")
    parser.add_argument("--simulator", action="store_true", 
                        help="Use IonQ simulator instead of QPU")
    parser.add_argument("--noise-model", type=str, default=None,
                        choices=["ideal", "aria-1", "harmony"],
                        help="Noise model for simulator (aria-1 for realistic)")
    
    args = parser.parse_args()
    
    # Calculate data qubits from distance
    # For repetition code: distance = number of data qubits
    n_data_qubits = args.distance
    
    print(f"\n{'#'*60}")
    print(f"# IonQ SINGLE EXPERIMENT")
    print(f"# Distance: {args.distance} (data qubits: {n_data_qubits})")
    print(f"# Shots: {args.shots}")
    print(f"# Mode: {'Simulator' if args.simulator else 'Hardware QPU'}")
    print(f"{'#'*60}\n")
    
    try:
        result = run_ionq_experiment(
            api_token=args.token,
            n_data_qubits=n_data_qubits,
            n_rounds=args.rounds,
            shots=args.shots,
            use_simulator=args.simulator,
            noise_model=args.noise_model,
        )
        
        # Save result
        output_dir = project_root / "results" / "multi_platform"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        mode = "simulator" if args.simulator else "hardware"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"ionq_{mode}_d{args.distance}_{timestamp}.json"
        
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
