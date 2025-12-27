#!/usr/bin/env python3
"""
Single IBM Quantum experiment runner.

This script runs ONE experiment at a time on IBM hardware, following the 
principle of testing methodically before committing real resources.

Usage:
    python scripts/run_ibm_single.py --token YOUR_TOKEN --drift-aware
    python scripts/run_ibm_single.py --token YOUR_TOKEN --no-drift-aware
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class IBMExperimentResult:
    """Result from IBM Quantum experiment."""
    platform: str
    backend_name: str
    selection_method: str
    code_distance: int
    n_rounds: int
    shots: int
    logical_error_rate: float
    raw_counts: dict
    selected_qubits: list
    timestamp: str
    execution_time_s: float
    job_id: str
    circuit_depth: int


def run_ibm_experiment(
    api_token: str,
    code_distance: int = 3,
    n_rounds: int = 3,
    shots: int = 1000,
    use_drift_aware: bool = True,
    backend_name: Optional[str] = None,
) -> IBMExperimentResult:
    """Run a single IBM Quantum experiment.
    
    Parameters
    ----------
    api_token : str
        IBM Quantum API token
    code_distance : int
        Repetition code distance (3 or 5)
    n_rounds : int
        Number of syndrome extraction rounds
    shots : int
        Number of measurement shots
    use_drift_aware : bool
        Whether to use drift-aware qubit selection
    backend_name : str, optional
        Specific backend name, or None for auto-selection
        
    Returns
    -------
    IBMExperimentResult
        Experiment results
    """
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    
    print("\n" + "=" * 70)
    print("IBM QUANTUM HARDWARE EXPERIMENT")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Code Distance: {code_distance}")
    print(f"Syndrome Rounds: {n_rounds}")
    print(f"Shots: {shots}")
    print(f"Selection Method: {'Drift-Aware' if use_drift_aware else 'Calibration-Based'}")
    print()
    
    # Initialize service
    print("Connecting to IBM Quantum...")
    try:
        service = QiskitRuntimeService(channel="ibm_quantum_platform", token=api_token)
    except Exception:
        print("Saving credentials...")
        QiskitRuntimeService.save_account(
            channel="ibm_quantum_platform",
            token=api_token,
            overwrite=True
        )
        service = QiskitRuntimeService(channel="ibm_quantum_platform")
    
    # Calculate qubit requirements
    n_data_qubits = 2 * code_distance - 1
    n_ancilla_qubits = n_data_qubits - 1  # Each ancilla checks adjacent pairs
    total_qubits = n_data_qubits + n_ancilla_qubits
    
    print(f"Qubit requirements: {n_data_qubits} data + {n_ancilla_qubits} ancilla = {total_qubits} total")
    
    # Get backend
    if backend_name:
        backend = service.backend(backend_name)
    else:
        # Auto-select least busy backend
        print("Auto-selecting backend...")
        backends = service.backends(
            filters=lambda x: (
                x.num_qubits >= total_qubits
                and x.status().operational
            )
        )
        if not backends:
            raise RuntimeError(f"No operational backend with >= {total_qubits} qubits")
        
        # Sort by queue length
        backend = min(backends, key=lambda x: x.status().pending_jobs)
    
    print(f"Selected Backend: {backend.name}")
    print(f"Backend Qubits: {backend.num_qubits}")
    print(f"Queue Depth: {backend.status().pending_jobs} jobs")
    print()
    
    # Select qubits
    if use_drift_aware:
        print("Running drift-aware qubit selection...")
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        
        try:
            from daqec.probes import run_lightweight_probes
            from daqec.selection import select_qubits_drift_aware
            
            # Get candidate qubits
            n_candidates = min(20, backend.configuration().n_qubits)
            candidate_qubits = list(range(n_candidates))
            
            print(f"Probing {len(candidate_qubits)} candidate qubits...")
            probe_results = run_lightweight_probes(
                backend, 
                candidate_qubits, 
                shots_per_circuit=30
            )
            
            # Convert to dict format
            probe_dict = {qid: r.to_dict() for qid, r in probe_results.items()}
            
            # Select best chain
            chains = select_qubits_drift_aware(
                probe_dict,
                code_distance=code_distance,
                backend_topology=backend.coupling_map,
                top_k=1
            )
            
            if chains:
                best_chain = chains[0]
                data_qubits = best_chain.data_qubits
                ancilla_qubits = best_chain.ancilla_qubits
                selection_method = "drift_aware"
                print(f"Selected data qubits: {data_qubits}")
                print(f"Selected ancilla qubits: {ancilla_qubits}")
                print(f"Chain score: {best_chain.score:.4f}")
            else:
                raise RuntimeError("No valid qubit chain found")
                
        except ImportError as e:
            print(f"Warning: Could not import DAQEC modules: {e}")
            print("Falling back to calibration-based selection...")
            use_drift_aware = False
    
    if not use_drift_aware:
        # Use default consecutive qubits
        data_qubits = list(range(n_data_qubits))
        ancilla_qubits = list(range(n_data_qubits, total_qubits))
        selection_method = "calibration_based"
        print(f"Using default qubits: {data_qubits + ancilla_qubits}")
    
    print()
    
    # Create circuit
    print("Creating repetition code circuit...")
    
    data_reg = QuantumRegister(n_data_qubits, 'data')
    ancilla_reg = QuantumRegister(n_ancilla_qubits, 'ancilla')
    syndrome_reg = ClassicalRegister(n_ancilla_qubits * n_rounds, 'syndrome')
    final_reg = ClassicalRegister(n_data_qubits, 'final')
    
    qc = QuantumCircuit(data_reg, ancilla_reg, syndrome_reg, final_reg)
    
    # Syndrome extraction rounds
    for r in range(n_rounds):
        if r > 0:
            for i in range(n_ancilla_qubits):
                qc.reset(ancilla_reg[i])
        
        # CNOT gates for parity checks (ZZ stabilizers)
        for i in range(n_ancilla_qubits):
            qc.cx(data_reg[i], ancilla_reg[i])
            qc.cx(data_reg[i + 1], ancilla_reg[i])
        
        # Measure ancillas
        for i in range(n_ancilla_qubits):
            qc.measure(ancilla_reg[i], syndrome_reg[r * n_ancilla_qubits + i])
        
        qc.barrier()
    
    # Final measurement
    qc.measure(data_reg, final_reg)
    
    print(f"Circuit created: {qc.num_qubits} qubits, depth {qc.depth()}")
    
    # Transpile for hardware
    print("Transpiling for hardware...")
    transpiled = transpile(
        qc,
        backend=backend,
        initial_layout=data_qubits + ancilla_qubits,
        optimization_level=1,
    )
    print(f"Transpiled circuit depth: {transpiled.depth()}")
    print()
    
    # Submit job
    print("Submitting job to hardware...")
    start_time = time.time()
    
    sampler = SamplerV2(backend)
    job = sampler.run([transpiled], shots=shots)
    job_id = job.job_id()
    print(f"Job ID: {job_id}")
    print("Waiting for results...")
    
    # Wait for result
    result = job.result()
    execution_time = time.time() - start_time
    
    # Extract counts
    pub_result = result[0]
    counts = pub_result.data.final.get_counts()
    
    # Calculate logical error rate
    logical_errors = 0
    total_shots = sum(counts.values())
    
    for bitstring, count in counts.items():
        n_ones = bitstring.count('1')
        if n_ones > n_data_qubits // 2:
            logical_errors += count
    
    ler = logical_errors / total_shots
    
    print()
    print("=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Total Shots: {total_shots}")
    print(f"Logical Errors: {logical_errors}")
    print(f"Logical Error Rate (LER): {ler:.6f}")
    print(f"Execution Time: {execution_time:.2f}s")
    
    # Top outcomes
    top_5 = dict(list(sorted(counts.items(), key=lambda x: -x[1]))[:5])
    print(f"Top 5 Outcomes: {top_5}")
    
    return IBMExperimentResult(
        platform="IBM_Quantum",
        backend_name=backend.name,
        selection_method=selection_method,
        code_distance=code_distance,
        n_rounds=n_rounds,
        shots=shots,
        logical_error_rate=ler,
        raw_counts=counts,
        selected_qubits=data_qubits + ancilla_qubits,
        timestamp=datetime.now().isoformat(),
        execution_time_s=execution_time,
        job_id=job_id,
        circuit_depth=transpiled.depth(),
    )


def save_result(result: IBMExperimentResult, filename: str = None):
    """Save experiment result to JSON file."""
    results_dir = "results/multi_platform"
    os.makedirs(results_dir, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ibm_{result.selection_method}_d{result.code_distance}_{timestamp}.json"
    
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(asdict(result), f, indent=2)
    
    print(f"\nResults saved to: {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(
        description="Run a single IBM Quantum experiment"
    )
    parser.add_argument(
        "--token", 
        required=True, 
        help="IBM Quantum API token"
    )
    parser.add_argument(
        "--distance", 
        type=int, 
        default=3,
        choices=[3, 5],
        help="Code distance (3 or 5)"
    )
    parser.add_argument(
        "--rounds", 
        type=int, 
        default=3,
        help="Number of syndrome extraction rounds"
    )
    parser.add_argument(
        "--shots", 
        type=int, 
        default=1000,
        help="Number of measurement shots"
    )
    parser.add_argument(
        "--drift-aware",
        dest="drift_aware",
        action="store_true",
        help="Use drift-aware qubit selection"
    )
    parser.add_argument(
        "--no-drift-aware",
        dest="drift_aware",
        action="store_false",
        help="Use calibration-based qubit selection"
    )
    parser.set_defaults(drift_aware=True)
    parser.add_argument(
        "--backend",
        default=None,
        help="Specific backend name (default: auto-select)"
    )
    
    args = parser.parse_args()
    
    try:
        result = run_ibm_experiment(
            api_token=args.token,
            code_distance=args.distance,
            n_rounds=args.rounds,
            shots=args.shots,
            use_drift_aware=args.drift_aware,
            backend_name=args.backend,
        )
        
        save_result(result)
        
        print("\n" + "=" * 70)
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nEXPERIMENT FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
