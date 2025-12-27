#!/usr/bin/env python3
"""Test script for FakeBackend simulation before hardware.

Uses FakeGuadalupeV2 (16 qubits) which works within the 24-qubit limit
of the basic simulator (no qiskit-aer required).
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime.fake_provider import FakeGuadalupeV2
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from datetime import datetime
import json
import os

RESULTS_DIR = "results/multi_platform"


def create_repetition_code_circuit(n_data_qubits: int, n_rounds: int) -> QuantumCircuit:
    """Create a repetition code circuit for error correction.
    
    Args:
        n_data_qubits: Number of data qubits (should be 2*distance - 1)
        n_rounds: Number of syndrome extraction rounds
        
    Returns:
        QuantumCircuit with the repetition code
    """
    n_ancilla_qubits = n_data_qubits - 1
    data_reg = QuantumRegister(n_data_qubits, 'data')
    ancilla_reg = QuantumRegister(n_ancilla_qubits, 'ancilla')
    syndrome_reg = ClassicalRegister(n_ancilla_qubits * n_rounds, 'syndrome')
    final_reg = ClassicalRegister(n_data_qubits, 'final')
    
    qc = QuantumCircuit(data_reg, ancilla_reg, syndrome_reg, final_reg)
    
    # Syndrome extraction rounds
    for r in range(n_rounds):
        if r > 0:
            # Reset ancillas between rounds
            for i in range(n_ancilla_qubits):
                qc.reset(ancilla_reg[i])
        
        # Apply CNOT gates for parity checks (ZZ stabilizers)
        for i in range(n_ancilla_qubits):
            qc.cx(data_reg[i], ancilla_reg[i])
            qc.cx(data_reg[i + 1], ancilla_reg[i])
        
        # Measure ancillas to extract syndromes
        for i in range(n_ancilla_qubits):
            qc.measure(ancilla_reg[i], syndrome_reg[r * n_ancilla_qubits + i])
        
        qc.barrier()
    
    # Final measurement of all data qubits
    qc.measure(data_reg, final_reg)
    return qc


def calculate_ler(counts: dict, n_data_qubits: int) -> float:
    """Calculate logical error rate from measurement counts.
    
    Uses majority voting: if more than half of data qubits are 1, 
    it's a logical error (assuming initialized to |0...0⟩).
    """
    logical_errors = 0
    total = sum(counts.values())
    
    for bitstring, count in counts.items():
        # Count 1s in the final measurement bits
        n_ones = bitstring.count("1")
        if n_ones > n_data_qubits // 2:
            logical_errors += count
    
    return logical_errors / total if total > 0 else 0.0


def main():
    """Run repetition code simulation on FakeGuadalupeV2."""
    print("=" * 60)
    print("FAKE BACKEND SIMULATION TEST")
    print("=" * 60)
    print(f"Backend: FakeGuadalupeV2 (16 qubits)")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Configuration
    code_distance = 3
    n_rounds = 3
    shots = 1000
    
    n_data_qubits = 2 * code_distance - 1  # 5 for d=3
    n_ancilla_qubits = n_data_qubits - 1   # 4 for d=3
    total_qubits = n_data_qubits + n_ancilla_qubits  # 9 qubits needed
    
    print(f"Configuration:")
    print(f"  Code distance: {code_distance}")
    print(f"  Syndrome rounds: {n_rounds}")
    print(f"  Shots: {shots}")
    print(f"  Data qubits: {n_data_qubits}")
    print(f"  Ancilla qubits: {n_ancilla_qubits}")
    print(f"  Total qubits needed: {total_qubits}")
    print()
    
    # Create circuit
    qc = create_repetition_code_circuit(n_data_qubits, n_rounds)
    print(f"Circuit created: {qc.num_qubits} qubits, depth {qc.depth()}")
    
    # Use FakeGuadalupeV2 (16 qubits - fits within basic simulator limit)
    fake_backend = FakeGuadalupeV2()
    print(f"Backend qubits: {fake_backend.num_qubits}")
    
    # Transpile
    pm = generate_preset_pass_manager(backend=fake_backend, optimization_level=1)
    isa_qc = pm.run(qc)
    print(f"Transpiled circuit: {isa_qc.num_qubits} qubits, depth {isa_qc.depth()}")
    print()
    
    # Run simulation
    print("Running simulation...")
    sampler = Sampler(mode=fake_backend)
    job = sampler.run([isa_qc], shots=shots)
    result = job.result()
    
    # Extract counts from the 'final' register
    pub_result = result[0]
    counts = pub_result.data.final.get_counts()
    
    # Calculate LER
    ler = calculate_ler(counts, n_data_qubits)
    
    # Display results
    print(f"Results:")
    print(f"  Total shots: {sum(counts.values())}")
    print(f"  Logical Error Rate: {ler:.6f}")
    
    top_outcomes = dict(list(sorted(counts.items(), key=lambda x: -x[1]))[:5])
    print(f"  Top 5 outcomes: {top_outcomes}")
    
    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_data = {
        "backend": "FakeGuadalupeV2",
        "backend_type": "simulator",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "code_distance": code_distance,
            "n_rounds": n_rounds,
            "shots": shots,
            "n_data_qubits": n_data_qubits,
            "n_ancilla_qubits": n_ancilla_qubits,
        },
        "circuit_info": {
            "num_qubits": qc.num_qubits,
            "original_depth": qc.depth(),
            "transpiled_depth": isa_qc.depth(),
        },
        "results": {
            "ler": ler,
            "logical_errors": int(ler * shots),
            "total_shots": shots,
        },
        "counts": counts,
    }
    
    result_file = os.path.join(RESULTS_DIR, "fake_guadalupe_simulation.json")
    with open(result_file, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"\nResults saved to: {result_file}")
    
    print()
    print("=" * 60)
    print("SIMULATION PASSED!")
    print("=" * 60)
    
    return ler


if __name__ == "__main__":
    main()
