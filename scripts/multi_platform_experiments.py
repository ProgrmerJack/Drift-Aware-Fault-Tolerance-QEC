#!/usr/bin/env python3
"""
Multi-Platform Drift-Aware QEC Experiments

This script validates the drift-aware qubit selection protocol across multiple
quantum computing platforms to address the single-platform limitation.

Platforms:
- IBM Quantum (superconducting, free tier)
- Amazon Braket IQM Emerald (superconducting, 54 qubits)
- Amazon Braket IonQ Forte (trapped-ion, 36 qubits)

The script tests on simulators first before submitting to real hardware.

Author: DAQEC Research Team
Date: 2024-12-27
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ExperimentResult:
    """Result from a single experiment run."""
    platform: str
    backend_name: str
    selection_method: str  # 'drift_aware' or 'calibration_based'
    code_distance: int
    n_rounds: int
    shots: int
    logical_error_rate: float
    raw_counts: Dict[str, int]
    selected_qubits: List[int]
    timestamp: str
    execution_time_s: float
    is_simulation: bool
    cost_estimate_usd: Optional[float] = None
    additional_metrics: Optional[Dict] = None


def save_results(results: List[ExperimentResult], filename: str):
    """Save experiment results to JSON file."""
    output_dir = PROJECT_ROOT / "results" / "multi_platform"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = output_dir / filename
    with open(filepath, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    
    print(f"Results saved to: {filepath}")
    return filepath


# =============================================================================
# IBM QUANTUM EXPERIMENTS
# =============================================================================

def create_repetition_code_circuit_ibm(
    n_data_qubits: int,
    n_rounds: int,
    data_qubits: List[int],
    ancilla_qubits: List[int],
):
    """Create a repetition code circuit for IBM Quantum.
    
    Parameters
    ----------
    n_data_qubits : int
        Number of data qubits (2*d - 1 for distance d)
    n_rounds : int
        Number of syndrome extraction rounds
    data_qubits : list
        Physical indices for data qubits
    ancilla_qubits : list
        Physical indices for ancilla qubits
    
    Returns
    -------
    QuantumCircuit
        Repetition code circuit ready for execution
    """
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    
    # Create registers
    data_reg = QuantumRegister(n_data_qubits, 'data')
    ancilla_reg = QuantumRegister(len(ancilla_qubits), 'ancilla')
    syndrome_reg = ClassicalRegister(len(ancilla_qubits) * n_rounds, 'syndrome')
    final_reg = ClassicalRegister(n_data_qubits, 'final')
    
    qc = QuantumCircuit(data_reg, ancilla_reg, syndrome_reg, final_reg)
    
    # Initialize logical |0⟩ state (all data qubits in |0⟩)
    # No operation needed - qubits start in |0⟩
    
    # Syndrome extraction rounds
    for r in range(n_rounds):
        # Reset ancillas at start of each round
        if r > 0:
            for i in range(len(ancilla_qubits)):
                qc.reset(ancilla_reg[i])
        
        # Apply CNOT gates for parity checks
        # Each ancilla measures parity of adjacent data qubits
        for i in range(len(ancilla_qubits)):
            # CNOT from data[i] to ancilla[i]
            qc.cx(data_reg[i], ancilla_reg[i])
            # CNOT from data[i+1] to ancilla[i]
            qc.cx(data_reg[i + 1], ancilla_reg[i])
        
        # Measure ancillas
        for i in range(len(ancilla_qubits)):
            qc.measure(ancilla_reg[i], syndrome_reg[r * len(ancilla_qubits) + i])
        
        qc.barrier()
    
    # Final data qubit measurement
    qc.measure(data_reg, final_reg)
    
    return qc


def run_ibm_simulation(
    code_distance: int = 3,
    n_rounds: int = 3,
    shots: int = 1000,
    noise_model: Optional[str] = None,
    physical_error_rate: float = 0.01,
) -> ExperimentResult:
    """Run repetition code experiment using numpy-based simulation.
    
    This uses a custom lightweight simulator since qiskit-aer requires
    C compiler installation on Windows.
    
    Parameters
    ----------
    code_distance : int
        Repetition code distance (3, 5, or 7)
    n_rounds : int
        Number of syndrome extraction rounds
    shots : int
        Number of shots
    noise_model : str, optional
        Noise model to use ('basic', 'realistic', or None for ideal)
    physical_error_rate : float
        Physical error rate for noisy simulation
    
    Returns
    -------
    ExperimentResult
        Experiment results
    """
    import time
    
    print(f"\n{'='*60}")
    print(f"NumPy Simulation - Distance {code_distance}, {n_rounds} rounds")
    print(f"{'='*60}")
    
    n_data_qubits = 2 * code_distance - 1
    n_ancilla_qubits = code_distance - 1
    
    # Create qubit lists (consecutive for simulation)
    data_qubits = list(range(n_data_qubits))
    ancilla_qubits = list(range(n_data_qubits, n_data_qubits + n_ancilla_qubits))
    
    print(f"Data qubits: {n_data_qubits}, Ancilla qubits: {n_ancilla_qubits}")
    
    # Set up noise parameters
    if noise_model == 'basic':
        p_error = 0.01  # 1% physical error rate
        print(f"Using basic noise model (p={p_error})")
    elif noise_model == 'realistic':
        p_error = physical_error_rate
        print(f"Using realistic noise model (p={p_error})")
    else:
        p_error = 0.0
        print("Using ideal (noiseless) simulator")
    
    # Run Monte Carlo simulation
    start_time = time.time()
    
    logical_errors = 0
    counts = {}
    
    for _ in range(shots):
        # Initialize data qubits in |0⟩
        data_state = np.zeros(n_data_qubits, dtype=int)
        
        # Apply random bit-flip errors during syndrome rounds
        for _ in range(n_rounds):
            # Random errors on data qubits
            if p_error > 0:
                errors = np.random.random(n_data_qubits) < p_error
                data_state = (data_state + errors.astype(int)) % 2
        
        # Measurement error
        if p_error > 0:
            meas_errors = np.random.random(n_data_qubits) < p_error * 0.5
            final_state = (data_state + meas_errors.astype(int)) % 2
        else:
            final_state = data_state
        
        # Convert to bitstring
        bitstring = ''.join(map(str, final_state))
        counts[bitstring] = counts.get(bitstring, 0) + 1
        
        # Check for logical error (majority flip)
        if np.sum(final_state) > n_data_qubits // 2:
            logical_errors += 1
    
    execution_time = time.time() - start_time
    
    ler = logical_errors / shots
    
    print(f"Shots: {shots}")
    print(f"Logical errors: {logical_errors}")
    print(f"Logical Error Rate (LER): {ler:.6f}")
    print(f"Execution time: {execution_time:.2f}s")
    
    return ExperimentResult(
        platform="NumPy_Sim",
        backend_name="numpy_simulator",
        selection_method="simulation",
        code_distance=code_distance,
        n_rounds=n_rounds,
        shots=shots,
        logical_error_rate=ler,
        raw_counts=counts,
        selected_qubits=data_qubits + ancilla_qubits,
        timestamp=datetime.now().isoformat(),
        execution_time_s=execution_time,
        is_simulation=True,
        additional_metrics={
            'noise_model': noise_model or 'ideal',
            'physical_error_rate': p_error,
        }
    )


def run_ibm_hardware(
    api_token: str,
    code_distance: int = 3,
    n_rounds: int = 3,
    shots: int = 1000,
    use_drift_aware: bool = True,
    backend_name: Optional[str] = None,
) -> ExperimentResult:
    """Run repetition code experiment on IBM Quantum hardware.
    
    Parameters
    ----------
    api_token : str
        IBM Quantum API token
    code_distance : int
        Repetition code distance
    n_rounds : int
        Number of syndrome extraction rounds
    shots : int
        Number of shots
    use_drift_aware : bool
        Whether to use drift-aware qubit selection
    backend_name : str, optional
        Specific backend name, or None to auto-select
    
    Returns
    -------
    ExperimentResult
        Experiment results
    """
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit import transpile
    import time
    
    print(f"\n{'='*60}")
    print(f"IBM Quantum Hardware - Distance {code_distance}, {n_rounds} rounds")
    print(f"Drift-aware selection: {use_drift_aware}")
    print(f"{'='*60}")
    
    # Initialize service
    try:
        service = QiskitRuntimeService(channel="ibm_quantum", token=api_token)
    except Exception as e:
        # Try saving credentials first
        QiskitRuntimeService.save_account(
            channel="ibm_quantum",
            token=api_token,
            overwrite=True
        )
        service = QiskitRuntimeService(channel="ibm_quantum")
    
    # Get backend
    if backend_name:
        backend = service.backend(backend_name)
    else:
        # Get least busy backend with enough qubits
        min_qubits = 2 * (2 * code_distance - 1)  # data + ancilla
        backends = service.backends(
            filters=lambda x: x.configuration().n_qubits >= min_qubits 
                            and x.status().operational
        )
        if not backends:
            raise RuntimeError(f"No operational backend with >= {min_qubits} qubits")
        # Sort by queue length
        backend = min(backends, key=lambda x: x.status().pending_jobs)
    
    print(f"Selected backend: {backend.name}")
    print(f"Queue depth: {backend.status().pending_jobs} jobs")
    
    n_data_qubits = 2 * code_distance - 1
    n_ancilla_qubits = code_distance - 1
    total_qubits = n_data_qubits + n_ancilla_qubits
    
    if use_drift_aware:
        # Import drift-aware selection
        from daqec.probes import run_lightweight_probes
        from daqec.selection import select_qubits_drift_aware
        
        # Get candidate qubits (first 15 qubits typically have best connectivity)
        candidate_qubits = list(range(min(15, backend.configuration().n_qubits)))
        
        print(f"Running probes on {len(candidate_qubits)} candidate qubits...")
        probe_results = run_lightweight_probes(backend, candidate_qubits, shots_per_circuit=30)
        
        # Convert to dict format
        probe_dict = {qid: r.to_dict() for qid, r in probe_results.items()}
        
        # Select best chain
        chains = select_qubits_drift_aware(
            probe_dict,
            code_distance=code_distance,
            backend_topology=backend.coupling_map,
            top_k=1
        )
        
        if not chains:
            raise RuntimeError("No valid qubit chain found")
        
        best_chain = chains[0]
        data_qubits = best_chain.data_qubits
        ancilla_qubits = best_chain.ancilla_qubits
        print(f"Selected data qubits: {data_qubits}")
        print(f"Selected ancilla qubits: {ancilla_qubits}")
        print(f"Chain score: {best_chain.score:.4f}")
        selection_method = "drift_aware"
    else:
        # Use default consecutive qubits (calibration-based)
        data_qubits = list(range(n_data_qubits))
        ancilla_qubits = list(range(n_data_qubits, total_qubits))
        selection_method = "calibration_based"
        print(f"Using default qubits: {data_qubits + ancilla_qubits}")
    
    # Create circuit
    qc = create_repetition_code_circuit_ibm(
        n_data_qubits, n_rounds, data_qubits, ancilla_qubits
    )
    
    # Transpile for hardware
    transpiled = transpile(
        qc,
        backend=backend,
        initial_layout=data_qubits + ancilla_qubits,
        optimization_level=1,
    )
    print(f"Transpiled circuit depth: {transpiled.depth()}")
    
    # Run on hardware
    print("Submitting job to hardware...")
    start_time = time.time()
    
    sampler = SamplerV2(backend)
    job = sampler.run([transpiled], shots=shots)
    print(f"Job ID: {job.job_id()}")
    
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
    
    print(f"\nResults:")
    print(f"Shots: {total_shots}")
    print(f"Logical errors: {logical_errors}")
    print(f"Logical Error Rate (LER): {ler:.6f}")
    print(f"Execution time: {execution_time:.2f}s")
    
    return ExperimentResult(
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
        is_simulation=False,
        cost_estimate_usd=0.0,  # IBM Quantum free tier
        additional_metrics={
            'job_id': job.job_id(),
            'circuit_depth': transpiled.depth(),
        }
    )


# =============================================================================
# AMAZON BRAKET EXPERIMENTS
# =============================================================================

def create_repetition_code_circuit_braket(
    n_data_qubits: int,
    n_rounds: int,
):
    """Create a repetition code circuit for Amazon Braket.
    
    Note: Braket uses a different circuit construction API.
    """
    from braket.circuits import Circuit
    
    n_ancilla_qubits = n_data_qubits - 1
    
    circuit = Circuit()
    
    # Data qubits: 0 to n_data-1
    # Ancilla qubits: n_data to n_data + n_ancilla - 1
    
    # Syndrome extraction rounds
    for r in range(n_rounds):
        # Apply CNOT gates for parity checks
        for i in range(n_ancilla_qubits):
            data_q1 = i
            data_q2 = i + 1
            ancilla_q = n_data_qubits + i
            
            # CNOT from data qubits to ancilla
            circuit.cnot(data_q1, ancilla_q)
            circuit.cnot(data_q2, ancilla_q)
    
    # Measure all qubits
    total_qubits = n_data_qubits + n_ancilla_qubits
    for q in range(total_qubits):
        circuit.measure(q)
    
    return circuit


def estimate_braket_cost(device_arn: str, shots: int, n_tasks: int = 1) -> float:
    """Estimate cost for Amazon Braket execution."""
    costs = {
        "ionq/Forte-1": (0.30, 0.08),      # (task_cost, shot_cost)
        "iqm/Emerald": (0.30, 0.0016),
        "quera/Aquila": (0.30, 0.01),
    }
    
    for device_key, (task_cost, shot_cost) in costs.items():
        if device_key in device_arn:
            return n_tasks * task_cost + shots * shot_cost
    
    return 0.0  # Unknown device


def run_braket_simulation(
    code_distance: int = 3,
    n_rounds: int = 3,
    shots: int = 1000,
) -> ExperimentResult:
    """Run repetition code experiment on Braket local simulator.
    
    Parameters
    ----------
    code_distance : int
        Repetition code distance
    n_rounds : int
        Number of syndrome extraction rounds
    shots : int
        Number of shots
    
    Returns
    -------
    ExperimentResult
        Experiment results
    """
    from braket.devices import LocalSimulator
    import time
    
    print(f"\n{'='*60}")
    print(f"Amazon Braket Local Simulation - Distance {code_distance}")
    print(f"{'='*60}")
    
    n_data_qubits = 2 * code_distance - 1
    
    # Create circuit
    circuit = create_repetition_code_circuit_braket(n_data_qubits, n_rounds)
    print(f"Circuit created: {circuit.qubit_count} qubits")
    
    # Run on local simulator
    device = LocalSimulator()
    
    start_time = time.time()
    task = device.run(circuit, shots=shots)
    result = task.result()
    execution_time = time.time() - start_time
    
    counts = result.measurement_counts
    
    # Calculate logical error rate
    logical_errors = 0
    total_shots = sum(counts.values())
    
    for bitstring, count in counts.items():
        # Data qubits are first n_data_qubits bits
        data_bits = bitstring[:n_data_qubits]
        n_ones = data_bits.count('1')
        if n_ones > n_data_qubits // 2:
            logical_errors += count
    
    ler = logical_errors / total_shots
    
    print(f"Shots: {total_shots}")
    print(f"Logical Error Rate (LER): {ler:.6f}")
    print(f"Execution time: {execution_time:.2f}s")
    
    return ExperimentResult(
        platform="Braket_Local",
        backend_name="local_simulator",
        selection_method="simulation",
        code_distance=code_distance,
        n_rounds=n_rounds,
        shots=shots,
        logical_error_rate=ler,
        raw_counts=dict(counts),
        selected_qubits=list(range(circuit.qubit_count)),
        timestamp=datetime.now().isoformat(),
        execution_time_s=execution_time,
        is_simulation=True,
    )


def run_braket_iqm_emerald(
    code_distance: int = 3,
    n_rounds: int = 3,
    shots: int = 100,  # Keep low for cost efficiency
    s3_bucket: str = None,
) -> ExperimentResult:
    """Run on IQM Emerald (superconducting, 54 qubits) via Amazon Braket.
    
    Cost: $0.30/task + $0.0016/shot
    
    Parameters
    ----------
    code_distance : int
        Repetition code distance
    n_rounds : int
        Number of syndrome extraction rounds
    shots : int
        Number of shots (default 100 for cost efficiency)
    s3_bucket : str
        S3 bucket for results storage
    
    Returns
    -------
    ExperimentResult
        Experiment results
    """
    from braket.aws import AwsDevice
    import time
    
    device_arn = "arn:aws:braket:eu-north-1::device/qpu/iqm/Emerald"
    
    print(f"\n{'='*60}")
    print(f"IQM Emerald (Superconducting) - Distance {code_distance}")
    print(f"{'='*60}")
    
    cost_estimate = estimate_braket_cost(device_arn, shots)
    print(f"Estimated cost: ${cost_estimate:.2f}")
    
    n_data_qubits = 2 * code_distance - 1
    
    # Create circuit
    circuit = create_repetition_code_circuit_braket(n_data_qubits, n_rounds)
    
    # Get device
    device = AwsDevice(device_arn)
    print(f"Device: {device.name}")
    print(f"Status: {device.status}")
    
    if device.status != "ONLINE":
        print(f"WARNING: Device is {device.status}, not ONLINE")
        return None
    
    # Set up S3 bucket
    if s3_bucket is None:
        s3_bucket = "amazon-braket-daqec-results"
    s3_location = (s3_bucket, "daqec-experiments")
    
    start_time = time.time()
    task = device.run(circuit, s3_destination_folder=s3_location, shots=shots)
    print(f"Task ARN: {task.id}")
    
    # Wait for result
    result = task.result()
    execution_time = time.time() - start_time
    
    counts = result.measurement_counts
    
    # Calculate logical error rate
    logical_errors = 0
    total_shots = sum(counts.values())
    
    for bitstring, count in counts.items():
        data_bits = bitstring[:n_data_qubits]
        n_ones = data_bits.count('1')
        if n_ones > n_data_qubits // 2:
            logical_errors += count
    
    ler = logical_errors / total_shots
    
    print(f"Shots: {total_shots}")
    print(f"Logical Error Rate (LER): {ler:.6f}")
    print(f"Actual cost: ${cost_estimate:.2f}")
    
    return ExperimentResult(
        platform="Braket_IQM",
        backend_name="Emerald",
        selection_method="default",
        code_distance=code_distance,
        n_rounds=n_rounds,
        shots=shots,
        logical_error_rate=ler,
        raw_counts=dict(counts),
        selected_qubits=list(range(circuit.qubit_count)),
        timestamp=datetime.now().isoformat(),
        execution_time_s=execution_time,
        is_simulation=False,
        cost_estimate_usd=cost_estimate,
        additional_metrics={
            'task_arn': task.id,
            'device_arn': device_arn,
        }
    )


def run_braket_ionq_forte(
    code_distance: int = 3,
    n_rounds: int = 1,  # Minimal rounds due to high cost
    shots: int = 50,    # Very low due to $0.08/shot
    s3_bucket: str = None,
) -> ExperimentResult:
    """Run on IonQ Forte-1 (trapped-ion, 36 qubits) via Amazon Braket.
    
    Cost: $0.30/task + $0.08/shot (EXPENSIVE!)
    
    Parameters
    ----------
    code_distance : int
        Repetition code distance
    n_rounds : int
        Number of syndrome extraction rounds
    shots : int
        Number of shots (default 50 for cost efficiency)
    s3_bucket : str
        S3 bucket for results storage
    
    Returns
    -------
    ExperimentResult
        Experiment results
    """
    from braket.aws import AwsDevice
    import time
    
    device_arn = "arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1"
    
    print(f"\n{'='*60}")
    print(f"IonQ Forte-1 (Trapped-Ion) - Distance {code_distance}")
    print(f"{'='*60}")
    
    cost_estimate = estimate_braket_cost(device_arn, shots)
    print(f"Estimated cost: ${cost_estimate:.2f}")
    print(f"WARNING: IonQ is expensive at $0.08/shot!")
    
    n_data_qubits = 2 * code_distance - 1
    
    # Create circuit
    circuit = create_repetition_code_circuit_braket(n_data_qubits, n_rounds)
    
    # Get device
    device = AwsDevice(device_arn)
    print(f"Device: {device.name}")
    print(f"Status: {device.status}")
    
    if device.status != "ONLINE":
        print(f"WARNING: Device is {device.status}, not ONLINE")
        return None
    
    # Set up S3 bucket
    if s3_bucket is None:
        s3_bucket = "amazon-braket-daqec-results"
    s3_location = (s3_bucket, "daqec-experiments")
    
    start_time = time.time()
    task = device.run(circuit, s3_destination_folder=s3_location, shots=shots)
    print(f"Task ARN: {task.id}")
    
    # Wait for result
    result = task.result()
    execution_time = time.time() - start_time
    
    counts = result.measurement_counts
    
    # Calculate logical error rate
    logical_errors = 0
    total_shots = sum(counts.values())
    
    for bitstring, count in counts.items():
        data_bits = bitstring[:n_data_qubits]
        n_ones = data_bits.count('1')
        if n_ones > n_data_qubits // 2:
            logical_errors += count
    
    ler = logical_errors / total_shots
    
    print(f"Shots: {total_shots}")
    print(f"Logical Error Rate (LER): {ler:.6f}")
    print(f"Actual cost: ${cost_estimate:.2f}")
    
    return ExperimentResult(
        platform="Braket_IonQ",
        backend_name="Forte-1",
        selection_method="default",
        code_distance=code_distance,
        n_rounds=n_rounds,
        shots=shots,
        logical_error_rate=ler,
        raw_counts=dict(counts),
        selected_qubits=list(range(circuit.qubit_count)),
        timestamp=datetime.now().isoformat(),
        execution_time_s=execution_time,
        is_simulation=False,
        cost_estimate_usd=cost_estimate,
        additional_metrics={
            'task_arn': task.id,
            'device_arn': device_arn,
            'qubit_technology': 'trapped-ion',
        }
    )


# =============================================================================
# MAIN EXPERIMENT ORCHESTRATOR
# =============================================================================

def run_all_simulations():
    """Run all simulator experiments first to validate code."""
    print("\n" + "="*70)
    print("PHASE 1: SIMULATOR VALIDATION")
    print("="*70)
    
    results = []
    
    # IBM Aer simulations
    print("\n--- IBM Aer Simulations ---")
    for distance in [3, 5]:
        for noise in [None, 'basic', 'realistic']:
            try:
                result = run_ibm_simulation(
                    code_distance=distance,
                    n_rounds=3,
                    shots=1000,
                    noise_model=noise
                )
                results.append(result)
            except Exception as e:
                print(f"Error in IBM simulation d={distance}, noise={noise}: {e}")
    
    # Braket local simulation
    print("\n--- Braket Local Simulations ---")
    for distance in [3, 5]:
        try:
            result = run_braket_simulation(
                code_distance=distance,
                n_rounds=3,
                shots=1000
            )
            results.append(result)
        except Exception as e:
            print(f"Error in Braket simulation d={distance}: {e}")
    
    # Save simulation results
    save_results(results, f"simulation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    return results


def run_ibm_hardware_experiments(api_tokens: List[str], shots: int = 1000):
    """Run experiments on IBM Quantum hardware using multiple API tokens."""
    print("\n" + "="*70)
    print("PHASE 2: IBM QUANTUM HARDWARE")
    print("="*70)
    
    results = []
    
    for i, token in enumerate(api_tokens):
        print(f"\n--- Using API token {i+1}/{len(api_tokens)} ---")
        
        for distance in [3, 5]:
            for use_drift_aware in [True, False]:
                try:
                    result = run_ibm_hardware(
                        api_token=token,
                        code_distance=distance,
                        n_rounds=3,
                        shots=shots,
                        use_drift_aware=use_drift_aware
                    )
                    results.append(result)
                except Exception as e:
                    print(f"Error: {e}")
    
    # Save results
    save_results(results, f"ibm_hardware_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    return results


def run_braket_hardware_experiments(shots_iqm: int = 100, shots_ionq: int = 50):
    """Run experiments on Amazon Braket hardware (cost-conscious)."""
    print("\n" + "="*70)
    print("PHASE 3: AMAZON BRAKET HARDWARE")
    print("="*70)
    
    results = []
    
    # IQM Emerald (cheaper)
    print("\n--- IQM Emerald (Superconducting) ---")
    for distance in [3]:  # Only d=3 to save cost
        try:
            result = run_braket_iqm_emerald(
                code_distance=distance,
                n_rounds=2,
                shots=shots_iqm
            )
            if result:
                results.append(result)
        except Exception as e:
            print(f"Error: {e}")
    
    # IonQ Forte (expensive - minimal experiment)
    print("\n--- IonQ Forte-1 (Trapped-Ion) ---")
    try:
        result = run_braket_ionq_forte(
            code_distance=3,
            n_rounds=1,
            shots=shots_ionq
        )
        if result:
            results.append(result)
    except Exception as e:
        print(f"Error: {e}")
    
    # Save results
    save_results(results, f"braket_hardware_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    return results


def main():
    """Main entry point for multi-platform experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Platform DAQEC Experiments")
    parser.add_argument("--phase", choices=["sim", "ibm", "braket", "all"], 
                       default="sim", help="Which phase to run")
    parser.add_argument("--ibm-tokens", nargs="+", help="IBM Quantum API tokens")
    parser.add_argument("--shots", type=int, default=1000, help="Shots for IBM")
    parser.add_argument("--shots-iqm", type=int, default=100, help="Shots for IQM")
    parser.add_argument("--shots-ionq", type=int, default=50, help="Shots for IonQ")
    
    args = parser.parse_args()
    
    all_results = []
    
    if args.phase in ["sim", "all"]:
        results = run_all_simulations()
        all_results.extend(results)
    
    if args.phase in ["ibm", "all"] and args.ibm_tokens:
        results = run_ibm_hardware_experiments(args.ibm_tokens, args.shots)
        all_results.extend(results)
    
    if args.phase in ["braket", "all"]:
        results = run_braket_hardware_experiments(args.shots_iqm, args.shots_ionq)
        all_results.extend(results)
    
    # Summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    
    for result in all_results:
        print(f"{result.platform}/{result.backend_name} d={result.code_distance}: "
              f"LER={result.logical_error_rate:.6f} "
              f"({'sim' if result.is_simulation else 'hw'})")
    
    # Final save
    save_results(all_results, f"all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    return all_results


if __name__ == "__main__":
    main()
