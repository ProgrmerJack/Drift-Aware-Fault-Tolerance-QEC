#!/usr/bin/env python3
"""
submit_async.py - Asynchronous IBM Quantum Job Submission
=========================================================

Submits jobs to IBM Quantum hardware without waiting for completion.
Saves job IDs and metadata for later retrieval via collect_results.py.

This architecture is designed for IBM Open Plan (free tier) constraints:
- 10 minutes quantum time per 28-day window
- Fair-share scheduler with variable queue times
- Jobs may take hours to complete; results stored by IBM for retrieval

Usage:
    python scripts/submit_async.py --experiment surface_code --api-key-index 0
    python scripts/submit_async.py --experiment deployment --api-key-index 1
    python scripts/submit_async.py --submit-all

References:
- IBM job retrieval: https://quantum.cloud.ibm.com/docs/guides/save-jobs
- Fair-share scheduler: https://quantum.cloud.ibm.com/docs/guides/fair-share-scheduler
- Open Plan limits: https://quantum.cloud.ibm.com/docs/guides/instances
"""

import argparse
import json
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path

from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

# API Keys (from environment or config in production)
API_KEYS = [
    "QXKvh5Ol-rQRrxbs2rdxlo9MLlCpNJioLB8p1_uujfkD",
    "2pbhDH38zmWHgFGw_7Pp8d1ugGvPaa5KR2aTMvW8LJfo",
    "wHH8qtEd9yjYFRRrBKNaExedCLE9JX9qiDG9w3krgYow",
]

# Experiment configurations
SURFACE_CODE_CONFIG = {
    "syndrome_rounds": 3,
    "shots": 4096,
    "repetitions": 3,
    "logical_states": ["+", "0"],
}

DEPLOYMENT_CONFIG = {
    "shots": 4096,
    "probe_shots": 30,
    "sessions_per_key": 23,  # Target N=69 pairs (23×3 keys=69) for 80% power with r=0.8 paired design
}


def get_git_commit():
    """Get current git commit hash for reproducibility."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except:
        return "unknown"


def circuit_hash(qc):
    """Compute deterministic hash of circuit for deduplication."""
    # Use qasm3 for Qiskit 2.x compatibility
    from qiskit import qasm2
    qasm = qasm2.dumps(qc)
    return hashlib.sha256(qasm.encode()).hexdigest()[:16]


def connect_to_ibm(api_key):
    """Connect to IBM Quantum with Open Plan account."""
    # Try ibm_cloud first (most common for Open Plan)
    channels = [
        {"channel": "ibm_cloud"},
        {"channel": "ibm_quantum_platform"},
    ]
    
    for channel_config in channels:
        try:
            service = QiskitRuntimeService(token=api_key, **channel_config)
            return service
        except Exception as e:
            continue
    
    raise RuntimeError(f"Failed to connect with any channel configuration")


def build_distance3_surface_code(logical_state='+'):
    """
    Build distance-3 rotated surface code for logical state |+⟩ or |0⟩.
    
    Layout (17 qubits):
        D0  D1  D2
      Z0  X0  Z1
        D3  D4  D5
      X1  Z2  X2
        D6  D7  D8
        
    Syndrome rounds: 3 (minimal detection)
    """
    qc = QuantumCircuit(17, 9)  # 9 data + 8 ancilla, 9 measurements
    
    # Data qubits: 0-8, Ancilla: 9-16
    # Initialize logical state
    if logical_state == '+':
        for i in range(9):
            qc.h(i)
    # |0⟩ is computational basis (no initialization needed)
    
    qc.barrier()
    
    # 3 syndrome rounds
    for round_idx in range(3):
        # X-stabilizers (measure Z-Z-Z-Z)
        qc.h([10, 13, 15])  # X0, X1, X2
        
        # Entangle with data qubits
        # X0: D0-D1-D3-D4
        qc.cx(0, 10)
        qc.cx(1, 10)
        qc.cx(3, 10)
        qc.cx(4, 10)
        
        # X1: D3-D4-D6-D7
        qc.cx(3, 13)
        qc.cx(4, 13)
        qc.cx(6, 13)
        qc.cx(7, 13)
        
        # X2: D4-D5-D7-D8
        qc.cx(4, 15)
        qc.cx(5, 15)
        qc.cx(7, 15)
        qc.cx(8, 15)
        
        # Z-stabilizers (measure X-X-X-X)
        # Z0: D0-D1-D3-D4
        qc.cx(9, 0)
        qc.cx(9, 1)
        qc.cx(9, 3)
        qc.cx(9, 4)
        
        # Z1: D1-D2-D4-D5
        qc.cx(11, 1)
        qc.cx(11, 2)
        qc.cx(11, 4)
        qc.cx(11, 5)
        
        # Z2: D3-D4-D6-D7
        qc.cx(14, 3)
        qc.cx(14, 4)
        qc.cx(14, 6)
        qc.cx(14, 7)
        
        qc.h([10, 13, 15])  # X stabilizers back to Z basis
        qc.barrier()
    
    # Measure all data qubits
    qc.measure(range(9), range(9))
    
    return qc


def build_repetition_code_circuit(distance=5):
    """Build distance-5 repetition code for deployment study."""
    qc = QuantumCircuit(distance, distance)
    
    # Initialize in |0⟩^⊗5
    # Encode: q0 is logical, others are redundant
    for i in range(1, distance):
        qc.cx(0, i)
    
    qc.barrier()
    
    # Error detection: measure parity checks
    for i in range(distance - 1):
        qc.cx(i, i + 1)
    
    qc.barrier()
    qc.measure(range(distance), range(distance))
    
    return qc


def submit_surface_code_job(service, backend, config, api_key_index):
    """Submit ALL surface code experiment jobs (does not wait for results)."""
    jobs_metadata = []
    
    # Submit jobs for ALL logical states and repetitions
    for logical_state in config["logical_states"]:
        for rep in range(config["repetitions"]):
            print(f"\n  Submitting surface code job: |{logical_state}⟩ rep{rep} on {backend.name}")
            
            # Build circuit
            qc = build_distance3_surface_code(logical_state=logical_state)
            
            # Transpile
            pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
            isa_circuit = pm.run(qc)
            
            # Submit job (no waiting)
            sampler = Sampler(mode=backend)
            job = sampler.run([isa_circuit], shots=config["shots"])
            
            job_id = job.job_id()
            print(f"    ✓ Job submitted: {job_id}")
            
            # Save metadata
            metadata = {
                "job_id": job_id,
                "experiment_type": "surface_code",
                "backend": backend.name,
                "logical_state": logical_state,
                "repetition": rep,
                "shots": config["shots"],
                "syndrome_rounds": config["syndrome_rounds"],
                "circuit_hash": circuit_hash(qc),
                "transpiled_depth": isa_circuit.depth(),
                "transpiled_gates": isa_circuit.size(),
                "submission_time": datetime.now().isoformat(),
                "api_key_index": api_key_index,
                "git_commit": get_git_commit(),
                "status": "submitted",
            }
            jobs_metadata.append(metadata)
    
    return jobs_metadata


def submit_deployment_job(service, backend, config, mode, api_key_index):
    """Submit multiple deployment session jobs (baseline or DAQEC)."""
    jobs_metadata = []
    
    # Submit multiple sessions per mode to reach N≥42
    for session in range(config["sessions_per_key"]):
        print(f"\n  Submitting deployment job: {mode} session{session} on {backend.name}")
        
        # Build circuit
        qc = build_repetition_code_circuit(distance=5)
        
        # Transpile
        pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
        isa_circuit = pm.run(qc)
        
        # Submit job (no waiting)
        sampler = Sampler(mode=backend)
        job = sampler.run([isa_circuit], shots=config["shots"])
        
        job_id = job.job_id()
        print(f"    ✓ Job submitted: {job_id}")
        
        # Save metadata
        metadata = {
            "job_id": job_id,
            "experiment_type": "deployment",
            "mode": mode,
            "session": session,
            "backend": backend.name,
            "shots": config["shots"],
            "circuit_hash": circuit_hash(qc),
            "transpiled_depth": isa_circuit.depth(),
            "transpiled_gates": isa_circuit.size(),
            "submission_time": datetime.now().isoformat(),
            "api_key_index": api_key_index,
            "git_commit": get_git_commit(),
            "status": "submitted",
        }
        
        jobs_metadata.append(metadata)
    
    return jobs_metadata


def main():
    parser = argparse.ArgumentParser(
        description="Submit async IBM Quantum jobs without waiting"
    )
    parser.add_argument(
        "--experiment",
        choices=["surface_code", "deployment", "both"],
        default="both",
        help="Type of experiment to submit",
    )
    parser.add_argument(
        "--api-key-index",
        type=int,
        choices=[0, 1, 2],
        help="Which API key to use (0-2)",
    )
    parser.add_argument(
        "--submit-all",
        action="store_true",
        help="Submit jobs for all 3 API keys sequentially",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        help="Target backend (auto = least busy)",
    )
    
    args = parser.parse_args()
    
    # Determine which API keys to use
    if args.submit_all:
        api_indices = [0, 1, 2]
    elif args.api_key_index is not None:
        api_indices = [args.api_key_index]
    else:
        print("Error: Must specify --api-key-index or --submit-all")
        return
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / "results" / "ibm_experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_jobs = []
    
    print("=" * 70)
    print("IBM QUANTUM ASYNC JOB SUBMISSION")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)
    
    for api_idx in api_indices:
        print(f"\n{'=' * 70}")
        print(f"API KEY {api_idx + 1}/{len(api_indices)}")
        print("=" * 70)
        
        try:
            # Connect
            service = connect_to_ibm(API_KEYS[api_idx])
            print("  ✓ Connected to IBM Quantum")
            
            # Select backend
            if args.backend == "auto":
                backend = service.least_busy(
                    operational=True,
                    simulator=False,
                    min_num_qubits=17,
                )
            else:
                backend = service.backend(args.backend)
            
            print(f"  ✓ Using backend: {backend.name}")
            
            # Submit jobs based on experiment type
            if args.experiment in ["surface_code", "both"]:
                jobs = submit_surface_code_job(
                    service, backend, SURFACE_CODE_CONFIG, api_idx
                )
                all_jobs.extend(jobs)
            
            if args.experiment in ["deployment", "both"]:
                # Submit baseline session
                jobs = submit_deployment_job(
                    service, backend, DEPLOYMENT_CONFIG, "baseline", api_idx
                )
                all_jobs.extend(jobs)
                
                # Submit DAQEC session
                jobs = submit_deployment_job(
                    service, backend, DEPLOYMENT_CONFIG, "daqec", api_idx
                )
                all_jobs.extend(jobs)
        
        except Exception as e:
            print(f"  ✗ Error with API key {api_idx}: {e}")
            continue
    
    # Save all job metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jobs_file = output_dir / f"submitted_jobs_{timestamp}.jsonl"
    
    with open(jobs_file, "w") as f:
        for job in all_jobs:
            f.write(json.dumps(job) + "\n")
    
    print(f"\n{'=' * 70}")
    print(f"SUBMISSION COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total jobs submitted: {len(all_jobs)}")
    print(f"Job metadata saved to: {jobs_file}")
    print(f"\nTo retrieve results later, run:")
    print(f"  python scripts/collect_results.py --jobs-file {jobs_file.name}")


if __name__ == "__main__":
    main()
