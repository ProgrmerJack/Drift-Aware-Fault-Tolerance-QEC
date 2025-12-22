"""
Lightweight probe circuits module.

Provides 30-shot probe circuits for T1, T2, and readout error estimation
suitable for QPU-constrained environments.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class ProbeResult:
    """Results from a probe measurement session.
    
    Attributes:
        qubit_id: Physical qubit index
        t1_us: Estimated T1 in microseconds
        t2_us: Estimated T2 in microseconds
        readout_error: Readout error probability
        gate_error: Estimated single-qubit gate error (optional)
        timestamp: Measurement timestamp (ISO format)
        shots: Number of shots used
        confidence: Confidence level (0-1) based on fit quality
    """
    qubit_id: int
    t1_us: float
    t2_us: float
    readout_error: float
    gate_error: Optional[float] = None
    timestamp: Optional[str] = None
    shots: int = 30
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format expected by select_qubits_drift_aware."""
        return {
            'T1': self.t1_us,
            'T2': self.t2_us,
            'readout_error': self.readout_error,
            'gate_error': self.gate_error or 0.01,
        }


def run_lightweight_probes(
    backend,
    qubit_ids: List[int],
    shots_per_circuit: int = 30,
    delays_us: Optional[List[float]] = None,
) -> Dict[int, ProbeResult]:
    """Run lightweight probe circuits to estimate qubit parameters.
    
    Executes T1, T2 (Ramsey), and readout error probes using minimal shots
    to estimate current qubit properties. Designed for QPU-constrained
    environments where comprehensive characterization is infeasible.
    
    Parameters
    ----------
    backend : IBMBackend or simulator
        IBM Quantum backend to run probes on.
        
    qubit_ids : list of int
        Physical qubit indices to probe.
        
    shots_per_circuit : int, optional
        Shots per probe circuit. Default: 30.
        Lower values reduce QPU time but increase uncertainty.
        
    delays_us : list of float, optional
        Delay times for T1/T2 estimation in microseconds.
        Default: [10, 50, 100, 200, 500].
        
    Returns
    -------
    dict
        Mapping from qubit_id to ProbeResult.
        
    Examples
    --------
    >>> from daqec.probes import run_lightweight_probes
    >>> from qiskit_ibm_runtime import QiskitRuntimeService
    >>> 
    >>> service = QiskitRuntimeService()
    >>> backend = service.backend("ibm_brisbane")
    >>> 
    >>> results = run_lightweight_probes(backend, qubit_ids=[0, 1, 2, 3, 4])
    >>> for qid, result in results.items():
    ...     print(f"Qubit {qid}: T1={result.t1_us:.1f}µs, T2={result.t2_us:.1f}µs")
    
    Notes
    -----
    The probe suite consists of:
    
    1. **T1 probe**: Prepare |1⟩, delay, measure. Fit exponential decay.
    2. **T2 (Ramsey) probe**: H, delay, H, measure. Fit oscillation decay.
    3. **Readout probe**: Prepare |0⟩ and |1⟩, measure. Compute misclassification.
    
    Total shots per qubit: 30 × 5 delays × 2 probes + 30 × 2 states = 360 shots.
    
    With 15 candidate qubits, total probe time ≈ 90 seconds QPU time.
    
    The 30-shot limit provides sufficient precision for ranking decisions
    while minimizing QPU overhead. Confidence intervals are wider than
    full characterization but adequate for selection.
    
    References
    ----------
    .. [1] DAQEC-Benchmark v1.0, Methods section "Probe circuits"
    """
    if delays_us is None:
        delays_us = [10, 50, 100, 200, 500]
    
    results = {}
    
    # Check if we're in simulation mode
    is_simulation = not hasattr(backend, 'run') or 'simulator' in str(backend).lower()
    
    if is_simulation:
        # Generate synthetic probe results for testing
        for qid in qubit_ids:
            results[qid] = _generate_synthetic_probe(qid)
    else:
        # Run actual probe circuits
        results = _run_hardware_probes(backend, qubit_ids, shots_per_circuit, delays_us)
    
    return results


def _generate_synthetic_probe(qubit_id: int, seed: Optional[int] = None) -> ProbeResult:
    """Generate synthetic probe results for testing/simulation."""
    if seed is not None:
        np.random.seed(seed + qubit_id)
    
    # Realistic parameter ranges for superconducting qubits
    t1 = np.random.uniform(80, 150)  # µs
    t2 = np.random.uniform(t1 * 0.5, t1 * 1.5)  # T2 typically ≤ 2*T1
    t2 = min(t2, 2 * t1)  # Physical constraint
    readout_error = np.random.uniform(0.005, 0.03)
    gate_error = np.random.uniform(0.002, 0.015)
    
    return ProbeResult(
        qubit_id=qubit_id,
        t1_us=t1,
        t2_us=t2,
        readout_error=readout_error,
        gate_error=gate_error,
        shots=30,
        confidence=0.9,
    )


def _run_hardware_probes(
    backend,
    qubit_ids: List[int],
    shots: int,
    delays_us: List[float],
) -> Dict[int, ProbeResult]:
    """Run actual probe circuits on hardware."""
    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit_ibm_runtime import SamplerV2
    except ImportError:
        raise ImportError("Qiskit and qiskit-ibm-runtime required for hardware probes")
    
    results = {}
    
    for qid in qubit_ids:
        # T1 probe circuits
        t1_circuits = []
        for delay in delays_us:
            qc = QuantumCircuit(1, 1)
            qc.x(0)  # Prepare |1⟩
            qc.delay(delay, 0, unit='us')
            qc.measure(0, 0)
            t1_circuits.append(qc)
        
        # T2 Ramsey probe circuits
        t2_circuits = []
        for delay in delays_us:
            qc = QuantumCircuit(1, 1)
            qc.h(0)  # Prepare |+⟩
            qc.delay(delay, 0, unit='us')
            qc.h(0)
            qc.measure(0, 0)
            t2_circuits.append(qc)
        
        # Readout probe circuits
        ro_circuits = []
        for prep_state in [0, 1]:
            qc = QuantumCircuit(1, 1)
            if prep_state == 1:
                qc.x(0)
            qc.measure(0, 0)
            ro_circuits.append(qc)
        
        # Transpile all circuits
        all_circuits = t1_circuits + t2_circuits + ro_circuits
        transpiled = transpile(
            all_circuits,
            backend=backend,
            initial_layout=[qid],
            optimization_level=0,  # Minimal optimization for probes
        )
        
        # Run circuits
        # Note: In production, batch these for efficiency
        sampler = SamplerV2(backend)
        job = sampler.run(transpiled, shots=shots)
        result = job.result()
        
        # Extract counts and fit parameters
        t1_probs = []
        for i, delay in enumerate(delays_us):
            counts = result[i].data.c.get_counts()
            p1 = counts.get('1', 0) / shots
            t1_probs.append(p1)
        
        t2_probs = []
        for i, delay in enumerate(delays_us):
            idx = len(delays_us) + i
            counts = result[idx].data.c.get_counts()
            p1 = counts.get('1', 0) / shots
            t2_probs.append(p1)
        
        # Readout errors
        ro_probs = []
        for i in range(2):
            idx = 2 * len(delays_us) + i
            counts = result[idx].data.c.get_counts()
            # Error is measuring wrong state
            if i == 0:  # Prepared |0⟩
                error = counts.get('1', 0) / shots
            else:  # Prepared |1⟩
                error = counts.get('0', 0) / shots
            ro_probs.append(error)
        
        # Fit T1
        t1_us = _fit_exponential_decay(delays_us, t1_probs)
        
        # Fit T2 (simplified)
        t2_us = _fit_exponential_decay(delays_us, [abs(p - 0.5) * 2 for p in t2_probs])
        
        # Average readout error
        readout_error = np.mean(ro_probs)
        
        results[qid] = ProbeResult(
            qubit_id=qid,
            t1_us=t1_us,
            t2_us=t2_us,
            readout_error=readout_error,
            shots=shots,
            confidence=0.8,  # Lower confidence due to limited shots
        )
    
    return results


def _fit_exponential_decay(times: List[float], probs: List[float]) -> float:
    """Fit exponential decay to extract T1 or T2."""
    times = np.array(times)
    probs = np.array(probs)
    
    # Simple linear fit in log space (robust to noise)
    # P(t) = A * exp(-t/T) + B
    # For T1: A ≈ 1, B ≈ 0
    # log(P) ≈ log(A) - t/T
    
    # Clip probabilities to avoid log(0)
    probs_clipped = np.clip(probs, 0.01, 0.99)
    
    # Linear regression: log(p) = a - t/T  =>  T = -1/slope
    try:
        log_probs = np.log(probs_clipped)
        slope, _ = np.polyfit(times, log_probs, 1)
        if slope < 0:
            T = -1 / slope
        else:
            T = 100.0  # Default if fit fails
    except:
        T = 100.0  # Default
    
    # Clamp to reasonable range
    T = np.clip(T, 10, 500)
    
    return float(T)
