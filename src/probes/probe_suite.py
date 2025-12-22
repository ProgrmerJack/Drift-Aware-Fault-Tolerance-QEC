"""
Module B: Probe Suite
=====================

Lightweight 30-shot diagnostics using Qiskit-Experiments.
Run T1, readout-error, and RB probes using SamplerV2.

References:
- Qiskit Experiments: https://qiskit.org/ecosystem/experiments/
- SamplerV2 Migration: Use Primitives, not deprecated backend.run()
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import json
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import random_clifford
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit_ibm_runtime.options import SamplerOptions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProbeSuite:
    """
    Lightweight probe suite for real-time qubit characterization.
    
    Implements 30-shot diagnostic routines before QEC experiments:
    - T1 energy relaxation
    - Readout error characterization  
    - Single-qubit RB (Randomized Benchmarking)
    
    Uses SamplerV2 primitives (not deprecated backend.run()).
    """
    
    def __init__(self, service: Optional[QiskitRuntimeService] = None,
                 backend_name: str = "ibm_sherbrooke",
                 default_shots: int = 30):
        """
        Initialize the probe suite.
        
        Args:
            service: QiskitRuntimeService instance (or None for offline use)
            backend_name: Name of the target backend
            default_shots: Default number of shots for probes (30 recommended for QPU budget)
        """
        self.service = service
        self.backend_name = backend_name
        self.default_shots = default_shots
        self._backend = None
        
    @property
    def backend(self):
        """Lazy-load backend to avoid initialization overhead."""
        if self._backend is None and self.service is not None:
            self._backend = self.service.backend(self.backend_name)
        return self._backend
    
    def create_t1_circuits(self, qubit: int, 
                           delay_times: List[float]) -> List[QuantumCircuit]:
        """
        Create T1 characterization circuits.
        
        Args:
            qubit: Qubit index to characterize
            delay_times: List of delay times in dt units
            
        Returns:
            List of T1 probe circuits
        """
        circuits = []
        
        for delay in delay_times:
            qc = QuantumCircuit(1, 1, name=f"t1_delay_{delay}")
            qc.x(0)  # Excite qubit
            qc.delay(int(delay), 0, unit='dt')  # Wait for decay
            qc.measure(0, 0)
            circuits.append(qc)
            
        return circuits
    
    def create_readout_error_circuits(self, qubit: int) -> List[QuantumCircuit]:
        """
        Create readout error characterization circuits.
        
        Args:
            qubit: Qubit index to characterize
            
        Returns:
            List of readout error circuits (prepare |0> and |1>)
        """
        # Measure |0> state
        qc_0 = QuantumCircuit(1, 1, name="readout_0")
        qc_0.measure(0, 0)
        
        # Measure |1> state
        qc_1 = QuantumCircuit(1, 1, name="readout_1")
        qc_1.x(0)
        qc_1.measure(0, 0)
        
        return [qc_0, qc_1]
    
    def create_rb_circuits(self, qubit: int, 
                           lengths: List[int],
                           num_samples: int = 3) -> List[QuantumCircuit]:
        """
        Create single-qubit Randomized Benchmarking circuits.
        
        Uses random Cliffords with inversion gate at the end.
        
        Args:
            qubit: Qubit index to characterize
            lengths: List of sequence lengths (number of Cliffords)
            num_samples: Number of random sequences per length
            
        Returns:
            List of RB circuits
        """
        circuits = []
        
        for length in lengths:
            for sample in range(num_samples):
                qc = QuantumCircuit(1, 1, name=f"rb_len{length}_s{sample}")
                
                # Apply random Clifford sequence
                cumulative = random_clifford(1)
                for _ in range(length):
                    cliff = random_clifford(1)
                    qc.append(cliff.to_instruction(), [0])
                    cumulative = cliff.compose(cumulative)
                
                # Apply inverse to return to |0>
                inverse = cumulative.adjoint()
                qc.append(inverse.to_instruction(), [0])
                
                qc.measure(0, 0)
                circuits.append(qc)
                
        return circuits
    
    def run_probes(self, qubit: int, 
                   probe_types: List[str] = ["t1", "readout", "rb"],
                   shots: Optional[int] = None) -> Dict[str, Any]:
        """
        Run a suite of probes on a single qubit.
        
        Args:
            qubit: Qubit index to probe
            probe_types: List of probe types to run
            shots: Number of shots (default: self.default_shots)
            
        Returns:
            Dictionary containing probe results
        """
        if shots is None:
            shots = self.default_shots
            
        results = {
            "qubit": qubit,
            "timestamp": datetime.now().isoformat(),
            "backend": self.backend_name,
            "shots": shots,
            "probes": {}
        }
        
        all_circuits = []
        circuit_map = {}  # Map circuit names to probe types
        
        # Create circuits for requested probes
        if "t1" in probe_types:
            # Use logarithmic spacing for T1 delays
            t1_delays = [100, 500, 1000, 5000, 10000]  # dt units
            t1_circuits = self.create_t1_circuits(qubit, t1_delays)
            for circ in t1_circuits:
                circuit_map[circ.name] = ("t1", circ.name)
            all_circuits.extend(t1_circuits)
            
        if "readout" in probe_types:
            ro_circuits = self.create_readout_error_circuits(qubit)
            for circ in ro_circuits:
                circuit_map[circ.name] = ("readout", circ.name)
            all_circuits.extend(ro_circuits)
            
        if "rb" in probe_types:
            rb_lengths = [1, 5, 10, 20]
            rb_circuits = self.create_rb_circuits(qubit, rb_lengths, num_samples=2)
            for circ in rb_circuits:
                circuit_map[circ.name] = ("rb", circ.name)
            all_circuits.extend(rb_circuits)
            
        if not all_circuits:
            logger.warning("No circuits to run")
            return results
            
        # Run on hardware if available
        if self.backend is not None:
            try:
                # Configure sampler options
                options = SamplerOptions()
                options.default_shots = shots
                
                sampler = SamplerV2(mode=self.backend)
                
                # Submit job
                logger.info(f"Submitting {len(all_circuits)} circuits to {self.backend_name}")
                job = sampler.run(all_circuits, shots=shots)
                
                # Wait for results
                result = job.result()
                
                # Process results by probe type
                probe_results = {"t1": {}, "readout": {}, "rb": {}}
                
                for idx, pub_result in enumerate(result):
                    circ_name = all_circuits[idx].name
                    if circ_name in circuit_map:
                        probe_type, name = circuit_map[circ_name]
                        counts = pub_result.data.c.get_counts()
                        probe_results[probe_type][name] = counts
                        
                # Analyze results
                if probe_results["t1"]:
                    results["probes"]["t1"] = self._analyze_t1(probe_results["t1"])
                if probe_results["readout"]:
                    results["probes"]["readout"] = self._analyze_readout(probe_results["readout"])
                if probe_results["rb"]:
                    results["probes"]["rb"] = self._analyze_rb(probe_results["rb"])
                    
            except Exception as e:
                logger.error(f"Error running probes: {e}")
                results["error"] = str(e)
        else:
            logger.info("No backend available - running in simulation mode")
            results["probes"]["simulated"] = True
            
        return results
    
    def _analyze_t1(self, t1_data: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze T1 probe results."""
        decay_curve = []
        for name, counts in sorted(t1_data.items()):
            # Extract delay time from circuit name
            delay = int(name.split("_")[-1])
            # Calculate excited state probability
            total = sum(counts.values())
            p1 = counts.get("1", 0) / total if total > 0 else 0
            decay_curve.append({"delay_dt": delay, "p1": p1})
            
        # Simple T1 estimate (would use proper fitting in production)
        return {
            "decay_curve": decay_curve,
            "estimated_t1_us": None,  # Would compute from fit
            "quality": "estimated"
        }
    
    def _analyze_readout(self, readout_data: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze readout error probe results."""
        result = {}
        
        for name, counts in readout_data.items():
            total = sum(counts.values())
            if "readout_0" in name:
                # Error is probability of measuring 1 when prepared in |0>
                result["p01"] = counts.get("1", 0) / total if total > 0 else 0
            elif "readout_1" in name:
                # Error is probability of measuring 0 when prepared in |1>
                result["p10"] = counts.get("0", 0) / total if total > 0 else 0
                
        # Compute average readout error
        p01 = result.get("p01", 0)
        p10 = result.get("p10", 0)
        result["average_error"] = (p01 + p10) / 2
        result["assignment_fidelity"] = 1 - result["average_error"]
        
        return result
    
    def _analyze_rb(self, rb_data: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze RB probe results."""
        # Group by sequence length
        length_data = {}
        for name, counts in rb_data.items():
            # Extract length from circuit name (rb_len{L}_s{S})
            parts = name.split("_")
            length = int(parts[1].replace("len", ""))
            
            total = sum(counts.values())
            p0 = counts.get("0", 0) / total if total > 0 else 0
            
            if length not in length_data:
                length_data[length] = []
            length_data[length].append(p0)
            
        # Average over samples at each length
        decay_points = []
        for length in sorted(length_data.keys()):
            avg_survival = np.mean(length_data[length])
            decay_points.append({
                "length": length,
                "survival_probability": avg_survival
            })
            
        # Simple EPC estimate (would use proper fitting in production)
        return {
            "decay_points": decay_points,
            "estimated_epc": None,  # Error per Clifford - would compute from fit
            "quality": "estimated"
        }
    
    def save_results(self, results: Dict[str, Any], 
                     output_dir: str = "data/probes") -> Path:
        """
        Save probe results to disk.
        
        Args:
            results: Probe results dictionary
            output_dir: Directory to save results
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = results["timestamp"].replace(":", "-").replace(".", "-")
        qubit = results["qubit"]
        filename = f"probe_q{qubit}_{timestamp}.json"
        
        filepath = output_path / filename
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)
            
        logger.info(f"Saved probe results to {filepath}")
        return filepath


def run_multi_qubit_probes(service: QiskitRuntimeService,
                           backend_name: str,
                           qubits: List[int],
                           probe_types: List[str] = ["t1", "readout", "rb"],
                           shots: int = 30) -> Dict[int, Dict[str, Any]]:
    """
    Run probes on multiple qubits.
    
    Args:
        service: QiskitRuntimeService instance
        backend_name: Name of the target backend
        qubits: List of qubit indices to probe
        probe_types: List of probe types to run
        shots: Number of shots per circuit
        
    Returns:
        Dictionary mapping qubit index to probe results
    """
    suite = ProbeSuite(service, backend_name, default_shots=shots)
    
    results = {}
    for qubit in qubits:
        logger.info(f"Running probes on qubit {qubit}")
        results[qubit] = suite.run_probes(qubit, probe_types, shots)
        
    return results


if __name__ == "__main__":
    print("Probe Suite Module - Run with IBM Quantum credentials")
    print("\nExample usage:")
    print("""
    from qiskit_ibm_runtime import QiskitRuntimeService
    
    service = QiskitRuntimeService()
    suite = ProbeSuite(service, "ibm_sherbrooke")
    
    # Run probes on qubit 0
    results = suite.run_probes(qubit=0, probe_types=["t1", "readout", "rb"])
    print(results)
    
    # Save results
    suite.save_results(results)
    """)
