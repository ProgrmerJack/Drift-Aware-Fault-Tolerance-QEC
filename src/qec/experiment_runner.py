"""
Module C: QEC Experiment Runner
===============================

Runs QEC experiments on IBM Quantum hardware using SamplerV2.
Manages job submission, result collection, and analysis.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import json
from pathlib import Path
import time

import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit_ibm_runtime.options import SamplerOptions

from .repetition_code import RepetitionCode, SyndromeDecoder, create_experiment_batch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QECExperimentRunner:
    """
    Manages QEC experiment execution on IBM Quantum hardware.
    
    Features:
    - QPU time budget management (~10 min per 28 days on Open Plan)
    - Job batching for efficiency
    - Automatic transpilation with layout specification
    - Result analysis and storage
    """
    
    def __init__(self, service: Optional[QiskitRuntimeService] = None,
                 backend_name: str = "ibm_sherbrooke",
                 output_dir: str = "data/experiments"):
        """
        Initialize the experiment runner.
        
        Args:
            service: QiskitRuntimeService instance
            backend_name: Target backend name
            output_dir: Directory for storing results
        """
        self.service = service
        self.backend_name = backend_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._backend = None
        self._pm = None  # Pass manager for transpilation
        
    @property
    def backend(self):
        """Lazy-load backend."""
        if self._backend is None and self.service is not None:
            self._backend = self.service.backend(self.backend_name)
        return self._backend
    
    @property
    def pass_manager(self):
        """Get preset pass manager for transpilation."""
        if self._pm is None and self.backend is not None:
            self._pm = generate_preset_pass_manager(
                optimization_level=1,
                backend=self.backend
            )
        return self._pm
    
    def transpile_circuits(self, circuits: List[QuantumCircuit],
                           initial_layout: Optional[List[int]] = None) -> List[QuantumCircuit]:
        """
        Transpile circuits for the target backend.
        
        Args:
            circuits: List of circuits to transpile
            initial_layout: Optional physical qubit layout
            
        Returns:
            List of transpiled circuits
        """
        if self.pass_manager is None:
            logger.warning("No pass manager available, returning original circuits")
            return circuits
            
        # Update pass manager with layout if specified
        if initial_layout is not None:
            pm = generate_preset_pass_manager(
                optimization_level=1,
                backend=self.backend,
                initial_layout=initial_layout
            )
        else:
            pm = self.pass_manager
            
        transpiled = pm.run(circuits)
        logger.info(f"Transpiled {len(circuits)} circuits for {self.backend_name}")
        
        return transpiled
    
    def estimate_job_time(self, circuits: List[QuantumCircuit], 
                          shots: int) -> float:
        """
        Estimate QPU time for a batch of circuits.
        
        This is a rough estimate based on circuit depth and shot count.
        Actual time depends on queue, calibration, etc.
        
        Args:
            circuits: List of circuits
            shots: Shots per circuit
            
        Returns:
            Estimated time in seconds
        """
        # Very rough estimate: ~1ms per circuit execution
        # Includes measurement time, reset, etc.
        total_executions = len(circuits) * shots
        base_time = total_executions * 0.001  # 1ms per shot
        
        # Add overhead for circuit depth
        avg_depth = np.mean([c.depth() for c in circuits])
        depth_factor = 1 + (avg_depth / 100)  # Scale with depth
        
        estimated_seconds = base_time * depth_factor
        
        logger.info(f"Estimated QPU time: {estimated_seconds:.1f}s for {len(circuits)} circuits × {shots} shots")
        return estimated_seconds
    
    def run_experiment(self, circuits: List[QuantumCircuit],
                       shots: int = 4000,
                       initial_layout: Optional[List[int]] = None,
                       job_tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run a QEC experiment on hardware.
        
        Args:
            circuits: Circuits to execute
            shots: Number of shots per circuit
            initial_layout: Physical qubit layout
            job_tags: Optional tags for job tracking
            
        Returns:
            Dictionary containing job info and results
        """
        if self.backend is None:
            logger.error("No backend available")
            return {"error": "No backend"}
            
        # Transpile circuits
        transpiled = self.transpile_circuits(circuits, initial_layout)
        
        # Estimate time
        est_time = self.estimate_job_time(transpiled, shots)
        logger.info(f"Submitting job with estimated time {est_time:.1f}s")
        
        # Configure sampler
        options = SamplerOptions()
        options.default_shots = shots
        
        # Create sampler and run
        sampler = SamplerV2(mode=self.backend)
        
        try:
            job = sampler.run(transpiled, shots=shots)
            job_id = job.job_id()
            logger.info(f"Submitted job {job_id}")
            
            # Wait for results
            result = job.result()
            
            # Process results
            processed_results = self._process_results(result, circuits)
            
            # Create experiment record
            record = {
                "job_id": job_id,
                "backend": self.backend_name,
                "timestamp": datetime.now().isoformat(),
                "num_circuits": len(circuits),
                "shots": shots,
                "estimated_time_s": est_time,
                "initial_layout": initial_layout,
                "tags": job_tags or [],
                "results": processed_results
            }
            
            # Save results
            self._save_results(record)
            
            return record
            
        except Exception as e:
            logger.error(f"Job failed: {e}")
            return {"error": str(e)}
    
    def _process_results(self, result, circuits: List[QuantumCircuit]) -> List[Dict[str, Any]]:
        """Process raw sampler results into analyzed data."""
        processed = []
        
        for idx, pub_result in enumerate(result):
            circuit_name = circuits[idx].name if idx < len(circuits) else f"circuit_{idx}"
            
            # Get counts from result
            counts = pub_result.data.c.get_counts() if hasattr(pub_result.data, 'c') else {}
            
            processed.append({
                "circuit_name": circuit_name,
                "counts": counts,
                "total_shots": sum(counts.values())
            })
            
        return processed
    
    def _save_results(self, record: Dict[str, Any]) -> Path:
        """Save experiment results to disk."""
        timestamp = record["timestamp"].replace(":", "-").replace(".", "-")
        filename = f"experiment_{timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(record, f, indent=2, default=str)
            
        logger.info(f"Saved results to {filepath}")
        return filepath
    
    def run_qec_experiment_suite(self, 
                                  distances: List[int] = [3, 5],
                                  num_rounds: List[int] = [1, 3],
                                  code_type: str = "bit_flip",
                                  shots: int = 4000,
                                  physical_layout: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Run a complete QEC experiment suite.
        
        Creates and runs experiments for multiple code parameters.
        
        Args:
            distances: Code distances to test
            num_rounds: Syndrome rounds to test
            code_type: "bit_flip" or "phase_flip"
            shots: Shots per circuit
            physical_layout: Optional physical qubit layout (must be large enough)
            
        Returns:
            Complete experiment results
        """
        # Create experiment batch
        circuits = create_experiment_batch(distances, num_rounds, code_type)
        logger.info(f"Created {len(circuits)} experiment circuits")
        
        # Run on hardware
        results = self.run_experiment(
            circuits,
            shots=shots,
            initial_layout=physical_layout,
            job_tags=[f"qec_{code_type}", f"distances_{distances}", f"rounds_{num_rounds}"]
        )
        
        # Add QEC-specific analysis
        if "results" in results:
            results["qec_analysis"] = self._analyze_qec_results(
                results["results"], distances, num_rounds, code_type
            )
            
        return results
    
    def _analyze_qec_results(self, results: List[Dict],
                              distances: List[int],
                              num_rounds: List[int],
                              code_type: str) -> Dict[str, Any]:
        """Perform QEC-specific analysis on results."""
        analysis = {
            "logical_error_rates": {},
            "threshold_estimate": None
        }
        
        for result in results:
            name = result["circuit_name"]
            counts = result["counts"]
            
            # Parse circuit name for parameters
            # Format: rep_{type}_d{d}_r{r}_state{s}
            parts = name.split("_")
            try:
                d = int(parts[2].replace("d", ""))
                r = int(parts[3].replace("r", ""))
                state = int(parts[4].replace("state", ""))
            except (IndexError, ValueError):
                continue
                
            # Decode results
            decoder = SyndromeDecoder(d)
            decoded = decoder.decode_counts(counts)
            
            # Calculate logical error rate
            if state == 0:
                logical_error = decoded["logical_1_probability"]
            else:
                logical_error = decoded["logical_0_probability"]
                
            key = f"d{d}_r{r}"
            if key not in analysis["logical_error_rates"]:
                analysis["logical_error_rates"][key] = []
            analysis["logical_error_rates"][key].append({
                "state": state,
                "logical_error_rate": logical_error,
                "syndrome_error_rate": decoded["error_rate"]
            })
            
        return analysis


def quick_qec_demo(service: QiskitRuntimeService,
                   backend_name: str = "ibm_sherbrooke") -> Dict[str, Any]:
    """
    Run a quick QEC demonstration experiment.
    
    Uses minimal parameters to conserve QPU time.
    
    Args:
        service: QiskitRuntimeService instance
        backend_name: Target backend
        
    Returns:
        Experiment results
    """
    runner = QECExperimentRunner(service, backend_name)
    
    # Minimal experiment: distance-3 code, 1 round, 1000 shots
    return runner.run_qec_experiment_suite(
        distances=[3],
        num_rounds=[1],
        code_type="bit_flip",
        shots=1000
    )


if __name__ == "__main__":
    print("QEC Experiment Runner Module")
    print("\nExample usage:")
    print("""
    from qiskit_ibm_runtime import QiskitRuntimeService
    
    service = QiskitRuntimeService()
    runner = QECExperimentRunner(service, "ibm_sherbrooke")
    
    # Run quick demo
    results = quick_qec_demo(service)
    print(results)
    
    # Or run full experiment suite
    results = runner.run_qec_experiment_suite(
        distances=[3, 5],
        num_rounds=[1, 3, 5],
        code_type="bit_flip",
        shots=4000
    )
    """)
