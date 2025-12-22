"""
Utilities Module: Job Management
================================

Manages IBM Quantum job submission, batching, and QPU budget tracking.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QPUBudgetTracker:
    """
    Tracks QPU time usage for Open Plan (~10 min per 28-day window).
    
    Helps optimize job scheduling to stay within budget constraints.
    """
    
    def __init__(self, monthly_budget_seconds: float = 600,
                 tracking_file: str = "data/qpu_budget.json"):
        """
        Initialize budget tracker.
        
        Args:
            monthly_budget_seconds: QPU time budget in seconds (default: 600 = 10 min)
            tracking_file: File to persist usage tracking
        """
        self.monthly_budget = monthly_budget_seconds
        self.tracking_file = Path(tracking_file)
        self.tracking_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.usage_history = self._load_history()
        
    def _load_history(self) -> List[Dict[str, Any]]:
        """Load usage history from file."""
        if self.tracking_file.exists():
            with open(self.tracking_file) as f:
                return json.load(f)
        return []
    
    def _save_history(self):
        """Save usage history to file."""
        with open(self.tracking_file, 'w') as f:
            json.dump(self.usage_history, f, indent=2, default=str)
            
    def record_usage(self, job_id: str, qpu_seconds: float, 
                     description: str = ""):
        """
        Record QPU time usage for a job.
        
        Args:
            job_id: IBM Quantum job ID
            qpu_seconds: QPU time consumed in seconds
            description: Optional job description
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "job_id": job_id,
            "qpu_seconds": qpu_seconds,
            "description": description
        }
        self.usage_history.append(entry)
        self._save_history()
        
        logger.info(f"Recorded {qpu_seconds:.1f}s QPU usage for job {job_id}")
        
    def get_current_window_usage(self) -> float:
        """
        Get total QPU usage in the current 28-day window.
        
        Returns:
            Total seconds used in current window
        """
        window_start = datetime.now() - timedelta(days=28)
        
        total = 0.0
        for entry in self.usage_history:
            entry_time = datetime.fromisoformat(entry["timestamp"])
            if entry_time >= window_start:
                total += entry["qpu_seconds"]
                
        return total
    
    def get_remaining_budget(self) -> float:
        """
        Get remaining QPU budget for current window.
        
        Returns:
            Remaining seconds available
        """
        used = self.get_current_window_usage()
        return max(0, self.monthly_budget - used)
    
    def can_run_job(self, estimated_seconds: float) -> bool:
        """
        Check if a job can be run within budget.
        
        Args:
            estimated_seconds: Estimated QPU time for job
            
        Returns:
            True if job fits within remaining budget
        """
        remaining = self.get_remaining_budget()
        return estimated_seconds <= remaining
    
    def get_usage_report(self) -> Dict[str, Any]:
        """
        Generate a usage report for the current window.
        
        Returns:
            Report dictionary with usage statistics
        """
        window_start = datetime.now() - timedelta(days=28)
        
        window_jobs = [
            entry for entry in self.usage_history
            if datetime.fromisoformat(entry["timestamp"]) >= window_start
        ]
        
        total_used = sum(entry["qpu_seconds"] for entry in window_jobs)
        
        return {
            "window_start": window_start.isoformat(),
            "window_end": (window_start + timedelta(days=28)).isoformat(),
            "total_budget_seconds": self.monthly_budget,
            "total_used_seconds": total_used,
            "remaining_seconds": self.monthly_budget - total_used,
            "utilization_percent": (total_used / self.monthly_budget) * 100,
            "num_jobs": len(window_jobs),
            "jobs": window_jobs
        }


class JobBatcher:
    """
    Batches circuits for efficient job submission.
    
    Combines multiple experiments into single jobs to reduce overhead
    and optimize QPU time usage.
    """
    
    def __init__(self, max_circuits_per_job: int = 300,
                 max_shots_per_job: int = 100000):
        """
        Initialize job batcher.
        
        Args:
            max_circuits_per_job: Maximum circuits in one job
            max_shots_per_job: Maximum total shots in one job
        """
        self.max_circuits = max_circuits_per_job
        self.max_shots = max_shots_per_job
        
    def create_batches(self, circuits: List[Any], 
                       shots_per_circuit: int) -> List[List[Any]]:
        """
        Split circuits into optimal batches.
        
        Args:
            circuits: List of QuantumCircuit objects
            shots_per_circuit: Shots for each circuit
            
        Returns:
            List of circuit batches
        """
        batches = []
        current_batch = []
        current_shots = 0
        
        for circuit in circuits:
            circuit_shots = shots_per_circuit
            
            # Check if adding this circuit exceeds limits
            if (len(current_batch) >= self.max_circuits or
                current_shots + circuit_shots > self.max_shots):
                # Start new batch
                if current_batch:
                    batches.append(current_batch)
                current_batch = [circuit]
                current_shots = circuit_shots
            else:
                current_batch.append(circuit)
                current_shots += circuit_shots
                
        # Add remaining circuits
        if current_batch:
            batches.append(current_batch)
            
        logger.info(f"Split {len(circuits)} circuits into {len(batches)} batches")
        return batches
    
    def estimate_batch_time(self, batch: List[Any], 
                            shots_per_circuit: int) -> float:
        """
        Estimate QPU time for a batch.
        
        Args:
            batch: List of circuits
            shots_per_circuit: Shots per circuit
            
        Returns:
            Estimated time in seconds
        """
        # Rough estimate: ~1ms per circuit execution
        total_executions = len(batch) * shots_per_circuit
        base_time = total_executions * 0.001
        
        # Add overhead for circuit depth
        try:
            avg_depth = sum(c.depth() for c in batch) / len(batch)
            depth_factor = 1 + (avg_depth / 100)
        except AttributeError:
            depth_factor = 1.5  # Default if depth() not available
            
        return base_time * depth_factor


class ExperimentRecord:
    """
    Structured experiment record for tracking and reproducibility.
    
    Records all parameters, results, and metadata for each experiment.
    """
    
    def __init__(self, experiment_type: str,
                 backend_name: str,
                 timestamp: Optional[datetime] = None):
        """
        Initialize experiment record.
        
        Args:
            experiment_type: Type of experiment (e.g., "qec_repetition", "probe")
            backend_name: Target backend
            timestamp: Optional timestamp (default: now)
        """
        self.experiment_type = experiment_type
        self.backend_name = backend_name
        self.timestamp = timestamp or datetime.now()
        
        self.record = {
            "experiment_type": experiment_type,
            "backend_name": backend_name,
            "timestamp": self.timestamp.isoformat(),
            "parameters": {},
            "circuits": [],
            "job_ids": [],
            "results": {},
            "analysis": {},
            "metadata": {}
        }
        
    def set_parameters(self, **kwargs):
        """Set experiment parameters."""
        self.record["parameters"].update(kwargs)
        
    def add_circuit_info(self, name: str, num_qubits: int, 
                         depth: int, **kwargs):
        """Add circuit information."""
        self.record["circuits"].append({
            "name": name,
            "num_qubits": num_qubits,
            "depth": depth,
            **kwargs
        })
        
    def add_job(self, job_id: str, **kwargs):
        """Record a submitted job."""
        self.record["job_ids"].append({
            "job_id": job_id,
            "submitted_at": datetime.now().isoformat(),
            **kwargs
        })
        
    def set_results(self, results: Dict[str, Any]):
        """Set experiment results."""
        self.record["results"] = results
        
    def set_analysis(self, analysis: Dict[str, Any]):
        """Set analysis results."""
        self.record["analysis"] = analysis
        
    def add_metadata(self, **kwargs):
        """Add metadata."""
        self.record["metadata"].update(kwargs)
        
    def save(self, output_dir: str = "data/experiments") -> Path:
        """
        Save experiment record to file.
        
        Args:
            output_dir: Output directory
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp_str = self.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.experiment_type}_{timestamp_str}.json"
        filepath = output_path / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.record, f, indent=2, default=str)
            
        logger.info(f"Saved experiment record to {filepath}")
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> 'ExperimentRecord':
        """
        Load experiment record from file.
        
        Args:
            filepath: Path to record file
            
        Returns:
            Loaded ExperimentRecord
        """
        with open(filepath) as f:
            data = json.load(f)
            
        record = cls(
            experiment_type=data["experiment_type"],
            backend_name=data["backend_name"],
            timestamp=datetime.fromisoformat(data["timestamp"])
        )
        record.record = data
        
        return record
    
    def to_dict(self) -> Dict[str, Any]:
        """Return record as dictionary."""
        return self.record


def create_experiment_session(backend_name: str = "ibm_sherbrooke",
                               budget_tracker: Optional[QPUBudgetTracker] = None) -> Dict[str, Any]:
    """
    Initialize an experiment session with budget tracking.
    
    Args:
        backend_name: Target backend
        budget_tracker: Optional existing budget tracker
        
    Returns:
        Session configuration dictionary
    """
    if budget_tracker is None:
        budget_tracker = QPUBudgetTracker()
        
    remaining = budget_tracker.get_remaining_budget()
    
    session = {
        "backend": backend_name,
        "session_start": datetime.now().isoformat(),
        "budget_remaining_seconds": remaining,
        "can_run_experiments": remaining > 30,  # Minimum for small experiment
        "recommendations": []
    }
    
    # Add recommendations based on remaining budget
    if remaining < 60:
        session["recommendations"].append(
            "⚠️ Low QPU budget remaining. Consider waiting for next window."
        )
    elif remaining < 300:
        session["recommendations"].append(
            "💡 Limited budget. Prioritize most important experiments."
        )
    else:
        session["recommendations"].append(
            "✅ Sufficient budget for comprehensive experiments."
        )
        
    logger.info(f"Created session with {remaining:.0f}s QPU budget remaining")
    return session


if __name__ == "__main__":
    print("Job Management Utilities")
    print("\nExample usage:")
    print("""
    # Budget tracking
    tracker = QPUBudgetTracker()
    print(f"Remaining budget: {tracker.get_remaining_budget():.0f}s")
    
    # Record usage
    tracker.record_usage("job_123", 45.5, "QEC experiment d=5")
    
    # Usage report
    report = tracker.get_usage_report()
    print(f"Utilization: {report['utilization_percent']:.1f}%")
    
    # Job batching
    batcher = JobBatcher()
    batches = batcher.create_batches(circuits, shots_per_circuit=4000)
    """)
