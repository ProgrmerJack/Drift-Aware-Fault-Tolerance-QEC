"""
Calibration Data Schemas
========================

Dataclass definitions for calibration snapshots, qubit properties,
and experiment records. These schemas enforce type safety and enable
serialization to JSON/Parquet for reproducible data pipelines.

References:
- IBM Quantum Backend Properties: https://quantum.cloud.ibm.com/docs/api/qiskit-ibm-runtime
- Qiskit Target: https://docs.quantum.ibm.com/api/qiskit/qiskit.transpiler.Target
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
import json
from pathlib import Path

import pandas as pd
import numpy as np


@dataclass
class QubitProperties:
    """
    Properties for a single qubit from calibration data.
    
    Attributes:
        qubit_index: Physical qubit index on the device
        t1_us: T1 relaxation time in microseconds
        t2_us: T2 dephasing time in microseconds
        readout_error: Assignment error probability
        readout_length_ns: Measurement duration in nanoseconds
        frequency_ghz: Qubit frequency in GHz
        anharmonicity_ghz: Anharmonicity (α) in GHz
    """
    qubit_index: int
    t1_us: Optional[float] = None
    t2_us: Optional[float] = None
    readout_error: Optional[float] = None
    readout_length_ns: Optional[float] = None
    frequency_ghz: Optional[float] = None
    anharmonicity_ghz: Optional[float] = None
    
    def quality_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Compute a quality score for this qubit.
        
        Args:
            weights: Optional custom weights for each property
            
        Returns:
            Quality score (higher is better)
        """
        if weights is None:
            weights = {
                "t1": 0.30,
                "t2": 0.30,
                "readout": 0.40,
            }
        
        score = 0.0
        
        # T1 contribution (normalize to ~100 us scale)
        if self.t1_us is not None and self.t1_us > 0:
            score += weights.get("t1", 0) * min(self.t1_us / 100.0, 1.0)
        
        # T2 contribution (normalize to ~100 us scale)
        if self.t2_us is not None and self.t2_us > 0:
            score += weights.get("t2", 0) * min(self.t2_us / 100.0, 1.0)
        
        # Readout contribution (lower error is better)
        if self.readout_error is not None:
            score += weights.get("readout", 0) * (1.0 - min(self.readout_error, 1.0))
        
        return score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QubitProperties":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class GateProperties:
    """
    Properties for a gate type across all qubits/edges.
    
    Attributes:
        gate_name: Name of the gate (e.g., 'x', 'cx', 'ecr')
        num_qubits: Number of qubits the gate acts on
        avg_error: Average error rate across all instances
        min_error: Minimum error rate
        max_error: Maximum error rate
        avg_duration_ns: Average gate duration in nanoseconds
    """
    gate_name: str
    num_qubits: int
    avg_error: Optional[float] = None
    min_error: Optional[float] = None
    max_error: Optional[float] = None
    avg_duration_ns: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GateProperties":
        return cls(**data)


@dataclass
class EdgeProperties:
    """
    Properties for a two-qubit coupling (edge).
    
    Attributes:
        qubit_pair: Tuple of qubit indices (canonical ordering)
        gate_name: Name of the 2Q gate (e.g., 'ecr', 'cx')
        error: Gate error rate
        duration_ns: Gate duration in nanoseconds
    """
    qubit_pair: Tuple[int, int]
    gate_name: str
    error: Optional[float] = None
    duration_ns: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['qubit_pair'] = list(self.qubit_pair)  # JSON-serializable
        return d
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EdgeProperties":
        data['qubit_pair'] = tuple(data['qubit_pair'])
        return cls(**data)


@dataclass
class CalibrationSnapshot:
    """
    Complete calibration snapshot for a backend at a point in time.
    
    This is the core data structure for tracking drift over time.
    
    Attributes:
        backend_name: Name of the IBM Quantum backend
        collection_timestamp: When this snapshot was collected
        calibration_timestamp: When IBM last calibrated the device
        num_qubits: Total number of qubits on the device
        qubit_properties: Dict mapping qubit index to QubitProperties
        gate_properties: Dict mapping gate name to GateProperties
        edge_properties: Dict mapping qubit pairs to EdgeProperties
        supports_dynamic_circuits: Whether backend supports dynamic circuits
        processor_type: Processor type (e.g., 'Eagle r3')
        version: Backend version string
    """
    backend_name: str
    collection_timestamp: datetime
    calibration_timestamp: Optional[datetime]
    num_qubits: int
    qubit_properties: Dict[int, QubitProperties]
    gate_properties: Dict[str, GateProperties]
    edge_properties: Dict[Tuple[int, int], EdgeProperties]
    supports_dynamic_circuits: bool = True
    processor_type: str = "unknown"
    version: str = "unknown"
    
    def get_qubit_scores(self, weights: Optional[Dict[str, float]] = None) -> Dict[int, float]:
        """Get quality scores for all qubits."""
        return {
            q: props.quality_score(weights) 
            for q, props in self.qubit_properties.items()
        }
    
    def get_best_qubits(self, n: int, weights: Optional[Dict[str, float]] = None) -> List[int]:
        """Get the top-n qubits by quality score."""
        scores = self.get_qubit_scores(weights)
        sorted_qubits = sorted(scores.keys(), key=lambda q: scores[q], reverse=True)
        return sorted_qubits[:n]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            'backend_name': self.backend_name,
            'collection_timestamp': self.collection_timestamp.isoformat(),
            'calibration_timestamp': self.calibration_timestamp.isoformat() if self.calibration_timestamp else None,
            'num_qubits': self.num_qubits,
            'qubit_properties': {str(k): v.to_dict() for k, v in self.qubit_properties.items()},
            'gate_properties': {k: v.to_dict() for k, v in self.gate_properties.items()},
            'edge_properties': {f"{k[0]}_{k[1]}": v.to_dict() for k, v in self.edge_properties.items()},
            'supports_dynamic_circuits': self.supports_dynamic_circuits,
            'processor_type': self.processor_type,
            'version': self.version,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalibrationSnapshot":
        """Create from dictionary."""
        # Parse timestamps
        collection_ts = datetime.fromisoformat(data['collection_timestamp'])
        calibration_ts = datetime.fromisoformat(data['calibration_timestamp']) if data.get('calibration_timestamp') else None
        
        # Parse qubit properties
        qubit_props = {
            int(k): QubitProperties.from_dict(v) 
            for k, v in data.get('qubit_properties', {}).items()
        }
        
        # Parse gate properties
        gate_props = {
            k: GateProperties.from_dict(v) 
            for k, v in data.get('gate_properties', {}).items()
        }
        
        # Parse edge properties
        edge_props = {}
        for k, v in data.get('edge_properties', {}).items():
            parts = k.split('_')
            edge = (int(parts[0]), int(parts[1]))
            edge_props[edge] = EdgeProperties.from_dict(v)
        
        return cls(
            backend_name=data['backend_name'],
            collection_timestamp=collection_ts,
            calibration_timestamp=calibration_ts,
            num_qubits=data['num_qubits'],
            qubit_properties=qubit_props,
            gate_properties=gate_props,
            edge_properties=edge_props,
            supports_dynamic_circuits=data.get('supports_dynamic_circuits', True),
            processor_type=data.get('processor_type', 'unknown'),
            version=data.get('version', 'unknown'),
        )
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert qubit properties to a DataFrame for analysis."""
        records = []
        for q, props in self.qubit_properties.items():
            record = {
                'backend': self.backend_name,
                'timestamp': self.collection_timestamp,
                'qubit': q,
                't1_us': props.t1_us,
                't2_us': props.t2_us,
                'readout_error': props.readout_error,
                'frequency_ghz': props.frequency_ghz,
            }
            records.append(record)
        return pd.DataFrame(records)
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "CalibrationSnapshot":
        """Create from DataFrame (limited reconstruction)."""
        # Extract common fields from first row
        first_row = df.iloc[0]
        
        qubit_props = {}
        for _, row in df.iterrows():
            qubit_props[int(row['qubit'])] = QubitProperties(
                qubit_index=int(row['qubit']),
                t1_us=row.get('t1_us'),
                t2_us=row.get('t2_us'),
                readout_error=row.get('readout_error'),
                frequency_ghz=row.get('frequency_ghz'),
            )
        
        return cls(
            backend_name=first_row['backend'],
            collection_timestamp=first_row['timestamp'],
            calibration_timestamp=None,
            num_qubits=len(qubit_props),
            qubit_properties=qubit_props,
            gate_properties={},
            edge_properties={},
        )
    
    def save(self, filepath: Union[str, Path], format: str = "json") -> Path:
        """Save snapshot to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
        elif format == "parquet":
            self.to_dataframe().to_parquet(filepath)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        return filepath
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "CalibrationSnapshot":
        """Load snapshot from file."""
        filepath = Path(filepath)
        
        if filepath.suffix == ".json":
            with open(filepath, 'r') as f:
                return cls.from_dict(json.load(f))
        elif filepath.suffix == ".parquet":
            return cls.from_dataframe(pd.read_parquet(filepath))
        else:
            raise ValueError(f"Unknown file format: {filepath.suffix}")


@dataclass
class ExperimentRecord:
    """
    Record of a single QEC experiment run.
    
    This schema ensures all experimental data has consistent structure
    for reproducible analysis.
    
    Attributes:
        experiment_id: UUID for this experiment
        backend_name: IBM Quantum backend used
        timestamp_start: Experiment start time (UTC)
        timestamp_end: Experiment end time (UTC)
        selection_method: Qubit selection strategy used
        code_distance: Repetition code distance
        num_rounds: Number of syndrome extraction rounds
        shots: Number of shots executed
        logical_error_rate: Measured logical error rate
        raw_counts: Full measurement outcomes
        qubit_layout: Physical qubit mapping
        job_ids: IBM job IDs for provenance
        probe_results: Pre-experiment probe data
        calibration_snapshot_id: Reference to calibration snapshot
        git_hash: Git commit hash at experiment time
        protocol_hash: Hash of protocol.yaml for reproducibility
        notes: Free-text annotations
    """
    experiment_id: str
    backend_name: str
    timestamp_start: datetime
    timestamp_end: datetime
    selection_method: str  # 'static' | 'realtime' | 'drift_aware'
    code_distance: int
    num_rounds: int
    shots: int
    logical_error_rate: float
    raw_counts: Dict[str, int]
    
    # Optional fields
    qubit_layout: Optional[List[int]] = None
    job_ids: List[str] = field(default_factory=list)
    probe_results: Optional[Dict[str, Any]] = None
    calibration_snapshot_id: Optional[str] = None
    git_hash: Optional[str] = None
    protocol_hash: Optional[str] = None
    notes: Optional[str] = None
    
    # Computed fields (populated during analysis)
    logical_error_ci_low: Optional[float] = None
    logical_error_ci_high: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        d = asdict(self)
        d['timestamp_start'] = self.timestamp_start.isoformat()
        d['timestamp_end'] = self.timestamp_end.isoformat()
        return d
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentRecord":
        """Create from dictionary."""
        data['timestamp_start'] = datetime.fromisoformat(data['timestamp_start'])
        data['timestamp_end'] = datetime.fromisoformat(data['timestamp_end'])
        return cls(**data)
    
    def to_row(self) -> Dict[str, Any]:
        """Convert to a flat dictionary for DataFrame row."""
        return {
            'experiment_id': self.experiment_id,
            'backend_name': self.backend_name,
            'timestamp_start': self.timestamp_start,
            'timestamp_end': self.timestamp_end,
            'selection_method': self.selection_method,
            'code_distance': self.code_distance,
            'num_rounds': self.num_rounds,
            'shots': self.shots,
            'logical_error_rate': self.logical_error_rate,
            'logical_error_ci_low': self.logical_error_ci_low,
            'logical_error_ci_high': self.logical_error_ci_high,
            'n_qubits': len(self.qubit_layout) if self.qubit_layout else None,
            'git_hash': self.git_hash,
            'protocol_hash': self.protocol_hash,
        }
    
    def save(self, filepath: Union[str, Path]) -> Path:
        """Save record to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        
        return filepath
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "ExperimentRecord":
        """Load record from JSON file."""
        with open(filepath, 'r') as f:
            return cls.from_dict(json.load(f))


def records_to_dataframe(records: List[ExperimentRecord]) -> pd.DataFrame:
    """Convert a list of experiment records to a DataFrame."""
    rows = [r.to_row() for r in records]
    return pd.DataFrame(rows)


def dataframe_to_records(df: pd.DataFrame) -> List[ExperimentRecord]:
    """Convert a DataFrame back to experiment records (partial)."""
    records = []
    for _, row in df.iterrows():
        record = ExperimentRecord(
            experiment_id=row['experiment_id'],
            backend_name=row['backend_name'],
            timestamp_start=row['timestamp_start'],
            timestamp_end=row['timestamp_end'],
            selection_method=row['selection_method'],
            code_distance=row['code_distance'],
            num_rounds=row['num_rounds'],
            shots=row['shots'],
            logical_error_rate=row['logical_error_rate'],
            raw_counts={},  # Not stored in flat format
            logical_error_ci_low=row.get('logical_error_ci_low'),
            logical_error_ci_high=row.get('logical_error_ci_high'),
            git_hash=row.get('git_hash'),
            protocol_hash=row.get('protocol_hash'),
        )
        records.append(record)
    return records
