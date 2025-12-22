"""
Backend Snapshot Collector
==========================

Collects and stores daily snapshots of IBM Quantum backend properties
for drift analysis and historical tracking.

References:
- https://quantum.cloud.ibm.com/docs/guides/get-qpu-information
- https://quantum.cloud.ibm.com/docs/api/qiskit-ibm-runtime/ibm-backend
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from qiskit_ibm_runtime import QiskitRuntimeService

from .schemas import (
    CalibrationSnapshot,
    QubitProperties,
    GateProperties,
    EdgeProperties,
)

logger = logging.getLogger(__name__)


class BackendSnapshotCollector:
    """
    Collects calibration snapshots from IBM Quantum backends.
    
    This class pulls backend properties and stores them in a structured
    format for drift analysis. Properties update approximately daily
    after calibration cycles.
    
    Attributes:
        service: QiskitRuntimeService instance
        data_dir: Directory for storing snapshots
    
    Example:
        >>> service = QiskitRuntimeService()
        >>> collector = BackendSnapshotCollector(service)
        >>> snapshot = collector.collect_snapshot("ibm_brisbane")
        >>> collector.save_snapshot(snapshot, "data/calibration_snapshots/")
    """
    
    def __init__(
        self,
        service: Optional[QiskitRuntimeService] = None,
        data_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the snapshot collector.
        
        Args:
            service: QiskitRuntimeService instance. If None, creates a new one.
            data_dir: Directory for storing snapshots. Defaults to "data/calibration_snapshots/"
        """
        self.service = service or QiskitRuntimeService()
        self.data_dir = Path(data_dir) if data_dir else Path("data/calibration_snapshots/")
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_snapshot(self, backend_name: str) -> CalibrationSnapshot:
        """
        Collect a calibration snapshot from the specified backend.
        
        Args:
            backend_name: Name of the IBM Quantum backend (e.g., "ibm_brisbane")
        
        Returns:
            CalibrationSnapshot containing all backend properties
        
        Raises:
            ValueError: If backend is not accessible
        """
        logger.info(f"Collecting snapshot from {backend_name}")
        
        # Get backend instance
        backend = self.service.backend(backend_name)
        
        # Get current properties (updates after calibration, ~daily)
        # Note: Using target instead of deprecated properties() where possible
        target = backend.target
        
        # Collection timestamp
        collection_time = datetime.now(timezone.utc)
        
        # Extract qubit properties
        qubit_props = self._extract_qubit_properties(backend, target)
        
        # Extract gate properties  
        gate_props = self._extract_gate_properties(target)
        
        # Extract edge (2Q coupling) properties
        edge_props = self._extract_edge_properties(target)
        
        # Get calibration timestamp from backend
        calibration_time = self._get_calibration_timestamp(backend)
        
        # Check dynamic circuit support
        supports_dynamic = self._check_dynamic_circuits(backend)
        
        snapshot = CalibrationSnapshot(
            backend_name=backend_name,
            collection_timestamp=collection_time,
            calibration_timestamp=calibration_time,
            num_qubits=backend.num_qubits,
            qubit_properties=qubit_props,
            gate_properties=gate_props,
            edge_properties=edge_props,
            supports_dynamic_circuits=supports_dynamic,
            processor_type=getattr(backend, 'processor_type', 'unknown'),
            version=str(getattr(backend, 'version', 'unknown')),
        )
        
        logger.info(f"Collected snapshot with {len(qubit_props)} qubits, {len(gate_props)} gates")
        return snapshot
    
    def _extract_qubit_properties(
        self, 
        backend: Any, 
        target: Any
    ) -> Dict[int, QubitProperties]:
        """Extract T1, T2, readout error, frequency for each qubit."""
        qubit_props = {}
        
        for qubit in range(backend.num_qubits):
            props = QubitProperties(
                qubit_index=qubit,
                t1_us=None,
                t2_us=None,
                readout_error=None,
                readout_length_ns=None,
                frequency_ghz=None,
            )
            
            try:
                # T1 and T2 from target qubit properties
                if hasattr(target, 'qubit_properties') and target.qubit_properties:
                    q_props = target.qubit_properties(qubit)
                    if q_props:
                        if hasattr(q_props, 't1') and q_props.t1:
                            props.t1_us = q_props.t1 * 1e6  # Convert to microseconds
                        if hasattr(q_props, 't2') and q_props.t2:
                            props.t2_us = q_props.t2 * 1e6
                        if hasattr(q_props, 'frequency') and q_props.frequency:
                            props.frequency_ghz = q_props.frequency / 1e9
                
                # Readout error from measure instruction
                if 'measure' in target.operation_names:
                    measure_props = target['measure'].get((qubit,))
                    if measure_props and hasattr(measure_props, 'error'):
                        props.readout_error = measure_props.error
                    if measure_props and hasattr(measure_props, 'duration'):
                        props.readout_length_ns = measure_props.duration * 1e9
                        
            except Exception as e:
                logger.warning(f"Could not extract properties for qubit {qubit}: {e}")
            
            qubit_props[qubit] = props
        
        return qubit_props
    
    def _extract_gate_properties(self, target: Any) -> Dict[str, GateProperties]:
        """Extract error rates and durations for all gates."""
        gate_props = {}
        
        for gate_name in target.operation_names:
            try:
                # Skip measurement (handled in qubit properties)
                if gate_name in ['measure', 'delay', 'reset']:
                    continue
                
                gate_info = target[gate_name]
                
                # Aggregate statistics across all qubits/edges
                errors = []
                durations = []
                
                for qargs, props in gate_info.items():
                    if props:
                        if hasattr(props, 'error') and props.error is not None:
                            errors.append(props.error)
                        if hasattr(props, 'duration') and props.duration is not None:
                            durations.append(props.duration)
                
                gate_props[gate_name] = GateProperties(
                    gate_name=gate_name,
                    num_qubits=len(next(iter(gate_info.keys()))) if gate_info else 0,
                    avg_error=sum(errors) / len(errors) if errors else None,
                    min_error=min(errors) if errors else None,
                    max_error=max(errors) if errors else None,
                    avg_duration_ns=(sum(durations) / len(durations) * 1e9) if durations else None,
                )
                
            except Exception as e:
                logger.warning(f"Could not extract properties for gate {gate_name}: {e}")
        
        return gate_props
    
    def _extract_edge_properties(self, target: Any) -> Dict[tuple, EdgeProperties]:
        """Extract 2Q gate properties for each connected qubit pair."""
        edge_props = {}
        
        # Find the primary 2Q gate (usually 'ecr', 'cx', or 'cz')
        two_qubit_gates = ['ecr', 'cx', 'cz', 'rzz']
        primary_2q_gate = None
        
        for gate in two_qubit_gates:
            if gate in target.operation_names:
                primary_2q_gate = gate
                break
        
        if not primary_2q_gate:
            logger.warning("No recognized 2Q gate found in target")
            return edge_props
        
        try:
            gate_info = target[primary_2q_gate]
            
            for qargs, props in gate_info.items():
                if len(qargs) == 2 and props:
                    edge = tuple(sorted(qargs))  # Canonical ordering
                    
                    edge_props[edge] = EdgeProperties(
                        qubit_pair=edge,
                        gate_name=primary_2q_gate,
                        error=props.error if hasattr(props, 'error') else None,
                        duration_ns=props.duration * 1e9 if hasattr(props, 'duration') and props.duration else None,
                    )
                    
        except Exception as e:
            logger.warning(f"Could not extract edge properties: {e}")
        
        return edge_props
    
    def _get_calibration_timestamp(self, backend: Any) -> Optional[datetime]:
        """Get the timestamp of the last calibration."""
        try:
            # Try to get from backend properties
            if hasattr(backend, 'properties') and callable(backend.properties):
                props = backend.properties()
                if props and hasattr(props, 'last_update_date'):
                    return props.last_update_date
                    
        except Exception as e:
            logger.warning(f"Could not get calibration timestamp: {e}")
        
        return None
    
    def _check_dynamic_circuits(self, backend: Any) -> bool:
        """Check if backend supports dynamic circuits (qasm3)."""
        try:
            if hasattr(backend, 'target'):
                target = backend.target
                # Check for supported features
                if hasattr(target, 'supported_features'):
                    return 'qasm3' in target.supported_features
            
            # Alternative check via configuration
            if hasattr(backend, 'configuration'):
                config = backend.configuration()
                if hasattr(config, 'supported_features'):
                    return 'qasm3' in config.supported_features
                    
        except Exception as e:
            logger.warning(f"Could not check dynamic circuit support: {e}")
        
        return False
    
    def save_snapshot(
        self,
        snapshot: CalibrationSnapshot,
        output_dir: Optional[Union[str, Path]] = None,
        file_format: str = "json",
    ) -> Path:
        """
        Save a calibration snapshot to disk.
        
        Args:
            snapshot: CalibrationSnapshot to save
            output_dir: Directory to save to (uses self.data_dir if None)
            file_format: Output format ("json" or "parquet")
        
        Returns:
            Path to the saved file
        """
        output_dir = Path(output_dir) if output_dir else self.data_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp_str = snapshot.collection_timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"{snapshot.backend_name}_{timestamp_str}"
        
        if file_format == "json":
            filepath = output_dir / f"{filename}.json"
            with open(filepath, 'w') as f:
                json.dump(snapshot.to_dict(), f, indent=2, default=str)
        
        elif file_format == "parquet":
            filepath = output_dir / f"{filename}.parquet"
            df = snapshot.to_dataframe()
            df.to_parquet(filepath)
        
        else:
            raise ValueError(f"Unknown format: {file_format}")
        
        logger.info(f"Saved snapshot to {filepath}")
        return filepath
    
    def load_snapshot(self, filepath: Union[str, Path]) -> CalibrationSnapshot:
        """Load a calibration snapshot from disk."""
        filepath = Path(filepath)
        
        if filepath.suffix == ".json":
            with open(filepath, 'r') as f:
                data = json.load(f)
            return CalibrationSnapshot.from_dict(data)
        
        elif filepath.suffix == ".parquet":
            df = pd.read_parquet(filepath)
            return CalibrationSnapshot.from_dataframe(df)
        
        else:
            raise ValueError(f"Unknown file format: {filepath.suffix}")
    
    def load_all_snapshots(
        self,
        backend_name: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[CalibrationSnapshot]:
        """
        Load all snapshots from the data directory.
        
        Args:
            backend_name: Filter by backend name
            start_date: Filter by start date
            end_date: Filter by end date
            
        Returns:
            List of CalibrationSnapshot objects
        """
        snapshots = []
        
        for filepath in sorted(self.data_dir.glob("*.json")):
            try:
                snapshot = self.load_snapshot(filepath)
                
                # Apply filters
                if backend_name and snapshot.backend_name != backend_name:
                    continue
                if start_date and snapshot.collection_timestamp < start_date:
                    continue
                if end_date and snapshot.collection_timestamp > end_date:
                    continue
                
                snapshots.append(snapshot)
                
            except Exception as e:
                logger.warning(f"Failed to load {filepath}: {e}")
        
        return snapshots
    
    def list_available_backends(self) -> List[Dict[str, Any]]:
        """List all available backends with basic info."""
        backends_info = []
        
        for backend in self.service.backends():
            info = {
                'name': backend.name,
                'num_qubits': backend.num_qubits,
                'status': backend.status().status_msg,
                'operational': backend.status().operational,
            }
            backends_info.append(info)
        
        return backends_info


def collect_daily_snapshot(
    service: Optional[QiskitRuntimeService] = None,
    backend_name: str = "ibm_brisbane",
    output_dir: str = "data/calibration",
) -> Path:
    """
    Convenience function to collect and save a daily calibration snapshot.
    
    Args:
        service: QiskitRuntimeService instance (creates new one if None)
        backend_name: Name of the backend
        output_dir: Directory to save snapshot
        
    Returns:
        Path to saved snapshot file
    """
    collector = BackendSnapshotCollector(service, data_dir=output_dir)
    snapshot = collector.collect_snapshot(backend_name)
    return collector.save_snapshot(snapshot)


def main():
    """CLI entry point for snapshot collection."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect backend calibration snapshots")
    parser.add_argument("backend", help="Backend name (e.g., ibm_brisbane)")
    parser.add_argument("--output-dir", "-o", default="data/calibration_snapshots/",
                        help="Output directory")
    parser.add_argument("--format", "-f", choices=["json", "parquet"], default="json",
                        help="Output format")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    collector = BackendSnapshotCollector(data_dir=args.output_dir)
    snapshot = collector.collect_snapshot(args.backend)
    filepath = collector.save_snapshot(snapshot, file_format=args.format)
    
    print(f"Snapshot saved to: {filepath}")


if __name__ == "__main__":
    main()
