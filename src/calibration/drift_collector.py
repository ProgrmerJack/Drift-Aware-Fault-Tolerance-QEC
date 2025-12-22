"""
Module A: Calibration & Drift Data Lake
========================================

Daily snapshots of backend properties (T1/T2, gate errors, readout errors, timestamps).
Computes drift features: variance, rate-of-change, rolling z-scores, change-point detection.

References:
- IBM Quantum Documentation: Get backend information with Qiskit
  https://quantum.cloud.ibm.com/docs/guides/get-qpu-information
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
import logging

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CalibrationCollector:
    """
    Collects and stores backend calibration data for drift analysis.
    
    This class implements the "Calibration & drift data lake" from the roadmap,
    enabling daily snapshots and historical analysis of QPU properties.
    """
    
    def __init__(self, data_dir: str = "data/calibration"):
        """
        Initialize the calibration collector.
        
        Args:
            data_dir: Directory to store calibration snapshots
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir = self.data_dir / "snapshots"
        self.snapshots_dir.mkdir(exist_ok=True)
        
    def collect_backend_properties(self, backend) -> Dict[str, Any]:
        """
        Collect current backend properties.
        
        IBM explicitly notes these properties update after calibration (≈24h cadence)
        and can be used for noise modeling / optimization.
        
        Args:
            backend: IBMBackend instance
            
        Returns:
            Dictionary containing backend properties with timestamps
        """
        timestamp = datetime.now().isoformat()
        
        # Get backend configuration
        config = backend.configuration()
        
        # Get backend properties (calibration data)
        properties = backend.properties()
        
        # Extract relevant data
        snapshot = {
            "timestamp": timestamp,
            "backend_name": backend.name,
            "backend_version": str(getattr(backend, 'version', 'unknown')),
            "num_qubits": config.num_qubits if hasattr(config, 'num_qubits') else len(properties.qubits),
            "qubits": {},
            "gates": {},
            "readout": {},
            "coupling_map": list(backend.coupling_map.get_edges()) if hasattr(backend, 'coupling_map') else [],
            "calibration_time": None,
        }
        
        # Extract qubit properties
        for qubit_idx, qubit_props in enumerate(properties.qubits):
            qubit_data = {}
            for prop in qubit_props:
                qubit_data[prop.name] = {
                    "value": prop.value,
                    "unit": prop.unit,
                    "date": prop.date.isoformat() if prop.date else None
                }
                # Track latest calibration time
                if prop.date and (snapshot["calibration_time"] is None or 
                                  prop.date.isoformat() > snapshot["calibration_time"]):
                    snapshot["calibration_time"] = prop.date.isoformat()
            snapshot["qubits"][str(qubit_idx)] = qubit_data
            
        # Extract gate properties
        for gate in properties.gates:
            gate_key = f"{gate.gate}_{gate.qubits}"
            snapshot["gates"][gate_key] = {
                "gate": gate.gate,
                "qubits": list(gate.qubits),
                "parameters": {}
            }
            for param in gate.parameters:
                snapshot["gates"][gate_key]["parameters"][param.name] = {
                    "value": param.value,
                    "unit": param.unit,
                    "date": param.date.isoformat() if param.date else None
                }
                
        # Extract readout errors
        for qubit_idx, qubit_props in enumerate(properties.qubits):
            for prop in qubit_props:
                if prop.name == "readout_error":
                    snapshot["readout"][str(qubit_idx)] = {
                        "error": prop.value,
                        "date": prop.date.isoformat() if prop.date else None
                    }
                    
        return snapshot
    
    def save_snapshot(self, snapshot: Dict[str, Any], backend_name: str) -> Path:
        """
        Save a calibration snapshot to disk.
        
        Args:
            snapshot: Calibration data dictionary
            backend_name: Name of the backend
            
        Returns:
            Path to saved file
        """
        # Create backend-specific directory
        backend_dir = self.snapshots_dir / backend_name
        backend_dir.mkdir(exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = snapshot["timestamp"].replace(":", "-").replace(".", "-")
        filename = f"calibration_{timestamp}.json"
        filepath = backend_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(snapshot, f, indent=2, default=str)
            
        logger.info(f"Saved calibration snapshot to {filepath}")
        return filepath
    
    def load_snapshots(self, backend_name: str, 
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Load calibration snapshots from disk.
        
        Args:
            backend_name: Name of the backend
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            List of calibration snapshots
        """
        backend_dir = self.snapshots_dir / backend_name
        if not backend_dir.exists():
            logger.warning(f"No snapshots found for backend {backend_name}")
            return []
            
        snapshots = []
        for filepath in sorted(backend_dir.glob("calibration_*.json")):
            with open(filepath, "r") as f:
                snapshot = json.load(f)
                
            # Filter by date if specified
            snapshot_time = datetime.fromisoformat(snapshot["timestamp"])
            if start_date and snapshot_time < start_date:
                continue
            if end_date and snapshot_time > end_date:
                continue
                
            snapshots.append(snapshot)
            
        return snapshots


class DriftAnalyzer:
    """
    Analyzes calibration drift from collected snapshots.
    
    Computes drift features including:
    - Variance and rate-of-change (daily differences)
    - Rolling z-scores
    - Change-point candidates
    - Cross-qubit correlations (crosstalk proxy)
    """
    
    def __init__(self):
        self.property_names = ["T1", "T2", "readout_error", "gate_error"]
        
    def extract_time_series(self, snapshots: List[Dict[str, Any]], 
                            qubit_idx: int, 
                            property_name: str) -> pd.Series:
        """
        Extract a time series for a specific qubit property.
        
        Args:
            snapshots: List of calibration snapshots
            qubit_idx: Index of the qubit
            property_name: Name of the property (T1, T2, readout_error, etc.)
            
        Returns:
            Pandas Series with datetime index
        """
        data = []
        timestamps = []
        
        for snapshot in snapshots:
            qubit_key = str(qubit_idx)
            if qubit_key in snapshot.get("qubits", {}):
                qubit_data = snapshot["qubits"][qubit_key]
                if property_name in qubit_data:
                    value = qubit_data[property_name]["value"]
                    if value is not None:
                        data.append(value)
                        timestamps.append(datetime.fromisoformat(snapshot["timestamp"]))
                        
        if not data:
            return pd.Series(dtype=float)
            
        return pd.Series(data, index=pd.DatetimeIndex(timestamps), name=f"q{qubit_idx}_{property_name}")
    
    def compute_drift_features(self, time_series: pd.Series, 
                               window_size: int = 7) -> Dict[str, float]:
        """
        Compute drift features for a time series.
        
        Args:
            time_series: Pandas Series with datetime index
            window_size: Window size for rolling statistics
            
        Returns:
            Dictionary of drift features
        """
        if len(time_series) < 2:
            return {
                "variance": np.nan,
                "rate_of_change": np.nan,
                "rolling_zscore": np.nan,
                "max_change": np.nan,
                "stability_score": np.nan
            }
            
        # Basic statistics
        variance = time_series.var()
        
        # Rate of change (daily differences)
        diff = time_series.diff()
        rate_of_change = diff.mean()
        max_change = diff.abs().max()
        
        # Rolling z-score
        rolling_mean = time_series.rolling(window=min(window_size, len(time_series))).mean()
        rolling_std = time_series.rolling(window=min(window_size, len(time_series))).std()
        zscore = (time_series - rolling_mean) / (rolling_std + 1e-10)
        current_zscore = zscore.iloc[-1] if len(zscore) > 0 else np.nan
        
        # Stability score (inverse of coefficient of variation)
        cv = time_series.std() / (time_series.mean() + 1e-10)
        stability_score = 1.0 / (1.0 + cv)
        
        return {
            "variance": variance,
            "rate_of_change": rate_of_change,
            "rolling_zscore": current_zscore,
            "max_change": max_change,
            "stability_score": stability_score
        }
    
    def detect_change_points(self, time_series: pd.Series, 
                             threshold_sigma: float = 2.0) -> List[datetime]:
        """
        Detect potential change points in calibration data.
        
        Args:
            time_series: Pandas Series with datetime index
            threshold_sigma: Number of standard deviations for detection
            
        Returns:
            List of datetime objects where change points were detected
        """
        if len(time_series) < 3:
            return []
            
        diff = time_series.diff()
        mean_diff = diff.mean()
        std_diff = diff.std()
        
        if std_diff < 1e-10:
            return []
            
        # Find points where change exceeds threshold
        z_diff = (diff - mean_diff) / std_diff
        change_points = time_series.index[z_diff.abs() > threshold_sigma].tolist()
        
        return change_points
    
    def compute_cross_qubit_correlation(self, snapshots: List[Dict[str, Any]], 
                                         property_name: str = "T1") -> pd.DataFrame:
        """
        Compute cross-qubit correlations as a crosstalk proxy.
        
        Args:
            snapshots: List of calibration snapshots
            property_name: Property to analyze
            
        Returns:
            Correlation matrix as DataFrame
        """
        if not snapshots:
            return pd.DataFrame()
            
        # Get number of qubits from first snapshot
        num_qubits = snapshots[0].get("num_qubits", 0)
        
        # Extract time series for all qubits
        series_dict = {}
        for q in range(num_qubits):
            ts = self.extract_time_series(snapshots, q, property_name)
            if len(ts) > 0:
                series_dict[f"q{q}"] = ts
                
        if len(series_dict) < 2:
            return pd.DataFrame()
            
        # Align all series to common timestamps
        df = pd.DataFrame(series_dict)
        
        # Compute correlation matrix
        return df.corr()
    
    def generate_drift_report(self, snapshots: List[Dict[str, Any]], 
                               backend_name: str) -> Dict[str, Any]:
        """
        Generate a comprehensive drift analysis report.
        
        Args:
            snapshots: List of calibration snapshots
            backend_name: Name of the backend
            
        Returns:
            Dictionary containing drift analysis results
        """
        if not snapshots:
            return {"error": "No snapshots available for analysis"}
            
        report = {
            "backend_name": backend_name,
            "analysis_timestamp": datetime.now().isoformat(),
            "num_snapshots": len(snapshots),
            "date_range": {
                "start": snapshots[0]["timestamp"],
                "end": snapshots[-1]["timestamp"]
            },
            "qubit_drift": {},
            "gate_drift": {},
            "cross_correlations": {},
            "change_points": {},
            "recommendations": []
        }
        
        num_qubits = snapshots[0].get("num_qubits", 0)
        
        # Analyze each qubit
        for q in range(num_qubits):
            qubit_report = {}
            for prop in ["T1", "T2"]:
                ts = self.extract_time_series(snapshots, q, prop)
                if len(ts) > 0:
                    features = self.compute_drift_features(ts)
                    change_pts = self.detect_change_points(ts)
                    qubit_report[prop] = {
                        "features": features,
                        "change_points": [cp.isoformat() for cp in change_pts]
                    }
            report["qubit_drift"][f"q{q}"] = qubit_report
            
        # Cross-qubit correlations
        for prop in ["T1", "T2"]:
            corr = self.compute_cross_qubit_correlation(snapshots, prop)
            if not corr.empty:
                report["cross_correlations"][prop] = corr.to_dict()
                
        # Generate recommendations
        recommendations = self._generate_recommendations(report)
        report["recommendations"] = recommendations
        
        return report
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """
        Generate actionable recommendations based on drift analysis.
        
        Args:
            report: Drift analysis report
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Find unstable qubits
        unstable_qubits = []
        for qubit, data in report.get("qubit_drift", {}).items():
            for prop, prop_data in data.items():
                features = prop_data.get("features", {})
                if features.get("stability_score", 1.0) < 0.8:
                    unstable_qubits.append(qubit)
                    break
                    
        if unstable_qubits:
            recommendations.append(
                f"Qubits {', '.join(unstable_qubits)} show significant drift. "
                "Consider using real-time probes before experiments on these qubits."
            )
            
        # Check for correlated drift
        for prop, corr in report.get("cross_correlations", {}).items():
            if corr:
                # Find high correlations (excluding diagonal)
                high_corr_pairs = []
                for q1 in corr:
                    for q2, val in corr[q1].items():
                        if q1 != q2 and abs(val) > 0.7:
                            pair = tuple(sorted([q1, q2]))
                            if pair not in high_corr_pairs:
                                high_corr_pairs.append(pair)
                if high_corr_pairs:
                    recommendations.append(
                        f"High {prop} correlation detected between: {high_corr_pairs}. "
                        "This may indicate crosstalk or shared noise sources."
                    )
                    
        # Check for recent change points
        recent_changes = []
        for qubit, data in report.get("qubit_drift", {}).items():
            for prop, prop_data in data.items():
                change_pts = prop_data.get("change_points", [])
                if change_pts:
                    recent_changes.append(f"{qubit} ({prop})")
                    
        if recent_changes:
            recommendations.append(
                f"Recent calibration changes detected in: {', '.join(recent_changes)}. "
                "Update layout selection based on current properties."
            )
            
        if not recommendations:
            recommendations.append(
                "Calibration appears stable. Standard layout selection from backend properties is suitable."
            )
            
        return recommendations


def collect_daily_snapshot(service, backend_name: str, 
                           collector: Optional[CalibrationCollector] = None) -> Path:
    """
    Convenience function to collect and save a daily calibration snapshot.
    
    Args:
        service: QiskitRuntimeService instance
        backend_name: Name of the backend
        collector: Optional CalibrationCollector instance
        
    Returns:
        Path to saved snapshot file
    """
    if collector is None:
        collector = CalibrationCollector()
        
    backend = service.backend(backend_name)
    snapshot = collector.collect_backend_properties(backend)
    return collector.save_snapshot(snapshot, backend_name)


if __name__ == "__main__":
    # Example usage (requires IBM Quantum credentials)
    print("Drift Collector Module - Run with IBM Quantum credentials to collect data")
    print("\nExample usage:")
    print("""
    from qiskit_ibm_runtime import QiskitRuntimeService
    
    # Initialize service
    service = QiskitRuntimeService()
    
    # Collect and save snapshot
    filepath = collect_daily_snapshot(service, "ibm_sherbrooke")
    
    # Analyze drift
    collector = CalibrationCollector()
    snapshots = collector.load_snapshots("ibm_sherbrooke")
    
    analyzer = DriftAnalyzer()
    report = analyzer.generate_drift_report(snapshots, "ibm_sherbrooke")
    print(report)
    """)
