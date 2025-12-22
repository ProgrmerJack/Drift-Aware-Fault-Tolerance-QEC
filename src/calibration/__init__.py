"""
Calibration & Drift Data Lake Module
====================================

Daily snapshots of backend properties (T1/T2, gate errors, readout errors, timestamps).
Computes drift features: variance, rate-of-change, rolling z-scores, change-point detection.
"""

from .drift_collector import (
    CalibrationCollector,
    DriftAnalyzer,
    collect_daily_snapshot
)

from .schemas import (
    QubitProperties,
    GateProperties,
    EdgeProperties,
    CalibrationSnapshot,
    ExperimentRecord,
    records_to_dataframe,
    dataframe_to_records,
)

from .drift_tracker import DriftTracker

from .snapshot_collector import (
    BackendSnapshotCollector,
    collect_daily_snapshot as collect_backend_snapshot,
)

__all__ = [
    # Legacy drift_collector
    "CalibrationCollector",
    "DriftAnalyzer",
    "collect_daily_snapshot",
    # Schemas
    "QubitProperties",
    "GateProperties",
    "EdgeProperties",
    "CalibrationSnapshot",
    "ExperimentRecord",
    "records_to_dataframe",
    "dataframe_to_records",
    # Drift tracker (advanced)
    "DriftTracker",
    # Snapshot collector
    "BackendSnapshotCollector",
    "collect_backend_snapshot",
]
