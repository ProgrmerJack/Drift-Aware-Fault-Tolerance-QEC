"""
Drift-Aware Fault-Tolerance QEC
==============================

A research framework for studying drift-aware fault-tolerant quantum error
correction on IBM Quantum Open Plan hardware.

Modules:
--------
calibration : Module A - Calibration & drift data lake
probes : Module B - Lightweight probe suite
qec : Module C - QEC repetition code experiments
analysis : Statistical analysis and A/B testing
utils : Job management and data utilities

Usage:
------
>>> from src.calibration import CalibrationCollector, DriftAnalyzer
>>> from src.probes import ProbeSuite, QubitSelector
>>> from src.qec import RepetitionCode, QECExperimentRunner
>>> from src.analysis import ABTestFramework
>>> from src.utils import QPUBudgetTracker
"""

__version__ = "0.1.0"
__author__ = "QEC Research Team"

# Import main classes for convenient access
from .calibration import CalibrationCollector, DriftAnalyzer, collect_daily_snapshot
from .probes import ProbeSuite, QubitSelector, RepetitionCodeLayoutGenerator
from .qec import RepetitionCode, SyndromeDecoder, QECExperimentRunner
from .analysis import DriftErrorAnalyzer, ABTestFramework, LogicalErrorRateModel, ResultsVisualizer
from .utils import QPUBudgetTracker, JobBatcher, ExperimentRecord, DataManager

__all__ = [
    # Version info
    "__version__",
    # Calibration
    "CalibrationCollector",
    "DriftAnalyzer",
    "collect_daily_snapshot",
    # Probes
    "ProbeSuite",
    "QubitSelector",
    "RepetitionCodeLayoutGenerator",
    # QEC
    "RepetitionCode",
    "SyndromeDecoder",
    "QECExperimentRunner",
    # Analysis
    "DriftErrorAnalyzer",
    "ABTestFramework",
    "LogicalErrorRateModel",
    "ResultsVisualizer",
    # Utils
    "QPUBudgetTracker",
    "JobBatcher",
    "ExperimentRecord",
    "DataManager"
]
