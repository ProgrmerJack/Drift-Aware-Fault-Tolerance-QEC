"""
DAQEC Toolkit: Drift-Aware Quantum Error Correction
====================================================

A lightweight, cloud-deployable toolkit for drift-aware QEC on superconducting
quantum processors.

Installation:
    pip install daqec

Quick Start:
    >>> from daqec import select_qubits_drift_aware, recommend_probe_interval, decode_adaptive
    >>> 
    >>> # 1. Run probes and select best qubit chain
    >>> chains = select_qubits_drift_aware(probe_results, code_distance=5, topology=backend.coupling_map)
    >>> 
    >>> # 2. Get recommended probe interval
    >>> interval = recommend_probe_interval(drift_rate=0.05, budget_minutes=10)
    >>> 
    >>> # 3. Decode with adaptive priors
    >>> logical = decode_adaptive(syndromes, error_rates, decoder='mwpm')

Reference:
    DAQEC-Benchmark v1.0: https://doi.org/10.5281/zenodo.XXXXXXX
    Paper: [Nature Communications DOI]
"""

__version__ = "1.0.0"
__author__ = "DAQEC Research Team"
__license__ = "MIT"

from .selection import select_qubits_drift_aware
from .policy import recommend_probe_interval
from .decoding import decode_adaptive
from .probes import run_lightweight_probes, ProbeResult

__all__ = [
    "__version__",
    "select_qubits_drift_aware",
    "recommend_probe_interval", 
    "decode_adaptive",
    "run_lightweight_probes",
    "ProbeResult",
]
