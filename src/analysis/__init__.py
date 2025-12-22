"""
Analysis Module
===============

Statistical analysis for drift-error correlations and A/B testing.
"""

from .drift_error_analysis import (
    DriftErrorAnalyzer,
    ABTestFramework,
    LogicalErrorRateModel
)
from .visualization import (
    ResultsVisualizer,
    generate_publication_figures
)

__all__ = [
    "DriftErrorAnalyzer",
    "ABTestFramework",
    "LogicalErrorRateModel",
    "ResultsVisualizer",
    "generate_publication_figures"
]
