"""
Module C: QEC Bench
==================

Repetition code experiments using dynamic circuits.
"""

from .repetition_code import (
    RepetitionCode,
    SyndromeDecoder,
    create_experiment_batch
)
from .experiment_runner import (
    QECExperimentRunner,
    quick_qec_demo
)
from .syndrome_extractor import (
    SyndromeData,
    SyndromeExtractor,
)

__all__ = [
    "RepetitionCode",
    "SyndromeDecoder",
    "create_experiment_batch",
    "QECExperimentRunner",
    "quick_qec_demo",
    "SyndromeData",
    "SyndromeExtractor",
]
