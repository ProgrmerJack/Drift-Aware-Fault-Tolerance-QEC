# DAQEC: Drift-Aware Quantum Error Correction Toolkit

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight, reference implementation of drift-aware quantum error correction policies described in:

> **Drift-Aware Fault-Tolerance Improves Surface-Code Logical Error Rates on Real Hardware**  
> *Nature Communications* (2025)

## Installation

```bash
# Core functionality (NumPy only)
pip install daqec

# With IBM Quantum hardware support
pip install daqec[hardware]

# With PyMatching decoder
pip install daqec[decoding]

# Full installation
pip install daqec[full]
```

## Quick Start

```python
from daqec import select_qubits_drift_aware, recommend_probe_interval, decode_adaptive

# 1. Select optimal qubit chain from probe results
probe_data = {
    0: {'T1': 150, 'T2': 80, 'readout_error': 0.02, 'gate_error': 0.005},
    1: {'T1': 180, 'T2': 95, 'readout_error': 0.015, 'gate_error': 0.004},
    2: {'T1': 120, 'T2': 60, 'readout_error': 0.03, 'gate_error': 0.008},
    # ... more qubits
}
chain = select_qubits_drift_aware(probe_data, code_distance=5)
print(f"Selected qubits: {chain.qubits}, score: {chain.score:.3f}")

# 2. Set probe cadence based on observed drift
policy = recommend_probe_interval(
    drift_rate_per_hour=0.15,  # 15% T1 change per hour
    budget_minutes=10          # 10 min calibration budget per hour
)
print(f"Optimal interval: {policy.optimal_interval_hours:.2f}h")
print(f"Expected tail reduction: {policy.expected_benefit:.1%}")

# 3. Decode with adaptive priors
import numpy as np
syndromes = np.random.randint(0, 2, (1000, 8))  # Example syndrome data
current_error_rates = {i: 0.01 for i in range(9)}  # Initial error rates
logical_errors = decode_adaptive(syndromes, current_error_rates, decoder='mwpm')
```

## Core API

### `select_qubits_drift_aware(probe_results, code_distance, ...)`

Selects an optimal qubit chain for surface code execution based on current calibration data.

**Parameters:**
- `probe_results`: Dict mapping qubit indices to calibration metrics (T1, T2, readout_error, gate_error)
- `code_distance`: Target surface code distance (3, 5, 7, ...)
- `backend_topology`: Optional connectivity graph
- `weights`: Optional custom scoring weights
- `top_k`: Number of candidate chains to return

**Returns:** `QubitChain` with selected qubits, composite score, and per-qubit metrics.

### `recommend_probe_interval(drift_rate_per_hour, budget_minutes, ...)`

Calculates optimal calibration probe interval using dose-response model.

**Parameters:**
- `drift_rate_per_hour`: Observed calibration drift rate (fraction per hour)
- `budget_minutes`: Available calibration time budget per hour
- `min_interval`: Minimum allowed interval (hours)
- `max_interval`: Maximum allowed interval (hours)

**Returns:** `DriftPolicy` with optimal interval and expected tail-compression benefit.

### `decode_adaptive(syndromes, error_rates, decoder='mwpm', alpha=0.3)`

Runs decoding with Bayesian-updated edge weights from fresh probe data.

**Parameters:**
- `syndromes`: Binary syndrome array (shots × stabilizers)
- `error_rates`: Dict mapping data qubit indices to error probabilities
- `decoder`: Decoder type ('mwpm' or 'uf')
- `alpha`: Bayesian update weight (0 = ignore probes, 1 = full trust)

**Returns:** Binary array of logical error outcomes.

### `run_probe_circuits(backend, qubits, shots=1024)`

Executes lightweight T1/T2/readout probe circuits on hardware.

**Parameters:**
- `backend`: IBM Quantum backend instance
- `qubits`: List of qubit indices to probe
- `shots`: Measurement shots per probe circuit

**Returns:** `ProbeResult` with T1, T2, readout_error, gate_error, timestamp.

## Benchmarks

This toolkit accompanies the benchmark dataset deposited at:
- **Zenodo**: [10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)

The benchmark includes:
- 30 days of calibration snapshots from 3 IBM Quantum backends
- 15,000+ QEC rounds per backend
- Reproducible analysis scripts

## Reproducing Paper Results

```bash
# Clone repository
git clone https://github.com/ProgrmerJack/Drift-Aware-Fault-Tolerance-QEC.git
cd Drift-Aware-Fault-Tolerance-QEC

# Install dependencies
pip install -e ".[full]"

# Run complete analysis
python scripts/run_full_analysis.py --data-dir data/

# Regenerate all figures
python scripts/generate_figures.py --output-dir results/figures/
```

## Citation

```bibtex
@article{daqec2025,
  title={Drift-Aware Fault-Tolerance Improves Surface-Code Logical Error Rates on Real Hardware},
  journal={Nature Communications},
  year={2025},
  doi={10.1038/s41467-XXX-XXXXX-X}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
