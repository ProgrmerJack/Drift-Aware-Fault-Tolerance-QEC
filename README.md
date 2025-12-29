# Drift-Aware Fault-Tolerant Quantum Error Correction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.0+-blueviolet.svg)](https://qiskit.org/)

A research framework for studying the impact of calibration drift on quantum error correction fault-tolerance thresholds, with novel drift-aware qubit selection and adaptive-prior syndrome decoding.

## Abstract

This project investigates how temporal drift in qubit calibration parameters (T1, T2, gate fidelities, readout errors) affects quantum error correction performance on IBM Quantum hardware. We demonstrate that:

1. **Calibration drift** significantly impacts logical error rates beyond what daily calibration data predicts
2. **Real-time probe refresh** (30-shot lightweight diagnostics) improves qubit selection fidelity
3. **Drift-aware selection** using predictive stability scoring outperforms static and real-time strategies
4. **Adaptive-prior decoding** leveraging drift information further reduces logical error rates

## Project Structure

```
Drift-Aware-Fault-Tolerance-QEC/
├── src/
│   ├── calibration/          # Module A: Drift data collection
│   │   ├── drift_collector.py
│   │   └── __init__.py
│   ├── probes/               # Module B: Lightweight diagnostics
│   │   ├── probe_suite.py
│   │   ├── qubit_selector.py
│   │   └── __init__.py
│   ├── qec/                  # Module C: QEC implementation
│   │   ├── repetition_code.py
│   │   ├── experiment_runner.py
│   │   └── __init__.py
│   ├── analysis/             # Analysis & visualization
│   │   ├── drift_error_analysis.py
│   │   ├── visualization.py
│   │   └── __init__.py
│   ├── utils/                # Utilities
│   │   ├── job_management.py
│   │   ├── data_management.py
│   │   └── __init__.py
│   └── __init__.py
├── notebooks/
│   ├── phase0_infrastructure.ipynb  # Setup & data lake
│   ├── phase1_baseline.ipynb        # Static selection baseline
│   ├── phase2_realtime.ipynb        # RT probe experiments
│   ├── phase3_adaptive.ipynb        # Drift-aware comparison
│   └── phase4_analysis.ipynb        # Statistical analysis
├── tests/
│   ├── test_calibration.py
│   ├── test_probes.py
│   ├── test_qec.py
│   └── test_analysis.py
├── data/
│   ├── calibration/          # Calibration snapshots
│   ├── experiments/          # Experiment results
│   ├── figures/              # Publication figures
│   └── tables/               # Summary tables
├── config/
│   └── experiment_config.yaml
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Installation

### Prerequisites

- Python 3.10 or higher
- IBM Quantum account (Open Plan provides ~10 min QPU time per 28-day window)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Drift-Aware-Fault-Tolerance-QEC.git
cd Drift-Aware-Fault-Tolerance-QEC
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure IBM Quantum credentials:
```python
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(channel="ibm_quantum", token="YOUR_TOKEN")
```

## Quick Start

### Run the Research Pipeline

Execute notebooks in order:

1. **Phase 0**: Initialize infrastructure and data lake
2. **Phase 1**: Establish baseline error rates with static qubit selection
3. **Phase 2**: Compare real-time vs static selection under drift
4. **Phase 3**: Evaluate drift-aware selection and adaptive decoding
5. **Phase 4**: Generate statistical analysis and publication figures

### Example Usage

```python
from src.calibration import DriftCollector
from src.probes import ProbeSuite, QubitSelector
from src.qec import RepetitionCode, QECExperimentRunner

# Connect to IBM Quantum
from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService(channel="ibm_quantum")
backend = service.least_busy(simulator=False, operational=True, min_num_qubits=27)

# Collect calibration data
drift_collector = DriftCollector(backend)
calibration = drift_collector.collect_calibration_snapshot()

# Run lightweight probes
probe_suite = ProbeSuite(backend=backend, shots_per_probe=30)
probe_results = probe_suite.run_probes(qubits=[0, 1, 2, 3, 4])

# Select qubits using drift-aware strategy
selector = QubitSelector(backend, strategy='drift_aware')
selected = selector.select_qubits(n_qubits=9, probe_data=probe_results)

# Build and run repetition code
rep_code = RepetitionCode(distance=5, qubits=selected['qubits'])
circuit = rep_code.build_circuit(n_syndrome_rounds=3, initial_state='0')

runner = QECExperimentRunner(backend=backend)
results = runner.run_experiment(circuits=[circuit], shots=1000)
```

## Key Components

### DriftCollector
Collects and stores calibration snapshots from IBM Quantum backends, tracking T1, T2, readout errors, and gate fidelities over time.

### ProbeSuite
Implements lightweight 30-shot diagnostic probes for real-time qubit characterization:
- T1 probe (single delay time)
- Readout error probe (|0⟩ and |1⟩ preparation)
- RB probe (2-3 Clifford sequences)

### QubitSelector
Three-tier qubit selection strategies:
- **Static**: Uses daily calibration data
- **Real-time**: Incorporates probe measurements
- **Drift-aware**: Predicts stability using Holt's exponential smoothing

### RepetitionCode
Distance-3/5/7 repetition code implementation with:
- Dynamic circuit support (mid-circuit measurement)
- SamplerV2 primitive compatibility
- Syndrome extraction and decoding

### AdaptivePriorDecoder
Syndrome decoder with drift-informed error priors that update based on real-time probe measurements.

## QPU Budget Management

The IBM Quantum Open Plan provides approximately 10 minutes of QPU time per 28-day rolling window. The `QPUBudgetTracker` class helps manage this constraint:

```python
from src.utils import QPUBudgetTracker

tracker = QPUBudgetTracker(total_budget_seconds=600)
print(f"Remaining budget: {tracker.remaining_budget()}s")
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{drift_aware_qec_2025,
  title={Drift-Aware Fault-Tolerant Quantum Error Correction},
  author={Ashuraliyev, Abduxoliq},
  journal={arXiv preprint},
  year={2025},
  note={ORCID: 0009-0003-5482-5526}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Abduxoliq Ashuraliyev**  
Email: Jack00040008@outlook.com  
ORCID: [0009-0003-5482-5526](https://orcid.org/0009-0003-5482-5526)

## Acknowledgments

- IBM Quantum for providing access to quantum hardware
- Amazon Braket, IQM Cloud, and Rigetti for multi-platform validation access
- Qiskit development team for the quantum computing framework
