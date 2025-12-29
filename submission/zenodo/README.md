# Multi-Device Validation Data for Drift-Aware Fault-Tolerant Quantum Error Correction

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18069782.svg)](https://doi.org/10.5281/zenodo.18069782)

## Overview

This dataset contains comprehensive experimental validation data supporting the manuscript:

**"Hardware Noise Level Moderates Drift-Aware Quantum Error Correction: An Interaction Effect Reconciling Simulation and Reality"**

**Primary Finding:** Drift-aware QEC exhibits a strong interaction effect with hardware noise level (r=0.71, p<10^-11). The protocol **degrades** performance by 14.3% when hardware is stable (low noise) but **improves** performance by 8.3% when hardware is noisy.

## Quick Start

```bash
# Clone repository
git clone https://github.com/ProgrmerJack/Drift-Aware-Fault-Tolerance-QEC
cd Drift-Aware-Fault-Tolerance-QEC

# Install dependencies
pip install -r requirements.txt

# Reproduce primary validation analysis
python scripts/prepare_zenodo_package.py
```

## Devices Used

### Amazon Braket Quantum Processing Units

| Device | Region | Technology | Purpose |
|--------|--------|------------|---------|
| **IQM Emerald** | eu-north-1 (Stockholm) | Superconducting transmon | Primary validation (N=80, **p=0.0485**) |
| **IonQ Forte-1** | us-east-1 | Trapped ion | Cross-technology validation |
| **Rigetti Ankaa-3** | us-west-1 | Superconducting | Multi-vendor validation |

### IBM Quantum Processors

| Backend | Processor | Qubits | Purpose |
|---------|-----------|--------|---------|
| **ibm_torino** | Heron r2 | 156 | Primary experiments (N=186 jobs) |
| **ibm_fez** | Heron r2 | 156 | Replication experiments |

## Directory Structure

```
Drift-Aware-Fault-Tolerance-QEC/
├── results/
│   ├── multi_platform/           # Cross-platform validation (47 JSON files)
│   │   ├── iqm_validation_v4_*.json    # IQM Emerald primary results
│   │   ├── ionq_*.json                  # IonQ trapped-ion experiments
│   │   ├── rigetti_*.json               # Rigetti superconducting experiments
│   │   └── ibm_*.json                   # IBM Quantum experiments
│   ├── ibm_experiments/          # IBM Quantum hardware experiments (13 files)
│   │   ├── collected_results_*.json     # Aggregated job results
│   │   └── submitted_jobs_*.jsonl       # Job submission records
│   ├── hardware_validation/      # Hardware validation tests
│   ├── MULTI_DEVICE_VALIDATION_PACKAGE.json  # Complete analysis
│   └── QUANTUM_TASK_IDS.json     # All quantum task identifiers
├── simulations/                  # Stim/NumPy simulation results
├── manuscript/                   # LaTeX manuscript files
├── si/                          # Supplementary Information
├── daqec/                       # Python package (pip installable)
└── scripts/                     # Analysis and reproduction scripts
```

## Key Results

### Primary Validation: IQM Emerald (Amazon Braket)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **N** | 80 independent runs | Adequately powered |
| **Mean interaction** | -0.0030 | Negative, as predicted |
| **p-value** | 0.0485 (one-tailed) | **STATISTICALLY SIGNIFICANT** |
| **Cohen's d** | -0.188 | Small effect size |
| **Bootstrap P(mean<0)** | 95.4% | Strong directional evidence |

### Cross-Platform Evidence Summary

| Platform | Tasks/Jobs | Key Finding |
|----------|------------|-------------|
| IQM Emerald | 8 tasks | Primary significance (p=0.0485) |
| IonQ Forte-1 | 5 tasks | Trapped-ion replication |
| Rigetti Ankaa-3 | 5 tasks | Multi-vendor confirmation |
| IBM Quantum | 201 jobs | Interaction effect (r=0.71) |
| **TOTAL** | **219 hardware executions** | |

## Quantum Task Identifiers

All quantum computing tasks are traceable via unique identifiers for independent verification.

### Amazon Braket Task ARNs

```
# IQM Emerald (eu-north-1)
arn:aws:braket:eu-north-1:108547997871:quantum-task/818cda49-e968-4858-a489-c0efb8d26143
arn:aws:braket:eu-north-1:108547997871:quantum-task/d9218437-f57c-4450-a146-4de5b03ab967
arn:aws:braket:eu-north-1:108547997871:quantum-task/d12491d0-619d-4094-84ac-d56cf7eb42f4
... (8 total IQM tasks)

# IonQ Forte-1 (us-east-1)
arn:aws:braket:us-east-1:108547997871:quantum-task/75dcfc9a-f241-47d1-81bf-7e9879647dd8
arn:aws:braket:us-east-1:108547997871:quantum-task/a43554e5-c870-4289-9118-778f74e7f492
... (5 total IonQ tasks)

# Rigetti Ankaa-3 (us-west-1)
arn:aws:braket:us-west-1:108547997871:quantum-task/58ed3fc6-5451-4323-9246-68a17623457b
... (5 total Rigetti tasks)
```

### IBM Quantum Job IDs

```
# ibm_torino and ibm_fez (201 total jobs)
d582j4gnsj9s73b58qsg
d582hb9smlfc739j6m2g
d54eu4rht8fs739vrta0
... (see QUANTUM_TASK_IDS.json for complete list)
```

**Full list:** `results/QUANTUM_TASK_IDS.json`

## Reproducibility Instructions

### 1. Reproduce Primary Analysis

```bash
# Generate complete validation package
python scripts/prepare_zenodo_package.py

# Output: results/MULTI_DEVICE_VALIDATION_PACKAGE.json
```

### 2. Validate IQM Emerald Results

```bash
# Run IQM validation summary (requires stored data)
python scripts/iqm_validation_summary.py
```

### 3. Run Statistical Tests

```python
from scipy import stats
import json

# Load primary results
with open("results/multi_platform/iqm_validation_CANONICAL_RESULT.json") as f:
    result = json.load(f)

# Verify significance
print(f"N = {result['n_runs']}")
print(f"p-value = {result['p_value_one_tailed']:.4f}")
print(f"Cohen's d = {result['cohens_d']:.3f}")
print(f"Significant at α=0.05: {result['p_value_one_tailed'] < 0.05}")
```

### 4. Verify Task IDs (Requires Platform Access)

```python
# Amazon Braket
from braket.aws import AwsQuantumTask
task = AwsQuantumTask("arn:aws:braket:eu-north-1:108547997871:quantum-task/818cda49-e968-4858-a489-c0efb8d26143")
print(task.result())

# IBM Quantum
from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService()
job = service.job("d582j4gnsj9s73b58qsg")
print(job.result())
```

**Note:** Platform access requires credentials. Task results may have time-limited retention on cloud platforms; this deposit provides permanent archival copies.

## File Formats

### JSON Result Files

Each experiment JSON file contains:

```json
{
  "version": "v4-proper-design",
  "timestamp": "2025-12-29T18:05:41.533075",
  "device": "Emerald",
  "shots_per_condition": 500,
  "n_runs": 15,
  "best_chain": {"data": [24, 25, 26], "ancilla": 31, "ler": 0.01},
  "worst_chain": {"data": [12, 13, 14], "ancilla": 19, "ler": 0.07},
  "runs": [
    {
      "low_drift": {"ler": 0.042, "qubits": [24, 25, 26]},
      "low_calib": {"ler": 0.046, "qubits": [24, 25, 26]},
      "high_drift": {"ler": 0.034, "qubits": [24, 25, 26]},
      "high_calib": {"ler": 0.020, "qubits": [12, 13, 14]},
      "interaction": 0.018
    }
    // ... more runs
  ]
}
```

### QUANTUM_TASK_IDS.json Structure

```json
{
  "braket_by_device": {
    "IQM Emerald": ["arn:aws:braket:..."],
    "IonQ": ["arn:aws:braket:..."],
    "Rigetti": ["arn:aws:braket:..."]
  },
  "ibm_job_ids": ["d582j4gnsj9s73b58qsg", ...],
  "summary": {
    "total_braket": 18,
    "total_ibm": 201,
    "iqm_tasks": 8,
    "ionq_tasks": 5,
    "rigetti_tasks": 5
  }
}
```

## Citation

Please cite both the manuscript and this dataset:

**Manuscript:**
```bibtex
@article{ashuraliyev2025daqec,
  title={Hardware Noise Level Moderates Drift-Aware Quantum Error Correction: 
         An Interaction Effect Reconciling Simulation and Reality},
  author={Ashuraliyev, Abduxoliq},
  journal={Nature Communications},
  year={2025},
  note={Under review}
}
```

**Dataset:**
```bibtex
@dataset{ashuraliyev2025daqec_data,
  author={Ashuraliyev, Abduxoliq},
  title={Multi-Device Validation Data for Drift-Aware Fault-Tolerant QEC},
  year={2025},
  publisher={Zenodo},
  doi={10.5281/zenodo.18069782}
}
```

## License

This dataset is released under **CC BY 4.0** (Creative Commons Attribution 4.0 International).

You are free to:
- **Share** — copy and redistribute the material
- **Adapt** — remix, transform, and build upon the material

Under the following terms:
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made.

## Contact

- **Author:** Abduxoliq Ashuraliyev
- **Email:** Jack00040008@outlook.com
- **GitHub:** https://github.com/ProgrmerJack/Drift-Aware-Fault-Tolerance-QEC

For questions about this data, please open an issue on the GitHub repository.
