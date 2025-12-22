# Code Availability Statement

> This statement will be included verbatim in the manuscript.
> Update URLs after repository is made public and archived.

---

## Code Availability

All code supporting this study is publicly available under open-source licenses.

### Primary Repository

**GitHub Repository**: https://github.com/[USER]/Drift-Aware-Fault-Tolerance-QEC

**Archived Version (Zenodo)**: DOI [10.5281/zenodo.XXXXXXX] (placeholder - update after deposit)

The archived version corresponds exactly to the code used to generate all results in this manuscript.

### Repository Contents

```
Drift-Aware-Fault-Tolerance-QEC/
├── src/                    # Source modules
│   ├── calibration.py      # Device characterization
│   ├── probes.py           # Lightweight probe circuits
│   ├── qec.py              # QEC circuit construction and execution
│   ├── analysis.py         # Data analysis and figure generation
│   └── utils.py            # Utility functions
├── protocol/               # Pre-registered protocol
│   ├── protocol.yaml       # Locked experimental parameters
│   └── run_protocol.py     # Executable protocol script
├── analysis/               # Statistical analysis
│   └── stats_plan.py       # Pre-registered statistical plan
├── notebooks/              # Analysis notebooks
│   ├── phase0_understand_drift.ipynb
│   ├── phase1_design_probes.ipynb
│   ├── phase2_syndrome_pipeline.ipynb
│   ├── phase3_qec_benchmark.ipynb
│   └── phase4_write_up.ipynb
├── tests/                  # Unit tests
└── config/                 # Configuration files
```

### Software Dependencies

The code requires Python 3.10+ and the following key packages:
- qiskit >= 1.0.0
- qiskit-ibm-runtime >= 0.20.0
- qiskit-aer >= 0.13.0
- numpy >= 1.24.0
- pandas >= 2.0.0
- scipy >= 1.11.0
- matplotlib >= 3.7.0

A complete `requirements.txt` with pinned versions is included.

### Installation

```bash
git clone https://github.com/[USER]/Drift-Aware-Fault-Tolerance-QEC.git
cd Drift-Aware-Fault-Tolerance-QEC
pip install -r requirements.txt
python -c "import qiskit; print(qiskit.__version__)"  # Verify installation
```

### License

The code is released under the **MIT License**, permitting unrestricted use, modification, and distribution with attribution.

### Reproducibility

To reproduce all figures from deposited data:

```bash
# Ensure data is downloaded (see Data Availability)
python protocol/run_protocol.py --mode=figures
```

To rerun complete analysis including statistical tests:

```bash
python protocol/run_protocol.py --mode=analysis
```

To rerun data collection (requires IBM Quantum access and consumes QPU allocation):

```bash
export IBMQ_TOKEN="your_token"
python protocol/run_protocol.py --mode=collect
```

### Testing

```bash
pytest tests/ -v
```

### Third-Party Code

This project uses the following third-party libraries under their respective licenses:
- **Qiskit** (Apache 2.0): Quantum circuit construction and execution
- **PyMatching** (Apache 2.0): Minimum-weight perfect matching decoder
- **NumPy/SciPy** (BSD): Numerical computing
- **Pandas** (BSD): Data manipulation
- **Matplotlib** (PSF): Visualization

No proprietary or restricted code is required.

---

## Version Control

| Version | Date | Git Commit | Zenodo DOI |
|---------|------|------------|------------|
| 1.0 | [DATE] | [HASH] | [DOI] |

---

## Contact

For code-related questions, open an issue on GitHub or contact [EMAIL].
