# Contributing to Drift-Aware Fault-Tolerance QEC

Thank you for your interest in contributing to this research project!

## Project Overview

This project investigates how qubit calibration drift affects quantum error correction fault-tolerance thresholds using IBM Quantum hardware.

## Development Setup

### Prerequisites

- Python 3.10+
- IBM Quantum account with API token
- Git

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Drift-Aware-Fault-Tolerance-QEC.git
   cd Drift-Aware-Fault-Tolerance-QEC
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. Configure IBM Quantum credentials:
   ```bash
   python -c "from qiskit_ibm_runtime import QiskitRuntimeService; QiskitRuntimeService.save_account(channel='ibm_quantum', token='YOUR_TOKEN')"
   ```

## Code Standards

### Style Guide

- Follow PEP 8 for Python code
- Use type hints for all function signatures
- Maximum line length: 88 characters (Black default)
- Use docstrings for all public functions and classes

### Formatting

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Type checking
mypy src/
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_calibration.py -v

# Skip slow tests
pytest -m "not slow"
```

## Project Structure

```
├── src/                    # Source code modules
│   ├── calibration/       # Drift data collection
│   ├── probes/            # Lightweight diagnostic circuits
│   ├── qec/               # QEC benchmark implementations
│   ├── analysis/          # Statistical analysis
│   └── utils/             # Utilities (budget, data management)
├── notebooks/             # Jupyter notebooks for experiments
├── tests/                 # Test suite
├── config/                # Configuration files
└── data/                  # Data storage (gitignored)
```

## Workflow

### Branching Strategy

- `main`: Stable, tested code
- `develop`: Integration branch
- `feature/*`: New features
- `fix/*`: Bug fixes
- `experiment/*`: Experimental code

### Commit Messages

Follow conventional commits:

```
feat: add adaptive qubit selection algorithm
fix: correct T1 drift threshold calculation
docs: update README with new installation steps
test: add unit tests for probe suite
refactor: simplify budget tracking logic
```

### Pull Request Process

1. Create a feature branch from `develop`
2. Make your changes with tests
3. Ensure all tests pass locally
4. Update documentation if needed
5. Submit PR with clear description
6. Address review feedback

## QPU Budget Considerations

**IMPORTANT**: This project operates under IBM Quantum Open Plan constraints (~10 minutes QPU time per 28-day rolling window).

- Always use the `QPUBudgetTracker` for QPU jobs
- Test with simulators first
- Batch circuits when possible
- Use 30-shot probes instead of full characterization

## Experiment Guidelines

### Running Experiments

1. Check remaining QPU budget before starting
2. Run Phase 0 infrastructure setup first
3. Follow notebook sequence: Phase 0 → 1 → 2 → 3 → 4
4. Save all intermediate results to Parquet

### Data Management

- Store raw data in `data/raw/`
- Processed data goes in `data/processed/`
- Use timestamped filenames for versioning
- Document data schemas in docstrings

## Research Ethics

- Cite IBM Quantum appropriately
- Follow data sharing guidelines
- Document experimental conditions thoroughly
- Report negative results honestly

## Questions?

Open an issue for:
- Bug reports
- Feature requests
- Questions about methodology
- Suggestions for improvement

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
