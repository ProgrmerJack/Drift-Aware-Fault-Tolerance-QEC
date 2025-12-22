# Reproducibility Card

## Quick Reproduction Guide

### Prerequisites
- Python 3.10+
- pip install numpy pandas scipy matplotlib seaborn pyarrow openpyxl

### Reproduce All Figures

```bash
# Generate simulated data (if master.parquet not present)
python scripts/generate_simulation_data.py

# Run Nature-tier statistical analysis
python analysis/nature_tier_stats.py

# Generate all manuscript figures
python scripts/generate_all_figures.py

# Generate mechanism figures (Fig 6-8)
python scripts/generate_mechanism_figure.py
```

### Expected Output

| Step | Output File | Runtime |
|------|-------------|---------|
| Data generation | data/processed/master.parquet | ~2 min |
| Nature-tier analysis | analysis/nature_tier_manifest.json | ~30 sec |
| Main figures (1-5) | manuscript/figures/fig*.pdf | ~1 min |
| Mechanism figures (6-8) | manuscript/figures/fig6-8*.pdf | ~30 sec |

### Key Results to Verify

1. **Primary Effect**: 61.5% relative reduction in logical error rate
2. **Statistical Unit**: n = 42 sessions (not 1,512 shots)
3. **Effect Size**: Cohen's d = 4.70 (large)
4. **Drift-Benefit Correlation**: r = 0.64 (PASS negative control)
5. **All Holdouts Generalize**: Temporal and all 3 backends

### File Checksums (for verification)

```bash
# Verify data integrity
sha256sum data/processed/master.parquet
# Expected: [hash will be computed on your system]

# Verify protocol integrity  
sha256sum protocol/protocol.yaml
# Expected: Matches hash in manuscript Methods
```

### IBM Quantum Hardware Access

For real-data validation (Phase 2):
1. Obtain IBM Quantum account
2. Set IBM_QUANTUM_TOKEN environment variable
3. Run `python scripts/run_real_experiments.py` (not included in this release)

### Environment Lock

```bash
# Create exact environment
pip install -r requirements.txt

# Or use conda
conda env create -f environment.yml
```

### Contact for Reproducibility Issues

If you encounter issues reproducing results:
1. Open GitHub issue with error log
2. Include Python version and OS
3. Include pip freeze output

### Random Seed

All stochastic processes use seed = 42 for reproducibility.
Set via: `np.random.seed(42)` and `random.seed(42)`

---

*This reproducibility card follows Nature Communications guidelines for computational reproducibility.*
