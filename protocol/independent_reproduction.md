# Independent Reproduction Protocol

## Purpose

This document provides instructions for third-party verification of the DAQEC-Benchmark results. Independent reproduction is the highest-leverage credibility signal for high-impact claims.

## What We Ask Reproducers To Verify

1. **Figure regeneration**: All main-text figures can be reproduced from the deposited dataset
2. **Statistical claims**: Key statistics (effect sizes, p-values, CIs) match within rounding
3. **Checksums**: Data files match published checksums
4. **Environment**: Analysis runs on stated dependencies

## Reproduction Steps

### 1. Environment Setup

```bash
# Clone repository at verified commit
git clone https://github.com/ProgrmerJack/Drift-Aware-Fault-Tolerance-QEC
cd Drift-Aware-Fault-Tolerance-QEC
git checkout v1.0.0  # Tagged release

# Create isolated environment
python -m venv reproduction_env
source reproduction_env/bin/activate  # Linux/Mac
# reproduction_env\Scripts\activate  # Windows

# Install exact dependencies
pip install -r requirements.txt
```

### 2. Download Benchmark Data

```bash
# Auto-downloads from Zenodo if not present
python scripts/download_benchmark.py

# Verify checksums
python scripts/verify_checksums.py
```

Expected output:
```
master.parquet: SHA256 = [CHECKSUM] ✓
SourceData.xlsx: SHA256 = [CHECKSUM] ✓
All checksums verified.
```

### 3. Regenerate All Figures

```bash
python scripts/reproduce_all_figures.py
```

Expected runtime: <5 minutes on standard hardware.

Output location: `results/figures/`

### 4. Verify Key Statistics

```bash
python scripts/verify_statistics.py
```

This script outputs:
- Primary effect size (Cohen's d)
- 95% cluster-bootstrap CI
- Permutation test p-value
- Dose-response correlation (Spearman ρ)
- Tail compression percentiles (95th, 99th)

### 5. Record Verification

Please record:
- [ ] Git commit hash verified
- [ ] All checksums match
- [ ] Figures 1-5 regenerated successfully
- [ ] Key statistics within ±0.01 of reported values
- [ ] Any deviations (describe below)

## Reproducer Attestation Template

```
INDEPENDENT REPRODUCTION ATTESTATION

Reproducer: [Name, Affiliation]
Date: [YYYY-MM-DD]
Commit: [hash]

I confirm that I have no collaborative relationship with the manuscript authors
(no joint publications, no shared institutional affiliation, no ongoing research overlap).

Verification results:
- Environment setup: [SUCCESS/FAIL]
- Checksum verification: [SUCCESS/FAIL]
- Figure regeneration: [SUCCESS/FAIL]
- Statistical verification: [SUCCESS/FAIL]

Deviations observed: [None / Describe]

Signed: _______________
```

## What This Does NOT Verify

- Correctness of the experimental protocol (requires QPU access)
- Validity of the probe circuit design (requires domain expertise)
- Generality to other hardware platforms (requires additional experiments)

This reproduction verifies only that the **analysis pipeline** produces the **claimed results** from the **deposited data**.

## Contact

For questions about reproduction, contact: [email]

---

## For Inclusion in SI

Upon successful independent reproduction, the following statement may be added to the Supplementary Information:

> **Independent Reproducibility.** The analysis pipeline was independently verified by [N] external researchers with no collaborative relationship to the authors. Reproducers confirmed that: (i) all data checksums match deposited values, (ii) all main-text figures regenerate from the one-command pipeline, and (iii) key statistics (Cohen's d, bootstrap CIs, permutation p-values) match within rounding tolerance. Attestations are available in the repository.
