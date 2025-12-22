# Nature Research Reporting Summary

> **Instructions**: This document provides responses for the Nature Research Reporting Summary.
> The actual PDF form must be downloaded from Nature's website and completed there.
> This markdown serves as a draft/reference document.
>
> Form URL: https://www.nature.com/documents/nr-reporting-summary.pdf

---

## Study Design

### 1. Describe the study design

**Type**: Observational study with pre-registered protocol and statistical analysis plan.

**Design**: Paired comparison design. Each experimental session generates paired observations: (1) baseline QEC performance using vendor calibration data, and (2) drift-aware QEC performance using real-time probe characterization. Sessions were conducted across multiple days and backends to capture natural drift variation.

### 2. Sample size determination

**Actual sample size**: 126 paired sessions across 42 day×backend clusters (14 days × 3 backends: ibm_brisbane, ibm_kyoto, ibm_osaka). Each day had 3 time-stratified sessions (0-8h, 8-16h, 16-24h post-calibration) to capture drift evolution.

**Statistical power**: The achieved sample (n=126 pairs) provides power >0.999 to detect the observed effect (Cohen's d = 3.82) at α=0.05. Even at α=0.01 with Bonferroni correction for 5 distances, power remains >0.99.

**IBM Fez hardware validation**: Separate validation on 156-qubit Heron r2 processor with surface code d=3 (6 runs) and deployment study (N=2 sessions per condition). This is underpowered for statistical inference but provides functional validation that the probe pipeline executes successfully on production hardware.

### 3. Data inclusion/exclusion criteria

**Pre-registered exclusion criteria** (applied automatically before analysis):
1. Backend status = "maintenance" at session start
2. Time since last calibration > 24 hours
3. Mean T₁ across all qubits < 50 μs
4. Any data collection error or timeout

**Post-hoc exclusions**: None. All sessions meeting pre-registered criteria were analyzed.

### 4. Randomization

**Not applicable** for this observational study. Session timing was determined by backend availability and researcher schedule, not randomized. However, the paired design controls for session-level confounds.

### 5. Blinding

**Partial blinding**: Statistical analysis code was written before unblinding to primary endpoint. Figure generation scripts were finalized before seeing drift-aware vs. baseline results. However, data collection itself could not be blinded as the experimenter needed to verify probe execution.

---

## Statistical Analysis

### 6. Statistical tests used

**Primary endpoint** (session-level logical error rate reduction):
- Cluster-robust bootstrap 95% CI (10,000 resamples with replacement at session level)
- Cluster-robust permutation test (10,000 permutations with stratification by day×backend cluster)
- Paired t-test with cluster-robust standard errors

**Effect size metrics**:
- Cohen's d = 3.82 (standardized mean difference)
- Cliff's δ = 1.00 (nonparametric effect size, all 126 sessions favor drift-aware)
- Hodges-Lehmann median difference with 95% CI
- Relative risk reduction (RRR) = 59.9%

**Secondary analyses** (Holm-Bonferroni corrected):
- Wilcoxon signed-rank test for burst frequency reduction
- Paired t-test for tail risk metrics (P₉₀, P₉₉)
- Kendall τ correlation for ranking stability
- Spearman ρ for drift-benefit dose-response

**Holdout validation**:
- Temporal: Final 3 days withheld
- Backend: Leave-one-backend-out cross-validation
- Distance: Separate analysis per code distance d=3,5,7,9,11

### 7. Assumptions tested

- **Normality**: Shapiro-Wilk test on paired differences (reported in SI)
- **Independence**: Sessions separated by ≥4 hours to minimize autocorrelation
- **Exchangeability** (for permutation test): Assessed via time-series analysis in SI

### 8. Effect sizes and confidence intervals

All reported effects include:
- Point estimate of effect size
- 95% confidence interval (bootstrap, BCa method)
- Cohen's d or equivalent standardized effect size
- Exact p-values (not just significance thresholds)

---

## Data Availability

### 9. Data availability statement

All data are deposited at Zenodo (DOI: 10.5281/zenodo.17881116) under CC BY 4.0 license. See Data Availability section in manuscript. The deposit includes: master.parquet (756 experiments, 126 paired sessions), IBM Fez hardware validation results, syndrome statistics, drift characterization, effect sizes by condition, and SourceData.xlsx with all figure/table data.

### 10. Unique identifiers

- **Zenodo DOI**: 10.5281/zenodo.17881116
- **Repository URL**: https://github.com/ProgrmerJack/Drift-Aware-Fault-Tolerance-QEC

---

## Code Availability

### 11. Code availability statement

All code is available at https://github.com/ProgrmerJack/Drift-Aware-Fault-Tolerance-QEC under MIT license and archived at Zenodo (DOI: 10.5281/zenodo.17881116). See Code Availability section in manuscript. The repository includes drop-in API functions (select_qubits_drift_aware, recommend_probe_interval, decode_adaptive) enabling immediate adoption.

### 12. Custom algorithms

All algorithms used are either:
- Standard implementations from published libraries (Qiskit, PyMatching, SciPy)
- Novel contributions described in Methods and fully implemented in the repository

---

## Materials & Resources

### 13. Biological materials

Not applicable (quantum computing study).

### 14. Chemical compounds

Not applicable.

### 15. Antibodies

Not applicable.

### 16. Cell lines

Not applicable.

### 17. Organisms/strains

Not applicable.

### 18. Human research participants

Not applicable.

### 19. Clinical data

Not applicable.

### 20. Dual use research

Not applicable.

---

## Field-Specific Reporting

### Physics/Quantum Computing

**Hardware used**: 
- **Main experiments**: IBM Quantum cloud-accessible superconducting processors (ibm_brisbane, ibm_kyoto, ibm_osaka - all 127-qubit Eagle r3 generation) accessed via IBM Quantum Platform using Open Plan allocation. Execution period: 14 consecutive days with 3 time-stratified sessions per day×backend.
- **Hardware validation**: ibm_fez (156-qubit Heron r2 processor with tunable couplers and native two-qubit gate set) for surface code d=3 experiments and deployment study.

**Code distances**: Repetition codes d=3, 5, 7, 9, 11 (12 qubits per instance with physical ancilla parity checks). Surface code d=3 (17 qubits, 3 syndrome rounds) on IBM Fez.

**Measurement protocol**: 1,024 shots per repetition code circuit; 4,096 shots per surface code circuit; 2,048 shots per deployment study experiment.

**Software versions**: 
- Python 3.10.12
- Qiskit 1.0.0
- qiskit-ibm-runtime 0.20.0
- Pymatching 2.2.0 (minimum-weight perfect matching decoder)
- NumPy 1.24.0
- SciPy 1.11.0
- Pandas 2.0.3
- Complete dependency list in requirements.txt with pinned versions

**Reproducibility**: All random seeds are fixed and recorded. Complete provenance tracking is implemented in the protocol execution script.

---

## Checklist Completion Notes

| Section | Status | Notes |
|---------|--------|-------|
| Study design | ✓ | Pre-registered protocol |
| Sample size | ✓ | Power analysis documented |
| Exclusion criteria | ✓ | Pre-registered, applied automatically |
| Randomization | N/A | Observational study |
| Blinding | Partial | Analysis code written pre-unblinding |
| Statistical tests | ✓ | Documented in stats_plan.py |
| Effect sizes | ✓ | CIs for all effects |
| Data availability | ✓ | Zenodo deposit |
| Code availability | ✓ | GitHub + Zenodo archive |

---

## Form Submission Notes

Before submitting the official PDF form:
1. Download latest version from Nature website
2. Transfer all responses from this draft
3. Verify all DOIs are assigned and active
4. Have co-authors review responses
5. Export completed PDF form
