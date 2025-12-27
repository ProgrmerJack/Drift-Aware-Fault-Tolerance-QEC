# Independent Reproduction Report Template

**For: DAQEC-Benchmark Independent Verification**

Use this template to document your independent reproduction of the DAQEC results. Upon completion, upload as a supplementary report to Zenodo/OSF/Research Square, linking to the original manuscript DOI.

---

## 1. Reproducer Information

| Field | Value |
|-------|-------|
| **Reproducer Name** | |
| **Affiliation** | |
| **ORCID** | |
| **Date of Reproduction** | |
| **Time Spent** | |

### Independence Declaration

- [ ] I have **no** current or past collaboration with the manuscript author(s)
- [ ] I am **not** affiliated with the same institution as the author(s)
- [ ] I have **no** ongoing research overlap with the author(s)
- [ ] I have **no** financial interest in the outcome of this verification

Signature: _______________________

---

## 2. Environment Setup

### 2.1 System Information

| Component | Value |
|-----------|-------|
| Operating System | |
| Python Version | |
| CPU | |
| RAM | |

### 2.2 Repository Verification

```bash
# Record the exact commands used
git clone https://github.com/ProgrmerJack/Drift-Aware-Fault-Tolerance-QEC
cd Drift-Aware-Fault-Tolerance-QEC
git checkout [tag/commit]
git log -1 --format="%H"  # Record full commit hash
```

**Commit Hash**: `________________________________`

**Tag/Version**: `________________________________`

### 2.3 Dependency Installation

- [ ] Created fresh virtual environment
- [ ] Installed from `requirements.txt` without errors
- [ ] All dependencies resolved

Any installation issues:
```
[Describe any issues encountered]
```

---

## 3. Data Verification

### 3.1 Checksum Verification

Run: `python scripts/verify_checksums.py`

| File | Expected SHA256 | Actual SHA256 | Match? |
|------|-----------------|---------------|--------|
| master.parquet | | | ☐ Yes ☐ No |
| SourceData.xlsx | | | ☐ Yes ☐ No |

### 3.2 Data Integrity

- [ ] All required data files present
- [ ] No file corruption detected
- [ ] Data loads without errors

---

## 4. Figure Reproduction

### 4.1 Figure Generation

Run: `python scripts/reproduce_all_figures.py`

| Figure | Generated? | Visually Matches? | Notes |
|--------|------------|-------------------|-------|
| Figure 1 | ☐ Yes ☐ No | ☐ Yes ☐ No | |
| Figure 2 | ☐ Yes ☐ No | ☐ Yes ☐ No | |
| Figure 3 | ☐ Yes ☐ No | ☐ Yes ☐ No | |
| Figure 4 | ☐ Yes ☐ No | ☐ Yes ☐ No | |
| Figure 5 | ☐ Yes ☐ No | ☐ Yes ☐ No | |

### 4.2 Generation Time

Total time to generate all figures: _______ minutes

---

## 5. Statistical Verification

### 5.1 Key Statistics Comparison

Run: `python scripts/verify_statistics.py`

| Statistic | Published Value | Reproduced Value | Within Tolerance? |
|-----------|-----------------|------------------|-------------------|
| Primary improvement (mean) | 61% | | ☐ Yes ☐ No |
| 95% CI lower bound | 58% | | ☐ Yes ☐ No |
| 95% CI upper bound | 64% | | ☐ Yes ☐ No |
| Tail compression (95th) | 75.7% | | ☐ Yes ☐ No |
| Tail compression (99th) | 77.2% | | ☐ Yes ☐ No |
| Dose-response ρ | 0.56 | | ☐ Yes ☐ No |
| Sample size (sessions) | 756 | | ☐ Yes ☐ No |
| Cluster count | 42 | | ☐ Yes ☐ No |

**Tolerance used**: ±0.01 (default) / ±______ (custom)

### 5.2 Statistical Notes

Any deviations from expected values:
```
[Describe any discrepancies]
```

---

## 6. Additional Verification (Optional)

### 6.1 Code Review

- [ ] Reviewed primary analysis scripts
- [ ] Logic appears correct
- [ ] No obvious bugs or data manipulation

### 6.2 Alternative Analysis

Did you run any alternative analyses not specified in the reproduction protocol?

```
[Describe any additional verification steps]
```

---

## 7. Summary Verdict

### 7.1 Reproduction Outcome

**Overall reproduction status**: 
- [ ] ✅ **SUCCESSFUL** - All claims verified within tolerance
- [ ] ⚠️ **PARTIAL** - Some claims verified, some discrepancies (describe below)
- [ ] ❌ **FAILED** - Major discrepancies or errors

### 7.2 Confidence Assessment

Rate your confidence in the following (1-5, where 5 = highest confidence):

| Aspect | Confidence |
|--------|------------|
| Data integrity | /5 |
| Figure accuracy | /5 |
| Statistical claims | /5 |
| Overall reproducibility | /5 |

### 7.3 Detailed Notes

```
[Any additional observations, concerns, or commendations]
```

---

## 8. Attestation

I attest that:

1. I conducted this reproduction independently, following the provided protocol
2. I have accurately reported my findings without selective omission
3. I have no conflicts of interest that would bias this verification
4. I consent to this report being published alongside the manuscript

**Signature**: _______________________

**Date**: _______________________

---

## Upload Instructions

1. Save this completed report as PDF
2. Create Zenodo upload with:
   - **Type**: Report
   - **Title**: "Independent Reproduction Report: [Manuscript Title]"
   - **Related identifiers**: Link to manuscript DOI (Supplement to)
   - **Keywords**: reproducibility, quantum computing, verification
3. Share DOI with manuscript authors for SI citation

---

## For Authors: Citation Format

When citing this reproduction report in your SI:

```latex
Independent reproduction was verified by [Reproducer Name] 
([Affiliation]). The reproduction report is available at 
Zenodo DOI: [Reproducer's Zenodo DOI].
```
