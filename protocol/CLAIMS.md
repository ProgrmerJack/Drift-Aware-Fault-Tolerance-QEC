# CLAIMS.md - Pre-Registered Scientific Claims
# =============================================================================
# LOCKED: This file defines all testable claims for the manuscript.
# Once locked, NO modifications without creating protocol amendment.
# Protocol Version: 1.0
# Lock Date: 2025-01-XX (fill before data collection)
# =============================================================================

## Primary Claim

**Claim P1**: Drift-aware qubit selection combined with adaptive-prior decoding
achieves a statistically significant reduction in logical error rate compared
to static baseline selection.

| Parameter | Value |
|-----------|-------|
| Comparison | `drift_aware_full_stack` vs `baseline_static` |
| Test | Paired bootstrap (n=10,000) |
| Hypothesis | Two-sided |
| α (Type I error) | 0.05 |
| Power target | 0.80 |
| **Minimum detectable effect** | 20% relative reduction |
| **Expected effect size** | 25-40% relative reduction |
| Cohen's d threshold | ≥0.5 (medium effect) |

### Defensible Thresholds

To claim support for P1, we require:

1. **p-value < 0.05** on primary endpoint (paired log-odds ratio)
2. **95% CI excludes zero** for relative risk reduction
3. **Effect consistent across ≥2 backends** (direction only, not magnitude)
4. **Minimum sample size met**: ≥3 sessions per backend, ≥3 calibration days

---

## Secondary Claims

### Claim S1: Drift Invalidates Static Selection

**Statement**: Optimal qubit subsets change materially within a single calibration
day, invalidating once-per-day static selection strategies.

| Metric | Threshold |
|--------|-----------|
| Jaccard similarity (morning vs evening) | < 0.7 |
| Rank correlation (Spearman) | ρ < 0.8 |
| At least N backends showing effect | 2 of 3 |

### Claim S2: Syndrome Non-IID Structure

**Statement**: Syndrome measurement streams exhibit statistically significant
deviations from iid Bernoulli process, consistent with correlated error events.

| Test | Null Hypothesis | Rejection at |
|------|-----------------|--------------|
| Runs test (Wald-Wolfowitz) | iid sequence | p < 0.01 |
| Autocorrelation (lag-1) | ACF(1) = 0 | |ACF(1)| > 2/√n |
| Fano factor | F = 1 (Poisson) | F > 1.5 |

### Claim S3: Lightweight Probes Suffice

**Statement**: 30-shot probe circuits provide sufficient signal-to-noise for
qubit ranking, requiring ≤5% of QPU budget while maintaining ranking accuracy.

| Metric | Threshold |
|--------|-----------|
| Probe budget fraction | ≤5% of total |
| Ranking accuracy vs 1000-shot | Spearman ρ > 0.85 |
| Logical error rate degradation | <5% vs full characterization |

### Claim S4: Distance Scaling

**Statement**: Logical error rate decreases with code distance for drift-aware
strategy, demonstrating error suppression scaling.

| Distance | Expected Logical Error Rate |
|----------|----------------------------|
| d=3 | Baseline (measured) |
| d=5 | < d=3 rate × 0.7 |
| d=7 | < d=5 rate × 0.7 |

Test: Linear regression of log(p_L) vs distance should have negative slope
with p < 0.05.

---

## Null Results We Will Report

Even if not statistically significant, we will report:

1. **Day-to-day variability**: Variation in effect size across calibration days
2. **Backend heterogeneity**: Differences in baseline error rates across backends
3. **Correlated failure modes**: Frequency of multi-qubit failure events
4. **Probe failure rate**: How often 30-shot probes fail to converge

---

## Stopping Rules

### Early Stopping for Success
If after 3 backends × 3 days, primary endpoint achieves p < 0.01 with effect
size >30%, we may conclude data collection (but will report final sample size).

### Early Stopping for Futility
If after 3 backends × 5 days:
- Effect size <10%, OR
- Direction inconsistent across backends, OR
- 95% CI includes both +20% and -20%

We will conclude null result and report accordingly.

### Never Stop Early Due to p-Hacking Pressure
We will NOT:
- Add more data to achieve significance
- Remove "outlier" backends post-hoc
- Change primary endpoint definition

---

## Effect Size Reporting

All results will include:

| Measure | Definition |
|---------|------------|
| Relative Risk Reduction | (p_baseline - p_drift_aware) / p_baseline |
| Absolute Risk Reduction | p_baseline - p_drift_aware |
| Number Needed to Treat | 1 / ARR |
| Cohen's d | (μ₁ - μ₂) / σ_pooled |
| Log Odds Ratio | log(OR) with bootstrap CI |

---

## Protocol Integrity Hash

This file, combined with `protocol.yaml`, defines the complete experimental
protocol. Before data collection begins:

1. Compute SHA-256 hash of this file
2. Compute SHA-256 hash of `protocol.yaml`
3. Record combined hash in git commit
4. Tag as `protocol_v1_locked`

```
CLAIMS.md SHA-256: [TO BE COMPUTED AT LOCK TIME]
protocol.yaml SHA-256: [TO BE COMPUTED AT LOCK TIME]
Combined manifest hash: [TO BE COMPUTED AT LOCK TIME]
```

---

## Amendment Procedure

If protocol changes are required after lock:

1. Create `CLAIMS_AMENDMENT_1.md` documenting:
   - What changed and why
   - Scientific justification
   - Impact on pre-registered claims
2. Do NOT modify this file
3. Report all amendments in manuscript Methods section
4. All amendments are presumed exploratory, not confirmatory

---

## Claim Mapping to Figures

| Claim | Primary Figure | Supporting Data |
|-------|---------------|-----------------|
| P1 | Figure 3 (main result) | Table 1, Extended Data Table 1 |
| S1 | Figure 2 (drift heatmap) | Extended Data Fig 1 |
| S2 | Figure 4 (syndrome analysis) | SI Section 3 |
| S3 | Figure 5 (probe efficiency) | SI Section 4 |
| S4 | Figure 3b (distance scaling) | Extended Data Table 2 |

---

## Data Availability Commitment

Upon publication, we will deposit:

1. **Raw data**: All syndrome measurement bitstrings
2. **Processed data**: Master Parquet file per protocol schema
3. **Code**: Complete analysis pipeline with locked dependencies
4. **Protocol**: This file and protocol.yaml, hash-verified

Repository: GitHub (public on acceptance)
Archive: Zenodo (DOI assigned)

---

*This document follows SPIRIT 2013 guidelines adapted for physics experiments.*
