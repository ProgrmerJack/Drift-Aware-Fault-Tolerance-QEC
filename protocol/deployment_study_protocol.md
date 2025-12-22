# Deployment Study Protocol: DAQEC in the Wild

## Study Design

A 14-day operational deployment study to demonstrate that the drift-aware pipeline works as a **real operational tool**, not just a post-hoc analysis artifact.

## Objectives

1. Measure how often the probe-triggered policy activates under real conditions
2. Quantify actual QPU-time overhead distribution
3. Report "agreement vs disagreement" frequency between probe and calibration rankings
4. Measure on-device tail metrics (95th/99th percentile logical failure)

## Protocol

### Phase 1: Baseline Period (Days 1-7)

Run standard QEC experiments using calibration-only selection:
- 3 sessions per day (0-8h, 8-16h, 16-24h post-calibration)
- Distance-5 repetition code
- 4096 shots per session
- Record: logical error rate, syndrome statistics, calibration timestamps

### Phase 2: DAQEC Deployment (Days 8-14)

Deploy drift-aware pipeline as the **default operational mode**:
- Same session schedule
- 30-shot probes before each session
- Probe-informed qubit selection
- Adaptive-prior decoding

### Metrics to Report

#### 1. Policy Trigger Frequency
```python
# Probe suggests different qubit chain than calibration
trigger_rate = (n_disagreement_sessions / n_total_sessions) * 100
```

Expected: 30-50% of sessions (based on simulation)

#### 2. QPU-Time Overhead Distribution
```python
# Per-session overhead
overhead_seconds = probe_time + selection_time
overhead_percent = overhead_seconds / total_session_time * 100
```

Report: mean, median, 95th percentile of overhead distribution

#### 3. Agreement/Disagreement Analysis
```python
# Kendall tau between probe-ranking and calibration-ranking
agreement_sessions = sessions where tau > 0.8
disagreement_sessions = sessions where tau < 0.5
partial_agreement = sessions where 0.5 <= tau <= 0.8
```

#### 4. On-Device Tail Metrics

| Metric | Baseline (Days 1-7) | DAQEC (Days 8-14) |
|--------|---------------------|-------------------|
| Mean logical error rate | X | Y |
| Median | X | Y |
| 95th percentile | X | Y |
| 99th percentile | X | Y |
| P(catastrophic burst) | X | Y |

### Hardware Configuration

- **Backend**: ibm_brisbane (or ibm_kyoto/ibm_osaka as backup)
- **Code distance**: 5
- **Syndrome rounds**: 3
- **Shots**: 4096 per experimental condition
- **Probe budget**: 30 shots per qubit × 15 candidate qubits = 450 shots

### QPU Budget Estimate

| Component | Sessions | Shots | Total |
|-----------|----------|-------|-------|
| Baseline (7 days × 3 sessions) | 21 | 4096 | 86,016 |
| DAQEC (7 days × 3 sessions) | 21 | 4096 | 86,016 |
| Probes (21 sessions × 450) | 21 | 450 | 9,450 |
| **Total** | 42 | - | **181,482 shots** |

Estimated QPU time: ~15-20 minutes (within Open Plan budget with careful scheduling)

### Analysis Scripts

```bash
# Run deployment study analysis
python scripts/deployment_study_analysis.py \
    --baseline-dir data/deployment/baseline/ \
    --daqec-dir data/deployment/daqec/ \
    --output results/deployment_study/

# Generate deployment study figures
python scripts/deployment_study_figures.py
```

### Expected Outputs

1. **Figure: Policy trigger frequency over time**
   - Bar chart showing daily trigger rate
   - Annotation: "Probe disagrees with calibration in X% of sessions"

2. **Figure: QPU overhead distribution**
   - Histogram of per-session overhead
   - Vertical line at 2% (claimed budget)

3. **Figure: Tail compression on-device**
   - Cumulative distribution of logical error rates
   - Baseline vs DAQEC overlay
   - Annotation: "99th percentile reduced from X to Y"

4. **Table: Deployment study summary**
   - Trigger rate, overhead stats, tail metrics
   - Statistical comparison (paired t-test, bootstrap CI)

### Success Criteria

The deployment study is successful if:
1. Policy triggers in >20% of sessions (demonstrates drift is real)
2. QPU overhead is <5% per session (demonstrates practicality)
3. Tail metrics improve by >50% (demonstrates operational benefit)
4. No sessions where DAQEC performs significantly worse than baseline

### Limitations to Acknowledge

- Single backend (generalization limited)
- 14-day window (seasonal effects not captured)
- Open Plan constraints (shot budget limited)
- No blinding possible (but pre-registered protocol)

---

## For Manuscript

If deployment study is completed before submission, add to Results:

> **Operational deployment.** To demonstrate practical viability, we deployed the drift-aware pipeline as the default operational mode for 7 days on ibm_brisbane. The probe-triggered policy activated in X% of sessions (Y/Z), with mean QPU overhead of W% per session. On-device 99th-percentile logical error rate decreased from A to B, confirming that tail compression transfers from controlled experiments to operational deployment.
