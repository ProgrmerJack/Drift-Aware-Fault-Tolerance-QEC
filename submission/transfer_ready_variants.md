# Transfer-Ready Manuscript Variants

If Nature Communications determines that the scope is narrower than their bar, manuscript transfer is a standard outcome (not a failure). This document prepares instant-pivot framings for alternative journals.

---

## Primary Target: Nature Communications

**Framing:** "Field-level reliability intervention for quantum error correction"

**Key selling points:**
- First systematic demonstration that calibration drift limits QEC reliability
- Operational principle (probe before execute) immediately applicable
- Tail-risk compression (not just mean improvement)

**Editor pitch:** "Advance likely to influence thinking in the field" + practical deployment path

---

## Alternative 1: PRX Quantum

**If NC says:** "Interesting but too applied / not fundamental enough"

**Reframing:** "Drift-aware error correction reveals hidden timescales in superconducting qubit noise"

**Emphasis shifts:**
- Lead with drift characterization (SI Section 2 → main text)
- Frame as physics insight: noise has structure exploitable for QEC
- Theoretical framework (OU model, optimal cadence derivation) gets prominent placement
- Hardware demonstrations support theory rather than leading

**Title variant:** "Exploiting Drift Timescales in Superconducting Qubit Error Correction"

**Abstract rewrite focus:**
- "We identify characteristic timescales in qubit parameter fluctuations..."
- "These timescales suggest an operational principle for adaptive QEC..."
- Physics narrative → engineering application (reversed from NC version)

---

## Alternative 2: npj Quantum Information

**If NC says:** "Nice work, but scope more suitable for specialized journal"

**Reframing:** "Practical drift-aware protocol for QEC reliability enhancement"

**Emphasis shifts:**
- Protocol emphasis (how-to orientation)
- Benchmark release as community contribution
- Integration examples more prominent
- Less "why this matters for the field" narrative

**Title variant:** "DAQEC: A Drift-Aware Protocol for Reliable Quantum Error Correction"

**Structural changes:**
- Methods/protocol section earlier
- Results → Technical Results
- Add "Protocol Availability" section

---

## Alternative 3: Quantum

**If NC says:** "Technical contribution but not broad enough impact"

**Reframing:** "Optimal probe scheduling for drift-aware quantum error correction"

**Emphasis shifts:**
- Theory section (OU model, cost optimization) becomes central
- Empirical results support theoretical predictions
- Less "practical deployment" narrative
- More "fundamental operational principle"

**Title variant:** "Optimal Probe Cadence for Drift-Aware Quantum Error Correction"

---

## Alternative 4: IEEE Transactions on Quantum Engineering

**If NC says:** "Engineering contribution, not science contribution"

**Reframing:** "Engineering reliable QEC: A drift-aware operational framework"

**Emphasis shifts:**
- Systems engineering perspective
- Overhead analysis prominent
- Integration with existing workflows
- Deployment case study
- Less "physics insight" narrative

**Title variant:** "A Drift-Aware Framework for Operational Quantum Error Correction"

---

## Transfer Checklist

If transferring from Nature Communications:

### What transfers automatically:
- [ ] Manuscript files
- [ ] Reviewer reports (if any received)
- [ ] Editorial correspondence
- [ ] Submission metadata

### What you must adapt:
- [ ] Abstract (reframe for target journal)
- [ ] Introduction (adjust scope claims)
- [ ] Discussion (match journal expectations)
- [ ] References (format changes)
- [ ] Figures (size/style requirements)
- [ ] SI structure (varies by journal)

### Timeline for pivot:
- **1 day**: Rewrite abstract and introduction framing
- **1 day**: Adjust Discussion section emphasis
- **2 days**: Reformat figures and references
- **Total**: 3-4 days to submission-ready

---

## Pre-Computed Abstract Variants

### PRX Quantum Version
> Calibration drift in superconducting qubits occurs on timescales comparable to experimental sessions, yet quantum error correction protocols typically assume static noise. Here we characterize drift dynamics on three IBM Quantum processors and show that this temporal structure can be exploited: by probing qubit parameters before each session and adapting both qubit selection and decoder priors, we achieve a 61% reduction in logical error rate across 756 experimental sessions. A theoretical model based on Ornstein-Uhlenbeck dynamics predicts an optimal probe interval of ~4 hours, consistent with our empirical findings. These results reveal that drift is not merely noise to be tolerated but information that can be leveraged for more reliable quantum computation.

### npj Quantum Information Version
> We present DAQEC, a drift-aware protocol for quantum error correction that improves reliability by adapting to real-time qubit performance rather than relying on stale calibration data. The protocol combines lightweight probe circuits (~30 seconds) with adaptive qubit selection and decoder-prior updates. Across 756 experimental sessions on IBM Quantum hardware, DAQEC reduces logical error rates by 61% on average, with even larger improvements (77%) for worst-case sessions. We release the DAQEC-Benchmark dataset (Zenodo DOI) and pip-installable toolkit to enable community adoption and extension. The protocol requires no hardware modifications and integrates with existing QEC workflows.

### Quantum Version
> We derive an optimal scheduling policy for probe-based drift compensation in quantum error correction. Modeling qubit parameter evolution as an Ornstein-Uhlenbeck process, we show that the cost-minimizing probe interval scales as √(probe_cost / (failure_cost × drift_rate)). For typical superconducting qubit parameters, this yields a 3-6 hour optimal interval. We validate this prediction across 756 experimental sessions on IBM Quantum processors, finding that the empirically-optimal interval (4 hours) agrees with the theoretical prediction within 5%. The resulting drift-aware protocol achieves 61% mean error reduction and 77% tail compression, demonstrating that optimal scheduling principles translate directly to improved fault-tolerant quantum computing reliability.

---

## Decision Tree

```
NC Decision
    │
    ├── Accept → Done 🎉
    │
    ├── Revise → Address concerns, resubmit
    │
    └── Reject/Transfer
            │
            ├── "Not fundamental enough" → PRX Quantum
            │
            ├── "Too specialized" → npj Quantum Information
            │
            ├── "Theory needs more" → Quantum
            │
            └── "Engineering focus" → IEEE TQE
```

---

## Key Metrics to Preserve Across Variants

Regardless of framing, always include:
- 61% mean improvement [58-64% CI]
- 77% tail compression (99th percentile)
- 756 sessions, 42 clusters, 3 backends
- Dose-response: ρ = 0.56, P < 10⁻⁴
- Theory prediction: τ* = 3.8h (matches empirical 4h)
