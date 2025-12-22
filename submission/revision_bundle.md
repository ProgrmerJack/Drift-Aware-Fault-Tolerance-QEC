# Revision Response Bundle

Pre-written response paragraphs for likely reviewer concerns. Each section provides a complete, copy-ready response that can be adapted for the actual revision.

---

## 1. "IBM-Specific / Platform Generalizability"

**Likely phrasing:** "The results are demonstrated only on IBM Quantum hardware. How do we know this approach generalizes to other platforms?"

### Response Paragraph

> We appreciate the reviewer's concern about platform generalizability. We address this through three complementary approaches:
>
> **Mechanism-level argument:** Our method exploits a universal property of superconducting qubits—time-varying coherence properties due to environmental fluctuations, two-level system defects, and thermal drift. This mechanism has been documented on Google Sycamore (Klimov et al., PRL 2018), Rigetti processors (multiple publications), and is fundamental to superconducting qubit physics. The specific magnitude of drift varies, but the phenomenon itself is platform-independent.
>
> **Simulation validation (SI-12):** We conducted Monte Carlo simulations with generic drift models (no IBM-specific noise characteristics) and reproduced the dose-response relationship between drift magnitude and improvement. At zero drift, improvement vanishes; at IBM-range drift (2–16%), improvements match hardware observations. This confirms the method is mechanism-driven, not hardware-artifact-driven.
>
> **Minimal requirements:** Our approach requires only: (1) qubit parameter heterogeneity at calibration time, (2) drift that reshuffles relative qubit rankings, (3) ability to probe current performance. All superconducting platforms satisfy these conditions.
>
> We have added explicit discussion of platform transferability in SI Section 12 and revised the Discussion to clarify the mechanism-level generality of our findings.

---

## 2. "Effect Size Too Large / Suspiciously Strong"

**Likely phrasing:** "A 61% improvement in logical error rate seems surprisingly large. What explains this magnitude?"

### Response Paragraph

> We share the reviewer's initial surprise at the effect magnitude and took extensive steps to validate it.
>
> **Mechanistic explanation:** The large effect arises from a compounding mechanism. Calibration drift causes: (1) suboptimal qubit selection (estimated 30–40% of effect), (2) incorrect decoder priors (estimated 15–25% of effect), and (3) violation of error-independence assumptions in MWPM matching (remainder). These three factors multiply rather than add, producing larger combined effects than intuition might suggest.
>
> **Consistency with prior work:** Wilson et al. (IEEE QCE 2020) reported 18% improvement from JIT transpilation alone. Kurniawan et al. (arXiv:2407.21462) showed up to 42% fidelity improvement from calibration-aware compilation. Our larger effect reflects: (a) independent probe validation rather than trusting backend reports, (b) QEC circuits' heightened sensitivity to correlated errors, (c) adaptive decoding benefits beyond qubit selection.
>
> **Internal consistency:** The dose-response relationship (larger improvement at greater staleness) provides internal validation. Effect increases monotonically with calibration age: 57.5% (fresh) → 58.2% (moderate) → 59.1% (stale). If the effect were artifactual, we would not expect such systematic variation with a mechanistically relevant covariate.
>
> **Robustness checks:** The multiverse analysis (SI-13) shows effects ranging from 54–67% across 60 analytical specifications, all statistically significant. The specification curve shows no discontinuities or suspicious outliers.

---

## 3. "Confounding with Diurnal/Load Patterns"

**Likely phrasing:** "Backend performance varies with time-of-day and queue load. How do you rule out these confounders?"

### Response Paragraph

> We designed the experimental protocol specifically to address time-of-day and load confounding:
>
> **Paired execution:** Both strategies (drift-aware and baseline) were executed in the same experimental session, submitted as simultaneous batch jobs. Any diurnal or load effects impact both conditions equally.
>
> **Within-cluster analysis (SI-9):** We analyzed effect size within day×backend clusters (42 total). If diurnal patterns drove the effect, within-day analysis would show attenuated effects. Instead, 40/42 clusters (95.2%) show positive correlation between staleness and improvement (sign test P < 10⁻⁴).
>
> **Stratified robustness:** Supplementary Table S-Sensitivity shows the dose-response relationship holds within each backend individually (Brisbane: ρ=0.66; Kyoto: ρ=0.59; Osaka: ρ=0.65; all P < 10⁻⁵). Different backends have different load patterns; consistent effects across backends argues against load confounding.
>
> **Permutation test:** When staleness labels are permuted within clusters (preserving cluster structure but breaking the staleness-outcome relationship), the observed correlation falls 6.21 standard deviations above the null mean (P < 10⁻⁴ from 10,000 permutations).
>
> The combination of paired design, within-cluster analysis, cross-backend replication, and permutation testing provides strong evidence against diurnal/load confounding.

---

## 4. "Baseline Fairness / Is the Baseline Too Weak?"

**Likely phrasing:** "The baseline uses static calibration. Is this a fair comparison? What about JIT or other adaptive baselines?"

### Response Paragraph

> The reviewer raises an important point about baseline selection. We chose our baseline to represent the current standard practice in QEC experiments, but we also provide explicit comparison to stronger alternatives:
>
> **Baseline justification:** Our "static calibration" baseline corresponds exactly to standard JIT (Just-in-Time) transpilation: qubit selection based on backend-reported properties at job submission time. This matches Wilson et al. (2020) and Kurniawan et al. (2024), and represents how most QEC experiments are conducted today.
>
> **JIT comparison (SI-10):** We provide head-to-head comparison stratified by calibration age. Key finding: drift-aware outperforms JIT baseline across all strata, including the "fresh calibration" stratum where JIT should perform best. This demonstrates that our method adds value beyond simple JIT approaches.
>
> **Why not stronger baselines?** 
> - **DGR (Google, 2024):** Hardware-specific technique requiring Google-proprietary control electronics; not implementable on IBM hardware.
> - **Drifting noise estimation:** Requires continuous syndrome stream; our discrete-session design precludes this.
> - **In-situ recalibration:** Requires fabricator-level access to recalibration routines; not available to external users.
>
> Our contribution is the strongest method implementable by external users without hardware-vendor cooperation. We have clarified this positioning in the Related Work section.

---

## 5. "Why Only Repetition Codes?"

**Likely phrasing:** "The experiments use repetition codes rather than full surface codes. How do results transfer to practical QEC?"

### Response Paragraph

> We use repetition codes for the same reason as Google's landmark QEC demonstrations (Acharya et al., Nature 2023): they isolate the error-correction mechanism while remaining executable on current hardware.
>
> **Practical constraints:** Full surface codes at distance-5 require 41+ high-quality qubits with appropriate connectivity. Current IBM Eagle processors cannot reliably support full surface code experiments at scale. Repetition codes provide the cleanest test of drift-aware selection without conflating results with surface-code-specific challenges.
>
> **Transfer mechanisms:** The core mechanisms we exploit—drift-induced qubit ranking changes and decoder prior mismatch—apply equally to any stabilizer code. Qubit selection for surface codes follows similar composite-score optimization over larger candidate sets. Adaptive decoding priors transfer directly (surface code MWPM uses identical weight-update equations).
>
> **Conservative estimate:** Repetition codes have smaller minimum-weight error chains than surface codes, making them less sensitive to correlated errors. Our improvements may therefore represent a conservative lower bound on surface code benefits, where correlated errors from stale priors cause more damage.
>
> We have added discussion of surface code transferability in the main text Discussion section.

---

## 6. "Statistical Independence / Pseudo-Replication"

**Likely phrasing:** "With 756 experimental sessions, is there pseudo-replication? Are the statistical tests valid?"

### Response Paragraph

> We designed the statistical analysis specifically to avoid pseudo-replication:
>
> **Session-level aggregation:** All primary analyses use session-level metrics (one data point per experimental session), not shot-level metrics. This prevents inflating sample sizes through within-session pseudo-replication.
>
> **Cluster-bootstrap inference:** Rather than assuming session independence, we use cluster-bootstrap with 42 day×backend clusters. This accounts for within-cluster correlation (same backend and day may share unmeasured confounders) while providing valid 95% confidence intervals.
>
> **Conservative degrees of freedom:** Our effective sample size is 42 clusters, not 756 sessions. All reported P-values use cluster-robust standard errors. Even with this conservative approach, effects remain highly significant (P < 10⁻¹⁰ for primary outcome).
>
> **Mixed-effects validation (SI-9):** The mixed-effects model with random intercepts for clusters confirms low ICC (0.062), indicating that cluster-level heterogeneity explains only 6.2% of variance. The staleness coefficient remains highly significant (P = 2.2 × 10⁻¹³) after accounting for cluster structure.
>
> We appreciate the reviewer's attention to statistical rigor and have added explicit discussion of our clustering strategy in the Methods section.

---

## 7. "Probe Overhead / Practical Deployment"

**Likely phrasing:** "The probe circuits consume QPU time. Is this overhead justified in practice?"

### Response Paragraph

> We quantify probe overhead in SI-5 (Table S-Overhead):
>
> **Measurement:** Probe circuits (T1, T2, readout characterization for 5 data qubits) require approximately 30 seconds of QPU time. A typical QEC session (10,000 shots × 10 syndrome rounds) requires approximately 15–20 minutes. Probe overhead is therefore 2.5–3.3% of total QPU time.
>
> **Break-even analysis:** Given 61% error rate reduction, the method is justified whenever error rates matter more than 3% throughput loss. For fault-tolerant applications, where error accumulation determines circuit depth limits, this trade-off is strongly favorable.
>
> **Amortization:** In operational deployment, probe overhead can be amortized across multiple QEC sessions. Probes remain useful for 1–4 hours depending on observed drift rates; a single 30-second probe investment supports dozens of subsequent sessions.
>
> **Comparison to alternatives:** Our probe-based approach is substantially cheaper than full recalibration (which requires vendor-level access and multi-hour turnaround) or continuous monitoring (which requires dedicated instrumentation). We provide the benefit of fresh calibration data at a fraction of the cost.
>
> We have added a practical deployment subsection to the Discussion addressing overhead considerations.

---

## Extension Script

The following describes how to extend results with additional data if requested:

### Adding +2 Backends

```bash
# Configure new backends (requires IBM Quantum access)
python scripts/extend_backends.py \
    --new-backends ibm_sherbrooke ibm_torino \
    --sessions-per-backend 50 \
    --output-dir data/extension/

# Re-run primary analysis with extended dataset
python scripts/run_full_analysis.py \
    --data-dir data/ data/extension/ \
    --output results/extended/
```

### Adding +7 Days

```bash
# Extend data collection period
python scripts/extend_timeline.py \
    --additional-days 7 \
    --backends ibm_brisbane ibm_kyoto ibm_osaka \
    --output-dir data/extension_7d/

# Re-run analysis
python scripts/run_full_analysis.py \
    --data-dir data/ data/extension_7d/ \
    --output results/extended_7d/
```

### Expected Outcomes

Based on power analysis:
- **+2 backends:** Expected to narrow CIs by ~20%, strengthen generalizability argument
- **+7 days:** Expected to narrow CIs by ~15%, add more staleness strata data points
- **Both:** Expected to convert marginal findings (if any) to definitive conclusions

---

## Quick Reference: Key Statistics

| Metric | Value | 95% CI | Source |
|--------|-------|--------|--------|
| Primary improvement | 61% | [58%, 64%] | Main text |
| Tail compression (95th) | 75.7% | [71%, 80%] | SI-11 |
| Tail compression (99th) | 77.2% | [72%, 82%] | SI-11 |
| JIT comparison (fresh) | 57.5% | [53%, 62%] | SI-10 |
| JIT comparison (stale) | 59.1% | [55%, 63%] | SI-10 |
| Multiverse range | 54–67% | All P < 0.001 | SI-13 |
| Within-cluster sign test | 40/42 positive | P < 10⁻⁴ | SI-9 |
