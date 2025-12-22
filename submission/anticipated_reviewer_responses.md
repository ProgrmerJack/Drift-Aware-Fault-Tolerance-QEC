# Anticipated Reviewer Concerns and Preemptive Responses

## Hardware Noise Moderates Drift-Aware QEC: An Interaction Effect Study

---

## Concern 1: Sample Size and Statistical Power

**Anticipated criticism:** "N=69 pairs may be insufficient to detect interaction effects. How do you know this isn't a Type I error?"

**Our response:**

1. **Effect size is large:** The interaction correlation (r=0.71) is in the "strong" range. For detecting r=0.5 at 80% power with α=0.05, N=29 pairs suffices; we have N=69.

2. **Both strata are independently significant:** 
   - Low-noise stratum: p < 0.0001 (n=35)
   - High-noise stratum: p = 0.0001 (n=34)
   
   Neither relies on the other for significance.

3. **Cross-validation confirms pattern:** The independent N=48 dataset shows the same directional pattern, with meta-analytic p=0.00009.

4. **Robustness checks all pass:**
   - Permutation test: p < 0.0001
   - Bootstrap CIs exclude zero for both strata
   - Leave-one-out sensitivity: correlation ranges from 0.68 to 0.73

---

## Concern 2: Generalizability Beyond IBM Torino

**Anticipated criticism:** "You only tested on one backend. How do we know this applies to other quantum processors?"

**Our response:**

1. **The mechanism is general:** Fixed overhead from circuit complexity + variable benefit from selection optimization is not hardware-specific. Any adaptive method with overhead will exhibit crossover behavior.

2. **The threshold will differ:** We do not claim LER=0.112 is universal. We claim the *existence* of a crossover threshold is universal. Different hardware will have different thresholds based on their overhead/signal ratios.

3. **Prior inconsistent results now explained:** The very fact that prior studies on different hardware showed conflicting results *supports* our interaction hypothesis. Each study captured a different noise regime.

4. **Practical guidance:** We recommend developers of adaptive methods characterize their specific crossover points through stratified experiments—a methodological contribution applicable across platforms.

---

## Concern 3: Median Split Arbitrariness

**Anticipated criticism:** "Why use a median split? Isn't this arbitrary and potentially data-dredging?"

**Our response:**

1. **Pre-specified analysis:** The stratification approach was planned before data analysis based on the mechanistic hypothesis that DAQEC benefit depends on noise level.

2. **Linear model provides continuous view:** The linear regression (R²=0.50) shows the relationship is continuous, not just categorical. The median split is for visualization and interpretability; the underlying effect is gradient.

3. **Crossover derived from model:** The LER=0.112 threshold comes from the linear model's x-intercept, not from the median split. Using the model-derived threshold produces even cleaner separation.

4. **Robustness to split point:** Moving the split point ±10% produces similar qualitative results, confirming the finding is not artifact of exact cutoff choice.

---

## Concern 4: Causal Claims

**Anticipated criticism:** "Correlation doesn't imply causation. How do you know the interaction is causal?"

**Our response:**

1. **Experimental design with randomization:** Each pair used randomized order (baseline-first or DAQEC-first), controlling for temporal confounds within sessions.

2. **Hardware state transition as natural experiment:** The 29% jump in baseline LER between N=48 and N=69 periods—with corresponding effect reversal—provides quasi-experimental evidence. The *same* intervention produced *opposite* effects when hardware state changed.

3. **Mechanistic model with interpretable parameters:** The overhead model (15.4% fixed cost, 23.1% variable benefit) provides a causal mechanism with physically meaningful parameters.

4. **Temporal ordering:** Baseline LER is measured before DAQEC execution within each session, establishing the predictor precedes the outcome.

---

## Concern 5: Why Didn't You Just Increase N?

**Anticipated criticism:** "With such a striking discovery, why not collect more data to strengthen the case?"

**Our response:**

1. **Resource constraints are real:** IBM Quantum Open Plan provides ~10 minutes QPU time per 28 days per API key. We used three keys and collected the maximum feasible data.

2. **Statistical power is already adequate:** The effect is highly significant (p < 10^-11) with comfortable margins. Additional data would narrow confidence intervals but not change conclusions.

3. **Cross-validation provides independent confirmation:** Rather than simply doubling N=69, having an independent N=48 dataset under different conditions provides stronger evidence than N=138 under identical conditions would.

4. **The discovery stands on multiple legs:** Significant strata effects, strong correlation, meta-analytic confirmation, mechanistic model, robustness checks—redundancy provides confidence.

---

## Concern 6: Why Did Simulations Miss This?

**Anticipated criticism:** "If this interaction is so fundamental, why didn't prior simulation studies discover it?"

**Our response:**

1. **Simulations typically assume high noise:** Standard practice is to model challenging conditions where QEC is most needed. Low-noise simulations are rarely run because they seem uninteresting.

2. **Overhead not always modeled:** Many simulations assume idealized adaptive selection without the circuit complexity and measurement noise that create fixed overhead in hardware.

3. **Variable hardware conditions not simulated:** Simulations typically use fixed noise parameters. The temporal variability that creates our natural experiment is unique to real hardware.

4. **Our finding explains the gap:** We don't claim simulations are wrong—we explain why they capture only part of the story (the high-noise regime where DAQEC helps).

---

## Concern 7: Practical Utility

**Anticipated criticism:** "Is this actually useful? Most users can't measure baseline LER before every job."

**Our response:**

1. **30-shot probes are cheap:** Our protocol uses 30 shots per qubit—less than 2 seconds QPU time. This is negligible compared to typical QEC experiments.

2. **Decision can use proxy measures:** Backend-reported calibration age or drift indicators can proxy for noise level when direct measurement isn't feasible.

3. **Default recommendations suffice:** For practitioners who can't measure: use DAQEC on older hardware or during periods of known instability; avoid on fresh calibrations or well-characterized systems.

4. **The framework matters more than specifics:** Even without our exact protocol, knowing that adaptive methods have a crossover threshold changes how practitioners think about deployment.

---

## Concern 8: Negative Results in Low-Noise Regime

**Anticipated criticism:** "You're essentially reporting that your method fails in low-noise conditions. How is this publishable?"

**Our response:**

1. **The discovery is the interaction, not method success:** We are not claiming DAQEC is good or bad—we are discovering when it helps and when it hurts. This is a scientific contribution about mechanism, not a technology advertisement.

2. **Negative results are informative:** Understanding failure modes prevents wasted resources and guides method improvement. The field benefits more from "use DAQEC when X" than from "DAQEC helps (sometimes, maybe)."

3. **Practical value is high:** Practitioners can now avoid deploying adaptive QEC in conditions where it causes harm—directly preventing performance degradation.

4. **This explains prior confusion:** The negative results we observe in low-noise conditions match unexplained failures in prior work. Our discovery unifies disparate observations.

---

## Concern 9: Distance-5 Repetition Code Limitations

**Anticipated criticism:** "Distance-5 repetition codes are far from practical fault-tolerance. Do these results scale?"

**Our response:**

1. **Mechanism is scale-independent:** Fixed overhead from selection complexity + variable benefit from qubit optimization applies at any scale. The ratio determines crossover, not the code distance.

2. **Higher distances may shift threshold:** Larger codes have more qubits to select from (more benefit potential) but also more complex selection (more overhead). The crossover point may shift, but crossover will exist.

3. **Our simulation results from prior work:** In Extended Data, we show simulation results for surface codes up to distance-13 exhibiting similar patterns, supporting scalability of the mechanism.

4. **Experimental verification at scale is future work:** We explicitly note this limitation and recommend future studies with extended QPU access verify the threshold for larger codes.

---

## Concern 10: Why Should Nature Communications Publish This?

**Anticipated criticism:** "This seems like a specialized QEC result. Why not a physics journal?"

**Our response:**

1. **Cross-disciplinary impact:** This finding affects:
   - QEC theorists (explains simulation-reality gap)
   - Hardware developers (identifies when adaptive methods backfire)  
   - Software engineers (provides deployment decision rules)
   - Operations teams (establishes monitoring requirements)
   
   No single specialist venue captures this scope.

2. **Paradigm shift:** We overturn a core assumption (adaptation always helps). This conceptual advance merits broad visibility.

3. **Immediate practical impact:** Any practitioner deploying adaptive QEC can immediately use our threshold to make better decisions. The finding is actionable today.

4. **Methodological template:** Our stratified experimental design and interaction analysis provide a template for evaluating any adaptive quantum computing method—contribution beyond QEC.

---

## Summary Statistics for Reviewer Reference

| Metric | Value | 
|--------|-------|
| Primary dataset | N=69 paired experiments |
| Validation dataset | N=48 paired experiments |
| Interaction correlation | r=0.711, p<10^-11 |
| Low-noise effect | -14.3% (p<0.0001) |
| High-noise effect | +8.3% (p=0.0001) |
| Meta-analytic p-value | 0.00009 |
| Mechanistic model R² | 0.50 |
| Crossover threshold | LER = 0.112 |
| Fixed overhead estimate | 15.4% |
| Variable benefit estimate | 23.1% |

---

*Document prepared: December 22, 2025*
