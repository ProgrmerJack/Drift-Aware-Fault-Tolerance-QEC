# Manuscript Scientific Integrity Verification

## Document Purpose

This document certifies that the manuscript has been comprehensively revised to ensure internal scientific consistency. All claims are now bounded by the real experimental evidence.

---

## Critical Problem Identified

The original manuscript contained **fundamental scientific inconsistencies**:

| Original Claim | Actual Evidence |
|----------------|-----------------|
| "756 experiments on IBM Quantum backends spanning 14 calibration cycles" | **10 real experiments** (4 deployment + 6 surface code) on 1 backend (ibm_fez) |
| "126 sessions across 3 backends (Brisbane, Kyoto, Osaka)" | **4 sessions on 1 backend** (ibm_fez only) |
| "60% relative reduction in mean logical error rate" | **~0% difference** (baseline: 0.360, DAQEC: 0.360) |
| "Cohen's d = 3.82, P < 10⁻¹⁵" | **Cohen's d ≈ 0.04, p = 0.97** (underpowered) |
| "42 day×backend clusters" | **1 experimental session** |
| "Holdout validation on ibm_fez" | ibm_fez is the **ONLY** backend tested |
| "756 syndrome-level experiments" | Came from **simulated data**, not real hardware |

### Root Cause

The manuscript conflated:
1. **Simulated projections** (`simulated_results_20251209_235438.json`, mode="simulated") containing fabricated 42 sessions with 40% improvement
2. **Real IBM hardware data** (`experiment_results_20251210_002938.json`) containing only 10 actual experiments

The simulated data was explicitly marked `"mode": "simulated"` and `"note": "Simulated results based on expected distributions from prior experiments"` but was cited as if it were real experimental evidence.

---

## Resolution: Honest Pilot Study Reframing

### Decision Made

**Option A: Reframe as Pilot Feasibility Study**

Given:
- Real hardware data: N=10 experiments total
- Real improvement: Negligible (~0%, p=0.97)
- Simulated data: Fabricated and not usable for empirical claims

Option B (keeping 60% claims) was impossible without fabrication.

### New Manuscript Structure

| Component | Original (Dishonest) | Revised (Honest) |
|-----------|---------------------|------------------|
| **Target journal** | Nature Communications | npj Quantum Information / PRX Quantum Technical Note |
| **Central claim** | "60% LER reduction with extreme significance" | "Pilot feasibility study demonstrating infrastructure" |
| **Sample size** | "N=756 experiments" | "N=10 experiments (explicitly underpowered)" |
| **Statistical claims** | "P < 10⁻¹⁵, Cohen's d = 3.82" | "p = 0.97, insufficient power for significance" |
| **Effect interpretation** | "Overwhelming evidence" | "Insufficient sample size for effect detection" |
| **Contribution** | "Definitive proof of method" | "Infrastructure validation enabling future scaled studies" |

---

## Files Created/Modified

### New Figures (Honest Representations)

| File | Size | Description |
|------|------|-------------|
| `fig1_pipeline_and_data.png` | 126.9 KB | Pipeline schematic + N=10 data summary |
| `fig2_deployment_pilot.png` | 225.3 KB | Deployment comparison with explicit N=2 underpowering note |
| `fig3_surface_code.png` | 217.4 KB | Surface code results with honest LER interpretation |
| `fig4_complete_summary.png` | 220.3 KB | All N=10 experiments with checkmarks for feasibility claims |

### New Manuscript Files

| File | Description |
|------|-------------|
| `main_pilot_honest.tex` | Complete rewritten manuscript (10 pages, 833 KB PDF) |
| `figure_legends_pilot_honest.tex` | Accurate legends matching actual figure content |
| `source_data/fig2_deployment.csv` | 4 rows of real deployment data |
| `source_data/fig3_surface_code.csv` | 6 rows of real surface code data |

### Generation Script

| File | Description |
|------|-------------|
| `scripts/generate_honest_figures.py` | Generates figures from real data only, no placeholders |

---

## Claim-Evidence Alignment

### New Abstract (149 words)

> We present a **pilot feasibility study** demonstrating a drift-aware quantum error correction protocol on IBM Quantum hardware. Using 30-shot probe circuits to refresh qubit rankings between calibrations, combined with adaptive-prior decoding, we establish the complete experimental infrastructure for drift-aware QEC. In pilot experiments on ibm_fez (**N=10 total**: 4 deployment sessions comparing baseline vs. drift-aware selection, plus 6 surface code runs), we demonstrate successful execution of both distance-3 surface codes and distance-5 repetition codes on 156-qubit hardware. **While the pilot sample size (N=2 per condition) is insufficient for statistical significance**, we observe ranking instability between sessions confirming sub-calibration drift, and validate that the probe-driven selection mechanism functions correctly in production.

### Key Honesty Statements

1. **Sample size acknowledgment**: "N=10 total experiments" stated explicitly
2. **Power limitation**: "insufficient for statistical significance" in abstract
3. **Effect interpretation**: "This result is expected given the small sample size"
4. **Future requirements**: "N≥21 sessions per condition" for 80% power stated
5. **Contribution scope**: "methodological rather than empirical"

---

## Figure-Legend-Source Data Consistency Check

| Figure | Legend Describes | Figure Shows | Source Data | ✓/✗ |
|--------|-----------------|--------------|-------------|-----|
| Fig 1a | Pipeline schematic | Pipeline schematic | N/A | ✓ |
| Fig 1b | N=10 experiment counts by category | Bar chart with 2+2+6 | N/A | ✓ |
| Fig 2a | N=2 paired scatter | Paired scatter with 2 points per condition | fig2_deployment.csv (4 rows) | ✓ |
| Fig 2b | Bar chart with SD | Bar chart with overlaid points | fig2_deployment.csv | ✓ |
| Fig 2c | Statistics with underpowering warning | Text panel with warning | Computed from CSV | ✓ |
| Fig 3a | 6 surface code runs | 6 bars with LER | fig3_surface_code.csv (6 rows) | ✓ |
| Fig 3b | Summary statistics | Text panel | Computed from CSV | ✓ |
| Fig 4 | All N=10 experiments | Strip plot with 10 points | All source data | ✓ |

**All figure panels now match their legends and are backed by source data.**

---

## Verification Commands

```bash
# Verify real data
python -c "import json; d=json.load(open('results/ibm_experiments/experiment_results_20251210_002938.json')); print('Deployment:', len(d.get('deployment_results',[]))); print('Surface code runs:', sum(len(s.get('runs',[])) for s in d.get('surface_code_results',[])))"
# Expected output: Deployment: 4, Surface code runs: 6

# Verify source data
wc -l manuscript/source_data/*.csv
# Expected: 5 lines in fig2 (header + 4 rows), 7 lines in fig3 (header + 6 rows)

# Verify figures exist
ls -la manuscript/figures/fig*_*.png
# Expected: 4 new honest figures

# Verify manuscript compiles
pdflatex manuscript/main_pilot_honest.tex
# Expected: 10 pages, ~833 KB
```

---

## Submission Pathway

### Recommended Venue Change

| Aspect | Original Target | Revised Target |
|--------|-----------------|----------------|
| Journal | Nature Communications | **npj Quantum Information** or **Physical Review Applied** |
| Article type | Full Article | **Technical Note** or **Letter** |
| Contribution | "Definitive proof" | "Feasibility demonstration + open infrastructure" |
| Main selling point | Effect size | **Reproducibility artifact enabling scaled replication** |

### Reviewer Preemption

The revised manuscript explicitly addresses:

1. **"Small N" concern**: Acknowledged in abstract, methods, and discussion
2. **"No significant effect" concern**: Explained as expected given underpowering
3. **"Why publish?" concern**: Infrastructure contribution for community replication
4. **"Future work" requirement**: Power analysis provided (N≥21 required)

---

## Certificate of Scientific Integrity

I certify that:

- [x] All claims in the revised manuscript are bounded by real experimental evidence
- [x] No simulated data is presented as real experimental data
- [x] Sample sizes are accurately stated (N=10 total)
- [x] Statistical limitations are explicitly acknowledged
- [x] Figure legends accurately describe figure contents
- [x] Source data files contain all data needed to reproduce figures
- [x] The manuscript makes no claims of statistical significance with N=2 per condition
- [x] The contribution is reframed as infrastructure/feasibility rather than effect proof

---

## Summary

The manuscript has been transformed from a **scientifically indefensible** document claiming "756 experiments with 60% improvement and P<10⁻¹⁵" (based on simulated data presented as real) into an **honest pilot feasibility study** accurately representing N=10 real IBM hardware experiments with explicit acknowledgment of statistical limitations.

This transformation preserves the genuine scientific value (validated infrastructure, observed drift, reproducibility artifact) while removing all unsupported claims.

**The revised manuscript is now internally consistent and scientifically defensible.**

---

*Document generated: 2025-01-15*
*Revision status: Complete*
