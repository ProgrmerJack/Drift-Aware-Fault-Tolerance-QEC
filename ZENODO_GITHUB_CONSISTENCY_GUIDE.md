=============================================================================
ZENODO AND GITHUB CONSISTENCY REQUIREMENTS
=============================================================================

Date: December 21, 2025
Manuscript: "A Reproducible Workflow for Drift Aware Cloud Quantum Error 
           Correction Experiments"

=============================================================================
THE CRITICAL CREDIBILITY ISSUE
=============================================================================

PROBLEM:
Your current public Zenodo record (if it exists from prior work) may contain
language consistent with the earlier "large-scale / strong-effect" narrative:
- "756 experiments"
- "60% improvement"
- "P < 10^-15"
- "126 sessions"

If you submit an honest pilot claiming N=10 with ~0% effect, but the linked
Zenodo/GitHub record advertises the opposite, reviewers/editors will treat
this as a CREDIBILITY RED FLAG:
- Best case: "Sloppy record-keeping"
- Worst case: "Data misrepresentation"

This alignment is MANDATORY before submission.

=============================================================================
REQUIRED ACTIONS
=============================================================================

OPTION A: Create New Zenodo Version (RECOMMENDED)
--------------------------------------------------

1. Go to your existing Zenodo deposit
2. Click "New version" 
3. Update the following fields to match honest pilot:

   Title:
   "Drift-Aware Fault-Tolerance in Quantum Error Correction: 
    Pilot Feasibility Study (N=10 IBM Hardware Experiments)"

   Description:
   "This dataset contains raw experimental results from a pilot feasibility
    study demonstrating drift-aware quantum error correction infrastructure
    on IBM Quantum hardware.
    
    EXPERIMENTAL SCOPE:
    - Total experiments: N=10 (4 deployment sessions + 6 surface code runs)
    - Single backend: ibm_fez (156-qubit Heron r2)
    - Experiment date: December 10, 2024
    - Shots per experiment: 4,096
    
    STATISTICAL POWER:
    This is a PILOT STUDY with explicitly insufficient sample size for
    statistical claims. With N=2 per condition in deployment comparison,
    power < 5% for detecting 50% effect. Future scaled studies require
    N≥42 sessions per condition for 80% power.
    
    NO PERFORMANCE CLAIMS:
    We make no claims of statistical significance or performance superiority.
    Both baseline and DAQEC strategies achieved mean LER = 0.360.
    
    CONTRIBUTION:
    This release provides validated experimental infrastructure and
    reproducibility artifacts enabling future properly powered studies."

   Keywords:
   - quantum error correction
   - drift-aware QEC
   - pilot feasibility study
   - reproducibility
   - IBM Quantum
   - cloud quantum computing
   - small sample size
   - infrastructure validation

   Files to include:
   ├─ experiment_results_20251210_002938.json (real data, 95.2 KB)
   ├─ simulated_results_20251209_235438.json (CLEARLY LABELED)
   │  └─ Add README_SIMULATION.txt explaining this is NOT real data
   ├─ source_data/fig2_deployment.csv
   ├─ source_data/fig3_surface_code.csv
   ├─ scripts/generate_honest_figures.py
   ├─ daqec/ (package source code)
   ├─ requirements.txt
   ├─ pyproject.toml
   ├─ README.md (UPDATED to match honest pilot)
   └─ LICENSE (MIT)

4. Publish new version
5. Update manuscript with new DOI

OPTION B: Create Separate Zenodo Records (CLEANER)
---------------------------------------------------

1. Create NEW record for honest pilot:
   DOI: 10.5281/zenodo.XXXXXXX (new number)
   Title: "Drift-Aware QEC Pilot Feasibility Study (N=10 Hardware Experiments)"
   Contains: Only real experimental data + infrastructure code

2. Create SEPARATE record for simulation:
   DOI: 10.5281/zenodo.YYYYYYY (different number)
   Title: "Simulated Projections for Drift-Aware QEC (NOT REAL DATA)"
   Contains: Only simulated data with prominent warnings

3. Link both in manuscript Data Availability:
   "Real experimental data: DOI 10.5281/zenodo.XXXXXXX
    Simulation code and projected results: DOI 10.5281/zenodo.YYYYYYY"

=============================================================================
GITHUB REPOSITORY REQUIREMENTS
=============================================================================

Repository: github.com/jackasher001/Drift-Aware-Fault-Tolerance-QEC

REQUIRED README.md STRUCTURE:
------------------------------

# Drift-Aware Fault-Tolerance in Quantum Error Correction

**Status:** Pilot Feasibility Study  
**Sample Size:** N=10 IBM hardware experiments  
**Statistical Power:** Insufficient for performance claims

## Quick Summary

This repository contains validated experimental infrastructure for drift-aware
quantum error correction on cloud quantum processors. The pilot study (N=10
experiments on IBM Fez) demonstrates:

✓ Successful surface code execution (d=3, 17 qubits)
✓ Probe-driven qubit selection mechanism
✓ Adaptive decoding infrastructure
✓ Complete reproducibility workflow

⚠ **Limitation:** Sample size (N=2 per condition) is explicitly insufficient
for statistical significance. Future scaled studies require N≥42 sessions.

## Experimental Data

**Real IBM Hardware Data:**
- File: `results/ibm_experiments/experiment_results_20251210_002938.json`
- Experiments: N=10 (4 deployment + 6 surface code)
- Backend: ibm_fez (156-qubit Heron r2)
- Date: December 10, 2024
- Shots: 4,096 per experiment

**Simulated Data (NOT REAL):**
- File: `results/simulated/simulated_results_20251209_235438.json`
- ⚠ This file contains SIMULATED PROJECTIONS, not real experimental results
- Mode: "simulated" (explicitly marked in file)
- Purpose: Projected expectations for properly powered future studies
- DO NOT cite as empirical evidence

## Installation

```bash
git clone https://github.com/jackasher001/Drift-Aware-Fault-Tolerance-QEC
cd Drift-Aware-Fault-Tolerance-QEC
pip install -e .
```

## One-Command Reproduction

```bash
python scripts/run_protocol.py --backend ibm_fez --shots 4096
```

## Citation

If you use this infrastructure, please cite:

```bibtex
@article{ashuraliyev2025daqec,
  title={A Reproducible Workflow for Drift Aware Cloud Quantum Error 
         Correction Experiments},
  author={Ashuraliyev, Abduxoliq},
  journal={npj Quantum Information},
  year={2025},
  note={Pilot feasibility study, N=10 experiments}
}
```

## Data Availability

Real experimental data: DOI 10.5281/zenodo.14536891

## License

MIT License - See LICENSE file

---

REQUIRED DIRECTORY STRUCTURE:
------------------------------

Drift-Aware-Fault-Tolerance-QEC/
├── README.md (UPDATED as above)
├── LICENSE (MIT)
├── requirements.txt
├── pyproject.toml
├── daqec/
│   ├── __init__.py
│   ├── probes.py
│   ├── selection.py
│   ├── decoding.py
│   └── policy.py
├── scripts/
│   ├── run_protocol.py
│   ├── generate_honest_figures.py
│   └── ...
├── results/
│   ├── ibm_experiments/
│   │   ├── experiment_results_20251210_002938.json (REAL)
│   │   └── README.md ("Real IBM hardware data")
│   └── simulated/
│       ├── simulated_results_20251209_235438.json (NOT REAL)
│       └── README_SIMULATION.md ("⚠ SIMULATED DATA - NOT REAL EXPERIMENTS")
├── manuscript/
│   ├── main_npjqi_brief.tex
│   ├── main_npjqi_brief.pdf
│   ├── figures/
│   └── source_data/
└── ...

REQUIRED README_SIMULATION.md:
-------------------------------

# ⚠ SIMULATED DATA - NOT REAL EXPERIMENTAL RESULTS

This directory contains SIMULATED projections, NOT real IBM hardware data.

**File:** simulated_results_20251209_235438.json

**Mode:** "simulated" (explicitly marked in file metadata)

**Purpose:** These are projected expectations for what a properly powered
study might observe, based on:
- Assumed 40% improvement effect size
- 66 simulated sessions (24 surface code + 42 deployment)
- Fabricated distributions matching expected behavior

**DO NOT cite this as empirical evidence.**

**For real experimental data, see:**
- `results/ibm_experiments/experiment_results_20251210_002938.json`
- N=10 real IBM hardware experiments
- Backend: ibm_fez
- Date: December 10, 2024

=============================================================================
VERIFICATION COMMANDS
=============================================================================

Before submission, run these checks:

1. Verify Zenodo record is public:
   ```
   curl -I https://doi.org/10.5281/zenodo.14536891
   # Should return HTTP 200, not 404
   ```

2. Verify GitHub repository is public:
   ```
   curl -I https://github.com/jackasher001/Drift-Aware-Fault-Tolerance-QEC
   # Should return HTTP 200, not 404
   ```

3. Check for "756" or "60%" in public records:
   ```
   # Go to Zenodo page in browser
   # Search page (Ctrl+F) for:
   - "756"
   - "60%"
   - "126 sessions"
   - "P<10"
   # Should find ZERO matches
   ```

4. Verify simulation clearly labeled:
   ```
   # Check that simulated_results file has:
   - "mode": "simulated" in JSON metadata
   - README_SIMULATION.md in same directory
   - Clear warning in main README.md
   ```

=============================================================================
MANUSCRIPT DATA AVAILABILITY SECTION (UPDATED IF NEEDED)
=============================================================================

If you separate real and simulated into different DOIs:

\section*{Data Availability}

Real experimental data are available at Zenodo with DOI 
10.5281/zenodo.14536891\cite{zenodo_real}. The deposit includes: 
(1) raw experimental results from N=10 IBM hardware experiments 
(experiment\_results\_20251210\_002938.json, 95.2 KB); (2) surface code 
syndrome measurements (3,391 total); (3) deployment session measurements 
(16,384 shots); (4) backend calibration snapshots; (5) source data CSV 
for all figures; (6) analysis notebooks.

Simulated projections (NOT included in manuscript claims) are available 
separately at DOI 10.5281/zenodo.YYYYYYY\cite{zenodo_simulated} for 
transparency. These simulations were used during experimental design but 
are not cited as empirical evidence.

=============================================================================
TIMELINE
=============================================================================

BEFORE SUBMISSION:
1. Update Zenodo record (1 hour)
2. Update GitHub README and directory structure (2 hours)
3. Verify all public links (30 minutes)
4. Re-compile manuscript with updated DOIs if needed (15 minutes)
5. Final verification scan (15 minutes)

TOTAL TIME: ~4 hours

DO NOT SKIP THESE STEPS. Inconsistent public records are a submission killer.

=============================================================================
SUMMARY CHECKLIST
=============================================================================

[ ] Zenodo record describes N=10 pilot (not "756 experiments")
[ ] Zenodo record makes no "60% improvement" claims
[ ] Simulated data clearly labeled as "NOT REAL DATA"
[ ] GitHub README matches honest pilot description
[ ] GitHub directory structure separates real from simulated
[ ] Both Zenodo and GitHub publicly accessible
[ ] No "756" or "60%" language in public-facing descriptions
[ ] Manuscript DOIs point to correct records
[ ] All verification commands pass

=============================================================================
END OF DOCUMENT
=============================================================================
