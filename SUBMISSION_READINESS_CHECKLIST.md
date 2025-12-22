# Nature Communications Submission Readiness Checklist

**Project:** Drift-Aware Fault-Tolerance in Quantum Error Correction  
**Target Journal:** Nature Communications  
**Generated:** December 10, 2024  
**Status:** Pre-submission preparation phase

---

## 1. Manuscript Requirements

### 1.1 Core Manuscript Components

- [x] **Title**: Clear, descriptive, no abbreviations (currently 17 words - acceptable)
- [x] **Abstract**: ≤150 words (currently 169 words - **NEEDS REDUCTION by 19 words**)
- [ ] **Main text**: ≤5,000 words (not counting abstract, refs, methods, figure legends, tables)
  - **ACTION REQUIRED**: Count current main text word count and verify compliance
  - Sections to count: Introduction, Related Work, Results, Discussion
  - Exclude: Methods, Data/Code Availability, Acknowledgments, References, Figure Legends
- [x] **Line numbers**: Enabled (`\linenumbers` active in LaTeX)
- [x] **Double spacing**: Enabled (`\doublespacing` active)
- [x] **Author affiliations**: Present (placeholder format - needs real names before submission)
- [ ] **Corresponding author**: Email present but needs real email before submission

### 1.2 Figure Requirements

- [ ] **Total display items**: ≤10 (main figures + tables combined)
  - Current: 5 figures + 2 tables in Results + 2 tables in hardware validation = **9 items ✓**
- [ ] **Figure quality**: All figures must be production-ready
  - **ACTION REQUIRED**: Verify all 5 referenced figures exist in publication-ready format
  - Required formats: PDF, EPS, or TIFF (min 300 DPI for photographs, 1000 DPI for line art)
  - Check: `fig:drift`, `fig:bursts`, `fig:primary`, `fig:mechanism`, `fig:ablation`
- [ ] **Figure legends**: Complete and detailed
  - **STATUS**: Figure legends present at end of manuscript but need verification against actual figures
- [ ] **Source data**: Excel file (SourceData.xlsx) for all data-containing figures
  - **ACTION REQUIRED**: Create SourceData.xlsx with tabs for each figure
  - Required tabs: Figure 1, Figure 2, Figure 3, Figure 4, Figure 5, Table 1, Table 2, Table 3, Table 4

### 1.3 Supplementary Information

- [x] **SI file exists**: `si/SI.tex` present and compiled to PDF
- [ ] **SI sections complete**:
  - [ ] Benchmark definition section referenced in main text
  - [ ] Probe circuit details and validation
  - [ ] Statistical analysis pre-registration
  - [ ] Extended methods (software versions, parameter settings)
  - [ ] Additional analyses (stratification methodology, confounder sensitivity)
  - [ ] Negative control experiments
  - [ ] Design rule derivation (4-hour probe cadence justification)
  - [ ] Full statistical tables
  - [ ] Extended data figures (if any beyond main 5)
  - **ACTION REQUIRED**: Review SI.tex to verify all referenced sections exist

### 1.4 References

- [ ] **Bibliography complete**: All in-text citations have corresponding entries
  - **ACTION REQUIRED**: Verify all `\cite{}` commands have entries in references
- [ ] **Format**: Nature style (numbered, sequential)
  - **STATUS**: Using `naturemag` bibliography style ✓
- [ ] **DOI links**: All references have DOIs where available
  - **ACTION REQUIRED**: Add DOI links to all references (use `\url{}` or hyperref)
- [ ] **Preprints**: arXiv preprints acceptable but prefer peer-reviewed where possible

---

## 2. Data and Code Availability

### 2.1 Data Deposition

- [ ] **Zenodo DOI**: Upload data and obtain permanent DOI
  - **CURRENT STATUS**: Placeholder `10.5281/zenodo.XXXXXXX` in manuscript
  - **ACTION REQUIRED**: 
    1. Create Zenodo deposit with main dataset (master.parquet - 756 experiments)
    2. Include IBM Fez hardware data (experiment_results_20251210_002938.json)
    3. Include HARDWARE_EXPERIMENT_RESULTS.md
    4. Add SourceData.xlsx when created
    5. Reserve DOI before submission (can upload under embargo during review)
    6. Update manuscript with real DOI
- [ ] **Data completeness**:
  - [ ] master.parquet (main 756-experiment dataset)
  - [ ] IBM Fez raw results JSON (3,391 lines)
  - [ ] IBM Fez analysis summary
  - [ ] SourceData.xlsx (all figure data)
  - [ ] Raw syndrome bitstrings (referenced in manuscript)
  - [ ] Probe measurement logs
  - [ ] Backend property snapshots (calibration data from IBM)
- [ ] **Data format**: Plain text/CSV or standard formats (Parquet, JSON, Excel accepted)
- [ ] **Metadata**: README in Zenodo deposit explaining file structure and data dictionary
- [ ] **License**: CC BY 4.0 or CC0 (Zenodo default) - **confirm before upload**

### 2.2 Code Availability

- [x] **GitHub repository public**: Repository already public
- [ ] **Code completeness**:
  - [x] Main analysis pipeline (`protocol/run_protocol.py`)
  - [x] IBM Fez experiment scripts (`scripts/run_ibm_experiments.py`, `scripts/analyze_ibm_results.py`)
  - [ ] Drop-in API functions referenced in manuscript:
    - [ ] `select_qubits_drift_aware()` - **verify exists and documented**
    - [ ] `recommend_probe_interval()` - **verify exists and documented**
    - [ ] `decode_adaptive()` - **verify exists and documented**
  - [x] Requirements.txt with pinned versions
  - [ ] Installation instructions in README
  - [ ] Usage examples/tutorials
- [ ] **License**: MIT License present ✓
- [ ] **Reproducibility**:
  - [x] Pre-registered protocol (`protocol/protocol.yaml`)
  - [ ] Instructions to reproduce all figures
  - [ ] Seed documentation for random number generators
  - [ ] Documented software environment (Docker/Conda environment file recommended)
- [ ] **GitHub DOI**: Consider archiving repository snapshot on Zenodo for permanent DOI
  - **BENEFIT**: Provides immutable version-of-record alongside live development repo

---

## 3. Ethical and Transparency Requirements

### 3.1 Mandatory Statements

- [x] **Data availability statement**: Present in manuscript ✓
- [x] **Code availability statement**: Present in manuscript ✓
- [x] **Acknowledgments**: Present (placeholder - verify before submission)
- [x] **Author contributions**: Present (placeholder - needs real authors)
- [x] **Competing interests**: Present ("no competing interests" declared)
- [ ] **Funding statement**: 
  - **CURRENT**: Not explicitly present
  - **ACTION REQUIRED**: Add funding statement section if any grants supported work
  - If unfunded: Add statement "This research received no specific grant from any funding agency"

### 3.2 Reporting Summary (Required for Experimental Studies)

- [ ] **Nature Reporting Summary**: Complete structured reporting checklist
  - **ACCESS**: Available at https://www.nature.com/documents/nr-reporting-summary.pdf
  - **ACTION REQUIRED**: Fill out applicable sections:
    - Statistics and reproducibility
    - Software availability
    - Life sciences (N/A for this study)
    - Materials availability (N/A for computational study)
  - **SUBMISSION**: Upload as separate PDF alongside manuscript

### 3.3 Pre-registration

- [x] **Protocol pre-registered**: `protocol/protocol.yaml` present
- [ ] **Public pre-registration**: Consider depositing protocol on OSF or similar before submission
  - **BENEFIT**: Strengthens claims of confirmatory analysis vs. exploratory
  - **ALTERNATIVE**: Can reference GitHub commit hash from before data collection

### 3.4 Ethics Statements (If Applicable)

- [x] **Not applicable**: No human participants, no animal studies, no field work
- [x] **Data ethics**: Using publicly accessible IBM Quantum infrastructure (no IRB required)

---

## 4. Submission Package Components

### 4.1 Required Files for Initial Submission

- [ ] **Main manuscript PDF**: Compiled from `manuscript/main.tex`
  - [ ] Verify PDF compiles without errors
  - [ ] Check that line numbers appear
  - [ ] Verify double spacing applied
  - [ ] Strip identifying metadata if opting for double-anonymized review
- [ ] **Supplementary Information PDF**: Compiled from `si/SI.tex`
- [ ] **Cover letter**: Use `submission/cover_letter.md` as template
  - [ ] Replace all [BRACKETED] placeholders
  - [ ] Add suggested reviewers (3-5 experts, no collaborators)
  - [ ] Declare if opting into double-anonymized peer review
  - [ ] Specify if manuscript was previously reviewed elsewhere (if applicable)
- [ ] **Source data file**: SourceData.xlsx (one tab per figure/table)
- [ ] **Reporting Summary PDF**: Nature's structured checklist
- [ ] **Figure files**: Individual high-resolution figures (if requested - some journals accept embedded only)

### 4.2 Optional but Recommended

- [ ] **Editor memo**: Consider creating one-page summary for editors (see `submission/editor_memo.tex`)
  - Highlight: novelty, significance, public impact, why Nature Communications
- [ ] **Suggested reviewers list**: Extended version with brief justifications
- [ ] **Author photo and bio**: For potential press coverage if accepted

---

## 5. Double-Anonymized Peer Review (Optional)

**Nature Communications offers this - decide whether to opt in**

### 5.1 If Opting In

- [ ] **Manuscript anonymization**:
  - [ ] Remove author names/affiliations from manuscript PDF
  - [ ] Anonymize self-citations (third person: "Previous work [ref]" not "We showed [ref]")
  - [ ] Remove acknowledgments or anonymize (replace names with "[removed for review]")
  - [ ] Check funding statements for identifying grant numbers
- [ ] **Repository anonymization**:
  - [ ] Create Anonymous GitHub mirror at https://anonymous.4open.science/
  - [ ] Update manuscript to reference anonymous GitHub URL during review
  - [ ] Check code for identifying comments, file paths, email addresses
- [ ] **Cover letter**: State "We opt into double-anonymized peer review"
- [ ] **After acceptance**: Restore real authorship to all materials

**RECOMMENDATION**: Given strong reproducibility focus, double-anonymization may limit review impact. Standard review allows reviewers to evaluate reproducibility claims by accessing real repository. Consider carefully.

---

## 6. Pre-Submission Verification

### 6.1 Content Verification

- [ ] **Abstract word count**: Reduce from 169 to ≤150 words
- [ ] **Main text word count**: Verify ≤5,000 words
- [ ] **Display items count**: Verify ≤10 (currently 9 ✓)
- [ ] **All citations present**: Run LaTeX, check for `[?]` or warnings
- [ ] **All figures referenced**: Verify every `\figref{}` and `\tabref{}` resolves
- [ ] **All tables complete**: Verify Tables 1-4 have all data filled in

### 6.2 Technical Checks

- [ ] **LaTeX compilation**: Clean compile with no errors
  - [ ] Run: `pdflatex main.tex` (twice for cross-references)
  - [ ] Run: `bibtex main` (if using BibTeX)
  - [ ] Verify SI.tex compiles independently
- [ ] **Code execution**: Test reproducibility
  - [ ] Fresh clone of repository
  - [ ] Install from requirements.txt
  - [ ] Run `python protocol/run_protocol.py --mode=test` (if test mode exists)
- [ ] **Links functional**: Test all URLs in manuscript (Zenodo, GitHub, IBM docs)
- [ ] **File paths**: Replace any Windows backslashes with forward slashes in manuscript

### 6.3 Statistical Review

- [ ] **Sample sizes reported**: All n values present
- [ ] **Effect sizes reported**: Cohen's d, Cliff's delta, etc. all present ✓
- [ ] **P-values reported**: All statistical tests include exact p-values ✓
- [ ] **Confidence intervals**: All estimates include 95% CIs ✓
- [ ] **Multiple comparisons**: Holm-Bonferroni mentioned ✓
- [ ] **Negative controls**: Pre-registered controls described ✓

---

## 7. Nature Communications Specific Requirements

### 7.1 Formatting

- [x] **Font**: Times/Times New Roman (using `mathptmx` package ✓)
- [x] **Font size**: 11-12pt (using 11pt ✓)
- [x] **Margins**: 1 inch all sides (using geometry package ✓)
- [x] **Spacing**: Double spacing ✓
- [x] **Page numbers**: Automatic in LaTeX ✓
- [x] **Line numbers**: Continuous (using lineno package ✓)

### 7.2 Structure

- [x] **Section headings**: No numbered sections (using `\section*{}` ✓)
- [x] **Subsection format**: No numbering in main text ✓
- [x] **Figure legends**: At end of manuscript (not with figures) ✓
- [x] **Tables**: Embedded in text or at end
  - **CURRENT**: Tables embedded in Results section ✓

### 7.3 Journal-Specific Policies

- [ ] **Transparent peer review**: Decide whether to opt in
  - If yes: Check box during submission; peer review reports will be published
- [ ] **Open access**: Nature Communications is fully OA - confirm APC funding
  - **APC Cost**: ~$5,890 USD (as of 2024)
  - **Waivers**: Check if institution has agreement or if waiver eligible
  - **ACTION REQUIRED**: Confirm funding source for APC before submission
- [ ] **Preprint policy**: Allowed and encouraged
  - Consider posting on arXiv before/during submission for visibility
  - **BENEFIT**: Establishes priority, increases citations if accepted

---

## 8. Post-Acceptance Checklist (Future)

### 8.1 Production Phase

- [ ] Accept proofs or request corrections
- [ ] Provide graphical abstract (optional)
- [ ] Provide author photo for "Behind the Paper" blog (optional)
- [ ] Complete copyright transfer forms
- [ ] Confirm open access license (CC BY default)

### 8.2 Post-Publication

- [ ] Update Zenodo DOI in GitHub README
- [ ] Add publication citation to README
- [ ] Announce on relevant platforms (Twitter/X, Reddit r/QuantumComputing, etc.)
- [ ] Upload to personal/institutional repositories
- [ ] Add to ORCID profile

---

## 9. Critical Path Summary

**Immediate Actions (Before Submission Possible)**:

1. **Reduce abstract to ≤150 words** (currently 169, need to cut 19 words)
2. **Create SourceData.xlsx** with all figure/table data in separate tabs
3. **Upload data to Zenodo** and obtain DOI (can use embargo during review)
4. **Update manuscript with real Zenodo DOI** (replace placeholder)
5. **Count main text words** and verify ≤5,000 word limit
6. **Complete Nature Reporting Summary PDF**
7. **Verify all 5 figures exist** in production-ready format (PDF/EPS/TIFF)
8. **Write cover letter** from template (add real author info, suggested reviewers)
9. **Replace author placeholders** in manuscript with real names/affiliations/email
10. **Test full reproducibility**: Fresh clone → install → run protocol script

**High Priority (Should Complete Before Submission)**:

11. Check all references have DOI links
12. Review SI.tex for completeness (all sections referenced in main text)
13. Verify drop-in API functions exist and are documented
14. Add funding statement (or "no funding" statement)
15. Decide on double-anonymized review (requires repository anonymization if yes)
16. Consider depositing pre-registration protocol on OSF for transparency

**Recommended (Enhances Submission Quality)**:

17. Create Docker/Conda environment file for reproducibility
18. Archive repository snapshot on Zenodo for permanent code DOI
19. Add installation/usage tutorial to README
20. Create editor memo (one-page impact summary)
21. Post preprint to arXiv for visibility and priority

---

## 10. Current Manuscript Status

**✅ COMPLETED**:
- Main manuscript structure complete
- IBM Fez hardware results integrated into Results, Methods, Discussion, Data Availability
- Statistical analysis follows Nature guidelines (cluster-robust, effect sizes, CIs)
- Pre-registered protocol available
- Code and data infrastructure ready
- Licensing in place (MIT for code)

**⚠️ NEEDS ATTENTION**:
- Abstract over word limit by 19 words
- Zenodo DOI placeholder needs replacement with real DOI
- Author information is placeholder format
- SourceData.xlsx not yet created
- Figure files existence not verified
- Main text word count not verified against 5,000 limit
- Nature Reporting Summary not completed
- Cover letter not personalized

**🔍 VERIFICATION NEEDED**:
- SI.tex completeness against all main text references
- All citations present in references section
- Drop-in API functions exist and documented
- Reproducibility from clean environment
- APC funding source confirmed

---

## 11. Estimated Time to Submission-Ready

**Assuming full-time work**:
- Critical path items: **2-3 days**
- High priority items: **1-2 days**
- Recommended enhancements: **2-3 days**

**Total estimate: 5-8 working days to complete submission package**

**Minimum viable submission (critical path only): 2-3 days**

---

## 12. Key Contacts and Resources

**Nature Communications Submission Portal**: https://mts-ncomms.nature.com/

**Author Guidelines**: https://www.nature.com/ncomms/submit

**Reporting Summary Template**: https://www.nature.com/documents/nr-reporting-summary.pdf

**Article Processing Charges**: https://www.nature.com/ncomms/submit/article-processing-charges

**Preprint Policy**: https://www.nature.com/ncomms/submit/preprint-and-publication

**Technical Support**: https://support.nature.com/

---

**Last Updated**: December 10, 2024  
**Next Review**: After completing critical path items

