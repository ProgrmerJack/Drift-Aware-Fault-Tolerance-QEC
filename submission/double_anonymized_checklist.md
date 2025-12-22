# Double-Anonymized Review Preparation

Nature Communications offers a double-anonymized peer review option. This document provides the checklist and materials needed to prepare a properly anonymized submission.

## NC Double-Anonymization Checklist

### Manuscript Anonymization

- [ ] **Title page**: Remove author names and affiliations (add separate title page file)
- [ ] **Acknowledgments**: Replace specific names with "[acknowledgment removed for review]"
- [ ] **Author contributions**: Replace names with "Author 1, Author 2, ..." or remove section
- [ ] **Competing interests**: Use generic statement if author-specific
- [ ] **Self-citations**: Replace "we previously showed [ref]" with "Previous work [ref]" using third person
- [ ] **Funding statements**: Remove grant numbers that could identify authors

### Repository/Data Anonymization

- [ ] **GitHub repository**: Create anonymous mirror using Anonymous GitHub (https://anonymous.4open.science/)
- [ ] **Zenodo deposit**: Use embargo or anonymous mode during review
- [ ] **Code comments**: Remove author names, email addresses, institutional affiliations
- [ ] **File paths**: Remove paths containing usernames or institutional identifiers
- [ ] **Git history**: Create fresh repository without commit author information

### Cover Letter

- [ ] Keep author names/affiliations in cover letter (not anonymized - editor-only)
- [ ] Note opt-in to double-anonymized review

## Anonymous Repository Mirror

### Using Anonymous GitHub

1. Go to https://anonymous.4open.science/
2. Enter original repository URL
3. Specify files to include (exclude any with identifying information)
4. Get anonymous URL for submission
5. Note: Anonymous URLs expire; re-create before submission deadline

### Manual Anonymization for Zenodo

1. Create new Zenodo upload with:
   - Author: "Anonymous"
   - Affiliation: "Anonymous Institution"
   - Description: Remove identifying prose
2. Set embargo until review completion
3. After acceptance: update with real authorship

## Self-Citation Transformations

| Original | Anonymized |
|----------|------------|
| "We previously demonstrated [our2023]" | "Previous work demonstrated [our2023]" |
| "Our lab has shown" | "Prior studies have shown" |
| "In our earlier analysis" | "In earlier analysis" |
| "We extend our prior work on X" | "This work extends prior work on X" |

## Files Requiring Attention

### `manuscript/main.tex`
- [ ] Remove `\author{}` block content
- [ ] Anonymize `\affiliation{}` blocks
- [ ] Check `\thanks{}` and `\acknowledgments{}` sections
- [ ] Review all self-citations in Related Work

### `si/SI.tex`
- [ ] Same author/affiliation removal
- [ ] Check figure captions for identifying information
- [ ] Review acknowledgment section

### `submission/cover_letter.tex`
- [ ] Keep author information (editor-only, not anonymized)
- [ ] Add statement: "We opt into double-anonymized peer review"

### `daqec/` Python package
- [ ] Remove author email from `pyproject.toml`
- [ ] Check `__init__.py` for author strings
- [ ] Review docstrings for identifying information

### `README.md`
- [ ] Replace author-specific acknowledgments
- [ ] Use anonymous GitHub URL
- [ ] Remove institutional logos/badges

## Pre-Submission Verification

Before submitting, verify anonymization by:

1. **Text search**: Search all files for author surnames, institutional names, grant numbers
2. **Git blame**: Check that anonymous repo has no identifying commits
3. **URL check**: Verify all links point to anonymized versions
4. **Metadata check**: PDF metadata can contain author information - use `pdftk` to strip

```bash
# Search for potentially identifying strings
grep -r "YourName\|YourInstitution\|grant-number" manuscript/ si/ daqec/

# Strip PDF metadata
pdftk manuscript.pdf dump_data output metadata.txt
# Review metadata.txt for identifying fields
```

## After Review

Upon acceptance (or if transferring to another journal):

1. Update Zenodo deposit with real authorship
2. Replace anonymous GitHub with real repository link
3. Restore acknowledgments section
4. Update self-citations to first person (optional)

---

## Template: Cover Letter Addition

Add to cover letter for double-anonymized review:

```latex
\textbf{Review preference:} We opt into Nature Communications' 
double-anonymized peer review scheme. The manuscript and supplementary 
files have been prepared according to the journal's anonymization guidelines. 
An anonymized repository mirror is available at [anonymous URL].
```
