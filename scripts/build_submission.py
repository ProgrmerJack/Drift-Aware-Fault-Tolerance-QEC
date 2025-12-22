#!/usr/bin/env python3
"""
build_submission.py - Build complete submission package for Nature Communications

Generates:
1. Cover letter (populated from template)
2. Highlights document
3. Suggested/excluded reviewers list
4. Author checklist
5. Submission manifest with all files

Nature Communications submission requirements:
- Main manuscript PDF
- Supplementary Information PDF
- Source Data files
- Cover letter
- Reporting Summary (completed form)

Usage:
    python scripts/build_submission.py [--validate] [--package]
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class SubmissionPackageBuilder:
    """Build complete submission package for journal submission."""
    
    project_root: Path
    output_dir: Optional[Path] = None
    
    # Package contents
    files: dict = field(default_factory=dict)
    checklist: dict = field(default_factory=dict)
    
    def __post_init__(self):
        self.project_root = Path(self.project_root)
        if self.output_dir is None:
            self.output_dir = self.project_root / "submission" / "package"
        self.output_dir = Path(self.output_dir)
        
    def setup_directories(self) -> None:
        """Create output directory structure."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        subdirs = ['manuscript', 'supplementary', 'source_data', 'forms']
        for subdir in subdirs:
            (self.output_dir / subdir).mkdir(exist_ok=True)
            
    def generate_cover_letter(self) -> Path:
        """Generate cover letter from template."""
        template_path = self.project_root / "submission" / "cover_letter.md"
        output_path = self.output_dir / "cover_letter.md"
        
        # Load template
        with open(template_path) as f:
            content = f.read()
            
        # Load stats for population
        stats_path = self.project_root / "results" / "statistics" / "stats_manifest.json"
        stats = {}
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
                
        # Populate placeholders
        replacements = {
            '[DATE]': datetime.now().strftime('%B %d, %Y'),
            '[REPOSITORY_URL]': 'https://github.com/[USER]/Drift-Aware-Fault-Tolerance-QEC',
            '[DOI]': '10.5281/zenodo.XXXXXXX',
        }
        
        # Add stats-based replacements
        primary = stats.get('primary_endpoint', {})
        if primary:
            effect = primary.get('effect_size', 15)
            replacements['≥15%'] = f'≥{effect:.0f}%'
            
        for placeholder, value in replacements.items():
            content = content.replace(placeholder, value)
            
        # Save
        with open(output_path, 'w') as f:
            f.write(content)
            
        self.files['cover_letter'] = output_path
        return output_path
        
    def generate_highlights(self) -> Path:
        """Generate research highlights for editors."""
        output_path = self.output_dir / "highlights.md"
        
        highlights = """# Research Highlights

> Bullet points summarizing key findings (for editors)

## Key Findings

1. **Calibration drift invalidates static qubit selection**: Qubit properties on cloud quantum processors drift significantly within 4-hour windows, causing ranking instability that undermines static error correction strategies.

2. **Syndrome streams show non-iid structure**: We demonstrate that syndrome measurements exhibit temporal bursts (Fano factor >> 1) and spatial correlations inconsistent with standard decoder assumptions.

3. **Lightweight probes enable real-time adaptation**: Our 30-shot probe protocol refreshes qubit rankings using only ~10 seconds of QPU time, compatible with open-access allocation limits.

4. **Drift-aware pipeline reduces logical error rates**: The combined approach of probe-informed selection and adaptive-prior decoding achieves significant error rate reductions across multiple backends.

## Novelty

- First systematic quantification of drift impact on QEC performance metrics
- Practical solution compatible with resource-constrained cloud access
- Pre-registered protocol with complete reproducibility package

## Impact

This work addresses a practical barrier to quantum error correction that affects all cloud-accessible quantum computing: the gap between vendor calibration cycles and actual device behavior. Our solution enables meaningful fault-tolerance experiments for researchers without dedicated hardware access.

## Target Audience

- Quantum computing researchers
- Quantum error correction specialists
- Cloud quantum computing users
- Fault-tolerant quantum computing roadmap planners
"""
        
        with open(output_path, 'w') as f:
            f.write(highlights)
            
        self.files['highlights'] = output_path
        return output_path
        
    def generate_reviewer_suggestions(self) -> Path:
        """Generate suggested and excluded reviewers list."""
        output_path = self.output_dir / "reviewers.md"
        
        # NOTE: These are placeholder entries - must be replaced with actual names
        reviewers = """# Reviewer Suggestions

> **IMPORTANT**: Replace all placeholder names with actual researchers before submission.
> Ensure no conflicts of interest exist with suggested reviewers.

## Suggested Reviewers

We suggest the following experts who can provide authoritative evaluation:

### 1. [REVIEWER NAME 1]
- **Institution**: [University/Institution]
- **Email**: [email@institution.edu]
- **Expertise**: Surface code implementations, fault-tolerant quantum computing
- **Relevant publications**: [Recent relevant paper]
- **No conflicts**: We confirm no collaborative or personal relationship.

### 2. [REVIEWER NAME 2]
- **Institution**: [University/Institution]
- **Email**: [email@institution.edu]
- **Expertise**: Superconducting qubit characterization, noise spectroscopy
- **Relevant publications**: [Recent relevant paper]
- **No conflicts**: We confirm no collaborative or personal relationship.

### 3. [REVIEWER NAME 3]
- **Institution**: [University/Institution]
- **Email**: [email@institution.edu]
- **Expertise**: Quantum decoder algorithms, MWPM optimization
- **Relevant publications**: [Recent relevant paper]
- **No conflicts**: We confirm no collaborative or personal relationship.

### 4. [REVIEWER NAME 4]
- **Institution**: [University/Institution]
- **Email**: [email@institution.edu]
- **Expertise**: Quantum device drift, benchmarking protocols
- **Relevant publications**: [Recent relevant paper]
- **No conflicts**: We confirm no collaborative or personal relationship.

### 5. [REVIEWER NAME 5]
- **Institution**: [University/Institution]
- **Email**: [email@institution.edu]
- **Expertise**: Cloud quantum computing, reproducibility in quantum experiments
- **Relevant publications**: [Recent relevant paper]
- **No conflicts**: We confirm no collaborative or personal relationship.

---

## Excluded Reviewers

We request that the following individuals not be invited to review:

### 1. [EXCLUDED NAME]
- **Institution**: [Institution]
- **Reason**: [Brief reason, e.g., "Direct competitor working on similar approach" or "Recent co-author on related work"]

---

## Notes for Editor

- All suggested reviewers have published in the areas of quantum error correction or quantum device characterization within the past 3 years.
- We have verified no co-authorship, institutional overlap, or personal relationships with suggested reviewers.
- Excluded reviewer request is based on potential competitive conflict, not personal issues.
"""
        
        with open(output_path, 'w') as f:
            f.write(reviewers)
            
        self.files['reviewers'] = output_path
        return output_path
        
    def generate_author_checklist(self) -> Path:
        """Generate submission checklist."""
        output_path = self.output_dir / "submission_checklist.md"
        
        checklist = """# Nature Communications Submission Checklist

## Manuscript Components

### Main Manuscript
- [ ] Title page with all author names and affiliations
- [ ] Abstract (≤150 words, no references)
- [ ] Main text (Introduction, Results, Discussion) ≤6,000 words
- [ ] Methods section
- [ ] Data availability statement
- [ ] Code availability statement
- [ ] Acknowledgements
- [ ] Author contributions
- [ ] Competing interests declaration
- [ ] References
- [ ] Figure legends

### Supplementary Information
- [ ] SI sections SI-1 through SI-6 complete
- [ ] All \todo{} placeholders filled
- [ ] SI figures generated and included
- [ ] SI tables with actual data

### Source Data
- [ ] SourceData.xlsx with all figure data
- [ ] Individual panel sheets labeled correctly
- [ ] Data formats documented

### Forms
- [ ] Reporting Summary completed (downloaded from Nature)
- [ ] Cover letter finalized
- [ ] Reviewer suggestions list complete
- [ ] Competing interests disclosed

---

## Quality Checks

### Scientific
- [ ] All claims supported by data shown
- [ ] Statistics correctly computed and reported
- [ ] Effect sizes with confidence intervals for all comparisons
- [ ] Negative results transparently reported

### Reproducibility
- [ ] Code repository public and accessible
- [ ] Zenodo deposit complete with DOI
- [ ] Protocol.yaml frozen and hashed
- [ ] requirements-lock.txt with exact versions
- [ ] All random seeds documented

### Compliance
- [ ] Word counts within limits
- [ ] Display items ≤10
- [ ] References complete and formatted
- [ ] No self-plagiarism or duplicate publication
- [ ] All data ethically obtained (N/A for this study)

---

## File Inventory

| File | Status | Location |
|------|--------|----------|
| main.tex | [ ] Ready | manuscript/ |
| main.pdf | [ ] Compiled | submission/package/ |
| SI.tex | [ ] Ready | si/ |
| SI.pdf | [ ] Compiled | submission/package/ |
| SourceData.xlsx | [ ] Generated | source_data/ |
| cover_letter.md | [ ] Finalized | submission/package/ |
| reporting_summary.pdf | [ ] Completed | submission/package/forms/ |
| reviewers.md | [ ] Finalized | submission/package/ |

---

## Pre-Submission Verification

Run these commands to verify:

```bash
# Validate manuscript
python scripts/finalize_manuscript.py --verbose

# Validate SI completeness
python scripts/populate_si.py --validate

# Generate all figures
python scripts/generate_figures.py --validate

# Build submission package
python scripts/build_submission.py --validate
```

---

## Submission Portal Steps

1. Log in to Nature Communications submission system
2. Start new submission
3. Select Article type
4. Upload main manuscript PDF
5. Upload Supplementary Information PDF
6. Upload Source Data files
7. Complete online forms
8. Upload Reporting Summary PDF
9. Enter suggested reviewers
10. Submit cover letter
11. Review all entries
12. Submit

---

## Post-Submission

- [ ] Record submission ID
- [ ] Archive complete submission package
- [ ] Note any queries from editorial system
- [ ] Prepare for potential desk-reject response (fast turnaround needed)

---

*Checklist generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        with open(output_path, 'w') as f:
            f.write(checklist)
            
        self.files['checklist'] = output_path
        return output_path
        
    def compile_pdfs(self) -> dict:
        """Compile LaTeX files to PDF."""
        results = {}
        
        # Main manuscript
        main_tex = self.project_root / "manuscript" / "main.tex"
        if main_tex.exists():
            pdf_path = self._compile_latex(main_tex, self.output_dir / "manuscript")
            if pdf_path:
                results['main_pdf'] = pdf_path
                self.files['main_pdf'] = pdf_path
            else:
                print("  ⚠ Failed to compile main.tex")
                
        # SI
        si_tex = self.project_root / "si" / "SI.tex"
        if si_tex.exists():
            pdf_path = self._compile_latex(si_tex, self.output_dir / "supplementary")
            if pdf_path:
                results['si_pdf'] = pdf_path
                self.files['si_pdf'] = pdf_path
            else:
                print("  ⚠ Failed to compile SI.tex")
                
        return results
        
    def _compile_latex(self, tex_path: Path, output_dir: Path) -> Optional[Path]:
        """Compile a single LaTeX file to PDF."""
        try:
            # Run pdflatex twice for references
            for _ in range(2):
                result = subprocess.run(
                    ['pdflatex', '-interaction=nonstopmode', '-output-directory', str(output_dir), str(tex_path)],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                
            pdf_name = tex_path.stem + '.pdf'
            pdf_path = output_dir / pdf_name
            
            if pdf_path.exists():
                return pdf_path
            return None
            
        except FileNotFoundError:
            print("  ⚠ pdflatex not found - skipping PDF compilation")
            return None
        except subprocess.TimeoutExpired:
            print("  ⚠ LaTeX compilation timed out")
            return None
        except Exception as e:
            print(f"  ⚠ LaTeX compilation error: {e}")
            return None
            
    def copy_source_data(self) -> list:
        """Copy source data files to package."""
        source_data_dir = self.project_root / "source_data"
        dest_dir = self.output_dir / "source_data"
        
        copied = []
        if source_data_dir.exists():
            for file in source_data_dir.glob("*"):
                if file.is_file():
                    dest = dest_dir / file.name
                    shutil.copy2(file, dest)
                    copied.append(dest)
                    self.files[f'source_data_{file.name}'] = dest
                    
        return copied
        
    def generate_manifest(self) -> Path:
        """Generate submission manifest with checksums."""
        manifest = {
            'generated': datetime.now().isoformat(),
            'project': 'Drift-Aware-Fault-Tolerance-QEC',
            'target_journal': 'Nature Communications',
            'files': {}
        }
        
        for name, path in self.files.items():
            path = Path(path)
            if path.exists():
                # Compute checksum
                sha256 = hashlib.sha256()
                with open(path, 'rb') as f:
                    for chunk in iter(lambda: f.read(8192), b''):
                        sha256.update(chunk)
                        
                manifest['files'][name] = {
                    'path': str(path.relative_to(self.output_dir) if path.is_relative_to(self.output_dir) else path),
                    'size_bytes': path.stat().st_size,
                    'sha256': sha256.hexdigest()
                }
                
        manifest_path = self.output_dir / "submission_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
            
        return manifest_path
        
    def validate_package(self) -> bool:
        """Validate completeness of submission package."""
        print("\n--- Package Validation ---")
        
        required_files = {
            'main_manuscript': self.project_root / "manuscript" / "main.tex",
            'si': self.project_root / "si" / "SI.tex",
            'cover_letter': self.output_dir / "cover_letter.md",
            'reporting_summary': self.project_root / "submission" / "reporting_summary.md",
            'source_data': self.project_root / "source_data" / "SourceData.xlsx",
        }
        
        issues = []
        for name, path in required_files.items():
            if path.exists():
                print(f"  ✓ {name}: {path.name}")
            else:
                print(f"  ✗ {name}: MISSING ({path})")
                issues.append(f"Missing: {name}")
                
        if issues:
            print(f"\n⚠ Package incomplete: {len(issues)} issue(s)")
            return False
        else:
            print("\n✓ Package validation passed")
            return True
            
    def build(self) -> bool:
        """Build complete submission package."""
        print("=" * 60)
        print("Building Submission Package")
        print("=" * 60)
        
        # Setup
        self.setup_directories()
        print(f"Output directory: {self.output_dir}")
        
        # Generate documents
        print("\n--- Generating Documents ---")
        
        cover = self.generate_cover_letter()
        print(f"  ✓ Cover letter: {cover.name}")
        
        highlights = self.generate_highlights()
        print(f"  ✓ Highlights: {highlights.name}")
        
        reviewers = self.generate_reviewer_suggestions()
        print(f"  ✓ Reviewer suggestions: {reviewers.name}")
        
        checklist = self.generate_author_checklist()
        print(f"  ✓ Submission checklist: {checklist.name}")
        
        # Copy source data
        print("\n--- Copying Source Data ---")
        copied = self.copy_source_data()
        if copied:
            print(f"  ✓ Copied {len(copied)} source data file(s)")
        else:
            print("  ⚠ No source data files found")
            
        # Compile PDFs (optional, may not have LaTeX installed)
        print("\n--- Compiling PDFs ---")
        pdfs = self.compile_pdfs()
        if pdfs:
            for name, path in pdfs.items():
                print(f"  ✓ {name}: {path.name}")
        else:
            print("  ⚠ PDF compilation skipped (LaTeX not available)")
            
        # Generate manifest
        print("\n--- Generating Manifest ---")
        manifest = self.generate_manifest()
        print(f"  ✓ Manifest: {manifest.name}")
        
        # Validate
        valid = self.validate_package()
        
        # Summary
        print("\n" + "=" * 60)
        print("Package Build Summary")
        print("=" * 60)
        print(f"  Total files: {len(self.files)}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Manifest: {manifest}")
        
        if valid:
            print("\n✓ Submission package ready for review")
        else:
            print("\n⚠ Package needs attention before submission")
            
        return valid


def main():
    parser = argparse.ArgumentParser(
        description="Build submission package for Nature Communications"
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Only validate existing package'
    )
    parser.add_argument(
        '--package',
        action='store_true',
        help='Create ZIP archive of package'
    )
    parser.add_argument(
        '--project-root',
        type=Path,
        default=Path(__file__).parent.parent,
        help='Project root directory'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Output directory for package'
    )
    
    args = parser.parse_args()
    
    builder = SubmissionPackageBuilder(
        project_root=args.project_root,
        output_dir=args.output_dir
    )
    
    if args.validate:
        valid = builder.validate_package()
        sys.exit(0 if valid else 1)
        
    success = builder.build()
    
    if args.package and success:
        # Create ZIP archive
        import zipfile
        
        zip_path = builder.output_dir.parent / f"submission_package_{datetime.now().strftime('%Y%m%d')}.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file in builder.output_dir.rglob('*'):
                if file.is_file():
                    zf.write(file, file.relative_to(builder.output_dir))
                    
        print(f"\n✓ Package archived: {zip_path}")
        
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
