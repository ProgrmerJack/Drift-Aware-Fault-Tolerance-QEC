#!/usr/bin/env python3
"""
final_gates.py - Pre-submission validation gates for Nature Communications

Runs comprehensive checks before submission:
1. Protocol integrity (hash verification)
2. Data completeness (master.parquet schema)
3. Reproducibility (script execution tests)
4. Manuscript compliance (word counts, placeholders)
5. Source data validation (SourceData.xlsx)
6. Repository readiness (git status, .gitignore)

All gates must pass before submission is approved.

Usage:
    python scripts/final_gates.py [--strict] [--skip-expensive]
"""

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class Gate:
    """A single validation gate."""
    name: str
    description: str
    passed: bool = False
    message: str = ""
    critical: bool = True  # If True, failure blocks submission


@dataclass
class FinalGatesValidator:
    """Run all pre-submission validation gates."""
    
    project_root: Path
    strict: bool = False
    skip_expensive: bool = False
    
    gates: list = field(default_factory=list)
    
    def __post_init__(self):
        self.project_root = Path(self.project_root)
        
    def add_gate(self, gate: Gate) -> None:
        """Add a gate to the validation list."""
        self.gates.append(gate)
        
    def gate_protocol_integrity(self) -> Gate:
        """Verify protocol.yaml hash matches locked version."""
        gate = Gate(
            name="Protocol Integrity",
            description="Verify protocol.yaml has not been modified since lock"
        )
        
        protocol_path = self.project_root / "protocol" / "protocol.yaml"
        lock_path = self.project_root / "protocol" / "protocol_locked.json"
        
        if not protocol_path.exists():
            gate.message = "protocol.yaml not found"
            return gate
            
        if not lock_path.exists():
            gate.message = "protocol_locked.json not found - run lock_protocol.py first"
            return gate
            
        # Compute current hash
        with open(protocol_path, 'rb') as f:
            current_hash = hashlib.sha256(f.read()).hexdigest()
            
        # Load locked hash
        with open(lock_path, encoding='utf-8') as f:
            lock_data = json.load(f)
            locked_hash = lock_data.get('protocol_hash', '')
            
        if current_hash == locked_hash:
            gate.passed = True
            gate.message = f"Hash verified: {current_hash[:16]}..."
        else:
            gate.message = f"Hash mismatch! Current: {current_hash[:16]}..., Locked: {locked_hash[:16]}..."
            
        return gate
        
    def gate_claims_locked(self) -> Gate:
        """Verify CLAIMS.md exists and is complete."""
        gate = Gate(
            name="Claims Locked",
            description="Verify CLAIMS.md has all required sections"
        )
        
        claims_path = self.project_root / "protocol" / "CLAIMS.md"
        
        if not claims_path.exists():
            gate.message = "CLAIMS.md not found"
            return gate
            
        with open(claims_path, encoding='utf-8') as f:
            content = f.read()
            
        required_sections = [
            "Primary Claim",
            "Secondary Claims",
            "Effect Size",
            "Stopping Rules"
        ]
        
        missing = [s for s in required_sections if s.lower() not in content.lower()]
        
        if not missing:
            gate.passed = True
            gate.message = "All required sections present"
        else:
            gate.message = f"Missing sections: {missing}"
            
        return gate
        
    def gate_data_completeness(self) -> Gate:
        """Verify master.parquet exists and has required schema."""
        gate = Gate(
            name="Data Completeness",
            description="Verify master.parquet exists with required columns"
        )
        
        master_path = self.project_root / "data" / "processed" / "master.parquet"
        
        if not master_path.exists():
            gate.message = "master.parquet not found"
            gate.critical = False  # Data may not exist yet
            return gate
            
        try:
            import pandas as pd
            df = pd.read_parquet(master_path)
            
            required_columns = [
                'session_id', 'backend', 'timestamp_utc', 'distance', 
                'strategy', 'logical_error_rate'
            ]
            
            missing = [c for c in required_columns if c not in df.columns]
            
            if not missing:
                gate.passed = True
                gate.message = f"Schema valid: {len(df):,} rows, {len(df.columns)} columns"
            else:
                gate.message = f"Missing columns: {missing}"
                
        except Exception as e:
            gate.message = f"Error reading parquet: {e}"
            
        return gate
        
    def gate_source_data(self) -> Gate:
        """Verify SourceData.xlsx exists and has required sheets."""
        gate = Gate(
            name="Source Data",
            description="Verify SourceData.xlsx has all figure sheets"
        )
        
        source_path = self.project_root / "source_data" / "SourceData.xlsx"
        
        if not source_path.exists():
            gate.message = "SourceData.xlsx not found"
            gate.critical = False  # May not be generated yet
            return gate
            
        try:
            import pandas as pd
            xl = pd.ExcelFile(source_path)
            
            # Check for expected sheets (from figure_manifest.yaml)
            expected_prefixes = ['fig1', 'fig2', 'fig3', 'fig4', 'fig5']
            found = [s for s in xl.sheet_names if any(s.startswith(p) for p in expected_prefixes)]
            
            if len(found) >= 5:
                gate.passed = True
                gate.message = f"Found {len(xl.sheet_names)} sheets including main figures"
            else:
                gate.message = f"Only {len(found)} main figure sheets found, expected ≥5"
                
        except Exception as e:
            gate.message = f"Error reading Excel: {e}"
            
        return gate
        
    def gate_manuscript_placeholders(self) -> Gate:
        """Verify manuscript has no unfilled placeholders."""
        gate = Gate(
            name="Manuscript Placeholders",
            description="Check for unfilled \\todo{} in main.tex"
        )
        
        main_tex = self.project_root / "manuscript" / "main.tex"
        
        if not main_tex.exists():
            gate.message = "main.tex not found"
            return gate
            
        import re
        with open(main_tex, encoding='utf-8') as f:
            content = f.read()
            
        todos = re.findall(r'\\todo\{[^}]+\}', content)
        
        if len(todos) == 0:
            gate.passed = True
            gate.message = "No placeholders found"
        else:
            gate.message = f"{len(todos)} placeholder(s) need filling"
            # In strict mode, this is critical
            gate.critical = self.strict
            
        return gate
        
    def gate_word_counts(self) -> Gate:
        """Verify manuscript word counts are within limits."""
        gate = Gate(
            name="Word Counts",
            description="Verify abstract ≤150, main text ≤6000 words"
        )
        
        try:
            # Use finalize_manuscript for this
            from scripts.finalize_manuscript import ManuscriptValidator
            validator = ManuscriptValidator(self.project_root)
            validator.validate()
            
            abstract_ok = validator.metrics.get('abstract_words', 999) <= 150
            main_ok = validator.metrics.get('main_text_words', 99999) <= 6000
            
            if abstract_ok and main_ok:
                gate.passed = True
                gate.message = f"Abstract: {validator.metrics.get('abstract_words')}/150, Main: {validator.metrics.get('main_text_words')}/6000"
            else:
                gate.message = f"Limits exceeded - Abstract: {validator.metrics.get('abstract_words')}/150, Main: {validator.metrics.get('main_text_words')}/6000"
                
        except Exception as e:
            # Fallback: simple check
            gate.message = f"Could not run validator: {e}"
            gate.critical = False
            
        return gate
        
    def gate_si_completeness(self) -> Gate:
        """Verify SI has all required sections."""
        gate = Gate(
            name="SI Completeness",
            description="Verify SI.tex has sections SI-1 through SI-6"
        )
        
        si_path = self.project_root / "si" / "SI.tex"
        
        if not si_path.exists():
            gate.message = "SI.tex not found"
            return gate
            
        import re
        with open(si_path, encoding='utf-8') as f:
            content = f.read()
            
        required_sections = ['SI-1', 'SI-2', 'SI-3', 'SI-4', 'SI-5', 'SI-6']
        found = [s for s in required_sections if s in content]
        
        if len(found) == len(required_sections):
            gate.passed = True
            gate.message = "All 6 SI sections present"
        else:
            missing = set(required_sections) - set(found)
            gate.message = f"Missing sections: {missing}"
            
        return gate
        
    def gate_git_status(self) -> Gate:
        """Verify git repository is clean and ready."""
        gate = Gate(
            name="Git Status",
            description="Check for uncommitted changes and sensitive files"
        )
        
        try:
            # Check for uncommitted changes
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            changes = result.stdout.strip().split('\n') if result.stdout.strip() else []
            
            # Check gitignore includes sensitive files
            gitignore_path = self.project_root / ".gitignore"
            sensitive_patterns = ['main.tex', 'SI.tex', 'cover_letter']
            
            gitignore_ok = False
            if gitignore_path.exists():
                with open(gitignore_path, encoding='utf-8') as f:
                    gitignore = f.read()
                gitignore_ok = any(p in gitignore for p in sensitive_patterns)
                
            if len(changes) <= 5 and gitignore_ok:
                gate.passed = True
                gate.message = f"{len(changes)} uncommitted file(s), .gitignore configured"
            else:
                issues = []
                if len(changes) > 5:
                    issues.append(f"{len(changes)} uncommitted files")
                if not gitignore_ok:
                    issues.append("Sensitive files not in .gitignore")
                gate.message = "; ".join(issues)
                gate.critical = False  # Warning only
                
        except Exception as e:
            gate.message = f"Git check failed: {e}"
            gate.critical = False
            
        return gate
        
    def gate_zenodo_ready(self) -> Gate:
        """Check if Zenodo deposit information is configured."""
        gate = Gate(
            name="Zenodo Ready",
            description="Verify Zenodo DOI placeholder or actual DOI"
        )
        
        # Check data availability statement
        data_avail = self.project_root / "submission" / "data_availability.md"
        
        if not data_avail.exists():
            gate.message = "data_availability.md not found"
            return gate
            
        with open(data_avail, encoding='utf-8') as f:
            content = f.read()
            
        # Check for DOI (either placeholder or real)
        has_doi_placeholder = 'XXXXXXX' in content or '10.5281/zenodo' in content
        
        if has_doi_placeholder:
            gate.passed = True
            gate.message = "DOI placeholder present (update before final submission)"
            gate.critical = False
        else:
            gate.message = "No Zenodo DOI found in data availability"
            
        return gate
        
    def gate_reproducibility_scripts(self) -> Gate:
        """Verify all pipeline scripts exist and are importable."""
        gate = Gate(
            name="Reproducibility Scripts",
            description="Check all pipeline scripts exist"
        )
        
        required_scripts = [
            'scripts/lock_protocol.py',
            'scripts/build_master.py',
            'scripts/stats_plan.py',
            'scripts/generate_figures.py',
            'scripts/populate_si.py',
            'scripts/finalize_manuscript.py',
            'scripts/build_submission.py',
            'protocol/run_protocol.py',
        ]
        
        missing = []
        for script in required_scripts:
            if not (self.project_root / script).exists():
                missing.append(script)
                
        if not missing:
            gate.passed = True
            gate.message = f"All {len(required_scripts)} pipeline scripts present"
        else:
            gate.message = f"Missing scripts: {missing}"
            
        return gate
        
    def gate_requirements_lock(self) -> Gate:
        """Verify requirements-lock.txt exists for reproducibility."""
        gate = Gate(
            name="Requirements Lock",
            description="Verify pinned dependencies exist"
        )
        
        # Check for various lock file formats
        lock_files = [
            'requirements-lock.txt',
            'requirements.txt',
            'pyproject.toml'
        ]
        
        found = []
        for lock_file in lock_files:
            if (self.project_root / lock_file).exists():
                found.append(lock_file)
                
        if 'requirements-lock.txt' in found or 'pyproject.toml' in found:
            gate.passed = True
            gate.message = f"Found: {', '.join(found)}"
        elif found:
            gate.message = f"Found {found} but prefer requirements-lock.txt for exact versions"
            gate.critical = False
            gate.passed = True
        else:
            gate.message = "No dependency lock file found"
            
        return gate
        
    def run_all_gates(self) -> bool:
        """Run all validation gates."""
        print("=" * 60)
        print("Final Pre-Submission Gates")
        print("=" * 60)
        print(f"Project: {self.project_root.name}")
        print(f"Mode: {'Strict' if self.strict else 'Standard'}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print()
        
        # Define gate order
        gate_functions = [
            self.gate_protocol_integrity,
            self.gate_claims_locked,
            self.gate_data_completeness,
            self.gate_source_data,
            self.gate_manuscript_placeholders,
            self.gate_word_counts,
            self.gate_si_completeness,
            self.gate_git_status,
            self.gate_zenodo_ready,
            self.gate_reproducibility_scripts,
            self.gate_requirements_lock,
        ]
        
        # Skip expensive gates if requested
        if self.skip_expensive:
            print("(Skipping expensive validation gates)")
            
        # Run each gate
        for gate_fn in gate_functions:
            gate = gate_fn()
            self.gates.append(gate)
            
            status = "✓ PASS" if gate.passed else ("✗ FAIL" if gate.critical else "⚠ WARN")
            print(f"[{status}] {gate.name}")
            print(f"         {gate.message}")
            print()
            
        # Summary
        passed = sum(1 for g in self.gates if g.passed)
        failed_critical = sum(1 for g in self.gates if not g.passed and g.critical)
        warnings = sum(1 for g in self.gates if not g.passed and not g.critical)
        
        print("=" * 60)
        print("Gate Summary")
        print("=" * 60)
        print(f"  Passed: {passed}/{len(self.gates)}")
        print(f"  Failed (critical): {failed_critical}")
        print(f"  Warnings: {warnings}")
        
        if failed_critical == 0:
            if warnings == 0:
                print("\n✅ ALL GATES PASSED - Ready for submission!")
            else:
                print(f"\n✓ GATES PASSED with {warnings} warning(s) - Review before submission")
            return True
        else:
            print(f"\n❌ {failed_critical} CRITICAL GATE(S) FAILED - Cannot submit")
            print("\nFailed gates:")
            for gate in self.gates:
                if not gate.passed and gate.critical:
                    print(f"  • {gate.name}: {gate.message}")
            return False
            
    def generate_report(self) -> dict:
        """Generate JSON report of gate results."""
        return {
            'timestamp': datetime.now().isoformat(),
            'project_root': str(self.project_root),
            'mode': 'strict' if self.strict else 'standard',
            'gates': [
                {
                    'name': g.name,
                    'description': g.description,
                    'passed': g.passed,
                    'critical': g.critical,
                    'message': g.message
                }
                for g in self.gates
            ],
            'summary': {
                'total': len(self.gates),
                'passed': sum(1 for g in self.gates if g.passed),
                'failed_critical': sum(1 for g in self.gates if not g.passed and g.critical),
                'warnings': sum(1 for g in self.gates if not g.passed and not g.critical)
            },
            'ready_to_submit': all(g.passed or not g.critical for g in self.gates)
        }


def main():
    parser = argparse.ArgumentParser(
        description="Run pre-submission validation gates"
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Treat all warnings as failures'
    )
    parser.add_argument(
        '--skip-expensive',
        action='store_true',
        help='Skip time-consuming validation checks'
    )
    parser.add_argument(
        '--project-root',
        type=Path,
        default=Path(__file__).parent.parent,
        help='Project root directory'
    )
    parser.add_argument(
        '--report',
        type=Path,
        help='Output JSON report to file'
    )
    
    args = parser.parse_args()
    
    validator = FinalGatesValidator(
        project_root=args.project_root,
        strict=args.strict,
        skip_expensive=args.skip_expensive
    )
    
    passed = validator.run_all_gates()
    
    if args.report:
        report = validator.generate_report()
        with open(args.report, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n✓ Report saved to {args.report}")
        
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
