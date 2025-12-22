#!/usr/bin/env python3
"""
finalize_manuscript.py - Validate and finalize manuscript for Nature Communications

Performs:
1. Word count validation (≤6,000 words main text, ≤150 words abstract)
2. Placeholder detection (\todo{} patterns)
3. Reference validation
4. Display item count (≤10 figures+tables)
5. Structural compliance check

Nature Communications Article requirements:
- Abstract: ≤150 words, no references
- Main text: ≤6,000 words
- Display items: ≤10 (figures + tables)
- Methods: separate section (not counted in main text limit if in Methods section)
- References: numbered superscripts

Usage:
    python scripts/finalize_manuscript.py [--fix] [--verbose]
"""

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ManuscriptValidator:
    """Validate manuscript against Nature Communications requirements."""
    
    project_root: Path
    verbose: bool = False
    fix_mode: bool = False
    
    # Validation results
    issues: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    
    # Nature Communications limits
    MAX_ABSTRACT_WORDS: int = 150
    MAX_MAIN_TEXT_WORDS: int = 6000
    MAX_DISPLAY_ITEMS: int = 10
    
    def __post_init__(self):
        self.project_root = Path(self.project_root)
        self.manuscript_path = self.project_root / "manuscript" / "main.tex"
        
    def load_manuscript(self) -> str:
        """Load manuscript content."""
        if not self.manuscript_path.exists():
            raise FileNotFoundError(f"Manuscript not found: {self.manuscript_path}")
        
        with open(self.manuscript_path, encoding='utf-8') as f:
            return f.read()
            
    def extract_abstract(self, content: str) -> str:
        """Extract abstract text from LaTeX."""
        # Match abstract environment
        match = re.search(
            r'\\begin\{abstract\}(.*?)\\end\{abstract\}',
            content,
            re.DOTALL
        )
        if match:
            abstract = match.group(1)
            # Remove LaTeX commands
            abstract = self._strip_latex(abstract)
            return abstract.strip()
        return ""
        
    def extract_main_text(self, content: str) -> str:
        """Extract main text (Introduction through Discussion, excluding Methods)."""
        # Find sections
        intro_match = re.search(r'\\section\*?\{Introduction\}', content)
        methods_match = re.search(r'\\section\*?\{Methods\}', content)
        
        if intro_match and methods_match:
            # Main text is from Introduction to Methods
            main_text = content[intro_match.start():methods_match.start()]
        elif intro_match:
            # No Methods section found, use to end
            main_text = content[intro_match.start():]
        else:
            return ""
            
        # Strip LaTeX commands
        return self._strip_latex(main_text)
        
    def extract_methods(self, content: str) -> str:
        """Extract Methods section."""
        methods_match = re.search(r'\\section\*?\{Methods\}', content)
        data_match = re.search(r'\\section\*?\{Data availability\}', content)
        
        if methods_match and data_match:
            methods = content[methods_match.start():data_match.start()]
        elif methods_match:
            methods = content[methods_match.start():]
        else:
            return ""
            
        return self._strip_latex(methods)
        
    def _strip_latex(self, text: str) -> str:
        """Remove LaTeX commands and environments for word counting."""
        # Remove comments
        text = re.sub(r'%.*$', '', text, flags=re.MULTILINE)
        
        # Remove common environments
        text = re.sub(r'\\begin\{(equation|figure|table|verbatim|itemize|enumerate)\*?\}.*?\\end\{\1\*?\}', '', text, flags=re.DOTALL)
        
        # Remove math environments
        text = re.sub(r'\$.*?\$', ' MATH ', text)
        text = re.sub(r'\\\[.*?\\\]', ' MATH ', text, flags=re.DOTALL)
        
        # Remove common commands
        commands_to_remove = [
            r'\\section\*?\{[^}]*\}',
            r'\\subsection\*?\{[^}]*\}',
            r'\\subsubsection\*?\{[^}]*\}',
            r'\\label\{[^}]*\}',
            r'\\ref\{[^}]*\}',
            r'\\cite\{[^}]*\}',
            r'\\citep?\{[^}]*\}',
            r'\\figref\{[^}]*\}',
            r'\\tabref\{[^}]*\}',
            r'\\eqref\{[^}]*\}',
            r'\\textbf\{([^}]*)\}',
            r'\\textit\{([^}]*)\}',
            r'\\emph\{([^}]*)\}',
            r'\\todo\{[^}]*\}',
            r'\\url\{[^}]*\}',
            r'\\item',
            r'\\noindent',
            r'\\vspace\{[^}]*\}',
            r'\\clearpage',
            r'\\newpage',
        ]
        
        for pattern in commands_to_remove:
            text = re.sub(pattern, ' \\1 ' if '(' in pattern else ' ', text)
            
        # Remove remaining backslash commands
        text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', ' ', text)
        text = re.sub(r'\\[a-zA-Z]+', ' ', text)
        
        # Clean up braces and special chars
        text = re.sub(r'[{}\[\]$&%#_^~]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
        
    def count_words(self, text: str) -> int:
        """Count words in cleaned text."""
        words = text.split()
        # Filter out very short tokens and numbers
        words = [w for w in words if len(w) > 1 or w.isalpha()]
        return len(words)
        
    def find_todos(self, content: str) -> list:
        """Find all \todo{} placeholders."""
        todos = re.findall(r'\\todo\{([^}]+)\}', content)
        return todos
        
    def count_display_items(self, content: str) -> dict:
        """Count figures and tables."""
        # Count figure environments
        figures = len(re.findall(r'\\begin\{figure\}', content))
        
        # Count figure legends (more accurate for Nature format)
        figure_legends = len(re.findall(r'\\textbf\{Figure\s+\d+', content))
        
        # Count tables
        tables = len(re.findall(r'\\begin\{table\}', content))
        
        return {
            'figures': max(figures, figure_legends),
            'tables': tables,
            'total': max(figures, figure_legends) + tables
        }
        
    def validate_references(self, content: str) -> dict:
        """Validate reference structure."""
        # Find all \cite commands
        citations = re.findall(r'\\cite[p]?\{([^}]+)\}', content)
        citation_keys = set()
        for cite in citations:
            keys = [k.strip() for k in cite.split(',')]
            citation_keys.update(keys)
            
        # Find bibliography entries
        bib_entries = re.findall(r'\\bibitem\{([^}]+)\}', content)
        bib_keys = set(bib_entries)
        
        # Check for missing/extra
        missing = citation_keys - bib_keys
        unused = bib_keys - citation_keys
        
        return {
            'total_citations': len(citation_keys),
            'total_entries': len(bib_keys),
            'missing': list(missing),
            'unused': list(unused)
        }
        
    def check_structure(self, content: str) -> list:
        """Check required structural elements."""
        required_sections = [
            ('Abstract', r'\\begin\{abstract\}'),
            ('Introduction', r'\\section\*?\{Introduction\}'),
            ('Results', r'\\section\*?\{Results\}'),
            ('Discussion', r'\\section\*?\{Discussion\}'),
            ('Methods', r'\\section\*?\{Methods\}'),
            ('Data availability', r'\\section\*?\{Data availability\}'),
            ('Code availability', r'\\section\*?\{Code availability\}'),
            ('Acknowledgements', r'\\section\*?\{Acknowledgements\}'),
            ('Author contributions', r'\\section\*?\{Author contributions\}'),
            ('Competing interests', r'\\section\*?\{Competing interests\}'),
        ]
        
        missing = []
        for name, pattern in required_sections:
            if not re.search(pattern, content):
                missing.append(name)
                
        return missing
        
    def validate(self) -> bool:
        """Run all validations."""
        print("=" * 60)
        print("Manuscript Validation - Nature Communications")
        print("=" * 60)
        
        # Load manuscript
        content = self.load_manuscript()
        print(f"Loaded: {self.manuscript_path.name}")
        
        # Word counts
        print("\n--- Word Counts ---")
        
        abstract = self.extract_abstract(content)
        abstract_words = self.count_words(abstract)
        self.metrics['abstract_words'] = abstract_words
        
        status = "✓" if abstract_words <= self.MAX_ABSTRACT_WORDS else "✗"
        print(f"  Abstract: {abstract_words}/{self.MAX_ABSTRACT_WORDS} words [{status}]")
        if abstract_words > self.MAX_ABSTRACT_WORDS:
            self.issues.append(f"Abstract exceeds limit: {abstract_words} > {self.MAX_ABSTRACT_WORDS}")
            
        main_text = self.extract_main_text(content)
        main_words = self.count_words(main_text)
        self.metrics['main_text_words'] = main_words
        
        status = "✓" if main_words <= self.MAX_MAIN_TEXT_WORDS else "✗"
        print(f"  Main text: {main_words}/{self.MAX_MAIN_TEXT_WORDS} words [{status}]")
        if main_words > self.MAX_MAIN_TEXT_WORDS:
            self.issues.append(f"Main text exceeds limit: {main_words} > {self.MAX_MAIN_TEXT_WORDS}")
            
        methods = self.extract_methods(content)
        methods_words = self.count_words(methods)
        self.metrics['methods_words'] = methods_words
        print(f"  Methods: {methods_words} words (separate section)")
        
        # Placeholders
        print("\n--- Placeholders ---")
        todos = self.find_todos(content)
        self.metrics['todos'] = len(todos)
        
        status = "✓" if len(todos) == 0 else "⚠"
        print(f"  \\todo{{}} placeholders: {len(todos)} [{status}]")
        
        if todos:
            self.warnings.append(f"{len(todos)} placeholder(s) need filling")
            if self.verbose:
                print("  Unfilled placeholders:")
                for todo in todos[:10]:
                    print(f"    - {todo[:50]}{'...' if len(todo) > 50 else ''}")
                if len(todos) > 10:
                    print(f"    ... and {len(todos) - 10} more")
                    
        # Display items
        print("\n--- Display Items ---")
        display = self.count_display_items(content)
        self.metrics['display_items'] = display
        
        status = "✓" if display['total'] <= self.MAX_DISPLAY_ITEMS else "✗"
        print(f"  Figures: {display['figures']}")
        print(f"  Tables: {display['tables']}")
        print(f"  Total: {display['total']}/{self.MAX_DISPLAY_ITEMS} [{status}]")
        
        if display['total'] > self.MAX_DISPLAY_ITEMS:
            self.issues.append(f"Too many display items: {display['total']} > {self.MAX_DISPLAY_ITEMS}")
            
        # References
        print("\n--- References ---")
        refs = self.validate_references(content)
        self.metrics['references'] = refs
        
        print(f"  Citations: {refs['total_citations']}")
        print(f"  Bibliography entries: {refs['total_entries']}")
        
        if refs['missing']:
            self.warnings.append(f"Missing bibliography entries: {refs['missing']}")
            print(f"  ⚠ Missing entries: {', '.join(refs['missing'][:5])}")
            
        if refs['unused']:
            self.warnings.append(f"Unused bibliography entries: {refs['unused']}")
            if self.verbose:
                print(f"  ⚠ Unused entries: {', '.join(refs['unused'][:5])}")
                
        # Structure
        print("\n--- Structure ---")
        missing_sections = self.check_structure(content)
        
        if missing_sections:
            self.issues.append(f"Missing required sections: {missing_sections}")
            print(f"  ✗ Missing sections: {', '.join(missing_sections)}")
        else:
            print("  ✓ All required sections present")
            
        # Summary
        print("\n" + "=" * 60)
        print("Validation Summary")
        print("=" * 60)
        
        if self.issues:
            print(f"\n❌ ISSUES ({len(self.issues)}):")
            for issue in self.issues:
                print(f"  • {issue}")
                
        if self.warnings:
            print(f"\n⚠ WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  • {warning}")
                
        if not self.issues and not self.warnings:
            print("\n✅ Manuscript passes all checks!")
            return True
        elif not self.issues:
            print("\n✓ Manuscript passes requirements (warnings should be reviewed)")
            return True
        else:
            print("\n✗ Manuscript has issues that must be resolved")
            return False
            
    def generate_report(self) -> dict:
        """Generate validation report."""
        return {
            'manuscript_path': str(self.manuscript_path),
            'metrics': self.metrics,
            'issues': self.issues,
            'warnings': self.warnings,
            'passed': len(self.issues) == 0
        }


@dataclass
class ManuscriptPopulator:
    """Populate manuscript placeholders with computed values."""
    
    project_root: Path
    dry_run: bool = False
    
    def __post_init__(self):
        self.project_root = Path(self.project_root)
        
    def load_stats_data(self) -> dict:
        """Load statistics from stats_manifest.json."""
        import json
        
        stats_path = self.project_root / "results" / "statistics" / "stats_manifest.json"
        if stats_path.exists():
            with open(stats_path) as f:
                return json.load(f)
        return {}
        
    def build_replacements(self, stats: dict) -> dict:
        """Build replacement dictionary from stats."""
        replacements = {}
        
        # Primary endpoint results
        primary = stats.get('primary_endpoint', {})
        if primary:
            replacements['primary_effect'] = f"{primary.get('effect_size', 'XX'):.1f}"
            replacements['primary_ci_lower'] = f"{primary.get('ci_lower', 'lower'):.2f}"
            replacements['primary_ci_upper'] = f"{primary.get('ci_upper', 'upper'):.2f}"
            replacements['primary_pvalue'] = f"{primary.get('p_value', 'val'):.4f}"
            replacements['primary_cohens_d'] = f"{primary.get('cohens_d', 'd'):.2f}"
            
        # Session counts
        data_summary = stats.get('data_summary', {})
        if data_summary:
            replacements['n_sessions'] = str(data_summary.get('n_sessions', 'XX'))
            replacements['n_backends'] = str(data_summary.get('n_backends', 'YY'))
            replacements['n_cal_days'] = str(data_summary.get('n_calibration_days', 'ZZ'))
            
        return replacements
        
    def apply_to_manuscript(self, replacements: dict) -> str:
        """Apply replacements to manuscript."""
        manuscript_path = self.project_root / "manuscript" / "main.tex"
        
        with open(manuscript_path) as f:
            content = f.read()
            
        # Pattern replacements
        for key, value in replacements.items():
            # Replace specific patterns like \todo{XX}% with actual values
            patterns = [
                (rf'\\todo\{{{key}\}}', str(value)),
                (rf'\\todo\{{~{key}\}}', str(value)),
            ]
            for pattern, replacement in patterns:
                content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
                
        return content


def main():
    parser = argparse.ArgumentParser(
        description="Validate manuscript for Nature Communications submission"
    )
    parser.add_argument(
        '--fix',
        action='store_true',
        help='Attempt to auto-populate placeholders from stats data'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed output'
    )
    parser.add_argument(
        '--project-root',
        type=Path,
        default=Path(__file__).parent.parent,
        help='Project root directory'
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate JSON validation report'
    )
    
    args = parser.parse_args()
    
    # Run validation
    validator = ManuscriptValidator(
        project_root=args.project_root,
        verbose=args.verbose
    )
    
    passed = validator.validate()
    
    # Generate report if requested
    if args.report:
        import json
        report = validator.generate_report()
        report_path = args.project_root / "manuscript" / "validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n✓ Report saved to {report_path}")
        
    # Attempt fixes if requested
    if args.fix:
        print("\n--- Attempting auto-population ---")
        populator = ManuscriptPopulator(args.project_root)
        stats = populator.load_stats_data()
        
        if stats:
            replacements = populator.build_replacements(stats)
            print(f"  Built {len(replacements)} replacements from stats data")
            
            if replacements:
                content = populator.apply_to_manuscript(replacements)
                output_path = args.project_root / "manuscript" / "main_populated.tex"
                with open(output_path, 'w') as f:
                    f.write(content)
                print(f"  ✓ Saved to {output_path}")
        else:
            print("  ⚠ No stats data found - run stats_plan.py first")
            
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
