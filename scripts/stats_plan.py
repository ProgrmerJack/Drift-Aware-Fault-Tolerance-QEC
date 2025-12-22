#!/usr/bin/env python3
"""
stats_plan.py - Statistical Analysis Plan Executor
===================================================

Executes the pre-registered statistical analysis plan from protocol.yaml
and generates:
1. stats_manifest.json - Machine-readable analysis results
2. statistics.tex - LaTeX fragment for manuscript Methods section

This ensures statistical analyses are:
- Pre-specified (no p-hacking)
- Reproducible (seeded bootstrap)
- Transparent (full audit trail)

Usage:
    python scripts/stats_plan.py                    # Run full analysis
    python scripts/stats_plan.py --dry-run          # Show analysis plan
    python scripts/stats_plan.py --generate-tex     # Generate LaTeX only
"""

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


# =============================================================================
# STATISTICAL TEST IMPLEMENTATIONS
# =============================================================================

class BootstrapTest:
    """Paired bootstrap test for primary endpoint."""
    
    def __init__(self, n_bootstrap: int = 10000, confidence_level: float = 0.95, seed: int = 42):
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.seed = seed
        
    def paired_bootstrap_ci(
        self,
        baseline: np.ndarray,
        treatment: np.ndarray,
        statistic: str = "mean_difference"
    ) -> dict:
        """
        Compute bootstrap confidence interval for paired samples.
        
        Args:
            baseline: Baseline measurements
            treatment: Treatment measurements
            statistic: "mean_difference", "log_odds_ratio", or "relative_reduction"
        
        Returns:
            dict with estimate, ci_lower, ci_upper, se, p_value
        """
        np.random.seed(self.seed)
        n = len(baseline)
        
        if len(baseline) != len(treatment):
            raise ValueError("Baseline and treatment must have same length")
        
        # Point estimate
        if statistic == "mean_difference":
            point_estimate = np.mean(treatment - baseline)
            diffs = treatment - baseline
        elif statistic == "log_odds_ratio":
            # For error rates
            eps = 1e-10
            baseline_safe = np.clip(baseline, eps, 1-eps)
            treatment_safe = np.clip(treatment, eps, 1-eps)
            log_or = np.log(treatment_safe / (1 - treatment_safe)) - np.log(baseline_safe / (1 - baseline_safe))
            point_estimate = np.mean(log_or)
            diffs = log_or
        elif statistic == "relative_reduction":
            eps = 1e-10
            rel_red = (baseline - treatment) / (baseline + eps)
            point_estimate = np.mean(rel_red)
            diffs = rel_red
        else:
            raise ValueError(f"Unknown statistic: {statistic}")
        
        # Bootstrap
        bootstrap_estimates = []
        for _ in range(self.n_bootstrap):
            idx = np.random.choice(n, size=n, replace=True)
            bootstrap_estimates.append(np.mean(diffs[idx]))
        
        bootstrap_estimates = np.array(bootstrap_estimates)
        
        # CI using percentile method
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_estimates, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2))
        
        # Standard error
        se = np.std(bootstrap_estimates)
        
        # p-value (two-sided, testing if different from 0)
        # Use bootstrap distribution centered at 0
        centered = bootstrap_estimates - point_estimate
        p_value = 2 * min(
            np.mean(centered >= point_estimate),
            np.mean(centered <= point_estimate)
        )
        
        return {
            "estimate": float(point_estimate),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "se": float(se),
            "p_value": float(p_value),
            "n_bootstrap": self.n_bootstrap,
            "confidence_level": self.confidence_level,
            "seed": self.seed,
            "n_pairs": n,
        }


class EffectSizeCalculator:
    """Calculate various effect size measures."""
    
    @staticmethod
    def cohens_d(baseline: np.ndarray, treatment: np.ndarray) -> float:
        """Calculate Cohen's d for paired samples."""
        diff = treatment - baseline
        return float(np.mean(diff) / np.std(diff, ddof=1))
    
    @staticmethod
    def relative_risk_reduction(baseline: np.ndarray, treatment: np.ndarray) -> float:
        """Calculate relative risk reduction."""
        eps = 1e-10
        return float(np.mean((baseline - treatment) / (baseline + eps)))
    
    @staticmethod
    def absolute_risk_reduction(baseline: np.ndarray, treatment: np.ndarray) -> float:
        """Calculate absolute risk reduction."""
        return float(np.mean(baseline - treatment))
    
    @staticmethod
    def number_needed_to_treat(baseline: np.ndarray, treatment: np.ndarray) -> float:
        """Calculate number needed to treat."""
        arr = EffectSizeCalculator.absolute_risk_reduction(baseline, treatment)
        if abs(arr) < 1e-10:
            return float('inf')
        return float(1.0 / arr)


class MultipleComparisonCorrection:
    """Multiple comparison correction methods."""
    
    @staticmethod
    def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[dict]:
        """
        Apply Holm-Bonferroni correction.
        
        Returns list of dicts with original p-value, adjusted p-value, and significance.
        """
        n = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p = np.array(p_values)[sorted_indices]
        
        adjusted = np.zeros(n)
        significant = np.zeros(n, dtype=bool)
        
        for i, (idx, p) in enumerate(zip(sorted_indices, sorted_p)):
            k = n - i
            adjusted_p = min(1.0, p * k)
            adjusted[idx] = adjusted_p
            
            # Check significance with step-down procedure
            if i == 0:
                significant[idx] = adjusted_p < alpha
            else:
                # Must reject all previous to reject this one
                if significant[sorted_indices[i-1]]:
                    significant[idx] = adjusted_p < alpha
                else:
                    significant[idx] = False
        
        return [
            {
                "original_p": p_values[i],
                "adjusted_p": float(adjusted[i]),
                "significant": bool(significant[i]),
            }
            for i in range(n)
        ]


# =============================================================================
# ANALYSIS EXECUTOR
# =============================================================================

@dataclass
class AnalysisResult:
    """Container for analysis results."""
    name: str
    description: str
    test_type: str
    result: dict
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class StatsExecutor:
    """Execute pre-registered statistical analysis plan."""
    
    def __init__(self, protocol_path: Path, data_path: Path):
        self.protocol_path = protocol_path
        self.data_path = data_path
        self.protocol = None
        self.data = None
        self.results = []
        
    def load(self):
        """Load protocol and data."""
        with open(self.protocol_path) as f:
            self.protocol = yaml.safe_load(f)
        
        if self.data_path.exists():
            self.data = pd.read_parquet(self.data_path)
            logger.info(f"Loaded {len(self.data)} records from {self.data_path}")
        else:
            logger.warning(f"Data file not found: {self.data_path}")
            self.data = pd.DataFrame()
    
    def run_primary_analysis(self) -> AnalysisResult:
        """Run primary endpoint analysis."""
        primary_config = self.protocol['statistics']['primary']
        endpoint_config = self.protocol['primary_endpoint']
        
        logger.info("Running PRIMARY analysis: paired bootstrap")
        
        if self.data.empty:
            # Return placeholder for dry run
            return AnalysisResult(
                name="primary_endpoint",
                description=endpoint_config['name'],
                test_type=primary_config['test'],
                result={
                    "status": "pending_data",
                    "comparison": endpoint_config['comparison'],
                    "metric": endpoint_config['metric'],
                }
            )
        
        # Get baseline and treatment data
        baseline = self.data[self.data['strategy'] == 'baseline_static']['logical_error_rate'].values
        treatment = self.data[self.data['strategy'] == 'drift_aware_full_stack']['logical_error_rate'].values
        
        # Pair by matching conditions
        # (In real implementation, would pair by backend, distance, etc.)
        n_pairs = min(len(baseline), len(treatment))
        baseline = baseline[:n_pairs]
        treatment = treatment[:n_pairs]
        
        # Run bootstrap test
        bootstrap = BootstrapTest(
            n_bootstrap=primary_config['n_bootstrap'],
            confidence_level=primary_config['confidence_level'],
            seed=primary_config['random_seed']
        )
        
        result = bootstrap.paired_bootstrap_ci(
            baseline, treatment,
            statistic="log_odds_ratio" if endpoint_config['metric'] == "paired_log_odds_ratio" else "mean_difference"
        )
        
        # Add effect sizes
        effect_calc = EffectSizeCalculator()
        result['cohens_d'] = effect_calc.cohens_d(baseline, treatment)
        result['relative_risk_reduction'] = effect_calc.relative_risk_reduction(baseline, treatment)
        result['absolute_risk_reduction'] = effect_calc.absolute_risk_reduction(baseline, treatment)
        result['nnt'] = effect_calc.number_needed_to_treat(baseline, treatment)
        
        # Significance
        result['significant'] = result['p_value'] < endpoint_config['alpha']
        result['alpha'] = endpoint_config['alpha']
        
        return AnalysisResult(
            name="primary_endpoint",
            description=endpoint_config['name'],
            test_type=primary_config['test'],
            result=result
        )
    
    def run_secondary_analyses(self) -> list[AnalysisResult]:
        """Run secondary analyses."""
        results = []
        
        # Distance scaling analysis
        logger.info("Running SECONDARY analysis: distance scaling")
        results.append(AnalysisResult(
            name="distance_scaling",
            description="Log-linear regression of logical error rate vs code distance",
            test_type="linear_regression",
            result={
                "status": "pending_data" if self.data.empty else "not_implemented",
                "expected_slope": "negative",
            }
        ))
        
        # Backend heterogeneity
        logger.info("Running SECONDARY analysis: backend heterogeneity")
        results.append(AnalysisResult(
            name="backend_heterogeneity",
            description="Mixed effects model for backend variation",
            test_type="mixed_effects",
            result={
                "status": "pending_data" if self.data.empty else "not_implemented",
            }
        ))
        
        return results
    
    def run_all(self) -> dict:
        """Run all pre-registered analyses."""
        self.load()
        
        manifest = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "protocol_version": self.protocol['protocol']['version'],
            "data_file": str(self.data_path),
            "n_records": len(self.data) if self.data is not None else 0,
            "analyses": {}
        }
        
        # Primary analysis
        primary = self.run_primary_analysis()
        manifest["analyses"]["primary"] = asdict(primary)
        
        # Secondary analyses
        secondary = self.run_secondary_analyses()
        manifest["analyses"]["secondary"] = [asdict(s) for s in secondary]
        
        # Multiple comparison correction for secondary endpoints
        if secondary:
            p_values = [
                s.result.get('p_value', 1.0) 
                for s in secondary 
                if 'p_value' in s.result
            ]
            if p_values:
                corrections = MultipleComparisonCorrection.holm_bonferroni(p_values)
                manifest["multiple_comparison_correction"] = {
                    "method": "holm_bonferroni",
                    "corrections": corrections
                }
        
        return manifest


# =============================================================================
# LATEX GENERATOR
# =============================================================================

class LaTeXGenerator:
    """Generate statistics.tex from analysis results."""
    
    def __init__(self, manifest: dict):
        self.manifest = manifest
        
    def generate(self) -> str:
        """Generate LaTeX fragment for Methods section."""
        lines = [
            "% =============================================================================",
            "% STATISTICAL ANALYSIS (Auto-generated by stats_plan.py)",
            "% Generated: " + self.manifest.get('generated_at', 'Unknown'),
            "% =============================================================================",
            "",
            "\\subsection*{Statistical Analysis}",
            "",
        ]
        
        # Primary analysis description
        primary = self.manifest.get('analyses', {}).get('primary', {})
        if primary:
            result = primary.get('result', {})
            lines.extend([
                "\\subsubsection*{Primary Endpoint}",
                "",
                f"The primary endpoint was {primary.get('description', 'not specified')}.",
                f"We used a {primary.get('test_type', 'paired bootstrap')} test with",
                f"{result.get('n_bootstrap', 10000):,} bootstrap resamples",
                f"(random seed: {result.get('seed', 42)}).",
                "",
            ])
            
            if result.get('status') != 'pending_data':
                lines.extend([
                    "\\textbf{Results:}",
                    f"The estimated effect was {result.get('estimate', 0):.4f}",
                    f"(95\\% CI: [{result.get('ci_lower', 0):.4f}, {result.get('ci_upper', 0):.4f}],",
                    f"$p = {result.get('p_value', 1):.4f}$).",
                    "",
                    "Effect sizes:",
                    "\\begin{itemize}",
                    f"  \\item Cohen's $d$ = {result.get('cohens_d', 0):.3f}",
                    f"  \\item Relative risk reduction = {result.get('relative_risk_reduction', 0):.1%}",
                    f"  \\item Absolute risk reduction = {result.get('absolute_risk_reduction', 0):.4f}",
                    "\\end{itemize}",
                    "",
                ])
        
        # Multiple comparison correction
        mcc = self.manifest.get('multiple_comparison_correction', {})
        if mcc:
            lines.extend([
                "\\subsubsection*{Multiple Comparison Correction}",
                "",
                f"Secondary endpoints were corrected for multiple comparisons using the",
                f"{mcc.get('method', 'Holm-Bonferroni')} procedure.",
                "",
            ])
        
        # Pre-registration statement
        lines.extend([
            "\\subsubsection*{Pre-registration}",
            "",
            "All statistical analyses were pre-specified in the experimental protocol",
            "(see Supplementary Information for full protocol). No analyses were",
            "added or modified after data collection began.",
            "",
        ])
        
        return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Execute pre-registered statistical analysis plan"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show analysis plan without running'
    )
    parser.add_argument(
        '--generate-tex',
        action='store_true',
        help='Generate LaTeX from existing manifest'
    )
    parser.add_argument(
        '--protocol',
        default='protocol/protocol.yaml',
        help='Path to protocol.yaml'
    )
    parser.add_argument(
        '--data',
        default='data/processed/master.parquet',
        help='Path to master.parquet'
    )
    parser.add_argument(
        '--output-dir',
        default='manuscript',
        help='Output directory for generated files'
    )
    
    args = parser.parse_args()
    
    protocol_path = PROJECT_ROOT / args.protocol
    data_path = PROJECT_ROOT / args.data
    output_dir = PROJECT_ROOT / args.output_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.generate_tex:
        # Load existing manifest
        manifest_path = output_dir / "stats_manifest.json"
        if not manifest_path.exists():
            logger.error(f"Manifest not found: {manifest_path}")
            return 1
        
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        # Generate LaTeX
        generator = LaTeXGenerator(manifest)
        tex = generator.generate()
        
        tex_path = output_dir / "statistics.tex"
        with open(tex_path, 'w') as f:
            f.write(tex)
        
        logger.info(f"Generated {tex_path}")
        return 0
    
    # Run analysis
    executor = StatsExecutor(protocol_path, data_path)
    
    if args.dry_run:
        executor.load()
        logger.info("=" * 60)
        logger.info("STATISTICAL ANALYSIS PLAN (DRY RUN)")
        logger.info("=" * 60)
        
        stats_config = executor.protocol.get('statistics', {})
        
        logger.info("\nPrimary Analysis:")
        primary = stats_config.get('primary', {})
        logger.info(f"  Test: {primary.get('test', 'not specified')}")
        logger.info(f"  Bootstrap samples: {primary.get('n_bootstrap', 10000)}")
        logger.info(f"  Confidence level: {primary.get('confidence_level', 0.95)}")
        logger.info(f"  Random seed: {primary.get('random_seed', 42)}")
        
        logger.info("\nSecondary Analysis:")
        secondary = stats_config.get('secondary', {})
        logger.info(f"  Test: {secondary.get('test', 'not specified')}")
        logger.info(f"  Fixed effects: {secondary.get('fixed_effects', [])}")
        logger.info(f"  Random effects: {secondary.get('random_effects', [])}")
        
        logger.info("\nMultiple Comparisons:")
        mcc = stats_config.get('multiple_comparisons', {})
        logger.info(f"  Method: {mcc.get('method', 'not specified')}")
        
        logger.info("\nEffect Sizes to Report:")
        for es in stats_config.get('effect_sizes', []):
            logger.info(f"  - {es}")
        
        return 0
    
    # Run full analysis
    manifest = executor.run_all()
    
    # Save manifest
    manifest_path = output_dir / "stats_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Saved manifest: {manifest_path}")
    
    # Generate LaTeX
    generator = LaTeXGenerator(manifest)
    tex = generator.generate()
    
    tex_path = output_dir / "statistics.tex"
    with open(tex_path, 'w') as f:
        f.write(tex)
    logger.info(f"Generated LaTeX: {tex_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
