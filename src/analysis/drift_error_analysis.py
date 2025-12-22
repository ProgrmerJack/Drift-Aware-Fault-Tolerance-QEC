"""
Analysis Module: Drift-Error Correlation Analysis
=================================================

Analyzes correlations between calibration drift and QEC error rates.
Implements statistical analysis for the A/B testing framework.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.optimize import curve_fit

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DriftErrorAnalyzer:
    """
    Analyzes correlations between calibration drift and QEC performance.
    
    Key analyses:
    - Correlation between drift features and logical error rates
    - Change-point impact on QEC metrics
    - Cross-qubit error correlation (crosstalk)
    """
    
    def __init__(self):
        """Initialize the analyzer."""
        self.correlation_methods = ["pearson", "spearman", "kendall"]
        
    def compute_drift_error_correlation(self,
                                         drift_features: pd.DataFrame,
                                         error_rates: pd.DataFrame,
                                         method: str = "spearman") -> Dict[str, Any]:
        """
        Compute correlation between drift features and error rates.
        
        Args:
            drift_features: DataFrame with drift features (variance, z-scores, etc.)
            error_rates: DataFrame with logical error rates
            method: Correlation method ("pearson", "spearman", "kendall")
            
        Returns:
            Dictionary with correlation results and p-values
        """
        results = {
            "method": method,
            "timestamp": datetime.now().isoformat(),
            "correlations": {},
            "significant_correlations": []
        }
        
        # Align dataframes by timestamp/qubit
        common_idx = drift_features.index.intersection(error_rates.index)
        if len(common_idx) == 0:
            logger.warning("No common indices between drift features and error rates")
            return results
            
        drift_aligned = drift_features.loc[common_idx]
        error_aligned = error_rates.loc[common_idx]
        
        # Compute correlations for each feature-error pair
        for drift_col in drift_aligned.columns:
            for error_col in error_aligned.columns:
                drift_vals = drift_aligned[drift_col].dropna()
                error_vals = error_aligned[error_col].loc[drift_vals.index]
                
                if len(drift_vals) < 5:
                    continue
                    
                # Compute correlation
                if method == "pearson":
                    corr, pval = sp_stats.pearsonr(drift_vals, error_vals)
                elif method == "spearman":
                    corr, pval = sp_stats.spearmanr(drift_vals, error_vals)
                else:  # kendall
                    corr, pval = sp_stats.kendalltau(drift_vals, error_vals)
                    
                results["correlations"][f"{drift_col}_vs_{error_col}"] = {
                    "correlation": corr,
                    "p_value": pval,
                    "n_samples": len(drift_vals)
                }
                
                # Flag significant correlations
                if pval < 0.05:
                    results["significant_correlations"].append({
                        "feature": drift_col,
                        "error_metric": error_col,
                        "correlation": corr,
                        "p_value": pval
                    })
                    
        return results
    
    def analyze_change_point_impact(self,
                                     change_points: List[datetime],
                                     error_time_series: pd.Series,
                                     window_size: int = 3) -> Dict[str, Any]:
        """
        Analyze impact of calibration change points on error rates.
        
        Compares error rates before and after each change point.
        
        Args:
            change_points: List of change point timestamps
            error_time_series: Time series of error rates
            window_size: Number of samples to compare before/after
            
        Returns:
            Analysis results for each change point
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "change_point_impacts": [],
            "summary": {}
        }
        
        error_increases = []
        
        for cp in change_points:
            # Find nearest index
            if cp not in error_time_series.index:
                nearest_idx = error_time_series.index.get_indexer([cp], method="nearest")[0]
            else:
                nearest_idx = error_time_series.index.get_loc(cp)
                
            # Get before and after windows
            start_before = max(0, nearest_idx - window_size)
            end_after = min(len(error_time_series), nearest_idx + window_size + 1)
            
            before_vals = error_time_series.iloc[start_before:nearest_idx]
            after_vals = error_time_series.iloc[nearest_idx:end_after]
            
            if len(before_vals) == 0 or len(after_vals) == 0:
                continue
                
            mean_before = before_vals.mean()
            mean_after = after_vals.mean()
            
            # Statistical test for difference
            if len(before_vals) >= 2 and len(after_vals) >= 2:
                t_stat, p_value = sp_stats.ttest_ind(before_vals, after_vals)
            else:
                t_stat, p_value = np.nan, np.nan
                
            impact = {
                "change_point": cp.isoformat() if hasattr(cp, 'isoformat') else str(cp),
                "mean_before": mean_before,
                "mean_after": mean_after,
                "relative_change": (mean_after - mean_before) / (mean_before + 1e-10),
                "t_statistic": t_stat,
                "p_value": p_value
            }
            
            results["change_point_impacts"].append(impact)
            error_increases.append(impact["relative_change"])
            
        # Summary statistics
        if error_increases:
            results["summary"] = {
                "num_change_points": len(change_points),
                "num_analyzed": len(error_increases),
                "mean_relative_change": np.mean(error_increases),
                "median_relative_change": np.median(error_increases),
                "max_increase": max(error_increases),
                "significant_impacts": sum(1 for imp in results["change_point_impacts"] 
                                           if imp["p_value"] < 0.05)
            }
            
        return results
    
    def compute_crosstalk_correlation(self,
                                       qubit_error_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze cross-qubit error correlations as crosstalk proxy.
        
        Args:
            qubit_error_df: DataFrame with columns = qubits, rows = time/experiments
            
        Returns:
            Correlation matrix and significant pairs
        """
        # Compute correlation matrix
        corr_matrix = qubit_error_df.corr()
        
        # Find significant correlations
        significant_pairs = []
        n_samples = len(qubit_error_df)
        
        # Bonferroni-corrected threshold
        n_comparisons = len(corr_matrix.columns) * (len(corr_matrix.columns) - 1) // 2
        alpha_corrected = 0.05 / n_comparisons if n_comparisons > 0 else 0.05
        
        for i, col1 in enumerate(corr_matrix.columns):
            for col2 in corr_matrix.columns[i+1:]:
                corr_val = corr_matrix.loc[col1, col2]
                
                # Compute p-value for correlation
                if n_samples > 2:
                    t_stat = corr_val * np.sqrt((n_samples - 2) / (1 - corr_val**2 + 1e-10))
                    p_value = 2 * (1 - sp_stats.t.cdf(abs(t_stat), n_samples - 2))
                else:
                    p_value = 1.0
                    
                if p_value < alpha_corrected:
                    significant_pairs.append({
                        "qubit_pair": (col1, col2),
                        "correlation": corr_val,
                        "p_value": p_value
                    })
                    
        return {
            "correlation_matrix": corr_matrix.to_dict(),
            "significant_pairs": significant_pairs,
            "alpha_corrected": alpha_corrected,
            "n_comparisons": n_comparisons
        }


class ABTestFramework:
    """
    A/B testing framework for comparing qubit selection strategies.
    
    Compares:
    - Static (backend properties only)
    - RT (real-time probes)
    - Drift-aware (probes + drift analysis)
    """
    
    def __init__(self, alpha: float = 0.05, power: float = 0.8):
        """
        Initialize A/B test framework.
        
        Args:
            alpha: Significance level
            power: Desired statistical power
        """
        self.alpha = alpha
        self.power = power
        
    def compute_sample_size(self, effect_size: float, 
                            baseline_error: float = 0.1) -> int:
        """
        Compute required sample size for detecting a given effect.
        
        Args:
            effect_size: Expected relative improvement (e.g., 0.1 = 10% reduction)
            baseline_error: Baseline logical error rate
            
        Returns:
            Required number of experiments per condition
        """
        # Cohen's h for proportions
        p1 = baseline_error
        p2 = baseline_error * (1 - effect_size)
        
        # Arcsine transformation
        phi1 = 2 * np.arcsin(np.sqrt(p1))
        phi2 = 2 * np.arcsin(np.sqrt(p2))
        h = abs(phi1 - phi2)
        
        if h < 0.01:
            return 10000  # Very small effect, need many samples
            
        # Sample size formula for two proportions
        z_alpha = sp_stats.norm.ppf(1 - self.alpha / 2)
        z_beta = sp_stats.norm.ppf(self.power)
        
        n = 2 * ((z_alpha + z_beta) / h) ** 2
        
        return int(np.ceil(n))
    
    def run_ab_test(self,
                    results_a: List[float],
                    results_b: List[float],
                    test_type: str = "two_sided") -> Dict[str, Any]:
        """
        Run A/B test comparing two selection strategies.
        
        Args:
            results_a: Error rates from strategy A (e.g., static)
            results_b: Error rates from strategy B (e.g., drift-aware)
            test_type: "two_sided", "greater", or "less"
            
        Returns:
            Test results including p-value, effect size, confidence interval
        """
        results_a = np.array(results_a)
        results_b = np.array(results_b)
        
        n_a, n_b = len(results_a), len(results_b)
        mean_a, mean_b = results_a.mean(), results_b.mean()
        std_a, std_b = results_a.std(ddof=1), results_b.std(ddof=1)
        
        # Two-sample t-test
        if test_type == "two_sided":
            alternative = "two-sided"
        elif test_type == "greater":
            alternative = "greater"
        else:
            alternative = "less"
            
        t_stat, p_value = sp_stats.ttest_ind(results_a, results_b)
        
        # Mann-Whitney U test (non-parametric alternative)
        u_stat, u_pvalue = sp_stats.mannwhitneyu(results_a, results_b, 
                                                   alternative='two-sided')
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / 
                            (n_a + n_b - 2))
        cohens_d = (mean_a - mean_b) / (pooled_std + 1e-10)
        
        # Confidence interval for difference
        se_diff = np.sqrt(std_a**2/n_a + std_b**2/n_b)
        t_crit = sp_stats.t.ppf(1 - self.alpha/2, n_a + n_b - 2)
        ci_lower = (mean_a - mean_b) - t_crit * se_diff
        ci_upper = (mean_a - mean_b) + t_crit * se_diff
        
        return {
            "strategy_a": {
                "n": n_a,
                "mean": mean_a,
                "std": std_a
            },
            "strategy_b": {
                "n": n_b,
                "mean": mean_b,
                "std": std_b
            },
            "t_test": {
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant": p_value < self.alpha
            },
            "mann_whitney": {
                "u_statistic": u_stat,
                "p_value": u_pvalue,
                "significant": u_pvalue < self.alpha
            },
            "effect_size": {
                "cohens_d": cohens_d,
                "interpretation": self._interpret_cohens_d(cohens_d)
            },
            "confidence_interval": {
                "lower": ci_lower,
                "upper": ci_upper,
                "level": 1 - self.alpha
            },
            "relative_improvement": (mean_a - mean_b) / (mean_a + 1e-10) * 100
        }
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        d = abs(d)
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"
    
    def run_three_way_comparison(self,
                                  static_results: List[float],
                                  rt_results: List[float],
                                  drift_results: List[float]) -> Dict[str, Any]:
        """
        Compare all three selection strategies.
        
        Args:
            static_results: Error rates with static selection
            rt_results: Error rates with RT selection
            drift_results: Error rates with drift-aware selection
            
        Returns:
            Complete comparison with pairwise tests and ANOVA
        """
        # One-way ANOVA
        f_stat, anova_p = sp_stats.f_oneway(static_results, rt_results, drift_results)
        
        # Kruskal-Wallis (non-parametric)
        h_stat, kw_p = sp_stats.kruskal(static_results, rt_results, drift_results)
        
        # Pairwise comparisons
        pairwise = {
            "static_vs_rt": self.run_ab_test(static_results, rt_results),
            "static_vs_drift": self.run_ab_test(static_results, drift_results),
            "rt_vs_drift": self.run_ab_test(rt_results, drift_results)
        }
        
        # Summary statistics
        summary = {
            "static": {
                "mean": np.mean(static_results),
                "std": np.std(static_results, ddof=1),
                "median": np.median(static_results)
            },
            "rt": {
                "mean": np.mean(rt_results),
                "std": np.std(rt_results, ddof=1),
                "median": np.median(rt_results)
            },
            "drift_aware": {
                "mean": np.mean(drift_results),
                "std": np.std(drift_results, ddof=1),
                "median": np.median(drift_results)
            }
        }
        
        # Determine best strategy
        means = {
            "static": summary["static"]["mean"],
            "rt": summary["rt"]["mean"],
            "drift_aware": summary["drift_aware"]["mean"]
        }
        best_strategy = min(means, key=means.get)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "anova": {
                "f_statistic": f_stat,
                "p_value": anova_p,
                "significant": anova_p < self.alpha
            },
            "kruskal_wallis": {
                "h_statistic": h_stat,
                "p_value": kw_p,
                "significant": kw_p < self.alpha
            },
            "pairwise_comparisons": pairwise,
            "summary_statistics": summary,
            "best_strategy": best_strategy,
            "best_mean_error": means[best_strategy]
        }


class LogicalErrorRateModel:
    """
    Models logical error rate scaling with code distance and rounds.
    
    Used to estimate error thresholds and predict performance.
    """
    
    def __init__(self):
        """Initialize the model."""
        pass
    
    def fit_threshold(self, distances: List[int],
                      error_rates: Dict[int, float]) -> Dict[str, Any]:
        """
        Fit error threshold model to experimental data.
        
        The logical error rate should scale as:
        p_L ∝ (p/p_th)^((d+1)/2)
        
        Args:
            distances: Code distances tested
            error_rates: Dictionary mapping distance to logical error rate
            
        Returns:
            Fitted parameters including threshold estimate
        """
        # Prepare data
        d_vals = []
        p_l_vals = []
        
        for d in distances:
            if d in error_rates:
                d_vals.append(d)
                p_l_vals.append(error_rates[d])
                
        if len(d_vals) < 2:
            return {"error": "Insufficient data for threshold fit"}
            
        d_arr = np.array(d_vals)
        p_l_arr = np.array(p_l_vals)
        
        # Log-linear fit: log(p_L) = a + b * d
        # This is approximate; exact scaling is more complex
        try:
            log_p = np.log(p_l_arr + 1e-10)
            coeffs = np.polyfit(d_arr, log_p, 1)
            
            a, b = coeffs[0], coeffs[1]
            
            # Estimate threshold from slope
            # If p < p_th, increasing d should decrease p_L
            # Slope b = ((d+1)/2) * log(p/p_th)
            
            # This is a simplified estimate
            if a < 0:
                # Error rate decreases with distance - below threshold
                below_threshold = True
                estimated_p_th = None
            else:
                below_threshold = False
                estimated_p_th = np.exp(-b / a) if a != 0 else None
                
            return {
                "distances_used": d_vals,
                "error_rates_used": p_l_vals,
                "slope": a,
                "intercept": b,
                "below_threshold": below_threshold,
                "estimated_threshold": estimated_p_th,
                "r_squared": self._compute_r_squared(d_arr, log_p, coeffs)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _compute_r_squared(self, x: np.ndarray, y: np.ndarray, 
                           coeffs: np.ndarray) -> float:
        """Compute R-squared for fit quality."""
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - ss_res / (ss_tot + 1e-10)
    
    def predict_error_rate(self, distance: int, 
                           fit_params: Dict[str, Any]) -> float:
        """
        Predict logical error rate for a given distance.
        
        Args:
            distance: Code distance
            fit_params: Parameters from fit_threshold()
            
        Returns:
            Predicted logical error rate
        """
        if "error" in fit_params:
            return np.nan
            
        a = fit_params["slope"]
        b = fit_params["intercept"]
        
        log_p = a * distance + b
        return np.exp(log_p)


if __name__ == "__main__":
    print("Analysis Module")
    print("\nExample usage:")
    print("""
    # A/B Testing
    ab_test = ABTestFramework()
    
    # Simulate results
    static_errors = [0.12, 0.11, 0.13, 0.10, 0.12, 0.11]
    rt_errors = [0.10, 0.09, 0.11, 0.09, 0.10, 0.08]
    drift_errors = [0.08, 0.07, 0.09, 0.07, 0.08, 0.07]
    
    # Run comparison
    comparison = ab_test.run_three_way_comparison(
        static_errors, rt_errors, drift_errors
    )
    print(f"Best strategy: {comparison['best_strategy']}")
    print(f"ANOVA p-value: {comparison['anova']['p_value']:.4f}")
    """)
