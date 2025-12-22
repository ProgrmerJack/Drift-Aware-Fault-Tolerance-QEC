"""
Analysis Module: Results Visualization
======================================

Visualization utilities for QEC experiments and drift analysis.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResultsVisualizer:
    """
    Creates visualizations for QEC experiments and drift analysis.
    
    Generates publication-quality figures for:
    - Logical error rate vs distance/rounds
    - Drift time series
    - A/B test comparisons
    - Correlation heatmaps
    """
    
    def __init__(self, output_dir: str = "figures"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory for saving figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Import matplotlib lazily
        self._plt = None
        self._sns = None
        
    @property
    def plt(self):
        """Lazy import matplotlib."""
        if self._plt is None:
            import matplotlib.pyplot as plt
            self._plt = plt
            # Set style for publication
            plt.style.use('seaborn-v0_8-whitegrid')
            plt.rcParams['figure.figsize'] = (8, 6)
            plt.rcParams['font.size'] = 12
            plt.rcParams['axes.labelsize'] = 14
            plt.rcParams['axes.titlesize'] = 16
            plt.rcParams['legend.fontsize'] = 11
        return self._plt
    
    @property
    def sns(self):
        """Lazy import seaborn."""
        if self._sns is None:
            import seaborn as sns
            self._sns = sns
        return self._sns
    
    def plot_logical_error_vs_distance(self,
                                        error_data: Dict[int, Dict[str, float]],
                                        code_type: str = "bit_flip",
                                        save: bool = True) -> Optional[Path]:
        """
        Plot logical error rate vs code distance.
        
        Args:
            error_data: Dict mapping distance to {strategy: error_rate}
            code_type: Code type for title
            save: Whether to save figure
            
        Returns:
            Path to saved figure or None
        """
        fig, ax = self.plt.subplots()
        
        distances = sorted(error_data.keys())
        strategies = list(error_data[distances[0]].keys()) if distances else []
        
        markers = ['o', 's', '^', 'D', 'v']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, strategy in enumerate(strategies):
            errors = [error_data[d].get(strategy, np.nan) for d in distances]
            ax.semilogy(distances, errors, 
                       marker=markers[i % len(markers)],
                       color=colors[i % len(colors)],
                       label=strategy,
                       linewidth=2,
                       markersize=8)
        
        ax.set_xlabel('Code Distance')
        ax.set_ylabel('Logical Error Rate')
        ax.set_title(f'{code_type.replace("_", " ").title()} Repetition Code')
        ax.legend(loc='upper right')
        ax.set_xticks(distances)
        ax.grid(True, alpha=0.3)
        
        self.plt.tight_layout()
        
        if save:
            filepath = self.output_dir / f'logical_error_vs_distance_{code_type}.pdf'
            self.plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {filepath}")
            return filepath
        else:
            self.plt.show()
            return None
    
    def plot_drift_time_series(self,
                                drift_data: pd.DataFrame,
                                property_name: str = "T1",
                                highlight_changes: Optional[List[datetime]] = None,
                                save: bool = True) -> Optional[Path]:
        """
        Plot calibration drift over time.
        
        Args:
            drift_data: DataFrame with datetime index and qubit columns
            property_name: Property being plotted
            highlight_changes: Optional change points to highlight
            save: Whether to save figure
            
        Returns:
            Path to saved figure or None
        """
        fig, ax = self.plt.subplots(figsize=(12, 6))
        
        # Plot each qubit
        for col in drift_data.columns[:10]:  # Limit to 10 qubits for clarity
            ax.plot(drift_data.index, drift_data[col], alpha=0.7, label=col)
        
        # Highlight change points
        if highlight_changes:
            for cp in highlight_changes:
                ax.axvline(cp, color='red', linestyle='--', alpha=0.5, linewidth=2)
        
        ax.set_xlabel('Date')
        ax.set_ylabel(f'{property_name} (μs)' if property_name in ['T1', 'T2'] else property_name)
        ax.set_title(f'{property_name} Drift Over Time')
        
        if len(drift_data.columns) <= 10:
            ax.legend(loc='upper right', ncol=2)
        
        # Format x-axis
        fig.autofmt_xdate()
        ax.grid(True, alpha=0.3)
        
        self.plt.tight_layout()
        
        if save:
            filepath = self.output_dir / f'drift_time_series_{property_name}.pdf'
            self.plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {filepath}")
            return filepath
        else:
            self.plt.show()
            return None
    
    def plot_ab_comparison(self,
                           comparison_results: Dict[str, Any],
                           save: bool = True) -> Optional[Path]:
        """
        Plot A/B test comparison results.
        
        Args:
            comparison_results: Results from ABTestFramework.run_three_way_comparison()
            save: Whether to save figure
            
        Returns:
            Path to saved figure or None
        """
        fig, axes = self.plt.subplots(1, 2, figsize=(14, 6))
        
        # Left: Bar plot with error bars
        ax1 = axes[0]
        summary = comparison_results.get("summary_statistics", {})
        
        strategies = list(summary.keys())
        means = [summary[s]["mean"] for s in strategies]
        stds = [summary[s]["std"] for s in strategies]
        
        x = np.arange(len(strategies))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        bars = ax1.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
        
        ax1.set_xlabel('Selection Strategy')
        ax1.set_ylabel('Logical Error Rate')
        ax1.set_title('Strategy Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels([s.replace("_", "\n") for s in strategies])
        
        # Add significance annotations
        best_idx = means.index(min(means))
        ax1.annotate('*', xy=(best_idx, means[best_idx] + stds[best_idx]),
                    ha='center', fontsize=20)
        
        # Right: Effect size comparison
        ax2 = axes[1]
        pairwise = comparison_results.get("pairwise_comparisons", {})
        
        comparisons = list(pairwise.keys())
        effect_sizes = [pairwise[c]["effect_size"]["cohens_d"] for c in comparisons]
        
        y = np.arange(len(comparisons))
        colors_effect = ['green' if es < 0 else 'red' for es in effect_sizes]
        
        ax2.barh(y, effect_sizes, color=colors_effect, alpha=0.8)
        ax2.set_yticks(y)
        ax2.set_yticklabels([c.replace("_vs_", "\nvs\n") for c in comparisons])
        ax2.set_xlabel("Cohen's d Effect Size")
        ax2.set_title('Pairwise Effect Sizes')
        ax2.axvline(0, color='black', linestyle='-', linewidth=1)
        
        # Add effect size interpretation zones
        ax2.axvspan(-0.2, 0.2, alpha=0.1, color='gray', label='Negligible')
        ax2.axvspan(-0.5, -0.2, alpha=0.1, color='yellow')
        ax2.axvspan(0.2, 0.5, alpha=0.1, color='yellow', label='Small')
        
        self.plt.tight_layout()
        
        if save:
            filepath = self.output_dir / 'ab_comparison.pdf'
            self.plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {filepath}")
            return filepath
        else:
            self.plt.show()
            return None
    
    def plot_correlation_heatmap(self,
                                  correlation_matrix: pd.DataFrame,
                                  title: str = "Cross-Qubit Correlation",
                                  save: bool = True) -> Optional[Path]:
        """
        Plot correlation matrix as heatmap.
        
        Args:
            correlation_matrix: DataFrame with correlation values
            title: Figure title
            save: Whether to save figure
            
        Returns:
            Path to saved figure or None
        """
        fig, ax = self.plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
        
        self.sns.heatmap(correlation_matrix, 
                        mask=mask,
                        annot=True if len(correlation_matrix) <= 10 else False,
                        fmt='.2f',
                        cmap='coolwarm',
                        center=0,
                        vmin=-1, vmax=1,
                        square=True,
                        ax=ax)
        
        ax.set_title(title)
        
        self.plt.tight_layout()
        
        if save:
            safe_title = title.lower().replace(" ", "_")
            filepath = self.output_dir / f'correlation_{safe_title}.pdf'
            self.plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {filepath}")
            return filepath
        else:
            self.plt.show()
            return None
    
    def plot_syndrome_history(self,
                               syndrome_data: List[str],
                               title: str = "Syndrome History",
                               save: bool = True) -> Optional[Path]:
        """
        Plot syndrome measurement history as a heatmap.
        
        Args:
            syndrome_data: List of syndrome bit strings
            title: Figure title
            save: Whether to save figure
            
        Returns:
            Path to saved figure or None
        """
        # Convert syndromes to binary matrix
        n_rounds = len(syndrome_data)
        n_bits = len(syndrome_data[0]) if syndrome_data else 0
        
        matrix = np.zeros((n_rounds, n_bits))
        for i, syn in enumerate(syndrome_data):
            for j, bit in enumerate(syn):
                matrix[i, j] = int(bit)
        
        fig, ax = self.plt.subplots(figsize=(max(6, n_bits), max(4, n_rounds * 0.3)))
        
        self.sns.heatmap(matrix,
                        cmap='Blues',
                        cbar_kws={'label': 'Syndrome Value'},
                        xticklabels=[f'S{i}' for i in range(n_bits)],
                        yticklabels=[f'R{i}' for i in range(n_rounds)],
                        ax=ax)
        
        ax.set_xlabel('Syndrome Qubit')
        ax.set_ylabel('Round')
        ax.set_title(title)
        
        self.plt.tight_layout()
        
        if save:
            filepath = self.output_dir / 'syndrome_history.pdf'
            self.plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {filepath}")
            return filepath
        else:
            self.plt.show()
            return None


def generate_publication_figures(results_dir: str = "data/experiments",
                                  output_dir: str = "figures") -> List[Path]:
    """
    Generate all publication figures from experiment results.
    
    Args:
        results_dir: Directory containing experiment results
        output_dir: Directory for output figures
        
    Returns:
        List of paths to generated figures
    """
    import json
    
    visualizer = ResultsVisualizer(output_dir)
    generated = []
    
    results_path = Path(results_dir)
    if not results_path.exists():
        logger.warning(f"Results directory {results_dir} does not exist")
        return generated
    
    # Load all experiment results
    for result_file in results_path.glob("experiment_*.json"):
        with open(result_file) as f:
            results = json.load(f)
            
        # Generate appropriate figures based on results content
        if "qec_analysis" in results:
            # Extract error rates by distance
            error_rates = results["qec_analysis"].get("logical_error_rates", {})
            if error_rates:
                # Restructure for plotting
                distances = set()
                for key in error_rates.keys():
                    d = int(key.split("_")[0].replace("d", ""))
                    distances.add(d)
                    
                # Create plot data
                plot_data = {}
                for d in sorted(distances):
                    for key, data in error_rates.items():
                        if key.startswith(f"d{d}_"):
                            r = key.split("_")[1]
                            if d not in plot_data:
                                plot_data[d] = {}
                            avg_error = np.mean([item["logical_error_rate"] for item in data])
                            plot_data[d][r] = avg_error
                            
                # This would need more structure to plot properly
                logger.info(f"Processed QEC data from {result_file}")
                
    return generated


if __name__ == "__main__":
    print("Visualization Module")
    print("\nExample usage:")
    print("""
    from src.analysis.visualization import ResultsVisualizer
    
    viz = ResultsVisualizer("figures")
    
    # Example: Plot error rate vs distance
    error_data = {
        3: {"static": 0.12, "rt": 0.10, "drift_aware": 0.08},
        5: {"static": 0.08, "rt": 0.06, "drift_aware": 0.05},
        7: {"static": 0.05, "rt": 0.04, "drift_aware": 0.03}
    }
    viz.plot_logical_error_vs_distance(error_data)
    """)
