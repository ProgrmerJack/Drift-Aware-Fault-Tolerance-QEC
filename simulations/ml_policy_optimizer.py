"""
Machine Learning Analysis for Optimal Drift-Aware Policies
===========================================================

Uses ML to discover optimal probe scheduling policies from simulated data.

Key novelty for Nature Communications:
1. Automated policy discovery (beyond manual 4-hour heuristic)
2. Platform-adaptive recommendations
3. Cost-benefit optimization
4. Predictive modeling of drift impact

Author: DAQEC Research Team
Date: December 2025
License: MIT
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import json


class DriftPolicyOptimizer:
    """ML-based optimization of drift-aware probe scheduling policies."""
    
    def __init__(self, output_dir: str = "simulations/ml_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.scaler = StandardScaler()
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features for ML model.
        
        Features:
        - time_since_calibration: Hours since last calibration
        - distance: Code distance
        - n_qubits: Number of qubits in code
        - platform parameters (T1, T2, drift magnitude)
        
        Target:
        - improvement_pct: DAQEC benefit in percentage
        """
        feature_cols = [
            "time_since_cal",  # Updated from time_since_calibration
            "distance" if "distance" in df.columns else "n_qubits",
        ]
        
        # Add platform-specific features if available
        if "drift_magnitude" in df.columns:
            feature_cols.append("drift_magnitude")
        if "calibration_interval" in df.columns:
            feature_cols.append("calibration_interval")
        
        X = df[feature_cols].values
        y = df["improvement_pct"].values
        
        return X, y
    
    def train_benefit_predictor(self, df: pd.DataFrame, model_type: str = "rf") -> Dict:
        """Train ML model to predict DAQEC benefit.
        
        Args:
            df: Simulation results dataframe
            model_type: 'rf' (Random Forest) or 'gb' (Gradient Boosting)
            
        Returns:
            Dictionary with model performance metrics
        """
        print(f"Training {model_type} model to predict DAQEC benefit...")
        
        X, y = self.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        if model_type == "rf":
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == "gb":
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        metrics = {
            "model_type": model_type,
            "train_r2": r2_score(y_train, y_pred_train),
            "test_r2": r2_score(y_test, y_pred_test),
            "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
            "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
            "cv_scores": cross_val_score(
                self.model, X_train_scaled, y_train, cv=5, scoring="r2"
            ).tolist()
        }
        
        print(f"  Train R²: {metrics['train_r2']:.3f}")
        print(f"  Test R²: {metrics['test_r2']:.3f}")
        print(f"  Test RMSE: {metrics['test_rmse']:.2f}%")
        print(f"  CV R² (mean±std): {np.mean(metrics['cv_scores']):.3f}±{np.std(metrics['cv_scores']):.3f}")
        
        # Feature importance
        if hasattr(self.model, "feature_importances_"):
            feature_names = [
                "time_since_calibration",
                "distance",
            ]
            if "drift_magnitude" in df.columns:
                feature_names.append("drift_magnitude")
            if "calibration_interval" in df.columns:
                feature_names.append("calibration_interval")
            
            importance = self.model.feature_importances_
            metrics["feature_importance"] = dict(zip(feature_names, importance.tolist()))
            
            print("\n  Feature importance:")
            for name, imp in sorted(metrics["feature_importance"].items(), 
                                   key=lambda x: x[1], reverse=True):
                print(f"    {name}: {imp:.3f}")
        
        return metrics
    
    def optimize_probe_interval(self, 
                               distance: int,
                               platform_params: Dict,
                               probe_cost_pct: float = 2.0,
                               benefit_threshold_pct: float = 50.0) -> Dict:
        """Find optimal probe interval for given constraints.
        
        Args:
            distance: Code distance
            platform_params: Platform-specific parameters
            probe_cost_pct: Maximum acceptable probe overhead (%)
            benefit_threshold_pct: Minimum acceptable benefit (%)
            
        Returns:
            Optimal policy recommendation
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_benefit_predictor first.")
        
        print(f"\nOptimizing probe interval for distance={distance}...")
        
        # Test different probe intervals
        intervals = np.linspace(1, 24, 24)  # 1 to 24 hours
        
        predictions = []
        for interval in intervals:
            # Feature vector (time_since_cal, distance)
            X = np.array([[interval, distance]])
            
            X_scaled = self.scaler.transform(X)
            benefit = self.model.predict(X_scaled)[0]
            
            # Compute cost (simplified: probe_cost_pct per probe)
            n_probes_per_day = 24 / interval
            daily_cost_pct = n_probes_per_day * probe_cost_pct
            
            # Net benefit (benefit - cost)
            net_benefit = benefit - daily_cost_pct
            
            predictions.append({
                "interval_hours": interval,
                "predicted_benefit_pct": benefit,
                "daily_cost_pct": daily_cost_pct,
                "net_benefit_pct": net_benefit
            })
        
        # Find optimal interval (max net benefit)
        optimal = max(predictions, key=lambda x: x["net_benefit_pct"])
        
        # Find interval that achieves 90% of max benefit with minimal cost
        max_benefit = max(p["predicted_benefit_pct"] for p in predictions)
        threshold_benefit = 0.9 * max_benefit
        
        efficient_intervals = [
            p for p in predictions 
            if p["predicted_benefit_pct"] >= threshold_benefit
        ]
        efficient_optimal = max(efficient_intervals, key=lambda x: x["interval_hours"])
        
        result = {
            "distance": distance,
            "optimal_interval_hours": optimal["interval_hours"],
            "optimal_benefit_pct": optimal["predicted_benefit_pct"],
            "optimal_cost_pct": optimal["daily_cost_pct"],
            "optimal_net_benefit_pct": optimal["net_benefit_pct"],
            "efficient_interval_hours": efficient_optimal["interval_hours"],
            "efficient_benefit_pct": efficient_optimal["predicted_benefit_pct"],
            "efficient_cost_pct": efficient_optimal["daily_cost_pct"],
            "max_benefit_pct": max_benefit,
            "all_predictions": predictions
        }
        
        print(f"  Optimal interval: {optimal['interval_hours']:.1f}h")
        print(f"    Benefit: {optimal['predicted_benefit_pct']:.1f}%")
        print(f"    Cost: {optimal['daily_cost_pct']:.2f}%")
        print(f"    Net benefit: {optimal['net_benefit_pct']:.1f}%")
        print(f"\n  Efficient interval (90% of max benefit):")
        print(f"    Interval: {efficient_optimal['interval_hours']:.1f}h")
        print(f"    Benefit: {efficient_optimal['predicted_benefit_pct']:.1f}%")
        print(f"    Cost: {efficient_optimal['daily_cost_pct']:.2f}%")
        
        return result
    
    def plot_benefit_vs_interval(self, optimization_result: Dict, save_path: Path = None):
        """Plot predicted benefit vs probe interval."""
        predictions = optimization_result["all_predictions"]
        df = pd.DataFrame(predictions)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Benefit and cost vs interval
        ax1.plot(df["interval_hours"], df["predicted_benefit_pct"], 
                'b-', linewidth=2, label="Predicted Benefit")
        ax1.plot(df["interval_hours"], df["daily_cost_pct"], 
                'r--', linewidth=2, label="Daily Cost")
        ax1.axvline(optimization_result["optimal_interval_hours"], 
                   color='green', linestyle=':', alpha=0.7, label="Optimal")
        ax1.axvline(optimization_result["efficient_interval_hours"],
                   color='orange', linestyle=':', alpha=0.7, label="Efficient (90%)")
        ax1.set_xlabel("Probe Interval (hours)", fontsize=12)
        ax1.set_ylabel("Percentage (%)", fontsize=12)
        ax1.set_title(f"DAQEC Benefit vs Probe Interval (d={optimization_result['distance']})", 
                     fontsize=14)
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot 2: Net benefit vs interval
        ax2.plot(df["interval_hours"], df["net_benefit_pct"], 
                'g-', linewidth=2)
        ax2.axvline(optimization_result["optimal_interval_hours"],
                   color='green', linestyle=':', alpha=0.7, label="Optimal")
        ax2.axhline(0, color='k', linestyle='-', alpha=0.3)
        ax2.set_xlabel("Probe Interval (hours)", fontsize=12)
        ax2.set_ylabel("Net Benefit (%)", fontsize=12)
        ax2.set_title("Net Benefit (Benefit - Cost)", fontsize=14)
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {save_path}")
        
        plt.close()
        
        return fig


class AdvancedDriftAnalysis:
    """Advanced statistical analysis of drift patterns."""
    
    def __init__(self, output_dir: str = "simulations/ml_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze_tail_compression_scaling(self, df: pd.DataFrame) -> Dict:
        """Analyze how tail compression scales with distance.
        
        Hypothesis: Larger codes show greater tail compression
        (more qubits = more tail events to compress)
        """
        print("\nAnalyzing tail compression scaling...")
        
        # Group by distance
        if "distance" not in df.columns:
            print("  Warning: No distance column, skipping analysis")
            return {}
        
        results = {}
        
        for distance, group in df.groupby("distance"):
            # Compute P95 improvement
            baseline_p95 = group["ler_baseline"].quantile(0.95)
            daqec_p95 = group["ler_daqec"].quantile(0.95)
            p95_compression = 100 * (baseline_p95 - daqec_p95) / baseline_p95
            
            # Compute P99 improvement
            baseline_p99 = group["ler_baseline"].quantile(0.99)
            daqec_p99 = group["ler_daqec"].quantile(0.99)
            p99_compression = 100 * (baseline_p99 - daqec_p99) / baseline_p99
            
            # Mean improvement
            mean_improvement = group["improvement_pct"].mean()
            
            results[int(distance)] = {
                "p95_compression_pct": p95_compression,
                "p99_compression_pct": p99_compression,
                "mean_improvement_pct": mean_improvement,
                "tail_vs_mean_ratio": (p95_compression + p99_compression) / (2 * mean_improvement)
            }
            
            print(f"  Distance {distance}:")
            print(f"    P95 compression: {p95_compression:.1f}%")
            print(f"    P99 compression: {p99_compression:.1f}%")
            print(f"    Mean improvement: {mean_improvement:.1f}%")
            print(f"    Tail/Mean ratio: {results[int(distance)]['tail_vs_mean_ratio']:.2f}")
        
        return results
    
    def discover_conditional_benefit_patterns(self, df: pd.DataFrame) -> Dict:
        """Discover when DAQEC provides largest benefits.
        
        Uses decision tree to find interpretable rules.
        """
        from sklearn.tree import DecisionTreeRegressor, export_text
        
        print("\nDiscovering conditional benefit patterns...")
        
        X, y = self._prepare_features_for_tree(df)
        
        # Shallow tree for interpretability
        tree = DecisionTreeRegressor(max_depth=4, min_samples_split=20, random_state=42)
        tree.fit(X, y)
        
        # Extract rules
        feature_names = ["time_since_calibration", "distance", "drift_magnitude"]
        rules = export_text(tree, feature_names=feature_names)
        
        print("\nDecision rules for DAQEC benefit:")
        print(rules)
        
        return {
            "decision_rules": rules,
            "feature_importance": dict(zip(feature_names, tree.feature_importances_.tolist()))
        }
    
    def _prepare_features_for_tree(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare simplified feature set for decision tree."""
        features = ["time_since_calibration"]
        
        if "distance" in df.columns:
            features.append("distance")
        elif "n_qubits" in df.columns:
            features.append("n_qubits")
        
        if "drift_magnitude" in df.columns:
            features.append("drift_magnitude")
        else:
            # Add default drift magnitude for IBM
            df = df.copy()
            df["drift_magnitude"] = 0.727
            features.append("drift_magnitude")
        
        X = df[features].values
        y = df["improvement_pct"].values
        
        return X, y


if __name__ == "__main__":
    print("="*80)
    print("ML-Based Policy Optimization for DAQEC")
    print("="*80)
    print()
    
    # Load simulation results
    sim_dir = Path("results")
    
    # Load distance scaling data
    df_scaling = pd.read_csv(sim_dir / "distance_scaling_ibm_v2.csv")
    
    # Initialize optimizer
    optimizer = DriftPolicyOptimizer()
    
    # Train benefit predictor
    print("\n" + "="*80)
    print("Training ML Model to Predict DAQEC Benefit")
    print("="*80)
    metrics = optimizer.train_benefit_predictor(df_scaling, model_type="rf")
    
    # Save metrics
    with open(optimizer.output_dir / "model_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Optimize probe intervals for different distances
    print("\n" + "="*80)
    print("Optimizing Probe Intervals")
    print("="*80)
    
    platform_params = {
        "drift_magnitude": 0.727,
        "calibration_interval": 24.0
    }
    
    optimal_policies = []
    for distance in [3, 5, 7, 9, 11, 13]:
        result = optimizer.optimize_probe_interval(
            distance=distance,
            platform_params=platform_params,
            probe_cost_pct=2.0
        )
        optimal_policies.append(result)
        
        # Plot
        fig_path = optimizer.output_dir / f"benefit_vs_interval_d{distance}.png"
        optimizer.plot_benefit_vs_interval(result, fig_path)
    
    # Save optimal policies
    policies_file = optimizer.output_dir / "optimal_policies.json"
    with open(policies_file, 'w') as f:
        json.dump(optimal_policies, f, indent=2)
    
    print(f"\nOptimal policies saved to {policies_file}")
    
    # Advanced analysis
    print("\n" + "="*80)
    print("Advanced Drift Pattern Analysis")
    print("="*80)
    
    # Commented out - columns not available in V2 CSV format
    # analyzer = AdvancedDriftAnalysis()
    
    # # Tail compression scaling
    # tail_results = analyzer.analyze_tail_compression_scaling(df_scaling)
    
    # with open(analyzer.output_dir / "tail_compression_scaling.json", 'w') as f:
    #     json.dump(tail_results, f, indent=2)
    
    # # Conditional benefit patterns
    # pattern_results = analyzer.discover_conditional_benefit_patterns(df_scaling)
    
    # with open(analyzer.output_dir / "conditional_benefit_patterns.json", 'w') as f:
    #     json.dump(pattern_results, f, indent=2)
    
    print("\n" + "="*80)
    print("ML ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {optimizer.output_dir}")
