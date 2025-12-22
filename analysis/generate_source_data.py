"""
Source Data Generator for Nature Publication

Generates SourceData.xlsx and individual CSV files per Nature policy.
All figure data must have corresponding machine-readable source data.

Usage:
    python analysis/generate_source_data.py
    python analysis/generate_source_data.py --figure 4
"""

import argparse
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
SOURCE_DATA_DIR = PROJECT_ROOT / "source_data"
MASTER_FILE = DATA_DIR / "master.parquet"


def load_master_data() -> pd.DataFrame:
    """Load the master analysis dataset."""
    if not MASTER_FILE.exists():
        raise FileNotFoundError(
            f"Master data file not found: {MASTER_FILE}\n"
            "Run the analysis pipeline first: python protocol/run_protocol.py --mode=analysis"
        )
    return pd.read_parquet(MASTER_FILE)


def ensure_dirs():
    """Create source data directory structure."""
    SOURCE_DATA_DIR.mkdir(exist_ok=True)
    for i in range(1, 6):
        (SOURCE_DATA_DIR / f"figure{i}").mkdir(exist_ok=True)


def generate_fig1_data(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Figure 1: Concept and Overview
    
    Panels:
    - 1a: Concept schematic (static/illustrative)
    - 1b: QPU budget allocation
    - 1c: Pipeline performance summary
    """
    data = {}
    
    # Fig 1a: Concept schematic data (illustrative)
    data["fig1a_concept_schematic"] = pd.DataFrame({
        "component": ["probe_t1", "probe_t2", "probe_readout", "qec_syndrome", "decoding"],
        "typical_error_rate": [0.01, 0.02, 0.03, 0.05, 0.02],
        "description": [
            "T1 decay characterization",
            "T2 dephasing characterization", 
            "Readout error measurement",
            "Syndrome extraction",
            "MWPM decoding"
        ]
    })
    
    # Fig 1b: QPU budget allocation
    data["fig1b_qpu_budget"] = pd.DataFrame({
        "method": ["Baseline (cal only)", "Probes (30 shots)", "QEC experiment"],
        "shots": [0, 30 * 27, 4096],  # 27 qubits probed
        "time_seconds": [0, 30 * 27 * 0.001, 4096 * 0.002],  # Approximate
        "percentage": [0, 5, 95]
    })
    
    # Fig 1c: Pipeline comparison (summary from data)
    if len(df) > 0:
        # Use 'strategy' column (from simulation) - maps baseline_static and drift_aware_full_stack
        baseline = df[df["strategy"] == "baseline_static"]["logical_error_rate"]
        adaptive = df[df["strategy"] == "drift_aware_full_stack"]["logical_error_rate"]
        
        data["fig1c_pipeline_performance"] = pd.DataFrame({
            "method": ["Baseline", "Drift-Aware"],
            "error_rate": [baseline.mean(), adaptive.mean()],
            "ci_lower": [
                baseline.mean() - 1.96 * baseline.std() / np.sqrt(max(1, len(baseline))),
                adaptive.mean() - 1.96 * adaptive.std() / np.sqrt(max(1, len(adaptive)))
            ],
            "ci_upper": [
                baseline.mean() + 1.96 * baseline.std() / np.sqrt(max(1, len(baseline))),
                adaptive.mean() + 1.96 * adaptive.std() / np.sqrt(max(1, len(adaptive)))
            ],
            "n_sessions": [len(baseline), len(adaptive)]
        })
    else:
        # Placeholder structure
        data["fig1c_pipeline_performance"] = pd.DataFrame({
            "method": ["Baseline", "Drift-Aware"],
            "error_rate": [np.nan, np.nan],
            "ci_lower": [np.nan, np.nan],
            "ci_upper": [np.nan, np.nan],
            "n_sessions": [0, 0]
        })
    
    return data


def generate_fig2_data(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Figure 2: Drift Characterization
    
    Panels:
    - 2a: T1 time series across sessions
    - 2b: Ranking stability (Kendall tau)
    - 2c: Drift heatmap by qubit
    - 2d: Calibration age vs accuracy gap
    """
    data = {}
    
    # Fig 2a: T1 time series (requires probe data columns)
    if "probe_t1_mean" in df.columns and len(df) > 0:
        data["fig2a_t1_timeseries"] = df[["timestamp", "backend", "probe_t1_mean", "probe_t1_std"]].copy()
        data["fig2a_t1_timeseries"].columns = ["timestamp", "backend", "t1_mean_us", "t1_std_us"]
    else:
        data["fig2a_t1_timeseries"] = pd.DataFrame({
            "timestamp": [],
            "backend": [],
            "t1_mean_us": [],
            "t1_std_us": []
        })
    
    # Fig 2b: Ranking stability (compute Kendall tau between sessions)
    # Placeholder structure - actual computation requires per-qubit data
    data["fig2b_ranking_kendall"] = pd.DataFrame({
        "session_pair": [],
        "kendall_tau": [],
        "hours_elapsed": [],
        "backend": []
    })
    
    # Fig 2c: Drift heatmap (z-scores by qubit)
    # Placeholder structure
    data["fig2c_drift_heatmap"] = pd.DataFrame({
        "qubit_id": [],
        "metric": [],
        "z_score": [],
        "session_id": []
    })
    
    # Fig 2d: Calibration age vs optimal qubit match
    if "cal_age_hours" in df.columns and len(df) > 0:
        data["fig2d_calibration_gap"] = df[["session_id", "cal_age_hours", "backend"]].copy()
        # Add placeholder for optimal qubit match metric
        data["fig2d_calibration_gap"]["optimal_qubit_match"] = np.nan
    else:
        data["fig2d_calibration_gap"] = pd.DataFrame({
            "session_id": [],
            "cal_age_hours": [],
            "backend": [],
            "optimal_qubit_match": []
        })
    
    return data


def generate_fig3_data(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Figure 3: Syndrome Analysis
    
    Panels:
    - 3a: Example syndrome sequences
    - 3b: Burst length distribution
    - 3c: Spatial correlation matrix
    - 3d: Temporal statistics summary
    """
    data = {}
    
    # Fig 3a: Syndrome sequences (example visualization data)
    # Structure for syndrome bit visualization
    data["fig3a_syndrome_sequences"] = pd.DataFrame({
        "shot_id": [],
        "round": [],
        "ancilla_id": [],
        "syndrome_bit": []
    })
    
    # Fig 3b: Burst distribution
    if "burst_index" in df.columns and len(df) > 0:
        # Aggregate burst statistics
        data["fig3b_burst_histogram"] = pd.DataFrame({
            "burst_length": [1, 2, 3, 4, 5],
            "observed_count": [np.nan] * 5,  # Placeholder
            "expected_iid": [np.nan] * 5,
            "ratio": [np.nan] * 5
        })
    else:
        data["fig3b_burst_histogram"] = pd.DataFrame({
            "burst_length": [],
            "observed_count": [],
            "expected_iid": [],
            "ratio": []
        })
    
    # Fig 3c: Spatial correlation
    data["fig3c_spatial_correlation"] = pd.DataFrame({
        "qubit_i": [],
        "qubit_j": [],
        "correlation": [],
        "distance": []
    })
    
    # Fig 3d: Temporal statistics
    if "burst_index" in df.columns and len(df) > 0:
        data["fig3d_temporal_statistics"] = pd.DataFrame({
            "metric": ["burst_index", "temporal_correlation", "max_burst_length"],
            "mean": [
                df["burst_index"].mean() if "burst_index" in df.columns else np.nan,
                df["temporal_correlation"].mean() if "temporal_correlation" in df.columns else np.nan,
                df["max_burst_length"].mean() if "max_burst_length" in df.columns else np.nan
            ],
            "ci_lower": [np.nan, np.nan, np.nan],
            "ci_upper": [np.nan, np.nan, np.nan]
        })
    else:
        data["fig3d_temporal_statistics"] = pd.DataFrame({
            "metric": [],
            "mean": [],
            "ci_lower": [],
            "ci_upper": []
        })
    
    return data


def generate_fig4_data(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Figure 4: Primary Endpoint (Main Result)
    
    Panels:
    - 4a: Error rate comparison scatter
    - 4b: Paired differences distribution
    - 4c: Confidence interval visualization
    - 4d: Effect size summary
    """
    data = {}
    
    if len(df) > 0 and "strategy" in df.columns:
        # Group by session and strategy, averaging across distances
        grouped = df.groupby(["session_id", "strategy", "backend"])["logical_error_rate"].mean().reset_index()
        
        # Reshape for paired comparison
        baseline = grouped[grouped["strategy"] == "baseline_static"].set_index("session_id")
        adaptive = grouped[grouped["strategy"] == "drift_aware_full_stack"].set_index("session_id")
        
        # Find matching sessions
        common_sessions = baseline.index.intersection(adaptive.index).unique()
        
        if len(common_sessions) > 0:
            # Get values for common sessions only
            baseline_vals = baseline.loc[common_sessions, "logical_error_rate"].values
            adaptive_vals = adaptive.loc[common_sessions, "logical_error_rate"].values
            backend_vals = baseline.loc[common_sessions, "backend"].values
            
            # Fig 4a: Scatter comparison
            data["fig4a_error_comparison"] = pd.DataFrame({
                "session_id": list(common_sessions),
                "baseline_error": baseline_vals,
                "adaptive_error": adaptive_vals,
                "backend": backend_vals
            })
            
            # Fig 4b: Paired differences
            diffs = baseline_vals - adaptive_vals
            data["fig4b_paired_diff"] = pd.DataFrame({
                "session_id": list(common_sessions),
                "paired_difference": diffs,
                "backend": backend_vals
            })
            
            # Fig 4c: CI visualization
            mean_diff = np.mean(diffs)
            se = np.std(diffs) / np.sqrt(len(diffs))
            data["fig4c_ci_plot"] = pd.DataFrame({
                "method": ["Drift-Aware vs Baseline"],
                "estimate": [mean_diff],
                "ci_lower": [mean_diff - 1.96 * se],
                "ci_upper": [mean_diff + 1.96 * se],
                "p_value": [np.nan]  # Placeholder - actual p from permutation test
            })
            
            # Fig 4d: Effect sizes
            pooled_std = np.sqrt((np.var(baseline_vals) + np.var(adaptive_vals)) / 2)
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else np.nan
            
            data["fig4d_effect_size"] = pd.DataFrame({
                "comparison": ["Primary endpoint"],
                "cohens_d": [cohens_d],
                "ci_lower": [np.nan],  # Placeholder - bootstrap CI
                "ci_upper": [np.nan],
                "interpretation": ["large" if abs(cohens_d) > 0.8 else "medium" if abs(cohens_d) > 0.5 else "small"]
            })
        else:
            # No matching sessions - create empty structure
            for name in ["fig4a_error_comparison", "fig4b_paired_diff", "fig4c_ci_plot", "fig4d_effect_size"]:
                data[name] = pd.DataFrame()
    else:
        # Create placeholder structures
        data["fig4a_error_comparison"] = pd.DataFrame({
            "session_id": [], "baseline_error": [], "adaptive_error": [], "backend": []
        })
        data["fig4b_paired_diff"] = pd.DataFrame({
            "session_id": [], "paired_difference": [], "backend": []
        })
        data["fig4c_ci_plot"] = pd.DataFrame({
            "method": [], "estimate": [], "ci_lower": [], "ci_upper": [], "p_value": []
        })
        data["fig4d_effect_size"] = pd.DataFrame({
            "comparison": [], "cohens_d": [], "ci_lower": [], "ci_upper": [], "interpretation": []
        })
    
    return data


def generate_fig5_data(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Figure 5: Generalization
    
    Panels:
    - 5a: Effect by backend
    - 5b: Effect by code distance
    - 5c: Resource efficiency analysis
    """
    data = {}
    
    if len(df) > 0:
        # Fig 5a: Backend comparison
        backends = df["backend"].unique() if "backend" in df.columns else []
        backend_effects = []
        for backend in backends:
            backend_df = df[df["backend"] == backend]
            baseline = backend_df[backend_df["strategy"] == "baseline_static"]["logical_error_rate"]
            adaptive = backend_df[backend_df["strategy"] == "drift_aware_full_stack"]["logical_error_rate"]
            if len(baseline) > 0 and len(adaptive) > 0:
                effect = baseline.mean() - adaptive.mean()
                backend_effects.append({
                    "backend": backend,
                    "effect_size": effect,
                    "n_sessions": min(len(baseline), len(adaptive)),
                    "p_value": np.nan  # Placeholder
                })
        data["fig5a_backend_comparison"] = pd.DataFrame(backend_effects) if backend_effects else pd.DataFrame({
            "backend": [], "effect_size": [], "n_sessions": [], "p_value": []
        })
        
        # Fig 5b: Distance scaling
        distances = df["distance"].unique() if "distance" in df.columns else []
        distance_effects = []
        for d in sorted(distances):
            dist_df = df[df["distance"] == d]
            baseline = dist_df[dist_df["strategy"] == "baseline_static"]["logical_error_rate"]
            adaptive = dist_df[dist_df["strategy"] == "drift_aware_full_stack"]["logical_error_rate"]
            if len(baseline) > 0 and len(adaptive) > 0:
                distance_effects.append({
                    "code_distance": d,
                    "baseline_error": baseline.mean(),
                    "adaptive_error": adaptive.mean(),
                    "improvement": baseline.mean() - adaptive.mean()
                })
        data["fig5b_distance_scaling"] = pd.DataFrame(distance_effects) if distance_effects else pd.DataFrame({
            "code_distance": [], "baseline_error": [], "adaptive_error": [], "improvement": []
        })
        
        # Fig 5c: Resource efficiency
        data["fig5c_resource_efficiency"] = pd.DataFrame({
            "probe_shots": [10, 20, 30, 50, 100],
            "improvement": [np.nan] * 5,  # Placeholder - from sensitivity analysis
            "overhead_percent": [1, 2, 3, 5, 10]
        })
    else:
        data["fig5a_backend_comparison"] = pd.DataFrame({
            "backend": [], "effect_size": [], "n_sessions": [], "p_value": []
        })
        data["fig5b_distance_scaling"] = pd.DataFrame({
            "code_distance": [], "baseline_error": [], "adaptive_error": [], "improvement": []
        })
        data["fig5c_resource_efficiency"] = pd.DataFrame({
            "probe_shots": [], "improvement": [], "overhead_percent": []
        })
    
    return data


def save_csv_files(fig_data: dict[str, pd.DataFrame], figure_num: int):
    """Save individual CSV files for a figure."""
    fig_dir = SOURCE_DATA_DIR / f"figure{figure_num}"
    for name, df in fig_data.items():
        filepath = fig_dir / f"{name}.csv"
        df.to_csv(filepath, index=False)
        print(f"  Saved: {filepath.name}")


def generate_excel(all_data: dict[str, pd.DataFrame]):
    """Generate consolidated SourceData.xlsx with all figures."""
    excel_path = SOURCE_DATA_DIR / "SourceData.xlsx"
    
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for sheet_name, df in all_data.items():
            # Excel sheet names max 31 chars
            short_name = sheet_name[:31]
            df.to_excel(writer, sheet_name=short_name, index=False)
    
    print(f"\nGenerated: {excel_path}")


def main(figure: Optional[int] = None):
    """Generate source data files."""
    print("=" * 60)
    print("Source Data Generator")
    print("=" * 60)
    
    ensure_dirs()
    
    # Load data
    try:
        df = load_master_data()
        print(f"Loaded master data: {len(df)} rows")
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("Generating placeholder structure without data...")
        df = pd.DataFrame()
    
    # Generate all figure data
    generators = {
        1: generate_fig1_data,
        2: generate_fig2_data,
        3: generate_fig3_data,
        4: generate_fig4_data,
        5: generate_fig5_data,
    }
    
    all_data = {}
    
    for fig_num, generator in generators.items():
        if figure is not None and fig_num != figure:
            continue
        
        print(f"\nFigure {fig_num}:")
        fig_data = generator(df)
        all_data.update(fig_data)
        save_csv_files(fig_data, fig_num)
    
    # Generate consolidated Excel
    if figure is None:
        generate_excel(all_data)
    
    print("\n" + "=" * 60)
    print("Source data generation complete")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate source data for figures")
    parser.add_argument("--figure", type=int, help="Generate data for specific figure only")
    args = parser.parse_args()
    
    main(figure=args.figure)
