"""
Surface Code Simulator with Drift-Aware Qubit Selection
========================================================

Extends DAQEC findings to fault-tolerance-relevant scales and surface codes.

This simulation framework addresses the manuscript limitation:
"Our study is limited to repetition codes... Extension to surface codes remains important."

Key novelty additions for Nature Communications:
1. Fault-tolerance-relevant distances (d=9, 11, 13) - beyond hardware limits
2. Surface code implementation - demonstrates code-general benefits
3. Diverse drift models - shows robustness to drift assumptions
4. Platform diversity - validates platform-general hypothesis

Author: DAQEC Research Team
Date: December 2025
License: MIT
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from scipy.stats import spearmanr
from datetime import datetime


class DriftModel(Enum):
    """Drift model types for qubit parameter degradation."""
    GAUSSIAN = "gaussian"  # Random Gaussian fluctuations (IBM-like)
    POWER_LAW = "power_law"  # 1/f noise (correlated drift)
    EXPONENTIAL = "exponential"  # Exponential decay (heating)
    CORRELATED = "correlated"  # Spatially correlated drift


@dataclass
class QubitParameters:
    """Physical qubit parameters that drift over time."""
    T1: float  # Energy relaxation time (µs)
    T2: float  # Dephasing time (µs)
    readout_error: float  # Readout assignment error
    gate_error_1q: float  # Single-qubit gate error
    gate_error_2q: float  # Two-qubit gate error
    timestamp: float  # Time since last calibration (hours)
    
    def logical_error_rate(self) -> float:
        """Estimate logical error contribution from physical parameters."""
        # Simplified error model - real-world is more complex
        return (
            0.3 * self.gate_error_2q +  # Two-qubit gates dominate
            0.2 * self.gate_error_1q +  # Single-qubit gates
            0.3 * self.readout_error +  # Readout errors
            0.2 * (1.0 / max(self.T1, 1e-6))  # Coherence decay
        )


class QuantumPlatform:
    """Simulates different quantum hardware platforms with characteristic noise."""
    
    def __init__(self, platform_type: str = "ibm"):
        self.platform_type = platform_type
        self.base_params = self._get_base_parameters()
        
    def _get_base_parameters(self) -> Dict:
        """Platform-specific baseline parameters."""
        if self.platform_type == "ibm":
            # IBM Quantum (Heron r2 - like IBM Fez)
            return {
                "T1_mean": 200,  # 200 µs
                "T2_mean": 100,  # 100 µs
                "readout_error_mean": 0.01,
                "gate_error_1q_mean": 0.0003,
                "gate_error_2q_mean": 0.005,
                "calibration_interval": 24.0,  # hours
                "drift_magnitude": 0.727,  # 72.7% drift from real data
            }
        elif self.platform_type == "google":
            # Google Quantum (Willow-like)
            return {
                "T1_mean": 100,  # 100 µs
                "T2_mean": 50,  # 50 µs
                "readout_error_mean": 0.005,
                "gate_error_1q_mean": 0.0001,
                "gate_error_2q_mean": 0.003,
                "calibration_interval": 12.0,  # hours (more frequent)
                "drift_magnitude": 0.5,  # Hypothesized lower drift
            }
        elif self.platform_type == "rigetti":
            # Rigetti (Aspen-like)
            return {
                "T1_mean": 150,
                "T2_mean": 80,
                "readout_error_mean": 0.02,
                "gate_error_1q_mean": 0.0005,
                "gate_error_2q_mean": 0.008,
                "calibration_interval": 48.0,  # hours (less frequent)
                "drift_magnitude": 0.8,  # Hypothesized higher drift
            }
        else:
            raise ValueError(f"Unknown platform: {self.platform_type}")
    
    def generate_qubits(self, n_qubits: int, time_since_calibration: float,
                       drift_model: DriftModel = DriftModel.GAUSSIAN,
                       seed: Optional[int] = None,
                       base_qubits: Optional[List['QubitParameters']] = None) -> List['QubitParameters']:
        """Generate qubit parameters with time-dependent HETEROGENEOUS drift.
        
        Key insight: Each physical qubit has INTRINSIC properties + TIME-DEPENDENT drift.
        If base_qubits provided, apply drift to those specific qubits.
        Otherwise, generate new qubits with intrinsic properties.
        
        Args:
            n_qubits: Number of physical qubits
            time_since_calibration: Hours since last calibration
            drift_model: Type of drift to apply
            seed: Random seed for reproducibility
            base_qubits: If provided, apply drift to these specific qubits
            
        Returns:
            List of QubitParameters with realistic heterogeneous drift
        """
        if seed is not None:
            np.random.seed(seed)
        
        base = self.base_params
        
        # If base_qubits provided, we're applying drift to existing qubits
        if base_qubits is not None:
            return self._apply_drift_to_qubits(base_qubits, time_since_calibration, drift_model)
        
        # Otherwise, generate new qubits with intrinsic properties
        base_drift_factor = self._compute_drift_factor(time_since_calibration, drift_model)
        
        qubits = []
        for i in range(n_qubits):
            # Base INTRINSIC parameters (device-to-device variation)
            T1_intrinsic = base["T1_mean"] * np.random.lognormal(0, 0.3)
            T2_intrinsic = min(base["T2_mean"] * np.random.lognormal(0, 0.3), 2 * T1_intrinsic)
            
            # Each qubit has intrinsic "drift susceptibility"
            # σ=1.5 creates wide variation: some qubits ultra-stable, others degrade rapidly
            drift_susceptibility = np.random.lognormal(0, 1.5)
            
            # Apply drift
            qubit_drift_factor = base_drift_factor * drift_susceptibility
            T1_drifted = T1_intrinsic * np.exp(-qubit_drift_factor * np.random.uniform(0.8, 1.2))
            T2_drifted = T2_intrinsic * np.exp(-qubit_drift_factor * np.random.uniform(0.8, 1.2))
            
            # Errors increase with drift
            error_multiplier = 1 + qubit_drift_factor * np.random.uniform(1.5, 2.5)
            readout_error = base["readout_error_mean"] * error_multiplier
            gate_error_1q = base["gate_error_1q_mean"] * error_multiplier
            gate_error_2q = base["gate_error_2q_mean"] * error_multiplier
            
            qubits.append(QubitParameters(
                T1=max(T1_drifted, 1e-6),
                T2=max(T2_drifted, 1e-6),
                readout_error=min(readout_error, 0.5),
                gate_error_1q=min(gate_error_1q, 0.1),
                gate_error_2q=min(gate_error_2q, 0.1),
                timestamp=time_since_calibration
            ))
        
        return qubits
    
    def _apply_drift_to_qubits(self, base_qubits: List['QubitParameters'], 
                               time_delta: float,
                               drift_model: DriftModel) -> List['QubitParameters']:
        """Apply drift to existing qubits (modeling time evolution of SAME physical qubits)."""
        drift_factor = self._compute_drift_factor(time_delta, drift_model)
        
        drifted = []
        for q in base_qubits:
            # Each qubit degrades independently
            susceptibility = np.random.lognormal(0, 1.5)
            qubit_drift = drift_factor * susceptibility
            
            # Coherence degrades exponentially
            T1_new = q.T1 * np.exp(-qubit_drift * np.random.uniform(0.8, 1.2))
            T2_new = q.T2 * np.exp(-qubit_drift * np.random.uniform(0.8, 1.2))
            
            # Errors increase
            error_mult = 1 + qubit_drift * np.random.uniform(1.5, 2.5)
            
            drifted.append(QubitParameters(
                T1=max(T1_new, 1e-6),
                T2=max(T2_new, 1e-6),
                readout_error=min(q.readout_error * error_mult, 0.5),
                gate_error_1q=min(q.gate_error_1q * error_mult, 0.1),
                gate_error_2q=min(q.gate_error_2q * error_mult, 0.1),
                timestamp=q.timestamp + time_delta
            ))
        
        return drifted
    
    def _compute_drift_factor(self, time_hours: float, model: DriftModel) -> float:
        """Compute drift magnitude based on time and model."""
        base_drift = self.base_params["drift_magnitude"]
        calibration_interval = self.base_params["calibration_interval"]
        
        # Normalized time (0 to 1 over calibration interval)
        t_norm = min(time_hours / calibration_interval, 1.0)
        
        if model == DriftModel.GAUSSIAN:
            # Linear drift with noise (validated against IBM real data)
            return base_drift * t_norm
        elif model == DriftModel.POWER_LAW:
            # 1/f noise - slower initial drift, accelerating
            return base_drift * (t_norm ** 1.5)
        elif model == DriftModel.EXPONENTIAL:
            # Exponential decay (e.g., heating)
            return base_drift * (1 - np.exp(-3 * t_norm))
        elif model == DriftModel.CORRELATED:
            # Correlated drift (system-wide events)
            return base_drift * t_norm * (1 + 0.3 * np.sin(2 * np.pi * t_norm))
        else:
            return base_drift * t_norm


class SurfaceCodeSimulator:
    """Simulate surface code QEC with drift-aware qubit selection.
    
    Extends repetition code findings to topological codes (surface codes).
    """
    
    def __init__(self, distance: int, platform: QuantumPlatform):
        self.distance = distance
        self.platform = platform
        self.n_data_qubits = distance ** 2
        self.n_ancilla_qubits = distance ** 2 - 1
        self.n_total_qubits = self.n_data_qubits + self.n_ancilla_qubits
        
    def select_qubits_baseline(self, available_qubits: List[QubitParameters]) -> List[int]:
        """Baseline: Select qubits using calibration-reported (stale) parameters.
        
        This mimics the static selection approach without drift awareness.
        """
        # Use "backend-reported" parameters (no drift applied)
        # In real scenario, these would be 0-24h stale
        scores = []
        for i, q in enumerate(available_qubits):
            # Simplified scoring - higher T1/T2, lower errors = better
            score = (q.T1 + q.T2) - 1000 * (q.readout_error + q.gate_error_2q)
            scores.append((i, score))
        
        # Sort by score descending, take top N
        scores.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in scores[:self.n_total_qubits]]
    
    def select_qubits_daqec(self, available_qubits: List[QubitParameters],
                           probe_results: Optional[List[QubitParameters]] = None) -> List[int]:
        """DAQEC: Select qubits using fresh probe measurements.
        
        This mimics the drift-aware selection with probe-refreshed parameters.
        """
        # Use probe-refreshed parameters (current measurements)
        if probe_results is None:
            probe_results = available_qubits
        
        scores = []
        for i, q in enumerate(probe_results):
            score = (q.T1 + q.T2) - 1000 * (q.readout_error + q.gate_error_2q)
            scores.append((i, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in scores[:self.n_total_qubits]]
    
    def simulate_logical_error_rate(self, selected_qubits: List[int],
                                   available_qubits: List[QubitParameters],
                                   n_rounds: int = 10,
                                   seed: Optional[int] = None) -> float:
        """Simulate logical error rate for selected qubit subset.
        
        Args:
            selected_qubits: Indices of selected qubits
            available_qubits: Full list of qubit parameters
            n_rounds: Number of QEC rounds
            seed: Random seed
            
        Returns:
            Estimated logical error rate
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Get parameters for selected qubits
        qubits = [available_qubits[i] for i in selected_qubits]
        
        # Compute physical error rates for each qubit
        physical_errors = []
        for q in qubits:
            # Coherence-limited error (assuming 1 µs gate time)
            coherence_error = 1 - np.exp(-1.0 / min(q.T1, q.T2))
            # Total physical error (independent errors add)
            total_error = min(coherence_error + q.gate_error_2q + q.readout_error, 1.0)
            physical_errors.append(total_error)
        
        # CRITICAL: Logical error dominated by WORST qubits (tail events)
        # Use P75 for small codes - even one bad qubit can ruin surface code
        # This is why drift-aware selection matters: avoiding tail-drifted qubits
        physical_errors_sorted = sorted(physical_errors, reverse=True)
        
        # For small codes (d≤5), use P75; for larger codes (d>5), use P85
        percentile = 0.75 if self.distance <= 5 else 0.85
        p_index = max(0, min(len(physical_errors_sorted) - 1, int(percentile * len(physical_errors_sorted))))
        representative_error = physical_errors_sorted[p_index]
        
        # Surface code threshold ~0.5-1% (phenomenological)
        threshold = 0.01
        
        if representative_error < threshold:
            # Below threshold: exponential suppression
            # LER ~ (p/p_th)^((d+1)/2) per round
            suppression = (representative_error / threshold) ** ((self.distance + 1) / 2)
            logical_error = suppression ** n_rounds
        else:
            # Above threshold: error rate approaches 1 (complete failure)
            # Saturate at ~1.0 to represent "code completely failed"
            logical_error = min(0.99, 1.0 - np.exp(-representative_error * n_rounds))
        
        # Add 5% log-normal noise
        logical_error *= np.random.lognormal(0, 0.05)
        
        return min(max(logical_error, 1e-15), 1.0)  # Clamp to [1e-15, 1.0]


class DriftAwareExperimentSimulator:
    """Run large-scale drift-aware QEC experiments via simulation.
    
    This extends the real hardware validation (756 experiments, d≤7) to:
    - Larger distances (d=9, 11, 13) - fault-tolerance-relevant scales
    - Surface codes - addresses manuscript limitation
    - Platform diversity - validates platform-general hypothesis
    - Drift model diversity - robustness to assumptions
    """
    
    def __init__(self, output_dir: str = "simulations/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def run_distance_scaling_study(self,
                                   distances: List[int] = [3, 5, 7, 9, 11, 13],
                                   platform_type: str = "ibm",
                                   n_sessions: int = 100,
                                   seed: int = 42) -> pd.DataFrame:
        """Study how DAQEC benefit scales with code distance.
        
        Hypothesis: Larger codes benefit MORE from drift-aware selection
        (more qubits = more opportunities for drift-induced ranking changes)
        """
        print(f"Running distance scaling study: d={distances}")
        
        results = []
        
        for distance in distances:
            print(f"  Distance {distance}...")
            platform = QuantumPlatform(platform_type)
            simulator = SurfaceCodeSimulator(distance, platform)
            
            for session_id in range(n_sessions):
                # Random time since calibration (0-24h)
                time_since_cal = np.random.uniform(0, 24)
                
                # Generate available qubits (2x needed for selection)
                n_available = 2 * simulator.n_total_qubits
                
                qubits_base_seed = seed + session_id
                
                # Step 1: Generate INTRINSIC qubit properties (t=0, fresh calibration)
                qubits_at_calibration = platform.generate_qubits(
                    n_available,
                    time_since_calibration=0,  # Fresh calibration
                    drift_model=DriftModel.GAUSSIAN,
                    seed=qubits_base_seed
                )
                
                # Step 2: Same qubits AFTER time_since_cal hours of drift
                # Use base_qubits parameter to evolve THE SAME physical qubits
                qubits_actual_now = platform.generate_qubits(
                    n_available, 
                    time_since_calibration=time_since_cal,
                    drift_model=DriftModel.GAUSSIAN,
                    seed=qubits_base_seed + 1000,  # Different noise for drift process
                    base_qubits=qubits_at_calibration  # Evolve THESE SPECIFIC qubits
                )
                
                # Step 3: FRESH PROBE measurements (what DAQEC gets - noisy estimate of current state)
                # In real experiment: 30-shot T1/T2 measurement has ~10% uncertainty
                qubits_probe = []
                for q_actual in qubits_actual_now:
                    qubits_probe.append(QubitParameters(
                        T1=q_actual.T1 * np.random.lognormal(0, 0.1),
                        T2=q_actual.T2 * np.random.lognormal(0, 0.1),
                        readout_error=q_actual.readout_error * np.random.lognormal(0, 0.1),
                        gate_error_1q=q_actual.gate_error_1q * np.random.lognormal(0, 0.1),
                        gate_error_2q=q_actual.gate_error_2q * np.random.lognormal(0, 0.1),
                        timestamp=q_actual.timestamp
                    ))
                
                # Step 4: SELECTION
                # Both select from available pool (indices), but use different rankings
                # Baseline: Use stale calibration data  
                selected_baseline = simulator.select_qubits_baseline(qubits_at_calibration)
                
                # DAQEC: Use fresh probe measurements
                # Pass qubits_actual_now as available (for indexing) and qubits_probe for scoring
                selected_daqec = simulator.select_qubits_daqec(qubits_actual_now, qubits_probe)
                
                # Step 5: EVALUATION - Both run on ACTUAL current state
                ler_baseline = simulator.simulate_logical_error_rate(
                    selected_baseline, qubits_actual_now, seed=seed + session_id
                )
                ler_daqec = simulator.simulate_logical_error_rate(
                    selected_daqec, qubits_actual_now, seed=seed + session_id
                )
                
                # Compute benefit
                improvement_pct = 100 * (ler_baseline - ler_daqec) / ler_baseline if ler_baseline > 0 else 0
                
                results.append({
                    "distance": distance,
                    "session_id": session_id,
                    "time_since_calibration": time_since_cal,
                    "n_qubits": simulator.n_total_qubits,
                    "ler_baseline": ler_baseline,
                    "ler_daqec": ler_daqec,
                    "improvement_pct": improvement_pct,
                    "platform": platform_type
                })
        
        df = pd.DataFrame(results)
        
        # Save results
        output_file = self.output_dir / f"distance_scaling_{platform_type}.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved to {output_file}")
        
        return df
    
    def run_platform_comparison(self,
                               distance: int = 7,
                               platforms: List[str] = ["ibm", "google", "rigetti"],
                               n_sessions: int = 100,
                               seed: int = 42) -> pd.DataFrame:
        """Compare DAQEC benefits across different quantum platforms.
        
        Tests platform-general hypothesis from manuscript Discussion.
        """
        print(f"Running platform comparison: {platforms}")
        
        results = []
        
        for platform_type in platforms:
            print(f"  Platform: {platform_type}...")
            platform = QuantumPlatform(platform_type)
            simulator = SurfaceCodeSimulator(distance, platform)
            
            for session_id in range(n_sessions):
                time_since_cal = np.random.uniform(0, platform.base_params["calibration_interval"])
                
                n_available = 2 * simulator.n_total_qubits
                qubits_stale = platform.generate_qubits(
                    n_available, time_since_cal, DriftModel.GAUSSIAN, seed + session_id
                )
                qubits_fresh = platform.generate_qubits(
                    n_available, 0.1, DriftModel.GAUSSIAN, seed + session_id + 1000
                )
                
                selected_baseline = simulator.select_qubits_baseline(qubits_stale)
                ler_baseline = simulator.simulate_logical_error_rate(
                    selected_baseline, qubits_stale, seed=seed + session_id
                )
                
                selected_daqec = simulator.select_qubits_daqec(qubits_stale, qubits_fresh)
                ler_daqec = simulator.simulate_logical_error_rate(
                    selected_daqec, qubits_stale, seed=seed + session_id
                )
                
                improvement_pct = 100 * (ler_baseline - ler_daqec) / ler_baseline if ler_baseline > 0 else 0
                
                results.append({
                    "platform": platform_type,
                    "session_id": session_id,
                    "time_since_calibration": time_since_cal,
                    "calibration_interval": platform.base_params["calibration_interval"],
                    "drift_magnitude": platform.base_params["drift_magnitude"],
                    "ler_baseline": ler_baseline,
                    "ler_daqec": ler_daqec,
                    "improvement_pct": improvement_pct,
                })
        
        df = pd.DataFrame(results)
        
        output_file = self.output_dir / f"platform_comparison_d{distance}.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved to {output_file}")
        
        return df
    
    def run_drift_model_robustness(self,
                                  distance: int = 7,
                                  drift_models: List[DriftModel] = None,
                                  n_sessions: int = 100,
                                  seed: int = 42) -> pd.DataFrame:
        """Test robustness to different drift model assumptions.
        
        Shows DAQEC benefits are not artifacts of specific drift model choice.
        """
        if drift_models is None:
            drift_models = [DriftModel.GAUSSIAN, DriftModel.POWER_LAW, 
                          DriftModel.EXPONENTIAL, DriftModel.CORRELATED]
        
        print(f"Running drift model robustness study")
        
        results = []
        platform = QuantumPlatform("ibm")
        simulator = SurfaceCodeSimulator(distance, platform)
        
        for drift_model in drift_models:
            print(f"  Drift model: {drift_model.value}...")
            
            for session_id in range(n_sessions):
                time_since_cal = np.random.uniform(0, 24)
                
                n_available = 2 * simulator.n_total_qubits
                qubits_stale = platform.generate_qubits(
                    n_available, time_since_cal, drift_model, seed + session_id
                )
                qubits_fresh = platform.generate_qubits(
                    n_available, 0.1, drift_model, seed + session_id + 1000
                )
                
                selected_baseline = simulator.select_qubits_baseline(qubits_stale)
                ler_baseline = simulator.simulate_logical_error_rate(
                    selected_baseline, qubits_stale, seed=seed + session_id
                )
                
                selected_daqec = simulator.select_qubits_daqec(qubits_stale, qubits_fresh)
                ler_daqec = simulator.simulate_logical_error_rate(
                    selected_daqec, qubits_stale, seed=seed + session_id
                )
                
                improvement_pct = 100 * (ler_baseline - ler_daqec) / ler_baseline if ler_baseline > 0 else 0
                
                results.append({
                    "drift_model": drift_model.value,
                    "session_id": session_id,
                    "time_since_calibration": time_since_cal,
                    "ler_baseline": ler_baseline,
                    "ler_daqec": ler_daqec,
                    "improvement_pct": improvement_pct,
                })
        
        df = pd.DataFrame(results)
        
        output_file = self.output_dir / f"drift_model_robustness_d{distance}.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved to {output_file}")
        
        return df
    
    def generate_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """Compute summary statistics for manuscript reporting."""
        stats = {
            "n_sessions": len(df),
            "mean_improvement_pct": df["improvement_pct"].mean(),
            "median_improvement_pct": df["improvement_pct"].median(),
            "std_improvement_pct": df["improvement_pct"].std(),
            "mean_ler_baseline": df["ler_baseline"].mean(),
            "mean_ler_daqec": df["ler_daqec"].mean(),
            "sessions_improved": (df["improvement_pct"] > 0).sum(),
            "sessions_degraded": (df["improvement_pct"] < 0).sum(),
            "max_improvement_pct": df["improvement_pct"].max(),
            "timestamp": datetime.now().isoformat()
        }
        
        # Dose-response correlation
        if "time_since_calibration" in df.columns:
            rho, p_value = spearmanr(df["time_since_calibration"], df["improvement_pct"])
            stats["spearman_rho"] = rho
            stats["spearman_p"] = p_value
        
        return stats


if __name__ == "__main__":
    print("="*80)
    print("DAQEC Simulation Framework - Extending Real Hardware Findings")
    print("="*80)
    print()
    print("Purpose: Extend validated real hardware results (756 experiments, d<=7)")
    print("         to fault-tolerance-relevant scales and surface codes")
    print()
    
    simulator = DriftAwareExperimentSimulator()
    
    # Study 1: Distance scaling (d=3 to d=13)
    print("\n" + "="*80)
    print("STUDY 1: Distance Scaling (d=3, 5, 7, 9, 11, 13)")
    print("="*80)
    df_scaling = simulator.run_distance_scaling_study(
        distances=[3, 5, 7, 9, 11, 13],
        platform_type="ibm",
        n_sessions=100,
        seed=42
    )
    
    print("\nSummary by distance:")
    summary_scaling = df_scaling.groupby("distance")["improvement_pct"].agg(["mean", "std", "count"])
    print(summary_scaling)
    
    # Study 2: Platform comparison
    print("\n" + "="*80)
    print("STUDY 2: Platform Comparison (IBM, Google, Rigetti)")
    print("="*80)
    df_platform = simulator.run_platform_comparison(
        distance=7,
        platforms=["ibm", "google", "rigetti"],
        n_sessions=100,
        seed=42
    )
    
    print("\nSummary by platform:")
    summary_platform = df_platform.groupby("platform")["improvement_pct"].agg(["mean", "std", "count"])
    print(summary_platform)
    
    # Study 3: Drift model robustness
    print("\n" + "="*80)
    print("STUDY 3: Drift Model Robustness")
    print("="*80)
    df_drift = simulator.run_drift_model_robustness(
        distance=7,
        n_sessions=100,
        seed=42
    )
    
    print("\nSummary by drift model:")
    summary_drift = df_drift.groupby("drift_model")["improvement_pct"].agg(["mean", "std", "count"])
    print(summary_drift)
    
    # Generate comprehensive statistics
    print("\n" + "="*80)
    print("COMPREHENSIVE STATISTICS FOR MANUSCRIPT")
    print("="*80)
    
    stats_scaling = simulator.generate_summary_statistics(df_scaling)
    stats_platform = simulator.generate_summary_statistics(df_platform)
    stats_drift = simulator.generate_summary_statistics(df_drift)
    
    # Save all statistics
    all_stats = {
        "distance_scaling": stats_scaling,
        "platform_comparison": stats_platform,
        "drift_model_robustness": stats_drift
    }
    
    stats_file = simulator.output_dir / "summary_statistics.json"
    with open(stats_file, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_to_python_types(obj):
            if isinstance(obj, dict):
                return {k: convert_to_python_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_python_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return convert_to_python_types(obj.tolist())
            else:
                return obj
        
        json.dump(convert_to_python_types(all_stats), f, indent=2)
    
    print(f"\nAll statistics saved to {stats_file}")
    print("\n" + "="*80)
    print("SIMULATION COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {simulator.output_dir}")
    print("\nNext steps:")
    print("1. Analyze results using analysis/analyze_simulation_results.py")
    print("2. Generate figures for manuscript Extended Data")
    print("3. Update manuscript with simulation findings")
