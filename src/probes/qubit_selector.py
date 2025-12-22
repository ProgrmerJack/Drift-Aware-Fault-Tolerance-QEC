"""
Module B: Qubit Selection Strategies
====================================

Implements three selection tiers as per the roadmap:
1. Static: Select top-k qubits from daily backend properties
2. Real-Time (RT): Re-select based on 30-shot probes at run time
3. Drift-Aware: Use rolling z-scores and change-point detection for layout

References:
- IBM Quantum Documentation: Get backend information
- Qiskit Transpiler: Generate preset pass managers
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QubitSelector:
    """
    Multi-tier qubit selection based on calibration and drift data.
    
    Implements the three selection strategies from the research roadmap:
    1. Static: Backend properties only
    2. RT: Real-time probes before experiment
    3. Drift-Aware: Historical drift analysis + probes
    """
    
    def __init__(self, num_qubits: int = 127):
        """
        Initialize the qubit selector.
        
        Args:
            num_qubits: Total number of qubits on the backend
        """
        self.num_qubits = num_qubits
        
    def static_selection(self, backend_properties: Dict[str, Any],
                         num_qubits: int,
                         connectivity_required: bool = True,
                         metric_weights: Optional[Dict[str, float]] = None) -> List[int]:
        """
        Select qubits based on static backend properties.
        
        This is the simplest selection strategy, using only the latest
        calibration data from backend.properties().
        
        Args:
            backend_properties: Backend properties snapshot
            num_qubits: Number of qubits to select
            connectivity_required: Whether selected qubits must be connected
            metric_weights: Weights for different metrics (default: equal)
            
        Returns:
            List of selected qubit indices
        """
        if metric_weights is None:
            metric_weights = {
                "T1": 0.3,
                "T2": 0.3,
                "readout_error": 0.2,
                "gate_error": 0.2
            }
            
        # Score each qubit
        qubit_scores = {}
        qubits_data = backend_properties.get("qubits", {})
        
        for qubit_idx_str, qubit_data in qubits_data.items():
            qubit_idx = int(qubit_idx_str)
            score = 0.0
            
            # T1 (higher is better)
            t1 = qubit_data.get("T1", {}).get("value", 0)
            if t1 > 0:
                score += metric_weights.get("T1", 0) * (t1 / 1e-4)  # Normalize to ~100us scale
                
            # T2 (higher is better)
            t2 = qubit_data.get("T2", {}).get("value", 0)
            if t2 > 0:
                score += metric_weights.get("T2", 0) * (t2 / 1e-4)
                
            # Readout error (lower is better)
            ro_error = qubit_data.get("readout_error", {}).get("value", 1)
            if ro_error < 1:
                score += metric_weights.get("readout_error", 0) * (1 - ro_error)
                
            qubit_scores[qubit_idx] = score
            
        # Sort by score (descending)
        sorted_qubits = sorted(qubit_scores.keys(), 
                               key=lambda q: qubit_scores[q], 
                               reverse=True)
        
        if connectivity_required:
            # Find connected subset
            coupling_map = backend_properties.get("coupling_map", [])
            selected = self._find_connected_subset(sorted_qubits, coupling_map, num_qubits)
        else:
            selected = sorted_qubits[:num_qubits]
            
        logger.info(f"Static selection: {selected}")
        return selected
    
    def rt_selection(self, probe_results: Dict[int, Dict[str, Any]],
                     num_qubits: int,
                     backend_properties: Optional[Dict[str, Any]] = None) -> List[int]:
        """
        Real-time selection based on 30-shot probe results.
        
        Uses recent probe data to select qubits with best current performance,
        overriding static properties if they differ significantly.
        
        Args:
            probe_results: Dictionary mapping qubit index to probe results
            num_qubits: Number of qubits to select
            backend_properties: Optional backend properties for connectivity
            
        Returns:
            List of selected qubit indices
        """
        qubit_scores = {}
        
        for qubit_idx, results in probe_results.items():
            score = 0.0
            probes = results.get("probes", {})
            
            # T1 quality (if available)
            if "t1" in probes:
                t1_data = probes["t1"]
                # Higher survival at longer delays = better T1
                decay_curve = t1_data.get("decay_curve", [])
                if decay_curve:
                    # Use final point's survival as proxy
                    final_survival = decay_curve[-1].get("p1", 0) if decay_curve else 0
                    score += 0.3 * final_survival
                    
            # Readout quality
            if "readout" in probes:
                ro_data = probes["readout"]
                fidelity = ro_data.get("assignment_fidelity", 0)
                score += 0.4 * fidelity
                
            # RB quality
            if "rb" in probes:
                rb_data = probes["rb"]
                decay_points = rb_data.get("decay_points", [])
                if decay_points:
                    # Use final survival probability
                    final_survival = decay_points[-1].get("survival_probability", 0)
                    score += 0.3 * final_survival
                    
            qubit_scores[qubit_idx] = score
            
        # Sort by score
        sorted_qubits = sorted(qubit_scores.keys(),
                               key=lambda q: qubit_scores[q],
                               reverse=True)
        
        selected = sorted_qubits[:num_qubits]
        logger.info(f"RT selection: {selected}")
        return selected
    
    def drift_aware_selection(self, 
                               drift_report: Dict[str, Any],
                               probe_results: Dict[int, Dict[str, Any]],
                               num_qubits: int,
                               stability_threshold: float = 0.8,
                               zscore_penalty: float = 0.5) -> List[int]:
        """
        Drift-aware selection combining historical drift analysis with probes.
        
        This is the most sophisticated selection strategy, incorporating:
        - Rolling z-scores from drift analysis
        - Stability scores from historical data
        - Real-time probe results
        - Change-point detection for recent calibration shifts
        
        Args:
            drift_report: Output from DriftAnalyzer.generate_drift_report()
            probe_results: Recent probe results
            num_qubits: Number of qubits to select
            stability_threshold: Minimum stability score to consider
            zscore_penalty: Penalty factor for high z-scores
            
        Returns:
            List of selected qubit indices
        """
        qubit_scores = {}
        
        # Get probe-based scores
        for qubit_idx, results in probe_results.items():
            score = 0.0
            probes = results.get("probes", {})
            
            # Base score from probes (same as RT selection)
            if "readout" in probes:
                ro_data = probes["readout"]
                score += 0.3 * ro_data.get("assignment_fidelity", 0)
            if "rb" in probes:
                rb_data = probes["rb"]
                decay_points = rb_data.get("decay_points", [])
                if decay_points:
                    score += 0.3 * decay_points[-1].get("survival_probability", 0)
                    
            qubit_scores[qubit_idx] = score
            
        # Adjust based on drift analysis
        qubit_drift = drift_report.get("qubit_drift", {})
        
        for qubit_key, drift_data in qubit_drift.items():
            qubit_idx = int(qubit_key.replace("q", ""))
            
            if qubit_idx not in qubit_scores:
                continue
                
            # Check T1 stability
            if "T1" in drift_data:
                t1_features = drift_data["T1"].get("features", {})
                stability = t1_features.get("stability_score", 1.0)
                zscore = abs(t1_features.get("rolling_zscore", 0))
                
                # Bonus for stable qubits
                if stability > stability_threshold:
                    qubit_scores[qubit_idx] += 0.2
                    
                # Penalty for high z-scores (recent drift)
                if zscore > 2.0:
                    qubit_scores[qubit_idx] -= zscore_penalty * (zscore - 2.0)
                    
                # Penalty if change point detected recently
                change_points = drift_data["T1"].get("change_points", [])
                if change_points:
                    qubit_scores[qubit_idx] -= 0.1
                    
            # Similar for T2
            if "T2" in drift_data:
                t2_features = drift_data["T2"].get("features", {})
                stability = t2_features.get("stability_score", 1.0)
                zscore = abs(t2_features.get("rolling_zscore", 0))
                
                if stability > stability_threshold:
                    qubit_scores[qubit_idx] += 0.2
                if zscore > 2.0:
                    qubit_scores[qubit_idx] -= zscore_penalty * (zscore - 2.0)
                    
        # Sort by adjusted score
        sorted_qubits = sorted(qubit_scores.keys(),
                               key=lambda q: qubit_scores[q],
                               reverse=True)
        
        selected = sorted_qubits[:num_qubits]
        logger.info(f"Drift-aware selection: {selected}")
        return selected
    
    def _find_connected_subset(self, sorted_qubits: List[int],
                                coupling_map: List[Tuple[int, int]],
                                num_qubits: int) -> List[int]:
        """
        Find a connected subset of qubits from the sorted list.
        
        Uses a greedy algorithm to grow a connected component.
        
        Args:
            sorted_qubits: Qubits sorted by score (best first)
            coupling_map: List of (q1, q2) edges
            num_qubits: Target number of qubits
            
        Returns:
            List of connected qubit indices
        """
        if not coupling_map:
            return sorted_qubits[:num_qubits]
            
        # Build adjacency dict
        adj = {}
        for q1, q2 in coupling_map:
            if q1 not in adj:
                adj[q1] = set()
            if q2 not in adj:
                adj[q2] = set()
            adj[q1].add(q2)
            adj[q2].add(q1)
            
        # Start with best qubit
        selected = [sorted_qubits[0]]
        selected_set = {sorted_qubits[0]}
        
        # Greedily add connected qubits
        while len(selected) < num_qubits:
            # Find best unselected qubit connected to current selection
            best_candidate = None
            
            for q in sorted_qubits:
                if q in selected_set:
                    continue
                    
                # Check if connected to any selected qubit
                if q in adj and any(n in selected_set for n in adj[q]):
                    best_candidate = q
                    break
                    
            if best_candidate is None:
                # No more connected qubits available
                logger.warning(f"Could only find {len(selected)} connected qubits")
                break
                
            selected.append(best_candidate)
            selected_set.add(best_candidate)
            
        return selected
    
    def compare_strategies(self, 
                           backend_properties: Dict[str, Any],
                           probe_results: Dict[int, Dict[str, Any]],
                           drift_report: Dict[str, Any],
                           num_qubits: int) -> Dict[str, List[int]]:
        """
        Compare all three selection strategies.
        
        Useful for analysis of how different strategies differ.
        
        Args:
            backend_properties: Backend calibration snapshot
            probe_results: Recent probe results
            drift_report: Drift analysis report
            num_qubits: Number of qubits to select
            
        Returns:
            Dictionary mapping strategy name to selected qubits
        """
        results = {
            "static": self.static_selection(backend_properties, num_qubits),
            "rt": self.rt_selection(probe_results, num_qubits, backend_properties),
            "drift_aware": self.drift_aware_selection(drift_report, probe_results, num_qubits)
        }
        
        # Compute overlap
        static_set = set(results["static"])
        rt_set = set(results["rt"])
        drift_set = set(results["drift_aware"])
        
        logger.info(f"Static vs RT overlap: {len(static_set & rt_set)}/{num_qubits}")
        logger.info(f"Static vs Drift overlap: {len(static_set & drift_set)}/{num_qubits}")
        logger.info(f"RT vs Drift overlap: {len(rt_set & drift_set)}/{num_qubits}")
        
        return results


class RepetitionCodeLayoutGenerator:
    """
    Generate qubit layouts specifically for repetition code experiments.
    
    Finds linear chains of qubits for distance-3, 5, 7 repetition codes.
    """
    
    def __init__(self, coupling_map: List[Tuple[int, int]]):
        """
        Initialize layout generator.
        
        Args:
            coupling_map: List of (q1, q2) edges from backend
        """
        self.coupling_map = coupling_map
        self.adj = self._build_adjacency()
        
    def _build_adjacency(self) -> Dict[int, set]:
        """Build adjacency dictionary from coupling map."""
        adj = {}
        for q1, q2 in self.coupling_map:
            if q1 not in adj:
                adj[q1] = set()
            if q2 not in adj:
                adj[q2] = set()
            adj[q1].add(q2)
            adj[q2].add(q1)
        return adj
    
    def find_linear_chains(self, length: int, 
                           qubit_scores: Optional[Dict[int, float]] = None) -> List[List[int]]:
        """
        Find all linear chains of the specified length.
        
        For repetition codes, we need chains where each qubit is connected
        to exactly 2 neighbors (except endpoints).
        
        Args:
            length: Required chain length
            qubit_scores: Optional scores for ranking chains
            
        Returns:
            List of qubit chains, sorted by quality if scores provided
        """
        chains = []
        visited_chains = set()
        
        # Try starting from each qubit
        for start in self.adj.keys():
            self._find_chains_from(start, [start], length, chains, visited_chains)
            
        # Sort chains by total score if provided
        if qubit_scores:
            chains.sort(key=lambda c: sum(qubit_scores.get(q, 0) for q in c), reverse=True)
            
        return chains
    
    def _find_chains_from(self, current: int, path: List[int], 
                          target_length: int, chains: List[List[int]],
                          visited_chains: set):
        """Recursive helper to find chains."""
        if len(path) == target_length:
            # Store canonical form to avoid duplicates
            canonical = tuple(path) if path[0] < path[-1] else tuple(reversed(path))
            if canonical not in visited_chains:
                visited_chains.add(canonical)
                chains.append(list(path))
            return
            
        # Extend chain
        for neighbor in self.adj.get(current, set()):
            if neighbor not in path:
                self._find_chains_from(neighbor, path + [neighbor], 
                                       target_length, chains, visited_chains)
    
    def select_best_layout(self, distance: int,
                           qubit_scores: Dict[int, float]) -> List[int]:
        """
        Select the best layout for a repetition code of given distance.
        
        Args:
            distance: Code distance (3, 5, or 7)
            qubit_scores: Quality scores for each qubit
            
        Returns:
            Best qubit layout as list of indices
        """
        # Distance-d repetition code needs 2d-1 physical qubits
        # (d data qubits + d-1 ancilla qubits interleaved)
        num_qubits = 2 * distance - 1
        
        chains = self.find_linear_chains(num_qubits, qubit_scores)
        
        if not chains:
            logger.warning(f"No valid chains found for distance-{distance} code")
            return []
            
        return chains[0]


if __name__ == "__main__":
    print("Qubit Selector Module")
    print("\nExample usage:")
    print("""
    from src.calibration import CalibrationCollector, DriftAnalyzer
    from src.probes import ProbeSuite
    
    # Load data
    collector = CalibrationCollector()
    snapshots = collector.load_snapshots("ibm_sherbrooke")
    
    # Analyze drift
    analyzer = DriftAnalyzer()
    drift_report = analyzer.generate_drift_report(snapshots, "ibm_sherbrooke")
    
    # Get latest properties
    latest_props = snapshots[-1] if snapshots else {}
    
    # Run probes on candidate qubits
    suite = ProbeSuite(service, "ibm_sherbrooke")
    probe_results = {}
    for q in range(20):  # Probe first 20 qubits
        probe_results[q] = suite.run_probes(q)
    
    # Select qubits using different strategies
    selector = QubitSelector()
    comparison = selector.compare_strategies(
        latest_props, probe_results, drift_report, num_qubits=7
    )
    print("Selection comparison:", comparison)
    """)
