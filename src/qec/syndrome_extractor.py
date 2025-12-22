"""
Syndrome Extractor
==================

Utilities for extracting and analyzing syndrome data from
QEC experiment results.

Syndromes are the key observable for:
- Detecting errors
- Feeding decoders
- Analyzing correlated error events (cosmic rays, TLS defects)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SyndromeData:
    """
    Container for syndrome data from a QEC experiment.
    
    Attributes:
        syndromes: Raw syndrome matrix (shots, rounds, n_detectors)
        detections: Detection events - syndrome flips (shots, rounds, n_detectors)
        logical_outcomes: Logical measurement results (shots,)
        distance: Code distance
        rounds: Number of syndrome rounds
        code_type: 'bit_flip' or 'phase_flip'
    """
    
    syndromes: np.ndarray  # (shots, rounds, n_detectors)
    detections: np.ndarray  # (shots, rounds, n_detectors)
    logical_outcomes: np.ndarray  # (shots,)
    distance: int
    rounds: int
    code_type: str
    
    @property
    def n_shots(self) -> int:
        """Number of shots in the data."""
        return self.syndromes.shape[0]
    
    @property
    def n_detectors(self) -> int:
        """Number of syndrome detectors per round."""
        return self.syndromes.shape[2]
    
    @property
    def total_detections(self) -> int:
        """Total number of detection events across all shots."""
        return int(np.sum(self.detections))
    
    @property
    def detection_rate(self) -> float:
        """Average detection rate per detector per round."""
        return float(np.mean(self.detections))
    
    def get_shot_syndrome(self, shot_idx: int) -> np.ndarray:
        """Get syndrome matrix for a single shot."""
        return self.syndromes[shot_idx]
    
    def get_shot_detections(self, shot_idx: int) -> np.ndarray:
        """Get detection events for a single shot."""
        return self.detections[shot_idx]


class SyndromeExtractor:
    """
    Extracts and processes syndrome data for analysis and decoding.
    
    Key concepts:
    - Syndromes: Raw measurement outcomes from ancilla qubits
    - Detections: Syndrome changes (flips) between rounds
    - Detection events feed into decoders (MWPM, etc.)
    
    Example:
        >>> extractor = SyndromeExtractor()
        >>> syndrome_data = extractor.extract(qec_result)
        >>> stats = extractor.compute_statistics(syndrome_data)
    """
    
    def __init__(self):
        """Initialize the syndrome extractor."""
        pass
    
    def extract_from_counts(
        self,
        counts: Dict[str, int],
        distance: int,
        rounds: int,
        code_type: str = "bit_flip",
    ) -> SyndromeData:
        """
        Extract syndrome data from measurement counts.
        
        Args:
            counts: Measurement outcomes from Sampler
            distance: Code distance
            rounds: Number of syndrome rounds
            code_type: 'bit_flip' or 'phase_flip'
            
        Returns:
            SyndromeData with extracted syndromes and detections
        """
        n_detectors = distance - 1
        n_data = distance
        
        # Expand counts to individual shots
        shots_data = []
        for outcome, count in counts.items():
            for _ in range(count):
                shots_data.append(outcome)
        
        n_shots = len(shots_data)
        
        syndromes = np.zeros((n_shots, rounds, n_detectors), dtype=int)
        logical_outcomes = np.zeros(n_shots, dtype=int)
        
        for shot_idx, outcome in enumerate(shots_data):
            # Parse outcome string
            # Expected format: "final_bits syn_r{n-1} ... syn_r1 syn_r0"
            # Or continuous string that needs to be parsed
            
            parts = outcome.replace(" ", "")  # Remove spaces
            
            # Extract bits
            bits = [int(b) for b in parts]
            
            # Final measurement is first n_data bits
            if len(bits) >= n_data:
                final_bits = bits[:n_data]
                # Majority vote for logical outcome
                logical_outcomes[shot_idx] = 1 if sum(final_bits) > n_data // 2 else 0
                
                # Remaining bits are syndromes
                syndrome_bits = bits[n_data:]
                
                # Parse syndrome rounds
                for r in range(rounds):
                    start_idx = r * n_detectors
                    end_idx = start_idx + n_detectors
                    if end_idx <= len(syndrome_bits):
                        syndromes[shot_idx, rounds - 1 - r, :] = syndrome_bits[start_idx:end_idx]
        
        # Compute detection events (syndrome flips)
        detections = self._compute_detections(syndromes)
        
        return SyndromeData(
            syndromes=syndromes,
            detections=detections,
            logical_outcomes=logical_outcomes,
            distance=distance,
            rounds=rounds,
            code_type=code_type,
        )
    
    def _compute_detections(self, syndromes: np.ndarray) -> np.ndarray:
        """
        Compute detection events from syndromes.
        
        Detection events are XOR of consecutive syndrome values:
        - First round: compare to assumed-zero initial state
        - Subsequent rounds: compare to previous round
        
        Args:
            syndromes: Array of shape (shots, rounds, n_detectors)
        
        Returns:
            Array of shape (shots, rounds, n_detectors) with detection events
        """
        shots, rounds, n_det = syndromes.shape
        detections = np.zeros_like(syndromes)
        
        # First round: detection = syndrome (compared to zero)
        detections[:, 0, :] = syndromes[:, 0, :]
        
        # Subsequent rounds: detection = XOR of consecutive syndromes
        for r in range(1, rounds):
            detections[:, r, :] = syndromes[:, r, :] ^ syndromes[:, r - 1, :]
        
        return detections
    
    def compute_statistics(self, data: SyndromeData) -> Dict[str, Any]:
        """
        Compute comprehensive statistics on syndrome data.
        
        Args:
            data: SyndromeData object
        
        Returns:
            Dictionary with various statistics
        """
        stats = {}
        
        # Basic rates
        stats['detection_rate'] = data.detection_rate
        stats['syndrome_rate'] = float(np.mean(data.syndromes))
        
        # Per-detector rates
        stats['per_detector_rate'] = np.mean(data.detections, axis=(0, 1)).tolist()
        
        # Per-round rates
        stats['per_round_rate'] = np.mean(data.detections, axis=(0, 2)).tolist()
        
        # Burstiness metrics
        stats['burstiness'] = self._compute_burstiness(data)
        
        # Spatial clustering
        stats['spatial_clustering'] = self._compute_spatial_clustering(data)
        
        # Temporal clustering
        stats['temporal_clustering'] = self._compute_temporal_clustering(data)
        
        # Correlation with logical errors
        stats['error_correlation'] = self._compute_error_correlation(data)
        
        # Logical error rate
        stats['logical_error_rate'] = float(np.mean(data.logical_outcomes))
        
        return stats
    
    def _compute_burstiness(self, data: SyndromeData) -> Dict[str, float]:
        """
        Compute burstiness metrics (deviation from Poisson).
        
        Fano factor > 1 indicates clustering/correlation.
        """
        # Total detections per shot
        total_per_shot = np.sum(data.detections, axis=(1, 2))
        
        mean = np.mean(total_per_shot)
        var = np.var(total_per_shot)
        
        fano = var / mean if mean > 0 else 0
        
        # Index of dispersion (another measure)
        iod = (var - mean) / mean if mean > 0 else 0
        
        return {
            'fano_factor': float(fano),
            'index_of_dispersion': float(iod),
            'mean_detections': float(mean),
            'var_detections': float(var),
        }
    
    def _compute_spatial_clustering(self, data: SyndromeData) -> Dict[str, Any]:
        """
        Analyze spatial clustering of detection events.
        
        Adjacent detections in the same round indicate potential
        correlated errors.
        """
        n_shots, n_rounds, n_det = data.detections.shape
        
        # Count adjacent pairs of detections in same round
        adjacent_pairs = 0
        total_detections = 0
        
        for shot in range(n_shots):
            for r in range(n_rounds):
                det_positions = np.where(data.detections[shot, r, :] == 1)[0]
                total_detections += len(det_positions)
                
                # Check for adjacent pairs
                for i in range(len(det_positions) - 1):
                    if det_positions[i + 1] - det_positions[i] == 1:
                        adjacent_pairs += 1
        
        # Expected adjacent pairs under independence
        p_det = data.detection_rate
        expected_pairs = n_shots * n_rounds * (n_det - 1) * p_det ** 2
        
        return {
            'adjacent_pairs': int(adjacent_pairs),
            'expected_pairs': float(expected_pairs),
            'clustering_ratio': float(adjacent_pairs / expected_pairs) if expected_pairs > 0 else 0,
        }
    
    def _compute_temporal_clustering(self, data: SyndromeData) -> Dict[str, Any]:
        """
        Analyze temporal clustering of detection events.
        
        Consecutive detections at the same position across rounds
        indicate persistent errors or measurement errors.
        """
        n_shots, n_rounds, n_det = data.detections.shape
        
        # Count consecutive detections at same position
        consecutive_pairs = 0
        
        for shot in range(n_shots):
            for d in range(n_det):
                for r in range(n_rounds - 1):
                    if data.detections[shot, r, d] == 1 and data.detections[shot, r + 1, d] == 1:
                        consecutive_pairs += 1
        
        # Expected under independence
        p_det = data.detection_rate
        expected_pairs = n_shots * n_det * (n_rounds - 1) * p_det ** 2
        
        return {
            'consecutive_pairs': int(consecutive_pairs),
            'expected_pairs': float(expected_pairs),
            'clustering_ratio': float(consecutive_pairs / expected_pairs) if expected_pairs > 0 else 0,
        }
    
    def _compute_error_correlation(self, data: SyndromeData) -> Dict[str, float]:
        """
        Analyze correlation between detection count and logical errors.
        """
        total_detections = np.sum(data.detections, axis=(1, 2))
        
        # Correlation coefficient
        if np.std(total_detections) > 0 and np.std(data.logical_outcomes) > 0:
            corr = np.corrcoef(total_detections, data.logical_outcomes)[0, 1]
        else:
            corr = 0
        
        # Error rate by detection count
        unique_counts = np.unique(total_detections)
        error_by_count = {}
        
        for count in unique_counts[:20]:  # Limit to first 20 values
            mask = total_detections == count
            if np.sum(mask) > 10:  # Need sufficient samples
                error_by_count[int(count)] = float(np.mean(data.logical_outcomes[mask]))
        
        return {
            'correlation': float(corr),
            'error_by_detection_count': error_by_count,
        }
    
    def find_burst_events(
        self,
        data: SyndromeData,
        threshold: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Find "burst" events with many detections in a short window.
        
        These bursts are candidates for correlated error events
        (e.g., cosmic rays, TLS defects).
        
        Args:
            data: SyndromeData object
            threshold: Minimum detections per round to flag as burst
        
        Returns:
            List of burst event descriptors
        """
        bursts = []
        
        for shot in range(data.n_shots):
            for r in range(data.rounds):
                det_count = np.sum(data.detections[shot, r, :])
                
                if det_count >= threshold:
                    det_positions = np.where(data.detections[shot, r, :] == 1)[0]
                    
                    bursts.append({
                        'shot': int(shot),
                        'round': int(r),
                        'count': int(det_count),
                        'positions': det_positions.tolist(),
                        'is_adjacent': self._check_adjacent(det_positions),
                        'logical_error': int(data.logical_outcomes[shot]),
                    })
        
        return bursts
    
    def _check_adjacent(self, positions: np.ndarray) -> bool:
        """Check if positions form a contiguous cluster."""
        if len(positions) <= 1:
            return True
        
        sorted_pos = np.sort(positions)
        gaps = np.diff(sorted_pos)
        
        return np.all(gaps == 1)
    
    def get_decoder_input(
        self,
        data: SyndromeData,
        shot: int,
    ) -> Dict[str, np.ndarray]:
        """
        Get the input needed for a decoder for a specific shot.
        
        Args:
            data: SyndromeData object
            shot: Shot index
        
        Returns:
            Dictionary with decoder inputs
        """
        return {
            'syndromes': data.syndromes[shot],
            'detections': data.detections[shot],
            'n_rounds': data.rounds,
            'n_detectors': data.n_detectors,
            'distance': data.distance,
        }
    
    def to_stim_format(
        self,
        data: SyndromeData,
    ) -> np.ndarray:
        """
        Convert detection events to Stim-compatible format.
        
        This enables using Stim's MWPM decoder (PyMatching).
        
        Args:
            data: SyndromeData object
            
        Returns:
            Detection events in Stim format (shots, n_detectors * rounds)
        """
        # Flatten detections to (shots, rounds * n_detectors)
        flat = data.detections.reshape(data.n_shots, -1)
        return flat.astype(np.uint8)
    
    def analyze_error_model(
        self,
        data: SyndromeData,
    ) -> Dict[str, Any]:
        """
        Analyze the error model from syndrome statistics.
        
        Extracts parameters that characterize the noise:
        - Physical error rate estimate
        - Measurement error contribution
        - Correlation strength
        
        Args:
            data: SyndromeData object
            
        Returns:
            Error model parameters
        """
        stats = self.compute_statistics(data)
        
        # Estimate physical error rate from detection rate
        # For repetition code: p_detection ≈ 2 * p_physical
        p_det = stats['detection_rate']
        p_physical_est = p_det / 2
        
        # Estimate measurement error from temporal clustering
        temporal = stats['temporal_clustering']
        # High temporal clustering suggests measurement errors
        p_meas_est = temporal['clustering_ratio'] - 1 if temporal['clustering_ratio'] > 1 else 0
        p_meas_est = max(0, min(p_meas_est * p_det, 0.5))  # Bounded
        
        # Correlation strength from spatial clustering
        spatial = stats['spatial_clustering']
        correlation_strength = spatial['clustering_ratio'] - 1 if spatial['clustering_ratio'] > 1 else 0
        
        # Logical error rate vs. expected
        # For d-round repetition code: p_L ≈ (p * rounds)^((d+1)/2) for low p
        p_L = stats['logical_error_rate']
        
        return {
            'physical_error_estimate': float(p_physical_est),
            'measurement_error_estimate': float(p_meas_est),
            'correlation_strength': float(correlation_strength),
            'logical_error_rate': float(p_L),
            'fano_factor': stats['burstiness']['fano_factor'],
            'detection_rate': float(p_det),
        }
