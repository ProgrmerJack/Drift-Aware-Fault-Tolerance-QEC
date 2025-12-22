"""
Drift Tracker
=============

Analyzes calibration snapshots to detect drift patterns, compute
drift features, and identify change points in qubit properties.

Uses the ruptures library for change-point detection and provides
features for drift-aware qubit selection.

References:
- ruptures: https://centre-borelli.github.io/ruptures-docs/
- Change-point detection in time series: Truong et al., Signal Processing (2020)
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False
    rpt = None

from .schemas import CalibrationSnapshot, QubitProperties

logger = logging.getLogger(__name__)


class DriftTracker:
    """
    Tracks and analyzes calibration drift over time.
    
    Provides drift features for qubit selection:
    - Rolling z-scores (recent outliers)
    - Stability scores (coefficient of variation)
    - Change-point detection (calibration shifts)
    - Cross-qubit correlations (crosstalk proxy)
    
    Example:
        >>> tracker = DriftTracker()
        >>> tracker.load_snapshots("data/calibration/ibm_brisbane/")
        >>> features = tracker.compute_drift_features(qubit=42)
        >>> change_points = tracker.detect_change_points(qubit=42, property="T1")
    """
    
    def __init__(self, window_days: int = 7, min_snapshots: int = 3):
        """
        Initialize the drift tracker.
        
        Args:
            window_days: Rolling window size for drift analysis
            min_snapshots: Minimum snapshots required for analysis
        """
        self.window_days = window_days
        self.min_snapshots = min_snapshots
        self.snapshots: List[CalibrationSnapshot] = []
        self._df_cache: Optional[pd.DataFrame] = None
    
    def load_snapshots(self, 
                       directory: Union[str, Path],
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None) -> int:
        """
        Load calibration snapshots from a directory.
        
        Args:
            directory: Directory containing snapshot JSON files
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Number of snapshots loaded
        """
        directory = Path(directory)
        
        if not directory.exists():
            logger.warning(f"Directory not found: {directory}")
            return 0
        
        self.snapshots = []
        self._df_cache = None
        
        for filepath in sorted(directory.glob("*.json")):
            try:
                snapshot = CalibrationSnapshot.load(filepath)
                
                # Apply date filters
                if start_date and snapshot.collection_timestamp < start_date:
                    continue
                if end_date and snapshot.collection_timestamp > end_date:
                    continue
                
                self.snapshots.append(snapshot)
            except Exception as e:
                logger.warning(f"Failed to load {filepath}: {e}")
        
        logger.info(f"Loaded {len(self.snapshots)} snapshots from {directory}")
        return len(self.snapshots)
    
    def add_snapshot(self, snapshot: CalibrationSnapshot) -> None:
        """Add a snapshot to the tracker."""
        self.snapshots.append(snapshot)
        self._df_cache = None  # Invalidate cache
    
    def _build_dataframe(self) -> pd.DataFrame:
        """Build a DataFrame from all snapshots for analysis."""
        if self._df_cache is not None:
            return self._df_cache
        
        records = []
        for snapshot in self.snapshots:
            for qubit_idx, props in snapshot.qubit_properties.items():
                records.append({
                    'timestamp': snapshot.collection_timestamp,
                    'backend': snapshot.backend_name,
                    'qubit': qubit_idx,
                    'T1': props.t1_us,
                    'T2': props.t2_us,
                    'readout_error': props.readout_error,
                    'frequency': props.frequency_ghz,
                })
        
        self._df_cache = pd.DataFrame(records)
        return self._df_cache
    
    def get_time_series(self, 
                        qubit: int, 
                        property_name: str) -> pd.Series:
        """
        Get the time series for a qubit property.
        
        Args:
            qubit: Qubit index
            property_name: Property name ('T1', 'T2', 'readout_error')
            
        Returns:
            Time series with datetime index
        """
        df = self._build_dataframe()
        
        qubit_data = df[df['qubit'] == qubit].copy()
        qubit_data = qubit_data.sort_values('timestamp')
        qubit_data = qubit_data.set_index('timestamp')
        
        if property_name not in qubit_data.columns:
            return pd.Series(dtype=float)
        
        return qubit_data[property_name].dropna()
    
    def compute_drift_features(self, 
                               qubit: int,
                               property_name: str = "T1") -> Dict[str, float]:
        """
        Compute drift features for a qubit property.
        
        Args:
            qubit: Qubit index
            property_name: Property to analyze
            
        Returns:
            Dictionary of drift features
        """
        ts = self.get_time_series(qubit, property_name)
        
        if len(ts) < self.min_snapshots:
            return {
                'variance': np.nan,
                'rate_of_change': np.nan,
                'rolling_zscore': np.nan,
                'stability_score': np.nan,
                'trend': np.nan,
                'n_points': len(ts),
            }
        
        # Basic statistics
        variance = ts.var()
        mean = ts.mean()
        std = ts.std()
        
        # Rate of change (daily differences)
        diff = ts.diff()
        rate_of_change = diff.mean() if len(diff) > 0 else np.nan
        
        # Rolling z-score (how unusual is the latest value?)
        window = min(self.window_days, len(ts))
        rolling_mean = ts.rolling(window=window, min_periods=1).mean()
        rolling_std = ts.rolling(window=window, min_periods=1).std()
        
        zscore = (ts.iloc[-1] - rolling_mean.iloc[-1]) / (rolling_std.iloc[-1] + 1e-10)
        
        # Stability score (inverse coefficient of variation)
        cv = std / (mean + 1e-10) if mean != 0 else 0
        stability_score = 1.0 / (1.0 + cv)
        
        # Linear trend (slope of least squares fit)
        x = np.arange(len(ts))
        if len(x) > 1:
            slope, _ = np.polyfit(x, ts.values, 1)
        else:
            slope = np.nan
        
        return {
            'variance': float(variance),
            'rate_of_change': float(rate_of_change) if not np.isnan(rate_of_change) else 0.0,
            'rolling_zscore': float(zscore),
            'stability_score': float(stability_score),
            'trend': float(slope) if not np.isnan(slope) else 0.0,
            'n_points': len(ts),
            'mean': float(mean),
            'std': float(std),
            'latest_value': float(ts.iloc[-1]),
        }
    
    def detect_change_points(self,
                             qubit: int,
                             property_name: str = "T1",
                             algorithm: str = "pelt",
                             penalty: str = "bic",
                             min_segment_length: int = 2) -> List[datetime]:
        """
        Detect change points in a qubit property time series.
        
        Uses the ruptures library for change-point detection.
        
        Args:
            qubit: Qubit index
            property_name: Property to analyze
            algorithm: Detection algorithm ('pelt', 'binseg', 'window')
            penalty: Penalty method ('bic', 'aic', 'linear')
            min_segment_length: Minimum points between change points
            
        Returns:
            List of datetime objects where changes were detected
        """
        ts = self.get_time_series(qubit, property_name)
        
        if len(ts) < self.min_snapshots:
            return []
        
        if not RUPTURES_AVAILABLE:
            # Fallback to simple z-score based detection
            return self._simple_change_detection(ts)
        
        # Prepare data for ruptures
        signal = ts.values.reshape(-1, 1)
        
        try:
            # Select algorithm
            if algorithm == "pelt":
                algo = rpt.Pelt(model="rbf", min_size=min_segment_length)
            elif algorithm == "binseg":
                algo = rpt.Binseg(model="rbf", min_size=min_segment_length)
            elif algorithm == "window":
                algo = rpt.Window(width=min_segment_length, model="rbf")
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            # Fit and predict
            algo.fit(signal)
            
            # Use penalty method
            if penalty == "bic":
                pen = np.log(len(signal)) * signal.var()
            elif penalty == "aic":
                pen = 2 * signal.var()
            else:
                pen = signal.var()  # linear
            
            change_indices = algo.predict(pen=pen)
            
            # Convert indices to timestamps
            timestamps = ts.index.tolist()
            change_points = []
            
            for idx in change_indices[:-1]:  # Last index is end of signal
                if 0 < idx < len(timestamps):
                    change_points.append(timestamps[idx])
            
            return change_points
            
        except Exception as e:
            logger.warning(f"Change-point detection failed: {e}")
            return self._simple_change_detection(ts)
    
    def _simple_change_detection(self, 
                                  ts: pd.Series, 
                                  threshold_sigma: float = 2.5) -> List[datetime]:
        """
        Simple change detection using z-scores of differences.
        
        Fallback when ruptures is not available.
        """
        if len(ts) < 3:
            return []
        
        diff = ts.diff().dropna()
        
        if diff.std() < 1e-10:
            return []
        
        z_diff = (diff - diff.mean()) / diff.std()
        change_mask = z_diff.abs() > threshold_sigma
        
        return ts.index[1:][change_mask].tolist()
    
    def compute_correlations(self,
                             property_name: str = "T1",
                             min_samples: int = 5) -> pd.DataFrame:
        """
        Compute cross-qubit correlations as a crosstalk proxy.
        
        Args:
            property_name: Property to analyze
            min_samples: Minimum samples for valid correlation
            
        Returns:
            Correlation matrix as DataFrame
        """
        df = self._build_dataframe()
        
        # Pivot to get qubit columns
        pivot = df.pivot_table(
            index='timestamp',
            columns='qubit',
            values=property_name,
            aggfunc='mean'
        )
        
        # Drop columns with too few valid samples
        pivot = pivot.dropna(thresh=min_samples, axis=1)
        
        if pivot.shape[1] < 2:
            return pd.DataFrame()
        
        return pivot.corr()
    
    def find_unstable_qubits(self,
                             stability_threshold: float = 0.7,
                             zscore_threshold: float = 2.0) -> List[int]:
        """
        Find qubits with significant drift or instability.
        
        Args:
            stability_threshold: Qubits with score below this are unstable
            zscore_threshold: Qubits with |z| above this are outliers
            
        Returns:
            List of unstable qubit indices
        """
        df = self._build_dataframe()
        qubits = df['qubit'].unique()
        
        unstable = []
        
        for qubit in qubits:
            for prop in ['T1', 'T2']:
                features = self.compute_drift_features(qubit, prop)
                
                if features['stability_score'] < stability_threshold:
                    unstable.append(qubit)
                    break
                
                if abs(features['rolling_zscore']) > zscore_threshold:
                    unstable.append(qubit)
                    break
        
        return list(set(unstable))
    
    def get_best_qubits(self,
                        n: int,
                        weights: Optional[Dict[str, float]] = None,
                        exclude_unstable: bool = True) -> List[int]:
        """
        Get the best qubits considering both quality and stability.
        
        Args:
            n: Number of qubits to select
            weights: Weights for different scoring factors
            exclude_unstable: Whether to exclude unstable qubits
            
        Returns:
            List of best qubit indices
        """
        if not self.snapshots:
            return []
        
        if weights is None:
            weights = {
                'quality': 0.6,
                'stability': 0.4,
            }
        
        # Get latest snapshot for quality scores
        latest = self.snapshots[-1]
        quality_scores = latest.get_qubit_scores()
        
        # Get stability scores
        stability_scores = {}
        for qubit in quality_scores.keys():
            features = self.compute_drift_features(qubit, "T1")
            stability_scores[qubit] = features.get('stability_score', 0.5)
        
        # Combine scores
        combined_scores = {}
        for qubit in quality_scores.keys():
            q_score = quality_scores.get(qubit, 0)
            s_score = stability_scores.get(qubit, 0)
            combined_scores[qubit] = (
                weights['quality'] * q_score +
                weights['stability'] * s_score
            )
        
        # Filter unstable qubits if requested
        if exclude_unstable:
            unstable = set(self.find_unstable_qubits())
            combined_scores = {
                q: s for q, s in combined_scores.items() 
                if q not in unstable
            }
        
        # Sort and return top n
        sorted_qubits = sorted(
            combined_scores.keys(),
            key=lambda q: combined_scores[q],
            reverse=True
        )
        
        return sorted_qubits[:n]
    
    def generate_report(self, backend_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive drift analysis report.
        
        Args:
            backend_name: Backend name for report header
            
        Returns:
            Report dictionary
        """
        if not self.snapshots:
            return {"error": "No snapshots loaded"}
        
        df = self._build_dataframe()
        qubits = df['qubit'].unique()
        
        # Backend name from snapshots if not provided
        if backend_name is None:
            backend_name = self.snapshots[0].backend_name
        
        report = {
            'backend_name': backend_name,
            'analysis_timestamp': datetime.now().isoformat(),
            'num_snapshots': len(self.snapshots),
            'date_range': {
                'start': self.snapshots[0].collection_timestamp.isoformat(),
                'end': self.snapshots[-1].collection_timestamp.isoformat(),
            },
            'qubit_summary': {},
            'unstable_qubits': self.find_unstable_qubits(),
            'correlations': {},
            'recommendations': [],
        }
        
        # Per-qubit analysis
        for qubit in qubits:
            report['qubit_summary'][int(qubit)] = {
                'T1': self.compute_drift_features(int(qubit), 'T1'),
                'T2': self.compute_drift_features(int(qubit), 'T2'),
                'T1_change_points': [
                    cp.isoformat() for cp in 
                    self.detect_change_points(int(qubit), 'T1')
                ],
                'T2_change_points': [
                    cp.isoformat() for cp in 
                    self.detect_change_points(int(qubit), 'T2')
                ],
            }
        
        # Correlations
        for prop in ['T1', 'T2']:
            corr = self.compute_correlations(prop)
            if not corr.empty:
                report['correlations'][prop] = corr.to_dict()
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(report)
        
        return report
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Flag unstable qubits
        unstable = report.get('unstable_qubits', [])
        if unstable:
            recommendations.append(
                f"⚠️  Qubits {unstable[:5]}{'...' if len(unstable) > 5 else ''} "
                f"show significant drift. Use real-time probes before experiments."
            )
        
        # Flag recent change points
        recent_changes = []
        cutoff = datetime.now() - timedelta(days=3)
        
        for qubit, data in report.get('qubit_summary', {}).items():
            for prop in ['T1', 'T2']:
                cps = data.get(f'{prop}_change_points', [])
                for cp_str in cps:
                    try:
                        cp = datetime.fromisoformat(cp_str)
                        if cp > cutoff:
                            recent_changes.append((qubit, prop, cp_str))
                    except (ValueError, TypeError):
                        pass
        
        if recent_changes:
            recommendations.append(
                f"📍 Recent calibration changes detected in {len(recent_changes)} "
                f"qubit-property pairs. Consider re-running probe characterization."
            )
        
        # Check for high correlations (crosstalk)
        high_corr_pairs = []
        for prop, corr in report.get('correlations', {}).items():
            for q1 in corr:
                for q2, val in corr[q1].items():
                    if q1 < q2 and abs(val) > 0.7:
                        high_corr_pairs.append((q1, q2, prop, val))
        
        if high_corr_pairs:
            recommendations.append(
                f"🔗 High correlations detected between {len(high_corr_pairs)} "
                f"qubit pairs. This may indicate crosstalk - avoid using "
                f"correlated qubits in the same chain."
            )
        
        if not recommendations:
            recommendations.append(
                "✅ Calibration appears stable. Standard qubit selection is suitable."
            )
        
        return recommendations
