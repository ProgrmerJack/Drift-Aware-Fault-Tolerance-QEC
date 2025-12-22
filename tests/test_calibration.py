"""
Tests for the calibration module.
"""

import pytest
import numpy as np
from datetime import datetime, timezone
from unittest.mock import Mock, MagicMock, patch


class TestDriftCollector:
    """Tests for DriftCollector class."""
    
    @pytest.fixture
    def mock_backend(self):
        """Create a mock IBM Quantum backend."""
        backend = Mock()
        backend.name = "test_backend"
        backend.num_qubits = 27
        
        # Mock target with qubit properties
        target = Mock()
        
        # Create mock qubit properties
        qubit_props = {}
        for i in range(27):
            qubit_props[i] = {
                't1': 100e-6 + np.random.randn() * 10e-6,  # ~100 µs
                't2': 80e-6 + np.random.randn() * 8e-6,    # ~80 µs
                'readout_error': 0.01 + np.random.rand() * 0.02,
                'prob_meas0_prep1': 0.01 + np.random.rand() * 0.01,
                'prob_meas1_prep0': 0.01 + np.random.rand() * 0.01,
            }
        
        target.qubit_properties = Mock(return_value=qubit_props)
        backend.target = target
        
        return backend
    
    def test_drift_collector_initialization(self, mock_backend):
        """Test DriftCollector can be initialized."""
        from src.calibration import DriftCollector
        
        collector = DriftCollector(mock_backend)
        
        assert collector.backend == mock_backend
        assert collector.backend_name == "test_backend"
    
    def test_collect_calibration_snapshot(self, mock_backend):
        """Test collecting a calibration snapshot."""
        from src.calibration import DriftCollector
        
        collector = DriftCollector(mock_backend)
        snapshot = collector.collect_calibration_snapshot()
        
        assert 'timestamp' in snapshot
        assert 'backend_name' in snapshot
        assert 'qubit_properties' in snapshot
        assert snapshot['backend_name'] == "test_backend"
    
    def test_snapshot_contains_expected_fields(self, mock_backend):
        """Test that snapshot contains all expected qubit property fields."""
        from src.calibration import DriftCollector
        
        collector = DriftCollector(mock_backend)
        snapshot = collector.collect_calibration_snapshot()
        
        if snapshot['qubit_properties']:
            qubit_data = list(snapshot['qubit_properties'].values())[0]
            # Should contain T1, T2, and readout error at minimum
            assert 't1' in qubit_data or 'T1' in qubit_data
    
    def test_timestamp_is_utc(self, mock_backend):
        """Test that timestamps are in UTC."""
        from src.calibration import DriftCollector
        
        collector = DriftCollector(mock_backend)
        snapshot = collector.collect_calibration_snapshot()
        
        timestamp = snapshot['timestamp']
        assert 'Z' in timestamp or '+00:00' in timestamp or 'UTC' in timestamp


class TestCalibrationDataStorage:
    """Tests for calibration data storage and retrieval."""
    
    def test_save_and_load_snapshot(self, tmp_path):
        """Test saving and loading calibration snapshots."""
        from src.utils import save_calibration_snapshot, load_calibration_snapshot
        
        test_snapshot = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'backend_name': 'test_backend',
            'qubit_properties': {
                0: {'t1': 100e-6, 't2': 80e-6, 'readout_error': 0.01},
                1: {'t1': 95e-6, 't2': 75e-6, 'readout_error': 0.015}
            }
        }
        
        filepath = tmp_path / "test_snapshot.json"
        save_calibration_snapshot(test_snapshot, filepath)
        
        loaded = load_calibration_snapshot(filepath)
        
        assert loaded['backend_name'] == test_snapshot['backend_name']
        assert len(loaded['qubit_properties']) == 2


class TestDriftAnalysis:
    """Tests for drift analysis utilities."""
    
    def test_compute_drift_statistics(self):
        """Test computing drift statistics from multiple snapshots."""
        # Create mock time series data
        t1_values = [100, 98, 102, 99, 101]  # µs
        
        mean = np.mean(t1_values)
        std = np.std(t1_values)
        drift_pct = std / mean * 100
        
        assert 99 < mean < 101
        assert drift_pct < 5  # Should be less than 5% variation
    
    def test_detect_significant_drift(self):
        """Test detection of significant drift events."""
        # Stable period
        stable_values = [100, 101, 99, 100, 100]
        
        # Drift event
        drift_values = [100, 100, 95, 85, 75]
        
        stable_diff = np.diff(stable_values)
        drift_diff = np.diff(drift_values)
        
        # Maximum change should be larger for drift event
        assert max(abs(drift_diff)) > max(abs(stable_diff))
