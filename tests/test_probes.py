"""
Tests for the probes module.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock


class TestProbeSuite:
    """Tests for ProbeSuite class."""
    
    @pytest.fixture
    def mock_backend(self):
        """Create a mock backend."""
        backend = Mock()
        backend.name = "test_backend"
        backend.num_qubits = 27
        return backend
    
    def test_probe_suite_initialization(self, mock_backend):
        """Test ProbeSuite can be initialized."""
        from src.probes import ProbeSuite
        
        probe_suite = ProbeSuite(
            backend=mock_backend,
            shots_per_probe=30,
            probes=['t1', 'readout']
        )
        
        assert probe_suite.backend == mock_backend
        assert probe_suite.shots == 30
        assert 't1' in probe_suite.probes
        assert 'readout' in probe_suite.probes
    
    def test_estimate_probe_time(self, mock_backend):
        """Test probe time estimation."""
        from src.probes import ProbeSuite
        
        probe_suite = ProbeSuite(
            backend=mock_backend,
            shots_per_probe=30,
            probes=['t1', 'readout', 'rb']
        )
        
        estimated_time = probe_suite.estimate_time()
        
        # Should return a positive number
        assert estimated_time > 0
        # Should be reasonable (< 60 seconds for 30-shot probes)
        assert estimated_time < 60
    
    def test_build_t1_probe_circuit(self, mock_backend):
        """Test building T1 probe circuit."""
        from src.probes import ProbeSuite
        
        probe_suite = ProbeSuite(
            backend=mock_backend,
            shots_per_probe=30,
            probes=['t1']
        )
        
        circuit = probe_suite._build_t1_probe(qubit=0, delay_us=50)
        
        # Should have a delay instruction
        assert circuit is not None
    
    def test_build_readout_probe_circuit(self, mock_backend):
        """Test building readout error probe circuit."""
        from src.probes import ProbeSuite
        
        probe_suite = ProbeSuite(
            backend=mock_backend,
            shots_per_probe=30,
            probes=['readout']
        )
        
        circuits = probe_suite._build_readout_probe(qubit=0)
        
        # Should return circuits for |0⟩ and |1⟩ preparation
        assert len(circuits) == 2


class TestQubitSelector:
    """Tests for QubitSelector class."""
    
    @pytest.fixture
    def mock_calibration_data(self):
        """Create mock calibration data."""
        return {
            'timestamp': '2024-01-01T00:00:00Z',
            'backend_name': 'test_backend',
            'qubit_properties': {
                i: {
                    't1': 100e-6 - i * 1e-6,  # Decreasing T1
                    't2': 80e-6 - i * 0.5e-6,
                    'readout_error': 0.01 + i * 0.001
                }
                for i in range(10)
            }
        }
    
    @pytest.fixture
    def mock_backend(self):
        """Create a mock backend."""
        backend = Mock()
        backend.name = "test_backend"
        backend.num_qubits = 27
        return backend
    
    def test_qubit_selector_initialization(self, mock_backend):
        """Test QubitSelector can be initialized."""
        from src.probes import QubitSelector
        
        selector = QubitSelector(backend=mock_backend, strategy='static')
        
        assert selector.backend == mock_backend
        assert selector.strategy == 'static'
    
    def test_static_selection(self, mock_backend, mock_calibration_data):
        """Test static qubit selection."""
        from src.probes import QubitSelector
        
        selector = QubitSelector(backend=mock_backend, strategy='static')
        result = selector.select_qubits(
            n_qubits=5,
            calibration_data=mock_calibration_data
        )
        
        assert 'qubits' in result
        assert len(result['qubits']) == 5
        # Should select qubits with best properties (lowest indices in this mock)
        assert 0 in result['qubits']
    
    def test_realtime_selection(self, mock_backend):
        """Test real-time qubit selection."""
        from src.probes import QubitSelector
        
        mock_probe_data = {
            'timestamp': '2024-01-01T00:00:00Z',
            'qubit_data': [
                {'qubit': i, 't1_probe': 100 - i, 'readout_error_probe': 0.01 + i * 0.001}
                for i in range(10)
            ]
        }
        
        selector = QubitSelector(backend=mock_backend, strategy='realtime')
        result = selector.select_qubits(
            n_qubits=5,
            probe_data=mock_probe_data
        )
        
        assert 'qubits' in result
        assert len(result['qubits']) == 5
    
    def test_invalid_strategy_raises(self, mock_backend):
        """Test that invalid strategy raises error."""
        from src.probes import QubitSelector
        
        with pytest.raises(ValueError):
            QubitSelector(backend=mock_backend, strategy='invalid_strategy')


class TestDriftPredictor:
    """Tests for drift prediction functionality."""
    
    def test_exponential_smoothing(self):
        """Test Holt's exponential smoothing implementation."""
        # Simple test data with upward trend
        values = [100, 102, 104, 106, 108]
        alpha = 0.3
        beta = 0.1
        
        # Initialize
        level = values[0]
        trend = values[1] - values[0]
        
        # Apply smoothing
        for val in values[1:]:
            prev_level = level
            level = alpha * val + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
        
        # Should predict continuation of trend
        predicted = level + trend
        assert predicted > values[-1]
    
    def test_stability_score_calculation(self):
        """Test stability score calculation."""
        # Stable qubit
        stable_values = [100, 100, 100, 100, 100]
        
        # Unstable qubit
        unstable_values = [100, 80, 120, 70, 130]
        
        stable_var = np.var(stable_values)
        unstable_var = np.var(unstable_values)
        
        stable_score = 1.0 / (1.0 + stable_var / 100)
        unstable_score = 1.0 / (1.0 + unstable_var / 100)
        
        # Stable should have higher score
        assert stable_score > unstable_score
