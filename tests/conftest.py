"""
Pytest configuration and shared fixtures for the test suite.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock


# =============================================================================
# Mock Backend Fixtures
# =============================================================================

@pytest.fixture
def mock_backend():
    """Create a mock IBM Quantum backend."""
    backend = Mock()
    backend.name = "ibm_brisbane"
    backend.num_qubits = 127
    
    # Mock properties
    backend.properties = Mock(return_value=Mock())
    backend.properties.return_value.t1 = Mock(return_value=150e-6)  # 150 µs
    backend.properties.return_value.t2 = Mock(return_value=100e-6)  # 100 µs
    backend.properties.return_value.readout_error = Mock(return_value=0.02)
    
    # Mock target for gate information
    backend.target = Mock()
    
    return backend


@pytest.fixture
def mock_service():
    """Create a mock QiskitRuntimeService."""
    service = Mock()
    service.least_busy = Mock(return_value=mock_backend())
    return service


@pytest.fixture
def mock_sampler():
    """Create a mock SamplerV2 primitive."""
    sampler = Mock()
    
    # Mock job submission
    job = Mock()
    job.job_id.return_value = "test_job_123"
    job.status.return_value = "DONE"
    
    # Mock result
    result = Mock()
    result.quasi_dists = [{0: 0.9, 1: 0.1}]
    job.result.return_value = result
    
    sampler.run.return_value = job
    
    return sampler


# =============================================================================
# Data Fixtures
# =============================================================================

@pytest.fixture
def sample_drift_telemetry():
    """Create sample drift telemetry DataFrame."""
    n_points = 24  # 24 hours of data
    timestamps = [datetime.now() - timedelta(hours=i) for i in range(n_points)]
    
    # Simulate realistic T1/T2 drift
    base_t1 = 150.0
    base_t2 = 100.0
    
    t1_values = base_t1 + np.random.normal(0, 10, n_points)
    t2_values = base_t2 + np.random.normal(0, 8, n_points)
    
    # Inject a drift event at hour 12
    t1_values[12:15] = base_t1 * 0.7  # 30% T1 drop
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'qubit': [0] * n_points,
        't1': t1_values,
        't2': t2_values,
        'readout_error': np.random.uniform(0.01, 0.03, n_points),
        'gate_error_1q': np.random.uniform(0.001, 0.003, n_points),
        'gate_error_2q': np.random.uniform(0.005, 0.015, n_points),
    })


@pytest.fixture
def sample_qec_results():
    """Create sample QEC benchmark results."""
    return pd.DataFrame({
        'distance': [3, 3, 3, 5, 5, 5, 7, 7, 7],
        'syndrome_rounds': [1, 2, 3, 1, 2, 3, 1, 2, 3],
        'logical_error_rate': [0.08, 0.07, 0.065, 0.05, 0.04, 0.035, 0.03, 0.025, 0.02],
        'error_rate_ci_lower': [0.07, 0.06, 0.055, 0.04, 0.03, 0.028, 0.02, 0.018, 0.015],
        'error_rate_ci_upper': [0.09, 0.08, 0.075, 0.06, 0.05, 0.042, 0.04, 0.032, 0.025],
        'shots': [4096] * 9,
        'job_id': [f"job_{i}" for i in range(9)],
        'timestamp': [datetime.now() - timedelta(hours=i) for i in range(9)],
    })


@pytest.fixture
def sample_probe_results():
    """Create sample probe measurement results."""
    return {
        'qubit_id': 0,
        'timestamp': datetime.now(),
        't1_estimate': 148.5,
        't2_estimate': 95.2,
        'readout_fidelity': 0.97,
        'measurement_duration_seconds': 5.2,
        'shots_used': 30,
    }


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture
def experiment_config():
    """Create sample experiment configuration."""
    return {
        'backend': {
            'name': 'ibm_brisbane',
            'min_qubits': 27,
            'optimization_level': 3,
        },
        'qec': {
            'distances': [3, 5, 7],
            'syndrome_rounds': 3,
            'shots': 4096,
            'initial_states': ['0', '1'],
        },
        'probes': {
            'shots': 30,
            'timeout_seconds': 300,
            'refresh_interval_hours': 4,
        },
        'budget': {
            'monthly_minutes': 10,
            'alert_threshold': 0.8,
        },
    }


@pytest.fixture
def qubit_chain_d5():
    """Create a sample qubit chain for distance-5 code."""
    # 2*5 - 1 = 9 qubits needed
    return [0, 1, 2, 3, 4, 5, 6, 7, 8]


@pytest.fixture
def qubit_chain_d7():
    """Create a sample qubit chain for distance-7 code."""
    # 2*7 - 1 = 13 qubits needed
    return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


# =============================================================================
# Helper Functions
# =============================================================================

@pytest.fixture
def make_test_circuit():
    """Factory fixture for creating test circuits."""
    from qiskit import QuantumCircuit
    
    def _make_circuit(n_qubits=5, n_clbits=5, add_measurements=True):
        qc = QuantumCircuit(n_qubits, n_clbits)
        
        # Add some gates
        for i in range(n_qubits):
            qc.h(i)
        
        if add_measurements:
            qc.measure(range(n_qubits), range(n_clbits))
        
        return qc
    
    return _make_circuit


# =============================================================================
# Cleanup Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def cleanup_matplotlib():
    """Automatically close matplotlib figures after each test."""
    yield
    import matplotlib.pyplot as plt
    plt.close('all')


# =============================================================================
# Markers Configuration
# =============================================================================

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests requiring real backend connection"
    )
    config.addinivalue_line(
        "markers", "qpu: marks tests that require QPU execution"
    )
