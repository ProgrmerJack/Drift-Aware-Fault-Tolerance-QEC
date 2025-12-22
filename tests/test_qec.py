"""
Tests for the QEC module.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock


class TestRepetitionCode:
    """Tests for RepetitionCode class."""
    
    def test_repetition_code_initialization(self):
        """Test RepetitionCode can be initialized."""
        from src.qec import RepetitionCode
        
        code = RepetitionCode(distance=5, qubits=[0, 1, 2, 3, 4, 5, 6, 7, 8])
        
        assert code.distance == 5
        assert len(code.data_qubits) == 9  # 2*d - 1
        assert len(code.ancilla_qubits) == 4  # d - 1
    
    def test_invalid_distance_raises(self):
        """Test that even distance raises error."""
        from src.qec import RepetitionCode
        
        with pytest.raises(ValueError):
            RepetitionCode(distance=4)  # Must be odd
    
    def test_insufficient_qubits_raises(self):
        """Test that insufficient qubits raises error."""
        from src.qec import RepetitionCode
        
        with pytest.raises(ValueError):
            # Need at least 2*5-1 = 9 qubits for distance 5
            RepetitionCode(distance=5, qubits=[0, 1, 2, 3, 4])
    
    def test_build_circuit_basic(self):
        """Test building a basic repetition code circuit."""
        from src.qec import RepetitionCode
        
        code = RepetitionCode(distance=3, qubits=[0, 1, 2, 3, 4, 5, 6])
        circuit = code.build_circuit(
            n_syndrome_rounds=1,
            initial_state='0',
            measure_final=True
        )
        
        # Should create a valid circuit
        assert circuit is not None
        assert circuit.num_qubits >= 5  # At least data qubits
    
    def test_initial_state_preparation(self):
        """Test that different initial states are supported."""
        from src.qec import RepetitionCode
        
        code = RepetitionCode(distance=3, qubits=[0, 1, 2, 3, 4, 5, 6])
        
        for state in ['0', '1', '+']:
            circuit = code.build_circuit(
                n_syndrome_rounds=1,
                initial_state=state,
                measure_final=True
            )
            assert circuit is not None
    
    def test_syndrome_round_structure(self):
        """Test that syndrome rounds have correct structure."""
        from src.qec import RepetitionCode
        
        code = RepetitionCode(distance=3, qubits=[0, 1, 2, 3, 4, 5, 6])
        circuit = code.build_circuit(
            n_syndrome_rounds=3,
            initial_state='0',
            measure_final=True
        )
        
        # Should have measurements for syndrome extraction
        # Count measure operations
        measure_count = 0
        for instruction in circuit.data:
            if instruction.operation.name == 'measure':
                measure_count += 1
        
        # Should have multiple measurements (syndromes + final)
        assert measure_count > 0


class TestSyndromeDecoder:
    """Tests for SyndromeDecoder class."""
    
    def test_decoder_initialization(self):
        """Test SyndromeDecoder can be initialized."""
        from src.qec import SyndromeDecoder
        
        decoder = SyndromeDecoder(distance=5)
        
        assert decoder.distance == 5
    
    def test_decode_no_error(self):
        """Test decoding with no errors (zero syndrome)."""
        from src.qec import SyndromeDecoder
        
        decoder = SyndromeDecoder(distance=3)
        syndrome = np.array([[0, 0]])  # No syndrome triggers
        
        result = decoder.decode(syndrome)
        
        # Should predict no correction needed
        assert result['logical_correction'] == 0
    
    def test_decode_single_error(self):
        """Test decoding with single qubit error."""
        from src.qec import SyndromeDecoder
        
        decoder = SyndromeDecoder(distance=3)
        # Error on middle qubit triggers both ancillas
        syndrome = np.array([[1, 1]])
        
        result = decoder.decode(syndrome)
        
        # Should identify error location
        assert 'correction' in result
    
    def test_majority_vote_decoding(self):
        """Test majority vote logic for repetition code."""
        # For repetition code, logical value is majority of data qubits
        data_qubits = [0, 0, 0, 1, 0]  # Logical 0 (3 zeros vs 2 ones would still be 0)
        
        logical = int(sum(data_qubits) > len(data_qubits) // 2)
        
        assert logical == 0  # Majority is 0


class TestQECExperimentRunner:
    """Tests for QECExperimentRunner class."""
    
    @pytest.fixture
    def mock_backend(self):
        """Create a mock backend."""
        backend = Mock()
        backend.name = "test_backend"
        backend.num_qubits = 27
        return backend
    
    @pytest.fixture
    def mock_budget_tracker(self):
        """Create a mock budget tracker."""
        tracker = Mock()
        tracker.remaining_budget.return_value = 300.0
        tracker.can_afford.return_value = True
        return tracker
    
    def test_runner_initialization(self, mock_backend, mock_budget_tracker):
        """Test QECExperimentRunner can be initialized."""
        from src.qec import QECExperimentRunner
        
        runner = QECExperimentRunner(
            backend=mock_backend,
            budget_tracker=mock_budget_tracker
        )
        
        assert runner.backend == mock_backend
    
    def test_estimate_experiment_time(self, mock_backend, mock_budget_tracker):
        """Test experiment time estimation."""
        from src.qec import QECExperimentRunner
        
        runner = QECExperimentRunner(
            backend=mock_backend,
            budget_tracker=mock_budget_tracker
        )
        
        estimated_time = runner.estimate_time(n_circuits=10, shots=1000)
        
        # Should return positive estimate
        assert estimated_time > 0


class TestLogicalErrorCalculation:
    """Tests for logical error rate calculation."""
    
    def test_error_rate_calculation(self):
        """Test basic error rate calculation."""
        total_shots = 1000
        logical_errors = 50
        
        error_rate = logical_errors / total_shots
        
        assert error_rate == 0.05
    
    def test_error_rate_with_confidence_interval(self):
        """Test error rate with binomial confidence interval."""
        from scipy import stats
        
        total_shots = 1000
        logical_errors = 50
        confidence = 0.95
        
        error_rate = logical_errors / total_shots
        
        # Wilson score interval
        z = stats.norm.ppf((1 + confidence) / 2)
        denominator = 1 + z**2 / total_shots
        center = (error_rate + z**2 / (2 * total_shots)) / denominator
        margin = z * np.sqrt(error_rate * (1 - error_rate) / total_shots + z**2 / (4 * total_shots**2)) / denominator
        
        ci_lower = center - margin
        ci_upper = center + margin
        
        # Confidence interval should contain the point estimate
        assert ci_lower < error_rate < ci_upper
    
    def test_zero_error_rate(self):
        """Test handling of zero error rate."""
        total_shots = 1000
        logical_errors = 0
        
        error_rate = logical_errors / total_shots
        
        assert error_rate == 0.0
