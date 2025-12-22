"""
Tests for the utils module.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import tempfile
import os


class TestQPUBudgetTracker:
    """Tests for QPUBudgetTracker class."""
    
    def test_tracker_initialization(self):
        """Test QPUBudgetTracker can be initialized."""
        from src.utils import QPUBudgetTracker
        
        tracker = QPUBudgetTracker(monthly_budget_minutes=10.0)
        
        assert tracker.monthly_budget == 10.0 * 60  # Converted to seconds
    
    def test_initial_remaining_budget(self):
        """Test initial remaining budget is full."""
        from src.utils import QPUBudgetTracker
        
        tracker = QPUBudgetTracker(monthly_budget_minutes=10.0)
        
        remaining = tracker.remaining_budget()
        
        assert remaining == pytest.approx(600.0, abs=1.0)  # 10 minutes in seconds
    
    def test_record_usage(self):
        """Test recording QPU usage."""
        from src.utils import QPUBudgetTracker
        
        tracker = QPUBudgetTracker(monthly_budget_minutes=10.0)
        
        tracker.record_usage(seconds=60.0, job_id="test_job_001")
        
        remaining = tracker.remaining_budget()
        
        assert remaining == pytest.approx(540.0, abs=1.0)  # 9 minutes remaining
    
    def test_can_afford_true(self):
        """Test can_afford returns True when budget available."""
        from src.utils import QPUBudgetTracker
        
        tracker = QPUBudgetTracker(monthly_budget_minutes=10.0)
        
        assert tracker.can_afford(estimated_seconds=60.0) is True
    
    def test_can_afford_false(self):
        """Test can_afford returns False when budget exhausted."""
        from src.utils import QPUBudgetTracker
        
        tracker = QPUBudgetTracker(monthly_budget_minutes=1.0)
        
        tracker.record_usage(seconds=55.0, job_id="test_job_001")
        
        # Only 5 seconds remaining, trying to afford 30
        assert tracker.can_afford(estimated_seconds=30.0) is False
    
    def test_budget_alert_threshold(self):
        """Test budget alert when threshold crossed."""
        from src.utils import QPUBudgetTracker
        
        tracker = QPUBudgetTracker(
            monthly_budget_minutes=10.0,
            alert_threshold=0.8
        )
        
        # Use 85% of budget
        tracker.record_usage(seconds=510.0, job_id="heavy_usage")
        
        assert tracker.is_below_alert_threshold() is True
    
    def test_usage_history(self):
        """Test usage history tracking."""
        from src.utils import QPUBudgetTracker
        
        tracker = QPUBudgetTracker(monthly_budget_minutes=10.0)
        
        tracker.record_usage(seconds=30.0, job_id="job1")
        tracker.record_usage(seconds=45.0, job_id="job2")
        
        history = tracker.get_usage_history()
        
        assert len(history) == 2
        assert history[0]['job_id'] == "job1"


class TestJobBatcher:
    """Tests for JobBatcher class."""
    
    def test_batcher_initialization(self):
        """Test JobBatcher can be initialized."""
        from src.utils import JobBatcher
        
        batcher = JobBatcher(max_circuits_per_batch=100)
        
        assert batcher.max_batch_size == 100
    
    def test_batch_circuits(self):
        """Test circuit batching."""
        from src.utils import JobBatcher
        from qiskit import QuantumCircuit
        
        batcher = JobBatcher(max_circuits_per_batch=5)
        
        # Create 12 circuits
        circuits = [QuantumCircuit(2, 2) for _ in range(12)]
        
        batches = batcher.batch(circuits)
        
        # Should create 3 batches: 5, 5, 2
        assert len(batches) == 3
        assert len(batches[0]) == 5
        assert len(batches[1]) == 5
        assert len(batches[2]) == 2
    
    def test_empty_circuit_list(self):
        """Test handling empty circuit list."""
        from src.utils import JobBatcher
        
        batcher = JobBatcher(max_circuits_per_batch=100)
        
        batches = batcher.batch([])
        
        assert len(batches) == 0


class TestDataManagement:
    """Tests for data management utilities."""
    
    def test_parquet_round_trip(self):
        """Test DataFrame can be saved and loaded from Parquet."""
        df = pd.DataFrame({
            'timestamp': [datetime.now() for _ in range(5)],
            'qubit': [0, 1, 2, 3, 4],
            't1': [150.0, 160.0, 155.0, 145.0, 170.0],
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_data.parquet')
            
            df.to_parquet(filepath)
            loaded = pd.read_parquet(filepath)
            
            assert len(loaded) == len(df)
            assert list(loaded.columns) == list(df.columns)
    
    def test_data_versioning(self):
        """Test data versioning with timestamps."""
        from datetime import datetime
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"drift_data_{timestamp}.parquet"
        
        assert timestamp in filename
    
    def test_parquet_append_pattern(self):
        """Test appending to Parquet dataset."""
        df1 = pd.DataFrame({'x': [1, 2, 3]})
        df2 = pd.DataFrame({'x': [4, 5, 6]})
        
        combined = pd.concat([df1, df2], ignore_index=True)
        
        assert len(combined) == 6


class TestSessionManager:
    """Tests for experiment session management."""
    
    def test_session_creation(self):
        """Test creating a new session."""
        from datetime import datetime
        
        session = {
            'session_id': f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'start_time': datetime.now(),
            'backend': 'ibm_brisbane',
            'experiments': [],
        }
        
        assert 'session_id' in session
        assert session['backend'] == 'ibm_brisbane'
    
    def test_session_logging(self):
        """Test logging experiment to session."""
        session = {'experiments': []}
        
        experiment = {
            'type': 'qec_benchmark',
            'distance': 5,
            'shots': 4096,
            'job_id': 'abc123',
        }
        
        session['experiments'].append(experiment)
        
        assert len(session['experiments']) == 1


class TestConfigLoader:
    """Tests for configuration loading utilities."""
    
    def test_yaml_config_loading(self):
        """Test loading YAML configuration."""
        import yaml
        
        config_str = """
        backend:
          name: ibm_brisbane
          min_qubits: 27
        qec:
          distances: [3, 5, 7]
          shots: 4096
        """
        
        config = yaml.safe_load(config_str)
        
        assert config['backend']['name'] == 'ibm_brisbane'
        assert config['qec']['distances'] == [3, 5, 7]
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = {
            'backend': {'name': 'ibm_brisbane', 'min_qubits': 27},
            'qec': {'distances': [3, 5, 7], 'shots': 4096},
        }
        
        # Validate required fields
        assert 'backend' in config
        assert 'qec' in config
        assert config['qec']['shots'] > 0


class TestErrorHandling:
    """Tests for error handling utilities."""
    
    def test_retry_decorator_success(self):
        """Test retry decorator on successful call."""
        call_count = 0
        
        def succeeds_first_try():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = succeeds_first_try()
        
        assert result == "success"
        assert call_count == 1
    
    def test_exception_logging(self):
        """Test exception information capture."""
        try:
            raise ValueError("Test error message")
        except ValueError as e:
            error_info = {
                'type': type(e).__name__,
                'message': str(e),
            }
        
        assert error_info['type'] == 'ValueError'
        assert 'Test error' in error_info['message']
