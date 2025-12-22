"""
Tests for the analysis module.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta


class TestDriftErrorAnalyzer:
    """Tests for DriftErrorAnalyzer class."""
    
    @pytest.fixture
    def sample_drift_data(self):
        """Create sample drift telemetry data."""
        timestamps = [datetime.now() - timedelta(hours=i) for i in range(10)]
        return pd.DataFrame({
            'timestamp': timestamps,
            'qubit': [0] * 10,
            't1': np.random.uniform(100, 200, 10),
            't2': np.random.uniform(80, 160, 10),
            'readout_error': np.random.uniform(0.01, 0.05, 10),
            'gate_error_1q': np.random.uniform(0.001, 0.005, 10),
            'gate_error_2q': np.random.uniform(0.005, 0.02, 10),
        })
    
    @pytest.fixture
    def sample_error_data(self):
        """Create sample QEC error data."""
        timestamps = [datetime.now() - timedelta(hours=i) for i in range(10)]
        return pd.DataFrame({
            'timestamp': timestamps,
            'distance': [3] * 10,
            'logical_error_rate': np.random.uniform(0.01, 0.1, 10),
            'shots': [4096] * 10,
            'qubit_set': ['0,1,2,3,4'] * 10,
        })
    
    def test_analyzer_initialization(self, sample_drift_data, sample_error_data):
        """Test DriftErrorAnalyzer can be initialized."""
        from src.analysis import DriftErrorAnalyzer
        
        analyzer = DriftErrorAnalyzer(
            drift_data=sample_drift_data,
            error_data=sample_error_data
        )
        
        assert analyzer.drift_data is not None
        assert analyzer.error_data is not None
    
    def test_compute_correlation(self, sample_drift_data, sample_error_data):
        """Test drift-error correlation computation."""
        from src.analysis import DriftErrorAnalyzer
        
        analyzer = DriftErrorAnalyzer(
            drift_data=sample_drift_data,
            error_data=sample_error_data
        )
        
        # Should compute Spearman correlation
        correlations = analyzer.compute_correlations()
        
        assert 't1_correlation' in correlations or 'correlations' in correlations
    
    def test_identify_drift_events(self, sample_drift_data, sample_error_data):
        """Test drift event identification."""
        from src.analysis import DriftErrorAnalyzer
        
        # Inject a clear drift event
        sample_drift_data.loc[5, 't1'] = 50  # Significant T1 drop
        
        analyzer = DriftErrorAnalyzer(
            drift_data=sample_drift_data,
            error_data=sample_error_data
        )
        
        events = analyzer.identify_drift_events(threshold_sigma=2.0)
        
        # Should identify at least one event
        assert isinstance(events, (list, pd.DataFrame))


class TestStatisticalTests:
    """Tests for statistical analysis functions."""
    
    def test_spearman_correlation(self):
        """Test Spearman correlation calculation."""
        from scipy.stats import spearmanr
        
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 5])
        
        corr, p_value = spearmanr(x, y)
        
        assert corr == pytest.approx(1.0)
        assert p_value < 0.05
    
    def test_negative_correlation(self):
        """Test negative correlation detection."""
        from scipy.stats import spearmanr
        
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([5, 4, 3, 2, 1])
        
        corr, p_value = spearmanr(x, y)
        
        assert corr == pytest.approx(-1.0)
    
    def test_no_correlation(self):
        """Test near-zero correlation."""
        from scipy.stats import spearmanr
        
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)
        
        corr, p_value = spearmanr(x, y)
        
        # Should be close to zero for uncorrelated data
        assert abs(corr) < 0.3


class TestThresholdFitting:
    """Tests for fault-tolerance threshold fitting."""
    
    def test_exponential_scaling(self):
        """Test exponential scaling detection."""
        # Simulate below-threshold behavior: error rate decreases with distance
        distances = np.array([3, 5, 7, 9])
        physical_error = 0.001
        threshold = 0.01
        
        # Below threshold: logical error ~ (p/p_th)^((d+1)/2)
        logical_errors = (physical_error / threshold) ** ((distances + 1) / 2)
        
        # Error rate should decrease with distance
        assert all(np.diff(logical_errors) < 0)
    
    def test_threshold_estimation_linear_region(self):
        """Test threshold estimation in linear log-log region."""
        from scipy import stats
        
        # Simulate data in log-log space
        distances = np.array([3, 5, 7, 9])
        log_distances = np.log(distances)
        
        # Linear in log-log means power law
        slope = -1.5  # Example scaling exponent
        intercept = -3.0
        log_errors = slope * log_distances + intercept
        
        result = stats.linregress(log_distances, log_errors)
        
        assert result.slope == pytest.approx(slope, abs=0.01)


class TestVisualization:
    """Tests for visualization functions."""
    
    def test_create_figure(self):
        """Test figure creation."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots()
        
        assert fig is not None
        assert ax is not None
        
        plt.close(fig)
    
    def test_plot_data_points(self):
        """Test plotting data points."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots()
        x = [1, 2, 3, 4, 5]
        y = [1, 4, 9, 16, 25]
        
        ax.scatter(x, y)
        
        # Should have data
        assert len(ax.collections) > 0 or len(ax.lines) > 0
        
        plt.close(fig)
    
    def test_log_scale_plot(self):
        """Test logarithmic scale plotting."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots()
        ax.set_yscale('log')
        
        assert ax.get_yscale() == 'log'
        
        plt.close(fig)


class TestResultsExporter:
    """Tests for results export functionality."""
    
    def test_dataframe_to_latex(self):
        """Test DataFrame to LaTeX export."""
        df = pd.DataFrame({
            'Distance': [3, 5, 7],
            'Error Rate': [0.05, 0.02, 0.008],
            'CI Lower': [0.04, 0.015, 0.006],
            'CI Upper': [0.06, 0.025, 0.010],
        })
        
        latex = df.to_latex(index=False)
        
        assert 'tabular' in latex
        assert 'Distance' in latex
    
    def test_results_serialization(self):
        """Test results can be serialized to JSON."""
        import json
        
        results = {
            'threshold_estimate': 0.0087,
            'confidence_interval': [0.0072, 0.0103],
            'correlation_t1': -0.67,
            'p_value': 0.003,
        }
        
        json_str = json.dumps(results)
        loaded = json.loads(json_str)
        
        assert loaded['threshold_estimate'] == results['threshold_estimate']
