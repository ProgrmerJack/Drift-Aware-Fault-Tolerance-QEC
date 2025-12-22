"""Quick test to verify drift-aware qubit selection logic works correctly."""

import sys
sys.path.append('simulations')

import numpy as np
from surface_code_simulator import QuantumPlatform, DriftModel, QubitParameters

# Remove old code - now using simulator directly

def simulate_experiment(n_available: int, n_select: int, time_since_cal: float, seed: int):
    """Simulate one experimental session."""
    
    # Use actual simulator
    platform = QuantumPlatform("ibm")
    drift_model = DriftModel.GAUSSIAN  # Use Gaussian drift model
    
    # Generate qubits at calibration (t=0)
    qubits_at_cal = platform.generate_qubits(n_available, time_since_calibration=0, 
                                            drift_model=drift_model, seed=seed)
    
    # Same qubits evolved to current time
    qubits_actual = platform.generate_qubits(n_available, time_since_calibration=time_since_cal,
                                            drift_model=drift_model, seed=seed+1000,
                                            base_qubits=qubits_at_cal)
    
    # Add probe noise
    qubits_probe = []
    for q in qubits_actual:
        noise = np.random.lognormal(0, 0.1)
        qubits_probe.append(QubitParameters(
            T1=q.T1 * noise,
            T2=q.T2 * noise,
            gate_error_1q=q.gate_error_1q * noise,
            gate_error_2q=q.gate_error_2q * noise,
            readout_error=q.readout_error * noise,
            timestamp=time_since_cal
        ))
    
    # BASELINE: Select using stale calibration data
    scores_baseline = []
    for q in qubits_at_cal:
        score = (q.T1 + q.T2) / 2 - 1000 * q.gate_error_2q
        scores_baseline.append(score)
    selected_baseline = sorted(range(n_available), key=lambda i: scores_baseline[i], reverse=True)[:n_select]
    
    # DAQEC: Select using fresh probe measurements
    scores_daqec = []
    for q in qubits_probe:
        score = (q.T1 + q.T2) / 2 - 1000 * q.gate_error_2q
        scores_daqec.append(score)
    selected_daqec = sorted(range(n_available), key=lambda i: scores_daqec[i], reverse=True)[:n_select]
    
    # Compute ACTUAL errors on current state
    errors_baseline = []
    for i in selected_baseline:
        q = qubits_actual[i]
        coherence_error = 1 - np.exp(-1.0 / min(q.T1, q.T2))
        total_error = coherence_error + q.gate_error_2q + q.readout_error
        errors_baseline.append(total_error)
    
    errors_daqec = []
    for i in selected_daqec:
        q = qubits_actual[i]
        coherence_error = 1 - np.exp(-1.0 / min(q.T1, q.T2))
        total_error = coherence_error + q.gate_error_2q + q.readout_error
        errors_daqec.append(total_error)
    
    avg_error_baseline = np.mean(errors_baseline)
    avg_error_daqec = np.mean(errors_daqec)
    
    improvement_pct = 100 * (avg_error_baseline - avg_error_daqec) / avg_error_baseline if avg_error_baseline > 0 else 0
    
    return {
        'time_since_cal': time_since_cal,
        'baseline_selection': selected_baseline,
        'daqec_selection': selected_daqec,
        'selections_differ': selected_baseline != selected_daqec,
        'avg_error_baseline': avg_error_baseline,
        'avg_error_daqec': avg_error_daqec,
        'improvement_pct': improvement_pct
    }

if __name__ == "__main__":
    print("="*80)
    print("TESTING DRIFT-AWARE QUBIT SELECTION LOGIC")
    print("="*80)
    
    # Run 10 test experiments
    n_improved = 0
    n_differ = 0
    improvements = []
    
    for i in range(10):
        # Random time since calibration (0-24h)
        time_since_cal = np.random.uniform(0, 24)
        
        result = simulate_experiment(
            n_available=20,  # 20 qubits available
            n_select=9,      # Select 9 (for d=3 surface code)
            time_since_cal=time_since_cal,
            seed=42 + i
        )
        
        if result['selections_differ']:
            n_differ += 1
        
        if result['improvement_pct'] > 0:
            n_improved += 1
            improvements.append(result['improvement_pct'])
        
        print(f"\nExperiment {i+1}:")
        print(f"  Time since calibration: {result['time_since_cal']:.1f} hours")
        print(f"  Selections differ: {result['selections_differ']}")
        print(f"  Baseline error: {result['avg_error_baseline']*100:.3f}%")
        print(f"  DAQEC error: {result['avg_error_daqec']*100:.3f}%")
        print(f"  Improvement: {result['improvement_pct']:.1f}%")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Selections differed: {n_differ}/10 ({100*n_differ/10:.0f}%)")
    print(f"DAQEC improved performance: {n_improved}/10 ({100*n_improved/10:.0f}%)")
    if improvements:
        print(f"Mean improvement: {np.mean(improvements):.1f}% ± {np.std(improvements):.1f}%")
        print(f"Median improvement: {np.median(improvements):.1f}%")
    else:
        print("No improvements observed - LOGIC BUG!")
    
    if n_differ == 0:
        print("\n⚠️  WARNING: Selections never differed - stale vs fresh data generation is broken!")
    elif n_improved < 5:
        print("\n⚠️  WARNING: Low improvement rate - drift model may need tuning!")
    else:
        print("\n✅ SUCCESS: Logic working correctly!")
