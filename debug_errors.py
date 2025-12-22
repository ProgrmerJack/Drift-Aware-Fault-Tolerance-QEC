"""Debug: Print actual error rates to understand why improvement is only 0.2%."""

import sys
sys.path.append('simulations')
from surface_code_simulator import QuantumPlatform, SurfaceCodeSimulator, DriftModel
import numpy as np

platform = QuantumPlatform("ibm")
simulator = SurfaceCodeSimulator(distance=3, platform=platform)

# One test case
np.random.seed(42)
time_since_cal = 12.0  # hours

n_available = 2 * simulator.n_total_qubits  # 34 qubits available

# Generate calibration state
qubits_cal = platform.generate_qubits(n_available, time_since_calibration=0, seed=100)

# Generate current state (evolved from calibration)
qubits_now = platform.generate_qubits(
    n_available, time_since_calibration=time_since_cal, seed=101, base_qubits=qubits_cal
)

# Baseline selects from calibration
selected_baseline = simulator.select_qubits_baseline(qubits_cal)
print(f"Baseline selected: {selected_baseline}")

# DAQEC selects from current (probe)
selected_daqec = simulator.select_qubits_daqec(qubits_cal, qubits_now)
print(f"DAQEC selected: {selected_daqec}")

# Compute errors for each qubit in both selections
print("\n" + "="*80)
print("BASELINE SELECTION - Error rates:")
baseline_errors = []
for idx in selected_baseline:
    q = qubits_now[idx]
    coherence_error = 1 - np.exp(-1.0 / min(q.T1, q.T2))
    total_error = coherence_error + q.gate_error_2q + q.readout_error
    baseline_errors.append(total_error)
    print(f"  Qubit {idx}: T1={q.T1:.1f}, T2={q.T2:.1f}, error={total_error:.6f}")

print(f"\nBaseline errors: mean={np.mean(baseline_errors):.6f}, P95={np.percentile(baseline_errors, 95):.6f}, max={np.max(baseline_errors):.6f}")

print("\n" + "="*80)
print("DAQEC SELECTION - Error rates:")
daqec_errors = []
for idx in selected_daqec:
    q = qubits_now[idx]
    coherence_error = 1 - np.exp(-1.0 / min(q.T1, q.T2))
    total_error = coherence_error + q.gate_error_2q + q.readout_error
    daqec_errors.append(total_error)
    print(f"  Qubit {idx}: T1={q.T1:.1f}, T2={q.T2:.1f}, error={total_error:.6f}")

print(f"\nDAQEC errors: mean={np.mean(daqec_errors):.6f}, P95={np.percentile(daqec_errors, 95):.6f}, max={np.max(daqec_errors):.6f}")

print("\n" + "="*80)
print("COMPARISON:")
print(f"  Mean error reduction: {100*(np.mean(baseline_errors) - np.mean(daqec_errors))/np.mean(baseline_errors):.1f}%")
print(f"  P95 error reduction: {100*(np.percentile(baseline_errors, 95) - np.percentile(daqec_errors, 95))/np.percentile(baseline_errors, 95):.1f}%")
print(f"  Max error reduction: {100*(np.max(baseline_errors) - np.max(daqec_errors))/np.max(baseline_errors):.1f}%")
