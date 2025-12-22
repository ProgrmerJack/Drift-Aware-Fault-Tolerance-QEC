"""Debug LER calculation."""

import sys
sys.path.append('simulations')

import numpy as np
from surface_code_simulator import QuantumPlatform, SurfaceCodeSimulator, DriftModel, QubitParameters

# Run ONE session
np.random.seed(42)

distance = 3
platform = QuantumPlatform("ibm")
simulator = SurfaceCodeSimulator(distance, platform)

time_since_cal = 12.0
n_available = 34
seed = 42

# Generate qubits
qubits_at_calibration = platform.generate_qubits(n_available, 0, DriftModel.GAUSSIAN, seed)
qubits_actual_now = platform.generate_qubits(n_available, time_since_cal, DriftModel.GAUSSIAN, seed+1000, base_qubits=qubits_at_calibration)

qubits_probe = []
for q in qubits_actual_now:
    qubits_probe.append(QubitParameters(
        T1=q.T1 * np.random.lognormal(0, 0.1),
        T2=q.T2 * np.random.lognormal(0, 0.1),
        readout_error=q.readout_error * np.random.lognormal(0, 0.1),
        gate_error_1q=q.gate_error_1q * np.random.lognormal(0, 0.1),
        gate_error_2q=q.gate_error_2q * np.random.lognormal(0, 0.1),
        timestamp=q.timestamp
    ))

# Selection
selected_baseline = simulator.select_qubits_baseline(qubits_at_calibration)
selected_daqec = simulator.select_qubits_daqec(qubits_at_calibration, qubits_probe)

print("Selected baseline:", selected_baseline)
print("Selected DAQEC:", selected_daqec)
print("Overlap:", len(set(selected_baseline) & set(selected_daqec)), "/ 17")

# Compute LER
ler_baseline = simulator.simulate_logical_error_rate(selected_baseline, qubits_actual_now, seed=seed)
ler_daqec = simulator.simulate_logical_error_rate(selected_daqec, qubits_actual_now, seed=seed)

print(f"\nLER baseline: {ler_baseline:.6f}")
print(f"LER DAQEC: {ler_daqec:.6f}")
print(f"Improvement: {100*(ler_baseline - ler_daqec)/ler_baseline:.1f}%")

# Detailed error analysis
print("\n" + "="*80)
print("BASELINE SELECTION - Errors:")
baseline_errors = []
for idx in selected_baseline:
    q = qubits_actual_now[idx]
    coherence_error = 1 - np.exp(-1.0 / min(q.T1, q.T2))
    total_error = coherence_error + q.gate_error_2q + q.readout_error
    baseline_errors.append(total_error)
baseline_errors.sort(reverse=True)
print(f"  Top 5 worst: {[f'{e:.6f}' for e in baseline_errors[:5]]}")
print(f"  Mean: {np.mean(baseline_errors):.6f}")
print(f"  P95: {baseline_errors[int(0.95*len(baseline_errors))]:.6f}")

print("\nDAQEC SELECTION - Errors:")
daqec_errors = []
for idx in selected_daqec:
    q = qubits_actual_now[idx]
    coherence_error = 1 - np.exp(-1.0 / min(q.T1, q.T2))
    total_error = coherence_error + q.gate_error_2q + q.readout_error
    daqec_errors.append(total_error)
daqec_errors.sort(reverse=True)
print(f"  Top 5 worst: {[f'{e:.6f}' for e in daqec_errors[:5]]}")
print(f"  Mean: {np.mean(daqec_errors):.6f}")
print(f"  P95: {daqec_errors[int(0.95*len(daqec_errors))]:.6f}")
