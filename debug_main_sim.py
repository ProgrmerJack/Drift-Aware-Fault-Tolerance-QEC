"""Debug single session to see why selections are identical."""

import sys
sys.path.append('simulations')

import numpy as np
from surface_code_simulator import QuantumPlatform, SurfaceCodeSimulator, DriftModel, QubitParameters

# Run ONE session
np.random.seed(42)

distance = 3
platform = QuantumPlatform("ibm")
simulator = SurfaceCodeSimulator(distance, platform)

time_since_cal = 12.0  # hours
n_available = 2 * simulator.n_total_qubits  # 34 qubits
seed = 42

# Step 1: Calibration state
qubits_at_calibration = platform.generate_qubits(
    n_available,
    time_since_calibration=0,
    drift_model=DriftModel.GAUSSIAN,
    seed=seed
)

# Step 2: Actual state after drift
qubits_actual_now = platform.generate_qubits(
    n_available, 
    time_since_calibration=time_since_cal,
    drift_model=DriftModel.GAUSSIAN,
    seed=seed + 1000,
    base_qubits=qubits_at_calibration
)

# Step 3: Probe measurements
qubits_probe = []
for q_actual in qubits_actual_now:
    qubits_probe.append(QubitParameters(
        T1=q_actual.T1 * np.random.lognormal(0, 0.1),
        T2=q_actual.T2 * np.random.lognormal(0, 0.1),
        readout_error=q_actual.readout_error * np.random.lognormal(0, 0.1),
        gate_error_1q=q_actual.gate_error_1q * np.random.lognormal(0, 0.1),
        gate_error_2q=q_actual.gate_error_2q * np.random.lognormal(0, 0.1),
        timestamp=q_actual.timestamp
    ))

# Step 4: Selection
selected_baseline = simulator.select_qubits_baseline(qubits_at_calibration)
selected_daqec = simulator.select_qubits_daqec(qubits_at_calibration, qubits_probe)

print("Selected baseline:", selected_baseline)
print("Selected DAQEC:", selected_daqec)
print("Selections differ?", selected_baseline != selected_daqec)

# Compute scores for first 5 qubits
print("\n" + "="*80)
print("BASELINE SCORING (from calibration state):")
for i in range(5):
    q = qubits_at_calibration[i]
    score = (q.T1 + q.T2) - 1000 * (q.readout_error + q.gate_error_2q)
    print(f"  Qubit {i}: T1={q.T1:.1f}, T2={q.T2:.1f}, error_2q={q.gate_error_2q:.6f}, score={score:.2f}")

print("\n" + "="*80)
print("DAQEC SCORING (from probe measurements):")
for i in range(5):
    q = qubits_probe[i]
    score = (q.T1 + q.T2) - 1000 * (q.readout_error + q.gate_error_2q)
    print(f"  Qubit {i}: T1={q.T1:.1f}, T2={q.T2:.1f}, error_2q={q.gate_error_2q:.6f}, score={score:.2f}")

print("\n" + "="*80)
print("ACTUAL STATE (what they execute on):")
for i in range(5):
    q = qubits_actual_now[i]
    coherence_error = 1 - np.exp(-1.0 / min(q.T1, q.T2))
    total_error = coherence_error + q.gate_error_2q + q.readout_error
    print(f"  Qubit {i}: T1={q.T1:.1f}, T2={q.T2:.1f}, total_error={total_error:.6f}")
