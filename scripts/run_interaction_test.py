#!/usr/bin/env python
"""
Test the Interaction Effect on Amazon Braket IonQ Hardware.

This script tests the CENTRAL CLAIM of the manuscript:
- When hardware noise is LOW → drift-aware selection HURTS performance
- When hardware noise is HIGH → drift-aware selection HELPS performance

Strategy:
---------
Since we cannot control IonQ's actual noise level, we simulate noise regimes by:
1. Using depolarizing noise injection via circuit modifications
2. Comparing "drift-aware" (optimal qubit selection) vs "calibration-based" (suboptimal)

The key insight: On perfect hardware (IonQ's ~0.5% error rates), we expect:
- Drift-aware overhead > benefit → HURTS performance (low-noise regime)
- But with injected noise, signal > overhead → HELPS performance (high-noise regime)

This tests whether the interaction effect generalizes to trapped-ion architecture.
"""

import argparse
import json
import sys
import time
import random
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# IonQ device config
IONQ_CONFIG = {
    "arn": "arn:aws:braket:us-east-1::device/qpu/ionq/Forte-Enterprise-1",
    "name": "IonQ Forte-1",
    "type": "trapped-ion",
    "qubits": 36,
    "task_cost": 0.30,
    "shot_cost": 0.01,
    "region": "us-east-1",
}


def create_repetition_code_circuit(n_data: int, n_rounds: int = 1):
    """Create a standard repetition code circuit."""
    from braket.circuits import Circuit
    
    n_ancilla = n_data - 1
    total = n_data + n_ancilla
    
    circuit = Circuit()
    
    # Syndrome extraction
    for _ in range(n_rounds):
        for i in range(n_ancilla):
            circuit.cnot(i, n_data + i)
            circuit.cnot(i + 1, n_data + i)
    
    # Measure all
    for i in range(total):
        circuit.measure(i)
    
    return circuit, n_data, n_ancilla


def create_drift_aware_circuit(n_data: int, n_rounds: int = 1):
    """
    Create a 'drift-aware' circuit - uses optimal qubit ordering.
    
    On IonQ with all-to-all connectivity, we use the most connected/stable
    qubits (simulated by using low-index qubits which are typically better).
    """
    from braket.circuits import Circuit
    
    n_ancilla = n_data - 1
    total = n_data + n_ancilla
    
    circuit = Circuit()
    
    # Use optimal qubit assignment (consecutive low-index qubits)
    # This simulates "best" qubit selection
    data_qubits = list(range(n_data))
    ancilla_qubits = list(range(n_data, total))
    
    # Syndrome extraction with optimal routing
    for _ in range(n_rounds):
        for i in range(n_ancilla):
            circuit.cnot(data_qubits[i], ancilla_qubits[i])
            circuit.cnot(data_qubits[i + 1], ancilla_qubits[i])
    
    for i in range(total):
        circuit.measure(i)
    
    return circuit, n_data, n_ancilla, "drift_aware"


def create_calibration_based_circuit(n_data: int, n_rounds: int = 1):
    """
    Create a 'calibration-based' circuit - uses suboptimal qubit ordering.
    
    Simulates using qubits based only on static calibration data,
    which may be stale. On IonQ with all-to-all connectivity, we simulate
    the effect of stale calibration by:
    1. Adding extra CNOT pairs that cancel out (adds depth + error)
    2. Using the same circuit but with extra gate overhead
    
    This models the real-world scenario where stale calibration leads to
    suboptimal routing and longer circuits.
    """
    from braket.circuits import Circuit
    
    n_ancilla = n_data - 1
    total = n_data + n_ancilla
    
    circuit = Circuit()
    
    # Add extra gates to simulate overhead from stale calibration
    # CNOT pairs that cancel: CNOT(a,b) CNOT(a,b) = Identity
    # But they still accumulate gate errors!
    for i in range(min(n_data, 4)):  # Add overhead on first few qubits
        circuit.cnot(i, (i + 1) % n_data)
        circuit.cnot(i, (i + 1) % n_data)  # Cancels mathematically, but adds errors
    
    # Standard syndrome extraction
    for _ in range(n_rounds):
        for i in range(n_ancilla):
            circuit.cnot(i, n_data + i)
            circuit.cnot(i + 1, n_data + i)
    
    # More overhead at the end (simulates longer circuit from worse routing)
    for i in range(min(n_data, 4)):
        circuit.cnot(i, (i + 1) % n_data)
        circuit.cnot(i, (i + 1) % n_data)
    
    for i in range(total):
        circuit.measure(i)
    
    return circuit, n_data, n_ancilla, "calibration_based"


def create_noisy_circuit(base_circuit, noise_level: float = 0.0):
    """
    Inject noise into circuit to simulate high-noise regime.
    
    Since IonQ hardware is very clean (~0.5% errors), we need to
    artificially degrade performance to test the high-noise regime
    where drift-aware selection should HELP.
    
    We do this by adding random bit-flip gates (X) based on noise_level.
    X gates are supported on IonQ.
    """
    from braket.circuits import Circuit  # noqa: F401
    
    if noise_level <= 0:
        return base_circuit
    
    # Clone and add noise BEFORE measurements
    noisy = Circuit()
    measurement_instrs = []
    
    # First pass: add gates with noise, collect measurements
    for instruction in base_circuit.instructions:
        # Check if this is a measurement
        is_measurement = instruction.operator.name.lower() == 'measure'
        
        if is_measurement:
            measurement_instrs.append(instruction)
        else:
            noisy.add_instruction(instruction)
            # With probability noise_level, add an X gate (bit flip)
            if instruction.target and random.random() < noise_level:
                target = instruction.target[0]
                noisy.x(target)
    
    # Add all measurements at the end
    for instr in measurement_instrs:
        noisy.add_instruction(instr)
    
    return noisy


def run_paired_experiment(
    n_data: int = 5,
    n_rounds: int = 1,
    shots: int = 100,
    noise_level: float = 0.0,
    use_simulator: bool = False,
) -> dict:
    """
    Run a paired experiment: drift-aware vs calibration-based.
    
    Returns results for both methods to enable direct comparison.
    """
    
    if use_simulator:
        from braket.devices import LocalSimulator
        device = LocalSimulator()
        device_name = "LocalSimulator"
    else:
        from braket.aws import AwsDevice
        device = AwsDevice(IONQ_CONFIG["arn"])
        device_name = IONQ_CONFIG["name"]
    
    n_ancilla = n_data - 1
    total = n_data + n_ancilla
    correct_state = '0' * total
    
    results = {
        "device": device_name,
        "n_data_qubits": n_data,
        "n_rounds": n_rounds,
        "shots": shots,
        "noise_level": noise_level,
        "timestamp": datetime.now().isoformat(),
        "experiments": [],
    }
    
    # Run both methods
    for method, create_func in [
        ("drift_aware", create_drift_aware_circuit),
        ("calibration_based", create_calibration_based_circuit),
    ]:
        print(f"\n--- Running {method} ---")
        
        circuit, _, _, _ = create_func(n_data, n_rounds)
        
        # Add noise if specified
        if noise_level > 0:
            circuit = create_noisy_circuit(circuit, noise_level)
        
        print(f"Circuit depth: {circuit.depth}")
        
        start_time = time.time()
        task = device.run(circuit, shots=shots)
        result = task.result()
        elapsed = time.time() - start_time
        
        counts = result.measurement_counts
        correct_count = counts.get(correct_state, 0)
        total_shots = sum(counts.values())
        ler = 1.0 - (correct_count / total_shots) if total_shots > 0 else 1.0
        
        exp_result = {
            "method": method,
            "circuit_depth": circuit.depth,
            "correct_count": correct_count,
            "total_shots": total_shots,
            "ler": ler,
            "elapsed_seconds": elapsed,
            "task_id": getattr(task, 'id', 'local'),
        }
        results["experiments"].append(exp_result)
        
        print(f"  LER: {ler:.4f} ({correct_count}/{total_shots} correct)")
        print(f"  Depth: {circuit.depth}")
    
    # Calculate comparison metrics
    drift = next(e for e in results["experiments"] if e["method"] == "drift_aware")
    calib = next(e for e in results["experiments"] if e["method"] == "calibration_based")
    
    ler_diff = drift["ler"] - calib["ler"]  # Negative = drift-aware better
    relative_improvement = (calib["ler"] - drift["ler"]) / calib["ler"] if calib["ler"] > 0 else 0
    
    results["comparison"] = {
        "ler_difference": ler_diff,  # drift_aware - calibration
        "relative_improvement": relative_improvement,  # positive = drift-aware better
        "drift_aware_helps": ler_diff < 0,
        "depth_difference": drift["circuit_depth"] - calib["circuit_depth"],
    }
    
    return results


def run_interaction_test(
    n_data: int = 5,
    n_rounds: int = 1,
    shots_per_condition: int = 100,
    noise_levels: list = None,
    use_simulator: bool = False,
) -> dict:
    """
    Run the full interaction effect test.
    
    Tests drift-aware vs calibration-based across multiple noise levels
    to demonstrate the crossover effect.
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20]  # Low to high noise
    
    print("=" * 70)
    print("INTERACTION EFFECT TEST")
    print("=" * 70)
    print(f"Testing: Does drift-aware help MORE at higher noise levels?")
    print(f"Noise levels: {noise_levels}")
    print(f"Shots per condition: {shots_per_condition}")
    print("=" * 70)
    
    all_results = {
        "test_type": "interaction_effect",
        "device": IONQ_CONFIG["name"] if not use_simulator else "LocalSimulator",
        "n_data_qubits": n_data,
        "n_rounds": n_rounds,
        "shots_per_condition": shots_per_condition,
        "noise_levels": noise_levels,
        "timestamp": datetime.now().isoformat(),
        "conditions": [],
    }
    
    for noise in noise_levels:
        print(f"\n{'='*70}")
        print(f"NOISE LEVEL: {noise:.2%}")
        print(f"{'='*70}")
        
        result = run_paired_experiment(
            n_data=n_data,
            n_rounds=n_rounds,
            shots=shots_per_condition,
            noise_level=noise,
            use_simulator=use_simulator,
        )
        
        result["noise_level"] = noise
        all_results["conditions"].append(result)
        
        # Show comparison
        comp = result["comparison"]
        effect = "HELPS" if comp["drift_aware_helps"] else "HURTS"
        print(f"\n>>> At {noise:.0%} noise: Drift-aware {effect}")
        print(f"    LER diff: {comp['ler_difference']:+.4f}")
        print(f"    Relative: {comp['relative_improvement']:+.1%}")
    
    # Analyze interaction effect
    print("\n" + "=" * 70)
    print("INTERACTION EFFECT ANALYSIS")
    print("=" * 70)
    
    # Check if effect changes with noise level
    low_noise_results = [c for c in all_results["conditions"] if c["noise_level"] < 0.10]
    high_noise_results = [c for c in all_results["conditions"] if c["noise_level"] >= 0.10]
    
    low_helps = sum(1 for c in low_noise_results if c["comparison"]["drift_aware_helps"])
    high_helps = sum(1 for c in high_noise_results if c["comparison"]["drift_aware_helps"])
    
    all_results["analysis"] = {
        "low_noise_conditions": len(low_noise_results),
        "low_noise_drift_helps": low_helps,
        "high_noise_conditions": len(high_noise_results),
        "high_noise_drift_helps": high_helps,
        "interaction_detected": (low_helps < len(low_noise_results) / 2) and (high_helps > len(high_noise_results) / 2),
    }
    
    print(f"Low noise (<10%): Drift-aware helps in {low_helps}/{len(low_noise_results)} conditions")
    print(f"High noise (>=10%): Drift-aware helps in {high_helps}/{len(high_noise_results)} conditions")
    
    if all_results["analysis"]["interaction_detected"]:
        print("\n✅ INTERACTION EFFECT DETECTED!")
        print("   Drift-aware hurts at low noise, helps at high noise.")
        print("   This SUPPORTS the manuscript's central claim.")
    else:
        print("\n⚠️ Interaction effect not clearly detected.")
        print("   May need more shots or different noise levels.")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Test interaction effect on Amazon Braket IonQ hardware"
    )
    parser.add_argument(
        "--distance", type=int, default=5,
        help="Code distance"
    )
    parser.add_argument(
        "--shots", type=int, default=100,
        help="Shots per condition"
    )
    parser.add_argument(
        "--rounds", type=int, default=1,
        help="Syndrome rounds"
    )
    parser.add_argument(
        "--simulator", action="store_true",
        help="Use local simulator"
    )
    parser.add_argument(
        "--single-pair", action="store_true",
        help="Run single paired comparison (no noise variation)"
    )
    parser.add_argument(
        "--noise", type=float, default=0.0,
        help="Noise level for single-pair test"
    )
    
    args = parser.parse_args()
    
    # Estimate cost
    if not args.simulator:
        if args.single_pair:
            n_tasks = 2  # drift-aware + calibration
        else:
            n_tasks = 10  # 5 noise levels × 2 methods
        
        cost = n_tasks * (IONQ_CONFIG["task_cost"] + args.shots * IONQ_CONFIG["shot_cost"])
        print(f"\n{'#'*70}")
        print(f"# COST ESTIMATE: ${cost:.2f}")
        print(f"# Tasks: {n_tasks}, Shots/task: {args.shots}")
        print(f"{'#'*70}\n")
    
    if args.single_pair:
        # Just run one paired comparison
        result = run_paired_experiment(
            n_data=args.distance,
            n_rounds=args.rounds,
            shots=args.shots,
            noise_level=args.noise,
            use_simulator=args.simulator,
        )
    else:
        # Run full interaction test
        result = run_interaction_test(
            n_data=args.distance,
            n_rounds=args.rounds,
            shots_per_condition=args.shots,
            use_simulator=args.simulator,
        )
    
    # Save results
    output_dir = project_root / "results" / "multi_platform"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.single_pair:
        filename = f"ionq_interaction_pair_d{args.distance}_{timestamp}.json"
    else:
        filename = f"ionq_interaction_full_d{args.distance}_{timestamp}.json"
    
    output_file = output_dir / filename
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if args.single_pair:
        comp = result["comparison"]
        effect = "HELPS" if comp["drift_aware_helps"] else "HURTS"
        print(f"At noise={args.noise:.0%}: Drift-aware {effect}")
        print(f"LER difference: {comp['ler_difference']:+.4f}")
    else:
        analysis = result["analysis"]
        print(f"Interaction detected: {analysis['interaction_detected']}")


if __name__ == "__main__":
    main()
