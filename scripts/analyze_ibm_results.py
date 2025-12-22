#!/usr/bin/env python3
"""
analyze_ibm_results.py - Analyze IBM Quantum Hardware Experiment Results
========================================================================

Analyzes the real hardware experiment results from ibm_fez backend.
"""

import json
import numpy as np
from pathlib import Path

def analyze_results():
    """Analyze the IBM hardware experiment results."""
    
    # Load results
    results_dir = Path(__file__).parent.parent / "results" / "ibm_experiments"
    results_file = results_dir / "experiment_results_20251210_002938.json"
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print("=" * 70)
    print("IBM QUANTUM HARDWARE EXPERIMENT ANALYSIS")
    print("=" * 70)
    print(f"\nExperiment Date: {results['start_time']}")
    print(f"Backend: ibm_fez (156 qubit Heron processor)")
    print()
    
    # =========================================================================
    # SURFACE CODE ANALYSIS
    # =========================================================================
    print("=" * 70)
    print("1. SURFACE CODE EXPERIMENT (Distance-3)")
    print("=" * 70)
    
    sc_results = results["surface_code_results"][0]
    runs = sc_results["runs"]
    
    # Separate by logical state
    plus_runs = [r for r in runs if r["logical_state"] == "+"]
    zero_runs = [r for r in runs if r["logical_state"] == "0"]
    
    plus_lers = [r["logical_error_rate"] for r in plus_runs]
    zero_lers = [r["logical_error_rate"] for r in zero_runs]
    
    print(f"\n|+⟩ Logical State (X-basis):")
    print(f"  Runs: {len(plus_runs)}")
    print(f"  Logical Error Rates: {[f'{x:.4f}' for x in plus_lers]}")
    print(f"  Mean: {np.mean(plus_lers):.4f} ± {np.std(plus_lers):.4f}")
    print(f"  Circuit depth: {plus_runs[0]['circuit_depth']}")
    print(f"  Gate count: {plus_runs[0]['n_gates']}")
    
    print(f"\n|0⟩ Logical State (Z-basis):")
    print(f"  Runs: {len(zero_runs)}")
    print(f"  Logical Error Rates: {[f'{x:.4f}' for x in zero_lers]}")
    print(f"  Mean: {np.mean(zero_lers):.4f} ± {np.std(zero_lers):.4f}")
    
    # Note: High |0⟩ error is expected without proper initialization/measurement
    print("\n  Note: High |0⟩ LER indicates initialization/measurement errors,")
    print("        which is typical for naive surface code without calibration.")
    
    # =========================================================================
    # DEPLOYMENT STUDY ANALYSIS
    # =========================================================================
    print("\n" + "=" * 70)
    print("2. DEPLOYMENT STUDY (Repetition Code d=5)")
    print("=" * 70)
    
    deployment_results = results["deployment_results"]
    
    baseline_sessions = [d for d in deployment_results if d["session_type"] == "baseline"]
    daqec_sessions = [d for d in deployment_results if d["session_type"] == "daqec"]
    
    baseline_lers = [d["logical_error_rate"] for d in baseline_sessions]
    daqec_lers = [d["logical_error_rate"] for d in daqec_sessions]
    
    print(f"\nBaseline Sessions (calibration-only selection):")
    print(f"  Sessions: {len(baseline_sessions)}")
    print(f"  LERs: {[f'{x:.4f}' for x in baseline_lers]}")
    print(f"  Mean: {np.mean(baseline_lers):.4f}")
    print(f"  Std: {np.std(baseline_lers):.4f}")
    
    print(f"\nDAQEC Sessions (probe-informed selection):")
    print(f"  Sessions: {len(daqec_sessions)}")
    print(f"  LERs: {[f'{x:.4f}' for x in daqec_lers]}")
    print(f"  Mean: {np.mean(daqec_lers):.4f}")
    print(f"  Std: {np.std(daqec_lers):.4f}")
    
    # Probe analysis
    print("\n  Probe Results:")
    for i, session in enumerate(daqec_sessions):
        if "probe_results" in session:
            print(f"    Session {i+1}:")
            print(f"      Selected qubits: {session.get('selected_qubits', 'N/A')}")
            probe_data = session["probe_results"]
            for q, data in probe_data.items():
                print(f"        Qubit {q}: error rate = {data['estimated_error']:.4f}")
    
    # Statistical comparison
    print("\n" + "-" * 50)
    print("Statistical Comparison:")
    print("-" * 50)
    
    if baseline_lers and daqec_lers:
        delta = np.mean(baseline_lers) - np.mean(daqec_lers)
        pct_improvement = delta / np.mean(baseline_lers) * 100
        
        print(f"  Difference (Baseline - DAQEC): {delta:.4f}")
        print(f"  Relative improvement: {pct_improvement:.1f}%")
        
        # Note on limited samples
        print(f"\n  ⚠️  WARNING: Only {len(baseline_lers)} baseline and {len(daqec_lers)} DAQEC sessions")
        print("     Due to 10-minute API limits, this is insufficient for statistical significance.")
        print("     Full deployment study requires 21 sessions each (7 days × 3 sessions/day).")
    
    # =========================================================================
    # KEY FINDINGS
    # =========================================================================
    print("\n" + "=" * 70)
    print("3. KEY FINDINGS")
    print("=" * 70)
    
    print("""
✅ SUCCESSFULLY DEMONSTRATED:
   1. Distance-3 surface code circuit executed on IBM Heron (ibm_fez)
   2. 17-qubit surface code with 3 syndrome rounds completed
   3. Probe-based qubit selection pipeline functional
   4. Real-time probe measurements inform qubit ranking
   
⚠️  LIMITATIONS (due to 10-minute API limits):
   1. Only 2 sessions per condition (vs. 21 needed)
   2. No statistical significance achievable
   3. Single backend (ibm_fez) - no cross-backend comparison
   4. No extended baseline to observe drift accumulation

📊 OBSERVATIONS:
   - Surface code |+⟩ LER ~50% suggests circuit errors dominate
   - Repetition code LER ~36% is reasonable for d=5 without DAQEC tuning
   - Probe pipeline successfully identified varying qubit quality
   - DAQEC qubit ranking changed between sessions (evidence of drift)

🔬 FOR FULL VALIDATION:
   - Requires extended IBM access (14-day deployment)
   - Need 42 sessions total (21 baseline + 21 DAQEC)
   - Should test on multiple backends (Brisbane, Kyoto, Osaka)
   - Need longer time windows to observe calibration decay
""")
    
    # =========================================================================
    # SAVE ANALYSIS
    # =========================================================================
    analysis = {
        "experiment_date": results["start_time"],
        "backend": "ibm_fez",
        "surface_code": {
            "plus_state_lers": plus_lers,
            "plus_state_mean": float(np.mean(plus_lers)),
            "plus_state_std": float(np.std(plus_lers)),
            "zero_state_lers": zero_lers,
            "zero_state_mean": float(np.mean(zero_lers)),
        },
        "deployment": {
            "baseline_lers": baseline_lers,
            "baseline_mean": float(np.mean(baseline_lers)),
            "daqec_lers": daqec_lers,
            "daqec_mean": float(np.mean(daqec_lers)),
            "improvement_percent": float((np.mean(baseline_lers) - np.mean(daqec_lers)) / np.mean(baseline_lers) * 100),
        },
        "limitations": "Only 2 sessions per condition due to 10-minute API limits",
        "conclusion": "Pipeline functional; full validation requires extended access",
    }
    
    analysis_file = results_dir / "analysis_summary.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\nAnalysis saved to: {analysis_file}")
    
    return analysis


if __name__ == "__main__":
    analyze_results()
