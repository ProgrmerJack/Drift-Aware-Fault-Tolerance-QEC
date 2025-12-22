"""
Validate IBM Fez hardware experiment claims from manuscript.
Manuscript claims (lines 166-226):
1. Surface code: |+⟩ LER = 0.5026 ± 0.0103 (3 runs)
2. Surface code: |0⟩ LER = 0.9908 ± 0.0028 (3 runs)
3. Deployment: Baseline LER = 0.3600 ± 0.0079 (2 sessions)
4. Deployment: DAQEC LER = 0.3604 ± 0.0010 (2 sessions)
5. Drift detection: Qubit 3 degraded from 0.4333 to 0.6667
6. Circuit specs: d=3, 17 qubits, 409 depth, 1,170 gates (602 two-qubit)
"""
import json
import numpy as np

print("=" * 80)
print("IBM FEZ HARDWARE VALIDATION")
print("=" * 80)

with open('results/ibm_experiments/experiment_results_20251210_002938.json') as f:
    fez = json.load(f)

print("\n### SURFACE CODE RESULTS ###")
if 'surface_code_results' in fez and len(fez['surface_code_results']) > 0:
    sc = fez['surface_code_results'][0]
    runs = sc.get('runs', [])
    
    # Filter by logical state
    plus_runs = [r for r in runs if r.get('logical_state') == '+']
    zero_runs = [r for r in runs if r.get('logical_state') == '0']
    
    print(f"\n|+⟩ state results (n={len(plus_runs)}):")
    if plus_runs:
        plus_lers = [r['logical_error_rate'] for r in plus_runs]
        plus_mean = np.mean(plus_lers)
        plus_std = np.std(plus_lers, ddof=1) / np.sqrt(len(plus_lers))  # Standard error
        
        print(f"  LER values: {[f'{ler:.4f}' for ler in plus_lers]}")
        print(f"  Mean LER: {plus_mean:.4f}")
        print(f"  Std Error: {plus_std:.4f}")
        print(f"  Manuscript claims: 0.5026 ± 0.0103")
        print(f"  Validation: {'PASS' if abs(plus_mean - 0.5026) < 0.01 else 'CHECK'}")
        
        # Check circuit specs
        sample = plus_runs[0]
        print(f"\n  Circuit specs:")
        print(f"    Depth: {sample.get('circuit_depth', 'N/A')} (manuscript claims: 409)")
        print(f"    Gates: {sample.get('n_gates', 'N/A')} (manuscript claims: 1,170)")
        
        depth_match = sample.get('circuit_depth') == 409
        gates_match = sample.get('n_gates') == 1170
        print(f"    Validation: {'PASS' if depth_match and gates_match else 'CHECK'}")
    else:
        print("  ERROR: No |+⟩ runs found")
    
    print(f"\n|0⟩ state results (n={len(zero_runs)}):")
    if zero_runs:
        zero_lers = [r['logical_error_rate'] for r in zero_runs]
        zero_mean = np.mean(zero_lers)
        zero_std = np.std(zero_lers, ddof=1) / np.sqrt(len(zero_lers))
        
        print(f"  LER values: {[f'{ler:.4f}' for ler in zero_lers]}")
        print(f"  Mean LER: {zero_mean:.4f}")
        print(f"  Std Error: {zero_std:.4f}")
        print(f"  Manuscript claims: 0.9908 ± 0.0028")
        print(f"  Validation: {'PASS' if abs(zero_mean - 0.9908) < 0.01 else 'CHECK'}")
    else:
        print("  ERROR: No |0⟩ runs found")
else:
    print("  ERROR: No surface code results found")

print("\n### DEPLOYMENT STUDY RESULTS ###")
deployment = fez.get('deployment_results', [])

if isinstance(deployment, list) and len(deployment) > 0:
    # Deployment is flat list with 'session_type' field
    baseline_sessions = [s for s in deployment if s.get('session_type') == 'baseline']
    daqec_sessions = [s for s in deployment if s.get('session_type') == 'daqec']
    
    print(f"\nBaseline strategy (n={len(baseline_sessions)}):")
    if baseline_sessions:
        baseline_lers = [s.get('logical_error_rate', s.get('mean_ler', 0)) for s in baseline_sessions]
        baseline_lers = [ler for ler in baseline_lers if ler > 0]
        if baseline_lers:
            baseline_mean = np.mean(baseline_lers)
            baseline_se = np.std(baseline_lers, ddof=1) / np.sqrt(len(baseline_lers)) if len(baseline_lers) > 1 else 0
            print(f"  LER values: {[f'{ler:.4f}' for ler in baseline_lers]}")
            print(f"  Mean LER: {baseline_mean:.4f} ± {baseline_se:.4f}")
            print(f"  Manuscript claims: 0.3600 ± 0.0079")
            print(f"  Validation: {'PASS' if abs(baseline_mean - 0.3600) < 0.01 else 'CHECK'}")
    else:
        print("  No baseline sessions found - check JSON structure")
    
    print(f"\nDAQEC strategy (n={len(daqec_sessions)}):")
    if daqec_sessions:
        daqec_lers = [s.get('logical_error_rate', s.get('mean_ler', 0)) for s in daqec_sessions]
        daqec_lers = [ler for ler in daqec_lers if ler > 0]
        if daqec_lers:
            daqec_mean = np.mean(daqec_lers)
            daqec_se = np.std(daqec_lers, ddof=1) / np.sqrt(len(daqec_lers)) if len(daqec_lers) > 1 else 0
            print(f"  LER values: {[f'{ler:.4f}' for ler in daqec_lers]}")
            print(f"  Mean LER: {daqec_mean:.4f} ± {daqec_se:.4f}")
            print(f"  Manuscript claims: 0.3604 ± 0.0010")
            print(f"  Validation: {'PASS' if abs(daqec_mean - 0.3604) < 0.01 else 'CHECK'}")
    else:
        print("  No DAQEC sessions found - check JSON structure")
else:
    print("  Deployment results structure not as expected. Examining...")
    print(f"  Type: {type(deployment)}")
    if isinstance(deployment, dict):
        print(f"  Keys: {list(deployment.keys())}")
    elif isinstance(deployment, list):
        print(f"  Length: {len(deployment)}")
        if len(deployment) > 0:
            print(f"  First element type: {type(deployment[0])}")
            if isinstance(deployment[0], dict):
                print(f"  First element keys: {list(deployment[0].keys())}")

print("\n### DRIFT DETECTION VALIDATION ###")
# Check for qubit ranking changes
if 'qubit_rankings' in deployment or 'probe_measurements' in deployment:
    print("Checking for qubit ranking stability...")
    print("Manuscript claims: Qubit 3 degraded from 0.4333 to 0.6667")
    # This requires more detailed parsing of deployment results
    print("  (Detailed validation requires probe measurement data)")
else:
    print("  Qubit ranking data not found in top-level deployment results")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("IBM Fez validation requires detailed examination of JSON structure.")
print("Manuscript claims are very specific - verify against raw bitstring counts.")
print("=" * 80)
