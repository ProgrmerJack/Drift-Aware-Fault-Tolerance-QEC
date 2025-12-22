#!/usr/bin/env python3
"""Test IBM Quantum connection and run a simple experiment."""

import sys
from datetime import datetime

print("=" * 70)
print("IBM QUANTUM CONNECTION TEST")
print("=" * 70)
print()

# Test imports
print("1. Testing imports...")
try:
    from qiskit import QuantumCircuit
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    print("   ✓ All imports successful")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test API keys
print("\n2. Testing API connections...")
API_KEYS = [
    "QXKvh5Ol-rQRrxbs2rdxlo9MLlCpNJioLB8p1_uujfkD",
    "2pbhDH38zmWHgFGw_7Pp8d1ugGvPaa5KR2aTMvW8LJfo",
    "wHH8qtEd9yjYFRRrBKNaExedCLE9JX9qiDG9w3krgYow",
]

working_service = None
working_backend = None

for idx, key in enumerate(API_KEYS):
    try:
        print(f"   Testing API key {idx+1}/3...")
        service = QiskitRuntimeService(channel="ibm_cloud", token=key)
        
        # Get backends
        backends = service.backends(operational=True, simulator=False)
        if backends:
            backend_name = backends[0].name
            print(f"   ✓ API key {idx+1} works! Backend: {backend_name}")
            if working_service is None:
                working_service = service
                working_backend = backend_name
        else:
            print(f"   ✗ API key {idx+1}: No backends available")
    except Exception as e:
        print(f"   ✗ API key {idx+1} failed: {e}")

if working_service is None:
    print("\n✗ No working API keys found!")
    sys.exit(1)

print(f"\n3. Running simple test experiment on {working_backend}...")
try:
    # Create simple Bell state circuit
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    
    print(f"   Circuit created with {qc.num_qubits} qubits")
    
    # Get backend and transpile
    backend = working_service.backend(working_backend)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuit = pm.run(qc)
    
    print(f"   Circuit transpiled (depth: {isa_circuit.depth()})")
    
    # Run with minimal shots
    print("   Submitting job to IBM Quantum...")
    sampler = Sampler(mode=backend)
    job = sampler.run([isa_circuit], shots=100)
    
    print(f"   Job ID: {job.job_id()}")
    print("   Waiting for result...")
    
    result = job.result()
    pub_result = result[0]
    counts = pub_result.data.meas.get_counts()
    
    print(f"   ✓ SUCCESS! Got {len(counts)} measurement outcomes")
    print(f"   Sample counts: {dict(list(counts.items())[:4])}")
    
    print("\n" + "=" * 70)
    print("✓ IBM QUANTUM CONNECTION SUCCESSFUL!")
    print("=" * 70)
    
except Exception as e:
    print(f"\n✗ Experiment failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nReady to run full experiments!")
