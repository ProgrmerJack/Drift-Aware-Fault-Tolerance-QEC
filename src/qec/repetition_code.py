"""
Module C: QEC Repetition Code Implementation
=============================================

Implements distance-3, 5, 7 repetition codes using dynamic circuits.
Uses SamplerV2 primitives for mid-circuit measurement and conditional reset.

References:
- IBM Quantum Documentation: Dynamic Circuits Guide
- Qiskit-QEC: Repetition code implementations
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
import json
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Clbit

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RepetitionCode:
    """
    Repetition code implementation using dynamic circuits.
    
    Implements both bit-flip and phase-flip repetition codes with:
    - Multiple syndrome extraction rounds
    - Mid-circuit measurement
    - Optional conditional reset
    
    The repetition code encodes 1 logical qubit in d physical data qubits,
    with d-1 ancilla qubits for syndrome extraction.
    """
    
    def __init__(self, distance: int = 3, 
                 code_type: str = "bit_flip",
                 num_rounds: int = 1):
        """
        Initialize a repetition code.
        
        Args:
            distance: Code distance (3, 5, or 7)
            code_type: "bit_flip" or "phase_flip"
            num_rounds: Number of syndrome extraction rounds
        """
        self.distance = distance
        self.code_type = code_type
        self.num_rounds = num_rounds
        
        # Number of qubits
        self.num_data = distance
        self.num_ancilla = distance - 1
        self.num_total = self.num_data + self.num_ancilla
        
        # Validate
        if distance not in [3, 5, 7]:
            raise ValueError(f"Distance must be 3, 5, or 7, got {distance}")
        if code_type not in ["bit_flip", "phase_flip"]:
            raise ValueError(f"Code type must be 'bit_flip' or 'phase_flip', got {code_type}")
            
    def build_encoding_circuit(self, initial_state: int = 0) -> QuantumCircuit:
        """
        Build the encoding circuit for the repetition code.
        
        For bit-flip: Encodes |0>_L or |1>_L using CNOT cascade
        For phase-flip: Encodes |+>_L or |->_L using H gates and CNOTs
        
        Args:
            initial_state: 0 for |0>_L/|+>_L or 1 for |1>_L/|->_L
            
        Returns:
            Encoding circuit
        """
        data_reg = QuantumRegister(self.num_data, 'data')
        qc = QuantumCircuit(data_reg, name=f"encode_{self.code_type}_d{self.distance}")
        
        if initial_state == 1:
            qc.x(data_reg[0])  # Prepare |1> on first data qubit
            
        if self.code_type == "bit_flip":
            # Bit-flip code: spread |0>/|1> to all data qubits
            for i in range(1, self.num_data):
                qc.cx(data_reg[0], data_reg[i])
        else:
            # Phase-flip code: create |+...+> or |-...->
            for i in range(self.num_data):
                qc.h(data_reg[i])
                
        return qc
    
    def build_syndrome_extraction_circuit(self,
                                           include_measurement: bool = True,
                                           use_reset: bool = True) -> QuantumCircuit:
        """
        Build syndrome extraction circuit for one round.
        
        For bit-flip code: measures Z_i Z_{i+1} stabilizers
        For phase-flip code: measures X_i X_{i+1} stabilizers (requires H basis change)
        
        Args:
            include_measurement: Whether to include ancilla measurements
            use_reset: Whether to reset ancillas after measurement (for repeated rounds)
            
        Returns:
            Syndrome extraction circuit
        """
        data_reg = QuantumRegister(self.num_data, 'data')
        ancilla_reg = QuantumRegister(self.num_ancilla, 'ancilla')
        
        if include_measurement:
            syndrome_reg = ClassicalRegister(self.num_ancilla, 'syndrome')
            qc = QuantumCircuit(data_reg, ancilla_reg, syndrome_reg,
                                name=f"syndrome_{self.code_type}")
        else:
            qc = QuantumCircuit(data_reg, ancilla_reg,
                                name=f"syndrome_{self.code_type}")
        
        if self.code_type == "bit_flip":
            # Z_i Z_{i+1} stabilizers using CNOT
            # Ancilla i measures parity of data[i] and data[i+1]
            for i in range(self.num_ancilla):
                qc.cx(data_reg[i], ancilla_reg[i])
                qc.cx(data_reg[i + 1], ancilla_reg[i])
        else:
            # X_i X_{i+1} stabilizers
            # Change to X basis
            for i in range(self.num_data):
                qc.h(data_reg[i])
                
            # Same parity checks in X basis
            for i in range(self.num_ancilla):
                qc.cx(data_reg[i], ancilla_reg[i])
                qc.cx(data_reg[i + 1], ancilla_reg[i])
                
            # Return to Z basis
            for i in range(self.num_data):
                qc.h(data_reg[i])
                
        if include_measurement:
            qc.measure(ancilla_reg, syndrome_reg)
            
            if use_reset:
                # Reset ancillas for next round
                for i in range(self.num_ancilla):
                    qc.reset(ancilla_reg[i])
                    
        return qc
    
    def build_full_experiment_circuit(self, 
                                       initial_state: int = 0,
                                       inject_error: Optional[int] = None,
                                       error_round: int = 0) -> QuantumCircuit:
        """
        Build a complete QEC experiment circuit.
        
        Includes:
        1. Encoding
        2. Optional error injection (for testing)
        3. Multiple syndrome extraction rounds
        4. Final data qubit measurement
        
        Args:
            initial_state: 0 or 1 for logical state
            inject_error: Optional qubit index to inject X error on
            error_round: Round after which to inject error
            
        Returns:
            Complete experiment circuit
        """
        data_reg = QuantumRegister(self.num_data, 'data')
        ancilla_reg = QuantumRegister(self.num_ancilla, 'ancilla')
        
        # Classical registers for each syndrome round + final measurement
        syndrome_regs = [ClassicalRegister(self.num_ancilla, f'syn_r{r}') 
                        for r in range(self.num_rounds)]
        final_reg = ClassicalRegister(self.num_data, 'final')
        
        qc = QuantumCircuit(data_reg, ancilla_reg, *syndrome_regs, final_reg,
                           name=f"rep_code_{self.code_type}_d{self.distance}_r{self.num_rounds}")
        
        # Step 1: Encoding
        encoding = self.build_encoding_circuit(initial_state)
        qc.compose(encoding, qubits=data_reg, inplace=True)
        qc.barrier()
        
        # Step 2-3: Syndrome extraction rounds with optional error injection
        for r in range(self.num_rounds):
            # Inject error after specified round
            if inject_error is not None and r == error_round:
                qc.x(data_reg[inject_error])
                qc.barrier()
                
            # Syndrome extraction
            if self.code_type == "bit_flip":
                for i in range(self.num_ancilla):
                    qc.cx(data_reg[i], ancilla_reg[i])
                    qc.cx(data_reg[i + 1], ancilla_reg[i])
            else:
                for i in range(self.num_data):
                    qc.h(data_reg[i])
                for i in range(self.num_ancilla):
                    qc.cx(data_reg[i], ancilla_reg[i])
                    qc.cx(data_reg[i + 1], ancilla_reg[i])
                for i in range(self.num_data):
                    qc.h(data_reg[i])
                    
            # Measure syndromes
            qc.measure(ancilla_reg, syndrome_regs[r])
            
            # Reset ancillas for next round (except last)
            if r < self.num_rounds - 1:
                for i in range(self.num_ancilla):
                    qc.reset(ancilla_reg[i])
                    
            qc.barrier()
            
        # Step 4: Final measurement
        if self.code_type == "phase_flip":
            # Measure in X basis
            for i in range(self.num_data):
                qc.h(data_reg[i])
                
        qc.measure(data_reg, final_reg)
        
        return qc
    
    def get_qubit_layout(self, physical_layout: List[int]) -> Dict[str, List[int]]:
        """
        Map logical qubits to physical layout.
        
        For repetition code, physical_layout should be a linear chain of
        2*distance - 1 qubits, alternating data and ancilla.
        
        Args:
            physical_layout: List of physical qubit indices in order
            
        Returns:
            Dictionary with 'data' and 'ancilla' qubit lists
        """
        if len(physical_layout) != self.num_total:
            raise ValueError(f"Layout must have {self.num_total} qubits, got {len(physical_layout)}")
            
        # Interleaved layout: D A D A D A D (for d=4)
        # or: D A D A D (for d=3)
        data_qubits = physical_layout[::2]  # Even indices
        ancilla_qubits = physical_layout[1::2]  # Odd indices
        
        return {
            "data": data_qubits,
            "ancilla": ancilla_qubits
        }


class SyndromeDecoder:
    """
    Simple majority vote decoder for repetition codes.
    
    For more advanced decoding, use MWPM (Minimum Weight Perfect Matching).
    """
    
    def __init__(self, distance: int):
        """
        Initialize decoder.
        
        Args:
            distance: Code distance
        """
        self.distance = distance
        self.num_data = distance
        self.num_ancilla = distance - 1
        
    def decode_syndrome(self, syndrome: str) -> int:
        """
        Decode a single syndrome measurement.
        
        Uses boundary matching: finds the error location that
        explains the syndrome pattern with minimum weight.
        
        Args:
            syndrome: Binary string of syndrome bits (e.g., "01")
            
        Returns:
            Estimated error location (-1 if no error)
        """
        syndrome_bits = [int(b) for b in reversed(syndrome)]
        
        # No syndrome -> no error
        if sum(syndrome_bits) == 0:
            return -1
            
        # Single syndrome triggered -> error at boundary or that position
        if sum(syndrome_bits) == 1:
            pos = syndrome_bits.index(1)
            # Could be data qubit pos or pos+1
            return pos
            
        # Find defect pair (adjacent syndromes that differ)
        error_pos = -1
        for i in range(len(syndrome_bits) - 1):
            if syndrome_bits[i] != syndrome_bits[i + 1]:
                error_pos = i + 1  # Error on data qubit between ancillas
                break
                
        if error_pos == -1 and syndrome_bits[0] == 1:
            error_pos = 0  # Error on first data qubit
            
        return error_pos
    
    def decode_multi_round(self, syndromes: List[str]) -> Tuple[int, Dict[str, Any]]:
        """
        Decode multiple syndrome rounds using temporal correlation.
        
        Looks for syndrome changes between rounds to identify errors.
        
        Args:
            syndromes: List of syndrome strings, one per round
            
        Returns:
            Tuple of (corrected_logical, analysis_dict)
        """
        # Track syndrome changes
        changes = []
        prev = "0" * self.num_ancilla
        
        for syn in syndromes:
            # XOR with previous to find changes
            change = "".join(str(int(a) ^ int(b)) for a, b in zip(syn, prev))
            changes.append(change)
            prev = syn
            
        # Count error events
        error_events = sum(sum(int(b) for b in c) for c in changes)
        
        # Simple majority vote on final syndrome
        final_syndrome = syndromes[-1]
        error_pos = self.decode_syndrome(final_syndrome)
        
        # Estimate if logical error occurred
        # For repetition code, odd number of errors = logical error
        logical_error = error_events % 2 == 1
        
        return int(logical_error), {
            "error_position": error_pos,
            "syndrome_changes": changes,
            "total_events": error_events
        }
    
    def decode_counts(self, counts: Dict[str, int]) -> Dict[str, Any]:
        """
        Decode measurement results from multiple shots.
        
        Args:
            counts: Dictionary of measurement outcomes -> counts
            
        Returns:
            Analysis dictionary with decoded results
        """
        # Parse measurement results
        # Format depends on circuit structure
        # Typically: "final_bits syndrome_r1 syndrome_r0"
        
        logical_0_count = 0
        logical_1_count = 0
        total_errors = 0
        
        for outcome, count in counts.items():
            parts = outcome.split()
            
            if len(parts) >= 1:
                # Final measurement result
                final = parts[0]
                # Majority vote on data qubits for logical state
                ones = sum(int(b) for b in final)
                if ones > self.num_data // 2:
                    logical_1_count += count
                else:
                    logical_0_count += count
                    
                # Check for errors (non-zero syndromes)
                for part in parts[1:]:
                    if any(b == '1' for b in part):
                        total_errors += count
                        break
                        
        total = logical_0_count + logical_1_count
        
        return {
            "logical_0_probability": logical_0_count / total if total > 0 else 0,
            "logical_1_probability": logical_1_count / total if total > 0 else 0,
            "error_rate": total_errors / (total * self.num_rounds) if total > 0 else 0,
            "total_shots": total
        }


def create_experiment_batch(distances: List[int] = [3, 5, 7],
                            num_rounds: List[int] = [1, 3, 5],
                            code_type: str = "bit_flip") -> List[QuantumCircuit]:
    """
    Create a batch of QEC experiment circuits for different parameters.
    
    Args:
        distances: List of code distances to test
        num_rounds: List of syndrome rounds to test
        code_type: "bit_flip" or "phase_flip"
        
    Returns:
        List of experiment circuits
    """
    circuits = []
    
    for d in distances:
        for r in num_rounds:
            code = RepetitionCode(distance=d, code_type=code_type, num_rounds=r)
            
            # Create circuits for both logical states
            for state in [0, 1]:
                circ = code.build_full_experiment_circuit(initial_state=state)
                circ.name = f"rep_{code_type}_d{d}_r{r}_state{state}"
                circuits.append(circ)
                
    return circuits


if __name__ == "__main__":
    print("Repetition Code Module")
    print("\nExample circuit (distance-3 bit-flip code):")
    
    code = RepetitionCode(distance=3, code_type="bit_flip", num_rounds=2)
    circuit = code.build_full_experiment_circuit(initial_state=0)
    print(circuit)
    
    print("\nExperiment batch for distances 3, 5:")
    batch = create_experiment_batch(distances=[3, 5], num_rounds=[1, 2])
    for circ in batch:
        print(f"  - {circ.name}: {circ.num_qubits} qubits, {circ.depth()} depth")
