"""
Drift-aware qubit selection module.

Provides the primary API for selecting optimal qubit chains based on
probe-validated measurements rather than stale calibration data.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np


@dataclass
class QubitChain:
    """A candidate qubit chain for repetition code execution.
    
    Attributes:
        data_qubits: List of data qubit indices
        ancilla_qubits: List of ancilla qubit indices
        score: Composite quality score (higher is better)
        probe_t1: Mean T1 from probes (microseconds)
        probe_t2: Mean T2 from probes (microseconds)
        probe_readout_error: Mean readout error from probes
        calibration_t1: Mean T1 from calibration (for comparison)
        drift_magnitude: Absolute difference between probe and calibration T1
    """
    data_qubits: List[int]
    ancilla_qubits: List[int]
    score: float
    probe_t1: float
    probe_t2: float
    probe_readout_error: float
    calibration_t1: Optional[float] = None
    drift_magnitude: Optional[float] = None


def select_qubits_drift_aware(
    probe_results: Dict[int, Dict[str, float]],
    code_distance: int,
    backend_topology,  # rustworkx.PyGraph or nx.Graph
    weights: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
    calibration_data: Optional[Dict[int, Dict[str, float]]] = None,
    top_k: int = 5,
) -> List[QubitChain]:
    """Select optimal qubit chains for repetition code based on probe measurements.
    
    This is the primary entry point for drift-aware qubit selection. It ranks
    candidate qubit chains using probe-validated measurements rather than
    potentially stale calibration data.
    
    Parameters
    ----------
    probe_results : dict
        Probe measurement results per qubit. Format:
        {qubit_id: {'T1': float, 'T2': float, 'readout_error': float, 'gate_error': float}}
        
    code_distance : int
        Repetition code distance (3, 5, or 7). Determines chain length.
        
    backend_topology : Graph
        Coupling map as a graph. Must have qubit indices as nodes and
        edges representing valid two-qubit gate connections.
        
    weights : tuple of 4 floats, optional
        Weights for (T1, T2, -readout_error, -gate_error) in composite score.
        Default: equal weights (0.25, 0.25, 0.25, 0.25).
        
    calibration_data : dict, optional
        Calibration data for comparison. If provided, drift magnitude is computed.
        
    top_k : int, optional
        Number of top-ranked chains to return. Default: 5.
        
    Returns
    -------
    list of QubitChain
        Top-k qubit chains ranked by composite score (best first).
        
    Examples
    --------
    >>> from daqec import select_qubits_drift_aware
    >>> 
    >>> probe_results = {
    ...     0: {'T1': 95.2, 'T2': 120.1, 'readout_error': 0.015, 'gate_error': 0.008},
    ...     1: {'T1': 102.3, 'T2': 135.7, 'readout_error': 0.012, 'gate_error': 0.006},
    ...     # ... more qubits
    ... }
    >>> 
    >>> chains = select_qubits_drift_aware(
    ...     probe_results,
    ...     code_distance=5,
    ...     backend_topology=backend.coupling_map
    ... )
    >>> best_chain = chains[0]
    >>> print(f"Best chain: {best_chain.data_qubits}, score={best_chain.score:.3f}")
    
    Notes
    -----
    The composite score is computed as:
    
    .. math::
        S_i = w_1 \\cdot \\hat{T}_{1,i} + w_2 \\cdot \\hat{T}_{2,i} 
              - w_3 \\cdot \\hat{\\epsilon}_{ro,i} - w_4 \\cdot \\hat{\\epsilon}_{2q,i}
    
    where :math:`\\hat{T}_1, \\hat{T}_2` are coherence times and 
    :math:`\\hat{\\epsilon}_{ro}, \\hat{\\epsilon}_{2q}` are error rates,
    all from probe measurements.
    
    References
    ----------
    .. [1] DAQEC-Benchmark v1.0, Zenodo DOI: 10.5281/zenodo.XXXXXXX
    """
    # Validate inputs
    if code_distance not in [3, 5, 7]:
        raise ValueError(f"code_distance must be 3, 5, or 7, got {code_distance}")
    
    n_data_qubits = 2 * code_distance - 1
    n_ancilla_qubits = n_data_qubits - 1  # Each ancilla checks adjacent pairs
    
    # Find all valid linear chains of required length
    candidate_chains = _find_linear_chains(
        backend_topology, 
        n_data_qubits + n_ancilla_qubits,
        probe_results.keys()
    )
    
    if not candidate_chains:
        raise ValueError(
            f"No valid {n_data_qubits + n_ancilla_qubits}-qubit chains found "
            f"in topology with probed qubits {list(probe_results.keys())}"
        )
    
    # Score each chain
    scored_chains = []
    w1, w2, w3, w4 = weights
    
    for chain in candidate_chains:
        # Separate data and ancilla qubits (alternating pattern)
        data_qubits = chain[::2][:n_data_qubits]
        ancilla_qubits = chain[1::2][:n_ancilla_qubits]
        
        # Compute mean metrics across chain
        t1_vals = [probe_results[q]['T1'] for q in chain if q in probe_results]
        t2_vals = [probe_results[q]['T2'] for q in chain if q in probe_results]
        ro_vals = [probe_results[q]['readout_error'] for q in chain if q in probe_results]
        ge_vals = [probe_results[q].get('gate_error', 0.01) for q in chain if q in probe_results]
        
        if not t1_vals:
            continue
            
        mean_t1 = np.mean(t1_vals)
        mean_t2 = np.mean(t2_vals)
        mean_ro = np.mean(ro_vals)
        mean_ge = np.mean(ge_vals)
        
        # Normalize T1/T2 to [0,1] range (assuming max ~200 µs)
        norm_t1 = min(mean_t1 / 200.0, 1.0)
        norm_t2 = min(mean_t2 / 200.0, 1.0)
        
        # Composite score
        score = w1 * norm_t1 + w2 * norm_t2 - w3 * mean_ro - w4 * mean_ge
        
        # Compute drift magnitude if calibration data provided
        drift_mag = None
        cal_t1 = None
        if calibration_data:
            cal_t1_vals = [calibration_data[q]['T1'] for q in chain 
                          if q in calibration_data]
            if cal_t1_vals:
                cal_t1 = np.mean(cal_t1_vals)
                drift_mag = abs(mean_t1 - cal_t1) / cal_t1 if cal_t1 > 0 else None
        
        scored_chains.append(QubitChain(
            data_qubits=list(data_qubits),
            ancilla_qubits=list(ancilla_qubits),
            score=score,
            probe_t1=mean_t1,
            probe_t2=mean_t2,
            probe_readout_error=mean_ro,
            calibration_t1=cal_t1,
            drift_magnitude=drift_mag,
        ))
    
    # Sort by score (descending) and return top-k
    scored_chains.sort(key=lambda x: x.score, reverse=True)
    return scored_chains[:top_k]


def _find_linear_chains(topology, length: int, valid_qubits: set) -> List[List[int]]:
    """Find all linear chains of given length in the topology."""
    chains = []
    valid_set = set(valid_qubits)
    
    # Get adjacency from topology
    if hasattr(topology, 'get_edges'):
        # Qiskit CouplingMap (new API)
        edges = set()
        for edge in topology.get_edges():
            edges.add((edge[0], edge[1]))
            edges.add((edge[1], edge[0]))
    elif hasattr(topology, 'edge_list'):
        # rustworkx PyGraph
        edges = set()
        for edge in topology.edge_list():
            edges.add((edge[0], edge[1]))
            edges.add((edge[1], edge[0]))
    else:
        # networkx Graph or similar with edges() method
        edges = set()
        for u, v in topology.edges():
            edges.add((u, v))
            edges.add((v, u))
    
    # Build adjacency dict
    adj = {}
    for u, v in edges:
        if u in valid_set and v in valid_set:
            adj.setdefault(u, set()).add(v)
    
    # DFS to find all paths of given length
    def dfs(path: List[int]):
        if len(path) == length:
            chains.append(path.copy())
            return
        
        last = path[-1]
        for neighbor in adj.get(last, []):
            if neighbor not in path:  # No revisiting
                path.append(neighbor)
                dfs(path)
                path.pop()
    
    # Start DFS from each valid qubit
    for start in valid_set:
        dfs([start])
    
    return chains
