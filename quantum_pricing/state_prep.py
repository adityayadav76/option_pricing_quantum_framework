import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate


def build_distribution_circuit(probs: np.ndarray, n_qubits: int, name: str = "StatePrep") -> QuantumCircuit:
    """
    Prepare sum_i sqrt(probs[i]) |i> using binary-tree (isometry) decomposition.

    This loads an arbitrary probability distribution exactly into quantum amplitudes.
    The resulting circuit applies at most 2^n_qubits - 1 multi-controlled RY rotations.
    """
    n_states = 2 ** n_qubits
    probs = np.array(probs, dtype=float)

    # Pad or truncate to n_states
    if len(probs) < n_states:
        probs = np.pad(probs, (0, n_states - len(probs)))
    else:
        probs = probs[:n_states]

    # Clamp negatives and normalize
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if total > 1e-12:
        probs = probs / total

    qc = QuantumCircuit(n_qubits, name=name)
    _tree_load(qc, probs, list(range(n_qubits)), [])
    return qc


def _tree_load(qc: QuantumCircuit, probs: np.ndarray, qubits: list, controls: list):
    """Recursive binary-tree probability loader."""
    if not qubits or probs.sum() < 1e-12:
        return

    q = qubits[0]
    rest = qubits[1:]
    mid = len(probs) // 2

    p_left = float(probs[:mid].sum())
    p_right = float(probs[mid:].sum())
    total = p_left + p_right

    if total < 1e-12:
        return

    theta = 2.0 * np.arcsin(np.sqrt(np.clip(p_right / total, 0.0, 1.0)))

    if abs(theta) > 1e-10:
        if not controls:
            qc.ry(theta, q)
        else:
            ctrl_qubits = [c[0] for c in controls]
            # ctrl_state string: Qiskit uses big-endian (last control = MSB)
            ctrl_state = "".join("1" if c[1] else "0" for c in reversed(controls))
            gate = RYGate(theta).control(len(controls), ctrl_state=ctrl_state)
            qc.append(gate, ctrl_qubits + [q])

    if p_left > 1e-12 and rest:
        _tree_load(qc, probs[:mid], rest, controls + [(q, 0)])
    if p_right > 1e-12 and rest:
        _tree_load(qc, probs[mid:], rest, controls + [(q, 1)])


def build_path_state_prep(p_rn: float, n_steps: int, name: str = "PathPrep") -> QuantumCircuit:
    """
    Build state preparation for independent binomial-tree path distribution.

    Creates tensor product: (sqrt(1-p)|0> + sqrt(p)|1>)^{⊗ n_steps}
    which encodes the risk-neutral probability of each path as an amplitude.
    """
    theta = 2.0 * np.arcsin(np.sqrt(np.clip(p_rn, 0.0, 1.0)))
    qc = QuantumCircuit(n_steps, name=name)
    if abs(theta) > 1e-10:
        for i in range(n_steps):
            qc.ry(theta, i)
    return qc
