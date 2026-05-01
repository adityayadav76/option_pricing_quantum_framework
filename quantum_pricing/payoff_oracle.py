import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate


def build_payoff_oracle(payoffs: np.ndarray, n_reg_qubits: int, name: str = "PayoffOracle"):
    """
    Build the payoff oracle that rotates the ancilla qubit proportional to the payoff.

    For each basis state |i> of the n_reg_qubits register, applies:
        |i>|0>  →  |i>(sqrt(1 - f_i/f_max)|0> + sqrt(f_i/f_max)|1>)

    Circuit acts on n_reg_qubits + 1 qubits (register qubits then ancilla).
    Payoffs should already be discounted (pre-multiplied by e^{-rT} or e^{-r*tau}).

    Returns (circuit, f_max).
    """
    payoffs = np.asarray(payoffs, dtype=float)
    payoffs = np.clip(payoffs, 0.0, None)
    f_max = float(payoffs.max()) if payoffs.size > 0 else 0.0

    n_total = n_reg_qubits + 1
    ancilla = n_reg_qubits
    qc = QuantumCircuit(n_total, name=name)

    if f_max < 1e-12:
        return qc, max(f_max, 1e-12)

    n_states = 2 ** n_reg_qubits

    for i in range(min(len(payoffs), n_states)):
        val = payoffs[i] / f_max
        if val < 1e-12:
            continue

        theta = 2.0 * np.arcsin(np.sqrt(np.clip(val, 0.0, 1.0)))
        if abs(theta) < 1e-10:
            continue

        # Bit-pattern of index i; bit j is the j-th qubit (LSB = qubit 0)
        bit_str = format(i, f"0{n_reg_qubits}b")  # MSB first in string

        # Flip qubits where the corresponding bit is '0' so that
        # after the X gates, |i> maps to |1...1> for the control condition.
        flip_qubits = [j for j, b in enumerate(reversed(bit_str)) if b == "0"]
        for j in flip_qubits:
            qc.x(j)

        _mcry(qc, theta, list(range(n_reg_qubits)), ancilla)

        for j in flip_qubits:
            qc.x(j)

    return qc, f_max


def _mcry(qc: QuantumCircuit, theta: float, ctrl_qubits: list, target: int):
    """Apply multi-controlled RY gate; transpiler will decompose to safe basis."""
    n_ctrl = len(ctrl_qubits)
    if n_ctrl == 0:
        qc.ry(theta, target)
    elif n_ctrl == 1:
        # Standard CRY decomposition using CX + RY
        qc.ry(theta / 2, target)
        qc.cx(ctrl_qubits[0], target)
        qc.ry(-theta / 2, target)
        qc.cx(ctrl_qubits[0], target)
    else:
        gate = RYGate(theta).control(n_ctrl)
        qc.append(gate, ctrl_qubits + [target])
