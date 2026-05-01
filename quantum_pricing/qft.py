import numpy as np
from qiskit import QuantumCircuit


def build_qft(n_qubits: int, inverse: bool = False) -> QuantumCircuit:
    """Build Quantum Fourier Transform (or its inverse) on n_qubits."""
    name = "IQFT" if inverse else "QFT"
    qc = QuantumCircuit(n_qubits, name=name)

    if not inverse:
        for j in range(n_qubits):
            qc.h(j)
            for k in range(j + 1, n_qubits):
                angle = 2.0 * np.pi / (2 ** (k - j + 1))
                qc.cp(angle, k, j)
        for j in range(n_qubits // 2):
            qc.swap(j, n_qubits - j - 1)
    else:
        for j in range(n_qubits // 2):
            qc.swap(j, n_qubits - j - 1)
        for j in range(n_qubits - 1, -1, -1):
            for k in range(n_qubits - 1, j, -1):
                angle = -2.0 * np.pi / (2 ** (k - j + 1))
                qc.cp(angle, k, j)
            qc.h(j)

    return qc
