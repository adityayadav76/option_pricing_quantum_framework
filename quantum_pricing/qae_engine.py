"""
Quantum Amplitude Estimation engine via Quantum Phase Estimation.

Architecture
------------
Operator A (state preparation + payoff oracle):
    A|0> = sum_i sqrt(p_i) |i> ⊗ (sqrt(1 - f_i/F)|0> + sqrt(f_i/F)|1>)
    where F = f_max = max discounted payoff

Good amplitude:  a = sum_i p_i * f_i / F  =  E[discounted_payoff] / F
Option price:    P = F * a

Grover operator  Q = A · S_0 · A† · S_good
    S_good : Z on ancilla qubit            (flips phase of good states)
    S_0    : phase flip of |0…0>           (A·S_0·A† = reflection about |ψ_A>)

QPE applied to Q extracts a phase θ where a = sin²(θ).
Measurement y  →  θ = π·y/2^m  →  a = sin²(θ)

Implementation notes
--------------------
All sub-circuits are inlined via `compose()` rather than wrapped with
`append(QuantumCircuit, ...)`.  This keeps every node in the circuit as a
proper Gate (unitary) so that `.to_gate()` and `.control()` succeed.
For S_0, the multi-controlled X is built with XGate().control(n-1) which
is a proper ControlledGate decomposable by the transpiler via ccx.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import XGate

from .qft import build_qft
from .state_prep import build_distribution_circuit, build_path_state_prep
from .payoff_oracle import build_payoff_oracle


SAFE_BASIS = [
    "ccx", "ccz", "cp", "crz", "cs", "csdg", "cswap", "cu",
    "cx", "cy", "cz", "h", "id", "measure", "reset",
    "p", "rx", "ry", "rz", "s", "sdg", "swap", "sx", "sxdg",
    "t", "tdg", "u", "x", "y", "z",
]


# ---------------------------------------------------------------------------
# A-operator builders (using compose() to keep gates unitary)
# ---------------------------------------------------------------------------

def build_A_operator(probs: np.ndarray, payoffs: np.ndarray, n_reg_qubits: int):
    """
    Build QAE operator A using a general probability distribution.

    probs   : probability of each discrete price/path level
    payoffs : discounted payoff for each level (same length as probs)
    Returns (A_circuit, f_max)
    """
    state_prep = build_distribution_circuit(probs, n_reg_qubits)
    payoff_oracle, f_max = build_payoff_oracle(payoffs, n_reg_qubits)

    n_total = n_reg_qubits + 1
    qc = QuantumCircuit(n_total, name="A")
    # compose() inlines all gates directly (no Instruction wrapper)
    qc.compose(state_prep,    qubits=list(range(n_reg_qubits)), inplace=True)
    qc.compose(payoff_oracle, qubits=list(range(n_total)),      inplace=True)
    return qc, f_max


def build_path_A_operator(p_rn: float, n_steps: int, payoffs: np.ndarray):
    """
    Build QAE operator A for a binomial path model.

    State preparation: n_steps independent RY rotations encoding risk-neutral
    probabilities.  The path |b_0 b_1 … b_{N-1}> encodes up(1)/down(0) moves.

    p_rn    : risk-neutral up-move probability at each step
    n_steps : number of time steps (= number of path qubits)
    payoffs : discounted payoff indexed by integer encoding of path bits
    Returns (A_circuit, f_max)
    """
    state_prep = build_path_state_prep(p_rn, n_steps)
    payoff_oracle, f_max = build_payoff_oracle(payoffs, n_steps)

    n_total = n_steps + 1
    qc = QuantumCircuit(n_total, name="A_path")
    qc.compose(state_prep,    qubits=list(range(n_steps)),  inplace=True)
    qc.compose(payoff_oracle, qubits=list(range(n_total)), inplace=True)
    return qc, f_max


# ---------------------------------------------------------------------------
# Q (Grover) operator
# ---------------------------------------------------------------------------

def build_Q_operator(A_circuit: QuantumCircuit, n_state_qubits: int) -> QuantumCircuit:
    """
    Build Q = A · S_0 · A† · S_good as a pure-gate QuantumCircuit.

    n_state_qubits : total state qubits (price/path register + ancilla)

    Using compose() keeps the result gate-only so .to_gate().control() works.
    MCX for S_0 is built from XGate().control(n-1) — a proper ControlledGate.
    """
    n = n_state_qubits
    qc = QuantumCircuit(n, name="Q")

    # S_good: Z on ancilla (last qubit) — flips phase of |…1> states
    qc.z(n - 1)

    # A†: inverse of state-preparation + payoff oracle
    qc.compose(A_circuit.inverse(), qubits=list(range(n)), inplace=True)

    # S_0: phase flip of |0…0>  =  X^n · MCZ · X^n
    # MCZ implemented as H · MCX · H on last qubit
    for i in range(n):
        qc.x(i)

    if n == 1:
        qc.z(0)
    elif n == 2:
        qc.cz(0, 1)
    else:
        # Multi-controlled-X as ControlledGate (transpile → ccx decomposition)
        mc_x = XGate().control(n - 1)
        qc.h(n - 1)
        qc.append(mc_x, list(range(n - 1)) + [n - 1])
        qc.h(n - 1)

    for i in range(n):
        qc.x(i)

    # A
    qc.compose(A_circuit, qubits=list(range(n)), inplace=True)

    return qc


# ---------------------------------------------------------------------------
# Full QAE circuit
# ---------------------------------------------------------------------------

def build_qae_circuit(A_circuit: QuantumCircuit, n_state_qubits: int, n_qpe_qubits: int) -> QuantumCircuit:
    """
    Build the complete QPE-based Quantum Amplitude Estimation circuit.

    Qubit layout: [QPE register (0…m-1) | State register (m…m+n-1)]
    Classical bits (0…m-1) hold the QPE measurement result.

    The controlled-Q is built by converting Q to a Gate first, which requires
    that Q contains only unitary (Gate) operations — guaranteed by compose().
    """
    m = n_qpe_qubits
    n = n_state_qubits
    n_total = m + n

    qc = QuantumCircuit(n_total, m, name="QAE")

    qpe_qubits   = list(range(m))
    state_qubits = list(range(m, m + n))

    # Hadamard on QPE register → |+>^m
    for q in qpe_qubits:
        qc.h(q)

    # Initialize state register with A (inlined via compose)
    qc.compose(A_circuit, qubits=state_qubits, inplace=True)

    # Build Q as a Gate so we can call .control()
    Q_circ  = build_Q_operator(A_circuit, n)
    Q_gate  = Q_circ.to_gate(label="Q")
    cQ_gate = Q_gate.control(1, label="cQ")

    # Controlled-Q^{2^k} for each QPE qubit k
    for k in range(m):
        reps = 2 ** k
        for _ in range(reps):
            qc.append(cQ_gate, [qpe_qubits[k]] + state_qubits)

    # Inverse QFT on QPE register (inlined)
    iqft = build_qft(m, inverse=True)
    qc.compose(iqft, qubits=qpe_qubits, inplace=True)

    # Measure QPE register into classical bits
    qc.measure(qpe_qubits, list(range(m)))

    return qc


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def extract_option_price(counts: dict, n_qpe_qubits: int, f_max: float) -> dict:
    """
    Extract option price from QPE measurement counts.

    QPE output y  →  canonical_y = min(y, 2^m - y)
                  →  θ = π · canonical_y / 2^m
                  →  a = sin²(θ)
    Option price  = f_max · a  (payoffs were pre-discounted)

    The weighted estimate uses all measurement outcomes for robustness;
    the symmetry sin²(x) = sin²(π-x) means both QPE peaks give the same a.
    """
    m = n_qpe_qubits
    N = 2 ** m

    total_shots = sum(counts.values())
    if total_shots == 0:
        return {"price": 0.0, "amplitude": 0.0, "confidence": 0.0,
                "best_y": 0, "f_max": f_max}

    a_weighted = 0.0
    max_count  = 0
    best_y_raw = 0

    for bitstring, count in counts.items():
        y = int(bitstring, 2)
        y_can = min(y, N - y)
        theta = np.pi * y_can / N
        a = float(np.sin(theta) ** 2)
        a_weighted += (count / total_shots) * a

        if count > max_count:
            max_count  = count
            best_y_raw = y

    best_y_can   = min(best_y_raw, N - best_y_raw)
    option_price = f_max * a_weighted

    return {
        "price":      option_price,
        "amplitude":  a_weighted,
        "f_max":      f_max,
        "best_y":     best_y_can,
        "confidence": max_count / total_shots,
        "n_shots":    total_shots,
    }
