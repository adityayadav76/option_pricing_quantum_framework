"""
Asian option pricing via QPE-based Quantum Amplitude Estimation.

An arithmetic-average Asian option pays max(A_T - K, 0) (call) or
max(K - A_T, 0) (put), where A_T = (1/N) Σ_{t=1}^{N} S_t.

Approach
--------
The full path space of N_steps binomial moves is enumerated.  For each
path the arithmetic average price and resulting payoff are computed
classically.  The QAE circuit uses N_steps independent RY rotations
for state preparation (encoding the risk-neutral path distribution)
and a payoff oracle that applies a conditional rotation to the ancilla
for each distinct path.

Quantum advantage: O(1/ε) vs O(1/ε²) sample complexity over Monte Carlo.
"""

import numpy as np
from math import exp, sqrt
from qiskit import transpile

from ..qae_engine import build_path_A_operator, build_qae_circuit, extract_option_price, SAFE_BASIS
from ..classical_models import mc_asian


def _path_payoffs_asian(S, K, T, r, sigma, N_steps, option_type):
    """
    Compute the discounted arithmetic-average payoff for every binomial path.
    """
    dt = T / N_steps
    u  = exp(sigma * sqrt(dt))
    d  = 1.0 / u
    disc = exp(-r * T)

    n_paths = 2 ** N_steps
    payoffs = np.zeros(n_paths)

    for path_idx in range(n_paths):
        k = 0
        price_sum = 0.0

        for step in range(N_steps):
            up = (path_idx >> step) & 1   # LSB = step 0
            k += up
            S_t = S * u ** k * d ** (step + 1 - k)
            price_sum += S_t

        avg = price_sum / N_steps
        if option_type == "call":
            payoffs[path_idx] = disc * max(avg - K, 0.0)
        else:
            payoffs[path_idx] = disc * max(K - avg, 0.0)

    return payoffs


def price_asian(
    S, K, T, r, sigma, option_type,
    n_steps, n_qpe_qubits,
    backend, shots=100000, top_k=10000,
):
    """
    Price an arithmetic-average Asian option using QPE-based QAE.

    Parameters
    ----------
    n_steps     : number of averaging time steps (= path qubits)
    n_qpe_qubits: QPE precision qubits
    backend     : AutomatskiKomencoQiskit instance
    """
    dt   = T / n_steps
    u    = exp(sigma * sqrt(dt))
    d    = 1.0 / u
    p_rn = (exp(r * dt) - d) / (u - d)

    payoffs = _path_payoffs_asian(S, K, T, r, sigma, n_steps, option_type)

    A_circ, f_max = build_path_A_operator(p_rn, n_steps, payoffs)
    n_state = n_steps + 1

    qae_circ   = build_qae_circuit(A_circ, n_state, n_qpe_qubits)
    transpiled = transpile(qae_circ, basis_gates=SAFE_BASIS, optimization_level=3)

    result = backend.run(transpiled, repetitions=shots, topK=top_k)
    counts = result.get_counts(None)

    price_data = extract_option_price(counts, n_qpe_qubits, f_max)

    classical = mc_asian(S, K, T, r, sigma, option_type.lower(), N_steps=n_steps)

    return {
        "quantum_price":   price_data["price"],
        "classical_price": classical,
        "amplitude":       price_data["amplitude"],
        "f_max":           f_max,
        "confidence":      price_data["confidence"],
        "circuit_stats": {
            "n_qubits": transpiled.num_qubits,
            "n_gates":  transpiled.size(),
            "depth":    transpiled.depth(),
        },
        "parameters": {
            "S": S, "K": K, "T": T, "r": r, "sigma": sigma,
            "option_type": option_type,
            "n_steps": n_steps,
            "n_qpe_qubits": n_qpe_qubits,
        },
    }
