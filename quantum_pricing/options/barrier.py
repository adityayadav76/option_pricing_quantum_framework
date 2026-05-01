"""
Barrier option pricing via QPE-based Quantum Amplitude Estimation.

Barrier types
-------------
  'up-out'   : knock-out if price rises above barrier; otherwise European payoff
  'down-out' : knock-out if price falls below barrier; otherwise European payoff
  'up-in'    : knock-in if price rises above barrier; payoff=0 otherwise
  'down-in'  : knock-in if price falls below barrier; payoff=0 otherwise

Approach
--------
The N_steps-qubit binomial path superposition is used.  For each path the
barrier-crossing check is performed step-by-step, and the terminal European
payoff is zeroed out (knock-out) or kept (knock-in) accordingly.
"""

import numpy as np
from math import exp, sqrt
from qiskit import transpile

from ..qae_engine import build_path_A_operator, build_qae_circuit, extract_option_price, SAFE_BASIS
from ..classical_models import mc_barrier


BARRIER_TYPES = ("up-out", "down-out", "up-in", "down-in")


def _path_payoffs_barrier(S, K, T, r, sigma, N_steps, option_type,
                           barrier, barrier_type):
    """
    Discounted terminal payoff for every path, respecting the barrier condition.
    """
    barrier_type = barrier_type.lower()
    is_up  = "up"  in barrier_type
    is_out = "out" in barrier_type
    dt   = T / N_steps
    u    = exp(sigma * sqrt(dt))
    d    = 1.0 / u
    disc = exp(-r * T)

    n_paths = 2 ** N_steps
    payoffs = np.zeros(n_paths)

    for path_idx in range(n_paths):
        k = 0
        barrier_hit = False

        for step in range(N_steps):
            up = (path_idx >> step) & 1
            k += up
            S_t = S * u ** k * d ** (step + 1 - k)

            if is_up and S_t >= barrier:
                barrier_hit = True
                break
            elif (not is_up) and S_t <= barrier:
                barrier_hit = True
                break

        if is_out and barrier_hit:
            payoffs[path_idx] = 0.0
            continue
        elif (not is_out) and (not barrier_hit):
            payoffs[path_idx] = 0.0
            continue

        S_T = S * u ** k * d ** (N_steps - k)
        raw = max(S_T - K, 0.0) if option_type == "call" else max(K - S_T, 0.0)
        payoffs[path_idx] = disc * raw

    return payoffs


def price_barrier(
    S, K, T, r, sigma, option_type,
    barrier, barrier_type,
    n_steps, n_qpe_qubits,
    backend, shots=100000, top_k=10000,
):
    """
    Price a barrier option using QPE-based Quantum Amplitude Estimation.

    Parameters
    ----------
    barrier      : barrier level (price threshold)
    barrier_type : one of 'up-out', 'down-out', 'up-in', 'down-in'
    n_steps      : number of monitoring time steps (= path qubits)
    n_qpe_qubits : QPE precision qubits
    backend      : AutomatskiKomencoQiskit instance
    """
    if barrier_type.lower() not in BARRIER_TYPES:
        raise ValueError(f"barrier_type must be one of {BARRIER_TYPES}")

    dt   = T / n_steps
    u    = exp(sigma * sqrt(dt))
    d    = 1.0 / u
    p_rn = (exp(r * dt) - d) / (u - d)

    payoffs = _path_payoffs_barrier(
        S, K, T, r, sigma, n_steps, option_type, barrier, barrier_type
    )

    A_circ, f_max = build_path_A_operator(p_rn, n_steps, payoffs)
    n_state = n_steps + 1

    qae_circ   = build_qae_circuit(A_circ, n_state, n_qpe_qubits)
    transpiled = transpile(qae_circ, basis_gates=SAFE_BASIS, optimization_level=3)

    result = backend.run(transpiled, repetitions=shots, topK=top_k)
    counts = result.get_counts(None)

    price_data = extract_option_price(counts, n_qpe_qubits, f_max)

    classical = mc_barrier(
        S, K, T, r, sigma, option_type.lower(),
        barrier, barrier_type.lower(), N_steps=n_steps
    )

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
            "barrier": barrier,
            "barrier_type": barrier_type,
            "n_steps": n_steps,
            "n_qpe_qubits": n_qpe_qubits,
        },
    }
