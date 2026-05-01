"""
Bermudan option pricing via QPE-based Quantum Amplitude Estimation.

A Bermudan option may only be exercised at a specified set of dates.

Approach
--------
Same as American option, except:
  - The optimal exercise boundary is computed with the restriction that
    early exercise is only allowed at the given exercise_steps.
  - All other time steps only allow holding (no exercise).
"""

import numpy as np
from math import exp, sqrt
from qiskit import transpile

from ..qae_engine import build_path_A_operator, build_qae_circuit, extract_option_price, SAFE_BASIS
from ..classical_models import binomial_bermudan, binomial_bermudan_boundary


def _path_payoffs_bermudan(S, K, T, r, sigma, N_steps, option_type,
                            boundary, exercise_steps):
    """
    Discounted payoff for every path with restricted exercise dates.
    """
    dt = T / N_steps
    u  = exp(sigma * sqrt(dt))
    d  = 1.0 / u
    exercise_set = set(exercise_steps)

    n_paths = 2 ** N_steps
    payoffs = np.zeros(n_paths)

    for path_idx in range(n_paths):
        k = 0
        exercised = False

        for step in range(N_steps):
            up = (path_idx >> step) & 1
            k += up
            t  = step + 1

            if t not in exercise_set:
                continue

            S_t = S * u ** k * d ** (t - k)
            tau = t * dt

            if option_type == "put" and S_t <= boundary[t]:
                payoffs[path_idx] = exp(-r * tau) * max(K - S_t, 0.0)
                exercised = True
                break
            elif option_type == "call" and S_t >= boundary[t]:
                payoffs[path_idx] = exp(-r * tau) * max(S_t - K, 0.0)
                exercised = True
                break

        if not exercised:
            S_T = S * u ** k * d ** (N_steps - k)
            raw = max(K - S_T, 0.0) if option_type == "put" else max(S_T - K, 0.0)
            payoffs[path_idx] = exp(-r * T) * raw

    return payoffs


def price_bermudan(
    S, K, T, r, sigma, option_type,
    n_steps, exercise_steps, n_qpe_qubits,
    backend, shots=100000, top_k=10000,
):
    """
    Price a Bermudan option using QPE-based Quantum Amplitude Estimation.

    Parameters
    ----------
    exercise_steps : list of step indices (1-based) on which exercise is allowed.
                     E.g. [30, 60, 90] for quarterly exercise over 90 steps.
    """
    dt   = T / n_steps
    u    = exp(sigma * sqrt(dt))
    d    = 1.0 / u
    p_rn = (exp(r * dt) - d) / (u - d)

    # Include expiry in exercise set automatically
    exercise_steps = sorted(set(list(exercise_steps) + [n_steps]))

    classical_price, boundary = binomial_bermudan_boundary(
        S, K, T, r, sigma, option_type, exercise_steps, n_steps
    )

    payoffs = _path_payoffs_bermudan(
        S, K, T, r, sigma, n_steps, option_type, boundary, exercise_steps
    )

    A_circ, f_max = build_path_A_operator(p_rn, n_steps, payoffs)
    n_state = n_steps + 1

    qae_circ   = build_qae_circuit(A_circ, n_state, n_qpe_qubits)
    transpiled = transpile(qae_circ, basis_gates=SAFE_BASIS, optimization_level=3)

    result = backend.run(transpiled, repetitions=shots, topK=top_k)
    counts = result.get_counts(None)

    price_data = extract_option_price(counts, n_qpe_qubits, f_max)

    return {
        "quantum_price":   price_data["price"],
        "classical_price": classical_price,
        "amplitude":       price_data["amplitude"],
        "f_max":           f_max,
        "confidence":      price_data["confidence"],
        "exercise_steps":  exercise_steps,
        "circuit_stats": {
            "n_qubits": transpiled.num_qubits,
            "n_gates":  transpiled.size(),
            "depth":    transpiled.depth(),
        },
        "parameters": {
            "S": S, "K": K, "T": T, "r": r, "sigma": sigma,
            "option_type": option_type,
            "n_steps": n_steps,
            "exercise_steps": exercise_steps,
            "n_qpe_qubits": n_qpe_qubits,
        },
    }
