"""
American option pricing via QPE-based Quantum Amplitude Estimation.

Approach
--------
1. Compute the optimal exercise boundary B[t] for each time step t via classical
   binomial backward induction  (O(N²) classical pre-processing).

2. Build a quantum circuit over the full space of 2^N_steps binomial paths.
   Each path qubit encodes an independent up(1)/down(0) move with risk-neutral
   probability p.  The state preparation is N_steps independent RY rotations.

3. The payoff oracle: for each of the 2^N_steps paths, simulate the path,
   check the first time step where early exercise is optimal (S_t <= B[t]
   for puts, S_t >= B[t] for calls), and record the pre-discounted payoff.

4. QAE estimates E[discounted payoff] with quantum speedup over Monte Carlo.

The quantum advantage over a classical MC simulation of American options is
O(1/ε) vs O(1/ε²) sample complexity, where ε is the pricing error.
"""

import numpy as np
from math import exp, sqrt
from qiskit import transpile

from ..qae_engine import build_path_A_operator, build_qae_circuit, extract_option_price, SAFE_BASIS
from ..classical_models import binomial_american, binomial_american_boundary


# ---------------------------------------------------------------------------
# Path payoff (classical simulation given the exercise boundary)
# ---------------------------------------------------------------------------

def _path_payoffs(S, K, T, r, sigma, N_steps, option_type, boundary):
    """
    For every path in {0,1}^N_steps compute the pre-discounted payoff.

    path integer encoding: bit j (LSB) = move at step j (1=up, 0=down).
    """
    dt = T / N_steps
    u  = exp(sigma * sqrt(dt))
    d  = 1.0 / u

    n_paths = 2 ** N_steps
    payoffs = np.zeros(n_paths)

    for path_idx in range(n_paths):
        k = 0  # running up-move count
        exercised = False

        for step in range(N_steps):
            up = (path_idx >> step) & 1   # LSB = step 0
            k += up
            S_t = S * u ** k * d ** (step + 1 - k)
            tau = (step + 1) * dt

            if option_type == "put" and S_t <= boundary[step + 1]:
                payoffs[path_idx] = exp(-r * tau) * max(K - S_t, 0.0)
                exercised = True
                break
            elif option_type == "call" and S_t >= boundary[step + 1]:
                payoffs[path_idx] = exp(-r * tau) * max(S_t - K, 0.0)
                exercised = True
                break

        if not exercised:
            S_T = S * u ** k * d ** (N_steps - k)
            raw = max(K - S_T, 0.0) if option_type == "put" else max(S_T - K, 0.0)
            payoffs[path_idx] = exp(-r * T) * raw

    return payoffs


# ---------------------------------------------------------------------------
# Main pricing function
# ---------------------------------------------------------------------------

def price_american(
    S, K, T, r, sigma, option_type,
    n_steps, n_qpe_qubits,
    backend, shots=100000, top_k=10000,
):
    """
    Price an American option using QPE-based Quantum Amplitude Estimation.

    Parameters
    ----------
    S, K        : spot price, strike price
    T           : time to expiry (years)
    r, sigma    : risk-free rate, volatility
    option_type : 'call' or 'put'
    n_steps     : binomial time steps (= path qubits)
    n_qpe_qubits: QPE precision qubits
    backend     : AutomatskiKomencoQiskit instance

    Returns
    -------
    dict with 'quantum_price', 'classical_price', 'circuit_stats', diagnostics
    """
    dt  = T / n_steps
    u   = exp(sigma * sqrt(dt))
    d   = 1.0 / u
    p_rn = (exp(r * dt) - d) / (u - d)

    # --- Classical pre-processing: compute optimal exercise boundary ---
    classical_price, boundary = binomial_american_boundary(
        S, K, T, r, sigma, option_type, n_steps
    )

    # --- Compute discounted payoff for every path ---
    payoffs = _path_payoffs(S, K, T, r, sigma, n_steps, option_type, boundary)

    # --- Build and run QAE circuit ---
    A_circ, f_max = build_path_A_operator(p_rn, n_steps, payoffs)
    n_state = n_steps + 1

    qae_circ  = build_qae_circuit(A_circ, n_state, n_qpe_qubits)
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
            "p_rn": p_rn,
        },
    }
