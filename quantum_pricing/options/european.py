"""
European option pricing via QPE-based Quantum Amplitude Estimation.

Model
-----
Under the risk-neutral measure, S_T is log-normal:
    log(S_T) ~ N(log(S) + (r - σ²/2)T, σ²T)

The price range is discretized into 2^n_price_qubits bins.
Each bin's probability is computed from the log-normal density.
The QAE circuit encodes E[e^{-rT} · payoff(S_T)].
"""

import numpy as np
from qiskit import transpile

from ..qae_engine import build_A_operator, build_qae_circuit, extract_option_price, SAFE_BASIS
from ..classical_models import black_scholes


def _lognormal_probs_payoffs(S, K, T, r, sigma, option_type, n_price_qubits):
    """Compute discretised log-normal probabilities and discounted payoffs."""
    mu    = np.log(S) + (r - 0.5 * sigma ** 2) * T
    s_dev = sigma * np.sqrt(T)

    log_lo = mu - 4.5 * s_dev
    log_hi = mu + 4.5 * s_dev
    n_bins = 2 ** n_price_qubits

    log_edges = np.linspace(log_lo, log_hi, n_bins + 1)
    log_mids  = 0.5 * (log_edges[:-1] + log_edges[1:])
    d_log     = (log_hi - log_lo) / n_bins
    S_mids    = np.exp(log_mids)

    # Log-normal PDF: pdf(x) = exp(-(ln x - μ)²/(2σ²)) / (x σ √2π)
    probs = (np.exp(-0.5 * ((log_mids - mu) / s_dev) ** 2)
             / (s_dev * np.sqrt(2 * np.pi))) * d_log
    probs = np.clip(probs, 0.0, None)
    probs /= probs.sum()

    disc = np.exp(-r * T)
    if option_type.lower() == "call":
        payoffs = disc * np.maximum(S_mids - K, 0.0)
    else:
        payoffs = disc * np.maximum(K - S_mids, 0.0)

    return probs, payoffs


def price_european(
    S, K, T, r, sigma, option_type,
    n_price_qubits, n_qpe_qubits,
    backend, shots=100000, top_k=10000,
):
    """
    Price a European option using QPE-based Quantum Amplitude Estimation.

    Parameters
    ----------
    S, K        : spot price, strike price
    T           : time to expiry (years)
    r, sigma    : risk-free rate, volatility
    option_type : 'call' or 'put'
    n_price_qubits : price discretisation qubits (controls granularity)
    n_qpe_qubits   : QPE precision qubits (controls accuracy)
    backend     : AutomatskiKomencoQiskit instance
    shots       : number of measurement repetitions
    top_k       : topK parameter for the backend

    Returns
    -------
    dict with 'quantum_price', 'classical_price', 'circuit_stats', and diagnostics
    """
    probs, payoffs = _lognormal_probs_payoffs(S, K, T, r, sigma, option_type, n_price_qubits)

    A_circ, f_max = build_A_operator(probs, payoffs, n_price_qubits)
    n_state = n_price_qubits + 1

    qae_circ = build_qae_circuit(A_circ, n_state, n_qpe_qubits)

    # Transpile to Automatski safe gate set
    transpiled = transpile(qae_circ, basis_gates=SAFE_BASIS, optimization_level=3)

    result  = backend.run(transpiled, repetitions=shots, topK=top_k)
    counts  = result.get_counts(None)

    price_data = extract_option_price(counts, n_qpe_qubits, f_max)

    classical = black_scholes(S, K, T, r, sigma, option_type.lower())

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
            "n_price_qubits": n_price_qubits,
            "n_qpe_qubits":   n_qpe_qubits,
        },
    }
