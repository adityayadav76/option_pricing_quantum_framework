"""
Binary (digital) option pricing via QPE-based Quantum Amplitude Estimation.

Payoff types
------------
  cash-or-nothing call : pays Q if S_T > K, else 0
  cash-or-nothing put  : pays Q if S_T < K, else 0
  asset-or-nothing call: pays S_T if S_T > K, else 0
  asset-or-nothing put : pays S_T if S_T < K, else 0

The price range is discretised into 2^n_price_qubits bins using the same
log-normal model as the European option.  The payoff is binary (fixed or
asset-value), so the oracle is simpler than for vanilla options.
"""

import numpy as np
from qiskit import transpile

from ..qae_engine import build_A_operator, build_qae_circuit, extract_option_price, SAFE_BASIS
from ..classical_models import binary_cash_or_nothing, black_scholes


PAYOFF_TYPES = ("cash-or-nothing", "asset-or-nothing")


def _binary_probs_payoffs(S, K, T, r, sigma, option_type, payoff_type,
                           cash_amount, n_price_qubits):
    """Compute discretised probabilities and discounted payoffs for the binary option."""
    mu    = np.log(S) + (r - 0.5 * sigma ** 2) * T
    s_dev = sigma * np.sqrt(T)

    log_lo = mu - 4.5 * s_dev
    log_hi = mu + 4.5 * s_dev
    n_bins = 2 ** n_price_qubits

    log_edges = np.linspace(log_lo, log_hi, n_bins + 1)
    log_mids  = 0.5 * (log_edges[:-1] + log_edges[1:])
    d_log     = (log_hi - log_lo) / n_bins
    S_mids    = np.exp(log_mids)

    probs = (np.exp(-0.5 * ((log_mids - mu) / s_dev) ** 2)
             / (s_dev * np.sqrt(2 * np.pi))) * d_log
    probs = np.clip(probs, 0.0, None)
    probs /= probs.sum()

    disc = np.exp(-r * T)

    if payoff_type == "cash-or-nothing":
        if option_type.lower() == "call":
            payoffs = disc * cash_amount * (S_mids > K).astype(float)
        else:
            payoffs = disc * cash_amount * (S_mids < K).astype(float)
    else:  # asset-or-nothing
        if option_type.lower() == "call":
            payoffs = disc * S_mids * (S_mids > K).astype(float)
        else:
            payoffs = disc * S_mids * (S_mids < K).astype(float)

    return probs, payoffs


def price_binary(
    S, K, T, r, sigma, option_type,
    payoff_type, cash_amount,
    n_price_qubits, n_qpe_qubits,
    backend, shots=100000, top_k=10000,
):
    """
    Price a binary (digital) option using QPE-based Quantum Amplitude Estimation.

    Parameters
    ----------
    option_type  : 'call' or 'put'
    payoff_type  : 'cash-or-nothing' or 'asset-or-nothing'
    cash_amount  : fixed payment for cash-or-nothing (ignored for asset-or-nothing)
    n_price_qubits : price discretisation qubits
    n_qpe_qubits   : QPE precision qubits
    backend      : AutomatskiKomencoQiskit instance
    """
    if payoff_type.lower() not in PAYOFF_TYPES:
        raise ValueError(f"payoff_type must be one of {PAYOFF_TYPES}")

    probs, payoffs = _binary_probs_payoffs(
        S, K, T, r, sigma, option_type, payoff_type.lower(),
        cash_amount, n_price_qubits
    )

    A_circ, f_max = build_A_operator(probs, payoffs, n_price_qubits)
    n_state = n_price_qubits + 1

    qae_circ   = build_qae_circuit(A_circ, n_state, n_qpe_qubits)
    transpiled = transpile(qae_circ, basis_gates=SAFE_BASIS, optimization_level=3)

    result = backend.run(transpiled, repetitions=shots, topK=top_k)
    counts = result.get_counts(None)

    price_data = extract_option_price(counts, n_qpe_qubits, f_max)

    # Classical reference
    if payoff_type.lower() == "cash-or-nothing":
        classical = binary_cash_or_nothing(S, K, T, r, sigma, option_type.lower(), cash_amount)
    else:
        # Asset-or-nothing = European + cash-or-nothing (put-call parity variant)
        # Analytical: call = S*N(d1), put = S*N(-d1)  (without discount, for asset-or-nothing)
        from math import log, sqrt as msqrt, exp as mexp, erf
        d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * msqrt(T))
        n_d1 = 0.5 * (1 + erf(d1 / msqrt(2)))
        disc = mexp(-r * T)
        classical = (disc * S * n_d1) if option_type.lower() == "call" else (disc * S * (1 - n_d1))

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
            "payoff_type": payoff_type,
            "cash_amount": cash_amount,
            "n_price_qubits": n_price_qubits,
            "n_qpe_qubits":   n_qpe_qubits,
        },
    }
