"""
Classical option pricing models used as reference values alongside quantum results.
No external dependencies beyond numpy; normal CDF implemented via math.erf.
"""

import numpy as np
from math import erf, sqrt, exp, log, pi


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


# ---------------------------------------------------------------------------
# European — Black-Scholes analytical
# ---------------------------------------------------------------------------

def black_scholes(S: float, K: float, T: float, r: float, sigma: float,
                  option_type: str) -> float:
    """Analytical Black-Scholes price for a European option."""
    if T <= 0.0:
        payoff = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        return payoff

    sqrtT = sqrt(T)
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    disc = exp(-r * T)

    if option_type == "call":
        return S * _norm_cdf(d1) - K * disc * _norm_cdf(d2)
    else:
        return K * disc * _norm_cdf(-d2) - S * _norm_cdf(-d1)


# ---------------------------------------------------------------------------
# American — binomial tree
# ---------------------------------------------------------------------------

def binomial_american(S: float, K: float, T: float, r: float, sigma: float,
                       option_type: str, N: int = 300) -> float:
    """American option price via binomial backward induction (N steps)."""
    dt = T / N
    u  = exp(sigma * sqrt(dt))
    d  = 1.0 / u
    p  = (exp(r * dt) - d) / (u - d)
    disc = exp(-r * dt)

    # Terminal payoffs (j = number of down-moves)
    S_T = np.array([S * u ** (N - j) * d ** j for j in range(N + 1)])
    V = np.maximum(S_T - K, 0) if option_type == "call" else np.maximum(K - S_T, 0)

    for t in range(N - 1, -1, -1):
        V_hold = disc * (p * V[:-1] + (1 - p) * V[1:])
        S_t    = np.array([S * u ** (t - j) * d ** j for j in range(t + 1)])
        V_ex   = (np.maximum(S_t - K, 0) if option_type == "call"
                  else np.maximum(K - S_t, 0))
        V = np.maximum(V_hold, V_ex)

    return float(V[0])


def binomial_american_boundary(S: float, K: float, T: float, r: float,
                                sigma: float, option_type: str, N: int):
    """
    Compute American option price AND optimal exercise boundary via backward induction.

    Returns (price, boundary) where boundary[t] is the critical stock price at step t:
      put : exercise if S_t <= boundary[t]
      call: exercise if S_t >= boundary[t]
    """
    dt = T / N
    u  = exp(sigma * sqrt(dt))
    d  = 1.0 / u
    p  = (exp(r * dt) - d) / (u - d)
    disc = exp(-r * dt)

    # boundary indexed 0..N (step), default = no exercise
    boundary = np.full(N + 1, 0.0 if option_type == "put" else np.inf)

    # Terminal layer (t = N)
    S_N = np.array([S * u ** k * d ** (N - k) for k in range(N + 1)])
    V   = (np.maximum(S_N - K, 0) if option_type == "call"
           else np.maximum(K - S_N, 0))

    if option_type == "put":
        ex_N = np.where(S_N <= K)[0]
        boundary[N] = float(S_N[ex_N[-1]]) if len(ex_N) else 0.0
    else:
        ex_N = np.where(S_N >= K)[0]
        boundary[N] = float(S_N[ex_N[0]]) if len(ex_N) else np.inf

    for t in range(N - 1, -1, -1):
        V_hold = disc * (p * V[:-1] + (1 - p) * V[1:])
        S_t    = np.array([S * u ** k * d ** (t - k) for k in range(t + 1)])
        V_ex   = (np.maximum(S_t - K, 0) if option_type == "call"
                  else np.maximum(K - S_t, 0))
        V_new  = np.maximum(V_hold, V_ex)

        # Record boundary: nodes where early exercise is optimal
        ex_mask = (V_new > V_hold + 1e-10) & (V_ex > 0)
        if np.any(ex_mask):
            if option_type == "put":
                boundary[t] = float(S_t[ex_mask].max())
            else:
                boundary[t] = float(S_t[ex_mask].min())
        else:
            boundary[t] = 0.0 if option_type == "put" else np.inf

        V = V_new

    return float(V[0]), boundary


# ---------------------------------------------------------------------------
# Bermudan — binomial tree with restricted exercise dates
# ---------------------------------------------------------------------------

def binomial_bermudan(S: float, K: float, T: float, r: float, sigma: float,
                       option_type: str, exercise_steps: list, N: int = 300) -> float:
    """Bermudan option via binomial backward induction."""
    dt = T / N
    u  = exp(sigma * sqrt(dt))
    d  = 1.0 / u
    p  = (exp(r * dt) - d) / (u - d)
    disc = exp(-r * dt)
    exercise_set = set(exercise_steps)

    S_T = np.array([S * u ** (N - j) * d ** j for j in range(N + 1)])
    V   = (np.maximum(S_T - K, 0) if option_type == "call"
           else np.maximum(K - S_T, 0))

    for t in range(N - 1, -1, -1):
        V_hold = disc * (p * V[:-1] + (1 - p) * V[1:])
        if t in exercise_set:
            S_t  = np.array([S * u ** (t - j) * d ** j for j in range(t + 1)])
            V_ex = (np.maximum(S_t - K, 0) if option_type == "call"
                    else np.maximum(K - S_t, 0))
            V = np.maximum(V_hold, V_ex)
        else:
            V = V_hold

    return float(V[0])


def binomial_bermudan_boundary(S, K, T, r, sigma, option_type, exercise_steps, N):
    """Bermudan boundary for quantum path payoff computation."""
    dt = T / N
    u  = exp(sigma * sqrt(dt))
    d  = 1.0 / u
    p  = (exp(r * dt) - d) / (u - d)
    disc = exp(-r * dt)
    exercise_set = set(exercise_steps)

    boundary = np.full(N + 1, 0.0 if option_type == "put" else np.inf)

    S_N = np.array([S * u ** k * d ** (N - k) for k in range(N + 1)])
    V   = (np.maximum(S_N - K, 0) if option_type == "call"
           else np.maximum(K - S_N, 0))

    if N in exercise_set:
        if option_type == "put":
            ex = np.where(S_N <= K)[0]
            boundary[N] = float(S_N[ex[-1]]) if len(ex) else 0.0
        else:
            ex = np.where(S_N >= K)[0]
            boundary[N] = float(S_N[ex[0]]) if len(ex) else np.inf

    for t in range(N - 1, -1, -1):
        V_hold = disc * (p * V[:-1] + (1 - p) * V[1:])
        if t in exercise_set:
            S_t  = np.array([S * u ** k * d ** (t - k) for k in range(t + 1)])
            V_ex = (np.maximum(S_t - K, 0) if option_type == "call"
                    else np.maximum(K - S_t, 0))
            V_new = np.maximum(V_hold, V_ex)
            ex_mask = (V_new > V_hold + 1e-10) & (V_ex > 0)
            if np.any(ex_mask):
                if option_type == "put":
                    boundary[t] = float(S_t[ex_mask].max())
                else:
                    boundary[t] = float(S_t[ex_mask].min())
            V = V_new
        else:
            V = V_hold

    return float(V[0]), boundary


# ---------------------------------------------------------------------------
# Asian — Monte Carlo reference (geometric Brownian motion, arithmetic average)
# ---------------------------------------------------------------------------

def mc_asian(S: float, K: float, T: float, r: float, sigma: float,
             option_type: str, N_steps: int = 252, n_sims: int = 50000,
             seed: int = 42) -> float:
    """Arithmetic Asian option price via Monte Carlo."""
    rng = np.random.default_rng(seed)
    dt  = T / N_steps
    drift = (r - 0.5 * sigma ** 2) * dt
    vol   = sigma * sqrt(dt)

    Z       = rng.standard_normal((n_sims, N_steps))
    log_ret = drift + vol * Z
    # Cumulative log-returns → price paths (n_sims × N_steps)
    log_paths = np.cumsum(log_ret, axis=1)
    paths = S * np.exp(log_paths)

    avg = paths.mean(axis=1)
    payoff = (np.maximum(avg - K, 0) if option_type == "call"
              else np.maximum(K - avg, 0))
    return float(exp(-r * T) * payoff.mean())


# ---------------------------------------------------------------------------
# Barrier — Monte Carlo reference
# ---------------------------------------------------------------------------

def mc_barrier(S: float, K: float, T: float, r: float, sigma: float,
               option_type: str, barrier: float, barrier_type: str,
               N_steps: int = 252, n_sims: int = 50000, seed: int = 42) -> float:
    """Barrier option price via Monte Carlo."""
    rng = np.random.default_rng(seed)
    dt    = T / N_steps
    drift = (r - 0.5 * sigma ** 2) * dt
    vol   = sigma * sqrt(dt)

    Z         = rng.standard_normal((n_sims, N_steps))
    log_ret   = drift + vol * Z
    log_paths = np.cumsum(log_ret, axis=1)
    paths     = S * np.exp(log_paths)   # shape (n_sims, N_steps)

    S_T = paths[:, -1]

    is_up  = "up"  in barrier_type.lower()
    is_out = "out" in barrier_type.lower()

    if is_up:
        hit = (paths >= barrier).any(axis=1)
    else:
        hit = (paths <= barrier).any(axis=1)

    if is_out:
        active = ~hit
    else:
        active = hit

    payoff_raw = (np.maximum(S_T - K, 0) if option_type == "call"
                  else np.maximum(K - S_T, 0))
    payoff = np.where(active, payoff_raw, 0.0)
    return float(exp(-r * T) * payoff.mean())


# ---------------------------------------------------------------------------
# Binary — analytical (cash-or-nothing)
# ---------------------------------------------------------------------------

def binary_cash_or_nothing(S: float, K: float, T: float, r: float,
                            sigma: float, option_type: str,
                            cash_amount: float = 1.0) -> float:
    """Cash-or-nothing binary option analytical price."""
    if T <= 0.0:
        if option_type == "call":
            return cash_amount if S > K else 0.0
        else:
            return cash_amount if S < K else 0.0

    sqrtT = sqrt(T)
    d2    = (log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    disc  = exp(-r * T)

    if option_type == "call":
        return cash_amount * disc * _norm_cdf(d2)
    else:
        return cash_amount * disc * _norm_cdf(-d2)
