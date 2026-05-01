"""
Quantum Options Pricing Framework — Gradio UI

Supports pricing of six option types and a portfolio using
Quantum Phase Estimation / Quantum Amplitude Estimation on the
Automatski Quantum Backend.
"""

import sys
import json
import traceback

import gradio as gr
import numpy as np

# ------------------------------------------------------------------
# Automatski backend
# ------------------------------------------------------------------
sys.path.append(".")
try:
    from AutomatskiKomencoQiskit import AutomatskiKomencoQiskit
    _backend_available = True
except ImportError:
    _backend_available = False

# ------------------------------------------------------------------
# Option pricers
# ------------------------------------------------------------------
from quantum_pricing.options.european import price_european
from quantum_pricing.options.american import price_american
from quantum_pricing.options.bermudan import price_bermudan
from quantum_pricing.options.asian    import price_asian
from quantum_pricing.options.barrier  import price_barrier
from quantum_pricing.options.binary   import price_binary
from quantum_pricing.portfolio        import price_portfolio, OptionLeg


# ------------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------------

def _make_backend(host: str, port: int):
    if not _backend_available:
        raise RuntimeError("AutomatskiKomencoQiskit not found. "
                           "Ensure AutomatskiKomencoQiskit.py is in the project root.")
    return AutomatskiKomencoQiskit(host=host.strip(), port=int(port))


def _fmt(result: dict) -> str:
    """Format result dict into a readable summary string."""
    lines = []
    lines.append("=" * 58)
    lines.append("  QUANTUM PRICING RESULT")
    lines.append("=" * 58)
    lines.append(f"  Quantum price   : {result.get('quantum_price', 'N/A'):>12.6f}")
    lines.append(f"  Classical ref.  : {result.get('classical_price', 'N/A'):>12.6f}")

    amp  = result.get("amplitude", None)
    fmax = result.get("f_max", None)
    conf = result.get("confidence", None)
    if amp  is not None: lines.append(f"  Amplitude (a)   : {amp:>12.6f}")
    if fmax is not None: lines.append(f"  Payoff max (F)  : {fmax:>12.6f}")
    if conf is not None: lines.append(f"  QPE confidence  : {conf:>12.3%}")

    cs = result.get("circuit_stats", {})
    if cs:
        lines.append("-" * 58)
        lines.append("  CIRCUIT STATISTICS  (after transpilation)")
        lines.append(f"    Qubits  : {cs.get('n_qubits','?')}")
        lines.append(f"    Gates   : {cs.get('n_gates','?'):,}")
        lines.append(f"    Depth   : {cs.get('depth','?')}")

    params = result.get("parameters", {})
    if params:
        lines.append("-" * 58)
        lines.append("  PARAMETERS")
        for k, v in params.items():
            lines.append(f"    {k:<22}: {v}")

    lines.append("=" * 58)
    return "\n".join(lines)


def _portfolio_fmt(result: dict) -> str:
    lines = []
    lines.append("=" * 68)
    lines.append("  PORTFOLIO PRICING RESULT")
    lines.append("=" * 68)
    lines.append(f"  Portfolio value (quantum)   : {result['portfolio_value_quantum']:>14.6f}")
    lines.append(f"  Portfolio value (classical) : {result['portfolio_value_classical']:>14.6f}")
    lines.append(f"  Number of legs              : {result['n_legs']}")
    lines.append("-" * 68)
    for i, leg in enumerate(result["legs"]):
        qty  = leg.get("quantity", 1.0)
        sign = "LONG" if qty > 0 else "SHORT"
        lines.append(f"  Leg {i+1}: {leg.get('label','?')}  [{sign} {abs(qty):.1f}]")
        lines.append(f"    Quantum price  : {leg.get('quantum_price','?'):>12.6f}")
        lines.append(f"    Classical price: {leg.get('classical_price','?'):>12.6f}")
        lines.append(f"    Leg value Q    : {leg.get('leg_value_quantum','?'):>12.6f}")
        cs = leg.get("circuit_stats", {})
        if cs:
            lines.append(f"    Qubits / Gates : {cs.get('n_qubits','?')} / {cs.get('n_gates','?'):,}")
    lines.append("=" * 68)
    return "\n".join(lines)


# ------------------------------------------------------------------
# Per-tab pricing handlers
# ------------------------------------------------------------------

def run_european(host, port, S, K, T, r, sigma, opt_type, n_price, n_qpe, shots, top_k):
    try:
        backend = _make_backend(host, port)
        result  = price_european(S, K, T, r, sigma, opt_type,
                                 int(n_price), int(n_qpe),
                                 backend, int(shots), int(top_k))
        return _fmt(result)
    except Exception:
        return f"ERROR\n{traceback.format_exc()}"


def run_american(host, port, S, K, T, r, sigma, opt_type, n_steps, n_qpe, shots, top_k):
    try:
        backend = _make_backend(host, port)
        result  = price_american(S, K, T, r, sigma, opt_type,
                                 int(n_steps), int(n_qpe),
                                 backend, int(shots), int(top_k))
        return _fmt(result)
    except Exception:
        return f"ERROR\n{traceback.format_exc()}"


def run_bermudan(host, port, S, K, T, r, sigma, opt_type,
                 n_steps, exercise_steps_str, n_qpe, shots, top_k):
    try:
        backend = _make_backend(host, port)
        exercise_steps = [int(x.strip()) for x in exercise_steps_str.split(",") if x.strip()]
        result  = price_bermudan(S, K, T, r, sigma, opt_type,
                                 int(n_steps), exercise_steps, int(n_qpe),
                                 backend, int(shots), int(top_k))
        return _fmt(result)
    except Exception:
        return f"ERROR\n{traceback.format_exc()}"


def run_asian(host, port, S, K, T, r, sigma, opt_type, n_steps, n_qpe, shots, top_k):
    try:
        backend = _make_backend(host, port)
        result  = price_asian(S, K, T, r, sigma, opt_type,
                               int(n_steps), int(n_qpe),
                               backend, int(shots), int(top_k))
        return _fmt(result)
    except Exception:
        return f"ERROR\n{traceback.format_exc()}"


def run_barrier(host, port, S, K, T, r, sigma, opt_type,
                barrier, barrier_type, n_steps, n_qpe, shots, top_k):
    try:
        backend = _make_backend(host, port)
        result  = price_barrier(S, K, T, r, sigma, opt_type,
                                 float(barrier), barrier_type,
                                 int(n_steps), int(n_qpe),
                                 backend, int(shots), int(top_k))
        return _fmt(result)
    except Exception:
        return f"ERROR\n{traceback.format_exc()}"


def run_binary(host, port, S, K, T, r, sigma, opt_type,
               payoff_type, cash_amount, n_price, n_qpe, shots, top_k):
    try:
        backend = _make_backend(host, port)
        result  = price_binary(S, K, T, r, sigma, opt_type,
                                payoff_type, float(cash_amount),
                                int(n_price), int(n_qpe),
                                backend, int(shots), int(top_k))
        return _fmt(result)
    except Exception:
        return f"ERROR\n{traceback.format_exc()}"


# ------------------------------------------------------------------
# Portfolio state (stored in a gr.State list of leg dicts)
# ------------------------------------------------------------------

def add_portfolio_leg(legs_state,
                      leg_type, leg_label, leg_quantity,
                      S, K, T, r, sigma, opt_type,
                      # per-type extras
                      n_price, n_steps, n_qpe,
                      exercise_steps_str,
                      barrier, barrier_type,
                      payoff_type, cash_amount):
    """Append a new leg to the portfolio state."""
    legs = legs_state or []

    base_params = dict(S=S, K=K, T=T, r=r, sigma=sigma,
                       option_type=opt_type,
                       n_qpe_qubits=int(n_qpe))

    leg_type_l = leg_type.lower()

    if leg_type_l == "european":
        params = {**base_params, "n_price_qubits": int(n_price)}
    elif leg_type_l == "american":
        params = {**base_params, "n_steps": int(n_steps)}
    elif leg_type_l == "bermudan":
        ex_steps = [int(x.strip()) for x in exercise_steps_str.split(",") if x.strip()]
        params = {**base_params, "n_steps": int(n_steps), "exercise_steps": ex_steps}
    elif leg_type_l == "asian":
        params = {**base_params, "n_steps": int(n_steps)}
    elif leg_type_l == "barrier":
        params = {**base_params, "barrier": float(barrier),
                  "barrier_type": barrier_type, "n_steps": int(n_steps)}
    elif leg_type_l == "binary":
        params = {**base_params, "n_price_qubits": int(n_price),
                  "payoff_type": payoff_type, "cash_amount": float(cash_amount)}
    else:
        return legs, _render_legs(legs)

    legs.append({
        "option_type_class": leg_type_l,
        "params": params,
        "quantity": float(leg_quantity),
        "label": leg_label or leg_type,
    })
    return legs, _render_legs(legs)


def clear_portfolio(legs_state):
    return [], "Portfolio cleared."


def _render_legs(legs):
    if not legs:
        return "Portfolio is empty."
    lines = [f"Portfolio: {len(legs)} leg(s)"]
    for i, leg in enumerate(legs):
        p = leg["params"]
        lines.append(
            f"  {i+1}. [{leg['option_type_class'].upper()}] "
            f"{leg['label']}  qty={leg['quantity']:+.1f}  "
            f"S={p.get('S')} K={p.get('K')} T={p.get('T')} "
            f"σ={p.get('sigma')} r={p.get('r')}"
        )
    return "\n".join(lines)


def run_portfolio(legs_state, host, port, shots, top_k):
    try:
        legs = legs_state or []
        if not legs:
            return "Portfolio is empty. Add legs first."
        backend = _make_backend(host, port)
        option_legs = [OptionLeg(**leg) for leg in legs]
        result = price_portfolio(option_legs, backend, int(shots), int(top_k))
        return _portfolio_fmt(result)
    except Exception:
        return f"ERROR\n{traceback.format_exc()}"


# ------------------------------------------------------------------
# Build UI
# ------------------------------------------------------------------

CSS = """
.result-box textarea { font-family: monospace; font-size: 0.82rem; }
"""

with gr.Blocks(title="Quantum Options Pricing") as demo:

    gr.Markdown(
        """
# ⚛ Quantum Options Pricing Framework
**Automatski Quantum Backend · Quantum Phase Estimation / Amplitude Estimation**

Price six types of options and full portfolios using pure QPE-based Quantum Amplitude
Estimation — no classical shortcuts or Monte Carlo approximations in the quantum circuit.
        """
    )

    # ---- Backend Configuration ----------------------------------------
    with gr.Accordion("🔧 Backend Configuration", open=True):
        with gr.Row():
            backend_host = gr.Textbox(value="localhost", label="Backend Host", scale=3)
            backend_port = gr.Number(value=8080, label="Port", scale=1, precision=0)
        with gr.Row():
            shots_global = gr.Number(value=100000, label="Shots (repetitions)", precision=0)
            topk_global  = gr.Number(value=10000,  label="topK", precision=0)

    gr.Markdown("---")

    # ---- Option Tabs --------------------------------------------------
    with gr.Tabs():

        # ============================
        # EUROPEAN
        # ============================
        with gr.Tab("🇪🇺 European"):
            gr.Markdown(
                "**European option** — may only be exercised at expiry. "
                "Priced by QAE over a discretised log-normal distribution."
            )
            with gr.Row():
                with gr.Column():
                    eu_S     = gr.Number(value=100.0,  label="Spot price S")
                    eu_K     = gr.Number(value=100.0,  label="Strike K")
                    eu_T     = gr.Number(value=1.0,    label="Expiry T (years)")
                    eu_r     = gr.Number(value=0.05,   label="Risk-free rate r")
                    eu_sigma = gr.Number(value=0.2,    label="Volatility σ")
                    eu_type  = gr.Radio(["call", "put"], value="call", label="Option type")
                with gr.Column():
                    eu_n_price = gr.Slider(1, 8, value=5, step=1,
                                           label="Price qubits (discretisation)")
                    eu_n_qpe   = gr.Slider(3, 10, value=7, step=1,
                                           label="QPE precision qubits")
                    eu_btn     = gr.Button("▶ Price European Option", variant="primary")
                    eu_out     = gr.Textbox(label="Result", lines=22,
                                           elem_classes="result-box")

            eu_btn.click(
                run_european,
                inputs=[backend_host, backend_port,
                        eu_S, eu_K, eu_T, eu_r, eu_sigma, eu_type,
                        eu_n_price, eu_n_qpe, shots_global, topk_global],
                outputs=eu_out,
            )

        # ============================
        # AMERICAN
        # ============================
        with gr.Tab("🇺🇸 American"):
            gr.Markdown(
                "**American option** — may be exercised on any trading day up to expiry. "
                "Optimal exercise boundary computed classically; QAE prices the path integral."
            )
            with gr.Row():
                with gr.Column():
                    am_S     = gr.Number(value=100.0, label="Spot price S")
                    am_K     = gr.Number(value=100.0, label="Strike K")
                    am_T     = gr.Number(value=1.0,   label="Expiry T (years)")
                    am_r     = gr.Number(value=0.05,  label="Risk-free rate r")
                    am_sigma = gr.Number(value=0.2,   label="Volatility σ")
                    am_type  = gr.Radio(["call", "put"], value="put", label="Option type")
                with gr.Column():
                    am_n_steps = gr.Slider(2, 20, value=8, step=1,
                                           label="Time steps N (= path qubits)")
                    am_n_qpe   = gr.Slider(3, 10, value=7, step=1,
                                           label="QPE precision qubits")
                    am_btn     = gr.Button("▶ Price American Option", variant="primary")
                    am_out     = gr.Textbox(label="Result", lines=22,
                                           elem_classes="result-box")

            am_btn.click(
                run_american,
                inputs=[backend_host, backend_port,
                        am_S, am_K, am_T, am_r, am_sigma, am_type,
                        am_n_steps, am_n_qpe, shots_global, topk_global],
                outputs=am_out,
            )

        # ============================
        # BERMUDAN
        # ============================
        with gr.Tab("🏝 Bermudan"):
            gr.Markdown(
                "**Bermudan option** — exercisable only on specified dates. "
                "Enter exercise step indices (1-based, comma-separated)."
            )
            with gr.Row():
                with gr.Column():
                    bm_S     = gr.Number(value=100.0, label="Spot price S")
                    bm_K     = gr.Number(value=100.0, label="Strike K")
                    bm_T     = gr.Number(value=1.0,   label="Expiry T (years)")
                    bm_r     = gr.Number(value=0.05,  label="Risk-free rate r")
                    bm_sigma = gr.Number(value=0.2,   label="Volatility σ")
                    bm_type  = gr.Radio(["call", "put"], value="put", label="Option type")
                with gr.Column():
                    bm_n_steps = gr.Slider(2, 20, value=12, step=1,
                                           label="Total time steps N")
                    bm_ex_str  = gr.Textbox(value="3,6,9,12",
                                            label="Exercise step indices (comma-separated)")
                    bm_n_qpe   = gr.Slider(3, 10, value=7, step=1,
                                           label="QPE precision qubits")
                    bm_btn     = gr.Button("▶ Price Bermudan Option", variant="primary")
                    bm_out     = gr.Textbox(label="Result", lines=22,
                                           elem_classes="result-box")

            bm_btn.click(
                run_bermudan,
                inputs=[backend_host, backend_port,
                        bm_S, bm_K, bm_T, bm_r, bm_sigma, bm_type,
                        bm_n_steps, bm_ex_str, bm_n_qpe, shots_global, topk_global],
                outputs=bm_out,
            )

        # ============================
        # ASIAN
        # ============================
        with gr.Tab("🌏 Asian"):
            gr.Markdown(
                "**Asian option** — payoff based on arithmetic average price over the period. "
                "QAE encodes the full path distribution via independent binomial moves."
            )
            with gr.Row():
                with gr.Column():
                    as_S     = gr.Number(value=100.0, label="Spot price S")
                    as_K     = gr.Number(value=100.0, label="Strike K")
                    as_T     = gr.Number(value=1.0,   label="Expiry T (years)")
                    as_r     = gr.Number(value=0.05,  label="Risk-free rate r")
                    as_sigma = gr.Number(value=0.2,   label="Volatility σ")
                    as_type  = gr.Radio(["call", "put"], value="call", label="Option type")
                with gr.Column():
                    as_n_steps = gr.Slider(2, 20, value=8, step=1,
                                           label="Averaging steps N (= path qubits)")
                    as_n_qpe   = gr.Slider(3, 10, value=7, step=1,
                                           label="QPE precision qubits")
                    as_btn     = gr.Button("▶ Price Asian Option", variant="primary")
                    as_out     = gr.Textbox(label="Result", lines=22,
                                           elem_classes="result-box")

            as_btn.click(
                run_asian,
                inputs=[backend_host, backend_port,
                        as_S, as_K, as_T, as_r, as_sigma, as_type,
                        as_n_steps, as_n_qpe, shots_global, topk_global],
                outputs=as_out,
            )

        # ============================
        # BARRIER
        # ============================
        with gr.Tab("🚧 Barrier"):
            gr.Markdown(
                "**Barrier option** — payoff contingent on price crossing (or not crossing) "
                "a barrier level during the option's life."
            )
            with gr.Row():
                with gr.Column():
                    ba_S     = gr.Number(value=100.0, label="Spot price S")
                    ba_K     = gr.Number(value=100.0, label="Strike K")
                    ba_T     = gr.Number(value=1.0,   label="Expiry T (years)")
                    ba_r     = gr.Number(value=0.05,  label="Risk-free rate r")
                    ba_sigma = gr.Number(value=0.2,   label="Volatility σ")
                    ba_type  = gr.Radio(["call", "put"], value="call", label="Option type")
                with gr.Column():
                    ba_barrier = gr.Number(value=120.0, label="Barrier level H")
                    ba_btype   = gr.Radio(
                        ["up-out", "down-out", "up-in", "down-in"],
                        value="up-out", label="Barrier type"
                    )
                    ba_n_steps = gr.Slider(2, 20, value=8, step=1,
                                           label="Monitoring steps N (= path qubits)")
                    ba_n_qpe   = gr.Slider(3, 10, value=7, step=1,
                                           label="QPE precision qubits")
                    ba_btn     = gr.Button("▶ Price Barrier Option", variant="primary")
                    ba_out     = gr.Textbox(label="Result", lines=22,
                                           elem_classes="result-box")

            ba_btn.click(
                run_barrier,
                inputs=[backend_host, backend_port,
                        ba_S, ba_K, ba_T, ba_r, ba_sigma, ba_type,
                        ba_barrier, ba_btype,
                        ba_n_steps, ba_n_qpe, shots_global, topk_global],
                outputs=ba_out,
            )

        # ============================
        # BINARY
        # ============================
        with gr.Tab("💰 Binary"):
            gr.Markdown(
                "**Binary (digital) option** — pays a fixed amount (cash-or-nothing) "
                "or the asset value (asset-or-nothing) if the condition is met at expiry."
            )
            with gr.Row():
                with gr.Column():
                    bi_S     = gr.Number(value=100.0, label="Spot price S")
                    bi_K     = gr.Number(value=100.0, label="Strike K")
                    bi_T     = gr.Number(value=1.0,   label="Expiry T (years)")
                    bi_r     = gr.Number(value=0.05,  label="Risk-free rate r")
                    bi_sigma = gr.Number(value=0.2,   label="Volatility σ")
                    bi_type  = gr.Radio(["call", "put"], value="call", label="Option type")
                with gr.Column():
                    bi_ptype  = gr.Radio(["cash-or-nothing", "asset-or-nothing"],
                                         value="cash-or-nothing", label="Payoff type")
                    bi_cash   = gr.Number(value=1.0, label="Cash amount Q (cash-or-nothing)")
                    bi_n_price = gr.Slider(1, 8, value=5, step=1,
                                           label="Price qubits (discretisation)")
                    bi_n_qpe   = gr.Slider(3, 10, value=7, step=1,
                                           label="QPE precision qubits")
                    bi_btn     = gr.Button("▶ Price Binary Option", variant="primary")
                    bi_out     = gr.Textbox(label="Result", lines=22,
                                           elem_classes="result-box")

            bi_btn.click(
                run_binary,
                inputs=[backend_host, backend_port,
                        bi_S, bi_K, bi_T, bi_r, bi_sigma, bi_type,
                        bi_ptype, bi_cash,
                        bi_n_price, bi_n_qpe, shots_global, topk_global],
                outputs=bi_out,
            )

        # ============================
        # PORTFOLIO
        # ============================
        with gr.Tab("📦 Portfolio"):
            gr.Markdown(
                "**Portfolio pricing** — add option legs of any type and "
                "price the combined portfolio. Each leg is priced via its own QAE circuit. "
                "Use positive quantity for long positions, negative for short."
            )

            portfolio_state = gr.State([])

            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("#### Add a Leg")
                    with gr.Row():
                        pf_type  = gr.Dropdown(
                            ["European", "American", "Bermudan",
                             "Asian", "Barrier", "Binary"],
                            value="European", label="Option type")
                        pf_label = gr.Textbox(value="", label="Label (optional)")
                        pf_qty   = gr.Number(value=1.0, label="Quantity (± for long/short)")

                    with gr.Row():
                        pf_S     = gr.Number(value=100.0, label="S")
                        pf_K     = gr.Number(value=100.0, label="K")
                        pf_T     = gr.Number(value=1.0,   label="T")
                        pf_r     = gr.Number(value=0.05,  label="r")
                        pf_sigma = gr.Number(value=0.2,   label="σ")
                        pf_otype = gr.Radio(["call", "put"], value="call", label="call/put")

                    with gr.Row():
                        pf_n_price  = gr.Slider(1, 8, value=5, step=1,  label="Price qubits")
                        pf_n_steps  = gr.Slider(2, 20, value=8, step=1, label="Path steps")
                        pf_n_qpe    = gr.Slider(3, 10, value=7, step=1, label="QPE qubits")

                    with gr.Row():
                        pf_ex_str   = gr.Textbox(value="4,8,12", label="Bermudan exercise steps")
                        pf_barrier  = gr.Number(value=120.0,      label="Barrier level")
                        pf_btype    = gr.Dropdown(
                            ["up-out", "down-out", "up-in", "down-in"],
                            value="up-out", label="Barrier type")

                    with gr.Row():
                        pf_paytype  = gr.Dropdown(
                            ["cash-or-nothing", "asset-or-nothing"],
                            value="cash-or-nothing", label="Binary payoff type")
                        pf_cash     = gr.Number(value=1.0, label="Cash amount")

                    with gr.Row():
                        pf_add_btn   = gr.Button("➕ Add Leg",   variant="primary")
                        pf_clear_btn = gr.Button("🗑 Clear All", variant="secondary")

                    pf_leg_display = gr.Textbox(label="Current Portfolio", lines=6,
                                                interactive=False)

                with gr.Column(scale=1):
                    gr.Markdown("#### Run Portfolio Pricing")
                    pf_run_btn = gr.Button("▶ Price Portfolio", variant="primary", size="lg")
                    pf_out     = gr.Textbox(label="Portfolio Result", lines=30,
                                            elem_classes="result-box")

            pf_add_btn.click(
                add_portfolio_leg,
                inputs=[portfolio_state,
                        pf_type, pf_label, pf_qty,
                        pf_S, pf_K, pf_T, pf_r, pf_sigma, pf_otype,
                        pf_n_price, pf_n_steps, pf_n_qpe,
                        pf_ex_str, pf_barrier, pf_btype,
                        pf_paytype, pf_cash],
                outputs=[portfolio_state, pf_leg_display],
            )

            pf_clear_btn.click(
                clear_portfolio,
                inputs=[portfolio_state],
                outputs=[portfolio_state, pf_leg_display],
            )

            pf_run_btn.click(
                run_portfolio,
                inputs=[portfolio_state, backend_host, backend_port,
                        shots_global, topk_global],
                outputs=pf_out,
            )

    # ---- Footer -------------------------------------------------------
    gr.Markdown(
        """
---
**Algorithm** — Quantum Phase Estimation (QPE) drives Quantum Amplitude Estimation (QAE).
For each option type the state preparation operator **A** encodes the risk-neutral price
distribution and discounted payoff. The Grover operator **Q = A·S₀·A†·S_good** is
repeatedly applied under phase kick-back; the inverse QFT extracts the amplitude
*a = sin²(θ)* which gives the option price *P = F·a* where *F* is the maximum payoff.
All circuits are transpiled to the Automatski safe gate set before execution.
        """
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
