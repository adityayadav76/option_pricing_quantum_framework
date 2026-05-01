"""
Portfolio pricing: price a collection of options of any type and
aggregate them into a portfolio value.

Each leg is priced individually using its own QAE circuit.
Portfolio value = Σ_i  w_i · price_i
where w_i is the signed quantity (positive = long, negative = short).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from .options.european  import price_european
from .options.american  import price_american
from .options.bermudan  import price_bermudan
from .options.asian     import price_asian
from .options.barrier   import price_barrier
from .options.binary    import price_binary


PRICERS = {
    "european": price_european,
    "american": price_american,
    "bermudan": price_bermudan,
    "asian":    price_asian,
    "barrier":  price_barrier,
    "binary":   price_binary,
}


@dataclass
class OptionLeg:
    """One option position in a portfolio."""
    option_type_class: str       # 'european', 'american', etc.
    params: dict                 # keyword arguments for the pricer (excl. backend/shots)
    quantity: float = 1.0        # positive = long, negative = short
    label: str = ""              # optional description

    def price(self, backend, shots: int, top_k: int) -> dict:
        pricer = PRICERS.get(self.option_type_class.lower())
        if pricer is None:
            raise ValueError(f"Unknown option class: {self.option_type_class}. "
                             f"Choose from {list(PRICERS)}")
        result = pricer(**self.params, backend=backend, shots=shots, top_k=top_k)
        result["quantity"] = self.quantity
        result["label"]    = self.label or self.option_type_class
        result["leg_value_quantum"]   = self.quantity * result["quantum_price"]
        result["leg_value_classical"] = self.quantity * result["classical_price"]
        return result


def price_portfolio(
    legs: list[OptionLeg],
    backend,
    shots: int = 100000,
    top_k: int = 10000,
) -> dict:
    """
    Price a portfolio of options by pricing each leg individually.

    Parameters
    ----------
    legs    : list of OptionLeg instances
    backend : AutomatskiKomencoQiskit instance

    Returns
    -------
    dict with per-leg results and aggregate portfolio metrics.
    """
    leg_results = []
    total_quantum   = 0.0
    total_classical = 0.0

    for i, leg in enumerate(legs):
        result = leg.price(backend, shots, top_k)
        leg_results.append(result)
        total_quantum   += result["leg_value_quantum"]
        total_classical += result["leg_value_classical"]

    return {
        "portfolio_value_quantum":   total_quantum,
        "portfolio_value_classical": total_classical,
        "legs": leg_results,
        "n_legs": len(legs),
    }
