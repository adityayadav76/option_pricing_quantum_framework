"""Quantum Options Pricing Framework — Automatski QPE/QAE backend."""

from .options.european  import price_european
from .options.american  import price_american
from .options.bermudan  import price_bermudan
from .options.asian     import price_asian
from .options.barrier   import price_barrier
from .options.binary    import price_binary
from .portfolio         import price_portfolio, OptionLeg
from .qae_engine        import SAFE_BASIS

__all__ = [
    "price_european",
    "price_american",
    "price_bermudan",
    "price_asian",
    "price_barrier",
    "price_binary",
    "price_portfolio",
    "OptionLeg",
    "SAFE_BASIS",
]
