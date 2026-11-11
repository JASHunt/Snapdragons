"""Snapdragons: stellar population synthesis from galaxy simulations.

Generate mock Gaia catalogues from hydrodynamical simulation snapshots.
"""

__version__ = "2.1.1"

from .config import Config
from .pipeline import Pipeline
from .reader import read_catalogue, Catalogue
from .simulation import SimulationSnapshot
from .isochrones import IsochroneGrid
from .imf import get_imf, KroupaIMF, SalpeterIMF
from .extinction import get_extinction_calculator
from .errors import GaiaErrorModel
from .photometry import grvs_from_g_rp

__all__ = [
    "grvs_from_g_rp",
    "Config",
    "Pipeline",
    "read_catalogue",
    "Catalogue",
    "SimulationSnapshot",
    "IsochroneGrid",
    "get_imf",
    "KroupaIMF",
    "SalpeterIMF",
    "get_extinction_calculator",
    "GaiaErrorModel",
]
