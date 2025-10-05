from __future__ import annotations
import numpy as np
from .core import from_stops
from .registry import register

# Minimal, permissive, CVD-safe core (no heavy deps)
# viridis-like (license-safe approximation)
_VIRIDIS_STOPS = [
    (0.0,  "#440154"), (0.13, "#472C7A"), (0.25, "#3B518B"),
    (0.38, "#2C718E"), (0.5, "#21918C"), (0.63, "#2DB27D"),
    (0.75, "#73D055"), (0.88, "#DCE319"), (1.0,  "#FDE725"),
]
_MAGMA_STOPS = [
    (0.0, "#000004"), (0.2, "#3B0F70"), (0.4, "#8C2981"),
    (0.6, "#DE4968"), (0.8, "#FE9F6D"), (1.0, "#FCFDBF"),
]
# vik-like (diverging around mid)
_VIK_STOPS = [
    (0.0, "#00224E"), (0.25, "#2E6DB4"), (0.5, "#F5F5F5"),
    (0.75, "#C14A3B"), (1.0, "#5A0000"),
]
# Okabe–Ito (categorical seed, exposed as utility)
OKABE_ITO = ["#000000","#E69F00","#56B4E9","#009E73","#F0E442","#0072B2","#D55E00","#CC79A7"]

def _register_core():
    register("forge3d:viridis", lambda: from_stops("forge3d:viridis", _VIRIDIS_STOPS, 256))
    register("forge3d:magma",   lambda: from_stops("forge3d:magma",   _MAGMA_STOPS,   256))
    register("forge3d:vik",     lambda: from_stops("forge3d:vik",     _VIK_STOPS,     256))
_register_core()
