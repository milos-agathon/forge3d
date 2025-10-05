from __future__ import annotations
import json
from .core import Colormap, from_stops

def load_json(path: str, n: int = 256) -> Colormap:
    with open(path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    name = spec.get("name", path)
    stops = spec["stops"]  # list of [pos, color]
    return from_stops(name, stops, n=n)

def load_cpt(path: str, n: int = 256) -> Colormap:
    try:
        from .providers import load_provider
        return load_provider("pycpt")(path)
    except Exception as e:
        raise RuntimeError("pycpt required for CPT files. Install forge3d[colormaps].") from e
