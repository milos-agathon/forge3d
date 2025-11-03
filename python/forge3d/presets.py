"""
python/forge3d/presets.py
High-level rendering presets for Python UX polish (P7-01).

Each preset returns a plain dict compatible with
python/forge3d/config.py::RendererConfig.from_mapping(), covering
lighting/shading/shadows/gi/atmosphere keys. Values are chosen to be
validation-friendly (e.g., GI modes empty by default to avoid HDRI
requirements; shadow map sizes are powers of two; CSM cascades 2..4).

Example
-------
>>> from forge3d import Renderer
>>> from forge3d import presets
>>> r = Renderer(1280, 720)
>>> cfg = presets.get("outdoor_sun")
>>> # Optionally add overrides (e.g., enable GI IBL once HDR is provided)
>>> cfg["gi"] = {"modes": ["ibl"]}
>>> cfg["atmosphere"]["hdr_path"] = "assets/sky.hdr"
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _dir_light(
    *,
    direction: tuple[float, float, float],
    intensity: float = 5.0,
    color: tuple[float, float, float] = (1.0, 0.97, 0.94),
) -> Dict[str, Any]:
    """Build a directional light mapping compatible with LightConfig.from_mapping()."""
    return {
        "type": "directional",
        "direction": [float(direction[0]), float(direction[1]), float(direction[2])],
        "intensity": float(intensity),
        "color": [float(color[0]), float(color[1]), float(color[2])],
    }


def _normalize_name(name: str) -> str:
    return "".join(c for c in str(name).strip().lower() if c not in {"-", "_", " ", "."})


# -----------------------------------------------------------------------------
# Preset definitions (schema-aligned with python/forge3d/config.py)
# -----------------------------------------------------------------------------

def studio_pbr() -> Dict[str, Any]:
    """Studio preset: directional key, Disney BRDF, PCF shadows.

    Notes
    -----
    - GI modes are left empty to avoid HDR asset requirements by default.
    - Atmosphere is disabled; users may enable HDR sky via overrides later.
    """
    return {
        "lighting": {
            "exposure": 1.0,
            "lights": [
                _dir_light(direction=(-0.30, -0.95, -0.20), intensity=6.0, color=(1.0, 0.98, 0.95)),
            ],
        },
        "shading": {
            "brdf": "disney-principled",
            "roughness": 0.35,
            "metallic": 0.0,
            "normal_maps": True,
        },
        "shadows": {
            "enabled": True,
            "technique": "pcf",
            "map_size": 2048,
            "cascades": 1,
        },
        "gi": {
            "modes": [],
        },
        "atmosphere": {
            "enabled": False,
        },
    }


def outdoor_sun() -> Dict[str, Any]:
    """Outdoor preset: Hosek–Wilkie sky, sun as directional light, CSM, GGX."""
    return {
        "lighting": {
            "exposure": 1.0,
            "lights": [
                _dir_light(direction=(-0.35, -1.00, -0.25), intensity=5.0, color=(1.0, 0.97, 0.92)),
            ],
        },
        "shading": {
            "brdf": "cooktorrance-ggx",
            "roughness": 0.5,
            "metallic": 0.0,
            "normal_maps": True,
        },
        "shadows": {
            "enabled": True,
            "technique": "csm",
            "map_size": 2048,
            "cascades": 3,  # Valid range [2,4] for CSM; pick 3 for balance
        },
        "gi": {
            "modes": [],  # Users can enable ["ibl"] if they provide an HDR path
        },
        "atmosphere": {
            "enabled": True,
            "sky": "hosek-wilkie",
            # "hdr_path": None  # provide via overrides if using GI IBL
        },
    }


def toon_viz() -> Dict[str, Any]:
    """Toon visualization: toon BRDF, hard shadows, no GI, flat background."""
    return {
        "lighting": {
            "exposure": 1.0,
            "lights": [
                _dir_light(direction=(-0.40, -0.90, -0.10), intensity=4.0, color=(1.0, 1.0, 1.0)),
            ],
        },
        "shading": {
            "brdf": "toon",
            "normal_maps": False,
        },
        "shadows": {
            "enabled": True,
            "technique": "hard",
            "map_size": 1024,
            "cascades": 1,
        },
        "gi": {
            "modes": [],
        },
        "atmosphere": {
            "enabled": False,
        },
    }


# -----------------------------------------------------------------------------
# Registry and lookup helpers
# -----------------------------------------------------------------------------

_PRESETS: Dict[str, Callable[[], Dict[str, Any]]] = {
    "studio_pbr": studio_pbr,
    "outdoor_sun": outdoor_sun,
    "toon_viz": toon_viz,
}

_ALIASES: Dict[str, str] = {
    "studio": "studio_pbr",
    "pbr": "studio_pbr",
    "sun": "outdoor_sun",
    "outdoor": "outdoor_sun",
    "toon": "toon_viz",
    "toonviz": "toon_viz",
}


def available() -> List[str]:
    """List available preset names."""
    return sorted(_PRESETS.keys())


def get(name: str) -> Dict[str, Any]:
    """Resolve a preset by name (case-insensitive; supports common aliases).

    Raises
    ------
    ValueError
        If the preset name is unknown.
    """
    key = _normalize_name(name)
    if key in _ALIASES:
        key = _ALIASES[key]
    if key not in _PRESETS:
        raise ValueError(f"Unknown preset: {name!r}. Available: {', '.join(available())}")
    # Return a shallow copy to avoid accidental mutation of registry entries
    out = _PRESETS[key]()
    assert isinstance(out, dict)
    return dict(out)


__all__ = [
    "studio_pbr",
    "outdoor_sun",
    "toon_viz",
    "available",
    "get",
]
