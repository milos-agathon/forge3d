# python/forge3d/tiles/__init__.py
# Tile providers, XYZ/WMTS client, and attribution overlay helpers.
# Enables basemap fetching/composition and simple overlays for demos/tests.
# RELEVANT FILES:python/forge3d/tiles/client.py,python/forge3d/tiles/overlay.py,examples/xyz_tile_compose_demo.py

from __future__ import annotations

from .client import TileClient, TileProvider, bbox_to_tiles  # noqa: F401
from .overlay import draw_attribution  # noqa: F401

__all__ = [
    "TileClient",
    "TileProvider",
    "bbox_to_tiles",
    "draw_attribution",
]

