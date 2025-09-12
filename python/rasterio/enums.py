# python/rasterio/enums.py
# Minimal enums stub exposing Resampling constants used in tests and demos.
# Exists to allow import `from rasterio.enums import Resampling`.
# RELEVANT FILES:python/forge3d/adapters/rasterio_tiles.py,python/forge3d/adapters/reproject.py

"""
Stub Resampling enum.
"""

class _Resampling:
    nearest = object()
    bilinear = object()
    cubic = object()

Resampling = _Resampling
