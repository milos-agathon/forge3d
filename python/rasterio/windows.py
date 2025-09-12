# python/rasterio/windows.py
# Minimal windows stub providing Window and transform for tests and demos.
# Exists to satisfy imports in adapters without real rasterio.
# RELEVANT FILES:python/forge3d/adapters/rasterio_tiles.py,python/forge3d/adapters/reproject.py

"""
Stub window utilities.
"""

class Window:
    def __init__(self, col_off, row_off, width, height):
        self.col_off = int(col_off)
        self.row_off = int(row_off)
        self.width = int(width)
        self.height = int(height)

def transform(window, base_transform):  # pragma: no cover - placeholder behavior
    # Return the base transform unchanged for tests that don't assert exact values.
    return base_transform
