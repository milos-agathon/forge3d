# python/pyproj/__init__.py
# Minimal pyproj stub exposing CRS and Transformer for tests/demos.
# Exists to allow imports and monkeypatching without installing pyproj.
# RELEVANT FILES:python/forge3d/adapters/reproject.py,tests/test_reproject_window.py

"""
Stub pyproj interfaces for local testing.
"""

class CRS:  # pragma: no cover
    @classmethod
    def from_string(cls, s):
        return cls()

class Transformer:  # pragma: no cover
    def __init__(self, *args, **kwargs):
        pass
    @classmethod
    def from_crs(cls, src, dst, always_xy=False):
        return cls()
    def transform(self, xs, ys):
        return xs, ys
