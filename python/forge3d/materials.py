"""PBR Materials - compatibility shim.

This module provides a compatibility layer for PBR materials.
All functionality is re-exported from the pbr module.
"""

# Re-export all PBR functionality from the pbr module
from .pbr import *

__all__ = [
    # Re-export everything from pbr
    *getattr(__import__('forge3d.pbr', fromlist=['__all__']), '__all__', [])
]