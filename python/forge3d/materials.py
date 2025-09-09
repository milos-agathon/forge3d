"""PBR Materials - legacy compatibility shim.

⚠️  MATERIALS MODULE POLICY:

This module provides legacy compatibility for code that imports `forge3d.materials`.
ALL functionality is re-exported from the `forge3d.pbr` module.

📋 RECOMMENDED USAGE:
    import forge3d.pbr as pbr           # Preferred - direct import
    material = pbr.PbrMaterial(...)     # Clear, explicit
    
🔧 LEGACY COMPATIBILITY:
    import forge3d.materials as mat     # Still supported
    material = mat.PbrMaterial(...)     # Works but not preferred
    
💡 MIGRATION PATH:
    - Existing code using `forge3d.materials` will continue to work
    - New code should use `forge3d.pbr` directly for better clarity
    - This shim will be maintained for backward compatibility
    
📚 RELATED MODULES:
    - forge3d.pbr: PBR materials system (primary implementation)
    - forge3d.shadows: Shadow mapping and CSM functionality
    
All classes, functions, and constants from `forge3d.pbr` are available here.
"""

# Re-export all PBR functionality from the pbr module
from .pbr import *

__all__ = [
    # Re-export everything from pbr
    *getattr(__import__('forge3d.pbr', fromlist=['__all__']), '__all__', [])
]