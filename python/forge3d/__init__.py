# python/forge3d/__init__.py
# Public Python API entry for forge3d package.
# Exists to expose minimal interfaces for textures, materials, and path tracing used in tests.
# RELEVANT FILES:python/forge3d/path_tracing.py,python/forge3d/materials.py,python/forge3d/textures.py

from .path_tracing import PathTracer, make_camera
from .materials import PbrMaterial
from .textures import load_texture, build_pbr_textures
from .sdf import (
    SdfPrimitive, SdfScene, SdfSceneBuilder, HybridRenderer,
    SdfPrimitiveType, CsgOperation, TraversalMode,
    create_sphere, create_box, create_simple_scene, render_simple_scene
)

__all__ = [
    "PathTracer",
    "make_camera",
    "PbrMaterial",
    "load_texture",
    "build_pbr_textures",
    # SDF functionality
    "SdfPrimitive",
    "SdfScene",
    "SdfSceneBuilder",
    "HybridRenderer",
    "SdfPrimitiveType",
    "CsgOperation",
    "TraversalMode",
    "create_sphere",
    "create_box",
    "create_simple_scene",
    "render_simple_scene",
]

