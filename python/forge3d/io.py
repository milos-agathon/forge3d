# python/forge3d/io.py
# High-level wrappers for OBJ import/export (Workstream F4/F5)

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from .geometry import MeshBuffers, _mesh_from_py, _mesh_to_py  # internal helpers

try:
    from . import _forge3d
except ImportError:  # pragma: no cover - handled in tests via skip
    _forge3d = None  # type: ignore[assignment]


def _ensure_native() -> None:
    if _forge3d is None:  # pragma: no cover - import guard
        raise RuntimeError("forge3d native module is not available; build the extension first")


@dataclass
class ObjMaterial:
    name: str
    diffuse_color: np.ndarray  # (3,)
    ambient_color: np.ndarray  # (3,)
    specular_color: np.ndarray  # (3,)
    diffuse_texture: Optional[str]


def load_obj(
    path: str,
    *,
    return_metadata: bool = False,
) -> MeshBuffers | Tuple[
    MeshBuffers,
    List[ObjMaterial],
    Dict[str, np.ndarray],  # material_groups
    Dict[str, np.ndarray],  # g_groups
    Dict[str, np.ndarray],  # o_groups
]:
    """Load a Wavefront OBJ file.

    Parameters
    ----------
    path: str
        Path to the OBJ file.
    return_metadata: bool, default False
        If True, also return a list of `ObjMaterial` entries and a mapping of material name to
        triangle ordinal indices (one per triangle in reading order).

    Returns
    -------
    MeshBuffers or (MeshBuffers, materials, material_groups, g_groups, o_groups)
        If `return_metadata` is False, returns only the mesh.
        Otherwise returns a tuple with materials parsed from MTL and group indices per material,
        plus `g` and `o` groupings as triangle ordinal arrays.

    Notes
    -----
    - Faces are triangulated on import.
    - MTL parsing is minimal: `Kd`, `Ka`, `Ks`, and `map_Kd` are supported.
    """
    _ensure_native()
    result = _forge3d.io_import_obj_py(str(path))
    # result is a dict with keys: mesh, materials, groups
    mesh = _mesh_from_py(result["mesh"])  # type: ignore[arg-type]
    if not return_metadata:
        return mesh

    materials: List[ObjMaterial] = []
    for md in result["materials"]:
        materials.append(
            ObjMaterial(
                name=str(md["name"]),
                diffuse_color=np.asarray(md["diffuse_color"], dtype=np.float32),
                ambient_color=np.asarray(md["ambient_color"], dtype=np.float32),
                specular_color=np.asarray(md["specular_color"], dtype=np.float32),
                diffuse_texture=(None if md["diffuse_texture"] is None else str(md["diffuse_texture"]))
            )
        )

    # Backward-compat: prefer "material_groups" if present, otherwise fallback to "groups"
    mat_groups_src: Dict[str, Any] = result.get("material_groups", result.get("groups", {}))
    material_groups: Dict[str, np.ndarray] = {}
    for k, v in mat_groups_src.items():
        material_groups[str(k)] = np.asarray(v, dtype=np.uint32)

    g_groups: Dict[str, np.ndarray] = {}
    for k, v in result.get("g_groups", {}).items():
        g_groups[str(k)] = np.asarray(v, dtype=np.uint32)

    o_groups: Dict[str, np.ndarray] = {}
    for k, v in result.get("o_groups", {}).items():
        o_groups[str(k)] = np.asarray(v, dtype=np.uint32)

    return mesh, materials, material_groups, g_groups, o_groups


def save_obj(
    mesh: MeshBuffers,
    path: str,
    *,
    materials: Optional[List[ObjMaterial]] = None,
    material_groups: Optional[Dict[str, np.ndarray]] = None,
    g_groups: Optional[Dict[str, np.ndarray]] = None,
    o_groups: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """Save a MeshBuffers as a Wavefront OBJ file, optionally with materials and groups.

    Parameters
    ----------
    mesh: MeshBuffers
        Geometry to export.
    path: str
        Destination OBJ path. If materials are provided, a sibling .mtl will be written.
    materials: Optional[List[ObjMaterial]]
        Materials to write to the MTL file (minimal fields: name, Kd/Ka/Ks, map_Kd).
    material_groups: Optional[Dict[str, np.ndarray]]
        Mapping material name -> triangle ordinal indices.
    g_groups: Optional[Dict[str, np.ndarray]]
        Mapping 'g' group name -> triangle ordinal indices.
    o_groups: Optional[Dict[str, np.ndarray]]
        Mapping 'o' object name -> triangle ordinal indices.

    Notes
    -----
    - Exports v/vt/vn/f records. If UV or normal arrays are empty, the corresponding tokens
      are omitted in faces.
    - Faces are written as triangle-list.
    - If materials are provided, mtllib/usemtl statements are emitted and a .mtl is written.
    - Groups are emitted as g/o prior to the corresponding faces.
    """
    _ensure_native()
    payload = _mesh_to_py(mesh)
    mats_payload = None
    if materials is not None:
        mats_payload = []
        for m in materials:
            md = {
                "name": m.name,
                "diffuse_color": tuple(np.asarray(m.diffuse_color, dtype=np.float32).tolist()),
                "ambient_color": tuple(np.asarray(m.ambient_color, dtype=np.float32).tolist()),
                "specular_color": tuple(np.asarray(m.specular_color, dtype=np.float32).tolist()),
            }
            if m.diffuse_texture is not None:
                md["diffuse_texture"] = m.diffuse_texture
            mats_payload.append(md)

    def _groups_to_py(d: Optional[Dict[str, np.ndarray]]):
        if d is None:
            return None
        out: Dict[str, np.ndarray] = {}
        for k, v in d.items():
            out[str(k)] = np.asarray(v, dtype=np.uint32)
        return out

    mg = _groups_to_py(material_groups)
    gg = _groups_to_py(g_groups)
    og = _groups_to_py(o_groups)

    _forge3d.io_export_obj_py(str(path), payload, mats_payload, mg, gg, og)
