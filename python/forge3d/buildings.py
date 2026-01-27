# python/forge3d/buildings.py
"""
P4: 3D Buildings Pipeline

High-level Python API for loading, processing, and rendering 3D buildings.
Supports GeoJSON, CityJSON, and OSM building data formats.

Example usage:
    from forge3d.buildings import add_buildings, BuildingLayer

    # Load buildings from GeoJSON
    layer = add_buildings("buildings.geojson")
    print(f"Loaded {layer.building_count} buildings")

    # Load buildings from CityJSON
    layer = add_buildings_cityjson("3dbag_tile.json")
    print(f"LOD: {layer.max_lod}")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Try to import native module
try:
    from ._native import get_native_module as _get_native_module
    _NATIVE = _get_native_module()
except ImportError:
    _NATIVE = None


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class BuildingMaterial:
    """PBR material properties for a building surface."""
    albedo: Tuple[float, float, float] = (0.7, 0.7, 0.7)
    roughness: float = 0.6
    metallic: float = 0.0
    ior: float = 1.5
    emissive: float = 0.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BuildingMaterial":
        """Create from dictionary (e.g., from native module)."""
        return cls(
            albedo=tuple(d.get("albedo", (0.7, 0.7, 0.7))),
            roughness=d.get("roughness", 0.6),
            metallic=d.get("metallic", 0.0),
            ior=d.get("ior", 1.5),
            emissive=d.get("emissive", 0.0),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "albedo": self.albedo,
            "roughness": self.roughness,
            "metallic": self.metallic,
            "ior": self.ior,
            "emissive": self.emissive,
        }


@dataclass
class Building:
    """A single building with geometry and attributes."""
    id: str
    positions: np.ndarray  # shape: (N, 3) or flat (N*3,)
    indices: np.ndarray    # shape: (M,) triangle indices
    normals: Optional[np.ndarray] = None  # shape: (N, 3) or flat
    height: Optional[float] = None
    ground_height: Optional[float] = None
    roof_type: str = "flat"
    material: BuildingMaterial = field(default_factory=BuildingMaterial)
    lod: int = 1
    attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def vertex_count(self) -> int:
        """Number of vertices."""
        if self.positions.ndim == 1:
            return len(self.positions) // 3
        return len(self.positions)

    @property
    def triangle_count(self) -> int:
        """Number of triangles."""
        return len(self.indices) // 3

    def bounds(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        """Get bounding box (min_x, min_y, min_z, max_x, max_y, max_z)."""
        if self.positions.size == 0:
            return None
        pos = self.positions.reshape(-1, 3)
        return (
            float(pos[:, 0].min()), float(pos[:, 1].min()), float(pos[:, 2].min()),
            float(pos[:, 0].max()), float(pos[:, 1].max()), float(pos[:, 2].max()),
        )


@dataclass
class BuildingLayer:
    """A collection of buildings ready for rendering."""
    name: str
    buildings: List[Building] = field(default_factory=list)
    crs_epsg: Optional[int] = None
    source_format: str = "unknown"

    @property
    def building_count(self) -> int:
        """Number of buildings in layer."""
        return len(self.buildings)

    @property
    def total_vertices(self) -> int:
        """Total vertex count across all buildings."""
        return sum(b.vertex_count for b in self.buildings)

    @property
    def total_triangles(self) -> int:
        """Total triangle count across all buildings."""
        return sum(b.triangle_count for b in self.buildings)

    @property
    def max_lod(self) -> int:
        """Maximum LOD level in layer."""
        if not self.buildings:
            return 0
        return max(b.lod for b in self.buildings)

    def bounds(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        """Get combined bounding box of all buildings."""
        all_bounds = [b.bounds() for b in self.buildings if b.bounds() is not None]
        if not all_bounds:
            return None
        min_x = min(b[0] for b in all_bounds)
        min_y = min(b[1] for b in all_bounds)
        min_z = min(b[2] for b in all_bounds)
        max_x = max(b[3] for b in all_bounds)
        max_y = max(b[4] for b in all_bounds)
        max_z = max(b[5] for b in all_bounds)
        return (min_x, min_y, min_z, max_x, max_y, max_z)


# ============================================================================
# Roof Type Inference
# ============================================================================

def infer_roof_type(properties: Dict[str, Any]) -> str:
    """
    Infer roof type from OSM/building properties.

    Args:
        properties: Dictionary of building properties/tags

    Returns:
        Roof type string: "flat", "gabled", "hipped", "pyramidal", etc.

    Example:
        roof = infer_roof_type({"building:roof:shape": "gabled"})
        assert roof == "gabled"
    """
    if _NATIVE is not None and hasattr(_NATIVE, "infer_roof_type_py"):
        return _NATIVE.infer_roof_type_py(json.dumps(properties))

    # Fallback Python implementation
    # Check explicit roof tags
    for key in ("building:roof:shape", "roof:shape", "roof_shape"):
        if key in properties:
            return _parse_roof_tag(properties[key])

    # Infer from building type
    building_type = properties.get("building", "").lower()
    roof_map = {
        "house": "gabled",
        "detached": "gabled",
        "residential": "gabled",
        "warehouse": "flat",
        "industrial": "flat",
        "commercial": "flat",
        "apartments": "hipped",
        "church": "gabled",
    }
    return roof_map.get(building_type, "flat")


def _parse_roof_tag(value: str) -> str:
    """Parse OSM roof shape tag value."""
    v = value.lower().strip()
    valid = {"flat", "gabled", "hipped", "pyramidal", "dome", "mansard",
             "shed", "gambrel", "onion", "skillion"}
    if v in valid:
        return v
    if v in ("lean_to", "lean-to"):
        return "shed"
    return "flat"


# ============================================================================
# Material Inference
# ============================================================================

def material_from_tags(tags: Dict[str, Any]) -> BuildingMaterial:
    """
    Infer building material from OSM tags.

    Args:
        tags: Dictionary of building tags/properties

    Returns:
        BuildingMaterial with inferred PBR properties

    Example:
        mat = material_from_tags({"building:material": "brick"})
        assert mat.albedo[0] > mat.albedo[2]  # Brick is reddish
    """
    if _NATIVE is not None and hasattr(_NATIVE, "material_from_tags_py"):
        result = _NATIVE.material_from_tags_py(json.dumps(tags))
        return BuildingMaterial.from_dict(result)

    # Fallback: return default
    return BuildingMaterial()


def material_from_name(name: str) -> BuildingMaterial:
    """
    Get material preset by name.

    Args:
        name: Material name (e.g., "brick", "glass", "concrete")

    Returns:
        BuildingMaterial with preset PBR properties
    """
    if _NATIVE is not None and hasattr(_NATIVE, "material_from_name_py"):
        result = _NATIVE.material_from_name_py(name)
        return BuildingMaterial.from_dict(result)

    # Fallback presets
    presets = {
        "brick": BuildingMaterial(albedo=(0.55, 0.25, 0.18), roughness=0.75),
        "concrete": BuildingMaterial(albedo=(0.6, 0.58, 0.55), roughness=0.7),
        "glass": BuildingMaterial(albedo=(0.04, 0.04, 0.05), roughness=0.1),
        "steel": BuildingMaterial(albedo=(0.56, 0.57, 0.58), roughness=0.35, metallic=0.9),
        "wood": BuildingMaterial(albedo=(0.5, 0.35, 0.2), roughness=0.7),
    }
    return presets.get(name.lower(), BuildingMaterial())


# ============================================================================
# GeoJSON Loading
# ============================================================================

def add_buildings(
    geojson_path: Union[str, Path],
    *,
    default_height: float = 10.0,
    height_key: Optional[str] = None,
    name: Optional[str] = None,
) -> BuildingLayer:
    """
    Load buildings from a GeoJSON file.

    Args:
        geojson_path: Path to GeoJSON file with building polygons
        default_height: Default extrusion height in meters (default: 10.0)
        height_key: Property key for building height (default: "height")
        name: Layer name (default: filename without extension)

    Returns:
        BuildingLayer with extruded building geometries

    Example:
        layer = add_buildings("buildings.geojson", default_height=12.0)
        print(f"Loaded {layer.building_count} buildings")
    """
    path = Path(geojson_path)
    if not path.exists():
        raise FileNotFoundError(f"GeoJSON file not found: {path}")

    layer_name = name or path.stem

    with open(path, "r", encoding="utf-8") as f:
        geojson_str = f.read()

    # Try native implementation
    if _NATIVE is not None and hasattr(_NATIVE, "import_osm_buildings_from_geojson_py"):
        result = _NATIVE.import_osm_buildings_from_geojson_py(
            geojson_str, default_height, height_key
        )
        # Result is a dict with positions, normals, uvs, indices
        positions = np.array(result["positions"], dtype=np.float32)
        indices = np.array(result["indices"], dtype=np.uint32)
        normals = np.array(result.get("normals", []), dtype=np.float32)

        # Create single merged building
        building = Building(
            id=layer_name,
            positions=positions,
            indices=indices,
            normals=normals if normals.size > 0 else None,
            height=default_height,
            roof_type="flat",
        )

        return BuildingLayer(
            name=layer_name,
            buildings=[building],
            source_format="geojson",
        )

    # Fallback: parse GeoJSON and create basic layer
    data = json.loads(geojson_str)
    if data.get("type") != "FeatureCollection":
        raise ValueError("GeoJSON must be a FeatureCollection")

    buildings = []
    for i, feature in enumerate(data.get("features", [])):
        props = feature.get("properties", {})
        geom = feature.get("geometry", {})

        # Get height
        h = default_height
        if height_key and height_key in props:
            try:
                h = float(props[height_key])
            except (ValueError, TypeError):
                pass
        elif "height" in props:
            try:
                h = float(props["height"])
            except (ValueError, TypeError):
                pass

        # Skip non-polygon geometries
        if geom.get("type") not in ("Polygon", "MultiPolygon"):
            continue

        # Create placeholder building (geometry would need actual extrusion)
        building = Building(
            id=props.get("id", f"building_{i}"),
            positions=np.zeros((0, 3), dtype=np.float32),
            indices=np.zeros(0, dtype=np.uint32),
            height=h,
            roof_type=infer_roof_type(props),
            material=material_from_tags(props),
            attributes=props,
        )
        buildings.append(building)

    return BuildingLayer(
        name=layer_name,
        buildings=buildings,
        source_format="geojson",
    )


# ============================================================================
# CityJSON Loading
# ============================================================================

def add_buildings_cityjson(
    cityjson_path: Union[str, Path],
    *,
    name: Optional[str] = None,
) -> BuildingLayer:
    """
    Load buildings from a CityJSON file.

    CityJSON is a JSON-based encoding of CityGML, commonly used for
    3D city models. Supports LOD1-LOD3 geometries.

    Args:
        cityjson_path: Path to CityJSON file
        name: Layer name (default: filename without extension)

    Returns:
        BuildingLayer with 3D building geometries

    Example:
        layer = add_buildings_cityjson("3dbag_tile.json")
        print(f"Loaded {layer.building_count} buildings at LOD {layer.max_lod}")
    """
    path = Path(cityjson_path)
    if not path.exists():
        raise FileNotFoundError(f"CityJSON file not found: {path}")

    layer_name = name or path.stem

    with open(path, "rb") as f:
        data = f.read()

    # Try native implementation
    if _NATIVE is not None and hasattr(_NATIVE, "parse_cityjson_py"):
        result = _NATIVE.parse_cityjson_py(data)

        meta = result.get("metadata", {})
        crs_epsg = meta.get("crs_epsg")

        buildings = []
        for bdata in result.get("buildings", []):
            positions = np.array(bdata["positions"], dtype=np.float32)
            indices = np.array(bdata["indices"], dtype=np.uint32)
            normals = bdata.get("normals")
            if normals is not None:
                normals = np.array(normals, dtype=np.float32)

            mat_data = bdata.get("material", {})
            material = BuildingMaterial.from_dict(mat_data)

            building = Building(
                id=bdata["id"],
                positions=positions,
                indices=indices,
                normals=normals,
                height=bdata.get("height"),
                ground_height=bdata.get("ground_height"),
                roof_type=bdata.get("roof_type", "flat"),
                material=material,
                lod=bdata.get("lod", 1),
            )
            buildings.append(building)

        return BuildingLayer(
            name=layer_name,
            buildings=buildings,
            crs_epsg=crs_epsg,
            source_format="cityjson",
        )

    # Fallback: basic JSON parsing
    json_data = json.loads(data.decode("utf-8"))
    if json_data.get("type") != "CityJSON":
        raise ValueError("Not a valid CityJSON file")

    # Parse metadata
    crs_epsg = None
    meta = json_data.get("metadata", {})
    ref_sys = meta.get("referenceSystem", "")
    if "::" in ref_sys:
        try:
            crs_epsg = int(ref_sys.split("::")[-1])
        except ValueError:
            pass

    # Parse transform
    transform = json_data.get("transform", {})
    scale = transform.get("scale", [1, 1, 1])
    translate = transform.get("translate", [0, 0, 0])

    # Parse vertices
    raw_verts = json_data.get("vertices", [])
    vertices = []
    for v in raw_verts:
        x = v[0] * scale[0] + translate[0]
        y = v[1] * scale[1] + translate[1]
        z = v[2] * scale[2] + translate[2]
        vertices.append([x, y, z])

    # Parse city objects
    buildings = []
    city_objects = json_data.get("CityObjects", {})
    for obj_id, obj in city_objects.items():
        obj_type = obj.get("type", "")
        if not obj_type.startswith("Building"):
            continue

        # Extract attributes
        attrs = obj.get("attributes", {})
        height = None
        for key in ("measuredHeight", "height", "h_dak", "h_max"):
            if key in attrs:
                try:
                    height = float(attrs[key])
                    break
                except (ValueError, TypeError):
                    pass

        # Create placeholder building
        building = Building(
            id=obj_id,
            positions=np.zeros((0, 3), dtype=np.float32),
            indices=np.zeros(0, dtype=np.uint32),
            height=height,
            roof_type=infer_roof_type(attrs),
            material=material_from_tags(attrs),
            lod=1,
            attributes=attrs,
        )
        buildings.append(building)

    return BuildingLayer(
        name=layer_name,
        buildings=buildings,
        crs_epsg=crs_epsg,
        source_format="cityjson",
    )


# ============================================================================
# 3D Tiles Loading
# ============================================================================

def add_buildings_3dtiles(
    tileset_path: Union[str, Path],
    *,
    name: Optional[str] = None,
) -> BuildingLayer:
    """
    Load buildings from a 3D Tiles tileset.

    Note: This loads the tileset metadata. Actual tile content (b3dm)
    is loaded on demand during rendering.

    Args:
        tileset_path: Path to tileset.json
        name: Layer name (default: filename without extension)

    Returns:
        BuildingLayer with 3D Tiles metadata

    Example:
        layer = add_buildings_3dtiles("path/to/tileset.json")
        print(f"Tileset bounds: {layer.bounds()}")
    """
    path = Path(tileset_path)
    if not path.exists():
        raise FileNotFoundError(f"Tileset not found: {path}")

    layer_name = name or path.stem

    with open(path, "r", encoding="utf-8") as f:
        tileset = json.load(f)

    # Basic validation
    if "root" not in tileset:
        raise ValueError("Invalid tileset: missing 'root' node")

    # For now, return empty layer with metadata
    # Actual tile loading happens during rendering
    return BuildingLayer(
        name=layer_name,
        buildings=[],
        source_format="3dtiles",
    )


# ============================================================================
# Module exports
# ============================================================================

__all__ = [
    # Data classes
    "Building",
    "BuildingLayer",
    "BuildingMaterial",
    # Loading functions
    "add_buildings",
    "add_buildings_cityjson",
    "add_buildings_3dtiles",
    # Inference functions
    "infer_roof_type",
    "material_from_tags",
    "material_from_name",
]
