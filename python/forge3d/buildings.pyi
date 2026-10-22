import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np
from .diagnostics import LayerSummary, ValidationReport, placeholder_fallback_diagnostic, python_public_3dtiles_incomplete_diagnostic

@dataclass
class BuildingMaterial:
    albedo: tuple[float, float, float] = (0.7, 0.7, 0.7)
    roughness: float = 0.6
    metallic: float = 0.0
    ior: float = 1.5
    emissive: float = 0.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> 'BuildingMaterial':
        ...

    def to_dict(self) -> dict[str, Any]:
        ...

@dataclass
class Building:
    id: str
    positions: np.ndarray
    indices: np.ndarray
    normals: np.ndarray | None = None
    height: float | None = None
    ground_height: float | None = None
    roof_type: str = 'flat'
    material: BuildingMaterial = field(default_factory=BuildingMaterial)
    lod: int = 1
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def vertex_count(self) -> int:
        ...

    @property
    def triangle_count(self) -> int:
        ...

    def bounds(self) -> tuple[float, float, float, float, float, float] | None:
        ...

@dataclass
class BuildingLayer:
    name: str
    buildings: list[Building] = field(default_factory=list)
    crs_epsg: int | None = None
    source_format: str = 'unknown'
    skipped_buildings: list[dict[str, str]] = field(default_factory=list)

    @property
    def building_count(self) -> int:
        ...

    @property
    def total_vertices(self) -> int:
        ...

    @property
    def total_triangles(self) -> int:
        ...

    @property
    def max_lod(self) -> int:
        ...

    def bounds(self) -> tuple[float, float, float, float, float, float] | None:
        ...

def infer_roof_type(properties: dict[str, Any]) -> str:
    ...

def material_from_tags(tags: dict[str, Any]) -> BuildingMaterial:
    ...

def material_from_name(name: str) -> BuildingMaterial:
    ...

def add_buildings(geojson_path: str | Path, *, default_height: float=10.0, height_key: str | None=None, name: str | None=None) -> BuildingLayer:
    ...

def add_buildings_cityjson(cityjson_path: str | Path, *, name: str | None=None) -> BuildingLayer:
    ...

def add_buildings_3dtiles(tileset_path: str | Path, *, name: str | None=None) -> BuildingLayer:
    ...

def validate_building_layer_support(layer: BuildingLayer, *, layer_id: str | None=None) -> ValidationReport:
    ...
__all__ = ['Building', 'BuildingLayer', 'BuildingMaterial', 'add_buildings', 'add_buildings_cityjson', 'add_buildings_3dtiles', 'validate_building_layer_support', 'infer_roof_type', 'material_from_tags', 'material_from_name']
