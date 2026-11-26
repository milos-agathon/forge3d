from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
from .map_scene import SceneRecipe
from .raster_style import CategoricalRasterStyle, RasterStyleResult

class MaterialGrid:
    crs: str
    origin: tuple[float, float]
    spacing: tuple[float, float]
    units: str
    elevation_units: str
    def __init__(self, crs: str, origin: tuple[float, float], spacing: tuple[float, float], units: str, elevation_units: str) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...

class MaterialTexture:
    path: str
    repeat_m: tuple[float, float]
    color_space: str
    channel: str
    anchor: tuple[float, float]
    normal_convention: str | None
    variation_seed: int | None
    def __init__(self, path: str, repeat_m: tuple[float, float], color_space: str, channel: str, anchor: tuple[float, float] = ..., *, normal_convention: str | None = ..., variation_seed: int | None = ...) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...
    def sample(self, grid: MaterialGrid, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]: ...
    def sample_points(self, x: Any, y: Any, *, footprint_m: tuple[float, float] | None = ...) -> tuple[np.ndarray, np.ndarray]: ...
    @property
    def resolved_path(self) -> Path: ...

class MaterialMask:
    operation: str
    field: str | None
    units: str | None
    lower: float | None
    upper: float | None
    class_ids: tuple[int, ...]
    children: tuple[MaterialMask, ...]
    def __init__(self, operation: str = ..., field: str | None = ..., units: str | None = ..., lower: float | None = ..., upper: float | None = ..., class_ids: tuple[int, ...] = ..., children: tuple[MaterialMask, ...] = ...) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MaterialMask: ...
    def evaluate(self, fields: Mapping[str, tuple[np.ndarray, str]], classes: np.ndarray | None, valid: np.ndarray) -> np.ndarray: ...

class TerrainMaterialLayer:
    name: str
    albedo: tuple[int, int, int]
    mask: MaterialMask
    texture: MaterialTexture | None
    opacity: float
    normal: MaterialTexture | None
    roughness: float | None
    roughness_texture: MaterialTexture | None
    def __init__(self, name: str, albedo: tuple[int, int, int], mask: MaterialMask = ..., texture: MaterialTexture | None = ..., opacity: float = ..., normal: MaterialTexture | None = ..., roughness: float | None = ..., roughness_texture: MaterialTexture | None = ...) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...

    def textures(self) -> tuple[MaterialTexture, ...]: ...

class TerrainMaterialRecipe:
    layers: tuple[TerrainMaterialLayer, ...]
    presentation: str
    relief_strength: float
    relief_azimuth: float
    relief_altitude: float
    categorical: CategoricalRasterStyle | None
    relief_mode: str
    def __init__(self, layers: tuple[TerrainMaterialLayer, ...], presentation: str = ..., relief_strength: float = ..., relief_azimuth: float = ..., relief_altitude: float = ..., categorical: CategoricalRasterStyle | None = ..., relief_mode: str = ...) -> None: ...
    @classmethod
    def land_cover(cls, categorical: CategoricalRasterStyle, *, albedo: Sequence[int], texture: MaterialTexture | None = ..., relief_strength: float = ...) -> TerrainMaterialRecipe: ...
    @classmethod
    def rock_snow(cls, ground: TerrainMaterialLayer, rock: TerrainMaterialLayer, snow: TerrainMaterialLayer, *, rock_slope_degrees: tuple[float, float], snow_elevation_m: tuple[float, float], snow_slope_degrees: tuple[float, float], presentation: str = ..., categorical: CategoricalRasterStyle | None = ..., relief_strength: float = ..., relief_mode: str = ...) -> TerrainMaterialRecipe: ...
    @property
    def has_surface_channels(self) -> bool: ...
    def resolved(self) -> TerrainMaterialRecipe: ...
    def with_layer(self, name: str, **overrides: Any) -> TerrainMaterialRecipe: ...
    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TerrainMaterialRecipe: ...
    def save(self, path: str | Path) -> Path: ...
    @classmethod
    def load(cls, path: str | Path) -> TerrainMaterialRecipe: ...
    def prepare(self, elevations: Any, grid: MaterialGrid, *, classes: Any = ..., fields: Mapping[str, tuple[Any, str]] | None = ..., valid_mask: Any = ..., derivative_halo: Any = ...) -> PreparedTerrainMaterial: ...
    def bind(self, scene: SceneRecipe, grid: MaterialGrid, *, classes: Any = ..., fields: Mapping[str, tuple[Any, str]] | None = ..., derivative_halo: Any = ...) -> SceneRecipe: ...

class PreparedTerrainMaterial:
    rgba: np.ndarray
    valid_mask: np.ndarray
    slope: np.ndarray
    aspect: np.ndarray
    masks: dict[str, np.ndarray]
    thematic_mask: np.ndarray
    legend: dict[str, Any] | None
    recipe: TerrainMaterialRecipe
    grid: MaterialGrid
    normals: np.ndarray | None
    roughness: np.ndarray | None
    def save_raster(self, path: str | Path) -> Path: ...

class TerrainMaterialStyle:
    recipe: TerrainMaterialRecipe
    grid: MaterialGrid
    classes: Any
    fields: Mapping[str, tuple[Any, str]]
    derivative_halo: Any
    def __init__(self, recipe: TerrainMaterialRecipe, grid: MaterialGrid, classes: Any = ..., fields: Mapping[str, tuple[Any, str]] | None = ..., derivative_halo: Any = ...) -> None: ...
    def prepare(self, source: Any) -> PreparedTerrainMaterial: ...
    def apply(self, source: Any, *, valid_mask: Any = ...) -> RasterStyleResult: ...
    def validate_scene(self, scene: SceneRecipe) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TerrainMaterialStyle: ...
