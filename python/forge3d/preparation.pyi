from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
from .diagnostics import Diagnostic
from .map_scene import MapScene

class PreparationError(ValueError):
    diagnostic: Diagnostic
    def __init__(self, code: str, message: str, remediation: str, source_id: str | None = ...) -> None: ...

class PreparationGrid:
    crs: str
    width: int
    height: int
    geotransform: Sequence[float]
    def __init__(self, crs: str, width: int, height: int, geotransform: Sequence[float]) -> None: ...
    @property
    def transform(self) -> Any: ...
    def to_dict(self) -> dict[str, Any]: ...
    def same_as(self, other: PreparationGrid) -> bool: ...
    def xy(self, column: float, row: float) -> tuple[float, float]: ...

class PreparationSource:
    source_id: str
    path: str | Path
    kind: str
    crs: str | None
    grid: PreparationGrid | None
    units: str | None
    vertical_reference: str | None
    nodata: float | int | None
    mask_path: str | Path | None
    resampling: str | None
    metadata: Mapping[str, Any]
    def __init__(self, source_id: str, path: str | Path, kind: str, crs: str | None = ..., grid: PreparationGrid | None = ..., units: str | None = ..., vertical_reference: str | None = ..., nodata: float | int | None = ..., mask_path: str | Path | None = ..., resampling: str | None = ..., metadata: Mapping[str, Any] = ...) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...

class PreparationRecipe:
    target_grid: PreparationGrid
    sources: Sequence[PreparationSource]
    vertical_reference: str
    checkpoints: Sequence[Mapping[str, Any]]
    display_styles: Mapping[str, Mapping[str, Any]]
    def __init__(self, target_grid: PreparationGrid, sources: Sequence[PreparationSource], vertical_reference: str, checkpoints: Sequence[Mapping[str, Any]] = ..., display_styles: Mapping[str, Mapping[str, Any]] = ...) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PreparationRecipe: ...
    def save(self, path: str | Path) -> None: ...
    @classmethod
    def load(cls, path: str | Path) -> PreparationRecipe: ...

class PreparedLayer:
    source_id: str
    kind: str
    grid: PreparationGrid | None
    data: np.ndarray | None
    valid_mask: np.ndarray | None
    features: list[dict[str, Any]] | None
    metadata: dict[str, Any]
    def __init__(self, source_id: str, kind: str, grid: PreparationGrid | None, data: np.ndarray | None, valid_mask: np.ndarray | None, features: list[dict[str, Any]] | None, metadata: dict[str, Any]) -> None: ...

class PreparedScene:
    recipe: PreparationRecipe
    layers: dict[str, PreparedLayer]
    report: dict[str, Any]
    root: Path | None
    def __init__(self, recipe: PreparationRecipe, layers: dict[str, PreparedLayer], report: dict[str, Any], root: Path | None = ...) -> None: ...
    def save(self, directory: str | Path) -> Path: ...
    @classmethod
    def load(cls, directory: str | Path) -> PreparedScene: ...
    def coverage_preview(self, path: str | Path) -> Path: ...
    def to_scene(self, *, lighting: Any, output: Any, camera: Any = ..., styles: Mapping[str, Mapping[str, Any]] | None = ..., terrain_void_fill: float | None = ..., assets_dir: str | Path | None = ...) -> MapScene: ...
    def viewer_assets(self, directory: str | Path, *, styles: Mapping[str, Mapping[str, Any]] | None = ..., terrain_void_fill: float | None = ...) -> dict[str, Any]: ...

def prepare_scene(recipe: PreparationRecipe, *, base_dir: str | Path = ...) -> PreparedScene: ...
