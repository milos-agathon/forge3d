from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
from .preparation import PreparationGrid, PreparationSource

class TerrainFusion:
    height: np.ndarray
    valid_mask: np.ndarray
    weights: np.ndarray
    grid: PreparationGrid
    vertical_reference: str
    source_ids: tuple[str, ...]
    overlap_statistics: dict[str, dict[str, Any]]
    record: dict[str, Any]
    def __init__(self, height: np.ndarray, valid_mask: np.ndarray, weights: np.ndarray, grid: PreparationGrid, vertical_reference: str, source_ids: tuple[str, ...], overlap_statistics: dict[str, dict[str, Any]], record: dict[str, Any]) -> None: ...
    @property
    def source_index(self) -> np.ndarray: ...
    @property
    def dominant_source_id(self) -> np.ndarray: ...
    def save(self, directory: str | Path) -> Path: ...
    @classmethod
    def load(cls, directory: str | Path) -> TerrainFusion: ...

def fuse_terrain(
    sources: Sequence[PreparationSource],
    target_grid: PreparationGrid,
    *,
    vertical_reference: str,
    surface_kinds: Mapping[str, str],
    transition_width_m: float,
    calibration_masks: Mapping[str, np.ndarray] | None = ...,
    bias_estimation_masks: Mapping[str, np.ndarray] | None = ...,
    max_bias_slope: float | None = ...,
    allow_mixed_surfaces: bool = ...,
    allow_unresolved_vertical_reference: bool = ...,
    base_dir: str | Path = ...,
) -> TerrainFusion: ...
