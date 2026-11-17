from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .observation_time import ObservationRecord
from .preparation import PreparationGrid

class RasterRecipeSource:
    path: str | Path
    source_id: str
    sha256: str
    observation: ObservationRecord | None
    def __init__(
        self,
        path: str | Path,
        source_id: str,
        sha256: str,
        observation: ObservationRecord | None = ...,
    ) -> None: ...
    @classmethod
    def from_file(
        cls,
        path: str | Path,
        source_id: str,
        *,
        observation: ObservationRecord | None = ...,
    ) -> RasterRecipeSource: ...
    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RasterRecipeSource: ...

class RasterRecipe:
    def __init__(
        self,
        a: RasterRecipeSource,
        b: RasterRecipeSource,
        mask: RasterRecipeSource,
        grid: PreparationGrid,
        bins: Sequence[float],
        palette: Sequence[Sequence[int]],
        *,
        right: bool = ...,
        quantity: str = ...,
    ) -> None: ...
    @property
    def grid(self) -> PreparationGrid: ...
    @property
    def identity(self) -> str: ...
    def legend(self) -> dict[str, Any]: ...
    def set_bins(self, bins: Sequence[float]) -> None: ...
    def replace_inputs(
        self, a: RasterRecipeSource, b: RasterRecipeSource, mask: RasterRecipeSource
    ) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...
    def to_json(self) -> str: ...
    @classmethod
    def from_json(cls, data: str) -> RasterRecipe: ...
    def sample(self, row: int, column: int) -> tuple[float | None, int]: ...
    def counts(self) -> dict[int, int]: ...
    def export_cells(self) -> tuple[np.ndarray, np.ndarray]: ...
    def metrics(self) -> dict[str, Any]: ...
    def close(self) -> None: ...
