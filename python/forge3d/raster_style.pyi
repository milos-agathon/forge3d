from typing import Any, Mapping, Sequence
import numpy as np

class RasterStyleResult:
    rgba: np.ndarray
    valid_mask: np.ndarray
    legend: dict[str, Any]
    heights: np.ndarray | None
    classes: np.ndarray | None
    x_classes: np.ndarray | None
    y_classes: np.ndarray | None
    def __init__(self, rgba: np.ndarray, valid_mask: np.ndarray, legend: dict[str, Any], heights: np.ndarray | None = ..., classes: np.ndarray | None = ..., x_classes: np.ndarray | None = ..., y_classes: np.ndarray | None = ...) -> None: ...

class RasterHeightSurfaceStyle:
    height_scale: float
    source_units: str
    height_units: str
    nodata: float | None
    low_color: tuple[int, int, int, int]
    high_color: tuple[int, int, int, int]
    def __init__(self, height_scale: float, source_units: str, height_units: str = ..., nodata: float | None = ..., low_color: Sequence[int] = ..., high_color: Sequence[int] = ...) -> None: ...
    def apply(self, source: Any, *, valid_mask: Any = ...) -> RasterStyleResult: ...
    def to_dict(self) -> dict[str, Any]: ...

class CategoricalRasterStyle:
    palette: Mapping[int, Sequence[int]]
    labels: Mapping[int, str]
    nodata: float | None
    units: str
    def __init__(self, palette: Mapping[int, Sequence[int]], labels: Mapping[int, str] | None = ..., nodata: float | None = ..., units: str = ...) -> None: ...
    def apply(self, source: Any, *, valid_mask: Any = ...) -> RasterStyleResult: ...
    def legend(self) -> dict[str, Any]: ...
    def to_dict(self) -> dict[str, Any]: ...

class BivariateRasterStyle:
    x_bins: Sequence[float]
    y_bins: Sequence[float]
    palette: Sequence[Sequence[Sequence[int]]]
    x_label: str
    y_label: str
    x_labels: Sequence[str]
    y_labels: Sequence[str]
    x_units: str
    y_units: str
    right: bool
    x_nodata: float | None
    y_nodata: float | None
    def __init__(self, x_bins: Sequence[float], y_bins: Sequence[float], palette: Sequence[Sequence[Sequence[int]]], x_label: str, y_label: str, x_labels: Sequence[str] = ..., y_labels: Sequence[str] = ..., x_units: str = ..., y_units: str = ..., right: bool = ..., x_nodata: float | None = ..., y_nodata: float | None = ...) -> None: ...
    def apply(self, x: Any, y: Any, *, valid_mask: Any = ...) -> RasterStyleResult: ...
    def legend(self) -> dict[str, Any]: ...
    def to_dict(self) -> dict[str, Any]: ...
