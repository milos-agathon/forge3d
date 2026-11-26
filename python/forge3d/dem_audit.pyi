from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
from PIL.Image import Image

from .diagnostics import Diagnostic

class AuditParameterError(ValueError):
    diagnostic: Diagnostic
    def __init__(self, parameter: str, requirement: str) -> None: ...

class AuditReport:
    findings: list[dict[str, Any]]
    metadata: dict[str, Any]
    hillshade: Image
    def __init__(
        self, findings: list[dict[str, Any]], metadata: dict[str, Any], hillshade: Image
    ) -> None: ...
    def save(self, stem: str | Path) -> None: ...

def audit(
    dem: np.ndarray | str | Path,
    transform: Any = ...,
    *,
    valid_mask: np.ndarray | None = ...,
    nodata: float | None = ...,
    band: int = ...,
    spike_multiple: float = ...,
    seam_multiple: float = ...,
    seam_length: int = ...,
    stripe_fraction: float = ...,
    quantized_fraction: float = ...,
    flat_cells: int = ...,
) -> AuditReport: ...
def fill_missing(
    dem: np.ndarray, *, region: Mapping[str, Any], method: str = ...
) -> tuple[np.ma.MaskedArray, dict[str, Any]]: ...
def remove_spikes(
    dem: np.ndarray, *, region: Mapping[str, Any]
) -> tuple[np.ma.MaskedArray, dict[str, Any]]: ...
def notch_filter(
    dem: np.ndarray,
    *,
    region: Mapping[str, Any],
    frequencies: Sequence[float],
    axis: str,
) -> tuple[np.ma.MaskedArray, dict[str, Any]]: ...
def dequantize(
    dem: np.ndarray, *, region: Mapping[str, Any], step: float
) -> tuple[np.ma.MaskedArray, dict[str, Any]]: ...
