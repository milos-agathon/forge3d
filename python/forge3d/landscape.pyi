from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
from .preparation import PreparedScene

class LandscapeLayout:
    quads_per_section: int
    sections_per_component: int
    components_x: int
    components_y: int
    def __init__(self, quads_per_section: int, sections_per_component: int, components_x: int, components_y: int) -> None: ...
    @property
    def shape(self) -> tuple[int, int]: ...

def export_landscape(prepared: PreparedScene, directory: str | Path, *, layout: LandscapeLayout, origin: Sequence[float], categorical_source: str, material_layers: Mapping[int, str], void_fill_m: float | None = ...) -> Path: ...
def verify_landscape_import(directory: str | Path, imported_positions_cm: np.ndarray, *, editor_version: str, measurement_tolerance_m: float) -> dict[str, Any]: ...
