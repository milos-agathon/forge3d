from pathlib import Path
from typing import Mapping, Sequence

from .map_plate import MapPlateConfig
from .map_scene import OrbitCamera
from .publication import PublicationExtent, PublicationSource

class ValueFaithfulRasterBinding:
    source: PublicationSource
    extent: PublicationExtent
    measurement_type: str
    height_scale: float
    colour_edges: Sequence[float]
    palette: Sequence[Sequence[int]]
    missing_code: int
    withheld_code: int
    def __init__(self, source: PublicationSource, extent: PublicationExtent, measurement_type: str,
                 height_scale: float, colour_edges: Sequence[float], palette: Sequence[Sequence[int]],
                 missing_code: int, withheld_code: int) -> None: ...
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, data: Mapping) -> ValueFaithfulRasterBinding: ...
    def render_series(self, cameras: Mapping[str, OrbitCamera], directory: str | Path,
                      *, config: MapPlateConfig | None = ...) -> list[dict]: ...
