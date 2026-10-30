from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from .map_plate import BBox, MapPlateConfig

@dataclass(frozen=True)
class PublicationSource:
    path: str | Path
    name: str
    year: str
    edition: str
    uri: str
    units: str
    missing_label: str
    provenance: Mapping[str, Any] = ...
    def description(self) -> dict[str, Any]: ...

@dataclass(frozen=True)
class PublicationExtent:
    page_id: str
    title: str
    bounds: BBox

@dataclass
class PublicationRecipe:
    source: PublicationSource
    breaks: Sequence[float]
    palette: Sequence[Sequence[int]]
    locator_bounds: BBox
    config: MapPlateConfig = ...
    missing_color: Sequence[int] = ...
    right: bool = ...
    camera_mode: str = ...
    def render_series(self, pages: Sequence[PublicationExtent], directory: str | Path,
                      *, cache: str | Path | None = ...) -> list[dict[str, Any]]: ...
