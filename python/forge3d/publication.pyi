from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from .map_plate import BBox, MapPlateConfig
from .observation_time import ObservationSelection, ObservationTrack
from .map_scene import OrbitCamera

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
    camera: OrbitCamera | None = ...

@dataclass(frozen=True)
class PublicationInset:
    inset_id: str
    title: str
    bounds: BBox
    rect: tuple[int, int, int, int]

@dataclass(frozen=True)
class PublicationLayout:
    config: MapPlateConfig
    insets: Sequence[PublicationInset] = ...

@dataclass(frozen=True)
class PublicationSubject:
    subject_id: str
    title: str
    source: PublicationSource
    breaks: Sequence[float]
    palette: Sequence[Sequence[int]]

def publication_output_support() -> dict[str, Any]: ...

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
    layout: PublicationLayout | None = ...
    def render_flyover(self, frames: Sequence[PublicationExtent], directory: str | Path,
                       *, layout: PublicationLayout, base_seed: int,
                       cache: str | Path | None = ...) -> dict[str, Any]: ...
    def render_times(self, track: ObservationTrack, presentation_times: Sequence[datetime | str],
                     pages: Sequence[PublicationExtent], directory: str | Path,
                     *, cache: str | Path | None = ...) -> list[dict[str, Any]]: ...
    def render_bookends(self, first: PublicationExtent, last: PublicationExtent, directory: str | Path,
                        *, layout: PublicationLayout, cache: str | Path | None = ...) -> list[dict[str, Any]]: ...
    def render_subjects(self, subjects: Sequence[PublicationSubject], pages: Sequence[PublicationExtent],
                        directory: str | Path, *, cache: str | Path | None = ...) -> list[dict[str, Any]]: ...
    def render_series(self, pages: Sequence[PublicationExtent], directory: str | Path,
                      *, cache: str | Path | None = ...,
                      selection: ObservationSelection | None = ...) -> list[dict[str, Any]]: ...
