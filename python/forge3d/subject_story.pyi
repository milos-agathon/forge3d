from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

from .camera_rigs import TerrainOrbitRig
from .chronos import FlythroughManifest
from .map_scene import LightingPreset, MapScene, OutputSpec, TerrainSource

@dataclass(frozen=True)
class SubjectFeature:
    feature_id: str
    name: str
    easting: float
    northing: float
    line_xy: tuple[tuple[float, float], ...] = ...
    area_triangles_xy: tuple[tuple[tuple[float, float], ...], ...] = ...

@dataclass(frozen=True)
class StoryObstacle:
    obstacle_id: str
    kind: Literal["building", "vegetation"]
    bounds_xyz_m: tuple[float, float, float, float, float, float]

@dataclass(frozen=True)
class StoryAnnotation:
    text: str
    anchor: tuple[float, float]
    font_size: float

class SubjectShotInfeasible(ValueError): ...

class CompiledSubjectShot:
    @property
    def fps(self) -> int: ...
    @property
    def reveal_frames(self) -> int: ...
    @property
    def frame_count(self) -> int: ...
    @property
    def key_frames(self) -> dict[str, int]: ...
    @property
    def frame_checks(self) -> tuple[Mapping[str, Any], ...]: ...
    def frame_scene(self, frame_index: int) -> MapScene: ...
    def render(
        self, out_dir: str | Path, *, base_seed: int, certificate: bool = ...,
        cache: str | PathLike[str] | None = ...,
    ) -> FlythroughManifest: ...

def compile_subject_shot(
    terrain: TerrainSource, features: Sequence[SubjectFeature], *, subject_id: str,
    bounds_wsen: Sequence[float], feature_crs: str,
    orbit: Callable[[tuple[float, float]], TerrainOrbitRig],
    fps: int, hold_seconds: float, output: OutputSpec, lighting: LightingPreset,
    framing: Sequence[float], reserved_text: Sequence[float],
    reveal_annotation: StoryAnnotation, hold_annotation: StoryAnnotation | None = ...,
    must_see: Sequence[str] = ..., establish_must_see: Sequence[str] = ...,
    obstacles: Sequence[StoryObstacle] = ...,
) -> CompiledSubjectShot: ...
