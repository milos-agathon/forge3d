from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

from .chronos import FlythroughManifest
from .map_scene import MapScene, SceneRecipe
from .observation_time import ObservationRecord, ObservationTrack

__all__: list[str]

@dataclass(frozen=True)
class ObservationRasterStyle:
    domain: tuple[float, float]

@dataclass(frozen=True)
class ObservationEvents:
    records: Sequence[ObservationRecord]
    lookback: timedelta
    crs: str
    label: str
    style: Mapping[str, Any]

class CompiledObservationMap:
    def frame(self, index: int) -> MapScene: ...
    def selection_table(self) -> dict[str, Any]: ...
    def render(self, directory: str | Path, *, base_seed: int, samples: int = ...,
               frames: Sequence[int] | None = ..., cache: str | Path | None = ...) -> FlythroughManifest: ...

def compile_observation_map(
    recipe: SceneRecipe,
    raster_track: ObservationTrack,
    presentation_times: Sequence[datetime | str],
    *,
    raster_style: ObservationRasterStyle,
    snapshot_dir: str | Path,
    events: ObservationEvents | None = ...,
    interventions: Sequence[str] = ...,
) -> CompiledObservationMap: ...
