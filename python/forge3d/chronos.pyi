from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .map_scene import MapScene

__all__: list[str]

class FlythroughManifest:
    base_seed: int
    samples: int
    frames: tuple[Mapping[str, Any], ...]
    schema: str
    def __init__(
        self,
        base_seed: int,
        samples: int,
        frames: Sequence[Mapping[str, Any]] = ...,
        schema: str = ...,
    ) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...
    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> FlythroughManifest: ...
    @staticmethod
    def load(path: str | Path) -> FlythroughManifest: ...
    def save(self, path: str | Path) -> None: ...
    def frame_record(self, frame_index: int) -> Mapping[str, Any]: ...
    def replay_frame(
        self, scene: MapScene, frame_index: int, out_path: str | Path
    ) -> Mapping[str, Any]: ...

def render_flythrough(
    camera_path: Sequence[MapScene] | Mapping[int, MapScene],
    *,
    base_seed: int,
    samples: int,
    out_dir: str | Path,
    certificate: bool = ...,
    cache: str | os.PathLike[str] | None = ...,
) -> FlythroughManifest: ...
