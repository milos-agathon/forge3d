from typing import Any, Literal, Mapping, Sequence
from dataclasses import dataclass
from .diagnostics import Diagnostic
from .terrain_params import VectorOverlayConfig
from .viewer_contract import WorldPosition

class VectorLineStyleError(ValueError):
    diagnostics: tuple[Diagnostic, ...]
    def __init__(self, diagnostics: Sequence[Diagnostic]) -> None: ...

@dataclass(frozen=True)
class VectorLineLayer:
    name: str
    points: Sequence[WorldPosition]
    width: float
    color: tuple[float, float, float, float]
    halo: float
    halo_color: tuple[float, float, float, float]
    cap: Literal["butt", "square"]
    join: Literal["bevel", "miter"]
    drape: bool
    z_offset: float
    opacity: float
    feature_id: int
    def __init__(
        self, name: str, points: Sequence[WorldPosition], width: float = ...,
        color: tuple[float, float, float, float] = ..., halo: float = ...,
        halo_color: tuple[float, float, float, float] = ...,
        cap: Literal["butt", "square"] = "butt",
        join: Literal["bevel", "miter"] = "miter", drape: bool = ..., z_offset: float = ...,
        opacity: float = ..., feature_id: int = ...,
    ) -> None: ...
    @classmethod
    def from_style(
        cls, layer: Any, points: Sequence[WorldPosition], *,
        properties: Mapping[str, Any] | None = ..., zoom: float = ...,
        **options: Any,
    ) -> VectorLineLayer: ...
    def to_overlay_config(self) -> VectorOverlayConfig: ...
