from __future__ import annotations

from typing import Any, Sequence


class GraticuleSpec:
    bounds: Sequence[float] | None
    interval_deg: float
    target_crs: str
    include_labels: bool
    precision: int
    line_steps: int
    def __init__(self, bounds: Sequence[float] | None = ..., interval_deg: float = ..., target_crs: str = ..., include_labels: bool = ..., precision: int = ..., line_steps: int = ...) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...


def generate_graticule(
    bounds: Sequence[float] | GraticuleSpec,
    *,
    interval_deg: float | None = ...,
    target_crs: str | None = ...,
    include_labels: bool | None = ...,
    precision: int | None = ...,
    line_steps: int | None = ...,
) -> dict[str, Any]: ...


__all__: list[str]
