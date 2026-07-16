from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt

class ShapedText:
    @property
    def text(self) -> str: ...
    @property
    def size(self) -> float: ...
    def to_dict(
        self, line_ranges: Sequence[tuple[int, int]] | None = ...
    ) -> dict[str, Any]: ...
    def svg_path(
        self,
        line_ranges: Sequence[tuple[int, int]] | None = ...,
        precision: int = ...,
    ) -> str: ...
    def outline_bounds(
        self, line_ranges: Sequence[tuple[int, int]] | None = ...
    ) -> tuple[float, float, float, float] | None: ...

class TextShapingError(ValueError):
    diagnostics: list[dict[str, Any]]

def shape(
    text: str,
    font_chain: Sequence[str | Path],
    size: float,
    script: str | None = ...,
    language: str | None = ...,
    features: Mapping[str, bool] | None = ...,
) -> ShapedText: ...
def rasterize_shaped_run(
    shaped: ShapedText,
    width: int,
    height: int,
    origin: tuple[float, float] = ...,
    line_ranges: Sequence[tuple[int, int]] | None = ...,
) -> npt.NDArray[np.float32]: ...
def bake_msdf_atlas(
    font_chain: Sequence[str | Path],
    charset: str | ShapedText,
    font_size: float,
    px_range: float = ...,
    padding: int = ...,
) -> dict[str, Any]: ...
