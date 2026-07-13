from pathlib import Path
from typing import Any, Mapping, Sequence

class ShapedText:
    @property
    def text(self) -> str: ...
    @property
    def size(self) -> float: ...
    def to_dict(self) -> dict[str, Any]: ...

class TextShapingError(ValueError):
    diagnostics: list[dict[str, Any]]

class TextRenderingDeferred(NotImplementedError):
    diagnostics: list[dict[str, Any]]

def shape(
    text: str,
    font_chain: Sequence[str | Path],
    size: float,
    script: str | None = ...,
    language: str | None = ...,
    features: Mapping[str, bool] | None = ...,
) -> ShapedText: ...
def rasterize_shaped_run(*args: Any, **kwargs: Any) -> Any: ...
def bake_msdf_atlas(*args: Any, **kwargs: Any) -> Any: ...
