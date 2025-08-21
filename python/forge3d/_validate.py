# T01-BEGIN:validate
from __future__ import annotations
from pathlib import Path
from typing import Tuple

_MAX_DIM = 8192  # conservative guardrail for headless targets

def _as_int(name: str, v) -> int:
    try:
        i = int(v)
    except Exception as e:
        raise ValueError(f"{name} must be an integer, got {type(v).__name__}") from e
    return i

def size_wh(width, height) -> Tuple[int, int]:
    w = _as_int("width", width)
    h = _as_int("height", height)
    if w <= 0 or h <= 0:
        raise ValueError("width and height must be > 0")
    if w > _MAX_DIM or h > _MAX_DIM:
        raise ValueError(f"width/height must be <= {_MAX_DIM}")
    return w, h

def grid(n) -> int:
    g = _as_int("grid", n)
    if g < 2:
        raise ValueError("grid must be >= 2")
    if g > 4096:
        raise ValueError("grid must be <= 4096")
    return g

def png_path(p: str | Path) -> str:
    s = str(p)
    if not s.lower().endswith(".png"):
        raise ValueError("path must end with .png")
    parent = Path(s).resolve().parent
    if not parent.exists():
        raise ValueError(f"directory does not exist: {parent}")
    return s
# T01-END:validate