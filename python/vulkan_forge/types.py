# T01-BEGIN:types
from __future__ import annotations
from typing import Protocol, Any

class SupportsPNG(Protocol):
    def render_png(self, path: str) -> Any: ...

# Public re-export set is defined in __init__.__all__
# T01-END:types