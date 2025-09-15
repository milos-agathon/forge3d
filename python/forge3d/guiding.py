# python/forge3d/guiding.py
# Simple online spatial/directional guiding utilities for experiments and tests.
# This exists to satisfy A13 by providing a minimal, deterministic histogram-based guider.
# RELEVANT FILES:python/forge3d/__init__.py,src/path_tracing/guiding.rs,tests/test_guiding.py,docs/api/guiding.md

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List

import numpy as np


@dataclass
class OnlineGuidingGrid:
    width: int
    height: int
    bins_per_cell: int = 8

    def __post_init__(self) -> None:
        self.bins_per_cell = max(1, int(self.bins_per_cell))
        self._counts = np.zeros((self.height, self.width, self.bins_per_cell), dtype=np.uint32)

    def update(self, x: int, y: int, bin_index: int, weight: float = 1.0) -> None:
        x = int(np.clip(x, 0, self.width - 1))
        y = int(np.clip(y, 0, self.height - 1))
        b = int(bin_index) % self.bins_per_cell
        w = float(weight)
        if not np.isfinite(w) or w <= 0.0:
            return
        inc = 1 if w >= 1.0 else (1 if (hash((x, y, b, int(w * 1e6))) & 1) else 0)
        if inc:
            self._counts[y, x, b] = np.minimum(self._counts[y, x, b] + 1, np.iinfo(np.uint32).max)

    def pdf(self, x: int, y: int) -> np.ndarray:
        x = int(np.clip(x, 0, self.width - 1))
        y = int(np.clip(y, 0, self.height - 1))
        row = self._counts[y, x].astype(np.float32)
        s = float(row.sum())
        if s <= 0.0:
            return np.full((self.bins_per_cell,), 1.0 / float(self.bins_per_cell), dtype=np.float32)
        return row / s

    def dims(self) -> Tuple[int, int, int]:
        return (self.width, self.height, self.bins_per_cell)

