from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

pytest.importorskip("forge3d")

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_PATH = REPO_ROOT / "examples" / "luxembourg_rail_overlay.py"


def _load_example_module():
    spec = importlib.util.spec_from_file_location("luxembourg_rail_overlay", EXAMPLE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_add_linestring_as_quads_uses_dem_height_for_z_span() -> None:
    example = _load_example_module()
    vertices: list[list[float]] = []
    indices: list[int] = []

    example._add_linestring_as_quads(
        [(0.0, 20.0), (0.0, 0.0)],
        [1.0, 0.0, 0.0, 1.0],
        vertices,
        indices,
        0.0,
        10.0,
        0.0,
        20.0,
        100.0,
        40.0,
        2.0,
    )

    zs = [vertex[2] for vertex in vertices]
    assert min(zs) == pytest.approx(0.0)
    assert max(zs) == pytest.approx(40.0)
    assert indices == [0, 1, 2, 1, 3, 2]
