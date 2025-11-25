# tests/test_p6_sunrise_to_noon_example.py
# Light smoke test for the P6 terrain sunrise-to-noon Python example.
# Ensures the example runs at low resolution, loads the DEM, and writes
# a small sequence of frames without requiring a GPU or heavy resources.

from __future__ import annotations

import importlib.util
import types
from pathlib import Path

import pytest


def _load_module_by_path(path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


@pytest.mark.parametrize("steps,size", [(3, (160, 90))])
def test_terrain_sunrise_to_noon_smoke(tmp_path: Path, steps: int, size: tuple[int, int]) -> None:
    repo = Path(__file__).resolve().parents[1]
    example_path = repo / "examples" / "terrain_demo.py"
    assert example_path.exists(), "terrain_demo.py not found"

    mod = _load_module_by_path(example_path)
    assert hasattr(mod, "render_sunrise_to_noon_sequence"), "terrain_demo.py is missing render_sunrise_to_noon_sequence()"

    dem = repo / "assets" / "Gore_Range_Albers_1m.tif"
    hdr = repo / "assets" / "snow_field_4k.hdr"
    outdir = tmp_path / "sunrise_seq"

    frames = mod.render_sunrise_to_noon_sequence(
        dem_path=dem,
        hdr_path=hdr,
        output_dir=outdir,
        size=size,
        steps=int(steps),
    )

    # Expect one frame per step
    assert len(frames) == steps

    for path in frames:
        # Either PNG or (rare) .npy fallback
        assert path.exists() or path.with_suffix(".npy").exists()

        # Paths should live under the requested output directory
        assert path.parent == outdir
