from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("forge3d")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_PATH = REPO_ROOT / "examples" / "terrain_camera_rigs_demo.py"


def _load_demo_module():
    spec = importlib.util.spec_from_file_location("terrain_camera_rigs_demo", DEMO_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_demo_main_applies_viewer_z_scale(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    demo = _load_demo_module()
    dem_path = tmp_path / "demo.npy"
    np.save(dem_path, np.zeros((64, 64), dtype=np.float32))

    calls: list[tuple[str, object]] = []

    class _ViewerStub:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def set_z_scale(self, value: float) -> None:
            calls.append(("set_z_scale", float(value)))

        def render_animation(self, animation, output_dir, fps=30, width=None, height=None) -> None:
            calls.append(
                (
                    "render_animation",
                    (
                        animation.keyframe_count,
                        Path(output_dir).name,
                        int(fps),
                        int(width) if width is not None else None,
                        int(height) if height is not None else None,
                    ),
                )
            )

    monkeypatch.setattr(
        demo.f3d,
        "open_viewer_async",
        lambda **kwargs: _ViewerStub(),
    )
    monkeypatch.setattr(
        demo,
        "_parse_args",
        lambda: demo.argparse.Namespace(
            preset="orbit_rainier",
            dem=dem_path,
            z_scale=2.5,
            fps=12,
            samples_per_second=8,
            width=640,
            height=360,
            export_dir=tmp_path / "frames",
            loop=False,
        ),
    )

    assert demo.main() == 0
    assert calls[0] == ("set_z_scale", 2.5)
    assert calls[1][0] == "render_animation"
