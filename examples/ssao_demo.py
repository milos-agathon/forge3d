# examples/ssao_demo.py
# SSAO demo for Workstream B.
# Exists to showcase SSAO configuration and generate comparison renders.
# RELEVANT FILES:python/forge3d/__init__.py,python/forge3d/postfx.py,shaders/ssao.wgsl,tests/test_b3_ssao.py

"""Generate SSAO comparison renders using the high-level Python API."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from forge3d import Scene, postfx


def _build_heightmap(size: int) -> np.ndarray:
    """Create a smooth bowl heightmap to emphasize ambient occlusion."""
    coords = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    ys, xs = np.meshgrid(coords, coords, indexing="ij")
    bowl = np.sqrt(xs * xs + ys * ys).astype(np.float32)
    return bowl


def main() -> None:
    output_dir = Path("diag_out")
    output_dir.mkdir(parents=True, exist_ok=True)

    scene = Scene(256, 256)
    scene.set_height_from_r32f(_build_heightmap(128))

    baseline_path = output_dir / "ssao_demo_baseline.png"
    scene.set_ssao_enabled(False)
    scene.render_png(baseline_path)

    postfx.enable_ssao(scene=scene, radius=2.0, intensity=1.1, bias=0.02)
    occluded_path = output_dir / "ssao_demo_occluded.png"
    scene.render_png(occluded_path)

    print(f"Baseline written to {baseline_path}")
    print(f"SSAO render written to {occluded_path}")


if __name__ == "__main__":
    main()
