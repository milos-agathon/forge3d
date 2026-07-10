"""Canonical MapScene terrain plus raster MVP example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from _import_shim import ensure_repo_import

ensure_repo_import()

import forge3d as f3d
from forge3d.helpers.offscreen import save_png_deterministic


def _write_example_assets(output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    terrain_path = output_dir / "terrain.npy"
    raster_path = output_dir / "ortho.png"

    heightmap = np.linspace(0.0, 1.0, 16 * 12, dtype=np.float32).reshape(12, 16)
    np.save(terrain_path, heightmap)

    yy, xx = np.mgrid[0:64, 0:96]
    raster = np.empty((64, 96, 4), dtype=np.uint8)
    raster[..., 0] = (32 + xx * 2).astype(np.uint8)
    raster[..., 1] = (96 + yy * 2).astype(np.uint8)
    raster[..., 2] = 144
    raster[..., 3] = 255
    save_png_deterministic(raster_path, raster)
    return terrain_path, raster_path


def build_scene(output_dir: Path) -> f3d.MapScene:
    terrain_path, raster_path = _write_example_assets(output_dir)
    return f3d.MapScene(
        terrain=f3d.TerrainSource(
            path=str(terrain_path),
            crs="EPSG:32610",
            metadata={"width": 16, "height": 12, "source_id": "terrain-raster-dem"},
            elevation_sampling_available=True,
        ),
        camera=f3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=900.0, azimuth_deg=25.0),
        lighting=f3d.LightingPreset(name="daylight", intensity=1.1),
        output=f3d.OutputSpec(width=96, height=64, format="png", path=str(output_dir / "terrain_raster.png")),
        reproducibility_profile=f3d.ReproducibilityProfile(seed=404),
        layers=[
            f3d.RasterOverlay(
                layer_id="ortho",
                path=str(raster_path),
                crs="EPSG:32610",
                opacity=0.8,
                metadata={"width": 96, "height": 64, "source_id": "terrain-raster-ortho"},
            )
        ],
    )


def _codes(report: f3d.ValidationReport) -> list[str]:
    return [diagnostic.code for diagnostic in report.diagnostics]


def run_example(output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scene = build_scene(output_dir)
    validation = scene.validate()
    render_status = "not_requested"
    png_path = scene.recipe.output.path
    try:
        render = scene.render()
        render_status = render.status
        png_path = scene.last_render_path
    except RuntimeError:
        render_status = "blocked_by_diagnostics"
    bundle_path = output_dir / "terrain_raster.forge3d"
    bundle = scene.save_bundle(bundle_path)
    return {
        "validation_status": validation.status,
        "render_status": render_status,
        "render_backend": scene.last_render_backend,
        "bundle_status": bundle.status,
        "diagnostic_codes": _codes(bundle),
        "png_path": str(png_path),
        "bundle_path": str(bundle_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the terrain+raster MapScene MVP example.")
    parser.add_argument("--output-dir", default="examples/out/mapscene_terrain_raster")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    payload = run_example(args.output_dir)
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print("validation:", payload["validation_status"])
        print("render:", payload["render_status"], payload["png_path"])
        print("bundle:", payload["bundle_status"], payload["bundle_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
