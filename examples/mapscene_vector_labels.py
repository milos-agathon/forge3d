"""Canonical MapScene terrain plus vector plus labels MVP example."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if PYTHON_DIR.exists():
    sys.path.insert(0, str(PYTHON_DIR))

import forge3d as f3d
from forge3d.label_plan import KeepoutRegion, PriorityClass


def _write_example_assets(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    terrain_path = output_dir / "terrain.npy"
    yy, xx = np.mgrid[0:32, 0:32]
    heightmap = ((xx + yy) / 62.0).astype(np.float32)
    np.save(terrain_path, heightmap)
    return terrain_path


def build_scene(output_dir: Path) -> f3d.MapScene:
    terrain_path = _write_example_assets(output_dir)
    return f3d.MapScene(
        terrain=f3d.TerrainSource(
            path=str(terrain_path),
            crs="EPSG:32610",
            metadata={"width": 32, "height": 32, "source_id": "vector-labels-dem"},
            elevation_sampling_available=True,
        ),
        camera=f3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=1200.0, azimuth_deg=35.0),
        lighting=f3d.LightingPreset(name="daylight", intensity=1.2),
        output=f3d.OutputSpec(width=256, height=256, format="png", path=str(output_dir / "vector_labels.png")),
        map_furniture=f3d.MapFurnitureLayer(
            title="Harbor overview",
            keepouts=[KeepoutRegion("title", "title", (0, 0, 64, 64), priority=100)],
        ),
        reproducibility_profile=f3d.ReproducibilityProfile(seed=1234),
        layers=[
            f3d.VectorOverlay(
                layer_id="roads",
                crs="EPSG:32610",
                features=[
                    {
                        "id": "road-1",
                        "geometry": {"type": "LineString", "coordinates": [(0.1, 0.2), (0.9, 0.75)]},
                        "properties": {"class": "primary"},
                    }
                ],
                style={"version": 8, "layers": [{"id": "roads", "type": "line", "paint": {"line-color": "#ffffff"}}]},
                metadata={"source_id": "roads-fixture"},
            ),
            f3d.LabelLayer(
                layer_id="labels",
                labels=[
                    {
                        "id": "city",
                        "kind": "point",
                        "text": "Alpha",
                        "geometry": {"type": "Point", "coordinates": (120.0, 120.0, 0.0)},
                        "priority_class": "cities",
                    },
                    {
                        "id": "park",
                        "kind": "polygon",
                        "text": "Park",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[(150.0, 150.0), (190.0, 150.0), (190.0, 190.0), (150.0, 190.0)]],
                        },
                        "priority_class": "parks",
                    },
                    {
                        "id": "blocked-title",
                        "kind": "point",
                        "text": "Beta",
                        "geometry": {"type": "Point", "coordinates": (24.0, 24.0, 0.0)},
                    },
                ],
                glyph_atlas={"glyphs": sorted(set("AlphaParkBeta"))},
                priority_rules=[PriorityClass("cities", rank=10), PriorityClass("parks", rank=5)],
                metadata={"source_id": "labels-fixture"},
            ),
        ],
    )


def _codes(report: f3d.ValidationReport) -> list[str]:
    return [diagnostic.code for diagnostic in report.diagnostics]


def run_example(output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scene = build_scene(output_dir)
    validation = scene.validate()
    plan = scene.compiled_label_plans["labels"]
    render_status = "not_requested"
    png_path = scene.recipe.output.path
    try:
        render = scene.render()
        render_status = render.status
        png_path = scene.last_render_path
    except RuntimeError:
        render_status = "blocked_by_diagnostics"
    bundle_path = output_dir / "vector_labels.forge3d"
    bundle = scene.save_bundle(bundle_path)
    return {
        "validation_status": validation.status,
        "render_status": render_status,
        "render_backend": scene.last_render_backend,
        "bundle_status": bundle.status,
        "diagnostic_codes": _codes(bundle),
        "accepted_label_ids": [label.label_id for label in plan.accepted],
        "rejected_label_reasons": {label.label_id: label.reason for label in plan.rejected},
        "png_path": str(png_path),
        "bundle_path": str(bundle_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the vector+labels MapScene MVP example.")
    parser.add_argument("--output-dir", default="examples/out/mapscene_vector_labels")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    payload = run_example(args.output_dir)
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print("validation:", payload["validation_status"], payload["diagnostic_codes"])
        print("render:", payload["render_status"], payload["png_path"])
        print("bundle:", payload["bundle_status"], payload["bundle_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
