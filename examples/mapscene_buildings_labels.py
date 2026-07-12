"""Canonical MapScene buildings plus labels example."""

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
from forge3d.label_plan import PriorityClass


def _heightmap(size: int = 24) -> np.ndarray:
    x = np.linspace(0.0, 1.0, size, dtype=np.float32)
    y = np.linspace(0.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    return (0.18 + 0.32 * xx + 0.18 * yy + 0.08 * np.sin(xx * np.pi * 3.0)).astype(np.float32)


def _building_features() -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []
    roof_types = ("flat", "gabled", "hipped", "pyramidal")
    for idx, roof_type in enumerate(roof_types):
        x0 = 0.08 + idx * 0.22
        x1 = x0 + 0.15
        y0 = 0.26 if idx % 2 == 0 else 0.36
        y1 = y0 + 0.30
        features.append(
            {
                "id": f"building-{roof_type}",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]],
                },
                "properties": {
                    "height": 24.0 + idx * 7.0,
                    "roof:shape": roof_type,
                    "building:material": "brick" if idx % 2 else "concrete",
                },
            }
        )
    return features


def build_scene(output_dir: Path) -> f3d.MapScene:
    features = _building_features()
    return f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=_heightmap(),
            crs="EPSG:32610",
            metadata={"width": 24, "height": 24, "source_id": "buildings-labels-dem", "asset_status": "fixture"},
            elevation_sampling_available=True,
        ),
        camera=f3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=1050.0, azimuth_deg=28.0, elevation_deg=48.0),
        lighting=f3d.LightingPreset(name="daylight", intensity=1.25),
        output=f3d.OutputSpec(width=160, height=112, format="png", path=str(output_dir / "buildings_labels.png")),
        reproducibility_profile=f3d.ReproducibilityProfile(seed=505),
        layers=[
            f3d.MapSceneBuildingLayer(
                layer_id="buildings",
                source={"source_id": "inline-mixed-roof-buildings", "asset_status": "fixture"},
                support_level="supported",
                geometry_count=len(features),
                material_status="scalar_pbr_underdeveloped",
                features=features,
                metadata={"source_id": "inline-mixed-roof-buildings", "asset_status": "fixture"},
            ),
            f3d.LabelLayer(
                layer_id="labels",
                labels=[
                    {
                        "id": "district",
                        "kind": "point",
                        "text": "District",
                        "geometry": {"type": "Point", "coordinates": (78.0, 58.0, 0.0)},
                        "priority_class": "districts",
                    }
                ],
                glyph_atlas={"glyphs": sorted(set("District"))},
                priority_rules=[PriorityClass("districts", rank=10)],
                metadata={"source_id": "building-labels"},
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
    render_status = "not_requested"
    render_backend = None
    try:
        render = scene.render()
        render_status = render.status
        render_backend = scene.last_render_backend
    except RuntimeError:
        render_status = "blocked_by_diagnostics"
    bundle_path = output_dir / "buildings_labels.forge3d"
    bundle = scene.save_bundle(bundle_path)
    return {
        "validation_status": validation.status,
        "render_status": render_status,
        "render_backend": render_backend,
        "bundle_status": bundle.status,
        "diagnostic_codes": _codes(bundle),
        "building_backend": (scene.last_render_metadata or {}).get("building_backend"),
        "building_batch_count": (scene.last_render_metadata or {}).get("building_batch_count"),
        "building_batch_ids": (scene.last_render_metadata or {}).get("building_batch_ids"),
        "building_roof_types": (scene.last_render_metadata or {}).get("building_roof_types"),
        "building_shadow_model": (scene.last_render_metadata or {}).get("building_shadow_model"),
        "png_path": str(output_dir / "buildings_labels.png"),
        "bundle_path": str(bundle_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the buildings+labels MapScene example.")
    parser.add_argument("--output-dir", default="examples/out/mapscene_buildings_labels")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    payload = run_example(args.output_dir)
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print("validation:", payload["validation_status"], payload["diagnostic_codes"])
        print("render:", payload["render_status"], payload.get("building_backend"))
        print("bundle:", payload["bundle_status"], payload["bundle_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
