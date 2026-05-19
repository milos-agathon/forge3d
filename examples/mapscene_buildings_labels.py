"""Canonical MapScene buildings plus labels MVP diagnostic example."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if PYTHON_DIR.exists():
    sys.path.insert(0, str(PYTHON_DIR))

import forge3d as f3d
from forge3d.label_plan import PriorityClass


def build_scene(output_dir: Path) -> f3d.MapScene:
    return f3d.MapScene(
        terrain=f3d.TerrainSource(
            path="fixtures/mapscene/dem.tif",
            crs="EPSG:32610",
            metadata={"width": 16, "height": 16, "source_id": "buildings-labels-dem", "asset_status": "fixture"},
            elevation_sampling_available=True,
        ),
        camera=f3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=1400.0, azimuth_deg=20.0),
        lighting=f3d.LightingPreset(name="daylight"),
        output=f3d.OutputSpec(width=128, height=96, format="png", path=str(output_dir / "buildings_labels.png")),
        reproducibility_profile=f3d.ReproducibilityProfile(seed=505),
        layers=[
            f3d.MapSceneBuildingLayer(
                layer_id="buildings",
                source="fixtures/mapscene/buildings.geojson",
                support_level="Pro-gated",
                geometry_count=24,
                material_status="scalar_pbr",
                metadata={"source_id": "buildings-fixture"},
            ),
            f3d.LabelLayer(
                layer_id="labels",
                labels=[
                    {
                        "id": "district",
                        "kind": "point",
                        "text": "District",
                        "geometry": {"type": "Point", "coordinates": (64.0, 48.0, 0.0)},
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
    try:
        render = scene.render()
        render_status = render.status
    except RuntimeError:
        render_status = "blocked_by_diagnostics"
    bundle_path = output_dir / "buildings_labels.forge3d"
    bundle = scene.save_bundle(bundle_path)
    return {
        "validation_status": validation.status,
        "render_status": render_status,
        "bundle_status": bundle.status,
        "diagnostic_codes": _codes(bundle),
        "png_path": str(output_dir / "buildings_labels.png"),
        "bundle_path": str(bundle_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the buildings+labels MapScene MVP diagnostic example.")
    parser.add_argument("--output-dir", default="examples/out/mapscene_buildings_labels")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    payload = run_example(args.output_dir)
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print("validation:", payload["validation_status"], payload["diagnostic_codes"])
        print("render:", payload["render_status"])
        print("bundle:", payload["bundle_status"], payload["bundle_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
