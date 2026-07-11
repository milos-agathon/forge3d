"""MapScene showcase using datasets bundled with forge3d.

The example covers the public contracts from specs 001-004 in one small
workflow: structured diagnostics, label planning, deterministic render, and
review-bundle save. It intentionally uses bundled data so it can run offline.
"""

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
from forge3d.datasets import mini_dem_path, sample_boundaries_path
from forge3d.label_plan import KeepoutRegion, PriorityClass


def _load_boundaries() -> list[dict[str, Any]]:
    payload = json.loads(sample_boundaries_path().read_text(encoding="utf-8"))
    features = []
    for index, feature in enumerate(payload["features"]):
        item = dict(feature)
        item["id"] = f"boundary-{index}"
        features.append(item)
    return features


def _label_point(feature: dict[str, Any]) -> tuple[float, float, float]:
    ring = feature["geometry"]["coordinates"][0]
    x = sum(point[0] for point in ring[:-1]) / max(1, len(ring) - 1)
    y = sum(point[1] for point in ring[:-1]) / max(1, len(ring) - 1)
    return (x * 256.0, (1.0 - y) * 256.0, 0.0)


def _labels_for_boundaries(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels = []
    for feature in features:
        name = str(feature["properties"]["name"])
        label_id = "label-" + name.lower().replace(" ", "-")
        labels.append(
            {
                "id": label_id,
                "kind": "point",
                "text": name,
                "geometry": {"type": "Point", "coordinates": _label_point(feature)},
                "priority_class": "areas",
            }
        )
    labels.append(
        {
            "id": "label-title-collision",
            "kind": "point",
            "text": "Overview",
            "geometry": {"type": "Point", "coordinates": (24.0, 24.0, 0.0)},
            "priority_class": "areas",
        }
    )
    return labels


def build_scene(output_dir: str | Path) -> f3d.MapScene:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features = _load_boundaries()
    labels = _labels_for_boundaries(features)
    glyphs = sorted(set("".join(str(label["text"]) for label in labels)))

    return f3d.MapScene(
        terrain=f3d.TerrainSource(
            path=str(mini_dem_path()),
            crs="EPSG:3857",
            metadata={
                "width": 256,
                "height": 256,
                "source_id": "forge3d.datasets.mini_dem",
            },
            elevation_sampling_available=True,
        ),
        camera=f3d.OrbitCamera(
            target=(0.5, 0.5, 0.0),
            distance=950.0,
            azimuth_deg=32.0,
            elevation_deg=48.0,
        ),
        lighting=f3d.LightingPreset(name="daylight", intensity=1.15),
        output=f3d.OutputSpec(
            width=256,
            height=256,
            format="png",
            path=str(output_dir / "bundled_datasets_showcase.png"),
        ),
        map_furniture=f3d.MapFurnitureLayer(
            title="Bundled dataset showcase",
            keepouts=[KeepoutRegion("title", "title", (0, 0, 72, 64), priority=100)],
            legend={"items": ["mini_dem", "sample_boundaries"]},
        ),
        reproducibility_profile=f3d.ReproducibilityProfile(
            seed=1004,
            asset_hashes_or_ids={
                "terrain": "forge3d.datasets.mini_dem",
                "vector": "forge3d.datasets.sample_boundaries",
            },
        ),
        layers=[
            f3d.VectorOverlay(
                layer_id="sample_boundaries",
                path=str(sample_boundaries_path()),
                features=features,
                crs="EPSG:3857",
                style={
                    "version": 8,
                    "layers": [
                        {
                            "id": "sample-boundaries-fill",
                            "type": "fill",
                            "paint": {"fill-color": "#4d8c57", "fill-opacity": 0.72},
                        },
                        {
                            "id": "sample-boundaries-outline",
                            "type": "line",
                            "paint": {"line-color": "#f7f7f7", "line-width": 1.5},
                        },
                    ],
                },
                metadata={"source_id": "forge3d.datasets.sample_boundaries"},
            ),
            f3d.LabelLayer(
                layer_id="boundary_labels",
                labels=labels,
                glyph_atlas={"glyphs": glyphs},
                priority_rules=[PriorityClass("areas", rank=10)],
                metadata={"source_id": "forge3d.datasets.sample_boundaries.labels"},
            ),
        ],
    )


def _codes(report: f3d.ValidationReport) -> list[str]:
    return [diagnostic.code for diagnostic in report.diagnostics]


def _render_with_optional_placeholder(scene: f3d.MapScene) -> f3d.ValidationReport:
    # SUTURA: MapScene renders natively or raises MapSceneNativeUnavailable;
    # the CPU placeholder escape hatch no longer exists.
    return scene.render()


def run_example(output_dir: str | Path) -> dict[str, Any]:
    scene = build_scene(output_dir)
    validation = scene.validate()
    plan = scene.compiled_label_plans["boundary_labels"]

    render_status = "not_requested"
    render_support: dict[str, str] = {}
    png_path = scene.recipe.output.path
    try:
        render = _render_with_optional_placeholder(scene)
        render_status = render.status
        render_support = dict(render.supported_features)
        png_path = scene.last_render_path
    except RuntimeError:
        render_status = "blocked_by_diagnostics"

    bundle_path = Path(output_dir) / "bundled_datasets_showcase.forge3d"
    bundle = scene.save_bundle(bundle_path)
    support_levels = dict(validation.supported_features)
    support_levels.update(render_support)
    support_levels.update(bundle.supported_features)

    return {
        "validation_status": validation.status,
        "render_status": render_status,
        "bundle_status": bundle.status,
        "dataset_names": ["mini_dem", "sample_boundaries"],
        "diagnostic_codes": _codes(bundle),
        "accepted_label_ids": [label.label_id for label in plan.accepted],
        "rejected_label_reasons": {label.label_id: label.reason for label in plan.rejected},
        "support_levels": support_levels,
        "png_path": str(png_path),
        "bundle_path": str(bundle_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the bundled dataset MapScene showcase.")
    parser.add_argument("--output-dir", default="examples/out/mapscene_bundled_datasets_showcase")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = run_example(args.output_dir)
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print("datasets:", ", ".join(payload["dataset_names"]))
        print("validation:", payload["validation_status"], payload["diagnostic_codes"])
        print("render:", payload["render_status"], payload["png_path"])
        print("bundle:", payload["bundle_status"], payload["bundle_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
