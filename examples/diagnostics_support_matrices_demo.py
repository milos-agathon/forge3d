"""Showcase feature 001 diagnostics and support-matrix behavior.

This example is intentionally validation-only: it does not require a GPU,
native rendering, or Pro-gated asset paths. It demonstrates how feature 001
surfaces incomplete map-rendering paths as structured diagnostics before a
render workflow treats them as successful.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _ensure_repo_import() -> None:
    from _import_shim import ensure_repo_import

    ensure_repo_import()


def _merge_reports(*reports):
    from forge3d.diagnostics import ValidationReport

    diagnostics = []
    layer_summaries = []
    supported_features: dict[str, str] = {}
    unsupported_features: dict[str, str] = {}

    for report in reports:
        diagnostics.extend(report.diagnostics)
        layer_summaries.extend(report.layer_summaries)
        supported_features.update(report.supported_features)
        unsupported_features.update(report.unsupported_features)

    return ValidationReport(
        diagnostics=diagnostics,
        layer_summaries=layer_summaries,
        supported_features=supported_features,
        unsupported_features=unsupported_features,
    )


def _style_report():
    from forge3d.style import validate_style_support

    return validate_style_support(
        {
            "version": 8,
            "name": "diagnostics-demo-style",
            "layers": [
                {
                    "id": "land",
                    "type": "fill",
                    "paint": {"fill-color": "#9fbf8f", "fill-opacity": 0.8},
                },
                {
                    "id": "roads",
                    "type": "line",
                    "paint": {
                        "line-color": "#303030",
                        "line-width": 2.0,
                        "line-gradient": ["interpolate", ["linear"], ["get", "speed"], 0, "#00f", 80, "#f00"],
                    },
                },
                {
                    "id": "places",
                    "type": "symbol",
                    "layout": {"text-field": ["get", "name"], "text-size": 14},
                    "paint": {"text-color": "#1f2933"},
                },
                {"id": "traffic-heat", "type": "heatmap"},
            ],
        }
    )


def _building_report():
    from forge3d.buildings import Building, BuildingLayer, validate_building_layer_support

    fallback_building = Building(
        id="building-zero-geometry",
        positions=np.asarray([], dtype=np.float32),
        indices=np.asarray([], dtype=np.uint32),
        height=18.0,
        attributes={"source": "public fallback"},
    )
    layer = BuildingLayer(
        name="buildings.public-fallback",
        buildings=[fallback_building],
        crs_epsg=3857,
        source_format="geojson",
    )
    return validate_building_layer_support(layer, layer_id=layer.name)


def _tiles_report():
    from forge3d.tiles3d import BoundingVolume, Tile, TileContent, Tileset, validate_tiles3d_support

    root = Tile(
        bounding_volume=BoundingVolume("sphere", [0.0, 0.0, 0.0, 10.0]),
        geometric_error=128.0,
        content=TileContent(uri="sample.b3dm"),
    )
    tileset = Tileset(
        base_path=Path("fixtures/tiles3d"),
        version="1.0",
        geometric_error=128.0,
        root=root,
    )
    return validate_tiles3d_support(tileset, layer_id="tiles3d.local-fixture")


def _vt_report():
    from forge3d.terrain_params import TerrainVTSettings, VTLayerFamily, validate_terrain_vt_support

    settings = TerrainVTSettings(
        enabled=True,
        layers=[
            VTLayerFamily("albedo"),
            VTLayerFamily("normal"),
            VTLayerFamily("mask"),
        ],
    )
    return validate_terrain_vt_support(settings, layer_id="terrain.material-vt")


def _label_report():
    from forge3d.diagnostics import validate_label_support

    labels = [
        {"id": "city-1", "kind": "point", "text": "Cafe"},
        {"id": "road-1", "kind": "line", "text": "A1"},
        {"id": "river-curve", "kind": "curved", "text": "River"},
    ]
    glyphs = set("ABCDFRivaor 1")
    return validate_label_support(labels, atlas_glyphs=glyphs, layer_id="labels.demo")


def _manual_report():
    from forge3d.diagnostics import (
        ValidationReport,
        crs_mismatch_diagnostic,
        estimated_gpu_memory_diagnostic,
        label_rejection_summary_diagnostic,
        pro_gated_path_diagnostic,
    )

    return ValidationReport(
        diagnostics=[
            crs_mismatch_diagnostic("EPSG:4326", "EPSG:3857", layer_id="roads.source"),
            pro_gated_path_diagnostic("native CityJSON import", layer_id="buildings.native"),
            estimated_gpu_memory_diagnostic(
                estimated_bytes=5_368_709_120,
                budget_bytes=4_294_967_296,
                layer_id="terrain.large-scene",
            ),
            label_rejection_summary_diagnostic(
                {"collision": 3, "missing_glyph": 1},
                layer_id="labels.demo",
            ),
        ]
    )


def build_demo_report():
    """Build one deterministic ValidationReport covering feature 001 paths."""
    _ensure_repo_import()
    return _merge_reports(
        _style_report(),
        _building_report(),
        _tiles_report(),
        _vt_report(),
        _label_report(),
        _manual_report(),
    )


def _print_summary(payload: dict[str, Any]) -> None:
    print(f"status: {payload['status']}")
    print(f"render_blocked_continue_on_warning: {payload['render_blocked']}")
    print("diagnostics:")
    for diagnostic in payload["diagnostics"]:
        layer = diagnostic["layer_id"] or "-"
        obj = diagnostic["object_id"] or "-"
        print(
            "  "
            f"{diagnostic['severity']:7s} "
            f"{diagnostic['code']:36s} "
            f"layer={layer} object={obj}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a feature 001 diagnostics/support-matrix example report."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/out/diagnostics_support_matrices_report.json"),
        help="Path for the deterministic JSON report.",
    )
    args = parser.parse_args()

    report = build_demo_report()
    payload = report.to_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    _print_summary(payload)
    print(f"wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
