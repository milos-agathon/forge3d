#!/usr/bin/env python3
"""Execute/revise a C05 preparation recipe; optionally render or export Landscape.

Run ``python examples/prepare_scene.py --help``. Inputs remain local. Import the
real rasterio before the example shim to avoid the repository's rasterio stub.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import rasterio  # noqa: F401 -- must precede _import_shim
from _import_shim import ensure_repo_import
ensure_repo_import()
import forge3d as f3d


def reject_constant(value):
    raise ValueError(f"Non-finite JSON constant {value}")


def json_input(path: Path, code: str, build):
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object")
        result = build(value)
        json.dumps(result.to_dict() if hasattr(result, "to_dict") else result, allow_nan=False)
        return result
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise f3d.PreparationError(code, f"Invalid JSON input {path}: {exc}", "Correct the JSON fields and finite values before retrying.") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path, help="PreparationRecipe JSON; source paths relative to this file")
    parser.add_argument("output", type=Path, help="New directory for prepared snapshot, report and recipe")
    parser.add_argument("--render", type=Path, help="Optional native MapScene still PNG; prepared rasters are draped on the terrain in their geographic position")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--camera", type=Path, help="OrbitCamera keyword arguments in JSON")
    parser.add_argument("--terrain-void-fill", type=float, help="Explicit render fill in metres")
    parser.add_argument("--landscape", type=Path, help="Optional new Landscape package directory")
    parser.add_argument("--layout", nargs=4, type=int, metavar=("QUADS", "SECTIONS", "COMPONENTS_X", "COMPONENTS_Y"))
    parser.add_argument("--origin", nargs=3, type=float, metavar=("EASTING", "NORTHING", "HEIGHT_M"))
    parser.add_argument("--categorical-source")
    parser.add_argument("--material-layers", type=Path, help="JSON object mapping class IDs to Unreal layer names")
    args = parser.parse_args()
    if args.render and (args.width is None or args.height is None or args.camera is None):
        parser.error("--render requires --width, --height and --camera")
    if args.landscape and any(v is None for v in (args.layout, args.origin, args.categorical_source, args.material_layers)):
        parser.error("--landscape requires --layout, --origin, --categorical-source and --material-layers")
    if args.output.exists():
        raise f3d.PreparationError("output_exists", str(args.output), "Choose a new output directory; existing files are never overwritten.")
    if args.landscape is not None and args.landscape.exists():
        raise f3d.PreparationError("output_exists", str(args.landscape), "Choose a new Landscape package directory; existing files are never overwritten.")
    camera = json_input(args.camera, "invalid_camera", lambda value: f3d.OrbitCamera(**value)) if args.render else None
    layers = json_input(args.material_layers, "invalid_material_layers", lambda value: {int(k): v for k, v in value.items()}) if args.landscape else None
    prepared = f3d.prepare_scene(f3d.PreparationRecipe.load(args.recipe))
    prepared.save(args.output)
    prepared.coverage_preview(args.output / "coverage.html")
    if args.render:
        scene = prepared.to_scene(lighting=f3d.LightingPreset(), output=f3d.OutputSpec(args.width, args.height),
                                  camera=camera, terrain_void_fill=args.terrain_void_fill)
        report = scene.render(str(args.render))
        (args.output / "render-report.json").write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        scene.save_bundle(args.output / "scene.bundle")
    if args.landscape:
        f3d.export_landscape(prepared, args.landscape, layout=f3d.LandscapeLayout(*args.layout),
                             origin=args.origin, categorical_source=args.categorical_source,
                             material_layers=layers, void_fill_m=args.terrain_void_fill)
    print(f"Prepared {len(prepared.layers)} inputs: {args.output.resolve()}")
    print(f"Coverage and operation history: {(args.output / 'coverage.html').resolve()}")


if __name__ == "__main__":
    try:
        main()
    except f3d.PreparationError as exc:
        print(json.dumps(vars(exc.diagnostic), ensure_ascii=True), file=sys.stderr)
        raise SystemExit(1) from None
