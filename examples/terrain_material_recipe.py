"""C01: one saved material recipe, new extent, native still and camera sequence.

Use a north-up projected metre DEM, already aligned categorical GeoTIFF, and a
recipe JSON. Required frame/output settings are explicit. No WGSL edits or
external renderer are needed. Import rasterio before the repository shim.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import file_digest
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import rasterio

try:
    import _import_shim  # noqa: F401
except ImportError:
    pass
import forge3d as f3d


def _encode_movie(frame_dir: Path, destination: Path, fps: int, frames: int) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(fps),
                    "-i", str(frame_dir / "frame-%06d.png"), "-frames:v", str(frames),
                    "-c:v", "libx264rgb", "-crf", "0", "-pix_fmt", "rgb24", str(destination)], check=True)


def run(args: argparse.Namespace) -> dict:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    window = rasterio.windows.Window(*args.window) if args.window else None
    with rasterio.open(args.dem) as src:
        if src.crs is None:
            raise ValueError("DEM requires a projected metre CRS")
        tagged_units = src.units[0]
        if tagged_units in ("metre", "meter", "metres", "meters"):
            tagged_units = "m"
        elevation_units = args.elevation_units or tagged_units
        if elevation_units != "m" or (tagged_units is not None and tagged_units != elevation_units):
            raise ValueError("DEM elevation units must be declared as 'm' and agree with its band metadata")
        if src.transform.b or src.transform.d or src.transform.a <= 0 or src.transform.e >= 0:
            raise ValueError("prepare a north-up, unrotated DEM before material authoring")
        heights = src.read(1, window=window, masked=True).astype(np.float32).filled(np.nan)
        transform = src.window_transform(window) if window else src.transform
        halo = None
        if window is not None:
            halo_window = rasterio.windows.Window(window.col_off - 1, window.row_off - 1, window.width + 2, window.height + 2)
            halo = src.read(1, window=halo_window, boundless=True, masked=True).astype(np.float32).filled(np.nan)
        crs = src.crs.to_string()
        source_info = {"path": str(Path(args.dem).resolve()), "crs": crs,
                       "source_credit": args.source_credit, "window": args.window,
                       "elevation_units": elevation_units, "vertical_datum": args.vertical_datum,
                       "source_transform": list(transform),
                       "source_nodata": float(src.nodata) if src.nodata is not None and np.isfinite(src.nodata) else None,
                       "source_nodata_nonfinite": src.nodata is not None and not np.isfinite(src.nodata)}
    grid = f3d.MaterialGrid(crs, transform * (.5, .5), (transform.a, -transform.e), "m", elevation_units)
    for name, path in (("dem", args.dem), ("classes", args.classes), ("recipe", args.recipe)):
        if path is not None:
            with Path(path).open("rb") as source:
                source_info[name + "_sha256"] = file_digest(source, "sha256").hexdigest()
    classes = None
    if args.classes:
        with rasterio.open(args.classes) as src:
            class_transform = src.window_transform(window) if window else src.transform
            classes = src.read(1, window=window, masked=True).astype(np.float64).filled(np.nan)
            if src.crs.to_string() != crs or class_transform != transform or classes.shape != heights.shape:
                raise ValueError("categorical data must use the DEM's CRS, affine transform and sample grid; align with C05 first")
    material = f3d.GeospatialMaterialRecipe.load(args.recipe)
    prepared = material.prepare(heights, grid, classes=classes, derivative_halo=halo)
    prepared.save_raster(output / "material.png")
    material.save(output / "recipe.json")
    span = float(np.hypot(heights.shape[1] * grid.spacing[0], heights.shape[0] * grid.spacing[1]))
    scene_recipe = f3d.SceneRecipe(
        f3d.TerrainSource(data=heights, crs=crs, metadata={"source_id": args.source_credit}, elevation_sampling_available=True),
        # Native Z-up terrain cameras take a polar angle from +Z. The CLI
        # presents the mapmaker with elevation above the geographic horizon.
        f3d.OrbitCamera(distance=span * args.camera_distance, elevation_deg=90 - args.elevation,
                       azimuth_deg=args.azimuth[0], fov_deg=args.fov),
        f3d.LightingPreset("custom", sun_direction=args.sun_direction,
                           settings={"cli_params": {"camera_mode": "mesh:zup:north"}, "exaggeration": args.exaggeration}),
        output=f3d.OutputSpec(width=args.size[0], height=args.size[1],
            samples=getattr(args, "samples", 1), hdr=getattr(args, "hdr", False),
            denoiser=getattr(args, "denoiser", "none")))
    if args.title:
        scene_recipe = replace(scene_recipe, map_furniture=f3d.MapFurnitureLayer(title=args.title))
    bound = material.bind(scene_recipe, grid, classes=classes, derivative_halo=halo)
    feature_material = (f3d.GeospatialMaterialRecipe.load(args.feature_recipe)
                        if getattr(args, "feature_recipe", None) else material)
    bound_layers = []
    for family in ("vectors", "buildings"):
        path = getattr(args, family, None)
        if not path:
            continue
        if getattr(args, "feature_crs", None) != grid.crs:
            raise ValueError("provided vector/building features must declare the prepared DEM CRS")
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        features = data.get("features")
        if data.get("type") != "FeatureCollection" or not features:
            raise ValueError("feature inputs must be nonempty prepared FeatureCollections")
        anchors = []
        for feature in features:
            coordinate = feature["geometry"]["coordinates"]
            while coordinate and isinstance(coordinate[0], list):
                coordinate = coordinate[0]
            anchors.append(coordinate)
        field_values = {name: (np.asarray([feature["properties"].get(prop) for feature in features], dtype=np.float64), units)
                        for name, prop, units in (getattr(args, "feature_field", None) or [])}
        class_values = None
        if feature_material.categorical is not None:
            key = getattr(args, "class_field", None)
            if not key:
                raise ValueError("categorical feature materials require --class-field")
            class_values = [feature["properties"][key] for feature in features]
        metadata = {"source_id": Path(path).name, "crs": grid.crs, "coordinate_space": "world"}
        if family == "vectors":
            vector_material = replace(feature_material, presentation=args.vector_presentation) if getattr(args, "vector_presentation", None) else feature_material
            layer = f3d.VectorOverlay("material-vectors", features=features, crs=grid.crs, metadata=metadata,
                                     width_world=getattr(args, "vector_width_world", None))
            layer = vector_material.bind_vector(layer, anchors, units="m", fields=field_values, classes=class_values)
        else:
            layer = f3d.MapSceneBuildingLayer("material-buildings", features=features, metadata=metadata)
            layer = feature_material.bind_buildings(layer, anchors, units="m", fields=field_values, classes=class_values,
                presentation=getattr(args, "building_presentation", "physical"))
        bound_layers.append(layer)
    if bound_layers:
        bound = feature_material.attach(bound, *bound_layers)
    scene = f3d.MapScene(recipe=bound)
    prepared_seconds = time.perf_counter() - started
    draw_started = time.perf_counter()
    scene.render(output / "still.png")
    scene.save_bundle(output / "scene.forge3d")
    frame_dir = output / "frames"
    frame_dir.mkdir(exist_ok=True)
    from forge3d.map_scene import _shared_terrain_render_context
    with _shared_terrain_render_context():
        for index, azimuth in enumerate(np.linspace(*args.azimuth, args.frames)):
            frame = f3d.MapScene(recipe=replace(bound, camera=replace(bound.camera, azimuth_deg=float(azimuth))))
            frame.render(frame_dir / f"frame-{index:06d}.png")
    render_seconds = time.perf_counter() - draw_started
    encode_seconds = None
    if args.movie:
        encode_started = time.perf_counter()
        _encode_movie(frame_dir, output / "flyover.mp4", args.fps, args.frames)
        encode_seconds = time.perf_counter() - encode_started
    native_path = Path(f3d._forge3d.__file__)
    with native_path.open("rb") as native:
        native_sha256 = file_digest(native, "sha256").hexdigest()
    record = {"source": source_info, "grid": grid.to_dict(), "display_transform": "sRGB",
              "runtime": {"source": f3d.__file__, "native": str(native_path), "native_sha256": native_sha256},
              "requested_quality": {"samples": scene_recipe.output.samples, "hdr": scene_recipe.output.hdr,
                                    "denoiser": scene_recipe.output.denoiser},
              "used_quality": {key: scene.last_render_metadata.get(key)
                               for key in ("samples_used", "denoiser_used", "visibility_sampling", "hdr_overlay_encoding")},
              "presentation": material.presentation, "recipe": material.to_dict(), "legend": prepared.legend,
              "feature_sources": {name: getattr(args, name, None) for name in ("vectors", "buildings")},
              "feature_recipe": feature_material.to_dict() if bound_layers else None,
              "feature_bindings": {layer.layer_id: layer.metadata["material_presentation"] for layer in bound_layers},
              "adapter": f3d.device_probe(), "backend": scene.last_render_backend,
              "camera": {"azimuth": args.azimuth, "elevation": args.elevation, "fov": args.fov,
                         "native_polar_angle": 90 - args.elevation, "mode": "mesh:zup:north",
                         "axes": "east/north/up", "azimuth_convention": "from east counterclockwise",
                         "distance": span * args.camera_distance},
              "size": args.size, "frames": args.frames,
              "fps": args.fps, "exaggeration": args.exaggeration, "prepared_seconds": prepared_seconds,
              "render_sequence_seconds": render_seconds, "encode_seconds": encode_seconds,
              "movie_encoding": {"codec": "libx264rgb", "crf": 0, "pixel_format": "rgb24", "interpretation": "lossless RGB"} if args.movie else None}
    (output / "evidence.json").write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dem", required=True)
    parser.add_argument("--classes")
    parser.add_argument("--vectors", help="provided projected FeatureCollection")
    parser.add_argument("--buildings", help="provided projected building FeatureCollection")
    parser.add_argument("--vector-presentation", choices=("physical", "protected"))
    parser.add_argument("--vector-width-world", type=float, help="line width in projected ground metres")
    parser.add_argument("--building-presentation", choices=("physical", "protected"), default="physical")
    parser.add_argument("--feature-crs")
    parser.add_argument("--feature-recipe", help="explicit scalar feature preset in the same recipe language")
    parser.add_argument("--class-field")
    parser.add_argument("--feature-field", action="append", nargs=3, metavar=("NAME", "PROPERTY", "UNITS"))
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--source-credit", required=True)
    parser.add_argument("--elevation-units", choices=("m",), help="required if the DEM band has no metre unit metadata")
    parser.add_argument("--vertical-datum", help="source vertical reference as documented by the provider")
    parser.add_argument("--sun-direction", type=float, nargs=3, default=(.8, .6, 0))
    parser.add_argument("--title", help="title rendered on the still and each frame")
    parser.add_argument("--window", type=int, nargs=4, metavar=("COL", "ROW", "WIDTH", "HEIGHT"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", required=True, type=int, nargs=2)
    parser.add_argument("--samples", type=int, default=1, help="native accumulation samples; 1 for preview")
    parser.add_argument("--hdr", action="store_true", help="write linear EXR sidecars as well as display PNGs")
    parser.add_argument("--denoiser", choices=("none", "off", "atrous", "oidn"), default="none")
    parser.add_argument("--frames", required=True, type=int)
    parser.add_argument("--fps", required=True, type=int)
    parser.add_argument("--azimuth", required=True, type=float, nargs=2, help="degrees from east counterclockwise")
    parser.add_argument("--elevation", required=True, type=float, help="degrees above the geographic horizon")
    parser.add_argument("--fov", required=True, type=float)
    parser.add_argument("--camera-distance", required=True, type=float, help="distance divided by scene diagonal")
    parser.add_argument("--exaggeration", type=float, default=1.0)
    parser.add_argument("--movie", action="store_true")
    args = parser.parse_args()
    if min(*args.size, args.frames, args.fps, args.samples) <= 0 or not np.isfinite(args.camera_distance) or args.camera_distance <= 0:
        parser.error("size, frames, fps, samples and camera distance must be positive")
    print(json.dumps(run(args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
