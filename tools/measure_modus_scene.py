"""Fresh-process qualification for MapScene.render and CHRONOS admission."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

import forge3d as f3d
from forge3d.chronos import render_flythrough
from measure_modus_poster import assert_tracked_cpu_peak, peak


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route", choices=("map", "fly"))
    parser.add_argument("mode", choices=("control", "admit", "unlimited", "refuse", "empty"))
    parser.add_argument("out", type=Path)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--dem", type=int, default=64)
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--pcf-cascades", type=int)
    parser.add_argument("--disabled-grid", action="store_true")
    parser.add_argument("--decoded-vector", action="store_true",
                        help="use the 96x64 public label/vector scene with its decoded GeoJSON")
    parser.add_argument("--float64-geotiff", action="store_true",
                        help="read the terrain from a Float64 GeoTIFF")
    parser.add_argument("--streaming", choices=("clipmap", "vt", "both"))
    parser.add_argument("--clipmap-resolution", type=int, default=16)
    parser.add_argument("--vt-atlas-size", type=int, default=1024)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--refusal-limit", type=int, default=1)
    parser.add_argument("--certificate", action="store_true")
    return parser.parse_args()


def build_scene(args, output):
    if args.decoded_vector:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
        from test_mapscene_render_png import _public_label_vector_scene

        base = _public_label_vector_scene(output / "map.png")
        vector, labels = base.recipe.layers
        source = args.out / "roads.geojson"
        features = [dict(type="Feature", properties={}, **row) for row in vector.features]
        source.write_text(json.dumps({"type": "FeatureCollection", "features": features}),
                          encoding="utf8")
        return f3d.MapScene(replace(base.recipe, layers=[replace(vector, path=str(source)), labels]))

    if args.mode in ("empty", "refuse"):
        # Avoid a temporary float64 linspace high-water mark masking call growth.
        dem = np.ones((args.dem, args.dem), dtype=np.float32)
    else:
        dem = np.linspace(0, 20, args.dem**2, dtype=np.float32).reshape(args.dem, args.dem)
    metadata = {"source_id": "modus-ramp"}
    if args.disabled_grid:
        metadata["terrain_geometry"] = {"mode": "grid", "enabled": False}
    if args.streaming in ("clipmap", "both"):
        metadata["terrain_geometry"] = {"mode": "clipmap", "ring_count": 3,
                                        "ring_resolution": args.clipmap_resolution,
                                        "center_resolution": args.clipmap_resolution}
    if args.streaming in ("vt", "both"):
        metadata["virtual_texture"] = dict(procedural_sources=True, source_size=(256, 512),
                                           families=["albedo", "normal", "mask"], atlas_size=args.vt_atlas_size)
    lighting = f3d.LightingPreset(name="daylight")
    if args.pcf_cascades is not None:
        lighting = replace(lighting, settings={"renderer_config": {"shadows": {
            "enabled": True, "technique": "pcf", "map_size": 4096, "cascades": args.pcf_cascades,
        }}})
    terrain = f3d.TerrainSource(data=dem, crs="EPSG:32610", elevation_sampling_available=True,
                                metadata=metadata)
    if args.float64_geotiff:
        from forge3d import gis
        path = args.out / "height.tif"
        gis.write_raster(path, dem.astype(np.float64), crs=terrain.crs,
                         transform=(1, 0, 0, 0, -1, args.dem), overwrite=True)
        terrain = replace(terrain, path=str(path), data=None)
    return f3d.MapScene(
        terrain=terrain,
        camera=f3d.OrbitCamera(target=(0, 0, 0), distance=75.0, azimuth_deg=35.0),
        lighting=lighting,
        output=f3d.OutputSpec(width=args.width, height=args.width, samples=args.samples),
    )


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / "render"
    scene = build_scene(args, output)
    return_peak = None
    if args.route == "map":
        job = dict(entry_point="MapScene.render", scene=scene, path=str(output / "map.png"))
        def render(**kw):
            nonlocal return_peak
            scene.render(job["path"], certificate=(output / "map.certificate.json"
                                                   if args.certificate else False), **kw)
            return_peak = peak()
            assert scene.last_render_backend == "gpu_terrain"
            assert scene.last_render_metadata["samples_used"] == args.samples
            if args.streaming in ("clipmap", "both"):
                assert scene.last_render_metadata["terrain_geometry_backend"] == "clipmap_indexed_pbr"
            if args.streaming in ("vt", "both"):
                stats = scene.material_vt_stats()
                assert stats["source_count"] == 12 and stats["atlas_device_local_bytes"] > 0
            return getattr(scene, "last_process_memory", None)
    else:
        scenes = [f3d.MapScene(replace(scene.recipe, camera=replace(scene.recipe.camera,
                    azimuth_deg=35.0 + i))) for i in range(args.frames)]
        options = dict(camera_path=scenes, out_dir=output, samples=args.samples, base_seed=7)
        job = dict(entry_point="render_flythrough", **options)
        def render(**kw):
            nonlocal return_peak
            result = render_flythrough(**options, certificate=args.certificate, **kw)
            return_peak = peak()
            return getattr(result, "process_memory", None)
    before = peak()
    import psutil
    counters = psutil.Process().memory_info()
    current = getattr(counters, "private", counters.rss)
    kw = {}
    if args.mode == "admit":
        from forge3d import mem
        kw["process_limit_bytes"] = mem.estimate(job)["predicted_peak_bytes"]
    elif args.mode in ("refuse", "empty"):
        kw["process_limit_bytes"] = args.refusal_limit
    memory = None
    refusal = None
    try:
        if args.mode != "empty":
            memory = render(**kw)
        assert args.mode != "refuse"
    except MemoryError as exc:
        assert args.mode == "refuse" and exc.code == "memory_admission_refused"
        assert exc.limit_bytes == args.refusal_limit < exc.estimate["predicted_peak_bytes"]
        refusal = exc.code
        memory = exc.estimate
    after = peak()
    counters = psutil.Process().memory_info()
    final_current = getattr(counters, "private", counters.rss)
    result = dict(mode=args.mode, route=args.route, width=scene.recipe.output.width,
                  height=scene.recipe.output.height, dem=args.dem if args.float64_geotiff else scene.recipe.terrain.data.shape[0],
                  samples=args.samples, pcf_cascades=args.pcf_cascades, disabled_grid=args.disabled_grid,
                  streaming=args.streaming, clipmap_resolution=args.clipmap_resolution if args.streaming in ("clipmap", "both") else None,
                  vt_atlas_size=args.vt_atlas_size if args.streaming in ("vt", "both") else None,
                  decoded_vector=args.decoded_vector, float64_geotiff=args.float64_geotiff,
                  frames=args.frames if args.route == "fly" else 1,
                  baseline=before,
                  pre_call_current=current, post_call_current=final_current,
                  return_peak=return_peak, peak=after, growth=after-before,
                  memory=memory, limits=kw, refusal=refusal)
    if args.mode not in ("refuse", "empty"):
        images = sorted(output.glob("*.png"))
        assert len(images) == result["frames"]
        hashes = []
        for image in images:
            with Image.open(image) as im:
                rgba = np.asarray(im.convert("RGBA"))
            hashes.append(hashlib.sha256(rgba).hexdigest())
            if args.reference:
                with Image.open(args.reference / "render" / image.name) as control:
                    np.testing.assert_array_equal(rgba, np.asarray(control.convert("RGBA")))
                assert image.read_bytes() == (args.reference / "render" / image.name).read_bytes()
        result["rgba_sha256"] = hashes
        if args.reference and args.route == "fly":
            for sidecar in ["flythrough_manifest.json", *(p.name for p in output.glob("*.provenance.json"))]:
                assert (output / sidecar).read_bytes() == (args.reference / "render" / sidecar).read_bytes()
        if memory is not None:
            from forge3d import diagnostics
            key = "process_memory" if args.route == "map" else "flythrough_process_memory"
            public = diagnostics.render_certificate(sign=False)
            assert public[key] == memory
            result["gpu_capture"] = public["allocations"]
            if args.certificate:
                cert = sorted(output.glob("*.certificate.json"))[-1]
                written = json.loads(cert.read_text("utf8"))
                assert written[key] == memory
                if args.decoded_vector and args.reference:
                    from forge3d.certificate import payload_sha256
                    control = json.loads((args.reference / "render" / cert.name).read_text("utf8"))
                    assert scene.last_render_metadata["certificate_payload_sha256"] == payload_sha256(control)
                    assert payload_sha256(written) == payload_sha256(control)
    else:
        assert not output.exists()
    (args.out / "measurement.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print({k: v for k, v in result.items() if k not in ("memory", "gpu_capture")})
    if args.mode in ("admit", "unlimited"):
        assert memory["measured_peak_bytes"] == return_peak, (memory["measured_peak_bytes"], return_peak)
        assert before <= return_peak <= after <= memory["predicted_peak_bytes"]
        assert memory["prediction_ratio"] >= 1
        assert sum(memory["ledger_samples"]["residual_by_phase_at_peak"].values()) == memory["sampled_unexplained_bytes"]
        assert result["gpu_capture"]["peak_host_visible_bytes"] <= memory["host_visible_peak_bytes"]
        assert result["gpu_capture"]["peak_device_local_bytes"] <= memory["device_local_peak_bytes"]
        assert_tracked_cpu_peak(memory)
        if kw:
            assert after <= kw["process_limit_bytes"]
        if args.reference:
            control = json.loads((args.reference / "measurement.json").read_text())
            control_growth = control["peak"] - control.get("pre_call_current", control["baseline"])
            assert memory["predicted_peak_bytes"] - memory["process_baseline_bytes"] >= control_growth
    if args.mode == "refuse":
        noise = json.loads((args.reference / "measurement.json").read_text())["growth"]
        assert result["growth"] <= noise


if __name__ == "__main__":
    main()
