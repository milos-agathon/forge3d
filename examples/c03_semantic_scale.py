"""Render the C03 Swiss itinerary from retained, edition-pinned Swisstopo inputs."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import shutil
import subprocess
import statistics
import time
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

# Import real rasterio before anything can add the repository's test stub.
import rasterio
from rasterio.merge import merge
import numpy as np
import shapefile
from pyproj import CRS

import forge3d
from forge3d import _forge3d as native
from forge3d.animation import CameraAnimation
from forge3d.chronos import render_flythrough
from forge3d.map_scene import (
    LightingPreset, OrbitCamera, OutputSpec,
    _mapscene_effective_camera_mode, _shared_terrain_render_context, _terrain_resources_for_recipe,
)
from forge3d.preparation import PreparationGrid, PreparationRecipe, PreparationSource, PreparedScene, prepare_scene
from forge3d.semantic_scale import Connection, Landmark, RouteEdge, ScaleFrame, ScalePolicy, compile_scale
from forge3d.text import shape
from forge3d.text_atlas import bake_atlas, save_atlas

BASE = "828df486b93094dcbf1d950106d04d6994b3af60"
BOUNDS = (2639000, 1156000, 2649000, 1166000)
CRS_LV95 = "EPSG:2056"
DATUM = "LN02"
GRID_M = 20  # integer decimation of the 2 m source for this 10 km overview
TRACKS = """
C14001C1-EFE4-463E-AF51-CF7A94BE8858 B1A7A9F0-C8FB-4C4A-94B2-5285A8900516
D42E9039-26F9-402A-A118-FEBB49C0131D 479076E7-96EA-49BB-AEDE-B5A08A20EAB0
EEC571BB-AD46-40E0-8DF0-10AFCBD35E0F 95AADDDE-7230-4B95-9598-5323878BA207
372DC520-7949-48E1-807B-20995D308748 DCA3B72A-9C84-4322-ADDF-CB943F85900E
65A09D9E-38A4-4513-A3FE-A51C10FEB5C5 B4B1387F-F26B-48DD-8012-82F4F4694A40
BDC811B4-89D1-427B-B906-C7B096880685 30469453-32E7-4D44-B679-C646E21099AB
80A271D7-849A-4D06-96E4-85918DD28D14 36AB09AF-096D-45E4-8F0A-BD12D3A195B1
1C2E10CE-7B95-4B05-8E1B-3CB928838AFB 1B1A7967-4A5F-4E68-8EAE-F00EB1D60276
FEEDFF37-F587-4391-B0FF-C2600BB2F2C5 91175F9C-C97A-4E75-BCD3-322FA30387C4
F6470D98-2A47-4486-A60F-06BCC6509318 AB4FA538-D1E6-465C-B511-DE557C13A54E
3020915D-0DC7-42F7-AFF8-AF621984347D
""".split()
# Source-ordered Wengernalpbahn connection from Kleine Scheidegg to Grund.
GRUND = """
C5DA50AE-2656-4AAA-A3F6-1987426D1636 D872BCBA-B08C-4FD7-BA0E-AB3BC149A488
99DE6E3A-122F-4336-AF8F-81CDA67342A5 700A06DC-907D-4C13-B7EA-B4DE73267456
BDCE1348-0FFD-487F-92AE-486C190B5DC6 054EAE78-C664-411D-9EFB-DE539BFF2442
A128490F-7159-4459-8BF1-52FF18F1465D 474E07CF-0E19-4919-A089-96C597B84F8D
A3914574-4F08-41F9-BA15-90098455412E 1AF8820C-EB5F-448E-9D48-C7D13499B056
6AD6437D-1D99-4706-A6E8-99C8134FCB37 8B104947-F224-45D9-8326-831EF8E9267F
0C7C64D2-277F-466D-A1B7-739976996851 04C4FDE3-DC12-4DA7-826E-BFC41D691204
108C1758-A36C-4DB5-9793-A4ED0D64A70E 1FDD5AF7-7DB6-491C-9518-0C35B1DEDCA6
66FD8D9F-DD69-4C0A-8121-E153A46ACFAE FEC0028A-A5D7-4F3C-86E1-E4962BA6E167
9451E1E4-94C1-4065-B687-8360A56061FB BE75C8E0-CEF5-421F-892B-D039FDC25592
98252974-3C89-4EBE-A125-8438E7C90B97 4F686012-F38C-440A-8C63-8F74DCB9504F
00458AAA-36EF-41D8-B6E7-CDBC655B04E2 81D8FAE3-7379-4A42-9F42-7752697E39D3
9EACD458-9DB5-4457-BBAF-7DCA80E16C5F 615BA39B-44B7-4DA3-BE3A-9D8506167A7C
4677E4B3-842A-491A-A9D2-5964E9C81853 7649B076-BAD2-4B7D-8438-4CDEC9730D8A
26BE75A6-E84A-4677-B2F0-524447E26641 30D3665C-6A32-4635-B506-BC9B9CB7EC78
D3ADDDC9-A332-4BC0-AF55-A8A00BE318C3 8041F5F5-6647-4439-B703-1CD2E71FA8EE
""".split()
SCHEIDEGG = "365D7D89-C3B3-40A3-89B3-9EB9E227CAF7"
NAMES = {
    "9733CE21-EE4B-4FB7-8502-7B2FF469DEEB": (True, "Western itinerary anchor", None),
    "2E3C1D00-71F1-4F81-8A16-5D5D17507F13": (True, "Valley destination", None),
    "5BE294B0-910F-4DA3-BECD-201FB42320E3": (True, "Regional summit control", None),
    "B9DD5D85-5089-4406-8C4C-212F1D97BA16": (False, "Local neighbourhood", 200),
    "6F38E732-F8C5-484B-8A3E-4EF4CAD4B1A6": (False, "Local valley-side neighbourhood", 200),
}


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def preflight():
    revision = json.loads(native.compile_frame(0, 0, 1, "{}", "{}").to_json())["engine_revision"]
    if not revision.endswith("+" + BASE[:12]):
        raise RuntimeError(f"native engine {revision!r} does not match {BASE}")
    return {"base": BASE, "engine_revision": revision, "native_sha256": sha(Path(native.__file__)),
            "package": forge3d.__file__, "native": native.__file__, "rasterio": rasterio.__file__}


def source_inputs(root, out):
    sources = json.loads((root / "sources.json").read_text(encoding="utf-8"))
    assert (sources["crs"], sources["vertical"], sources["units"], sources["tlm_edition"]) == (
        CRS_LV95, DATUM, "metres", "swisstlm3d_2026-02")
    raw = root / "sources"
    tiles = {raw / item["url"].rsplit("/", 1)[1]: item["sha256"] for item in sources["dem"]}
    dem_path = out / "mosaic.tif"
    controls = []
    with ExitStack() as stack:
        datasets = []
        for path, expected in tiles.items():
            if sha(path) != expected:
                raise ValueError(f"unexpected DEM tile hash: {path}")
            ds = stack.enter_context(rasterio.open(path))
            if ds.crs.to_string() != CRS_LV95 or ds.units != ("metre",) or ds.res != (2, 2):
                raise ValueError(f"unexpected DEM grid or units: {path}")
            datasets.append(ds)
        dem, affine = merge(datasets, bounds=BOUNDS, res=GRID_M)
        expected_transform = rasterio.transform.from_origin(BOUNDS[0], BOUNDS[3], GRID_M, GRID_M)
        assert affine == expected_transform and dem.shape == (1, 500, 500)
        # GDAL-independent grid arithmetic plus reads from the original tile
        # at the centres of the corresponding output samples.
        for ds in datasets:
            x, y = ds.bounds.left + GRID_M / 2, ds.bounds.top - GRID_M / 2
            col = int((x - BOUNDS[0]) / GRID_M)
            row = int((BOUNDS[3] - y) / GRID_M)
            # The output centre lies at a corner of four source cells.
            # Compare the exact recorded height with those equidistant cells.
            neighbours = [(x + dx, y + dy) for dx in (-1, 1) for dy in (-1, 1)]
            raw_heights = [float(value[0]) for value in ds.sample(neighbours)]
            value = float(dem[0, row, col])
            assert value in raw_heights, (ds.name, x, y, value, raw_heights)
            controls.append({"xy": [x, y], "height_m": value, "source": Path(ds.name).name,
                             "source_neighbour_heights_m": raw_heights})
    with rasterio.open(dem_path, "w", driver="GTiff", count=1, dtype=dem.dtype, crs=CRS_LV95,
                       width=dem.shape[2], height=dem.shape[1], transform=affine) as ds:
        ds.write(dem)
        ds.set_band_unit(1, "m")
        ds.update_tags(vertical_reference=DATUM)
    grid = PreparationGrid(CRS_LV95, dem.shape[2], dem.shape[1], affine.to_gdal())
    source = PreparationSource("swissalti3d-2025", dem_path, "elevation", units="m", vertical_reference=DATUM)
    prepared = prepare_scene(PreparationRecipe(grid, (source,), DATUM))
    prepared.save(out / "prepared")
    prepared = PreparedScene.load(out / "prepared")

    # Decode the original shapefiles independently of the cached subset.
    # Match every selected UUID, XY and name to that subset before use.
    subset_path = root / "features.json"
    assert sha(subset_path) == sources["features_sha256"]
    subset = json.loads(subset_path.read_text(encoding="utf-8"))
    wanted = {"{" + uid + "}" for uid in (*TRACKS, *GRUND, *NAMES, SCHEIDEGG)}
    features, components = {}, {}
    for layer, group in subset.items():
        stem = raw / ("swissTLM3D_" + layer)
        prj = CRS.from_wkt(stem.with_suffix(".prj").read_text())
        assert prj.sub_crs_list[0].to_epsg() == 2056
        assert prj.sub_crs_list[1].to_epsg() == 5728
        assert all(axis.unit_name == "metre" for axis in prj.axis_info)
        components[layer] = {path.name: sha(path) for path in sorted(raw.glob(stem.name + ".*"))}
        cached = {item["properties"]["UUID"]: item for item in group}
        with shapefile.Reader(str(stem)) as reader:
            for item in reader.iterShapeRecords(fields=["UUID", "NAME"], bbox=BOUNDS):
                key = item.record.UUID
                if key not in wanted:
                    continue
                source_xy = tuple(map(tuple, item.shape.points))
                expected = cached[key]["geometry"]
                points = expected["coordinates"] if expected["type"] == "LineString" else [expected["coordinates"]]
                assert source_xy == tuple(tuple(point[:2]) for point in points), key
                assert tuple(item.shape.z) == tuple(point[2] for point in points), key
                assert (item.record.NAME or None) == cached[key]["properties"]["NAME"], key
                assert key not in features and len(item.shape.parts) <= 1
                features[key] = (source_xy, item.record.NAME)
    assert features.keys() == wanted, sorted(wanted - features.keys())
    write_json(out / "sources.json", {"dem_edition": "2025", "dem": sources["dem"],
               "tlm_edition": sources["tlm_edition"], "tlm_url": sources["tlm"],
               "tlm_components": components, "subset_sha256": sha(subset_path),
               "horizontal_crs": CRS_LV95, "vertical_crs": "EPSG:5728", "vertical_reference": DATUM,
               "units": "m", "dem_control_samples": controls, "verified_ids": sorted(features),
               "mosaic_sha256": sha(dem_path)})
    return prepared, features


def policy_for(features, reuse):
    def uid(key):
        return "{" + key + "}"

    ids = tuple(map(uid, GRUND if reuse else TRACKS))
    edges = tuple(RouteEdge(key, features[key][0]) for key in ids)
    names = NAMES
    reasons = ("Lütschental and Grindelwald identify the main railway itinerary.",
               "Eiger provides a regional summit control.")
    if reuse:
        names = {
            SCHEIDEGG: (True, "Uphill start of the revised rail connection", None),
            "B9DD5D85-5089-4406-8C4C-212F1D97BA16": (True, "Downhill endpoint of the revised rail connection", None),
            "5BE294B0-910F-4DA3-BECD-201FB42320E3": NAMES["5BE294B0-910F-4DA3-BECD-201FB42320E3"],
            "6F38E732-F8C5-484B-8A3E-4EF4CAD4B1A6": NAMES["6F38E732-F8C5-484B-8A3E-4EF4CAD4B1A6"],
        }
        reasons = (
            "The longer Wengernalpbahn itinerary has bends and switchbacks that exercise route generalization.",
            "The locator extent includes both rail endpoints and the Eiger control while centring the revised route.",
            "The overview targets the climb from Kleine Scheidegg and the valley pose focuses on the approach to Grund.",
        )
    places = tuple(Landmark(uid(key), features[uid(key)][1], features[uid(key)][0][0], required, reason, span)
                   for key, (required, reason, span) in names.items())
    bounds = (2639800, 1158400, 2646600, 1164200) if reuse else BOUNDS
    connection = "Kleine Scheidegg to Grund" if reuse else "Lütschental to Grindelwald"
    return ScalePolicy(CRS_LV95, DATUM, edges, (Connection(connection, ids),), places,
                       bounds, (0.77, 0.66, 0.98, 0.96), GRID_M, 0.5, 3, 20,
                       ("Overview candidate tolerance follows the prepared 20 m DEM grid.",
                        "Half an output pixel is the route displacement criterion.",
                        "UTM 32N is the GENERALIS working CRS. The original LV95 vertices are retained.",
                        "A 200 m local neighbourhood witness should span one label em before admission.",
                        *reasons),
                       ("Other rail branches, other names and hydrography omitted to foreground this itinerary.",
                        "Source Z retained in verification; route and names drape on the LN02 terrain."),
                       "EPSG:32632")


def film_frames(prepared, fps, duration, reuse):
    dem = next(iter(prepared.layers.values())).data
    west, south, east, north = BOUNDS
    center = ((west + east) / 2, (south + north) / 2)
    animation = CameraAnimation()
    # A closer, wider overview keeps the full 10 km DEM inside the clip volume.
    overview = (2642900, 1160900, 2080, 8800, 192, 58, 68) if reuse else (2643300, 1161400, 2080, 8800, 192, 58, 68)
    close = (2644400, 1163050, 1300, 3200, 200, 61, 48) if reuse else (2645100, 1163690, 1170, 4250, 200, 62, 46)
    datum = (float(dem.min()) + float(dem.max())) / 2
    for t, (x, y, z, distance, azimuth, elevation, fov) in ((0, overview), (duration, close)):
        animation.add_keyframe(t, azimuth, elevation, distance, fov, (x - center[0], y - center[1], z - datum))
    frames = []
    for i in range(round(duration * fps) + 1):
        state = animation.evaluate(i / fps)
        x, y, _ = state.target
        frames.append(ScaleFrame(OrbitCamera(state.target, state.radius, state.phi_deg, state.theta_deg, state.fov_deg),
                                 (center[0] + x, center[1] + y)))
    cut = replace(frames[-1], camera=replace(frames[-1].camera, azimuth_deg=226))
    outage = replace(cut, unavailable_detail=("Scenario input: fine detail withheld; overview remains available",))
    timeline = (*frames, *((cut,) * fps), *((outage,) * fps), *reversed(frames))
    return timeline, {"overview": 0, "valley": len(frames) - 1, "cut": len(frames),
                      "partial": len(frames) + fps, "ascent": len(timeline) - 1}


def render_device():
    # Session reads the process-wide GPU context already used by the renderer.
    session = native.Session(window=False)
    return {"adapter": session.adapter_name, "backend": session.backend, "device_type": session.device_type}


def peak_working_set():
    if not hasattr(ctypes, "windll"):
        return None
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
            (field, ctypes.c_size_t) for field in (
                "peak", "current", "peak_paged", "paged", "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile")
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    get_process = ctypes.windll.kernel32.GetCurrentProcess
    get_process.restype = wintypes.HANDLE
    get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
    if not get_memory(get_process(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError()
    return counters.peak


def height_residency_probe(base, policy, frame, output, evidence):
    """Exercise actual height cache loss on a fixed mixed-layer pose."""
    # The production grid stays within the prepared DEM footprint and its
    # vertex shader reads the renderer's height page table in both states.
    terrain_span = BOUNDS[2] - BOUNDS[0]
    dem = np.ascontiguousarray(base.recipe.terrain.data, dtype=np.float32)
    x, y, z = frame.camera.target
    focus = (x, z, -y)  # The height scheduler takes X east, Y up, Z south.
    ready_fields = ("resident_fine_tiles", "resident_ancestor_tiles", "coarse_prefilled")
    capture = {}
    with _shared_terrain_render_context():
        resources, _ = _terrain_resources_for_recipe(base.recipe)
        renderer = resources.renderer
        renderer.enable_height_streaming(terrain_span, dem=dem)
        ready = dict(renderer.stream_height_tiles(focus))
        while not ready["converged"]:
            renderer._wait_height_streaming()
            next_stats = dict(renderer.stream_height_tiles(focus))
            if not next_stats["converged"] and all(next_stats[key] == ready[key] for key in ready_fields):
                raise RuntimeError(f"height streamer made no residency progress after the loader barrier: {next_stats}")
            ready = next_stats
        if ready["resident_fine_tiles"] == 0:
            raise RuntimeError("height streamer converged without fine tiles")

        for name in ("resident", "interrupted"):
            scenario = frame
            stats = ready
            if name == "interrupted":
                # Recreate the real cache, then request the same focus without
                # admitting uploads before the frame consumes its fallback.
                renderer.enable_height_streaming(terrain_span, dem=dem)
                stats = dict(renderer.stream_height_tiles(focus, max_uploads=0))
                if stats["resident_fine_tiles"] != 0 or stats["converged"]:
                    raise RuntimeError("height cache reset did not interrupt fine residency")
                scenario = replace(frame, unavailable_detail=("height streamer fine pages absent after cache reset",))
            compiled = compile_scale(base, policy, (scenario,))
            scene = compiled.scene(0)
            path = output / f"{name}.png"
            scene.render(str(path))
            after = dict(renderer.height_streaming_stats())
            if after["resident_fine_tiles"] != stats["resident_fine_tiles"]:
                raise RuntimeError("height residency changed while rendering the probe frame")
            capture[name] = {"before": stats, "after": after, "sha256": sha(path),
                             "render_backend": scene.last_render_backend,
                             "terrain_geometry_backend": scene.last_render_metadata.get("terrain_geometry_backend"),
                             "decision": compiled.frame_record(0)}
    write_json(output / "height-residency.json", {
        **evidence, "render_device": render_device(),
        "terrain_camera_mode": _mapscene_effective_camera_mode(base.recipe), "frames": capture,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--size", nargs=2, type=int, required=True)
    parser.add_argument("--fps", type=int, required=True)
    parser.add_argument("--descent-seconds", type=float, required=True)
    parser.add_argument("--font", type=Path, default=Path("C:/Windows/Fonts/verdana.ttf"))
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--plans-only", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--height-residency-probe", action="store_true")
    args = parser.parse_args()
    if args.fps <= 0 or not math.isfinite(args.descent_seconds) or args.descent_seconds <= 0:
        parser.error("positive frame rate and duration required")
    if args.height_residency_probe and (args.plans_only or args.baseline or args.reuse):
        parser.error("the height residency probe requires the original physical itinerary")
    provenance = preflight()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    prepared, features = source_inputs(args.inputs, args.output)
    authored = time.perf_counter()
    policy = policy_for(features, args.reuse)
    font = args.font.resolve(strict=True)
    for place in policy.landmarks:
        shape(place.name, [font], policy.label_size_px)
    atlas = bake_atlas(font, charset=" ".join(place.name for place in policy.landmarks))
    atlas_png, atlas_json = save_atlas(atlas, args.output / "names.png", args.output / "names.json")
    policy = replace(policy, glyph_atlas={"font_path": str(font),
                                         "image_path": str(atlas_png.resolve()),
                                         "metrics_path": str(atlas_json.resolve())})
    frames, stills = film_frames(prepared, args.fps, args.descent_seconds, args.reuse)
    base = prepared.to_scene(
        lighting=LightingPreset("daylight", sun_direction=(-0.46, -0.58, 0.67), intensity=1.8,
                                settings={"cli_params": {"camera_mode": "mesh:zup"},
                                          "albedo_mode": "colormap",
                                          "colormap": "27382d,52664e,758474,aaa59b,e2e2d9",
                                          "hue_variation_strength": 0}),
        output=OutputSpec(*args.size), camera=frames[0].camera,
    )
    if args.baseline:
        # Static MapScene control. Same raw sources, cameras, output, stroke,
        # label placement owner and locator; all route and label detail admitted.
        policy = replace(policy, overview_tolerance_m=0, landmarks=tuple(
            replace(place, detail_span_m=None) for place in policy.landmarks))
    if args.height_residency_probe:
        height_residency_probe(base, policy, frames[stills["cut"]], args.output, provenance)
        return
    compiled = compile_scale(base, policy, frames)
    receipt = compiled.to_dict()
    # Spatial decisions agree at reversed poses. Timeline admission can
    # differ when a pose borders a cut or outage in only one direction.
    def spatial_decision(record):
        decision = {k: v for k, v in record.items() if k not in ("frame", "labels_accepted")}
        decision["labels_omitted"] = {key: reason for key, reason in record["labels_omitted"].items()
                                     if reason != "single_frame_admission"}
        return decision

    for index, record in enumerate(receipt["frames"][:stills["cut"]]):
        reverse = compiled.frame_record(len(frames) - index - 1)
        assert spatial_decision(record) == spatial_decision(reverse)
        assert compiled.frame_record(index) == record
    write_json(args.output / "scale-plan.json", receipt)
    compiled_at = time.perf_counter()
    print(f"Compiled {len(frames)} frames at {args.size[0]}x{args.size[1]}", flush=True)
    evidence = {**provenance, "font_sha256": sha(font), "atlas_sha256": sha(atlas_png),
                "source_preparation_s": authored - start,
                "policy_and_compile_s": compiled_at - authored, "frame_count": len(frames), "stills": stills}
    write_json(args.output / "run.json", evidence)
    if args.plans_only:
        return
    frame_dir = args.output / "frames"
    render_start = time.perf_counter()
    render_start_ns = time.time_ns()
    manifest = render_flythrough(compiled.scenes(), base_seed=0, samples=1, out_dir=frame_dir)
    rendered = time.perf_counter()
    print(f"CHRONOS rendered {len(frames)} frames in {rendered - render_start:.2f}s", flush=True)
    native_memory = dict(native.global_memory_metrics())
    device = render_device()
    frame_times = [(frame_dir / f"frame_{i:08d}.png").stat().st_mtime_ns for i in range(len(frames))]
    time_to_first_png_s = (frame_times[0] - render_start_ns) / 1e9
    frame_intervals = [(b - a) / 1e9 for a, b in zip(frame_times, frame_times[1:])]
    if any(manifest.frame_record(i)["engine_revision"] != provenance["engine_revision"] for i in range(len(frames))):
        raise RuntimeError("CHRONOS returned a different engine revision")
    for name, index in stills.items():
        shutil.copyfile(frame_dir / f"frame_{index:08d}.png", args.output / f"{name}.png")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-framerate", str(args.fps),
                    "-i", str(frame_dir / "frame_%08d.png"), "-c:v", "libx264", "-crf", "17",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output / "grindelwald.mp4")],
                   check=True)
    write_json(args.output / "run.json", {**evidence, "render_device": device, "chronos_s": rendered - render_start,
               "chronos_s_per_frame": (rendered - render_start) / len(frames),
               "time_to_first_png_s": time_to_first_png_s,
               "median_png_interval_s": statistics.median(frame_intervals),
               "encode_s": time.perf_counter() - rendered, "peak_process_working_set_bytes": peak_working_set(),
               "native_tracked_peak_bytes": native_memory["peak_total_bytes"]})


if __name__ == "__main__":
    main()
