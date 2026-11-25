"""Render and measure the same prepared Rhodes workload through D02 or WGSL."""

from pathlib import Path
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import time

import rasterio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import forge3d as f3d
from forge3d import gis, map_scene as ms
from forge3d._native import get_native_module
from forge3d.observation_time import ObservationRecord
from forge3d.preparation import PreparationGrid

COLORS = [
    (150, 47, 36, 255),
    (230, 151, 87, 255),
    (236, 224, 192, 255),
    (96, 171, 146, 255),
    (27, 97, 101, 255),
]
BINS = [
    [-0.5, -0.125, 0.125, 0.5],
    [-0.375, -0.0625, 0.0625, 0.375],
    [-0.25, -0.0625, 0.125, 0.25],
]
BEFORE = "2023-07-18"
AFTER = ("2023-08-07", "2023-08-17")
QUANTITY = "Normalized difference of NDVI"
CAMERAS = [(azimuth, tilt) for azimuth, tilt in zip(range(-90, -66, 2), range(16, 28))]


def scene(
    directory,
    grid,
    *,
    azimuth=225,
    elevation=58,
    mode="mesh:zup:north",
    publication=False,
):
    return f3d.MapScene(
        terrain=f3d.TerrainSource(
            path=directory / "terrain.tif",
            crs=grid.crs,
            metadata=grid.to_dict()
            | {
                "source_id": "Copernicus GLO-30",
                "height_units": "m",
                "vertical_reference": "EGM2008",
                "resolution": [10, 10],
            },
        ),
        camera=f3d.OrbitCamera(
            distance=21000 if publication else 31500,
            elevation_deg=elevation,
            azimuth_deg=azimuth,
            fov_deg=58 if publication else 44,
        ),
        lighting=f3d.LightingPreset(
            sun_direction=(0.4, 0.7, 0.75),
            settings={
                "cli_params": {"camera_mode": mode},
                "colormap_strength": 1.0,
                "hue_variation_strength": 0.0,
            },
        ),
        output=f3d.OutputSpec(width=1024, height=768),
    )


def load(directory, name):
    with rasterio.open(directory / f"{name}.tif") as dataset:
        return dataset.read(1)


def reference(first, last, mask, edges):
    a, b = last + first, last - first
    valid = mask.astype(bool) & np.isfinite(a) & np.isfinite(b) & (a != 0)
    data = np.full_like(a, np.nan)
    np.divide(b, a, out=data, where=valid)
    return data, gis.classify_raster(data, bins=edges, dtype="uint8")["array"][0]


def cpu_style(map_scene, grid, ids):
    map_scene.recipe.layers = [
        f3d.RasterOverlay(
            "spectral change",
            data=ids,
            crs=grid.crs,
            style=f3d.CategoricalRasterStyle(dict(enumerate(COLORS, 1)), nodata=0),
        )
    ]


def sources(directory, declared, date):
    records = {record["observed_at"][:10]: record for record in declared["records"]}
    sources = []
    for day in (date, BEFORE):
        row = records[day]
        observation = ObservationRecord(
            row["source_id"],
            row["observed_at"],
            row["observed_at"],
            "2023-09-01T00:00:00Z",
            row,
        )
        sources.append(
            f3d.RasterRecipeSource(
                directory / f"{day}.tif", row["source_id"], row["ndvi"], observation
            )
        )
    mask = f3d.RasterRecipeSource(
        directory / f"{date}-mask.tif",
        f"joint SCL mask {BEFORE}/{date}",
        records[date]["joint_mask_sha256"],
    )
    return *sources, mask


def peak_cpu():
    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            *(
                (name, ctypes.c_size_t)
                for name in (
                    "PeakWorkingSetSize",
                    "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage",
                    "PagefileUsage",
                    "PeakPagefileUsage",
                )
            ),
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(Counters),
        wintypes.DWORD,
    ]
    if not ctypes.windll.psapi.GetProcessMemoryInfo(
        ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        raise OSError("GetProcessMemoryInfo failed")
    return counters.PeakWorkingSetSize


def benchmark(root, mode):
    directory = root / "prepared"
    declared = json.loads((directory / "sources.json").read_text())
    grid = PreparationGrid(**declared["grid"])
    output = root / mode
    output.mkdir(exist_ok=True)
    native = get_native_module()
    probe = f3d.device_probe("vulkan")
    assert probe["status"] == "ok" and probe["device_type"] == "DiscreteGpu", probe
    frames = []
    original = ms._render_terrain_renderer_result

    def timed_terrain(*args, **kwargs):
        started = time.perf_counter()
        result = original(*args, **kwargs)
        assert result is not None
        frames[-1]["native_seconds"] = time.perf_counter() - started
        frames[-1]["frame_readback_bytes"] = result.rgba.nbytes
        return result

    ms._render_terrain_renderer_result = timed_terrain
    native_overlay = f3d.OverlayLayer

    class TimedOverlay:
        from_colormap1d = staticmethod(native_overlay.from_colormap1d)

        @staticmethod
        def from_raster_rgba(rgba, **kwargs):
            start = time.perf_counter()
            result = native_overlay.from_raster_rgba(rgba, **kwargs)
            frames[-1]["upload_enqueue_seconds"] = time.perf_counter() - start
            frames[-1]["category_upload_payload_bytes"] = rgba.nbytes
            return result

    if mode == "cpu":
        f3d.OverlayLayer = TimedOverlay
    native_recipe = getattr(native, "_ResidentRasterRecipe", None)

    class TimedRecipe:
        def __init__(self):
            self.recipe = native_recipe()

        def __getattr__(self, name):
            return getattr(self.recipe, name)

        def replace(self, *args):
            start = time.perf_counter()
            self.recipe.replace(*args)
            frames[-1]["input_upload_and_normalize_seconds"] = (
                time.perf_counter() - start
            )

        def classify(self, *args):
            start = time.perf_counter()
            self.recipe.classify(*args)
            frames[-1]["classification_seconds"] = time.perf_counter() - start

    if mode == "gpu":
        assert native_recipe is not None, "Build matching native recipe bindings"
        native._ResidentRasterRecipe = TimedRecipe
    recipe = None
    try:
        for day in AFTER:
            for index, edges in enumerate(BINS):
                target = output / f"{day}-{index}.png"
                rendered = scene(directory, grid)
                stage = {"date": day, "bins": edges}
                frames.append(stage)
                started = time.perf_counter()
                kwargs = {}
                if mode == "cpu":
                    if index == 0:
                        if day == AFTER[0]:
                            first = load(directory, BEFORE)
                        last, mask = load(directory, day), load(
                            directory, f"{day}-mask"
                        )
                    _, ids = reference(first, last, mask, edges)
                    cpu_style(rendered, grid, ids)
                else:
                    if recipe is None:
                        recipe = f3d.RasterRecipe(
                            *sources(directory, declared, day),
                            grid,
                            edges,
                            COLORS,
                            quantity=QUANTITY,
                        )
                    elif index == 0:
                        recipe.replace_inputs(*sources(directory, declared, day))
                        recipe.set_bins(edges)
                    else:
                        recipe.set_bins(edges)
                    kwargs["raster_recipe"] = recipe
                render_start = time.perf_counter()
                rendered.render(str(target), **kwargs)
                stage["render_export_seconds"] = time.perf_counter() - render_start
                stage["seconds"] = time.perf_counter() - started
                assert rendered.last_render_backend == "gpu_terrain"
                stage.update(
                    bytes=target.stat().st_size,
                    peak_cpu_working_set=peak_cpu(),
                    memory=native.global_memory_metrics(),
                    backend=rendered.last_render_backend,
                )
                if mode == "cpu":
                    stage["counts"] = np.bincount(ids.ravel(), minlength=6).tolist()
                else:
                    serial = recipe.to_json()
                    assert f3d.RasterRecipe.from_json(serial).to_json() == serial
                    (output / f"{day}-{index}.recipe.json").write_text(
                        serial + "\n", encoding="utf8"
                    )
                    assert (
                        rendered.last_render_metadata["thematic_legend"]
                        == recipe.legend()
                    )
                    stage["metrics_at_frame"] = recipe.metrics()
                    stage["counts"] = list(recipe.counts().values())
                    stage["sample"] = recipe.sample(900, 900)
                    stage["metrics_after_inspection"] = recipe.metrics()
                    (output / f"{day}-{index}.legend.json").write_text(
                        json.dumps(recipe.legend(), indent=2), encoding="utf8"
                    )
                print(day, index, stage["seconds"], stage["counts"], flush=True)
    finally:
        ms._render_terrain_renderer_result = original
        f3d.OverlayLayer = native_overlay
        if mode == "gpu":
            native._ResidentRasterRecipe = native_recipe
    (output / "metrics.json").write_text(
        json.dumps({"probe": probe, "frames": frames}, indent=2)
    )
    if recipe is not None:
        recipe.close()


def publication_plate(frame, inset, date, legend):
    """Compose the same final publication for either native renderer."""
    fonts = Path(__file__).resolve().parents[1] / "assets" / "fonts"
    font_path = fonts / "NotoSansLatin-subset.ttf"
    title = ImageFont.truetype(str(font_path), 31)
    body = ImageFont.truetype(str(font_path), 20)
    small = ImageFont.truetype(str(font_path), 15)
    plate = Image.new("RGB", (1340, 870), "#f6f2e8")
    plate.paste(frame.convert("RGB"), (16, 74))
    plate.paste(inset.convert("RGB"), (1045, 107))
    draw = ImageDraw.Draw(plate)
    draw.text(
        (16, 14),
        "Rhodes  /  vegetation spectral change",
        font=title,
        fill="#252520",
    )
    draw.text(
        (16, 52),
        f"Sentinel-2 L2A   18 July - {int(date[-2:])} August 2023   |   Copernicus GLO-30 terrain",
        font=small,
        fill="#333330",
    )
    draw.text((1045, 80), "Grid north up", font=body, fill="#252520")
    draw.text((1045, 415), "(NDVI Aug - NDVI Jul)", font=body, fill="#252520")
    draw.text((1045, 438), "/ (NDVI Aug + NDVI Jul)", font=body, fill="#252520")
    for row, item in enumerate(legend["entries"]):
        y = 475 + 34 * row
        draw.rectangle((1045, y, 1071, y + 21), fill=tuple(item["rgba"][:3]))
        label = item["label"].replace("≥", ">=")
        draw.text((1084, y), label, font=body, fill="#252520")
    draw.rectangle((1045, 645, 1071, 666), fill=(25, 25, 38))
    draw.text((1084, 645), "Invalid / masked", font=body, fill="#252520")
    draw.text((1045, 699), "Spectral change only.", font=small, fill="#333330")
    return plate


def publish(root, *, revised=False):
    started = time.perf_counter()
    directory = root / "prepared"
    declared = json.loads((directory / "sources.json").read_text())
    grid = PreparationGrid(**declared["grid"])
    output = root / ("publication-revised" if revised else "publication")
    output.mkdir(exist_ok=True)
    date, edges = (AFTER[1], BINS[2]) if revised else (AFTER[0], BINS[0])
    recipe = f3d.RasterRecipe(
        *sources(directory, declared, date),
        grid,
        edges,
        COLORS,
        quantity=QUANTITY,
    )
    first = load(directory, BEFORE)
    palette = np.asarray([(25, 25, 38, 255), *COLORS], dtype=np.uint8)
    _, ids = reference(
        first, load(directory, date), load(directory, date + "-mask"), edges
    )
    inset = Image.fromarray(palette[ids]).resize((280, 280), Image.Resampling.NEAREST)
    inset.save(output / "reference-inset.png")
    timings = {
        "setup_seconds": time.perf_counter() - started,
        "native_seconds": 0.0,
        "layout_seconds": 0.0,
    }
    records = []
    for index, (azimuth, tilt) in enumerate(CAMERAS):
        target = output / f"{index:03d}.png"
        live = scene(directory, grid, azimuth=azimuth, elevation=tilt, publication=True)
        started = time.perf_counter()
        live.render(str(target), raster_recipe=recipe)
        timings["native_seconds"] += time.perf_counter() - started
        assert live.last_render_backend == "gpu_terrain"
        assert live.last_render_metadata["thematic_legend"] == recipe.legend()
        frame = np.asarray(Image.open(target))
        background = frame[0, 0]
        rows, columns = np.nonzero(np.any(frame != background, axis=2))
        assert 0 < rows.min() <= rows.max() < frame.shape[0] - 1
        assert 0 < columns.min() <= columns.max() < frame.shape[1] - 1
        pixels = hashlib.sha256(target.read_bytes()).hexdigest()
        records.append(
            {
                "frame": target.name,
                "azimuth": azimuth,
                "tilt": tilt,
                "sha256": pixels,
                "bounds": [
                    int(columns.min()),
                    int(rows.min()),
                    int(columns.max()),
                    int(rows.max()),
                ],
                "visible_pixels": int(rows.size),
            }
        )
        started = time.perf_counter()
        with Image.open(target) as image:
            publication_plate(image, inset, date, recipe.legend()).save(
                output / f"plate-{index:03d}.png"
            )
        timings["layout_seconds"] += time.perf_counter() - started
    assert len({record["sha256"] for record in records}) == len(CAMERAS)
    metrics = recipe.metrics()
    assert (
        metrics["normalization_dispatches"] == metrics["classification_dispatches"] == 1
    )
    assert metrics["readback_bytes"] == 0
    (output / "frames.json").write_text(
        json.dumps(
            {
                "frames": records,
                "metrics": metrics,
                "legend": recipe.legend(),
                "timings": timings,
            },
            indent=2,
        )
    )
    started = time.perf_counter()
    values, classes = recipe.export_cells()
    profile = {
        "driver": "GTiff",
        "width": grid.width,
        "height": grid.height,
        "count": 1,
        "transform": grid.transform,
        "crs": grid.crs,
        "compress": "deflate",
        "tiled": True,
    }
    for filename, data, nodata in (("values", values, np.nan), ("classes", classes, 0)):
        with rasterio.open(
            output / f"{filename}.tif", "w", **profile, dtype=data.dtype, nodata=nodata
        ) as dataset:
            dataset.write(data, 1)
        with rasterio.open(output / f"{filename}.tif") as dataset:
            np.testing.assert_array_equal(dataset.read(1), data)
            assert dataset.transform == grid.transform and str(dataset.crs) == grid.crs
    export_seconds = time.perf_counter() - started
    expected_values, expected_classes = reference(
        first, load(directory, date), load(directory, date + "-mask"), edges
    )
    np.testing.assert_array_equal(values, expected_values)
    valid = expected_classes != 0
    np.testing.assert_array_equal(
        values[valid].view(np.uint32), expected_values[valid].view(np.uint32)
    )
    np.testing.assert_array_equal(classes, expected_classes)
    (output / "export.json").write_text(
        json.dumps(
            {
                "export_seconds": export_seconds,
                "seconds_including_verification": time.perf_counter() - started,
                "analysis_payload_bytes": recipe.metrics()["readback_bytes"],
                "matching_cells": int(classes.size),
            },
            indent=2,
        )
    )
    recipe.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "mode", choices=["cpu", "gpu", "publication", "publication-revised"]
    )
    args = parser.parse_args()
    if args.mode.startswith("publication"):
        publish(args.directory.resolve(), revised=args.mode == "publication-revised")
    else:
        benchmark(args.directory.resolve(), args.mode)
