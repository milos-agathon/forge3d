"""Measure the existing Tracy Arm render_core path with local DEM/imagery.

Inputs remain read-only; render_core's temporary map directory is redirected
under --output. Run the same command with the baseline and corrected binaries.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import forge3d as f3d
import numpy as np
from scipy.ndimage import zoom


def digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def serialize(value):
    if dataclasses.is_dataclass(value):
        return {field.name: serialize(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, (tuple, list)):
        return [serialize(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    # Native overlay objects have no serialized public configuration. Their
    # exact source RGBA bytes are hashed separately below.
    return type(value).__name__


def run(core_path: Path, data: Path, output: Path, compare: Path | None):
    output.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("tracy_render_core", core_path)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    core.MAPDIR = str(output / "maps")
    original_config = core.make_terrain_params_config
    configs = []

    def capture_config(**kwargs):
        config = original_config(**kwargs)
        configs.append(serialize(config))
        return config

    core.make_terrain_params_config = capture_config
    renderer = core.Renderer()
    native = renderer.r
    calls = []

    class TimedNative:
        def render_with_aov(self, **kwargs):
            cpu_start = time.process_time()
            wall_start = time.perf_counter()
            result = native.render_with_aov(**kwargs)
            wall = time.perf_counter() - wall_start
            cpu = time.process_time() - cpu_start
            report = f3d.diagnostics.render_certificate(sign=False)
            calls.append(dict(native_wall_s=wall, native_process_cpu_s=cpu,
                              gpu_ms=sum(p.get("gpu_ms", 0.0) for p in report["passes"]),
                              adapter=report["adapter"], allocations=report["allocations"],
                              native_height_sha256=digest(kwargs["heightmap"])))
            return result

    renderer.r = TimedNative()
    render_settings = dict(target=core.utm_to_world(613200, 6413800, 300),
                           radius=9000, bearing=200, theta=62, want_depth=True)
    # Initialize the same pipelines outside the paired native-call timing.
    renderer.render(np.linspace(0, 1, 32 * 32, dtype=np.float32).reshape(32, 32),
                    np.full((32, 32, 4), 255, np.uint8), **render_settings)
    dem_path, rgba_path = data / "render_dem_10m.npy", data / "base_rgba.npy"
    full = np.maximum(np.load(dem_path), 0).astype(np.float32)
    rgba_full = np.load(rgba_path)
    rows = []
    metadata = dict(version=f3d.__version__, module=f3d.__file__, numpy_version=np.__version__,
                    native_sha256=hashlib.sha256(Path(importlib.util.find_spec("forge3d._forge3d").origin).read_bytes()).hexdigest(),
                    core_sha256=hashlib.sha256(core_path.read_bytes()).hexdigest(),
                    source_dem_sha256=digest(full), source_rgba_sha256=digest(rgba_full),
                    render_settings=render_settings, rows=rows)
    for size in (1490, 2080, 3000, 4160):
        if size == 1490:
            heights = np.maximum(np.load(data / "far_dem_40m.npy"), 0).astype(np.float32)
            image = zoom(rgba_full, (size / 4160, size / 4160, 1), order=1)
            source = "far_dem_40m.npy; resampled base_rgba.npy"
        elif size == 2080:
            heights, image = full[::2, ::2].copy(), rgba_full[::2, ::2].copy()
            source = "render_dem_10m.npy and base_rgba.npy, stride 2"
        elif size == 3000:
            heights = zoom(full, size / 4160, order=1)
            image = zoom(rgba_full, (size / 4160, size / 4160, 1), order=1)
            source = "render_dem_10m.npy and base_rgba.npy, bilinear to 3000 squared"
        else:
            heights, image = full, rgba_full
            source = "unresampled render_dem_10m.npy and base_rgba.npy"
        started = time.perf_counter()
        beauty, depth = renderer.render(heights, image, **render_settings)
        wrapper_wall = time.perf_counter() - started
        row = dict(calls[-1], size=size, source=source, wrapper_wall_s=wrapper_wall,
                   height_sha256=digest(heights), rgba_sha256=digest(image),
                   settings=configs[-1], hashes={})
        for name, array in (("beauty", beauty), ("depth", depth)):
            filename = f"tracy-{size}-{name}.npy"
            np.save(output / filename, array)
            row["hashes"][name] = digest(array)
            if compare is not None:
                before = np.load(compare / filename)
                assert (array.shape, array.dtype, array.tobytes()) == (before.shape, before.dtype, before.tobytes()), filename
        if compare is not None:
            previous = json.loads((compare / "timings.json").read_text())["rows"][len(rows)]
            for key in ("settings", "height_sha256", "rgba_sha256", "native_height_sha256"):
                assert row[key] == previous[key], key
        rows.append(row)
        (output / "timings.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps({key: row[key] for key in ("size", "native_wall_s", "native_process_cpu_s", "gpu_ms", "wrapper_wall_s")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    run(args.core, args.data, args.output, args.compare)
