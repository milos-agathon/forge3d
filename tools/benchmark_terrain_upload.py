"""Reproduce native terrain upload cost and compare unmodified frame bytes.

Run from the repository root with its freshly rebuilt native extension:
    python tools/benchmark_terrain_upload.py --output artifacts/terrain-native-cost/before
    python tools/benchmark_terrain_upload.py --output artifacts/terrain-native-cost/after --compare artifacts/terrain-native-cost/before

No golden or certificate is written. Timings exclude fixture creation and array readback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import forge3d as f3d
from forge3d import _forge3d as native
from forge3d import diagnostics
from forge3d.terrain_params import (
    AovSettings, DetailSettings, HeightAoSettings, PomSettings, ShadowSettings,
    make_terrain_params_config,
)


def heightmap(size: int, kind: str) -> np.ndarray:
    if kind == "random":
        return np.random.default_rng(0).uniform(0.0, 2000.0, (size, size)).astype(np.float32)
    x = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(x, x)
    return np.ascontiguousarray(2000.0 * np.exp(-4.0 * (xx * xx + yy * yy)), dtype=np.float32)


def run(output: Path, compare: Path | None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    session = f3d.Session(window=False)
    renderer = f3d.TerrainRenderer(session)
    material = f3d.MaterialSet.custom((1.0, 1.0, 1.0), 0.0, 0.9, 1.0, 0.0, 1.0)
    hdr = output / "neutral.hdr"
    hdr.write_bytes(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 4 +X 8\n" + bytes([128, 128, 128, 128]) * 32)
    ibl = f3d.IBL.from_hdr(str(hdr), intensity=1.0)
    settings = dict(
        size_px=(1080, 1080), render_scale=1.0, terrain_span=41600.0,
        msaa_samples=1, z_scale=1.0, exposure=1.0, domain=(0.0, 2000.0),
        albedo_mode="colormap", colormap_strength=1.0, hue_variation_strength=0.0,
        light_azimuth_deg=-135.0, light_elevation_deg=38.0, sun_intensity=2.4,
        ibl_intensity=0.55, cam_radius=30000.0, cam_phi_deg=-110.0,
        cam_theta_deg=62.0, fov_y_deg=32.0, camera_mode="mesh:zup",
        clip=(600.0, 221600.0),
        shadows=ShadowSettings(False, "NONE", 4096, 1, 83200.0, 1.2, 0.75,
                               0.002, 0.0008, 0.0004, 1e-4, 0.0, 1.0, 1.0),
        pom=PomSettings(False, "Occlusion", 0.0, 1, 1, 0, False, False),
        detail=DetailSettings(enabled=False), height_ao=HeightAoSettings(enabled=False),
        aov=AovSettings(enabled=True, albedo=True, normal=True, depth=True),
    )
    params = f3d.TerrainRenderParams(make_terrain_params_config(**settings))
    # Prime pipeline/resource initialization outside the measured native calls.
    renderer.render_with_aov(material, ibl, params, heightmap(32, "smooth"))
    rows = []
    for kind in ("random", "smooth"):
        for size in (1490, 2080, 3000):
            heights = heightmap(size, kind)
            cpu_start = time.process_time()
            wall_start = time.perf_counter()
            frame, aov = renderer.render_with_aov(material, ibl, params, heights)
            wall_s = time.perf_counter() - wall_start
            cpu_s = time.process_time() - cpu_start
            report = diagnostics.render_certificate(sign=False)
            passes = report.get("passes", [])
            arrays = {"beauty": frame.to_numpy(), "depth": aov.depth(),
                      "normal": aov.normal(), "albedo": aov.albedo()}
            hashes = {}
            for name, array in arrays.items():
                filename = f"{kind}-{size}-{name}.npy"
                np.save(output / filename, array)
                hashes[name] = hashlib.sha256(array.tobytes()).hexdigest()
                if compare is not None:
                    reference = np.load(compare / filename)
                    if array.dtype != reference.dtype or array.shape != reference.shape or array.tobytes() != reference.tobytes():
                        raise AssertionError(f"Frame bytes changed: {filename}")
            row = dict(kind=kind, size=size, wall_s=wall_s, process_cpu_s=cpu_s,
                       gpu_ms=sum(p.get("gpu_ms", 0.0) for p in passes),
                       gpu_passes=passes, hashes=hashes, adapter=report.get("adapter"),
                       allocations=report.get("allocations"),
                       input_sha256=hashlib.sha256(heights.tobytes()).hexdigest())
            rows.append(row)
            print(json.dumps({key: row[key] for key in
                              ("kind", "size", "wall_s", "process_cpu_s", "gpu_ms")}), flush=True)
            (output / "timings.json").write_text(json.dumps(dict(
                module=f3d.__file__, version=f3d.__version__,
                native_module=native.__file__,
                native_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                settings={k: str(v) for k, v in settings.items()}, rows=rows,
            ), indent=2), encoding="utf-8")

    # Exercise consumers of coarse AO and renderer reuse with small fixed scenes.
    fixed = heightmap(96, "smooth")
    for mode in ("mesh", "mesh:zup", "screen"):
        for ao_weight, debug_mode, env_debug in ((0.0, 0, ""), (0.7, 0, ""),
                                                (0.0, 28, ""), (0.0, 0, "28")):
            label = f"fixed-{mode.replace(':', '-')}-{ao_weight}-{debug_mode}-{env_debug}"
            config = dict(settings, size_px=(192, 128), terrain_span=2.8,
                          domain=(0.0, 1.0), cam_radius=5.0, clip=(0.1, 100.0),
                          camera_mode=mode, ao_weight=ao_weight, debug_mode=debug_mode)
            small_params = f3d.TerrainRenderParams(make_terrain_params_config(**config))
            previous = os.environ.get("VF_COLOR_DEBUG_MODE")
            try:
                os.environ["VF_COLOR_DEBUG_MODE"] = env_debug
                beauty = renderer.render_terrain_pbr_pom(material, ibl, small_params, fixed / 2000.0)
                frame, aov = renderer.render_with_aov(material, ibl, small_params, fixed / 2000.0)
                offline_params = f3d.TerrainRenderParams(make_terrain_params_config(
                    **dict(config, aa_samples=16, aa_seed=12345),
                ))
                offline = f3d.render_offline(
                    renderer, material, ibl, offline_params, fixed / 2000.0,
                    settings=f3d.OfflineQualitySettings(enabled=True, adaptive=False, batch_size=4),
                )
                assert offline.metadata["samples_used"] == 16
            finally:
                if previous is None:
                    os.environ.pop("VF_COLOR_DEBUG_MODE", None)
                else:
                    os.environ["VF_COLOR_DEBUG_MODE"] = previous
            arrays = {"beauty": beauty.to_numpy(), "aov-beauty": frame.to_numpy(),
                      "depth": aov.depth(), "normal": aov.normal(), "albedo": aov.albedo(),
                      "offline-beauty": offline.frame.to_numpy(),
                      "offline-hdr": offline.hdr_frame.to_numpy_f32(),
                      "offline-depth": offline.aov_frame.depth(),
                      "offline-normal": offline.aov_frame.normal(),
                      "offline-albedo": offline.aov_frame.albedo()}
            for name, array in arrays.items():
                filename = f"{label}-{name}.npy"
                np.save(output / filename, array)
                if compare is not None:
                    reference = np.load(compare / filename)
                    if (array.shape, array.dtype, array.tobytes()) != (reference.shape, reference.dtype, reference.tobytes()):
                        raise AssertionError(f"Fixed scene bytes changed: {filename}")
    print("Fixed scenes completed; all saved frame bytes match." if compare else "Fixed scene baselines saved.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    run(args.output, args.compare)
