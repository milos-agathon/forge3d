"""D04: render color, relief and transparent vectors, then replay composition.

Run from an installed Forge3D environment:
    python docs/examples/mapscene_render_passes.py --output artifacts/d04/example
No image substitutes or adapter skips are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import forge3d as f3d
from forge3d._png import load_png_rgba
from forge3d.helpers.offscreen import save_png_deterministic


def compose_example(output: Path, width: int = 256, height: int = 192) -> dict:
    adapter = f3d.device_probe()
    if adapter.get("status") != "ok" or adapter.get("software_fallback") or adapter.get("device_type") not in ("DiscreteGpu", "IntegratedGpu"):
        raise RuntimeError(f"D04 example requires a physical GPU: {adapter}")
    output.mkdir(parents=True, exist_ok=True)
    x, y = np.meshgrid(np.linspace(-1, 1, 32), np.linspace(-1, 1, 32))
    terrain = f3d.TerrainSource(data=(120 * np.exp(-3 * (x*x + y*y))).astype(np.float32),
                              crs="EPSG:3857", metadata={"source_id": "d04-example-hill"})
    camera = f3d.OrbitCamera(distance=240, elevation_deg=60, azimuth_deg=30)
    source_reports = {}
    source_metrics = {}
    images = {}
    for name, settings in (("color", {"albedo_mode": "colormap"}),
                           ("relief", {"albedo_mode": "material", "colormap_strength": 0.0})):
        scene = f3d.MapScene(terrain=terrain, camera=camera,
                            lighting=f3d.LightingPreset(settings=settings),
                            output=f3d.OutputSpec(width, height),
                            reproducibility_profile=f3d.ReproducibilityProfile(seed=42))
        start = time.perf_counter()
        report = scene.render(str(output / f"{name}.png"))
        source_metrics[name] = {"wall_ms": (time.perf_counter() - start) * 1000,
                                "backend": scene.last_render_backend,
                                "render_metadata": scene.last_render_metadata}
        source_reports[name] = report.to_dict()
        images[name] = f3d.RenderPassInput(load_png_rgba(output / f"{name}.png"))
    start = time.perf_counter()
    overlay = np.asarray(f3d.vector_render_oit_py(
        # The native vector API takes normalized device coordinates.
        width, height, polylines=[[(-0.85, 0.0), (0.0, 0.35), (0.85, 0.0)]],
        polyline_rgba=[(0.05, 0.4, 1.0, 0.6)], stroke_width=[4.0]), dtype=np.uint8)
    source_metrics["overlay"] = {"wall_ms": (time.perf_counter() - start) * 1000,
                                 "backend": "native_vector_oit"}
    if not (overlay[..., 3] > 0).any() or not (overlay[..., 3] == 0).any():
        raise RuntimeError("native overlay must contain both coverage and transparency")
    save_png_deterministic(output / "overlay.png", overlay)
    # OIT resolves straight RGB into Rgba8Unorm, without an sRGB transfer.
    images["vectors"] = f3d.RenderPassInput(overlay, color_space="linear")
    passes = [f3d.RenderPassSpec("color", "color", ("color_image",)),
              f3d.RenderPassSpec("relief", "relief", ("color", "relief_image"), "multiply", {"opacity": 0.5}),
              f3d.RenderPassSpec("overlay", "overlay", ("relief", "vectors"), "alpha_over")]
    scene = f3d.MapScene(terrain=terrain, camera=camera, lighting=f3d.LightingPreset(),
                        output=f3d.OutputSpec(width, height))
    start = time.perf_counter()
    report = scene.render_passes(passes, {"color_image": images["color"], "relief_image": images["relief"],
                                         "vectors": images["vectors"]}, str(output / "composed.png"))
    composition_ms = (time.perf_counter() - start) * 1000
    scene.save_bundle(output / "composition")
    replay = f3d.MapScene.load_bundle(output / "composition.forge3d")
    replay_report = replay.render_passes(path=str(output / "replay.png"))
    if report.to_dict() != replay_report.to_dict() or (output / "composed.png").read_bytes() != (output / "replay.png").read_bytes():
        raise RuntimeError("D04 bundle replay changed pixels or validation report")
    result = {"adapter": adapter, "dimensions": [width, height], "source_metrics": source_metrics,
              "source_reports": source_reports, "composition_wall_ms": composition_ms,
              "composition_backend": scene.last_render_backend, "report": report.to_dict(),
              "replay_pixel_and_report_equal": True,
              "sha256": {name: hashlib.sha256((output / f"{name}.png").read_bytes()).hexdigest()
                         for name in ("color", "relief", "overlay", "composed", "replay")}}
    (output / "measurements.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/d04/example"))
    args = parser.parse_args()
    result = compose_example(args.output)
    print(json.dumps({key: result[key] for key in ("adapter", "composition_wall_ms", "sha256", "replay_pixel_and_report_equal")}, indent=2))
