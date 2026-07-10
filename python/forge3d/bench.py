import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import forge3d as f3d


def _percentiles(ms: List[float]) -> Tuple[float, float, float]:
    if not ms:
        return (0.0, 0.0, 0.0)
    a = np.asarray(ms, dtype=np.float64)
    # Median (p50) and p95 via numpy for stability across platforms
    p50 = float(np.percentile(a, 50))
    p95 = float(np.percentile(a, 95))
    return (p50, p95, float(a.max()))


def _env_info() -> Dict[str, str]:
    info = f3d.device_probe()
    if not isinstance(info, dict):
        return {"status": "unknown"}
    out = {k: info.get(k) for k in ("status", "adapter_name", "backend", "device_type")}
    return out


def _bench_loop(fn, *, iterations: int, warmup: int) -> List[float]:
    # Warmup
    for _ in range(max(0, warmup)):
        fn()

    # Timed runs
    times_s: List[float] = []
    for _ in range(max(1, iterations)):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times_s.append(t1 - t0)
    return [t * 1000.0 for t in times_s]  # ms


def _memory_snapshot() -> Dict[str, Any]:
    return dict(f3d.memory_metrics())


def _memory_delta(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, float]:
    delta: Dict[str, float] = {}
    for key, after_value in after.items():
        before_value = before.get(key)
        if isinstance(after_value, (int, float)) and isinstance(before_value, (int, float)):
            delta[key] = float(after_value) - float(before_value)
    return delta


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _gpu_timing_snapshot(render_metadata: Dict[str, Any] | None = None) -> Dict[str, float | None | bool]:
    timings: Dict[str, float | None | bool] = {
        "available": False,
        "terrain_main_pass_ms": None,
        "vt_upload_avg_ms": None,
        "offline_accumulation_ms": None,
    }
    metadata = render_metadata if isinstance(render_metadata, dict) else {}
    vt_stats = metadata.get("material_vt_stats")
    if isinstance(vt_stats, dict):
        timings["vt_upload_avg_ms"] = _float_or_none(vt_stats.get("avg_upload_ms"))
    timings["terrain_main_pass_ms"] = _float_or_none(metadata.get("terrain_main_pass_ms"))
    timings["offline_accumulation_ms"] = _float_or_none(metadata.get("offline_accumulation_ms"))
    timings["available"] = any(
        timings[key] is not None
        for key in ("terrain_main_pass_ms", "vt_upload_avg_ms", "offline_accumulation_ms")
    )
    return timings


def _memory_tracking_snapshot(
    width: int,
    height: int,
    memory_after: Dict[str, Any],
    render_metadata: Dict[str, Any] | None = None,
) -> Dict[str, float | int | str]:
    metadata = render_metadata if isinstance(render_metadata, dict) else {}
    output_bytes = int(width) * int(height) * 4
    vt_stats = metadata.get("material_vt_stats")
    vt_bytes = 0
    if isinstance(vt_stats, dict):
        for key in ("resident_bytes", "resident_tile_bytes", "atlas_bytes"):
            value = vt_stats.get(key)
            if isinstance(value, (int, float)):
                vt_bytes = max(vt_bytes, int(value))
    expected = output_bytes + vt_bytes
    tracked = int(max(memory_after.get("total_bytes", 0), memory_after.get("peak_total_bytes", 0)))
    coverage = 1.0 if expected <= 0 else tracked / float(expected)
    return {
        "expected_bytes": expected,
        "tracked_bytes": tracked,
        "coverage_ratio": coverage,
        "status": "supported" if coverage >= 0.95 else "underdeveloped",
    }


def _op_renderer_rgba(width: int, height: int):
    r = f3d.Renderer(width, height)
    def step():
        _ = r.render_triangle_rgba()
    return step


def _op_renderer_png(width: int, height: int):
    r = f3d.Renderer(width, height)
    tmpdir = Path(tempfile.mkdtemp(prefix="vf_bench_"))
    def step():
        p = tmpdir / f"frame_{time.time_ns()}.png"
        r.render_triangle_png(str(p))
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass
    return step


def _op_scene_rgba(width: int, height: int, *, grid: int = 16, colormap: str = "viridis"):
    sc = f3d.Scene(width, height, grid=grid, colormap=colormap)
    def step():
        _ = sc.render_rgba()
    return step


def _op_numpy_to_png(width: int, height: int, *, seed: int = 0):
    rng = np.random.default_rng(seed)
    arr = (rng.random((height, width, 4)) * 255).astype(np.uint8)

    tmpdir = Path(tempfile.mkdtemp(prefix="vf_bench_png_"))
    def step():
        p = tmpdir / f"img_{time.time_ns()}.png"
        f3d.numpy_to_png(str(p), arr)
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass
    return step


def _op_png_to_numpy(width: int, height: int, *, seed: int = 0):
    # Pre-create a PNG once, then measure load path only
    rng = np.random.default_rng(seed)
    arr = (rng.random((height, width, 4)) * 255).astype(np.uint8)
    tmpdir = Path(tempfile.mkdtemp(prefix="vf_bench_pngin_"))
    src = tmpdir / "src.png"
    f3d.numpy_to_png(str(src), arr)

    def step():
        _ = f3d.png_to_numpy(str(src))
    return step


def _op_mapscene_terrain_png(width: int, height: int, *, vt_active: bool = False):
    tmpdir = Path(tempfile.mkdtemp(prefix="vf_bench_mapscene_"))
    dem = np.zeros((max(2, height // 8), max(2, width // 8)), dtype=np.float32)
    metadata: Dict[str, Any] = {
        "width": int(dem.shape[1]),
        "height": int(dem.shape[0]),
        "source_id": "bench-dem",
    }
    lighting_settings: Dict[str, Any] = {}
    if vt_active:
        metadata["virtual_texture"] = {
            "enabled": True,
            "families": [
                {
                    "family": "albedo",
                    "virtual_size_px": [512, 512],
                    "tile_size": 120,
                    "tile_border": 4,
                }
            ],
            "atlas_size": 1024,
            "residency_budget_mb": 16.0,
            "max_mip_levels": 4,
            "use_feedback": True,
            "procedural_sources": True,
            "source_count": 4,
            "source_size": 512,
            "pattern": "checker",
        }
        lighting_settings.update({"albedo_mode": "material", "colormap_strength": 0.0})
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=dem,
            crs="EPSG:32610",
            metadata=metadata,
        ),
        camera=f3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=100.0),
        lighting=f3d.LightingPreset(name="daylight", settings=lighting_settings),
        output=f3d.OutputSpec(width=int(width), height=int(height), format="png"),
    )

    def step():
        p = tmpdir / f"mapscene_{time.time_ns()}.png"
        scene.render(str(p))
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass

    def metadata():
        value = getattr(scene, "last_render_metadata", None)
        return dict(value) if isinstance(value, dict) else {}

    return step, metadata


def run_benchmark(
    op: str,
    width: int,
    height: int,
    *,
    iterations: int = 100,
    warmup: int = 10,
    grid: int = 16,
    colormap: str = "viridis",
    seed: int = 0,
) -> Dict:
    """
    Run a timing benchmark for a given operation.

    Outside CENSOR's render-certificate scope: this is a benchmark harness,
    not one deliverable render.

    Parameters
    ----------
    op : {'renderer_rgba','renderer_png','scene_rgba','numpy_to_png','png_to_numpy',
          'mapscene_terrain_png','mapscene_terrain_vt_png'}
    width, height : int
        Resolution (for render ops) or array dimensions (for PNG ops).
    iterations : int, default 100
    warmup : int, default 10
    grid : int, default 16
        Grid resolution for Scene (ignored for other ops).
    colormap : str, default 'viridis'
        Colormap for Scene (ignored for other ops).
    seed : int, default 0
        RNG seed for PNG path workloads.

    Returns
    -------
    dict
        {
          'op': str, 'width': int, 'height': int, 'pixels': int,
          'iterations': int, 'warmup': int,
          'stats': {'min_ms','p50_ms','mean_ms','p95_ms','max_ms','std_ms'},
          'throughput': {'fps','mpix_per_s'},
          'env': {'status','adapter_name','backend','device_type'}
        }
    """
    op = op.lower().strip()
    env = _env_info()

    metadata_probe: Callable[[], Dict[str, Any]] = lambda: {}

    # GPU-dependent ops should be skipped/guarded by caller if env['status'] != 'ok'
    if op == "renderer_rgba":
        step = _op_renderer_rgba(width, height)
    elif op == "renderer_png":
        step = _op_renderer_png(width, height)
    elif op == "scene_rgba":
        step = _op_scene_rgba(width, height, grid=grid, colormap=colormap)
    elif op == "numpy_to_png":
        step = _op_numpy_to_png(width, height, seed=seed)
    elif op == "png_to_numpy":
        step = _op_png_to_numpy(width, height, seed=seed)
    elif op == "mapscene_terrain_png":
        step, metadata_probe = _op_mapscene_terrain_png(width, height)
    elif op == "mapscene_terrain_vt_png":
        step, metadata_probe = _op_mapscene_terrain_png(width, height, vt_active=True)
    else:
        raise ValueError(
            "unknown op; expected one of: renderer_rgba, renderer_png, scene_rgba, "
            "numpy_to_png, png_to_numpy, mapscene_terrain_png, mapscene_terrain_vt_png"
        )

    memory_before = _memory_snapshot()
    ms = _bench_loop(step, iterations=iterations, warmup=warmup)
    memory_after = _memory_snapshot()

    mean_ms = float(statistics.fmean(ms)) if ms else 0.0
    std_ms = float(statistics.pstdev(ms)) if len(ms) > 1 else 0.0
    p50_ms, p95_ms, max_ms = _percentiles(ms)
    min_ms = min(ms) if ms else 0.0
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    mpix_per_s = (width * height / 1e6) * fps
    render_metadata = metadata_probe()

    return {
        "op": op,
        "width": int(width),
        "height": int(height),
        "pixels": int(width * height),
        "iterations": int(iterations),
        "warmup": int(warmup),
        "stats": {
            "min_ms": float(min_ms),
            "p50_ms": float(p50_ms),
            "mean_ms": float(mean_ms),
            "p95_ms": float(p95_ms),
            "max_ms": float(max_ms),
            "std_ms": float(std_ms),
        },
        "throughput": {
            "fps": float(fps),
            "mpix_per_s": float(mpix_per_s),
        },
        "env": env,
        "memory": {
            "before": memory_before,
            "after": memory_after,
            "delta": _memory_delta(memory_before, memory_after),
            "tracking": _memory_tracking_snapshot(width, height, memory_after, render_metadata),
        },
        "gpu_timings": _gpu_timing_snapshot(render_metadata),
    }


def run_vt_frame_time_comparison(
    width: int,
    height: int,
    *,
    iterations: int = 10,
    warmup: int = 2,
) -> Dict[str, Any]:
    """Measure baseline and VT-active MapScene render times through the public path."""
    baseline = run_benchmark(
        "mapscene_terrain_png",
        width,
        height,
        iterations=iterations,
        warmup=warmup,
    )
    vt_active = run_benchmark(
        "mapscene_terrain_vt_png",
        width,
        height,
        iterations=iterations,
        warmup=warmup,
    )
    baseline_mean = float(baseline["stats"]["mean_ms"])
    vt_mean = float(vt_active["stats"]["mean_ms"])
    return {
        "width": int(width),
        "height": int(height),
        "iterations": int(iterations),
        "warmup": int(warmup),
        "baseline": baseline,
        "vt_active": vt_active,
        "delta_ms": vt_mean - baseline_mean,
        "delta_pct": ((vt_mean - baseline_mean) / baseline_mean * 100.0) if baseline_mean > 0.0 else 0.0,
        "vt_upload_avg_ms": vt_active["gpu_timings"].get("vt_upload_avg_ms"),
        "vt_gpu_timings_available": bool(vt_active["gpu_timings"].get("available")),
    }


def _print_table(result: Dict):
    def f(x): return f"{x:.3f}"
    s = result["stats"]
    t = result["throughput"]
    env = result.get("env", {})
    mem = result.get("memory", {}).get("after", {})
    gpu = result.get("gpu_timings", {})
    lines = [
        f"op={result['op']} | {result['width']}x{result['height']} px ({result['pixels']/1e6:.2f} MPix)",
        f"iters={result['iterations']} warmup={result['warmup']}",
        f"min/median/mean/p95/max/std (ms): {f(s['min_ms'])} / {f(s['p50_ms'])} / {f(s['mean_ms'])} / {f(s['p95_ms'])} / {f(s['max_ms'])} / {f(s['std_ms'])}",
        f"throughput: {f(t['fps'])} FPS, {f(t['mpix_per_s'])} MPix/s",
        f"memory: host_visible={mem.get('host_visible_bytes')} limit={mem.get('limit_bytes')} policy={mem.get('budget_policy')}",
        "gpu timings (ms): "
        f"terrain_main={gpu.get('terrain_main_pass_ms')} "
        f"vt_upload_avg={gpu.get('vt_upload_avg_ms')} "
        f"offline_accumulation={gpu.get('offline_accumulation_ms')}",
        f"env: status={env.get('status')} adapter={env.get('adapter_name')} backend={env.get('backend')} type={env.get('device_type')}",
    ]
    print("\n".join(lines))


def main(argv=None):
    p = argparse.ArgumentParser(description="forge3d timing harness")
    p.add_argument("--op", required=True,
                   choices=[
                       "renderer_rgba",
                       "renderer_png",
                       "scene_rgba",
                       "numpy_to_png",
                       "png_to_numpy",
                       "mapscene_terrain_png",
                       "mapscene_terrain_vt_png",
                   ])
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--grid", type=int, default=16)
    p.add_argument("--colormap", type=str, default="viridis")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", type=str, default=None, help="Write JSON result to this path")
    args = p.parse_args(argv)

    res = run_benchmark(
        args.op, args.width, args.height,
        iterations=args.iterations, warmup=args.warmup,
        grid=args.grid, colormap=args.colormap, seed=args.seed,
    )
    _print_table(res)
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2))
