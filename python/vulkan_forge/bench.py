import argparse
import json
import math
import os
import statistics
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import vulkan_forge as vf


def _percentiles(ms: List[float]) -> Tuple[float, float, float]:
    if not ms:
        return (0.0, 0.0, 0.0)
    a = np.asarray(ms, dtype=np.float64)
    # Median (p50) and p95 via numpy for stability across platforms
    p50 = float(np.percentile(a, 50))
    p95 = float(np.percentile(a, 95))
    return (p50, p95, float(a.max()))


def _env_info() -> Dict[str, str]:
    info = vf.device_probe()
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


def _op_renderer_rgba(width: int, height: int):
    r = vf.Renderer(width, height)
    def step():
        _ = r.render_triangle_rgba()
    return step


def _op_renderer_png(width: int, height: int):
    r = vf.Renderer(width, height)
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
    sc = vf.Scene(width, height, grid=grid, colormap=colormap)
    def step():
        _ = sc.render_rgba()
    return step


def _op_numpy_to_png(width: int, height: int, *, seed: int = 0):
    rng = np.random.default_rng(seed)
    arr = (rng.random((height, width, 4)) * 255).astype(np.uint8)

    tmpdir = Path(tempfile.mkdtemp(prefix="vf_bench_png_"))
    def step():
        p = tmpdir / f"img_{time.time_ns()}.png"
        vf.numpy_to_png(str(p), arr)
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
    vf.numpy_to_png(str(src), arr)

    def step():
        _ = vf.png_to_numpy(str(src))
    return step


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

    Parameters
    ----------
    op : {'renderer_rgba','renderer_png','scene_rgba','numpy_to_png','png_to_numpy'}
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
    else:
        raise ValueError("unknown op; expected one of: renderer_rgba, renderer_png, scene_rgba, numpy_to_png, png_to_numpy")

    ms = _bench_loop(step, iterations=iterations, warmup=warmup)

    mean_ms = float(statistics.fmean(ms)) if ms else 0.0
    std_ms = float(statistics.pstdev(ms)) if len(ms) > 1 else 0.0
    p50_ms, p95_ms, max_ms = _percentiles(ms)
    min_ms = min(ms) if ms else 0.0
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    mpix_per_s = (width * height / 1e6) * fps

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
    }


def _print_table(result: Dict):
    def f(x): return f"{x:.3f}"
    s = result["stats"]
    t = result["throughput"]
    env = result.get("env", {})
    lines = [
        f"op={result['op']} | {result['width']}x{result['height']} px ({result['pixels']/1e6:.2f} MPix)",
        f"iters={result['iterations']} warmup={result['warmup']}",
        f"min/median/mean/p95/max/std (ms): {f(s['min_ms'])} / {f(s['p50_ms'])} / {f(s['mean_ms'])} / {f(s['p95_ms'])} / {f(s['max_ms'])} / {f(s['std_ms'])}",
        f"throughput: {f(t['fps'])} FPS, {f(t['mpix_per_s'])} MPix/s",
        f"env: status={env.get('status')} adapter={env.get('adapter_name')} backend={env.get('backend')} type={env.get('device_type')}",
    ]
    print("\n".join(lines))


def main(argv=None):
    p = argparse.ArgumentParser(description="vulkan_forge timing harness")
    p.add_argument("--op", required=True,
                   choices=["renderer_rgba", "renderer_png", "scene_rgba", "numpy_to_png", "png_to_numpy"])
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