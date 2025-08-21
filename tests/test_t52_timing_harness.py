import json
import math
import sys
import pytest

import forge3d as f3d


def _gpu_ok():
    info = f3d.device_probe()
    return isinstance(info, dict) and info.get("status") == "ok"


def _check_schema(res: dict):
    for k in ("op", "width", "height", "pixels", "iterations", "warmup", "stats", "throughput", "env"):
        assert k in res
    s = res["stats"]
    t = res["throughput"]
    for k in ("min_ms", "p50_ms", "mean_ms", "p95_ms", "max_ms", "std_ms"):
        assert k in s and s[k] >= 0.0
    for k in ("fps", "mpix_per_s"):
        assert k in t and t[k] >= 0.0
    assert res["width"] > 0 and res["height"] > 0


@pytest.mark.parametrize("op", ["numpy_to_png", "png_to_numpy"])
def test_cpu_only_png_bench_smoke(op):
    res = f3d.run_benchmark(op, 64, 64, iterations=3, warmup=1, seed=1)
    _check_schema(res)
    assert res["op"] == op


@pytest.mark.skipif(not _gpu_ok(), reason="No suitable GPU adapter")
@pytest.mark.parametrize("op", ["renderer_rgba", "scene_rgba"])
def test_gpu_bench_smoke(op):
    res = f3d.run_benchmark(op, 64, 64, iterations=3, warmup=1)
    _check_schema(res)
    assert res["op"] == op


@pytest.mark.skipif(not _gpu_ok(), reason="No suitable GPU adapter")
def test_renderer_png_bench_smoke():
    res = f3d.run_benchmark("renderer_png", 64, 64, iterations=2, warmup=1)
    _check_schema(res)