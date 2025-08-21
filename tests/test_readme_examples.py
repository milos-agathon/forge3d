import sys, subprocess, pytest

def _has_gpu():
    try:
        import forge3d as f3d
        info = f3d.device_probe()
        return isinstance(info, dict) and info.get("status") in ("ok","error","unsupported")
    except Exception:
        return False

@pytest.mark.skipif(not _has_gpu(), reason="No compatible GPU / wgpu backend available")
@pytest.mark.parametrize("script", [
    "examples/triangle_png.py",
    "examples/scene_terrain_demo.py",
    "examples/terrain_normalize_demo.py",
])
def test_gpu_examples_run(script):
    subprocess.check_call([sys.executable, script])

def test_cpu_examples_run():
    subprocess.check_call([sys.executable, "examples/png_numpy_roundtrip.py"])

def test_diagnostics_runs():
    subprocess.check_call([sys.executable, "examples/diagnostics.py"])

def test_grid_generate_runs():
    subprocess.check_call([sys.executable, "examples/grid_generate_demo.py"])

def test_bench_runs():
    subprocess.check_call([sys.executable, "examples/run_bench.py"])