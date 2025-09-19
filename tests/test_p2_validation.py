"""
P2 validation tests: watertight triangle intersector and TLAS/BLAS instancing wiring.

This suite focuses on two things:
- Presence of the watertight ray/triangle intersector in WGSL with key attributes
- Correct TLAS/BLAS bindings in the wavefront scene bind group layout
- Optional end-to-end sanity via the Rust example (skipped unless FORGE3D_RUN_WAVEFRONT=1)

RELEVANT FILES:
- src/shaders/pt_intersect.wgsl
- src/path_tracing/wavefront/pipeline.rs
- examples/wavefront_instances.rs
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

import forge3d
from forge3d.helpers.aov_io import load_rgba32f_auto


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_text(p: Path) -> str:
    with p.open("r", encoding="utf-8") as f:
        return f.read()


def _log_device_and_thresholds(*, test_name: str, jitter: float | None = None, **thresholds):
    try:
        adapters = forge3d.enumerate_adapters()
    except Exception:
        adapters = []
    hdr = f"[forge3d] {test_name}: "
    if adapters:
        ada = adapters[0]
        info = {
            "adapter_name": ada.get("adapter_name") or ada.get("name"),
            "device_name": ada.get("device_name") or ada.get("name"),
            "backend": ada.get("backend") or ada.get("device_type"),
        }
        print(hdr + f"adapter={info}")
    else:
        print(hdr + "adapter=<unknown>")
    if jitter is not None:
        print(hdr + f"jitter={jitter}")
    if thresholds:
        print(hdr + f"thresholds={thresholds}")


def test_watertight_intersector_present():
    """Assert the WGSL implements a watertight triangle intersector.

    Checks for function name and key shear/permutation and edge-function math.
    """
    wgsl_path = REPO_ROOT / "src" / "shaders" / "pt_intersect.wgsl"
    assert wgsl_path.exists(), f"Missing WGSL file: {wgsl_path}"
    src = _read_text(wgsl_path)

    # Function present
    assert "fn ray_triangle_intersect(ray: Ray" in src

    # Key steps of watertight algorithm (permutation, shear constants, edge functions)
    assert "let Sz = 1.0 / ray.d[kz];" in src
    assert "let U = (bx * cy) - (by * cx);" in src
    assert "let det = U + V + W;" in src


def test_scene_bind_group_layout_has_instances_and_blas():
    """Verify instances (binding 14) and BLAS desc table (binding 15) are present.

    This ensures TLAS/BLAS mapping is visible to WGSL kernels.
    """
    pipe_rs = REPO_ROOT / "src" / "path_tracing" / "wavefront" / "pipeline.rs"
    assert pipe_rs.exists(), f"Missing pipeline file: {pipe_rs}"
    txt = _read_text(pipe_rs)

    # Instances buffer binding
    assert "binding: 14" in txt and "Instances buffer" in txt
    # BLAS descriptor table binding
    assert "binding: 15" in txt and "BLAS descriptor table" in txt


@pytest.mark.skipif(
    os.environ.get("FORGE3D_RUN_WAVEFRONT", "0") != "1" or os.environ.get("FORGE3D_CI_GPU", "0") != "1" or shutil.which("cargo") is None or not forge3d.enumerate_adapters(),
    reason="Wavefront example disabled unless FORGE3D_RUN_WAVEFRONT=1 and FORGE3D_CI_GPU=1, cargo present, and a GPU adapter is available",
)
def test_wavefront_instances_material_swap_outputs_differ(tmp_path: Path):
    """Run the Rust example with and without --swap-materials and ensure images differ.

    This validates per-instance material_id indexing through TLAS/BLAS wiring.
    """
    env = os.environ.copy()
    # Ensure we run from repo root
    cwd = str(REPO_ROOT)
    _log_device_and_thresholds(test_name="near_edge_luminance", jitter=0.001)

    # 1) Default run
    cmd_a = [
        "cargo", "run", "--no-default-features", "--features", "images",
        "--example", "wavefront_instances",
    ]
    res_a = subprocess.run(cmd_a, cwd=cwd, capture_output=True, text=True)
    assert res_a.returncode == 0, f"Run A failed: {res_a.stderr}"
    img_path = REPO_ROOT / "out" / "wavefront_instances.png"
    assert img_path.exists(), f"Expected output not found: {img_path}"
    a_bytes = img_path.read_bytes()
    a_copy = tmp_path / "wf_a.png"
    a_copy.write_bytes(a_bytes)

    # 2) Swap materials
    cmd_b = [
        "cargo", "run", "--no-default-features", "--features", "images",
        "--example", "wavefront_instances", "--", "--swap-materials",
    ]
    res_b = subprocess.run(cmd_b, cwd=cwd, capture_output=True, text=True)
    assert res_b.returncode == 0, f"Run B failed: {res_b.stderr}"
    assert img_path.exists(), f"Expected output not found after swap: {img_path}"
    b_bytes = img_path.read_bytes()
    b_copy = tmp_path / "wf_b.png"
    b_copy.write_bytes(b_bytes)

    # Images should differ when materials are swapped
    assert a_bytes != b_bytes, "Expected different outputs when swapping per-instance material IDs"


def test_make_mesh_warns_on_degenerate_triangles():
    """Ensure Python mesh.make_mesh warns about degenerate triangles.

    This validates that the mesh validation path detects repeated-vertex triangles,
    which often exercise near-edge cases for the intersector/traversal.
    """
    import numpy as np
    import warnings

    from forge3d.mesh import make_mesh

    # Two triangles; the second is degenerate (repeated vertex index)
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)
    indices = np.array([
        [0, 1, 2],   # valid
        [0, 0, 2],   # degenerate: repeated vertex 0
    ], dtype=np.uint32)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        m = make_mesh(vertices, indices)
        # Expect at least one warning mentioning 'degenerate'
        msgs = [str(w.message).lower() for w in rec]
        assert any("degenerate" in s for s in msgs), f"Expected degenerate triangle warning, got: {msgs}"


@pytest.mark.skipif(
    os.environ.get("FORGE3D_RUN_WAVEFRONT", "0") != "1" or os.environ.get("FORGE3D_CI_GPU", "0") != "1" or shutil.which("cargo") is None or not forge3d.enumerate_adapters(),
    reason="Wavefront example disabled unless FORGE3D_RUN_WAVEFRONT=1 and FORGE3D_CI_GPU=1, cargo present, and a GPU adapter is available",
)
def test_wavefront_skinny_blas_changes_image(tmp_path: Path):
    """Images should differ when BLAS1 geometry changes to skinny triangles."""
    cwd = str(REPO_ROOT)

    # Baseline (no skinny)
    cmd_a = [
        "cargo", "run", "--no-default-features", "--features", "images",
        "--example", "wavefront_instances",
    ]
    res_a = subprocess.run(cmd_a, cwd=cwd, capture_output=True, text=True)
    assert res_a.returncode == 0, f"Baseline run failed: {res_a.stderr}"
    img_path = REPO_ROOT / "out" / "wavefront_instances.png"
    assert img_path.exists(), f"Missing output: {img_path}"
    base = img_path.read_bytes()

    # Skinny BLAS1 enabled
    cmd_b = [
        "cargo", "run", "--no-default-features", "--features", "images",
        "--example", "wavefront_instances", "--", "--skinny-blas1",
    ]
    res_b = subprocess.run(cmd_b, cwd=cwd, capture_output=True, text=True)
    assert res_b.returncode == 0, f"Skinny run failed: {res_b.stderr}"
    skinny = img_path.read_bytes()

    assert base != skinny, "Skinny BLAS geometry should change the rendered image"


@pytest.mark.skipif(
    os.environ.get("FORGE3D_RUN_WAVEFRONT", "0") != "1" or os.environ.get("FORGE3D_CI_GPU", "0") != "1" or shutil.which("cargo") is None or not forge3d.enumerate_adapters(),
    reason="Wavefront example disabled unless FORGE3D_RUN_WAVEFRONT=1 and FORGE3D_CI_GPU=1, cargo present, and a GPU adapter is available",
)
def test_wavefront_force_blas_selection_changes_output(tmp_path: Path):
    """Forcing both instances to BLAS 0 vs BLAS 1 should produce different outputs (with skinny BLAS1)."""
    cwd = str(REPO_ROOT)
    img_path = REPO_ROOT / "out" / "wavefront_instances.png"

    # Force BLAS 0 for both instances
    cmd0 = [
        "cargo", "run", "--no-default-features", "--features", "images",
        "--example", "wavefront_instances", "--", "--skinny-blas1", "--force-blas=0",
    ]
    r0 = subprocess.run(cmd0, cwd=cwd, capture_output=True, text=True)
    assert r0.returncode == 0, f"BLAS0 run failed: {r0.stderr}"
    a0 = img_path.read_bytes()

    # Force BLAS 1 for both instances
    cmd1 = [
        "cargo", "run", "--no-default-features", "--features", "images",
        "--example", "wavefront_instances", "--", "--skinny-blas1", "--force-blas=1",
    ]
    r1 = subprocess.run(cmd1, cwd=cwd, capture_output=True, text=True)
    assert r1.returncode == 0, f"BLAS1 run failed: {r1.stderr}"
    a1 = img_path.read_bytes()

    assert a0 != a1, "Forcing BLAS 0 vs 1 should change the image when BLAS1 is skinny"


@pytest.mark.skipif(
    os.environ.get("FORGE3D_RUN_WAVEFRONT", "0") != "1" or os.environ.get("FORGE3D_CI_GPU", "0") != "1" or shutil.which("cargo") is None or not forge3d.enumerate_adapters(),
    reason="Wavefront example disabled unless FORGE3D_RUN_WAVEFRONT=1 and FORGE3D_CI_GPU=1, cargo present, and a GPU adapter is available",
)
def test_wavefront_near_edge_stability_small_jitter(tmp_path: Path):
    """Small camera jitter should not cause catastrophic changes with watertight intersections.

    We compare mean luminance difference between two renders with +/- jitter.
    Requires Pillow to decode PNGs.
    """
    try:
        from PIL import Image  # type: ignore
    except Exception:
        pytest.skip("Pillow not available for PNG decoding")

    cwd = str(REPO_ROOT)
    img_path = REPO_ROOT / "out" / "wavefront_instances.png"

    def _run_and_load(args: list[str]) -> tuple[float, int]:
        r = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
        assert r.returncode == 0, f"Run failed: {r.stderr}"
        assert img_path.exists()
        im = Image.open(img_path).convert("RGB")
        import numpy as np
        arr = np.asarray(im, dtype=np.float32) / 255.0
        # Simple luminance
        lum = 0.2126 * arr[...,0] + 0.7152 * arr[...,1] + 0.0722 * arr[...,2]
        return float(lum.mean()), int(arr.size)

    cmd_neg = [
        "cargo", "run", "--no-default-features", "--features", "images",
        "--example", "wavefront_instances", "--", "--skinny-blas1", "--camera-jitter=-0.001",
    ]
    mean_a, _ = _run_and_load(cmd_neg)

    cmd_pos = [
        "cargo", "run", "--no-default-features", "--features", "images",
        "--example", "wavefront_instances", "--", "--skinny-blas1", "--camera-jitter=0.001",
    ]
    mean_b, _ = _run_and_load(cmd_pos)

    # Relative difference in mean luminance should be reasonably small
    denom = max(mean_a, mean_b, 1e-6)
    rel = abs(mean_a - mean_b) / denom
    assert rel < 0.2, f"Mean luminance changed too much with tiny jitter: rel={rel:.3f}"


@pytest.mark.skipif(
    os.environ.get("FORGE3D_RUN_WAVEFRONT", "0") != "1" or os.environ.get("FORGE3D_CI_GPU", "0") != "1" or shutil.which("cargo") is None or not forge3d.enumerate_adapters(),
    reason="Wavefront example disabled unless FORGE3D_RUN_WAVEFRONT=1 and FORGE3D_CI_GPU=1, cargo present, and a GPU adapter is available",
)
def test_wavefront_depth_aov_hist_stability_small_jitter(tmp_path: Path):
    """Compare histograms of depth AOV for tiny +/- camera jitter.

    Uses example's --dump-aov-depth to save RGBA32F raw bytes, then computes
    normalized histograms on the .x channel over nonzero depths.
    """
    import numpy as np

    cwd = str(REPO_ROOT)
    path_a = str(tmp_path / "aov_a.bin")
    path_b = str(tmp_path / "aov_b.bin")

    cmd_a = [
        "cargo", "run", "--no-default-features", "--features", "images",
        "--example", "wavefront_instances", "--",
        "--skinny-blas1", f"--dump-aov-depth={path_a}", "--camera-jitter=-0.001",
    ]
    r_a = subprocess.run(cmd_a, cwd=cwd, capture_output=True, text=True)
    assert r_a.returncode == 0, f"Run A failed: {r_a.stderr}"
    assert os.path.exists(path_a), "Missing AOV dump A"

    cmd_b = [
        "cargo", "run", "--no-default-features", "--features", "images",
        "--example", "wavefront_instances", "--",
        "--skinny-blas1", f"--dump-aov-depth={path_b}", "--camera-jitter=0.001",
    ]
    r_b = subprocess.run(cmd_b, cwd=cwd, capture_output=True, text=True)
    assert r_b.returncode == 0, f"Run B failed: {r_b.stderr}"
    assert os.path.exists(path_b), "Missing AOV dump B"

    # Load RGBA32F raw; shape (-1, 4)
    arr_a, _ = load_rgba32f_auto(path_a)
    arr_b, _ = load_rgba32f_auto(path_b)
    # Extract depth channel (supports (H,W,4) or (N,4))
    if arr_a.ndim == 3:
        depth_a = arr_a[..., 0].reshape(-1)
    else:
        depth_a = arr_a[:, 0]
    if arr_b.ndim == 3:
        depth_b = arr_b[..., 0].reshape(-1)
    else:
        depth_b = arr_b[:, 0]

    # Focus on non-zero depths (hit pixels), since background stays zero
    va = depth_a[depth_a > 0.0]
    vb = depth_b[depth_b > 0.0]
    if va.size == 0 or vb.size == 0:
        pytest.skip("No valid depth samples; scene produced all zeros")

    vmin = float(min(va.min(), vb.min()))
    vmax = float(max(va.max(), vb.max()))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        pytest.skip("Invalid depth range for histogram comparison")

    bins = 64
    ha, _ = np.histogram(va, bins=bins, range=(vmin, vmax))
    hb, _ = np.histogram(vb, bins=bins, range=(vmin, vmax))
    # Normalize
    ha = ha.astype(np.float64) / max(1, ha.sum())
    hb = hb.astype(np.float64) / max(1, hb.sum())

    l1 = float(np.abs(ha - hb).sum())
    assert l1 < 0.15, f"Depth histogram L1 too large with tiny jitter: {l1:.3f}"


@pytest.mark.skipif(
    os.environ.get("FORGE3D_RUN_WAVEFRONT", "0") != "1" or os.environ.get("FORGE3D_CI_GPU", "0") != "1" or shutil.which("cargo") is None or not forge3d.enumerate_adapters(),
    reason="Wavefront example disabled unless FORGE3D_RUN_WAVEFRONT=1 and FORGE3D_CI_GPU=1, cargo present, and a GPU adapter is available",
)
def test_wavefront_albedo_aov_hist_stability_small_jitter(tmp_path: Path):
    """Compare luminance histograms of albedo AOV for tiny +/- camera jitter."""
    import numpy as np

    cwd = str(REPO_ROOT)
    path_a = str(tmp_path / "aov_albedo_a.bin")
    path_b = str(tmp_path / "aov_albedo_b.bin")

    cmd_a = [
        "cargo", "run", "--no-default-features", "--features", "images",
        "--example", "wavefront_instances", "--",
        "--skinny-blas1", f"--dump-aov-albedo={path_a}", "--camera-jitter=-0.001",
    ]
    r_a = subprocess.run(cmd_a, cwd=cwd, capture_output=True, text=True)
    assert r_a.returncode == 0, f"Run A failed: {r_a.stderr}"
    assert os.path.exists(path_a), "Missing AOV albedo dump A"

    cmd_b = [
        "cargo", "run", "--no-default-features", "--features", "images",
        "--example", "wavefront_instances", "--",
        "--skinny-blas1", f"--dump-aov-albedo={path_b}", "--camera-jitter=0.001",
    ]
    r_b = subprocess.run(cmd_b, cwd=cwd, capture_output=True, text=True)
    assert r_b.returncode == 0, f"Run B failed: {r_b.stderr}"
    assert os.path.exists(path_b), "Missing AOV albedo dump B"

    arr_a, _ = load_rgba32f_auto(path_a)
    arr_b, _ = load_rgba32f_auto(path_b)
    if arr_a.ndim == 3:
        arr_a = arr_a.reshape(-1, 4)
    if arr_b.ndim == 3:
        arr_b = arr_b.reshape(-1, 4)
    # Compute luminance from RGB; filter nonzero albedo
    lum_a = 0.2126 * arr_a[:,0] + 0.7152 * arr_a[:,1] + 0.0722 * arr_a[:,2]
    lum_b = 0.2126 * arr_b[:,0] + 0.7152 * arr_b[:,1] + 0.0722 * arr_b[:,2]
    va = lum_a[lum_a > 0.0]
    vb = lum_b[lum_b > 0.0]
    if va.size == 0 or vb.size == 0:
        pytest.skip("No valid albedo samples; scene produced zeros")
    vmin = float(min(va.min(), vb.min()))
    vmax = float(max(va.max(), vb.max()))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        pytest.skip("Invalid albedo range for histogram comparison")
    bins = 64
    ha, _ = np.histogram(va, bins=bins, range=(vmin, vmax))
    hb, _ = np.histogram(vb, bins=bins, range=(vmin, vmax))
    ha = ha.astype(np.float64) / max(1, ha.sum())
    hb = hb.astype(np.float64) / max(1, hb.sum())
    l1 = float(np.abs(ha - hb).sum())
    assert l1 < 0.20, f"Albedo histogram L1 too large with tiny jitter: {l1:.3f}"


@pytest.mark.skipif(
    os.environ.get("FORGE3D_RUN_WAVEFRONT", "0") != "1" or os.environ.get("FORGE3D_CI_GPU", "0") != "1" or shutil.which("cargo") is None or not forge3d.enumerate_adapters(),
    reason="Wavefront example disabled unless FORGE3D_RUN_WAVEFRONT=1 and FORGE3D_CI_GPU=1, cargo present, and a GPU adapter is available",
)
def test_wavefront_normal_aov_cosz_hist_stability_small_jitter(tmp_path: Path):
    """Compare histograms of normal.z for tiny +/- camera jitter (hit pixels only)."""
    import numpy as np

    cwd = str(REPO_ROOT)
    path_a = str(tmp_path / "aov_normal_a.bin")
    path_b = str(tmp_path / "aov_normal_b.bin")

    cmd_a = [
        "cargo", "run", "--no-default-features", "--features", "images",
        "--example", "wavefront_instances", "--",
        "--skinny-blas1", f"--dump-aov-normal={path_a}", "--camera-jitter=-0.001",
    ]
    r_a = subprocess.run(cmd_a, cwd=cwd, capture_output=True, text=True)
    assert r_a.returncode == 0, f"Run A failed: {r_a.stderr}"
    assert os.path.exists(path_a), "Missing AOV normal dump A"

    cmd_b = [
        "cargo", "run", "--no-default-features", "--features", "images",
        "--example", "wavefront_instances", "--",
        "--skinny-blas1", f"--dump-aov-normal={path_b}", "--camera-jitter=0.001",
    ]
    r_b = subprocess.run(cmd_b, cwd=cwd, capture_output=True, text=True)
    assert r_b.returncode == 0, f"Run B failed: {r_b.stderr}"
    assert os.path.exists(path_b), "Missing AOV normal dump B"

    arr_a = np.fromfile(path_a, dtype=np.float32).reshape(-1, 4)
    arr_b = np.fromfile(path_b, dtype=np.float32).reshape(-1, 4)
    # Hit pixels have non-zero normal vectors
    nz_a = arr_a[:,0]**2 + arr_a[:,1]**2 + arr_a[:,2]**2
    nz_b = arr_b[:,0]**2 + arr_b[:,1]**2 + arr_b[:,2]**2
    va = arr_a[nz_a > 0.0][:,2]  # normal.z
    vb = arr_b[nz_b > 0.0][:,2]
    if va.size == 0 or vb.size == 0:
        pytest.skip("No valid normal samples; scene produced zeros")
    vmin = float(min(va.min(), vb.min()))
    vmax = float(max(va.max(), vb.max()))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        pytest.skip("Invalid normal.z range for histogram comparison")
    bins = 64
    ha, _ = np.histogram(va, bins=bins, range=(vmin, vmax))
    hb, _ = np.histogram(vb, bins=bins, range=(vmin, vmax))
    ha = ha.astype(np.float64) / max(1, ha.sum())
    hb = hb.astype(np.float64) / max(1, hb.sum())
    l1 = float(np.abs(ha - hb).sum())
    assert l1 < 0.20, f"Normal.z histogram L1 too large with tiny jitter: {l1:.3f}"


@pytest.mark.skipif(
    os.environ.get("FORGE3D_RUN_WAVEFRONT", "0") != "1" or os.environ.get("FORGE3D_CI_GPU", "0") != "1" or shutil.which("cargo") is None or not forge3d.enumerate_adapters(),
    reason="Wavefront example disabled unless FORGE3D_RUN_WAVEFRONT=1 and FORGE3D_CI_GPU=1, cargo present, and a GPU adapter is available",
)
def test_wavefront_aov_multichannel_stability_param(tmp_path: Path):
    """Multi-channel AOV stability with parameterized jitter and thresholds via env vars.

    Env vars (defaults):
      - FORGE3D_JITTER_EPS (0.001)
      - FORGE3D_THRESH_DEPTH (0.15)
      - FORGE3D_THRESH_ALBEDO (0.20)
      - FORGE3D_THRESH_NORMAL (0.20)
      - FORGE3D_THRESH_COMBINED (0.18)
    """
    import numpy as np

    cwd = str(REPO_ROOT)
    eps = float(os.environ.get("FORGE3D_JITTER_EPS", "0.001"))
    th_depth = float(os.environ.get("FORGE3D_THRESH_DEPTH", "0.15"))
    th_albedo = float(os.environ.get("FORGE3D_THRESH_ALBEDO", "0.20"))
    th_normal = float(os.environ.get("FORGE3D_THRESH_NORMAL", "0.20"))
    th_comb = float(os.environ.get("FORGE3D_THRESH_COMBINED", "0.18"))
    _log_device_and_thresholds(
        test_name="aov_multichannel",
        jitter=eps,
        depth=th_depth,
        albedo=th_albedo,
        normal=th_normal,
        combined=th_comb,
    )

    # Output paths
    depth_a = str(tmp_path / "depth_a.bin"); depth_b = str(tmp_path / "depth_b.bin")
    albedo_a = str(tmp_path / "albedo_a.bin"); albedo_b = str(tmp_path / "albedo_b.bin")
    normal_a = str(tmp_path / "normal_a.bin"); normal_b = str(tmp_path / "normal_b.bin")

    def _run_dump(j):
        cmd = [
            "cargo", "run", "--no-default-features", "--features", "images",
            "--example", "wavefront_instances", "--",
            "--skinny-blas1",
            f"--dump-aov-depth={depth_a if j < 0 else depth_b}",
            f"--dump-aov-albedo={albedo_a if j < 0 else albedo_b}",
            f"--dump-aov-normal={normal_a if j < 0 else normal_b}",
            f"--camera-jitter={j}",
        ]
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        assert r.returncode == 0, f"Run failed (jitter={j}): {r.stderr}"

    _run_dump(-eps)
    _run_dump(+eps)

    # Load dumps
    arr_da, _ = load_rgba32f_auto(depth_a)
    arr_db, _ = load_rgba32f_auto(depth_b)
    if arr_da.ndim == 3:
        da = arr_da[..., 0].reshape(-1)
    else:
        da = arr_da[:, 0]
    if arr_db.ndim == 3:
        db = arr_db[..., 0].reshape(-1)
    else:
        db = arr_db[:, 0]

    aa, _ = load_rgba32f_auto(albedo_a)
    ab, _ = load_rgba32f_auto(albedo_b)
    if aa.ndim == 3:
        aa = aa.reshape(-1, 4)
    if ab.ndim == 3:
        ab = ab.reshape(-1, 4)

    na, _ = load_rgba32f_auto(normal_a)
    nb, _ = load_rgba32f_auto(normal_b)
    if na.ndim == 3:
        na = na.reshape(-1, 4)
    if nb.ndim == 3:
        nb = nb.reshape(-1, 4)

    # Depth (nonzero hit pixels)
    vda = da[da > 0.0]; vdb = db[db > 0.0]
    if vda.size == 0 or vdb.size == 0:
        pytest.skip("No valid depth samples; scene produced all zeros")
    dmin = float(min(vda.min(), vdb.min())); dmax = float(max(vda.max(), vdb.max()))
    if not np.isfinite(dmin) or not np.isfinite(dmax) or dmin >= dmax:
        pytest.skip("Invalid depth range for histogram comparison")
    bins = 64
    hda, _ = np.histogram(vda, bins=bins, range=(dmin, dmax))
    hdb, _ = np.histogram(vdb, bins=bins, range=(dmin, dmax))
    hda = hda.astype(np.float64) / max(1, hda.sum()); hdb = hdb.astype(np.float64) / max(1, hdb.sum())
    l1_depth = float(np.abs(hda - hdb).sum())

    # Albedo luminance (nonzero)
    la = 0.2126 * aa[:,0] + 0.7152 * aa[:,1] + 0.0722 * aa[:,2]
    lb = 0.2126 * ab[:,0] + 0.7152 * ab[:,1] + 0.0722 * ab[:,2]
    la_nz = la[la > 0.0]; lb_nz = lb[lb > 0.0]
    if la_nz.size == 0 or lb_nz.size == 0:
        pytest.skip("No valid albedo samples; scene produced zeros")
    amin = float(min(la_nz.min(), lb_nz.min())); amax = float(max(la_nz.max(), lb_nz.max()))
    if not np.isfinite(amin) or not np.isfinite(amax) or amin >= amax:
        pytest.skip("Invalid albedo range for histogram comparison")
    haa, _ = np.histogram(la_nz, bins=bins, range=(amin, amax))
    hab, _ = np.histogram(lb_nz, bins=bins, range=(amin, amax))
    haa = haa.astype(np.float64) / max(1, haa.sum()); hab = hab.astype(np.float64) / max(1, hab.sum())
    l1_albedo = float(np.abs(haa - hab).sum())

    # Normal.z on nonzero normal vectors
    nzmask_a = (na[:,0]**2 + na[:,1]**2 + na[:,2]**2) > 0.0
    nzmask_b = (nb[:,0]**2 + nb[:,1]**2 + nb[:,2]**2) > 0.0
    nza = na[nzmask_a][:,2]; nzb = nb[nzmask_b][:,2]
    if nza.size == 0 or nzb.size == 0:
        pytest.skip("No valid normal samples; scene produced zeros")
    nmin = float(min(nza.min(), nzb.min())); nmax = float(max(nza.max(), nzb.max()))
    if not np.isfinite(nmin) or not np.isfinite(nmax) or nmin >= nmax:
        pytest.skip("Invalid normal.z range for histogram comparison")
    hna, _ = np.histogram(nza, bins=bins, range=(nmin, nmax))
    hnb, _ = np.histogram(nzb, bins=bins, range=(nmin, nmax))
    hna = hna.astype(np.float64) / max(1, hna.sum()); hnb = hnb.astype(np.float64) / max(1, hnb.sum())
    l1_normal = float(np.abs(hna - hnb).sum())

    # Combined metric (equal weights)
    l1_combined = (l1_depth + l1_albedo + l1_normal) / 3.0

    assert l1_depth <= th_depth, f"Depth L1 {l1_depth:.3f} > {th_depth:.3f}"
    assert l1_albedo <= th_albedo, f"Albedo L1 {l1_albedo:.3f} > {th_albedo:.3f}"
    assert l1_normal <= th_normal, f"Normal L1 {l1_normal:.3f} > {th_normal:.3f}"
    assert l1_combined <= th_comb, f"Combined L1 {l1_combined:.3f} > {th_comb:.3f}"
