import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pytest

import forge3d as f3d
from forge3d.path_tracing import hybrid_render_terrain_reference

from _ssim import ssim
from test_hybrid_terrain_pt import (
    ALBEDO, CAM, MAX_FRAMES, RELIEF, SIZE, SPAN, VARIANCE_THRESHOLD,
    _assert_raster_parity_metrics, _require_gpu, _scene_kwargs,
)

ROOT = Path(__file__).resolve().parents[1]
DEM_PATH = ROOT / "tests/data/prometheus/gore_range_dem.npy"
PROVENANCE_PATH = DEM_PATH.with_suffix(".json")
GOLDEN_DIR = ROOT / "tests/golden/hybrid_terrain"
MEMORY_LIMIT = 512 * 1024 * 1024
SOURCE_SHA256 = "bcacfdfabefc7ef4e7b9a9cefe12375b491fe1769218ad3c3ddd3ae30f9d93bc"
REFERENCE_NAMES = ("gore_range_reference.png", "gore_range_aovs.npz")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_text_sha256(path):
    """Hash a UTF-8 provenance record independently of checkout newlines."""
    canonical = Path(path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(canonical).hexdigest()


def real_dem():
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    assert provenance["source_sha256"] == SOURCE_SHA256
    assert provenance["fixture_sha256"] == sha256(DEM_PATH) == "9dad0295bd571495b8d047e4586a6c1c8556ca02ceb67b3ce4a33f393d99c96c"
    assert provenance["source_archive_md5"] == "3df0d15994166512cca17893b7af9dc4"
    assert provenance["source_archive_member_sha256"] == SOURCE_SHA256
    assert provenance["license"] == "CC-BY-4.0"
    assert provenance["dataset_doi"] == "10.5281/zenodo.3940482"
    assert provenance["elevation_source"] == "USGS National Elevation Dataset"
    assert provenance["sampling"] == {"method": "point subsampling", "row_step": 12, "column_step": 12}
    dem = np.load(DEM_PATH, allow_pickle=False)
    assert dem.dtype == np.float32 and list(dem.shape) == provenance["shape"]
    assert dem.shape == (125, 125)
    assert np.isfinite(dem).all() and not np.any(dem == provenance["source_nodata"])
    assert float(dem.max()) > float(dem.min())
    dem = dem.copy()
    dem -= dem.min()
    dem /= dem.max()
    return dem


def scene_metadata():
    return {
        "width": SIZE, "height": SIZE, "camera": CAM,
        "span_world_units": SPAN, "relief_world_units": RELIEF,
        "elevation_transform": "subtract minimum, divide by range, multiply by relief",
        "parameters": _scene_kwargs(real_dem()),
        "environment": "constant white IBL, cosine-weighted visibility sampling",
    }


def assert_reference(out):
    assert out["converged"] is True
    assert out["convergence_metric"] == "frame_mean_estimator_variance"
    assert 2 <= out["frames"] <= MAX_FRAMES
    assert out["variance"] == float(out["luminance_variance"].max())
    assert np.isfinite(out["luminance_variance"]).all()
    assert np.all(out["luminance_variance"] >= 0)
    assert out["variance"] < VARIANCE_THRESHOLD
    assert np.isfinite(out["radiance"]).all()
    assert out["radiance"].shape == (SIZE, SIZE, 3)
    assert out["peak_host_visible_bytes"] < MEMORY_LIMIT
    assert out["minmax_pyramid_bytes"] < out["gpu_resource_bytes"] < MEMORY_LIMIT
    hits = np.isfinite(out["depth"])
    assert hits.any()
    assert np.isfinite(out["albedo"]).all() and np.isfinite(out["normal"]).all()
    np.testing.assert_allclose(out["albedo"][hits], np.broadcast_to(ALBEDO, out["albedo"][hits].shape), atol=2e-3)
    assert np.max(np.abs(np.linalg.norm(out["normal"][hits], axis=-1) - 1)) < 1e-2
    assert out["traversal"]["primary"]["rays"] > 0
    assert out["traversal"]["shadow"]["rays"] > 0


def independent_reference_metrics(first, second):
    luminance = np.array([0.2126, 0.7152, 0.0722])
    difference = (first["radiance"].astype(np.float64) - second["radiance"]) @ luminance
    return {
        "first_seed": 7, "second_seed": 19, "first_spp": 1, "second_spp": 8,
        "luminance_mse": float(np.mean(difference ** 2)),
        "first_variance": float(first["variance"]),
        "second_variance": float(second["variance"]),
        "second_frames": int(second["frames"]),
    }


def aov_metrics(actual, expected):
    hit_a, hit_b = np.isfinite(actual["depth"]), np.isfinite(expected["depth"])
    both = hit_a & hit_b
    assert both.any()
    na = actual["normal"][both].astype(np.float64)
    nb = expected["normal"][both].astype(np.float64)
    na /= np.linalg.norm(na, axis=-1, keepdims=True)
    nb /= np.linalg.norm(nb, axis=-1, keepdims=True)
    angle = np.degrees(np.arccos(np.clip(np.sum(na * nb, axis=-1), -1, 1)))
    return {
        "iou": float(both.sum() / (hit_a | hit_b).sum()),
        "median_depth_err": float(np.median(np.abs(actual["depth"][both] - expected["depth"][both]))),
        "median_normal_err": float(np.median(angle)),
        "albedo_err": float(np.mean(np.abs(actual["albedo"][both] - expected["albedo"][both]))),
        "cam_dist": float(np.linalg.norm(CAM["origin"])),
    }


def aligned_raster_metrics(dem, out, capture_path=None):
    from forge3d.terrain_params import make_terrain_params_config

    session = f3d.Session(window=False)
    renderer = f3d.TerrainRenderer(session)
    material = f3d.MaterialSet.custom(ALBEDO, 0.0, 0.8, 1.0, 0.0, 4.0)
    with tempfile.TemporaryDirectory() as temporary:
        hdr = Path(temporary) / "white.hdr"
        hdr.write_bytes(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 4 +X 8\n" + bytes([128, 128, 128, 128]) * 32)
        environment = f3d.IBL.from_hdr(str(hdr), intensity=1.0)
    center_height = 0.5 * RELIEF
    origin = np.asarray(CAM["origin"], dtype=np.float64)
    target = np.asarray(CAM["look_at"], dtype=np.float64)
    offset = origin - target
    radius = np.linalg.norm(offset)
    raster_target = [target[0], -target[2], target[1] - center_height]
    clip_near, clip_far = 5.0, 400.0
    config = make_terrain_params_config(
        size_px=(SIZE, SIZE), render_scale=1.0,
        terrain_span=SPAN,
        msaa_samples=1, z_scale=RELIEF, exposure=1.0, domain=(0.0, 1.0),
        ibl_enabled=True, colormap_strength=0.0, camera_mode="mesh:zup",
        cam_radius=float(radius),
        cam_phi_deg=float(np.degrees(np.arctan2(-offset[2], offset[0]))),
        cam_theta_deg=float(np.degrees(np.arccos(offset[1] / radius))),
        fov_y_deg=CAM["fov_y"], clip=(clip_near, clip_far), debug_mode=25,
    )
    config.cam_target = raster_target
    mesh_grid_size = 512
    gx = np.linspace(0, dem.shape[1] - 1, mesh_grid_size)
    gy = np.linspace(0, dem.shape[0] - 1, mesh_grid_size)
    x0, y0 = gx.astype(int), gy.astype(int)
    x1, y1 = np.minimum(x0 + 1, dem.shape[1] - 1), np.minimum(y0 + 1, dem.shape[0] - 1)
    tx, ty = gx - x0, (gy - y0)[:, None]
    rows = dem[:, x0] * (1 - tx) + dem[:, x1] * tx
    raster_dem = np.ascontiguousarray((rows[y0] * (1 - ty) + rows[y1] * ty)[::-1], dtype=np.float32)
    _, frame = renderer.render_with_aov(material, environment, f3d.TerrainRenderParams(config), raster_dem)
    if capture_path:
        np.savez_compressed(capture_path, normal=frame.normal(), depth=frame.depth(), albedo=frame.albedo())
    normal = frame.normal().astype(np.float64)
    normal[normal[..., 2] < 0] *= -1
    normal = normal[..., [0, 2, 1]]
    normal[..., 2] *= -1
    raw_depth = frame.depth().astype(np.float64)
    hit = np.linalg.norm(normal, axis=-1) > 0.5
    y, x = np.mgrid[:SIZE, :SIZE]
    half = np.tan(np.radians(CAM["fov_y"]) / 2)
    cos_angle = 1 / np.sqrt(1 + (((x + 0.5) / SIZE * 2 - 1) * half) ** 2 + ((1 - (y + 0.5) / SIZE * 2) * half) ** 2)
    forward_y = (target[1] - origin[1]) / radius
    depth = (clip_near + raw_depth * (clip_far - clip_near) - forward_y * center_height) / cos_angle
    depth[~hit] = np.nan
    albedo = frame.albedo().astype(np.float64)
    pt_albedo = out["albedo"].astype(np.float64)
    pt_linear = np.where(pt_albedo <= 0.04045, pt_albedo / 12.92, ((pt_albedo + 0.055) / 1.055) ** 2.4)
    metrics = aov_metrics({**out, "albedo": pt_linear}, {"normal": normal, "depth": depth, "albedo": albedo})
    metrics["normal_contract"] = "Geometric raster normals (debug mode 25), outward hemisphere, rotated from Z-up to Y-up; not stylized shading normals."
    metrics["camera_contract"] = "Identical eye/target via mesh:zup and (x,-z,y); raster height centering and normalized near/far depth undone."
    metrics["sampling_contract"] = "The same bilinear DEM surface is evaluated on the rasterizer's fixed 512-vertex mesh grid (upload.rs build_uniforms_with_matrices); nearest texture sampling then selects those vertex heights. Rows are reversed by coordinate rotation."
    return metrics


def raster_metrics_in_subprocess(out, capture_path=None):
    with tempfile.TemporaryDirectory() as temporary:
        capture = Path(temporary) / "pt.npz"
        np.savez(capture, **{key: out[key] for key in ("depth", "normal", "albedo")})
        code = (
            "import json,sys,numpy as np; from pathlib import Path; "
            "sys.path.insert(0,str(Path.cwd()/'tests')); "
            "import test_prometheus_dem_reference as t; "
            "d=t.real_dem(); o=np.load(sys.argv[1],allow_pickle=False); "
            "print('PROMETHEUS_RASTER='+json.dumps(t.aligned_raster_metrics(d,dict(o),sys.argv[2] or None)),flush=True)"
        )
        result = subprocess.run([sys.executable, "-c", code, str(capture), str(capture_path or "")], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, f"real-DEM raster comparison failed:\n{result.stdout}\n{result.stderr}"
    rows = [line.split("=", 1)[1] for line in result.stdout.splitlines() if line.startswith("PROMETHEUS_RASTER=")]
    assert len(rows) == 1, result.stdout
    return json.loads(rows[0])


def load_reference_files():
    scores = json.loads((GOLDEN_DIR / "gore_range_scores.json").read_text(encoding="utf-8"))
    assert scores["fixture_sha256"] == sha256(DEM_PATH)
    assert scores["provenance_sha256"] == canonical_text_sha256(PROVENANCE_PATH)
    assert scores["scene"] == json.loads(json.dumps(scene_metadata()))
    assert scores["convergence_metric"] == "frame_mean_estimator_variance"
    assert scores["variance"] < VARIANCE_THRESHOLD
    assert scores["gpu_resource_bytes"] < MEMORY_LIMIT
    assert scores["peak_host_visible_bytes"] < MEMORY_LIMIT
    assert scores["independent_reference"]["luminance_mse"] < VARIANCE_THRESHOLD
    assert scores["adapter"]["status"] == "ok"
    assert scores["adapter"]["device_type"].lower() in {"discretegpu", "integratedgpu", "virtualgpu"}
    for name in REFERENCE_NAMES:
        assert sha256(GOLDEN_DIR / name) == scores["artifact_sha256"][name]
    with np.load(GOLDEN_DIR / REFERENCE_NAMES[1], allow_pickle=False) as bundle:
        aovs = {key: bundle[key] for key in bundle.files}
    assert set(aovs) == {"albedo", "normal", "depth", "radiance", "luminance_variance"}
    assert float(aovs["luminance_variance"].max()) == scores["variance"]
    for key in ("albedo", "normal", "radiance"):
        assert aovs[key].shape == (SIZE, SIZE, 3) and aovs[key].dtype == np.float32
        assert np.isfinite(aovs[key]).all()
    assert aovs["depth"].shape == (SIZE, SIZE) and aovs["depth"].dtype == np.float32
    assert aovs["luminance_variance"].shape == (SIZE, SIZE)
    assert np.isfinite(aovs["luminance_variance"]).all()
    _assert_raster_parity_metrics(scores["raster_parity"])
    return f3d.png_to_numpy(str(GOLDEN_DIR / REFERENCE_NAMES[0])), aovs, scores


@pytest.fixture(scope="module")
def real_reference():
    _require_gpu()
    dem = real_dem()
    out = hybrid_render_terrain_reference(dem, SIZE, SIZE, CAM, **_scene_kwargs(dem))
    assert_reference(out)
    return dem, out


def test_measured_dem_provenance_and_reference_bundle():
    real_dem()
    load_reference_files()


def test_real_dem_convergence_and_memory(real_reference):
    _, out = real_reference
    print("PROMETHEUS_REAL_DEM=" + json.dumps({key: out[key] for key in (
        "frames", "variance", "convergence_metric", "peak_host_visible_bytes", "gpu_resource_bytes", "traversal"
    )}, sort_keys=True))
    assert_reference(out)


def test_real_dem_independent_radiance_reference(real_reference):
    dem, out = real_reference
    other = hybrid_render_terrain_reference(dem, SIZE, SIZE, CAM, **{**_scene_kwargs(dem), "seed": 19, "spp": 8})
    assert_reference(other)
    scores = independent_reference_metrics(out, other)
    print("PROMETHEUS_INDEPENDENT=" + json.dumps(scores, sort_keys=True))
    assert scores["luminance_mse"] < VARIANCE_THRESHOLD


def test_real_dem_raster_aov_tolerances(real_reference):
    _, out = real_reference
    _assert_raster_parity_metrics(raster_metrics_in_subprocess(out))


def test_real_dem_beauty_and_aov_golden(real_reference):
    _, out = real_reference
    beauty, aovs, _ = load_reference_files()
    actual = out["rgba"]
    assert actual.shape == beauty.shape
    mean_abs = float(np.mean(np.abs(actual[..., :3].astype(np.float64) - beauty[..., :3])))
    score = ssim(actual[..., :3], beauty[..., :3], data_range=255.0)
    print(f"PROMETHEUS_BEAUTY: SSIM={score:.8f}, mean_abs={mean_abs:.8f}")
    assert score >= 0.995
    assert mean_abs <= 2.0
    _assert_raster_parity_metrics(aov_metrics(out, aovs))


@pytest.mark.parametrize("component", ["depth", "normal", "albedo"])
def test_aov_golden_gate_rejects_corruption(component):
    _, aovs, _ = load_reference_files()
    bad = {key: value.copy() for key, value in aovs.items()}
    hit = np.isfinite(bad["depth"])
    if component == "depth":
        bad[component][hit] += 2 * 10.0
    elif component == "normal":
        bad[component][hit] *= -1
    else:
        bad[component][hit] += 2 * 0.01
    with pytest.raises(AssertionError):
        _assert_raster_parity_metrics(aov_metrics(bad, aovs))
