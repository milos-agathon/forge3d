# tests/test_splat_api.py
# SPLAT-FUSED Python surface: Gaussian splat loading, the analytic ray /
# Gaussian kernel, page stores, input diagnostics and `render_fused`.
# CPU-only sections always run; GPU sections follow the recipe-golden skip
# convention (skip when no adapter, hard-fail on regression).
# RELEVANT FILES: python/forge3d/splat.py, src/py_functions/splat.rs,
#                 tests/test_splat_fusion_occlusion.py

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import forge3d as f3d
from forge3d import splat

from _splat_fusion import FIXTURE_DIR, gpu_available, manifest, scene_kwargs

pytestmark = pytest.mark.skipif(
    not splat.splat_fusion_available(),
    reason="wheel built without the splat-fusion feature",
)


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Section 1: loader + cloud (no GPU)
# ---------------------------------------------------------------------------


def test_public_surface_is_exported():
    for name in (
        "splat",
        "load_gaussian_splats",
        "render_fused",
        "GaussianSplatCloud",
        "FusedCamera",
        "FusedTerrain",
        "FusedPointCloud",
        "FusedRenderResult",
        "SplatFusionUnavailable",
    ):
        assert name in f3d.__all__, name
        assert hasattr(f3d, name), name
    assert f3d.render_fused is splat.render_fused
    assert f3d.load_gaussian_splats is splat.load_gaussian_splats
    assert set(splat.__all__) <= set(dir(splat))


def test_load_gaussian_splats_fixture_fields():
    m = manifest()
    cloud = splat.load_gaussian_splats(FIXTURE_DIR / m["splat_file"])
    n = m["splat_count"]
    assert isinstance(cloud, splat.GaussianSplatCloud)
    assert cloud.count == n == len(cloud)
    assert cloud.sh_degree == 1
    assert "GaussianSplatCloud(count=3500" in repr(cloud)
    for name, shape in (
        ("positions", (n, 3)),
        ("scales", (n, 3)),
        ("rotations", (n, 4)),
        ("opacities", (n,)),
        ("sh0", (n, 3)),
        ("inverse_covariance", (n, 6)),
    ):
        value = getattr(cloud, name)
        assert value.shape == shape, name
        assert value.dtype == np.float32, name
        assert np.isfinite(value).all(), name
    assert (cloud.scales > 0).all()
    assert ((cloud.opacities > 0) & (cloud.opacities < 1)).all()
    np.testing.assert_allclose(np.linalg.norm(cloud.rotations, axis=1), 1.0, atol=1e-5)
    lo, hi = cloud.bounds
    assert all(a < b for a, b in zip(lo, hi))
    assert (cloud.positions >= np.array(lo)).all() and (cloud.positions <= np.array(hi)).all()
    # The 5 float columns (pos, scale, rot, opacity, sh0) plus Sigma^-1.
    assert cloud.byte_size >= n * (12 + 12 + 16 + 4 + 12 + 24)


def test_inverse_covariance_is_r_diag_inv_sigma2_rt():
    m = manifest()
    cloud = splat.load_gaussian_splats(FIXTURE_DIR / m["splat_file"])
    packed = cloud.inverse_covariance
    for i in (0, 1, 777, 2599, 2600, 3499):
        r = _quat_to_matrix(cloud.rotations[i].astype(np.float64))
        expected = r @ np.diag(1.0 / cloud.scales[i].astype(np.float64) ** 2) @ r.T
        xx, xy, xz, yy, yz, zz = packed[i]
        got = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
        np.testing.assert_allclose(got, expected, rtol=2e-4, atol=1e-4)
        # Sigma^-1 is symmetric positive definite.
        assert np.linalg.eigvalsh(got).min() > 0


def test_from_arrays_round_trip_and_validation(tmp_path):
    rng = np.random.default_rng(3)
    n = 64
    positions = rng.uniform(-2, 2, (n, 3)).astype(np.float32)
    scales = rng.uniform(0.1, 0.5, (n, 3)).astype(np.float32)
    rotations = rng.normal(size=(n, 4)).astype(np.float32)
    opacities = rng.uniform(0.05, 0.95, n).astype(np.float32)
    sh0 = rng.normal(size=(n, 3)).astype(np.float32)
    sh_rest = (0.1 * rng.normal(size=(n, 15, 3))).astype(np.float32)
    cloud = splat.GaussianSplatCloud.from_arrays(
        positions, scales, rotations, opacities, sh0, sh_rest
    )
    assert cloud.count == n and cloud.sh_degree == 3
    np.testing.assert_array_equal(cloud.positions, positions)
    np.testing.assert_allclose(
        cloud.rotations, rotations / np.linalg.norm(rotations, axis=1, keepdims=True), atol=1e-6
    )
    # View-dependent colour: higher bands make it depend on the direction.
    assert cloud.color(0, (0.0, 0.0, 1.0)) != cloud.color(0, (1.0, 0.0, 0.0))
    with pytest.raises(IndexError):
        cloud.color(n, (0.0, 0.0, 1.0))

    path = tmp_path / "cloud.ply"
    cloud.save(str(path))
    loaded = splat.load_gaussian_splats(path)
    assert loaded.count == n and loaded.sh_degree == 3
    np.testing.assert_array_equal(loaded.positions, positions)
    np.testing.assert_allclose(loaded.scales, scales, rtol=1e-5)
    np.testing.assert_allclose(loaded.opacities, opacities, atol=1e-5)
    np.testing.assert_allclose(
        loaded.inverse_covariance, cloud.inverse_covariance, rtol=2e-3, atol=1e-3
    )

    with pytest.raises(Exception, match="scale"):
        splat.GaussianSplatCloud.from_arrays(
            positions, np.zeros_like(scales), rotations, opacities, sh0
        )
    with pytest.raises(Exception, match="opacity"):
        splat.GaussianSplatCloud.from_arrays(
            positions, scales, rotations, opacities + 2.0, sh0
        )
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        splat.GaussianSplatCloud.from_arrays(
            positions[:, :2], scales, rotations, opacities, sh0
        )
    with pytest.raises(ValueError, match="sh_rest"):
        splat.GaussianSplatCloud.from_arrays(
            positions, scales, rotations, opacities, sh0, sh_rest[:, :4]
        )


def test_malformed_ply_raises_a_diagnostic(tmp_path):
    bad = tmp_path / "bad.ply"
    bad.write_text("ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nend_header\n0\n")
    with pytest.raises(Exception, match="required 3DGS property"):
        splat.load_gaussian_splats(bad)
    with pytest.raises(Exception, match="cannot open"):
        splat.load_gaussian_splats(tmp_path / "missing.ply")


# ---------------------------------------------------------------------------
# Section 2: the analytic ray / Gaussian kernel (no GPU)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("miss", [0.0, 0.3, 0.9, 1.8])
def test_ray_gaussian_matches_closed_form(miss):
    sigma = 0.75
    hit = splat.ray_gaussian(
        origin=(-3.0, 1.0 + miss, -2.0),
        direction=(1.0, 0.0, 0.0),
        center=(4.0, 1.0, -2.0),
        scale=(sigma, sigma, sigma),
        opacity=0.9,
        kappa=4.0,
    )
    g = (miss / sigma) ** 2
    assert hit["t_star"] == pytest.approx(7.0, abs=1e-5)
    assert hit["g_star"] == pytest.approx(g, abs=1e-4)
    assert hit["response"] == pytest.approx(0.9 * math.exp(-0.5 * g), abs=1e-5)
    assert hit["transmittance"] == pytest.approx(math.exp(-4.0 * hit["response"]), abs=1e-6)
    # The surface hit is the entry into the 1-sigma shell.
    expected_hit = 7.0 - math.sqrt(max(1.0 - g, 0.0)) * sigma
    assert hit["t_hit"] == pytest.approx(expected_hit, abs=1e-4)


def test_ray_gaussian_anisotropy_cutoff_and_segment():
    # Along the long axis the closest approach is the same but the transverse
    # miss is measured in the short axis' sigma.
    wide = splat.ray_gaussian((-5, 0.2, 0), (1, 0, 0), (0, 0, 0), (2.0, 0.1, 0.1))
    thin = splat.ray_gaussian((-5, 0.2, 0), (1, 0, 0), (0, 0, 0), (2.0, 1.0, 0.1))
    assert wide["g_star"] == pytest.approx(4.0, abs=1e-4)
    assert thin["g_star"] == pytest.approx(0.04, abs=1e-4)
    assert thin["response"] > wide["response"]
    # Beyond three sigma a splat contributes nothing (its proxy is the
    # 3-sigma ellipsoid's box).
    outside = splat.ray_gaussian((-5, 0.31, 0), (1, 0, 0), (0, 0, 0), (2.0, 0.1, 0.1))
    assert outside["response"] == 0.0 and outside["transmittance"] == 1.0
    # A segment that ends at the centre sees half of the line integral.
    full = splat.ray_gaussian((-5, 0, 0), (1, 0, 0), (0, 0, 0), (0.5, 0.5, 0.5))
    half = splat.ray_gaussian((-5, 0, 0), (1, 0, 0), (0, 0, 0), (0.5, 0.5, 0.5), tmax=5.0)
    assert half["response"] == pytest.approx(0.5 * full["response"], rel=1e-4)
    # Larger opacity -> lower transmittance, strictly.
    ts = [
        splat.ray_gaussian((-5, 0.1, 0), (1, 0, 0), (0, 0, 0), (0.5, 0.5, 0.5), opacity=a)[
            "transmittance"
        ]
        for a in (0.1, 0.3, 0.6, 0.9, 1.0)
    ]
    assert all(a > b for a, b in zip(ts, ts[1:]))
    with pytest.raises(ValueError):
        splat.ray_gaussian((0, 0, 0), (0, 0, 0), (1, 0, 0), (1, 1, 1))


# ---------------------------------------------------------------------------
# Section 3: out-of-core page stores (no GPU)
# ---------------------------------------------------------------------------


def test_page_stores_and_synthetic_billion_point_index(tmp_path):
    m = manifest()
    cloud = splat.load_gaussian_splats(FIXTURE_DIR / m["splat_file"])
    summary = cloud.write_page_store(str(tmp_path / "cloud.f3dpages"), 512)
    assert summary["logical_primitives"] == m["splat_count"]
    assert summary["page_count"] == math.ceil(m["splat_count"] / 512)

    built = splat.build_splat_page_store(
        FIXTURE_DIR / m["splat_file"],
        tmp_path / "built.f3dpages",
        page_capacity=512,
        memory_budget_bytes=1 << 20,
    )
    assert built["logical_primitives"] == m["splat_count"]
    with pytest.raises(f3d.MemoryBudgetExceeded):
        splat.build_splat_page_store(
            FIXTURE_DIR / m["splat_file"], tmp_path / "tiny.f3dpages", memory_budget_bytes=64
        )

    field = splat.write_synthetic_point_field(
        tmp_path / "field.f3dpages",
        grid=(496, 496),
        cell_size=16.0,
        origin=(-3968.0, 0.6, -3968.0),
        hole=(-32.0, -32.0, 32.0, 32.0),
        seed=1,
    )
    assert field["logical_primitives"] > 1_000_000_000
    # A billion-point index backed by a megabyte of payload on disk.
    assert field["payload_bytes"] <= 2 * 1024 * 1024
    assert field["file_bytes"] < 32 * 1024 * 1024


# ---------------------------------------------------------------------------
# Section 4: render_fused input diagnostics (no GPU reached)
# ---------------------------------------------------------------------------


def test_render_fused_rejects_missing_and_malformed_inputs(tmp_path):
    camera = splat.FusedCamera(origin=(0, 5, 10), look_at=(0, 0, 0))
    with pytest.raises(ValueError, match="at least one representation"):
        splat.render_fused(camera=camera)
    with pytest.raises(TypeError, match="camera"):
        splat.render_fused(terrain=np.zeros((4, 4), np.float32), camera=(0, 1, 2))
    with pytest.raises(ValueError, match="origin"):
        splat.render_fused(terrain=np.zeros((4, 4), np.float32), camera={"look_at": (0, 0, 0)})
    with pytest.raises(ValueError, match="2-D"):
        splat.render_fused(terrain=np.zeros(8, np.float32), camera=camera)
    with pytest.raises(ValueError, match="non-finite"):
        splat.render_fused(terrain=np.full((4, 4), np.nan, np.float32), camera=camera)
    with pytest.raises(ValueError, match="samples"):
        splat.render_fused(terrain=np.zeros((4, 4), np.float32), camera=camera, samples=0)
    with pytest.raises(TypeError, match="splats"):
        splat.render_fused(splats=object(), camera=camera)
    with pytest.raises(TypeError, match="pointcloud"):
        splat.render_fused(pointcloud=42, camera=camera)
    with pytest.raises(FileNotFoundError):
        splat.render_fused(splats=str(tmp_path / "missing.ply"), camera=camera)
    with pytest.raises(Exception, match="missing"):
        splat.render_fused(pointcloud=tmp_path / "missing.copc.laz", camera=camera)
    flat = np.zeros((4, 4), np.float32)
    with pytest.raises(ValueError, match="paging policy"):
        splat.render_fused(terrain=flat, camera=camera, policy="lazy")
    with pytest.raises(ValueError, match="unknown brdf"):
        splat.render_fused(terrain=flat, camera=camera, brdf="velvet")
    with pytest.raises(Exception, match="kappa"):
        splat.render_fused(terrain=flat, camera=camera, kappa=0.0)
    # A point page store is not a splat source (and vice versa).
    m = manifest()
    cloud = splat.load_gaussian_splats(FIXTURE_DIR / m["splat_file"])
    store = tmp_path / "cloud.f3dpages"
    cloud.write_page_store(str(store), 4096)
    with pytest.raises(ValueError, match="splat page store"):
        splat.render_fused(pointcloud=store, camera=camera)


# ---------------------------------------------------------------------------
# Section 5: render_fused on the GPU
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fused_result():
    if not gpu_available():
        pytest.skip("no usable GPU adapter for the fused render")
    m = manifest()
    return splat.render_fused(
        **scene_kwargs(m), samples=16, width=128, height=96, seed=7, return_aovs=True
    )


def test_render_fused_returns_rgba_image():
    if not gpu_available():
        pytest.skip("no usable GPU adapter for the fused render")
    m = manifest()
    kwargs = scene_kwargs(m)
    image = splat.render_fused(**kwargs, samples=8, width=80, height=64, seed=7)
    assert isinstance(image, np.ndarray)
    assert image.shape == (64, 80, 4) and image.dtype == np.uint8
    assert (image[..., 3] == 255).all()
    assert image[..., :3].std() > 5.0, "the image is flat"
    # Deterministic: same inputs, same bytes.
    again = splat.render_fused(**kwargs, samples=8, width=80, height=64, seed=7)
    np.testing.assert_array_equal(image, again)
    other_seed = splat.render_fused(**kwargs, samples=8, width=80, height=64, seed=8)
    assert (image != other_seed).any()


def test_render_fused_aovs_shapes_and_contents(fused_result):
    r = fused_result
    h, w = 96, 128
    assert isinstance(r, splat.FusedRenderResult)
    assert r.rgba.shape == (h, w, 4) and r.rgba.dtype == np.uint8
    for name in ("radiance", "albedo", "normal", "position", "direct"):
        value = getattr(r, name)
        assert value.shape == (h, w, 3) and value.dtype == np.float32, name
    assert r.transmittance.shape == (h, w, 4) and r.transmittance.dtype == np.float32
    for name in ("depth", "sun_cosine", "reservoir_visibility"):
        assert getattr(r, name).shape == (h, w), name
    assert r.hit_kind.shape == (h, w) and r.hit_kind.dtype == np.uint8
    assert np.isfinite(r.radiance).all() and (r.radiance >= 0).all()

    # All three representations are visible, and nothing else.
    kinds = set(np.unique(r.hit_kind).tolist())
    assert {splat.HIT_TERRAIN, splat.HIT_SPLAT, splat.HIT_LIDAR} <= kinds
    assert kinds <= {splat.HIT_MISS, splat.HIT_TERRAIN, splat.HIT_SPLAT, splat.HIT_LIDAR}

    # The unified model: every factor is a transmittance in [0, 1] and the
    # total is their product.
    t = r.transmittance
    assert (t >= 0).all() and (t <= 1 + 1e-3).all()
    np.testing.assert_allclose(t[..., 0], t[..., 1] * t[..., 2] * t[..., 3], atol=2e-3)
    hit = r.hit_kind != splat.HIT_MISS
    # Each occluder family actually shadows something in this scene.
    assert ((t[..., 1] < 0.5) & hit).sum() > 20, "no splat shadow"
    assert ((t[..., 2] < 0.5) & hit).sum() > 20, "no LiDAR shadow"
    assert ((t[..., 3] < 0.5) & hit).sum() > 20, "no terrain shadow"
    assert r.shadow_mask.dtype == bool and r.shadow_mask.any() and not r.shadow_mask.all()
    # Shadowed surfaces receive no direct sun.
    shadowed = r.shadow_mask & (t[..., 0] < 1e-3)
    assert r.direct[shadowed].max() < 1e-2
    np.testing.assert_allclose(np.linalg.norm(r.normal[hit], axis=1), 1.0, atol=1e-3)
    assert np.isnan(r.depth[~hit]).all() and np.isfinite(r.depth[hit]).all()


def test_render_fused_reports_restir_paging_and_budget(fused_result):
    stats = fused_result.stats
    m = manifest()
    assert stats["frames"] == 4
    assert stats["logical_primitives"] == m["splat_count"] + m["copc_point_count"]
    assert stats["reservoir_valid_count"] > 0, "the ReSTIR chain carried no reservoirs"
    assert stats["tlas_node_count"] > 0
    paging = stats["paging"]
    assert paging["loads"] == stats["page_count"] == paging["resident_pages"]
    assert paging["miss_events"] > 0 and paging["deferred_pages"] == 0
    limit = stats["tracker_limit_bytes"]
    assert limit == 512 * 1024 * 1024
    assert 0 < stats["peak_total_bytes"] <= limit
    assert stats["peak_host_visible_bytes"] <= limit
    assert paging["pool_bytes"] <= stats["peak_total_bytes"]
    # The reservoirs carry a last-known sun visibility in [0, 1].
    carried = fused_result.reservoir_visibility
    valid = carried >= 0
    assert valid.sum() == stats["reservoir_valid_count"]
    assert (carried[valid] <= 1.0 + 1e-6).all()


def test_render_fused_single_representations_and_options():
    if not gpu_available():
        pytest.skip("no usable GPU adapter for the fused render")
    m = manifest()
    kwargs = scene_kwargs(m)
    common = dict(
        camera=kwargs["camera"],
        samples=4,
        width=64,
        height=64,
        sun_azimuth_deg=kwargs["sun_azimuth_deg"],
        sun_elevation_deg=kwargs["sun_elevation_deg"],
        return_aovs=True,
    )
    terrain_only = splat.render_fused(terrain=kwargs["terrain"], **common)
    assert set(np.unique(terrain_only.hit_kind).tolist()) <= {splat.HIT_MISS, splat.HIT_TERRAIN}
    assert terrain_only.stats["paging"]["loads"] == 0
    assert (terrain_only.transmittance[..., 1:3] == 1.0).all()

    cloud = splat.load_gaussian_splats(kwargs["splats"])
    splat_only = splat.render_fused(splats=cloud, splat_page_size=512, **common)
    assert set(np.unique(splat_only.hit_kind).tolist()) == {splat.HIT_MISS, splat.HIT_SPLAT}
    assert splat_only.stats["page_count"] == 7

    lidar_only = splat.render_fused(
        pointcloud=kwargs["pointcloud"], lidar_radius=kwargs["lidar_radius"], **common
    )
    assert set(np.unique(lidar_only.hit_kind).tolist()) == {splat.HIT_MISS, splat.HIT_LIDAR}

    # Fog attenuates the sun beam: with density the direct AOV dims.
    foggy = splat.render_fused(terrain=kwargs["terrain"], fog_density=0.02, **common)
    lit = (terrain_only.transmittance[..., 0] > 0.99) & (terrain_only.hit_kind == splat.HIT_TERRAIN)
    assert foggy.direct[lit].mean() < 0.9 * terrain_only.direct[lit].mean()
    # The working set must fit the pool under the exact policy.
    with pytest.raises(f3d.MemoryBudgetExceeded, match="residency pool"):
        splat.render_fused(
            splats=cloud,
            splat_page_size=512,
            splat_slots=2,
            **{**common, "return_aovs": False},
        )
    progressive = splat.render_fused(
        splats=cloud, splat_page_size=512, splat_slots=2, policy="progressive", **common
    )
    assert progressive.stats["stale_frames"] > 0
    assert progressive.stats["paging"]["peak_resident_pages"] <= 2


# ---------------------------------------------------------------------------
# Section 6: CENSOR render certificate / ANAMNESIS cache contract
# ---------------------------------------------------------------------------


def test_render_entrypoints_accept_certificate_and_cache():
    import inspect

    for function in (splat.render_fused, splat.render_fused_reference):
        parameters = inspect.signature(function).parameters
        assert "certificate" in parameters, function.__name__
        assert "cache" in parameters, function.__name__


def test_render_fused_emits_a_signed_certificate(tmp_path):
    if not gpu_available():
        pytest.skip("no usable GPU adapter for the fused render")
    from forge3d import certificate as cert_module
    from forge3d.diagnostics import render_certificate

    m = manifest()
    kwargs = scene_kwargs(m)
    path = tmp_path / "fused.certificate.json"
    splat.render_fused(
        **kwargs, samples=4, samples_per_frame=2, width=48, height=48, certificate=path
    )
    live = render_certificate(sign=False)
    assert [entry["label"] for entry in live["passes"]] == [
        "hybrid_pt.fused_gbuffer",
        "hybrid_pt.fused",
        "hybrid_pt.restir_temporal",
        "hybrid_pt.restir_spatial",
        "hybrid_pt.fused_publish",
    ]
    assert set(live["engine"]["wgsl_module_hashes"]) == {
        "hybrid-pt-kernel",
        "hybrid-pt-restir-temporal",
        "hybrid-pt-restir-spatial",
    }
    assert "splat_fusion.unified_occlusion" in live["models"]
    assert live["inputs"]["fused.logical_primitives"] == str(
        m["splat_count"] + m["copc_point_count"]
    )
    assert live["inputs"]["fused.paging_policy"] == "exact"
    # No machine-specific path leaks into the signed inputs.
    assert not any(str(FIXTURE_DIR) in value for value in live["inputs"].values())

    written = json.loads(path.read_text(encoding="utf-8"))
    assert [entry["label"] for entry in written["passes"]] == [
        entry["label"] for entry in live["passes"]
    ]
    assert cert_module.payload_sha256(written)


def test_render_fused_reference_emits_a_certificate():
    if not gpu_available():
        pytest.skip("no usable GPU adapter for the fused render")
    from forge3d.diagnostics import render_certificate

    m = manifest()
    kwargs = scene_kwargs(m)
    half = 0.5 * (m["dem_width"] - 1) * m["dem_spacing"][0]
    splat.render_fused_reference(
        splats=kwargs["splats"],
        terrain=kwargs["terrain"],
        camera=kwargs["camera"],
        region=((-half, -1000.0, -half), (half, 1000.0, half)),
        samples=4,
        width=48,
        height=48,
        certificate=True,
    )
    certificate = render_certificate(sign=False)
    assert [entry["label"] for entry in certificate["passes"]] == ["fused_reference.path_trace"]
    assert "splat_fusion.hard_proxy_reference" in certificate["models"]


# ---------------------------------------------------------------------------
# Section 7: SPLAT-FUSED limits remediation
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not gpu_available(), reason="no usable GPU adapter for the fused render")
def test_self_bias_parameters_reach_the_kernel():
    m = manifest()
    kw = scene_kwargs(m)
    common = dict(samples=4, width=96, height=96, return_aovs=True)
    a = splat.render_fused(**kw, **common, splat_self_bias_sigmas=2.0, lidar_self_bias_radii=1.0)
    b = splat.render_fused(**kw, **common, splat_self_bias_sigmas=0.5, lidar_self_bias_radii=3.0)
    lidar = (a.hit_kind == splat.HIT_LIDAR) & (b.hit_kind == splat.HIT_LIDAR)
    assert lidar.sum() > 50
    # The self_bias AOV is stored in the Rgba16Float emission AOV; WGSL leaves
    # the f32 -> f16 storage rounding to the implementation, so the bound is
    # one half-float ulp (2**-10 relative).
    f16_ulp = 2.0**-10
    np.testing.assert_allclose(a.self_bias[lidar], 1.0 * m["lidar_radius"], rtol=f16_ulp)
    np.testing.assert_allclose(b.self_bias[lidar], 3.0 * m["lidar_radius"], rtol=f16_ulp)
    splats = (a.hit_kind == splat.HIT_SPLAT) & (b.hit_kind == splat.HIT_SPLAT)
    assert splats.sum() > 50
    np.testing.assert_allclose(b.self_bias[splats], 0.25 * a.self_bias[splats], rtol=1e-5)
    assert np.all(a.self_bias[a.hit_kind == splat.HIT_TERRAIN] == 0.0)
    with pytest.raises(RuntimeError, match="self-shadow bias"):
        splat.render_fused(**kw, **common, lidar_self_bias_radii=-1.0)


def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def test_transformed_applies_a_similarity_without_mutating_the_source():
    cloud = splat.load_gaussian_splats(FIXTURE_DIR / manifest()["splat_file"])
    before = np.array(cloud.positions, copy=True)
    q = np.array([0.9, 0.1, -0.3, 0.2])
    q /= np.linalg.norm(q)
    t = np.array([100.0, -20.0, 7.0])
    moved = cloud.transformed(2.5, tuple(q), tuple(t))
    R = _quat_to_matrix(q)
    np.testing.assert_allclose(moved.positions, 2.5 * before @ R.T + t, rtol=0, atol=1e-3)
    np.testing.assert_allclose(moved.scales, 2.5 * np.asarray(cloud.scales), rtol=1e-6)
    for qi, qm in zip(np.asarray(cloud.rotations), np.asarray(moved.rotations)):
        assert abs(np.dot(_quat_mul(q, qi), qm)) >= 1.0 - 1e-6
    np.testing.assert_array_equal(np.asarray(cloud.positions), before)
    assert moved.count == cloud.count
    for bad in (dict(scale=0.0), dict(scale=float("nan"))):
        with pytest.raises(ValueError, match="scale"):
            cloud.transformed(bad["scale"], (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="rotation"):
        cloud.transformed(1.0, (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
def _srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb_u8(v):
    v = np.clip(np.asarray(v, dtype=np.float64), 0.0, 1.0)
    s = np.where(v <= 0.0031308, v * 12.92, 1.055 * v ** (1.0 / 2.4) - 0.055)
    return np.round(s * 255.0).astype(np.int64)


def test_terrain_albedo_map_is_validated():
    h = np.zeros((6, 5), np.float32)
    with pytest.raises(TypeError, match="uint8"):
        splat.render_fused(
            terrain=splat.FusedTerrain(heights=h, albedo_map=np.zeros(h.shape + (3,), np.float32)),
            camera=splat.FusedCamera(origin=(0, 5, 10), look_at=(0, 0, 0)),
        )
    with pytest.raises(ValueError, match="albedo_map shape"):
        splat.render_fused(
            terrain=splat.FusedTerrain(heights=h, albedo_map=np.zeros((7, 5, 3), np.uint8)),
            camera=splat.FusedCamera(origin=(0, 5, 10), look_at=(0, 0, 0)),
        )
    with pytest.raises(ValueError, match="albedo_map shape"):
        splat.render_fused(
            terrain=splat.FusedTerrain(heights=h, albedo_map=np.zeros((6, 5, 2), np.uint8)),
            camera=splat.FusedCamera(origin=(0, 5, 10), look_at=(0, 0, 0)),
        )
    with pytest.raises(ValueError, match="albedo_sampling"):
        splat.render_fused(
            terrain=splat.FusedTerrain(
                heights=h, albedo_map=np.zeros((6, 5, 3), np.uint8), albedo_sampling="cubic"
            ),
            camera=splat.FusedCamera(origin=(0, 5, 10), look_at=(0, 0, 0)),
        )


@pytest.mark.skipif(not gpu_available(), reason="no usable GPU adapter for the fused render")
def test_terrain_albedo_map_sets_the_terrain_albedo():
    m = manifest()
    kw = scene_kwargs(m)
    t = kw["terrain"]
    grey = np.full(t.heights.shape + (3,), 128, np.uint8)
    kw["terrain"] = splat.FusedTerrain(
        heights=t.heights, spacing=t.spacing, albedo=t.albedo, albedo_map=grey
    )
    out = splat.render_fused(**kw, samples=2, width=96, height=96, return_aovs=True)
    terrain = out.hit_kind == splat.HIT_TERRAIN
    assert terrain.sum() > 500
    codes = _linear_to_srgb_u8(out.albedo[terrain])
    assert np.abs(codes - 128).max() <= 1, np.unique(codes)
    assert out.stats["terrain_bytes"] > 0
@pytest.mark.skipif(not gpu_available(), reason="no usable GPU adapter for the fused render")
def test_render_fused_1080p_fits_the_budget():
    m = manifest()
    out = splat.render_fused(
        **scene_kwargs(m), samples=2, width=1920, height=1080, return_aovs=True
    )
    assert out.rgba.shape == (1080, 1920, 4)
    assert out.albedo.shape == (1080, 1920, 3)
    assert out.hit_kind.shape == (1080, 1920)
    assert out.transmittance.shape == (1080, 1920, 4)
    assert out.stats["tiles"] == 4  # default 1024x1024 tiles
    assert out.stats["peak_total_bytes"] <= 512 * 1024 * 1024
    print(
        f"\n1920x1080 fused render: {out.stats['tiles']} tiles, peak "
        f"{out.stats['peak_total_bytes'] / 2**20:.1f} MiB"
    )


def test_render_fused_rejects_invalid_tiles():
    camera = splat.FusedCamera(origin=(0, 5, 10), look_at=(0, 0, 0))
    flat = np.zeros((4, 4), np.float32)
    for tile in ((0, 10), (10, 0), (65, 10), (10, 65)):
        with pytest.raises(ValueError, match="tile"):
            splat.render_fused(terrain=flat, camera=camera, width=64, height=64, tile=tile)


@pytest.mark.skipif(not gpu_available(), reason="no usable GPU adapter for the fused render")
def test_render_fused_sequence_matches_independent_renders_and_reuses_pages():
    m = manifest()
    kw = scene_kwargs(m)
    cam = kw.pop("camera")
    sun = {k: kw.pop(k) for k in ("sun_azimuth_deg", "sun_elevation_deg", "sun_intensity", "sun_color")}
    other = splat.FusedCamera(
        origin=tuple(np.array(cam.origin) + [6.0, 0.0, -4.0]),
        look_at=cam.look_at,
        up=cam.up,
        fov_y_deg=cam.fov_y_deg,
    )
    views = [
        splat.FusedView(camera=cam, **sun),
        splat.FusedView(camera=other, **sun),
        splat.FusedView(camera=cam, **sun),
    ]
    common = dict(samples=8, width=128, height=128, return_aovs=True)
    seq = splat.render_fused_sequence(**kw, views=views, **common)
    assert len(seq) == 3
    for view, got in zip(views, seq):
        alone = splat.render_fused(**kw, camera=view.camera, **sun, **common)
        np.testing.assert_array_equal(got.rgba, alone.rgba)
        np.testing.assert_array_equal(got.radiance.view(np.uint32), alone.radiance.view(np.uint32))
    loads = [r.stats["paging"]["loads"] for r in seq]  # cumulative over the sequence
    assert loads[2] == loads[1], f"repeated view re-streamed pages: {loads}"
    assert max(r.stats["peak_total_bytes"] for r in seq) <= 512 * 1024 * 1024
    print(f"\nsequence paging loads (cumulative): {loads}")
    with pytest.raises(ValueError, match="unknown view key"):
        splat._native("render_fused_sequence")(
            views=[{"cam_origin": (0, 1, 2), "cam_look_at": (0, 0, 0), "bogus": 1}],
            heights=np.zeros((4, 4), np.float32),
        )
