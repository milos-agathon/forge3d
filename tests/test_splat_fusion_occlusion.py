# tests/test_splat_fusion_occlusion.py
# SPLAT-FUSED measurable win, through the public Python API: cross-
# representation shadow correctness at billion-primitive scale inside the
# 512 MiB budget.
#
# The committed fixture (mini DEM with a ridge, a Gaussian splat cloud, a
# COPC LiDAR swath) is embedded in a synthetic billion-point page index and
# rendered twice:
#   * forge3d.splat.render_fused — the fused ReSTIR integrator; its shadow
#     mask is the unified transmittance T_total < 1/2;
#   * forge3d.splat.render_fused_reference — the AEQUITAS wavefront path
#     tracer over hard proxies of the same primitives; its shadow mask is
#     read from the path-traced radiance.
# Gate: IoU(fused_shadow, reference_shadow) > 0.9 for "splat shadow on
# terrain" and "terrain shadow on splat/LiDAR", peak residency <= 512 MiB,
# logical primitives > 1e9. A deterministic golden image of the fixture is
# drift-checked with the project's image tolerance.
#
# GPU-skip follows the recipe-golden convention (skip when no adapter,
# hard-fail on regression; FORGE3D_SPLAT_FUSION_REQUIRED_GPU=1 forbids the
# skip on the hardware lane).
# RELEVANT FILES: tests/test_splat_fusion_occlusion.rs, python/forge3d/splat.py,
#                 src/path_tracing/fused_reference.rs

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import forge3d as f3d
from forge3d import splat

from _splat_fusion import GOLDEN_PATH, gpu_available, manifest, scene_kwargs

pytestmark = pytest.mark.skipif(
    not splat.splat_fusion_available(),
    reason="wheel built without the splat-fusion feature",
)

SIZE = 256
IOU_GATE = 0.9
BUDGET_BYTES = 512 * 1024 * 1024
MIN_SUN_COSINE = 0.15
UPDATE_GOLDENS = os.environ.get("FORGE3D_UPDATE_SPLAT_FUSION_GOLDENS") == "1"
ARTIFACT_DIR = os.environ.get("FORGE3D_SPLAT_FUSION_ARTIFACT_DIR")

# (name, receiver hit kinds, world xz rectangle min_x, min_z, max_x, max_z).
# The rectangles are fixed in world space from the fixture layout (occluder
# footprint displaced along the sun direction), independent of either render.
CASES = (
    ("splat shadow on terrain", (splat.HIT_TERRAIN,), (10.0, -16.0, 27.0, -3.0)),
    ("LiDAR shadow on terrain", (splat.HIT_TERRAIN,), (8.0, 6.0, 21.0, 18.0)),
    ("terrain shadow on splat", (splat.HIT_SPLAT,), (-17.0, -12.0, -3.0, 0.0)),
    ("terrain shadow on LiDAR", (splat.HIT_LIDAR,), (-17.0, 3.0, -3.0, 15.0)),
    (
        "terrain shadow on splat/LiDAR",
        (splat.HIT_SPLAT, splat.HIT_LIDAR),
        (-17.0, -12.0, -3.0, 15.0),
    ),
)


@pytest.fixture(scope="module")
def dod(tmp_path_factory):
    """The definition-of-done renders: fused + path-traced reference."""
    if not gpu_available():
        pytest.skip("no usable GPU adapter for the fused render")
    m = manifest()
    work = tmp_path_factory.mktemp("splat_fusion")
    half = 0.5 * (m["dem_width"] - 1) * m["dem_spacing"][0]

    # A synthetic billion-point index around the fixture core: ~246k index
    # entries aliasing 16 on-disk payload pages. Nothing is resident until a
    # ray asks for it.
    field_path = work / "billion_point_field.f3dpages"
    field = splat.write_synthetic_point_field(
        field_path,
        page_capacity=4096,
        template_pages=16,
        grid=(496, 496),
        cell_size=16.0,
        origin=(-3968.0, 0.6, -3968.0),
        thickness=1.5,
        hole=(-half, -half, half, half),
        seed=0x5EEDF00D,
    )
    kwargs = scene_kwargs(m)
    fixture_cloud = kwargs["pointcloud"]
    fused = splat.render_fused(
        **{**kwargs, "pointcloud": [fixture_cloud, field_path]},
        splat_page_size=512,
        samples=96,
        samples_per_frame=2,
        width=SIZE,
        height=SIZE,
        seed=7,
        return_aovs=True,
    )
    reference = splat.render_fused_reference(
        splats=kwargs["splats"],
        pointcloud=[fixture_cloud, field_path],
        terrain=kwargs["terrain"],
        camera=kwargs["camera"],
        # The fixture core: strictly inside the DEM footprint.
        region=((-half + 0.5, -1000.0, -half + 0.5), (half - 0.5, 1000.0, half - 0.5)),
        samples=96,
        width=SIZE,
        height=SIZE,
        seed=11,
        sun_azimuth_deg=kwargs["sun_azimuth_deg"],
        sun_elevation_deg=kwargs["sun_elevation_deg"],
        sun_intensity=kwargs["sun_intensity"],
        sun_color=kwargs["sun_color"],
        lidar_radius=kwargs["lidar_radius"],
        splat_page_size=512,
        min_sun_cosine=MIN_SUN_COSINE,
    )
    if ARTIFACT_DIR:
        out = Path(ARTIFACT_DIR)
        out.mkdir(parents=True, exist_ok=True)
        f3d.numpy_to_png(str(out / "py_fused_beauty.png"), fused.rgba)
    return m, field, fused, reference


def _case_scores(fused, reference):
    fused_shadow = fused.transmittance[..., 0] < 0.5
    reference_shadow = reference["shadow"] == 1
    classified = reference["shadow"] >= 0
    x, z = fused.position[..., 0], fused.position[..., 2]
    scores = {}
    for name, kinds, (x0, z0, x1, z1) in CASES:
        receivers = np.isin(fused.hit_kind, kinds)
        valid = (
            receivers
            # Both renders agree on what the receiver is.
            & (reference["receiver_class"] == fused.hit_kind)
            & classified
            & (fused.sun_cosine > MIN_SUN_COSINE)
            & (x >= x0) & (x <= x1) & (z >= z0) & (z <= z1)
        )
        scores[name] = dict(
            iou=splat.shadow_iou(fused_shadow, reference_shadow, valid),
            valid=int(valid.sum()),
            fused=int((fused_shadow & valid).sum()),
            reference=int((reference_shadow & valid).sum()),
        )
    return scores


def test_cross_representation_shadow_iou_exceeds_gate(dod):
    m, _, fused, reference = dod
    assert reference["splat_proxies"] == m["splat_count"]
    assert reference["lidar_proxies"] == m["copc_point_count"]
    scores = _case_scores(fused, reference)
    print("\nshadow IoU, fused vs path-traced reference (gate > %.1f):" % IOU_GATE)
    for name, s in scores.items():
        print(
            f"  {name:<30} IoU {s['iou']:.4f}  (valid {s['valid']} px, "
            f"fused shadow {s['fused']}, reference shadow {s['reference']})"
        )
    for name, s in scores.items():
        # The case must be exercised: enough receivers, shadowed and lit.
        assert s["valid"] > 300, (name, s)
        assert 60 < s["fused"] < s["valid"], (name, s)
        assert 60 < s["reference"] < s["valid"], (name, s)
        assert s["iou"] > IOU_GATE, f"{name}: shadow IoU {s['iou']:.4f} <= {IOU_GATE} ({s})"


def test_billion_primitives_are_paged_inside_the_budget(dod):
    m, field, fused, _ = dod
    stats = fused.stats
    paging = stats["paging"]
    assert field["logical_primitives"] > 1_000_000_000
    assert stats["logical_primitives"] == (
        field["logical_primitives"] + m["splat_count"] + m["copc_point_count"]
    )
    assert stats["logical_primitives"] > 1_000_000_000
    print(
        f"\nlogical primitives: {stats['logical_primitives']} | peak resident pages: "
        f"{paging['peak_resident_pages']} of {stats['page_count']} | render peak total "
        f"{stats['peak_total_bytes'] / 2**20:.1f} MiB, host-visible "
        f"{stats['peak_host_visible_bytes'] / 2**20:.1f} MiB, pool "
        f"{paging['pool_bytes'] / 2**20:.1f} MiB (limit {stats['tracker_limit_bytes'] / 2**20:.0f} MiB)"
    )
    # Paged, not resident: far less than 1 % of the pages ever entered the pool.
    assert paging["peak_resident_pages"] < stats["page_count"] // 100
    assert paging["miss_events"] > 0, "the paging path never ran"
    # Far-field pages were streamed from the billion-point index on demand.
    fixture_pages = 7 + 3
    assert paging["loads"] > fixture_pages
    assert paging["deferred_pages"] == 0
    # 512 MiB through the memory tracker.
    assert stats["tracker_limit_bytes"] == BUDGET_BYTES
    assert stats["peak_total_bytes"] <= BUDGET_BYTES
    assert stats["peak_host_visible_bytes"] <= BUDGET_BYTES
    assert stats["tracker_peak_host_visible_bytes"] <= BUDGET_BYTES
    assert paging["pool_bytes"] <= BUDGET_BYTES // 2
    # One integrator: the ReSTIR chain ran and carried reservoirs.
    assert stats["frames"] == 48
    assert stats["reservoir_valid_count"] > 0


def test_fused_golden_image():
    if not gpu_available():
        pytest.skip("no usable GPU adapter for the fused render")
    from _ssim import ssim

    m = manifest()
    # The exact configuration of the Rust golden test (96x96, 8 spp x 256).
    rgba = splat.render_fused(
        **scene_kwargs(m),
        splat_page_size=512,
        samples=2048,
        samples_per_frame=8,
        width=96,
        height=96,
        seed=7,
    )
    if UPDATE_GOLDENS:
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        f3d.numpy_to_png(str(GOLDEN_PATH), rgba)
        pytest.skip(f"updated {GOLDEN_PATH}")
    assert GOLDEN_PATH.exists(), (
        f"Missing fused golden {GOLDEN_PATH}. "
        "Regenerate with FORGE3D_UPDATE_SPLAT_FUSION_GOLDENS=1."
    )
    expected = f3d.png_to_numpy(str(GOLDEN_PATH))
    assert expected.shape == rgba.shape
    mean_abs = float(
        np.abs(rgba[..., :3].astype(np.float64) - expected[..., :3].astype(np.float64)).mean()
    )
    score = ssim(rgba[..., :3], expected[..., :3], data_range=255.0)
    print(f"\nfused golden drift: SSIM {score:.6f}, mean abs {mean_abs:.4f}")
    assert score >= 0.995, f"fused golden drift: SSIM {score:.6f}"
    assert mean_abs <= 2.0, f"fused golden drift: mean abs {mean_abs:.4f}"
