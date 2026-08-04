"""SUBSTRATIA: full-PBR terrain VT normal + mask family gates.

Covers the moonshot definition-of-done:

- ``test_normal_family_changes_lighting_ssim`` — the gated measurable win: the
  normal VT family must change *beauty lighting* under grazing light by an
  SSIM difference > 0.05 (beyond the existing normal-AOV coverage in
  ``test_tv20_virtual_texturing.py``).
- ``test_all_families_page_within_budget`` — albedo, normal, and mask all page
  with per-family resident bytes > 0 whose sum stays within the VT residency
  budget and the 512 MiB host-visible ceiling.
- ``test_missing_family_is_fatal`` — a requested family with no registered
  source raises a fatal diagnostic instead of degrading silently.
- ``test_partial_normal_residency_degrades_gracefully`` — non-resident normal
  tiles fall back to the geometric surface normal (no corrupted/black-normal
  lighting) while the render completes.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

import forge3d as f3d
from _terrain_runtime import _build_heightmap, terrain_rendering_available
from forge3d.terrain_params import (
    AovSettings,
    PomSettings,
    TerrainVTSettings,
    VTLayerFamily,
    make_terrain_params_config,
)

GPU_AVAILABLE = terrain_rendering_available()
VT_MATERIAL_COUNT = 4
MIB = 1024.0 * 1024.0
MEMORY_BUDGET_LIMIT_BYTES = 512 * 1024 * 1024

# Labeled grazing-light detail region (fractions of image height/width) used
# by the SSIM gate: central band where the low-sun normal shading dominates.
GRAZING_REGION = (0.18, 0.85, 0.12, 0.88)


# ---------------------------------------------------------------------------
# Local SSIM harness (no external dependency)
# ---------------------------------------------------------------------------

def _box_mean(img: np.ndarray, radius: int) -> np.ndarray:
    """Edge-clamped box-filter mean with window (2*radius+1)^2."""
    size = 2 * radius + 1
    padded = np.pad(img, radius, mode="edge").astype(np.float64)
    csum = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    csum = np.pad(csum, ((1, 0), (1, 0)))
    h, w = img.shape
    total = (
        csum[size : size + h, size : size + w]
        - csum[0:h, size : size + w]
        - csum[size : size + h, 0:w]
        + csum[0:h, 0:w]
    )
    return total / float(size * size)


def _ssim(a: np.ndarray, b: np.ndarray, radius: int = 3) -> float:
    """Mean structural similarity of two grayscale images in [0, 1]."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1 = 0.01**2
    c2 = 0.03**2
    mu_a = _box_mean(a, radius)
    mu_b = _box_mean(b, radius)
    sigma_a = _box_mean(a * a, radius) - mu_a * mu_a
    sigma_b = _box_mean(b * b, radius) - mu_b * mu_b
    sigma_ab = _box_mean(a * b, radius) - mu_a * mu_b
    numerator = (2.0 * mu_a * mu_b + c1) * (2.0 * sigma_ab + c2)
    denominator = (mu_a * mu_a + mu_b * mu_b + c1) * (sigma_a + sigma_b + c2)
    return float(np.mean(numerator / denominator))


def _luminance(rgba: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgba, dtype=np.float64)[..., :3]
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _region_slices(shape: tuple[int, ...]) -> tuple[slice, slice]:
    top, bottom, left, right = GRAZING_REGION
    h, w = shape[0], shape[1]
    return (
        slice(int(h * top), int(h * bottom)),
        slice(int(w * left), int(w * right)),
    )


# ---------------------------------------------------------------------------
# Procedural VT sources
# ---------------------------------------------------------------------------

def _build_albedo_source(size: int, material_index: int) -> np.ndarray:
    coords = np.linspace(0.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(coords, coords)
    checker = ((np.floor(xx * 12) + np.floor(yy * 12)) % 2.0).astype(np.float32)
    palette = np.array(
        [
            [0.80, 0.25, 0.15],
            [0.20, 0.65, 0.25],
            [0.20, 0.35, 0.85],
            [0.90, 0.80, 0.20],
        ],
        dtype=np.float32,
    )
    base = palette[material_index % len(palette)]
    rgb = np.clip(base * (0.4 + 0.6 * checker[..., None]), 0.0, 1.0)
    rgba = np.concatenate([rgb, np.ones((size, size, 1), dtype=np.float32)], axis=-1)
    return np.ascontiguousarray((rgba * 255.0).round().astype(np.uint8))


def _build_bumpy_normal_source(
    size: int,
    material_index: int,
    frequency: float = 22.0,
    amplitude: float = 6.0,
) -> np.ndarray:
    """Tangent-space normal map with strong sinusoidal bumps.

    Encoded [0,1] RGB; decodes to normals with pronounced slopes so grazing
    light produces long shading variation across the tile.
    """
    coords = np.linspace(0.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(coords, coords)
    two_pi_f = 2.0 * np.pi * (frequency + material_index * 3.0)
    dzdx = amplitude * np.cos(two_pi_f * xx) * np.sin(two_pi_f * yy * 0.71 + 1.3)
    dzdy = amplitude * np.sin(two_pi_f * xx * 0.63 + 0.4) * np.cos(two_pi_f * yy)
    normal = np.stack([-dzdx, -dzdy, np.ones_like(dzdx)], axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[..., :3] = np.clip((normal * 0.5 + 0.5) * 255.0, 0.0, 255.0).astype(np.uint8)
    rgba[..., 3] = 255
    return np.ascontiguousarray(rgba)


def _build_mask_source(size: int, material_index: int) -> np.ndarray:
    """Mask family source: r = gate (on), g = roughness pattern, b = AO."""
    coords = np.linspace(0.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(coords, coords)
    rough = 0.25 + 0.5 * (
        0.5 + 0.5 * np.sin(xx * (18.0 + material_index)) * np.cos(yy * 15.0)
    )
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 1] = np.clip(rough * 255.0, 0.0, 255.0).astype(np.uint8)
    rgba[..., 2] = 255
    rgba[..., 3] = 255
    return np.ascontiguousarray(rgba)


def _register_family_sources(
    renderer: "f3d.TerrainRenderer",
    virtual_size: int,
    families: tuple[str, ...],
) -> None:
    builders = {
        "albedo": _build_albedo_source,
        "normal": _build_bumpy_normal_source,
        "mask": _build_mask_source,
    }
    fallbacks = {
        "albedo": [0.5, 0.5, 0.5, 1.0],
        "normal": [0.5, 0.5, 1.0, 1.0],
        "mask": [1.0, 1.0, 1.0, 1.0],
    }
    for material_index in range(VT_MATERIAL_COUNT):
        for family in families:
            renderer.register_material_vt_source(
                material_index,
                family,
                builders[family](virtual_size, material_index),
                (virtual_size, virtual_size),
                fallbacks[family],
            )


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

def _write_test_hdr(path: Path, width: int = 8, height: int = 4) -> None:
    with path.open("wb") as handle:
        handle.write(b"#?RADIANCE\n")
        handle.write(b"FORMAT=32-bit_rle_rgbe\n\n")
        handle.write(f"-Y {height} +X {width}\n".encode())
        for y in range(height):
            for x in range(width):
                r = int((x / max(width - 1, 1)) * 255)
                g = int((y / max(height - 1, 1)) * 255)
                handle.write(bytes([r, g, 164, 128]))


def _build_test_ibl():
    with tempfile.NamedTemporaryFile(suffix=".hdr", delete=False) as tmp:
        hdr_path = Path(tmp.name)
    try:
        _write_test_hdr(hdr_path)
        return f3d.IBL.from_hdr(str(hdr_path), intensity=1.0)
    finally:
        hdr_path.unlink(missing_ok=True)


def _build_render_params(
    *,
    vt_settings: TerrainVTSettings | None,
    size_px: tuple[int, int] = (256, 192),
    cam_target: tuple[float, float, float] = (0.0, 0.0, 0.0),
    cam_radius: float = 4.0,
    light_elevation_deg: float = 24.0,
    sun_intensity: float = 2.2,
    ibl_intensity: float = 1.8,
    normal_aov: bool = False,
) -> "f3d.TerrainRenderParams":
    config = make_terrain_params_config(
        size_px=size_px,
        render_scale=1.0,
        terrain_span=8.0,
        msaa_samples=1,
        z_scale=1.6,
        exposure=1.0,
        domain=(0.0, 1.0),
        albedo_mode="material",
        colormap_strength=0.0,
        ibl_enabled=True,
        ibl_intensity=ibl_intensity,
        light_azimuth_deg=136.0,
        light_elevation_deg=light_elevation_deg,
        sun_intensity=sun_intensity,
        cam_radius=cam_radius,
        cam_phi_deg=142.0,
        cam_theta_deg=58.0,
        fov_y_deg=50.0,
        camera_mode="mesh",
        pom=PomSettings(False, "Occlusion", 0.0, 1, 1, 0, False, False),
        aov=AovSettings(enabled=True, albedo=False, normal=normal_aov, depth=False),
    )
    config.cam_target = [float(cam_target[0]), float(cam_target[1]), float(cam_target[2])]
    config.vt = vt_settings
    return f3d.TerrainRenderParams(config)


def _render_beauty(env, params) -> np.ndarray:
    renderer, material_set, ibl, heightmap = env
    frame = renderer.render_terrain_pbr_pom(
        material_set=material_set,
        env_maps=ibl,
        params=params,
        heightmap=heightmap,
        target=None,
    )
    return np.asarray(frame.to_numpy())


def _render_beauty_and_normal_aov(env, params) -> tuple[np.ndarray, np.ndarray]:
    renderer, material_set, ibl, heightmap = env
    frame, aov_frame = renderer.render_with_aov(
        material_set=material_set,
        env_maps=ibl,
        params=params,
        heightmap=heightmap,
    )
    return (
        np.asarray(frame.to_numpy()),
        np.asarray(aov_frame.normal(), dtype=np.float32),
    )


def _dump_artifact(name: str, image: np.ndarray) -> None:
    artifact_dir = os.environ.get("FORGE3D_TERRAIN_GOLDEN_ARTIFACT_DIR")
    if not artifact_dir:
        return
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{name}.npy", image)


@pytest.fixture()
def vt_render_env():
    if not GPU_AVAILABLE:
        pytest.skip("terrain VT PBR tests require a terrain-capable GPU runtime")

    session = f3d.Session(window=False)
    renderer = f3d.TerrainRenderer(session)
    material_set = f3d.MaterialSet.terrain_default()
    heightmap = _build_heightmap(160)
    ibl = _build_test_ibl()
    renderer.clear_material_vt_sources()
    try:
        yield renderer, material_set, ibl, heightmap
    finally:
        try:
            renderer.clear_material_vt_sources()
        except RuntimeError:
            pass


# ---------------------------------------------------------------------------
# Static contracts (no GPU required)
# ---------------------------------------------------------------------------

def test_vt_settings_families_property() -> None:
    settings = TerrainVTSettings(
        layers=[
            VTLayerFamily(family="albedo"),
            VTLayerFamily(family="normal"),
            VTLayerFamily(family="mask"),
        ]
    )
    assert settings.families == ("albedo", "normal", "mask")


def test_shader_carries_family_info_and_residency_gate() -> None:
    shader = (
        Path(__file__).resolve().parents[1] / "src" / "shaders" / "terrain_pbr_pom.wgsl"
    ).read_text(encoding="utf-8")
    for token in (
        "family_info: array<vec4<u32>, 3>",
        "fn terrain_vt_resolve_family_uv(",
        "fn terrain_vt_sample_family_data(",
    ):
        assert token in shader, f"missing hardened VT shader token: {token}"


# ---------------------------------------------------------------------------
# GPU-gated DoD tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not GPU_AVAILABLE, reason="requires GPU-backed forge3d runtime")
class TestTerrainVTPbrFamilies:
    def test_normal_family_changes_lighting_ssim(self, vt_render_env) -> None:
        """Gated measurable win: the normal family must change grazing-light
        beauty output by SSIM difference > 0.05."""
        renderer = vt_render_env[0]
        renderer.clear_material_vt_sources()
        virtual_size = 2048

        grazing = dict(
            light_elevation_deg=7.0,
            sun_intensity=3.5,
            ibl_intensity=0.25,
            cam_radius=3.0,
        )

        baseline = _render_beauty(
            vt_render_env,
            _build_render_params(vt_settings=None, **grazing),
        )

        _register_family_sources(renderer, virtual_size, ("normal",))
        vt_settings = TerrainVTSettings(
            enabled=True,
            atlas_size=4096,
            residency_budget_mb=192.0,
            max_mip_levels=6,
            layers=[
                VTLayerFamily(
                    family="normal",
                    virtual_size_px=(virtual_size, virtual_size),
                    fallback=(0.5, 0.5, 1.0, 1.0),
                )
            ],
        )
        with_normal = _render_beauty(
            vt_render_env,
            _build_render_params(vt_settings=vt_settings, **grazing),
        )
        stats = renderer.get_material_vt_stats()

        _dump_artifact("vt_ssim_baseline", baseline)
        _dump_artifact("vt_ssim_with_normal", with_normal)

        rows, cols = _region_slices(baseline.shape)
        ssim_value = _ssim(
            _luminance(baseline)[rows, cols],
            _luminance(with_normal)[rows, cols],
        )
        ssim_delta = 1.0 - ssim_value

        # Attribution: the only difference between the renders is the resident
        # normal family.
        assert stats["resident_tiles_normal"] > 0.0
        assert stats["resident_tiles_albedo"] == pytest.approx(0.0)
        assert stats["resident_tiles_mask"] == pytest.approx(0.0)
        assert ssim_delta > 0.05, (
            f"normal family must change grazing-light beauty output: "
            f"SSIM delta {ssim_delta:.4f} <= 0.05 (SSIM {ssim_value:.4f})"
        )

    def test_all_families_page_within_budget(self, vt_render_env) -> None:
        renderer = vt_render_env[0]
        renderer.clear_material_vt_sources()
        virtual_size = 2048
        budget_mb = 96.0

        _register_family_sources(renderer, virtual_size, ("albedo", "normal", "mask"))
        vt_settings = TerrainVTSettings(
            enabled=True,
            atlas_size=4096,
            residency_budget_mb=budget_mb,
            max_mip_levels=6,
            layers=[
                VTLayerFamily(family="albedo", virtual_size_px=(virtual_size, virtual_size)),
                VTLayerFamily(
                    family="normal",
                    virtual_size_px=(virtual_size, virtual_size),
                    fallback=(0.5, 0.5, 1.0, 1.0),
                ),
                VTLayerFamily(
                    family="mask",
                    virtual_size_px=(virtual_size, virtual_size),
                    fallback=(1.0, 1.0, 1.0, 1.0),
                ),
            ],
        )

        # Camera sweep to drive feedback/paging across the virtual extent.
        for cam_target in ((-2.0, -2.0, 0.0), (2.0, 2.0, 0.0)):
            _render_beauty(
                vt_render_env,
                _build_render_params(
                    vt_settings=vt_settings,
                    cam_target=cam_target,
                    cam_radius=1.5,
                ),
            )
        stats = renderer.get_material_vt_stats()

        budget_bytes = budget_mb * MIB
        resident_sum = 0.0
        budget_sum = 0.0
        for family in ("albedo", "normal", "mask"):
            assert stats[f"resident_bytes_{family}"] > 0.0, (
                f"family '{family}' paged no tiles during the sweep"
            )
            assert stats[f"resident_bytes_{family}"] <= stats[f"budget_bytes_{family}"]
            resident_sum += stats[f"resident_bytes_{family}"]
            budget_sum += stats[f"budget_bytes_{family}"]

        assert resident_sum == pytest.approx(stats["resident_bytes_total"])
        assert resident_sum <= budget_bytes
        assert budget_sum <= budget_bytes
        assert resident_sum <= MEMORY_BUDGET_LIMIT_BYTES
        assert budget_bytes <= MEMORY_BUDGET_LIMIT_BYTES

    def test_missing_family_is_fatal(self, vt_render_env) -> None:
        renderer = vt_render_env[0]
        renderer.clear_material_vt_sources()
        virtual_size = 1024

        # Register albedo only, then request albedo + normal.
        _register_family_sources(renderer, virtual_size, ("albedo",))
        vt_settings = TerrainVTSettings(
            enabled=True,
            atlas_size=2048,
            residency_budget_mb=32.0,
            max_mip_levels=4,
            layers=[
                VTLayerFamily(family="albedo", virtual_size_px=(virtual_size, virtual_size)),
                VTLayerFamily(
                    family="normal",
                    virtual_size_px=(virtual_size, virtual_size),
                    fallback=(0.5, 0.5, 1.0, 1.0),
                ),
            ],
        )

        with pytest.raises(
            RuntimeError,
            match=r"family 'normal' requested but no source registered",
        ):
            _render_beauty(
                vt_render_env,
                _build_render_params(vt_settings=vt_settings),
            )

    def test_partial_normal_residency_degrades_gracefully(self, vt_render_env) -> None:
        renderer = vt_render_env[0]
        renderer.clear_material_vt_sources()
        virtual_size = 1024

        def normal_only_settings(budget_mb: float) -> TerrainVTSettings:
            # max_mip_levels=1 removes the coarse-mip rescue path so
            # non-resident tiles must fall back to the geometric normal;
            # use_feedback=False keeps the resident subset deterministic.
            return TerrainVTSettings(
                enabled=True,
                atlas_size=4096,
                residency_budget_mb=budget_mb,
                max_mip_levels=1,
                use_feedback=False,
                layers=[
                    VTLayerFamily(
                        family="normal",
                        virtual_size_px=(virtual_size, virtual_size),
                        fallback=(0.5, 0.5, 1.0, 1.0),
                    )
                ],
            )

        params_kwargs = dict(cam_radius=3.0, normal_aov=True)
        baseline_beauty, baseline_aov = _render_beauty_and_normal_aov(
            vt_render_env,
            _build_render_params(vt_settings=None, **params_kwargs),
        )

        _register_family_sources(renderer, virtual_size, ("normal",))
        full_beauty, full_aov = _render_beauty_and_normal_aov(
            vt_render_env,
            _build_render_params(vt_settings=normal_only_settings(64.0), **params_kwargs),
        )
        full_stats = renderer.get_material_vt_stats()

        partial_beauty, partial_aov = _render_beauty_and_normal_aov(
            vt_render_env,
            _build_render_params(vt_settings=normal_only_settings(1.0), **params_kwargs),
        )
        partial_stats = renderer.get_material_vt_stats()

        # The render completed with finite output.
        for image in (partial_beauty, partial_aov):
            assert np.all(np.isfinite(np.asarray(image, dtype=np.float64)))

        # Partial run holds strictly fewer normal tiles than the full run.
        assert partial_stats["resident_tiles_normal"] > 0.0
        assert (
            partial_stats["resident_tiles_normal"]
            < full_stats["resident_tiles_normal"]
        )

        # Identify fragments whose normal tiles fell back in the partial run:
        # their normal AOV matches the geometric baseline although the fully
        # resident render disagrees with it.
        aov_delta_partial = np.max(np.abs(partial_aov - baseline_aov), axis=-1)
        aov_delta_full = np.max(np.abs(full_aov - baseline_aov), axis=-1)
        fallback_region = (aov_delta_partial < 0.01) & (aov_delta_full > 0.03)
        assert fallback_region.mean() > 0.02, (
            "expected a visible region of non-resident normal tiles "
            f"(got {fallback_region.mean():.4f} coverage)"
        )

        # Graceful degradation: in the fallback region, beauty lighting matches
        # the geometric-normal baseline (no corrupted/black-normal lighting).
        baseline_lum = _luminance(baseline_beauty)
        partial_lum = _luminance(partial_beauty)
        region_diff = np.abs(partial_lum - baseline_lum)[fallback_region]
        assert float(region_diff.mean()) < 0.02, (
            f"fallback region deviates from geometric baseline: "
            f"mean {float(region_diff.mean()):.4f}"
        )
        # And nothing collapsed to black in the fallback region.
        assert float(partial_lum[fallback_region].min()) >= 0.0
        assert float(partial_lum[fallback_region].mean()) > 0.01
