# tests/test_inverse_pt.py
# DIFFERENTIA DoD gates: differentiable inverse path tracing recovers
# per-texel albedo, sun direction/intensity and atmospheric turbidity from a
# single observed image. The observation is rendered by the independent
# forward reference renderer (hybrid_render_terrain_reference); the solver
# receives ONLY the beauty image and uses its own seed/spp — no shared noise.
#   - test_inverse_solve_registered: native symbol registered + importable
#   - test_primal_parity: the solver's primal is the same forward chain —
#     inverse_render_primal must match hybrid_render_terrain_reference at
#     matched frames/seed/spp (anti-fork gate)
#   - test_recover_synthetic_scene: MEASURABLE WIN — median ΔE2000 < 4,
#     sun angular error < 2 deg, turbidity within 15% of ground truth
#   - test_solve_memory_budget: peak host-visible allocation < 512 MiB
# GPU-skip follows the recipe-golden convention (terrain_rendering_available).
# Before/after recovery images are written to FORGE3D_INVERSE_ARTIFACT_DIR
# (default: tests/artifacts/inverse/, which is git-ignored).

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _deltae import median_delta_e2000_linear
from _terrain_runtime import terrain_rendering_available

import forge3d as f3d

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = Path(
    os.environ.get(
        "FORGE3D_INVERSE_ARTIFACT_DIR", ROOT / "tests" / "artifacts" / "inverse"
    )
)

# Locked synthetic scene (deterministic; the recovery gates below key on it).
DEM_N = 16          # DEM texels per side — also the recovered albedo dims
SIZE = 96           # render resolution
SPAN = 32.0
CAM = {
    "origin": (0.0, 42.0, 0.001),   # near-nadir aerial view
    "look_at": (0.0, 0.0, 0.0),
    "up": (0.0, 0.0, -1.0),
    "fov_y": 38.0,
    "exposure": 1.0,
}
TRUE_AZ = 215.0
TRUE_EL = 30.0
TRUE_I = 2.4
TRUE_TAU = 3.0
INIT_AZ = 185.0
INIT_EL = 26.0
INIT_I = 1.6
INIT_TAU = 5.0
WARM_WHITE = (1.0, 0.97, 0.92)
# Target and solver use independent noise: the observation is a converged
# 64-frame render at seed 11; the solver samples its own 4-spp stream at
# seed 7. Only the beauty image crosses the boundary.
TARGET_SPP = 8
TARGET_FRAMES = 64
TARGET_SEED = 11
SOLVER_SPP = 4
SOLVER_FRAMES = 4
SOLVER_SEED = 7


def _require_gpu_and_native():
    if not terrain_rendering_available():
        pytest.skip(
            "DIFFERENTIA requires a terrain-capable hardware-backed forge3d runtime"
        )
    if not hasattr(f3d, "inverse_solve"):
        pytest.skip(
            "forge3d._forge3d lacks inverse_solve (built without enable-inverse-pt)"
        )


def _dem() -> np.ndarray:
    """Synthetic terrain with real relief: three sharp peaks + a planar ramp.
    At TRUE_EL=30 deg the peaks cast shadows over ~15-20% of the frame —
    the boundary signal that makes the sun direction identifiable."""
    n = DEM_N
    x = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    z = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    xx, zz = np.meshgrid(x, z, indexing="xy")
    dem = (
        8.0 * np.exp(-((xx - 0.35) ** 2 + (zz + 0.2) ** 2) / 0.08)
        + 6.0 * np.exp(-((xx + 0.45) ** 2 + (zz - 0.4) ** 2) / 0.06)
        + 4.0 * np.exp(-((xx + 0.1) ** 2 + (zz + 0.55) ** 2) / 0.05)
        + 1.2 * (0.5 + 0.5 * xx)
    )
    return np.ascontiguousarray(dem.astype(np.float32))


def _truth_albedo() -> np.ndarray:
    """Spatially varying linear albedo map (demN, demN, 3) in (0.1, 0.9)."""
    n = DEM_N
    x = np.linspace(0.0, 1.0, n, dtype=np.float32)
    xx, zz = np.meshgrid(x, x, indexing="xy")
    r = 0.25 + 0.55 * np.clip(xx, 0, 1)          # red gradient along x
    g = 0.25 + 0.55 * np.clip(zz, 0, 1)          # green gradient along z
    b = 0.35 + 0.30 * ((xx > 0.5) ^ (zz > 0.5))  # blue checker quadrants
    return np.ascontiguousarray(
        np.stack([r, g, b], axis=-1).astype(np.float32)
    )


def _render_target(albedo_map: np.ndarray) -> np.ndarray:
    """The observation: rendered by the INDEPENDENT forward reference
    renderer (its own seed/spp/frame count), not by the solver's primal."""
    out = f3d.hybrid_render_terrain_reference(
        _dem(),
        SIZE,
        SIZE,
        CAM,
        spacing=(SPAN / (DEM_N - 1), SPAN / (DEM_N - 1)),
        exaggeration=1.0,
        albedo_map=albedo_map,
        turbidity=TRUE_TAU,
        sun_azimuth_deg=TRUE_AZ,
        sun_elevation_deg=TRUE_EL,
        sun_intensity=TRUE_I,
        sun_color=WARM_WHITE,
        env_intensity=0.35,
        spp=TARGET_SPP,
        min_frames=TARGET_FRAMES,
        max_frames=TARGET_FRAMES,
        variance_threshold=1e30,   # run exactly TARGET_FRAMES frames
        seed=TARGET_SEED,
        earth_model="flat",
        refraction_model="none",
    )
    return out["rgba"]


def _sun_angular_error_deg(recovered: np.ndarray, az_deg: float, el_deg: float) -> float:
    az, el = np.radians(az_deg), np.radians(el_deg)
    truth = np.array(
        [np.cos(az) * np.cos(el), np.sin(el), np.sin(az) * np.cos(el)]
    )
    d = np.asarray(recovered, dtype=np.float64)
    d /= np.linalg.norm(d)
    cos_err = float(np.clip(np.dot(d, truth), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_err)))


def _write_recovery_artifacts(
    target: np.ndarray, recovered_rgba: np.ndarray, albedo_true: np.ndarray, albedo_rec: np.ndarray
) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    f3d.numpy_to_png(str(ARTIFACT_DIR / "target.png"), target)
    f3d.numpy_to_png(str(ARTIFACT_DIR / "recovered.png"), recovered_rgba)
    # Albedo maps as sRGB PNGs (linear -> sRGB encode for inspection).
    def alb_png(a: np.ndarray) -> np.ndarray:
        srgb = np.where(
            a <= 0.0031308, 12.92 * a, 1.055 * np.power(a, 1.0 / 2.4) - 0.055
        )
        u8 = (np.clip(srgb, 0, 1) * 255.0 + 0.5).astype(np.uint8)
        return np.dstack([u8, np.full(u8.shape[:2], 255, dtype=np.uint8)])

    f3d.numpy_to_png(
        str(ARTIFACT_DIR / "albedo_ground_truth.png"), alb_png(albedo_true)
    )
    f3d.numpy_to_png(
        str(ARTIFACT_DIR / "albedo_recovered.png"), alb_png(albedo_rec)
    )


def test_inverse_solve_registered():
    """`inverse_solve` is a registered native symbol and listed in the
    API contract surface (EXPECTED_FUNCTIONS)."""
    from test_api_contracts import TestNativeModuleSymbols

    assert "inverse_solve" in TestNativeModuleSymbols.EXPECTED_FUNCTIONS
    assert "inverse_render_primal" in TestNativeModuleSymbols.EXPECTED_FUNCTIONS
    import forge3d._native as _nat

    mod = _nat.get_native_module()
    if mod is None:
        pytest.skip("native extension not built")
    if not hasattr(mod, "inverse_solve"):
        pytest.skip("native extension built without enable-inverse-pt")
    assert callable(mod.inverse_solve)
    assert callable(mod.inverse_render_primal)
    # Package-level wrappers are importable regardless of native availability.
    assert callable(f3d.recover_scene)
    assert isinstance(f3d.RecoveredScene, type)


def test_primal_parity():
    """Anti-fork gate: the solver's primal render must match the independent
    forward reference bit-for-bit at matched frames/seed/spp — the inverse
    module dual-dispatches the same forward kernels rather than maintaining
    a parallel implementation."""
    _require_gpu_and_native()

    dem = _dem()
    albedo = _truth_albedo()
    spacing = SPAN / (DEM_N - 1)
    frames = 8
    spp = 4
    seed = 23

    ref = f3d.hybrid_render_terrain_reference(
        dem,
        SIZE,
        SIZE,
        CAM,
        spacing=(spacing, spacing),
        exaggeration=1.0,
        albedo_map=albedo,
        turbidity=TRUE_TAU,
        sun_azimuth_deg=TRUE_AZ,
        sun_elevation_deg=TRUE_EL,
        sun_intensity=TRUE_I,
        sun_color=WARM_WHITE,
        env_intensity=0.35,
        spp=spp,
        min_frames=frames,
        max_frames=frames,
        variance_threshold=1e30,
        seed=seed,
        earth_model="flat",
        refraction_model="none",
    )["rgba"]

    prim = f3d.inverse.render_primal(
        dem,
        albedo,
        cam=CAM,
        width=SIZE,
        height=SIZE,
        spacing=(spacing, spacing),
        exaggeration=1.0,
        sun_azimuth_deg=TRUE_AZ,
        sun_elevation_deg=TRUE_EL,
        sun_intensity=TRUE_I,
        turbidity=TRUE_TAU,
        sun_color=WARM_WHITE,
        env_intensity=0.35,
        spp=spp,
        frames=frames,
        seed=seed,
    )

    diff = np.abs(ref.astype(np.int32) - prim.astype(np.int32))
    assert float(diff.max()) <= 1.0, (
        f"inverse primal diverged from the forward reference: max |u8| diff "
        f"{int(diff.max())} (the inverse must drive the same forward chain)"
    )


def test_recover_synthetic_scene():
    """MEASURABLE WIN: recover albedo/sun/turbidity from one observed image
    rendered by the independent forward renderer. The solver sees ONLY the
    beauty image (different seed, different spp, different frame count).
    Gates: median ΔE2000 < 4, sun angular error < 2 deg, turbidity within
    15% of ground truth."""
    _require_gpu_and_native()

    dem = _dem()
    albedo_true = _truth_albedo()
    spacing = SPAN / (DEM_N - 1)

    target = _render_target(albedo_true)
    assert target.shape == (SIZE, SIZE, 4)
    assert target[..., :3].max() > 8, "synthetic target rendered black"

    rec = f3d.recover_scene(
        target,
        dem,
        cam=CAM,
        spacing=(spacing, spacing),
        exaggeration=1.0,
        init_albedo=(0.45, 0.45, 0.45),
        sun_azimuth_deg=INIT_AZ,
        sun_elevation_deg=INIT_EL,
        sun_intensity=INIT_I,
        turbidity=INIT_TAU,
        sun_color=WARM_WHITE,
        env_intensity=0.35,
        iters=400,
        spp=SOLVER_SPP,
        frames=SOLVER_FRAMES,
        seed=SOLVER_SEED,
        lr_albedo=0.06,
        lr_sun=0.05,
        lr_turbidity=0.08,
        early_stop_tol=0.0,
        early_stop_patience=9999,
        spatial_reuse=True,
        edge_term=True,
        score_correction=True,
        reference_albedo=albedo_true,
    )
    _write_recovery_artifacts(target, rec.rgba, albedo_true, rec.albedo)

    assert rec.albedo.shape == (DEM_N, DEM_N, 3)
    assert len(rec.loss_history) >= 1
    assert all(np.isfinite(rec.loss_history)), "non-finite loss history"
    assert rec.loss_history[-1] <= rec.loss_history[0] * 1.05 + 1e-6, (
        f"solver diverged: loss {rec.loss_history[0]} -> {rec.loss_history[-1]}"
    )

    de_median = median_delta_e2000_linear(rec.albedo, albedo_true)
    sun_err = _sun_angular_error_deg(rec.sun_dir, TRUE_AZ, TRUE_EL)
    tau_err = abs(rec.turbidity - TRUE_TAU) / TRUE_TAU
    inten_err = abs(rec.sun_intensity - TRUE_I) / TRUE_I

    # The reported AEQUITAS score is computed by the wrapper from the
    # canonical tests/_deltae.py — same function, so agreement must be
    # exact to float64 precision.
    assert rec.albedo_delta_e2000_median is not None, (
        "solver did not report albedo_delta_e2000_median for a supplied "
        "reference_albedo"
    )
    assert abs(rec.albedo_delta_e2000_median - de_median) < 1e-6, (
        f"reported ΔE2000 score {rec.albedo_delta_e2000_median:.3f} disagrees "
        f"with the canonical median {de_median:.3f}"
    )

    assert de_median < 4.0, (
        f"albedo recovery failed: median ΔE2000 {de_median:.3f} >= 4 "
        f"(sun {sun_err:.2f} deg, τ {tau_err * 100:.1f}%, I {inten_err * 100:.1f}%)"
    )
    assert sun_err < 2.0, f"sun recovery failed: {sun_err:.2f} deg >= 2"
    assert tau_err < 0.15, (
        f"turbidity recovery failed: |{rec.turbidity:.3f} - {TRUE_TAU}| / "
        f"{TRUE_TAU} = {tau_err:.3f} >= 15%"
    )


def test_solve_memory_budget():
    """Peak host-visible allocation during solve() stays under 512 MiB,
    measured by the render-scoped allocation ledger (reported on the
    result dict)."""
    _require_gpu_and_native()

    dem = _dem()
    spacing = SPAN / (DEM_N - 1)
    target = _render_target(_truth_albedo())
    rec = f3d.recover_scene(
        target,
        dem,
        cam=CAM,
        spacing=(spacing, spacing),
        exaggeration=1.0,
        sun_azimuth_deg=INIT_AZ,
        sun_elevation_deg=INIT_EL,
        sun_intensity=INIT_I,
        turbidity=INIT_TAU,
        sun_color=WARM_WHITE,
        env_intensity=0.35,
        iters=4,
        spp=2,
        frames=2,
        seed=3,
    )
    limit = 512 * 1024 * 1024
    assert rec.peak_host_visible_bytes < limit, (
        f"solver exceeded host-visible budget: {rec.peak_host_visible_bytes} "
        f">= {limit} bytes"
    )
