# tests/test_adjudication_gate.py
# AEQUITAS perceptual adjudication gate: PT-vs-raster ground-truth parity.
#
# Section 1 (no GPU): unit tests for tests/_deltae.py against the Sharma et al.
# (2005) CIEDE2000 reference vectors and mask/band sanity checks.
# Section 2 (GPU): renders the committed reference scene both ways via
# forge3d.render_adjudication_pair and asserts the measurable win:
#   dE2000 < 2.0 on >= 95% of lit pixels  AND  SSIM > 0.96 on the shadow band.
# Follows the recipe-golden skip convention (skip on unsupported hosted GPUs,
# hard-fail on regression) and persists goldens under tests/golden/adjudication/.

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _deltae import (
    band_bbox,
    delta_e_2000,
    lit_mask,
    shadow_boundary_band,
    srgb_to_lab,
    srgb_to_linear,
)
from _ssim import ssim
import _terrain_runtime

import forge3d as f3d

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = ROOT / "tests" / "golden" / "adjudication"
UPDATE_GOLDENS = os.environ.get("FORGE3D_UPDATE_ADJUDICATION_GOLDENS") == "1"
ARTIFACT_DIR = os.environ.get("FORGE3D_ADJUDICATION_ARTIFACT_DIR")

# Gate parameters (locked; changing them invalidates the committed goldens).
GATE_WIDTH = 512
GATE_HEIGHT = 512
GATE_SPP = 4096
DELTA_E_MAX = 2.0
LIT_PASS_FRACTION_MIN = 0.95
BAND_SSIM_MIN = 0.96
# Drift thresholds vs committed goldens (terrain-golden convention).
DRIFT_SSIM_MIN = 0.995
DRIFT_MEAN_ABS_MAX = 2.0


# ---------------------------------------------------------------------------
# Section 1: metric primitives (always run; no GPU required)
# ---------------------------------------------------------------------------

# Reference pairs from Sharma, Wu & Dalal (2005), Table 1 (kL=kC=kH=1).
SHARMA_VECTORS = [
    ((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425),
    ((50.0, 3.1571, -77.2803), (50.0, 0.0, -82.7485), 2.8615),
    ((50.0, 2.8361, -74.0200), (50.0, 0.0, -82.7485), 3.4412),
    ((50.0, -1.3802, -84.2814), (50.0, 0.0, -82.7485), 1.0000),
    ((50.0, -1.1848, -84.8006), (50.0, 0.0, -82.7485), 1.0000),
    ((50.0, 0.0, 0.0), (50.0, -1.0, 2.0), 2.3669),
    ((50.0, 2.5, 0.0), (50.0, 0.0, -2.5), 4.3065),
    ((50.0, 2.5, 0.0), (73.0, 25.0, -18.0), 27.1492),
    ((50.0, 2.5, 0.0), (61.0, -5.0, 29.0), 22.8977),
    ((2.0776, 0.0795, -1.1350), (0.9033, -0.0636, -0.5514), 0.9082),
]


@pytest.mark.parametrize("lab1,lab2,expected", SHARMA_VECTORS)
def test_deltae2000_sharma_vectors(lab1, lab2, expected):
    got = float(delta_e_2000(np.array(lab1), np.array(lab2)))
    assert abs(got - expected) < 1e-3, f"dE2000({lab1},{lab2}) = {got:.4f}, want {expected:.4f}"
    # Symmetry (kL=kC=kH=1 makes CIEDE2000 symmetric).
    rev = float(delta_e_2000(np.array(lab2), np.array(lab1)))
    assert abs(rev - expected) < 1e-3


def test_srgb_to_lab_white_and_black():
    lab = srgb_to_lab(np.array([[[255, 255, 255]]], dtype=np.uint8))
    assert np.allclose(lab[0, 0], [100.0, 0.0, 0.0], atol=1e-2)
    lab0 = srgb_to_lab(np.array([[[0, 0, 0]]], dtype=np.uint8))
    assert np.allclose(lab0[0, 0], [0.0, 0.0, 0.0], atol=1e-6)


@pytest.mark.parametrize("code", [0, 1, 10, 11, 255])
def test_srgb_uint8_matches_normalized_float(code):
    encoded = np.full((1, 1, 3), code, dtype=np.uint8)
    normalized = encoded.astype(np.float64) / 255.0
    np.testing.assert_array_equal(srgb_to_linear(encoded), srgb_to_linear(normalized))
    np.testing.assert_array_equal(srgb_to_lab(encoded), srgb_to_lab(normalized))
    if code == 1:
        np.testing.assert_array_equal(srgb_to_linear(encoded), np.full(encoded.shape, 1.0 / 255.0 / 12.92))
        assert not lit_mask(encoded).any()


@pytest.mark.parametrize("key", ["lit_pass_fraction", "shadow_band_ssim"])
def test_reference_consistency_rejects_lowered_scores(key):
    reference = {
        "width": GATE_WIDTH, "height": GATE_HEIGHT, "spp": GATE_SPP,
        "ssim_region": "shadow_boundary_pixels",
        "lit_pass_fraction": 1.0, "shadow_band_ssim": 1.0,
    }
    measured = dict(reference)
    # A committed value 1e-6 below the PNG-recomputed score is a real
    # mismatch; platform recompute noise is ~1e-14 and the tolerance is 1e-9.
    reference[key] = 1.0 - 1e-6
    with pytest.raises(AssertionError, match="reference does not describe PNG"):
        _assert_reference_consistency(measured, reference)


def test_lit_mask_and_boundary_band_shapes():
    img = np.zeros((32, 48, 3), dtype=np.uint8)
    img[:, 24:, :] = 200  # right half lit
    lit = lit_mask(img)
    assert lit.shape == (32, 48) and lit.dtype == bool
    assert not lit[:, :24].any() and lit[:, 24:].all()

    band = shadow_boundary_band(img, band_px=3)
    assert band.shape == (32, 48) and band.dtype == bool
    # Band straddles the lit/shadow edge at x=24 and is thin.
    assert band[:, 24].all()
    assert not band[:, :19].any() and not band[:, 40:].any()
    ys, xs = band_bbox(band)
    assert xs.start >= 19 and xs.stop <= 40


def test_ssim_boundary_mask_excludes_bounding_box_interior():
    reference = np.full((64, 64, 3), 200, dtype=np.uint8)
    band = np.zeros((64, 64), dtype=bool)
    band[8:12, 8:56] = True
    band[52:56, 8:56] = True
    changed = reference.copy()
    changed[24:40, 24:40] = 0
    assert ssim(reference, changed, mask=band) == 1.0
    ys, xs = band_bbox(band)
    assert ssim(reference[ys, xs], changed[ys, xs]) < 1.0
    changed[band] = 0
    assert ssim(reference, changed, mask=band) < BAND_SSIM_MIN


def test_ssim_full_mask_matches_unmasked():
    rng = np.random.default_rng(7)
    a = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    b = rng.integers(0, 256, a.shape, dtype=np.uint8)
    assert ssim(a, b, mask=np.ones(a.shape[:2], dtype=bool)) == ssim(a, b)


@pytest.mark.parametrize("mask", [
    np.zeros((16, 16), dtype=bool),
    np.ones((16, 15), dtype=bool),
    np.ones((16, 16), dtype=np.uint8),
])
def test_ssim_rejects_invalid_mask(mask):
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="mask"):
        ssim(image, image, mask=mask)


def test_deltae_identical_images_is_zero():
    rng = np.random.default_rng(7)
    img = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    lab = srgb_to_lab(img)
    de = delta_e_2000(lab, lab)
    assert de.shape == (16, 16)
    assert float(np.abs(de).max()) < 1e-9


# ---------------------------------------------------------------------------
# Section 2: the adjudication gate (GPU; recipe-golden skip convention)
# ---------------------------------------------------------------------------


def _save_png(path: Path, rgba: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    f3d.numpy_to_png(str(path), rgba)


def _write_failure_artifacts(name: str, actual: np.ndarray, expected: np.ndarray) -> None:
    if not ARTIFACT_DIR:
        return
    out = Path(ARTIFACT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    _save_png(out / f"{name}_actual.png", actual)
    _save_png(out / f"{name}_expected.png", expected)
    diff = np.abs(
        actual[..., :3].astype(np.int16) - expected[..., :3].astype(np.int16)
    ).astype(np.uint8)
    _save_png(out / f"{name}_diff.png", diff)


def _assert_matches_golden(name: str, actual: np.ndarray) -> None:
    golden_path = GOLDEN_DIR / f"{name}.png"
    if UPDATE_GOLDENS:
        _save_png(golden_path, actual)
        return
    assert golden_path.exists(), (
        f"Missing adjudication golden {golden_path}. "
        "Regenerate with FORGE3D_UPDATE_ADJUDICATION_GOLDENS=1."
    )
    expected = f3d.png_to_numpy(str(golden_path))
    assert actual.shape == expected.shape
    mean_abs = float(
        np.mean(np.abs(actual[..., :3].astype(np.float32) - expected[..., :3].astype(np.float32)))
    )
    score = ssim(actual[..., :3], expected[..., :3], data_range=255.0)
    if score < DRIFT_SSIM_MIN or mean_abs > DRIFT_MEAN_ABS_MAX:
        _write_failure_artifacts(name, actual, expected)
    assert score >= DRIFT_SSIM_MIN, f"{name} drift: SSIM too low vs golden: {score:.6f}"
    assert mean_abs <= DRIFT_MEAN_ABS_MAX, f"{name} drift: mean abs diff too high: {mean_abs:.4f}"


def _assert_score_reference(scores, reference):
    for key, expected in (
        ("width", GATE_WIDTH), ("height", GATE_HEIGHT), ("spp", GATE_SPP),
        ("ssim_region", "shadow_boundary_pixels"),
    ):
        assert reference[key] == expected, f"score reference configuration mismatch: {key}"
    for key, minimum in (
        ("lit_pass_fraction", LIT_PASS_FRACTION_MIN),
        ("shadow_band_ssim", BAND_SSIM_MIN),
    ):
        value = reference[key]
        assert type(value) in (int, float) and np.isfinite(value)
        assert minimum <= value <= 1.0, f"invalid reference score: {key}"
        if key == "shadow_band_ssim":
            assert value > minimum
        assert np.isfinite(scores[key]) and scores[key] >= value, (
            f"{key} score drift: {scores[key]} < committed {value}"
        )


def _assert_reference_consistency(scores, reference):
    _assert_score_reference(scores, reference)
    for key in ("lit_pass_fraction", "shadow_band_ssim"):
        # Recomputed float64 metrics differ across platforms in the last bits;
        # a 1e-9 absolute tolerance still catches a stale scores.json, whose
        # drift is orders of magnitude larger.
        assert abs(scores[key] - reference[key]) <= 1e-9, (
            f"reference does not describe PNG: {key} "
            f"({scores[key]} vs committed {reference[key]})"
        )


def _load_score_reference():
    pt = f3d.png_to_numpy(str(GOLDEN_DIR / "pt_reference.png"))
    raster = f3d.png_to_numpy(str(GOLDEN_DIR / "raster_reference.png"))
    assert pt.shape == raster.shape == (GATE_HEIGHT, GATE_WIDTH, 4)
    lit = lit_mask(pt)
    assert lit.any()
    de = delta_e_2000(srgb_to_lab(pt), srgb_to_lab(raster))
    scores = {
        "lit_pass_fraction": float((de[lit] < DELTA_E_MAX).mean()),
        "shadow_band_ssim": ssim(
            pt[..., :3], raster[..., :3], mask=shadow_boundary_band(pt)
        ),
    }
    reference = json.loads((GOLDEN_DIR / "scores.json").read_text())
    _assert_reference_consistency(scores, reference)
    return reference


def test_committed_adjudication_reference_metrics():
    _load_score_reference()


@pytest.mark.parametrize("key", ["lit_pass_fraction", "shadow_band_ssim"])
def test_numeric_score_drift_is_rejected(key):
    reference = {
        "width": GATE_WIDTH, "height": GATE_HEIGHT, "spp": GATE_SPP,
        "ssim_region": "shadow_boundary_pixels",
        "lit_pass_fraction": 1.0, "shadow_band_ssim": 1.0,
    }
    scores = dict(reference)
    scores[key] = np.nextafter(1.0, 0.0)
    with pytest.raises(AssertionError, match="score drift"):
        _assert_score_reference(scores, reference)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, True])
def test_invalid_numeric_reference_is_rejected(value):
    reference = {
        "width": GATE_WIDTH, "height": GATE_HEIGHT, "spp": GATE_SPP,
        "ssim_region": "shadow_boundary_pixels",
        "lit_pass_fraction": value, "shadow_band_ssim": 1.0,
    }
    with pytest.raises(AssertionError):
        _assert_score_reference(reference, reference)


def test_production_instanced_mesh_pbr_mode():
    if not _terrain_runtime.adjudication_rendering_available():
        pytest.skip("PBR mesh rendering requires a hardware-backed runtime")
    from forge3d.geometry import primitive_mesh, instance_mesh_gpu_render

    mesh = primitive_mesh("sphere")
    transforms = np.eye(4, dtype=np.float32).reshape(1, 16)
    legacy = instance_mesh_gpu_render(mesh, transforms)
    pbr = instance_mesh_gpu_render(mesh, transforms, pbr={
        "base_color": [0.63, 0.28, 0.22, 1.0],
        "roughness": 0.7,
        "metallic": 0.0,
        "environment": [0.4, 0.48, 0.62],
    })
    assert pbr.shape == legacy.shape and pbr.dtype == np.uint8
    assert pbr[..., :3].any()
    assert pbr[..., 0].sum() > pbr[..., 2].sum(), "PBR shader did not consume the red material"
    assert not np.array_equal(pbr, legacy)
    with pytest.raises(ValueError, match="finite solid materials"):
        instance_mesh_gpu_render(mesh, transforms, pbr={"roughness": float("nan")})
    with pytest.raises(ValueError, match="unknown PBR mesh setting"):
        instance_mesh_gpu_render(mesh, transforms, pbr={"unknown": 0.0})


def test_adjudication_availability_does_not_run_terrain_probe(monkeypatch):
    monkeypatch.setattr(_terrain_runtime, "_running_on_unsupported_hosted_macos_ci", lambda: False)
    monkeypatch.setattr(_terrain_runtime, "_running_on_unsupported_hosted_windows_ci", lambda: False)

    def fail_terrain_probe():
        raise AssertionError("adjudication availability ran a terrain render")

    monkeypatch.setattr(_terrain_runtime, "terrain_rendering_available", fail_terrain_probe)
    monkeypatch.setattr(f3d, "has_gpu", lambda: True)
    monkeypatch.setattr(f3d, "device_probe", lambda *_: {
        "status": "ok", "device_type": "DiscreteGpu", "name": "test GPU",
    })
    assert _terrain_runtime.adjudication_rendering_available()


def test_adjudication_render_failure_is_not_skipped(monkeypatch):
    monkeypatch.setattr(_terrain_runtime, "adjudication_rendering_available", lambda: True)
    monkeypatch.setattr(_terrain_runtime, "adjudication_shadow_kernel_compiles", lambda: (True, None))

    def fail_render(*_args):
        raise RuntimeError("adjudication render failed")

    monkeypatch.setattr(f3d, "render_adjudication_pair", fail_render)
    with pytest.raises(RuntimeError, match="adjudication render failed"):
        test_adjudication_gate()


def test_shadow_kernel_compile_failure_skips_before_render(monkeypatch):
    monkeypatch.setattr(_terrain_runtime, "adjudication_rendering_available", lambda: True)
    monkeypatch.setattr(
        _terrain_runtime,
        "adjudication_shadow_kernel_compiles",
        lambda: (False, "unsupported atomic operation"),
    )

    def fail_render(*_args):
        raise AssertionError("adjudication rendered after a failed shadow probe")

    monkeypatch.setattr(f3d, "render_adjudication_pair", fail_render)
    with pytest.raises(pytest.skip.Exception, match="unsupported atomic operation"):
        test_adjudication_gate()


def test_adjudication_gate():
    if not _terrain_runtime.adjudication_rendering_available():
        pytest.skip(
            "Adjudication gate requires a hardware-backed forge3d runtime with the capture API"
        )
    shadow_kernel_compiles, reason = _terrain_runtime.adjudication_shadow_kernel_compiles()
    if not shadow_kernel_compiles:
        pytest.skip(f"wavefront shadow kernel did not compile: {reason}")

    pt_rgba, raster_rgba, meta = f3d.render_adjudication_pair(
        GATE_WIDTH, GATE_HEIGHT, GATE_SPP
    )
    assert pt_rgba.shape == (GATE_HEIGHT, GATE_WIDTH, 4) and pt_rgba.dtype == np.uint8
    assert raster_rgba.shape == (GATE_HEIGHT, GATE_WIDTH, 4) and raster_rgba.dtype == np.uint8
    from forge3d._native import get_native_module

    memory = dict(get_native_module().global_memory_metrics())
    peak_host_visible = memory["peak_host_visible_bytes"]
    assert 0 < peak_host_visible <= 512 * 1024 * 1024
    print(f"ADJUDICATION: process peak tracked host-visible bytes = {peak_host_visible}")
    if ARTIFACT_DIR:
        out = Path(ARTIFACT_DIR)
        _save_png(out / "pt_actual.png", pt_rgba)
        _save_png(out / "raster_actual.png", raster_rgba)
        (out / "capture.json").write_text(json.dumps({
            "scene": meta,
            "adapter": f3d.device_probe(os.environ.get("WGPU_BACKEND")),
            "memory": memory,
        }, indent=2))

    # Strict routing: the raster image is rendered by the interactive
    # viewer's frame pipeline (headless), not by a private offscreen fork,
    # and its metadata is what that viewer frame actually consumed.
    assert meta["raster_route"] == "viewer.pbr_scene"
    assert meta["raster_metadata_source"] == "rendered_frame"

    # Both renders must come from the single ReferenceSceneDesc: the raster
    # metadata (viewer-consumed values) must be byte-identical to the PT
    # metadata for every shared key. `spp` is PT-only.
    pt_meta, raster_meta = dict(meta["pt"]), dict(meta["raster"])
    assert pt_meta.pop("spp") == GATE_SPP
    assert set(raster_meta) == set(pt_meta), (
        f"shared metadata keys differ: pt={sorted(pt_meta)} raster={sorted(raster_meta)}"
    )
    for key, value in pt_meta.items():
        assert np.float64(raster_meta[key]).tobytes() == np.float64(value).tobytes(), (
            f"metadata {key} differs: pt={value!r} raster={raster_meta[key]!r}"
        )

    # Constant sky/ambient contract: both paths must report the literal
    # constants from the single ReferenceSceneDesc (no gradient fields).
    for key in ("ambient_r", "ambient_g", "ambient_b", "sky_r", "sky_g", "sky_b"):
        assert key in pt_meta, f"missing constant ambient/sky metadata key {key}"

    # --- Metric 1: dE2000 over lit pixels of the PT reference ---
    lit = lit_mask(pt_rgba)
    assert lit.any(), "lit-pixel mask is empty; scene/exposure regression"
    de = delta_e_2000(srgb_to_lab(pt_rgba), srgb_to_lab(raster_rgba))
    lit_pass_fraction = float((de[lit] < DELTA_E_MAX).mean())

    # --- Metric 2: SSIM on the shadow-boundary band of the PT reference ---
    band = shadow_boundary_band(pt_rgba)
    band_ssim = ssim(
        pt_rgba[..., :3], raster_rgba[..., :3], data_range=255.0, mask=band
    )

    print(
        f"\nADJUDICATION: dE2000<{DELTA_E_MAX} on {lit_pass_fraction * 100.0:.4f}% "
        f"of lit pixels (need >= {LIT_PASS_FRACTION_MIN * 100.0:.1f}%); "
        f"shadow-boundary SSIM = {band_ssim:.6f} (need > {BAND_SSIM_MIN})"
    )

    if UPDATE_GOLDENS:
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        (GOLDEN_DIR / "scores.json").write_text(
            json.dumps(
                {
                    "width": GATE_WIDTH,
                    "height": GATE_HEIGHT,
                    "spp": GATE_SPP,
                    "ssim_region": "shadow_boundary_pixels",
                    # Truncate to 9 decimals: the committed reference must stay
                    # at or below the true score so a recompute on another
                    # platform (float64 reduction noise ~1e-14) still satisfies
                    # the drift check's `>=`.
                    "lit_pass_fraction": int(lit_pass_fraction * 1e9) / 1e9,
                    "shadow_band_ssim": int(band_ssim * 1e9) / 1e9,
                },
                indent=2,
            )
            + "\n"
        )

    # The measurable win (hard gate; not xfail, not skip-on-error).
    assert lit_pass_fraction >= LIT_PASS_FRACTION_MIN, (
        f"dE2000 gate failed: only {lit_pass_fraction * 100.0:.4f}% of lit pixels "
        f"under dE {DELTA_E_MAX}"
    )
    assert band_ssim > BAND_SSIM_MIN, (
        f"shadow-boundary SSIM gate failed: {band_ssim:.6f} <= {BAND_SSIM_MIN}"
    )

    if not UPDATE_GOLDENS:
        reference = _load_score_reference()
        _assert_score_reference(
            {"lit_pass_fraction": lit_pass_fraction, "shadow_band_ssim": band_ssim},
            reference,
        )

    # Drift detection against the committed reference renders.
    _assert_matches_golden("pt_reference", pt_rgba)
    _assert_matches_golden("raster_reference", raster_rgba)
