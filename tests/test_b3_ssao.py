# tests/test_b3_ssao.py
# SSAO integration tests for Workstream B.
# Exists to validate SSAO occlusion strength, parameter plumbing, and 1080p performance marker.
# RELEVANT FILES:python/forge3d/__init__.py,python/forge3d/postfx.py,shaders/ssao.wgsl,examples/ssao_demo.py

from __future__ import annotations

import time
import numpy as np
import pytest

from forge3d import Scene
from forge3d import postfx


def _mean_luma(image: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> float:
    region = image[y0:y1, x0:x1, :3].astype(np.float32)
    return float(region.mean())


@pytest.mark.opbr
@pytest.mark.olighting
def test_ssao_darkens_basin() -> None:
    size = 96
    scene = Scene(size, size)
    ys, xs = np.meshgrid(
        np.linspace(-1.0, 1.0, size, dtype=np.float32),
        np.linspace(-1.0, 1.0, size, dtype=np.float32),
        indexing="ij",
    )
    basin = np.sqrt(xs * xs + ys * ys).astype(np.float32)
    scene.set_height_from_r32f(basin)

    baseline = scene.render_rgba()
    baseline_center = _mean_luma(baseline, 40, 56, 40, 56)

    scene.set_ssao_parameters(radius=2.5, intensity=1.4, bias=0.02)
    scene.set_ssao_enabled(True)
    occluded = scene.render_rgba()
    occluded_center = _mean_luma(occluded, 40, 56, 40, 56)

    assert occluded_center < baseline_center * 0.85

    scene.set_ssao_enabled(False)
    reset = scene.render_rgba()
    reset_center = _mean_luma(reset, 40, 56, 40, 56)
    assert np.isclose(reset_center, baseline_center, rtol=0.05)


@pytest.mark.opbr
def test_ssao_parameter_roundtrip_and_postfx_link() -> None:
    scene = Scene(64, 64)
    postfx.enable_ssao(scene=scene, radius=1.5, intensity=0.75, bias=0.03)
    assert scene.ssao_enabled()
    params = scene.get_ssao_parameters()
    assert params == pytest.approx((1.5, 0.75, 0.03), rel=1e-3)
    assert 'ssao' in postfx.list_enabled_effects()

    scene.set_ssao_parameters(radius=2.0, intensity=0.5, bias=0.01)
    assert scene.get_ssao_parameters() == pytest.approx((2.0, 0.5, 0.01), rel=1e-3)

    postfx.disable_ssao(scene=scene)
    assert not scene.ssao_enabled()


@pytest.mark.olighting
def test_ssao_radius_influences_occlusion_strength() -> None:
    size = 96
    scene = Scene(size, size)
    ys, xs = np.meshgrid(
        np.linspace(-1.0, 1.0, size, dtype=np.float32),
        np.linspace(-1.0, 1.0, size, dtype=np.float32),
        indexing="ij",
    )
    ridge = (xs * xs - ys * ys).astype(np.float32)
    scene.set_height_from_r32f(ridge)

    baseline = scene.render_rgba()

    scene.set_ssao_parameters(radius=1.0, intensity=1.2, bias=0.015)
    scene.set_ssao_enabled(True)
    small_radius = scene.render_rgba()

    scene.set_ssao_parameters(radius=3.0, intensity=1.2, bias=0.015)
    large_radius = scene.render_rgba()
    scene.set_ssao_enabled(False)

    baseline_mean = _mean_luma(baseline, 30, 66, 30, 66)
    small_mean = _mean_luma(small_radius, 30, 66, 30, 66)
    large_mean = _mean_luma(large_radius, 30, 66, 30, 66)

    assert small_mean < baseline_mean
    assert large_mean < small_mean * 0.97


@pytest.mark.opbr
@pytest.mark.olighting
def test_ssao_1080p_performance_marker() -> None:
    scene = Scene(1920, 1080)

    scene.set_ssao_enabled(False)
    start = time.perf_counter()
    baseline_image = scene.render_rgba()
    baseline_elapsed = time.perf_counter() - start

    scene.set_ssao_parameters(radius=1.5, intensity=1.0, bias=0.02)
    scene.set_ssao_enabled(True)
    start = time.perf_counter()
    image = scene.render_rgba()
    occluded_elapsed = time.perf_counter() - start

    assert image.shape == (1080, 1920, 4)
    assert (occluded_elapsed - baseline_elapsed) < 0.75

    baseline_mean = _mean_luma(baseline_image, 400, 680, 880, 1040)
    occluded_mean = _mean_luma(image, 400, 680, 880, 1040)
    assert occluded_mean <= baseline_mean

    scene.set_ssao_enabled(False)
