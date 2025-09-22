# tests/test_b1_msaa.py
# Workstream B1 MSAA regression tests.
# Exists to validate MSAA toggles and edge-metric improvements.
# RELEVANT FILES:python/forge3d/__init__.py,python/forge3d/viewer.py,tests/edge_consistency.py,shaders/tone_map.wgsl

from __future__ import annotations

import numpy as np
import pytest

from forge3d import Renderer
from forge3d.viewer import set_msaa


def _edge_metric(image: np.ndarray) -> float:
    gray = image[..., :3].astype(np.float32).mean(axis=2) / 255.0
    diff_x = np.abs(np.diff(gray, axis=1))
    diff_y = np.abs(np.diff(gray, axis=0))
    threshold = 0.2
    return float(np.count_nonzero(diff_x > threshold) + np.count_nonzero(diff_y > threshold))


@pytest.mark.opbr
@pytest.mark.olighting
def test_msaa_reduces_edge_metric() -> None:
    set_msaa(1)
    renderer = Renderer(128, 128)
    base_image = renderer.render_triangle_rgba()
    baseline = _edge_metric(base_image)

    set_msaa(4)
    renderer.set_msaa_samples(4)
    msaa_image = renderer.render_triangle_rgba()
    improved = _edge_metric(msaa_image)

    assert improved <= baseline * 0.8
    assert np.all(msaa_image[..., 3] == 255)

    renderer.set_msaa_samples(1)
    set_msaa(1)


@pytest.mark.olighting
def test_msaa_api_validation() -> None:
    renderer = Renderer(64, 64)
    assert renderer.set_msaa_samples(2) == 2
    with pytest.raises(ValueError):
        renderer.set_msaa_samples(3)

    set_msaa(1)
    with pytest.raises(ValueError):
        set_msaa(5)
