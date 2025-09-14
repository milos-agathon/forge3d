# tests/test_path_tracing_api.py
# Smoke-test for the Workstream A path tracing API skeleton.
# This exists to verify importability and basic behavior of the placeholder API.
# RELEVANT FILES:python/forge3d/path_tracing.py,python/forge3d/__init__.py,docs/user/path_tracing.rst,pytest.ini

import numpy as np
import pytest


def test_import_and_construct():
    import forge3d.path_tracing as pt
    tracer = pt.create_path_tracer(16, 8, max_bounces=2, seed=42)
    assert tracer.size == (16, 8)


def test_render_rgba_shape_and_dtype():
    import forge3d.path_tracing as pt
    tracer = pt.PathTracer(10, 6)
    img = tracer.render_rgba(spp=1)
    assert isinstance(img, np.ndarray)
    assert img.dtype == np.uint8
    assert img.shape == (6, 10, 4)


@pytest.mark.skip(reason="GPU kernels not implemented yet (Workstream A scaffolding)")
def test_path_tracer_gpu_available_or_skipped():
    import forge3d.path_tracing as pt
    tracer = pt.PathTracer(8, 8)
    assert tracer.supports_gpu() is True

