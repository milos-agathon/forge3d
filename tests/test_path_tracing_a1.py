# tests/test_path_tracing_a1.py
# A1 validation: RNG determinism, HDR accumulation, tile scheduler coverage, basic shading.
# This exists to verify initial A1 features behave deterministically and efficiently.
# RELEVANT FILES:python/forge3d/path_tracing.py,docs/user/path_tracing.rst,tests/test_path_tracing_api.py

import numpy as np


def test_rng_determinism_and_repeatability():
    import forge3d.path_tracing as pt
    t1 = pt.PathTracer(16, 12, seed=7)
    t2 = pt.PathTracer(16, 12, seed=7)
    # add identical scene content
    t1.add_sphere((0.0, 0.0, 0.0), 0.5, (1.0, 0.2, 0.1))
    t2.add_sphere((0.0, 0.0, 0.0), 0.5, (1.0, 0.2, 0.1))
    a = t1.render_rgba(spp=2)
    b = t2.render_rgba(spp=2)
    assert a.shape == b.shape == (12, 16, 4)
    assert a.dtype == b.dtype == np.uint8
    assert np.array_equal(a, b)


def test_hdr_accumulation_and_tonemap_properties():
    import forge3d.path_tracing as pt
    tracer = pt.PathTracer(8, 8, seed=1)
    tracer.add_sphere((0.0, 0.0, 0.0), 0.6, {"type": "lambert", "base_color": (2.0, 2.0, 2.0)})  # bright to force tonemap
    img = tracer.render_rgba(spp=3)
    assert img.shape == (8, 8, 4)
    # alpha opaque
    assert np.all(img[..., 3] == 255)
    # ldr bounded
    assert img[..., :3].max() <= 255 and img[..., :3].min() >= 0


def test_tile_scheduler_covers_full_frame():
    import forge3d.path_tracing as pt
    tracer = pt.PathTracer(17, 19, seed=0, tile=7)
    tracer.add_sphere((0.0, 0.0, 0.0), 0.01, (1.0, 1.0, 1.0))
    img = tracer.render_rgba(spp=1)
    assert img.shape == (19, 17, 4)
    # Ensure every pixel visited: sum(alpha) equals 255 * W * H
    assert int(img[..., 3].sum()) == 255 * 17 * 19
