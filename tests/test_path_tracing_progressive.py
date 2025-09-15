# tests/test_path_tracing_progressive.py
# Progressive tiling and callback cadence tests for A15.
# This exists to validate tile scheduler correctness and checkpoint update cadence.
# RELEVANT FILES:python/forge3d/path_tracing.py,python/forge3d/path_tracing.pyi,docs/api/path_tracing.md

import numpy as np


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return float(np.sqrt(np.mean((a - b) ** 2))) / 255.0


def test_progressive_matches_fullframe():
    import forge3d.path_tracing as pt

    w, h = 64, 48
    t = pt.PathTracer(w, h, seed=3, tile=8)
    full = t.render_rgba(spp=1)

    prog = t.render_progressive(tile_size=8, spp=1)

    assert prog.shape == full.shape == (h, w, 4)
    assert prog.dtype == full.dtype == np.uint8
    assert _rmse(prog, full) <= 5e-3  # 0.5% RMSE bound; here should be 0


def test_progressive_callback_cadence_with_fake_clock():
    import forge3d.path_tracing as pt

    w, h = 64, 48
    t = pt.PathTracer(w, h, seed=7, tile=8)

    class FakeClock:
        def __init__(self):
            self.t = 0.0
        def now(self):
            return self.t
        def advance(self, dt: float):
            self.t += float(dt)

    clock = FakeClock()
    calls: list[float] = []

    def cb(info):
        calls.append(info["timestamp"])
        # Advance virtual time by 0.25s per callback opportunity
        clock.advance(0.25)
        # Stop after ~1.0s of virtual time to bound the test
        if clock.now() >= 1.0:
            return True
        return False

    _ = t.render_progressive(tile_size=8, min_updates_per_sec=2.0, time_source=clock.now, callback=cb)

    # In ~1.0s window with min 2 Hz cadence, expect at least 2 callbacks
    assert len(calls) >= 2
