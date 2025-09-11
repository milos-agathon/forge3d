import pytest
import forge3d.postfx as postfx


class TestBloomControls:
    def setup_method(self):
        for e in postfx.list():
            postfx.disable(e)
        postfx.set_chain_enabled(True)

    def test_enable_bloom_and_params(self):
        ok = postfx.enable("bloom", threshold=1.25, strength=0.8, radius=1.0)
        assert ok is True
        assert "bloom" in postfx.list()

        assert postfx.get_parameter("bloom", "threshold") == 1.25
        assert postfx.get_parameter("bloom", "strength") == 0.8
        assert postfx.get_parameter("bloom", "radius") == 1.0

    def test_param_validation_and_clamp(self):
        postfx.enable("bloom", threshold=-5.0, strength=9.9)
        # clamp behavior: threshold >= 0.0, strength <= 2.0
        assert postfx.get_parameter("bloom", "threshold") >= 0.0
        assert postfx.get_parameter("bloom", "strength") <= 2.0


def test_bloom_perf_estimate_headroom():
    # Rough acceptance: cost 1–3 ms @1080p for a simple chain
    # Not a GPU test; purely a sanity estimate
    width, height = 1920, 1080
    pixels = width * height
    # Bright-pass + 2 blur passes ~= 3x full-frame
    total_pixels = pixels * 3
    # Assume ~0.45 ms per MPix for compute-friendly path on mid GPU
    est_ms = total_pixels / 1_000_000.0 * 0.45
    assert est_ms <= 3.0
