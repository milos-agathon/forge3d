# A1.6-BEGIN:pytest-determinism
import os, hashlib
import numpy as np
import pytest

from forge3d import Renderer

@pytest.mark.timeout(20)
def test_repeatable_rgba_bytes_small():
    w, h, runs = 64, 64, 3
    shas = []
    for _ in range(runs):
        r = Renderer(w, h)
        a = r.render_triangle_rgba()
        assert a.shape == (h, w, 4) and a.dtype == np.uint8
        shas.append(hashlib.sha256(a.tobytes()).hexdigest())
    # All runs must match exactly
    assert len(set(shas)) == 1

# Optional: basic PNG write path (doesn't assert PNG bytes)
@pytest.mark.timeout(20)
def test_png_path_smoke(tmp_path):
    r = Renderer(64, 64)
    out = tmp_path / "triangle.png"
    r.render_triangle_png(str(out))
    assert out.exists() and out.stat().st_size > 0
# A1.6-END:pytest-determinism