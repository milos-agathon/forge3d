import math
import random


def triangle_reduction(tiles_full_res, tiles_with_lod, base_tris):
    full = len(tiles_full_res) * base_tris
    lod = 0
    for lod_level in tiles_with_lod:
        # 2^lod per axis => 4^lod fewer triangles
        lod += max(1, base_tris // (4 ** max(0, lod_level)))
    if full == 0:
        return 0.0
    return max(0.0, float(full - lod) / float(full))


def test_lod_triangle_reduction_reasonable():
    # Simulate a 16x16 tile grid at full-res
    tiles_full = [0] * (16 * 16)
    # Assign LODs: closer tiles low LOD index (more detail), far tiles higher
    rng = random.Random(123)
    tiles_lod = []
    for i in range(16 * 16):
        # Bias towards 1–3 with occasional 0 and 4
        lod = min(4, max(0, int(abs(rng.gauss(2.0, 1.0)))))
        tiles_lod.append(lod)

    reduction = triangle_reduction(tiles_full, tiles_lod, base_tris=2048)
    # Expect a healthy reduction in the 50–90% band
    assert 0.5 <= reduction <= 0.9

