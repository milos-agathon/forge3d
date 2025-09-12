# tests/test_tiles_overlay.py
# Tests attribution overlay placement and basic visibility across positions.
# RELEVANT FILES:python/forge3d/tiles/overlay.py,python/forge3d/tiles/client.py,examples/xyz_tile_compose_demo.py

import io

import pytest


@pytest.mark.parametrize("pos", ["tl", "tr", "bl", "br"])  # Basic positions
def test_draw_attribution_positions(pos: str):
    try:
        from PIL import Image  # type: ignore
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Pillow not installed: {e}")

    from forge3d.tiles import draw_attribution

    W, H = 256, 128
    bg = Image.new("RGBA", (W, H), (10, 10, 10, 255))
    out = draw_attribution(bg, text="Data © Provider", logo=None, position=pos, dpi=96, margin=8)
    assert out.size == (W, H)

    # Probe a small corner area to ensure non-background pixels exist
    def count_non_bg(x0: int, y0: int, w: int, h: int) -> int:
        px = out.crop((x0, y0, x0 + w, y0 + h)).getdata()
        return sum(1 for p in px if p != (10, 10, 10, 255))

    pad = 8
    if pos == "tl":
        changed = count_non_bg(0, 0, 64, 32)
    elif pos == "tr":
        changed = count_non_bg(W - 64, 0, 64, 32)
    elif pos == "bl":
        changed = count_non_bg(0, H - 32, 64, 32)
    else:
        changed = count_non_bg(W - 64, H - 32, 64, 32)
    assert changed > 0

