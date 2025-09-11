import forge3d as f3d
import forge3d.postfx as postfx


def test_renderer_postfx_toggle():
    r = f3d.Renderer(64, 64)
    assert postfx.is_renderer_postfx_enabled(r) is False
    postfx.set_renderer_postfx_enabled(r, True)
    assert postfx.is_renderer_postfx_enabled(r) is True
    postfx.set_renderer_postfx_enabled(r, False)
    assert postfx.is_renderer_postfx_enabled(r) is False
