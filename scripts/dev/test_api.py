import numpy as np

def test_public_api_imports():
    import vulkan_forge as vf
    from vulkan_forge import Renderer, render_triangle_rgba, render_triangle_png
    assert hasattr(vf, "__version__")

    r = Renderer(16, 16)
    a = render_triangle_rgba(16, 16)
    assert a.shape == (16, 16, 4) and a.dtype == np.uint8

def test_vshade_reexport():
    from vshade import Renderer as R2
    from vulkan_forge import Renderer as R1
    assert R1 is R2
