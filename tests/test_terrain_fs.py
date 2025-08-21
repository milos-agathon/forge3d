# T32-BEGIN:tests
import os, numpy as np, pytest

SKIP = os.environ.get("VF_ENABLE_TERRAIN_TESTS", "0") != "1"
pytestmark = pytest.mark.skipif(SKIP, reason="Enable with VF_ENABLE_TERRAIN_TESTS=1 after T3.3 wiring.")

def _luminance(img):
    rgb = img[..., :3].astype(np.float32) / 255.0
    return (0.2126*rgb[...,0] + 0.7152*rgb[...,1] + 0.0722*rgb[...,2])

def _bump(h, w, amp=1.0, sigma=0.18):
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = (w-1)/2.0, (h-1)/2.0
    r2 = ((xx - cx)**2 + (yy - cy)**2) / (max(h,w)**2)
    return (amp * np.exp(-r2 / (2*sigma**2))).astype(np.float32)

def _render_with_sun(azimuth_deg):
    import forge3d as f3d
    R = f3d.Renderer(128, 128)
    dem = _bump(128, 128, amp=50.0)
    R.add_terrain(dem, (30.0, 30.0), exaggeration=1.0, colormap="viridis")
    R.set_sun(elevation_deg=45.0, azimuth_deg=azimuth_deg)  # unified kw names
    return R.render_rgba()

def test_east_west_flip():
    east = _render_with_sun(90.0)
    west = _render_with_sun(270.0)
    assert np.isfinite(east).all() and np.isfinite(west).all()  # basic sanity

    Y_e = _luminance(east)
    Y_w = _luminance(west)

    y  = Y_e.shape[0] // 2
    cx = Y_e.shape[1] // 2
    dx = 16

    east_right = Y_e[y, cx+dx]
    east_left  = Y_e[y, cx-dx]
    west_right = Y_w[y, cx+dx]
    west_left  = Y_w[y, cx-dx]

    assert east_right > east_left, "East sun should light east slope more"
    assert west_left  > west_right, "West sun should light west slope more"
# T32-END:tests