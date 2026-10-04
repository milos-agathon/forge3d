"""Step 2 of 3: prepare the render inputs in ./data from fetch_data.py's downloads.

  context_dem_2m_smooth.tif  Gaussian blur (sigma 2 px = 4 m), so 1 m spires are not
                             stretched into needles by the viewer's coarser vertex grid
  naip_*_graded.png          saturation x1.3 and a gentle S-curve, offsetting the
                             renderer's desaturating sun/sky tint and tone mapping
"""

from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

DATA = Path(__file__).resolve().parent / "data"


def gaussian(a: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur with edge padding and a 3-sigma kernel."""
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    for axis in (1, 0):
        p = np.pad(a, [(r, r) if ax == axis else (0, 0) for ax in (0, 1)], mode="edge")
        a = np.zeros(a.shape)
        for i, w in enumerate(k):
            a += w * p[(slice(None),) * axis + (slice(i, i + a.shape[axis]),)]
    return a


def smooth_dem() -> None:
    with rasterio.open(DATA / "context_dem_2m.tif") as src:
        profile, h = src.profile, src.read(1).astype(np.float64)
    with rasterio.open(DATA / "context_dem_2m_smooth.tif", "w", **profile) as dst:
        dst.write(gaussian(h, 2.0).astype(np.float32), 1)
    print("wrote context_dem_2m_smooth.tif")


def grade(name: str, saturation: float = 1.3) -> None:
    rgb = np.asarray(Image.open(DATA / f"{name}.png").convert("RGB"), dtype=np.float64) / 255.0
    lum = (rgb @ np.array([0.2126, 0.7152, 0.0722]))[..., None]
    rgb = np.clip(lum + (rgb - lum) * saturation, 0.0, 1.0)
    rgb += 0.12 * (rgb - 0.5) * (1.0 - np.abs(2.0 * rgb - 1.0))
    Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).save(DATA / f"{name}_graded.png")
    print(f"wrote {name}_graded.png")


if __name__ == "__main__":
    smooth_dem()
    grade("naip_tile_1m")
    grade("naip_context_4m")
