# T32-BEGIN:example
import numpy as np
from PIL import Image
import forge3d as f3d

r = f3d.Renderer(256, 256)
yy, xx = np.mgrid[0:256, 0:256]
cx, cy = 127.5, 127.5
r2 = ((xx - cx)**2 + (yy - cy)**2) / (256.0**2)
dem = (80.0 * np.exp(-r2 / (2*0.15**2))).astype(np.float32)
r.add_terrain(dem, (25.0, 25.0), exaggeration=1.0, colormap="viridis")

for az in (90.0, 270.0):
    r.set_sun(elevation_deg=45.0, azimuth_deg=az)
    arr = r.render_rgba()
    Image.fromarray(arr, "RGBA").save(f"flip_az{int(az)}.png")
print("Wrote flip_az90.png / flip_az270.png")
# T32-END:example