from __future__ import annotations
import numpy as np
from forge3d import Renderer

H, W = 256, 256
x = np.linspace(-3, 3, W, dtype=np.float32)
y = np.linspace(-3, 3, H, dtype=np.float32)
X, Y = np.meshgrid(x, y)
Z = 0.25*np.sin(1.3*X) + 0.25*np.cos(1.1*Y)
Z = np.ascontiguousarray(Z, dtype=np.float32)

r = Renderer(800, 600)
r.add_terrain(Z, spacing=(1.0, 1.0), exaggeration=1.0, colormap="viridis")
# Reuse existing off-screen writer for now
r.render_triangle_png("terrain_overlay.png")
print("Wrote terrain_overlay.png")
