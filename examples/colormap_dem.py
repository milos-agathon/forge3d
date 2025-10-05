import numpy as np
from forge3d.colormaps import get

# Pseudocode for your existing render pipeline:

# dem = load_dem(...)

cm = get("forge3d:viridis")  # or "cmcrameri:batlow" if extras installed

# scene = forge3d.Scene()

# scene.set_colormap(cm, vmin=dem.min(), vmax=dem.max())

# scene.render_heightmap(dem)
