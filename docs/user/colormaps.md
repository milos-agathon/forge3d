# Colormaps in forge3d

## Quick start
```python
from forge3d.colormaps import get, available
cm = get("forge3d:viridis")
```

## Optional providers

Install extras: `pip install "forge3d[colormaps]"`

Use:

```python
get("cmcrameri:batlow")
get("cmocean:deep")
get("colorcet:glasbey")
from forge3d.colormaps import load_cpt
cm = load_cpt("imhof_1.cpt")
```

## GPU details

* LUT baked to linear sRGB, uploaded as 2D texture (N×1), sampled in WGSL.
* Set vmin/vmax and colormap during render pass.
