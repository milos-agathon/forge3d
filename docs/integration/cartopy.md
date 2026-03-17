# Cartopy Integration

There is no current `forge3d.tiles` Python package or bundled
`examples/cartopy_overlay.py` example in this repository. The practical
integration point today is NumPy image interop: render with forge3d, then place
 the resulting RGBA image inside a Cartopy axes.

## Current workflow

1. Render or snapshot a forge3d scene to an RGBA array or PNG.
2. Use Cartopy for projection-aware axes, coastlines, and reference data.
3. Add the forge3d image with `ax.imshow(...)` using the extent you want.

## Example

```python
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import forge3d as f3d

rgba = f3d.render_offscreen_rgba(800, 600, seed=42)

fig = plt.figure(figsize=(8, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.imshow(
    rgba,
    extent=(6.0, 7.0, 45.0, 46.0),
    transform=ccrs.PlateCarree(),
    origin="upper",
)
ax.coastlines(resolution="10m")
plt.show()
```

## Status

- Cartopy interop is image-based today.
- Projection-aware tile fetching and basemap composition are not wrapped by a
  dedicated forge3d module.
