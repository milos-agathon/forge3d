# Matplotlib Integration

Matplotlib interop in forge3d is straightforward because the public Python
surface already works with NumPy arrays and PNG bytes.

There is no current `forge3d.adapters.mpl_cmap` or
`forge3d.helpers.mpl_display` module in this package.

## Current workflow

- Render to an RGBA NumPy array with `forge3d.render_offscreen_rgba()`
- Or load/sample terrain data with `forge3d.mini_dem()` / dataset helpers
- Display the result with normal `matplotlib.pyplot.imshow(...)`

## Example

```python
import matplotlib.pyplot as plt
import forge3d as f3d

rgba = f3d.render_offscreen_rgba(800, 600, seed=7)

fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(rgba)
ax.set_title("forge3d offscreen render")
ax.axis("off")
plt.show()
```

## Colormaps

The public colormap helpers are:

- `forge3d.available_colormaps()`
- `forge3d.get_colormap(name)`

If you need Matplotlib-specific normalization or artist helpers, use
Matplotlib directly around the NumPy data returned by forge3d.
