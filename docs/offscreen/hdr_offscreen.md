# HDR Off-Screen Pipeline

forge3d does **not** currently ship a public Python `forge3d.hdr` module or a
`create_hdr_offscreen_pipeline(...)` helper.

## Current public offscreen API

- `forge3d.render_offscreen_rgba()`
- `forge3d.save_png_deterministic()`
- `forge3d.rgba_to_png_bytes()`
- `forge3d.helpers.offscreen.save_png_with_exif()`

## Example

```python
import forge3d as f3d
from forge3d.helpers.offscreen import save_png_with_exif

rgba = f3d.render_offscreen_rgba(1024, 768, seed=3, frames=4, denoiser="off")
save_png_with_exif(
    "offscreen.png",
    rgba,
    metadata={"description": "forge3d offscreen render"},
)
```

## Status

- Fine-grained HDR render-target management and tone-mapping pipelines currently
  live in Rust internals, not in a supported Python wrapper.
- For Python users today, the maintained interface is the offscreen helper
  layer plus scene/viewer snapshot paths.
