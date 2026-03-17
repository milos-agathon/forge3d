# External Image Import Integration

forge3d currently has an **internal Rust** `external_image` subsystem under
`src/external_image`, but it does **not** expose a public Python module named
`forge3d.external_image`.

## What is public today

- `ViewerHandle.load_overlay()` for draping an image file over terrain
- `forge3d.textures.build_pbr_textures()` for attaching image arrays to PBR
  material containers
- Standard NumPy / Pillow interop for loading image data in Python

## Current Python patterns

### Terrain overlay image

```python
import forge3d as f3d

with f3d.open_viewer_async(terrain_path=f3d.fetch_dem("swiss")) as viewer:
    viewer.load_overlay(
        "landcover",
        f3d.fetch("swiss-land-cover"),
        opacity=0.85,
    )
    viewer.snapshot("swiss-overlay.png")
```

### Normal or albedo texture arrays

```python
import numpy as np
from PIL import Image
from forge3d.textures import build_pbr_textures

normal = np.asarray(Image.open("brick_normal.png").convert("RGBA"), dtype=np.uint8)
albedo = np.asarray(Image.open("brick_albedo.png").convert("RGBA"), dtype=np.uint8)

texset = build_pbr_textures(base_color=albedo, normal=normal)
print(texset)
```

## Status

- There is no public `ImageImportConfig`, `import_image_to_texture()`, or
  `forge3d.RenderError` API in Python today.
- If you need direct access to the Rust `external_image` subsystem, it must be
  bound explicitly first; the current docs should not imply otherwise.
