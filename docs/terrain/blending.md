# TV21: Terrain Scatter Blend and Contact Shading

TV21 extends the terrain scatter workflow with terrain-aware mesh grounding controls. The goal is to reduce the hard "floating prop" seam where instanced meshes meet the terrain and to add localized darkening where the terrain should visually press into the mesh base.

## What shipped

- New per-batch Python settings on `forge3d.terrain_scatter.TerrainScatterBatch`:
  - `terrain_blend=TerrainMeshBlendSettings(...)`
  - `terrain_contact=TerrainContactSettings(...)`
- Shared offscreen and viewer support through the existing terrain scatter contract.
- Terrain-aware shader logic in the instanced mesh path:
  - seam alpha fade based on sampled terrain height under each fragment
  - localized contact darkening near the terrain intersection band
- Backward-compatible disabled path:
  - leaving both settings at defaults preserves the pre-TV21 image
  - explicit disabled settings are pixel-identical to the baseline path
- Regression coverage for three representative cases:
  - rock cluster
  - road edge
  - building foundation

## Public API

```python
import forge3d as f3d
from forge3d import terrain_scatter as ts

batch = ts.TerrainScatterBatch(
    name="foundation",
    color=(0.70, 0.71, 0.69, 1.0),
    transforms=transforms,
    terrain_blend=ts.TerrainMeshBlendSettings(
        enabled=True,
        bury_depth=1.4,
        fade_distance=3.4,
    ),
    terrain_contact=ts.TerrainContactSettings(
        enabled=True,
        distance=3.0,
        strength=0.28,
        vertical_weight=0.85,
    ),
    levels=[ts.TerrainScatterLevel(mesh=f3d.geometry.primitive_mesh("box"))],
)

ts.apply_to_renderer(renderer, [batch])
```

## Parameter semantics

- `terrain_blend.enabled`
  - Enables seam fading against the terrain heightfield.
- `terrain_blend.bury_depth`
  - How far the mesh can sink into the terrain before the blend is fully opaque again.
  - Use this to hide harsh mesh-ground cut lines on buried bases.
- `terrain_blend.fade_distance`
  - How far above the terrain the fade continues before the mesh returns to full opacity.
  - Larger values soften transitions over sloped terrain.
- `terrain_contact.enabled`
  - Enables terrain-proximity darkening near the mesh base.
- `terrain_contact.distance`
  - World-space distance band used for the contact effect.
- `terrain_contact.strength`
  - Darkening strength in `[0, 1]`.
- `terrain_contact.vertical_weight`
  - Bias toward flatter upward-facing surfaces.
  - Higher values keep the contact effect concentrated near horizontal bases instead of climbing steep side walls.

## Workflow notes

- Both settings serialize through:
  - `TerrainScatterBatch.to_native_dict()`
  - `TerrainScatterBatch.to_viewer_payload()`
- The viewer IPC payload carries the same `terrain_blend` and `terrain_contact` dictionaries as the offscreen renderer path.
- Validation is enforced in Python and Rust:
  - `bury_depth >= 0`
  - `fade_distance > 0`
  - `distance > 0`
  - `strength in [0, 1]`
  - `vertical_weight in [0, 1]`

## Example and tests

- Real-DEM example: `python examples/terrain_tv21_blending_demo.py`
- Example notes: `docs/examples/blending_demo.md`
- Feature regression tests: `tests/test_terrain_tv21_blending.py`
- Real-DEM example smoke test: `tests/test_terrain_tv21_demo.py`

The real-DEM example writes per-case baseline, TV21, and diff images plus a contact sheet and JSON summary so the visual change is inspectable without opening the viewer. Each crop preserves the DEM shape but normalizes local relief into a close-up-friendly range so the grounding controls remain legible at example scale.
