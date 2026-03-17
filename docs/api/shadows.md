# Shadows

There is no public `forge3d.shadows` Python module in the current package.
Current shadow controls are exposed through `forge3d.Scene`, terrain parameter
objects, and native-only CSM helpers on the extension module.

## Public scene controls

- `Scene.set_shadow_quality("off" | "low" | "medium" | "high")`
- `Scene.enable_cloud_shadows()`
- `Scene.disable_cloud_shadows()`
- `Scene.is_cloud_shadows_enabled()`
- `Scene.set_cloud_shadow_intensity(...)`
- `Scene.set_cloud_shadow_softness(...)`

## Example

```python
import forge3d as f3d

scene = f3d.Scene(800, 600)
scene.set_shadow_quality("high")
scene.enable_cloud_shadows()
scene.set_cloud_shadow_intensity(0.6)
scene.set_cloud_shadow_softness(0.4)

rgba = scene.render_rgba()
f3d.numpy_to_png("shadows.png", rgba)
```

## Native-only helpers

The compiled extension contains lower-level CSM utilities such as
`configure_csm`, `set_csm_*`, and `get_csm_cascade_info`, but they are not
wrapped as a public Python `forge3d.shadows` package. Access them through
`forge3d._native.get_native_module()` only if you are intentionally working
against internal APIs.
