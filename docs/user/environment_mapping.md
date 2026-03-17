# Environment Mapping and Image-Based Lighting (IBL)

There is no public `forge3d.envmap` Python module in the current package.

## Current public surface

IBL and environment-map controls live primarily on `forge3d.Scene`:

- `enable_ibl()` / `disable_ibl()` / `is_ibl_enabled()`
- `load_environment_map(hdr_data, width, height)`
- `generate_ibl_textures()`
- `set_ibl_quality(...)`
- `get_ibl_texture_info()`

## Status

- Environment-map preprocessing exists in the renderer and native layer.
- The old `forge3d.envmap.*` examples are outdated.
- Higher-level terrain/viewer workflows often manage lighting through scene
  settings or terrain parameter objects instead of a dedicated envmap module.
