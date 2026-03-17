# Compressed Textures

Compressed-texture support exists in the Rust codebase, but there is currently
no public Python API named `forge3d.get_compressed_texture_support()` or
`forge3d.load_compressed_texture()`.

## Current status

- Internal Rust pieces include texture-format utilities and KTX2 loader work
  under `src/core/texture_format.rs` and `src/loaders/ktx2`.
- The public Python package currently exposes `forge3d.textures` for texture
  containers, not direct compressed-texture upload helpers.

## What to use from Python today

- `forge3d.textures.build_pbr_textures()` for assembling texture sets from
  NumPy arrays
- `forge3d.textures.load_texture()` for lightweight path-or-array texture
  containers
- Standard image formats (PNG/JPEG/etc.) loaded via Pillow or paths passed into
  higher-level workflows

## Note

If compressed-texture upload becomes a public Python feature later, it should be
documented as a new explicit module or function family rather than implied by
these internal implementation notes.
