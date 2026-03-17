# HDR Off-Screen Rendering and Tone Mapping

forge3d does not currently expose a public `forge3d.hdr` Python module.

## Current public path

- `forge3d.render_offscreen_rgba()`
- `forge3d.save_png_deterministic()`
- `forge3d.rgba_to_png_bytes()`
- `forge3d.helpers.offscreen.save_png_with_exif()`

These helpers are the maintained Python interface for offscreen capture today.
Fine-grained HDR target management remains in the Rust renderer internals.
