# Virtual Texture Streaming

Virtual texturing is not exposed as a public Python module in the current
package. In particular, there is no supported `forge3d.streaming` API today.

## Current status

- The concept exists as renderer-side design and implementation work.
- Python workflows currently rely on concrete files, datasets, and viewer/scene
  APIs rather than a standalone virtual-texture manager.

## Recommended Python path

Use:

- `forge3d.datasets` for sample asset resolution
- `forge3d.cog` for COG-based terrain access where enabled
- viewer overlays / bundle workflows for higher-level scene composition
