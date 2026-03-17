# Render Bundles

There is no public `forge3d.bundles` Python module in the current package.

## What "bundle" means in forge3d today

The public Python API uses **scene bundles**, not GPU render bundles:

- `forge3d.bundle.save_bundle()`
- `forge3d.bundle.load_bundle()`
- `forge3d.bundle.is_bundle()`

These helpers write and read portable `.forge3d` directories containing
terrain, overlays, presets, and camera bookmarks.

## Internal status

The underlying Rust renderer does use command-recording and batching concepts,
but those internals are not exposed as a stable Python API. Older examples that
referenced `forge3d.bundles.BundleManager` or similar classes are outdated.

## Recommended path

If you need portable scene packaging from Python, use `forge3d.bundle`. If you
need lower-level GPU command optimization, work in the Rust renderer directly.
