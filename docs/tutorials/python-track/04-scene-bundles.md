# Scene Bundles

> **Pro Feature:** This tutorial uses features that require a
> commercial license. See https://github.com/milos-agathon/forge3d#license. You can read and learn from the code,
> but the highlighted functions will raise `LicenseError` without a valid key.

Bundles package terrain, presets, bookmarks, and the canonical TV16 scene
review registry into a portable directory with checksums.

## Save a bundle

```python
import forge3d as f3d

bookmarks = [
    f3d.CameraBookmark(
        name="overview",
        eye=(0.0, 2.0, 3.0),
        target=(0.0, 0.0, 0.0),
        fov_deg=42.0,
    )
]

bundle_path = f3d.save_bundle(
    "mini-scene.forge3d",
    name="Mini Scene",
    dem_path=f3d.mini_dem_path(),
    colormap_name="terrain",
    domain=(float(f3d.mini_dem().min()), float(f3d.mini_dem().max())),
    camera_bookmarks=bookmarks,
    preset={"sun": {"azimuth_deg": 315, "elevation_deg": 30}},
)
print(bundle_path)
```

## Load and inspect

```python
loaded = f3d.load_bundle(bundle_path)
print(loaded.dem_path)
print(loaded.manifest.camera_bookmarks[0].name)
print(loaded.preset)
```

## Load the same bundle into a running viewer

`ViewerHandle.load_bundle()` loads the terrain, installs the bundle's review
state, and applies the active scene variant automatically:

```python
with f3d.open_viewer_async() as viewer:
    loaded = viewer.load_bundle(bundle_path)
    print(loaded.get_active_variant_id())
    print([variant.id for variant in loaded.list_variants()])
    viewer.snapshot("bundle-loaded.png")
```

Pass `variant_id=` to override the bundle's active variant during load:

```python
with f3d.open_viewer_async() as viewer:
    viewer.load_bundle(bundle_path, variant_id="review")
    viewer.snapshot("bundle-loaded.png")
```

You can query and mutate the installed TV16 state directly from the handle:

```python
with f3d.open_viewer_async() as viewer:
    loaded = viewer.load_bundle(bundle_path)
    viewer.apply_scene_variant("review")
    viewer.set_review_layer_visible("annotations", True)
    print(viewer.list_review_layers())
```

## Expected output

![Expected output for the scene bundle tutorial](../../gallery/images/01-mount-rainier.png)
