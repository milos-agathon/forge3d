# Scene Bundle API

The Scene Bundle (`.forge3d`) format provides portable, reproducible scene packages.

## Overview

A `.forge3d` bundle is a directory containing:

```
my_scene.forge3d/
├── manifest.json        # Version, checksums, metadata
├── terrain/             # DEM files
│   └── dem.tif
├── overlays/            # Optional legacy raw overlay mirrors
│   ├── vectors.geojson
│   └── labels.json
├── camera/              # Camera bookmarks
│   └── bookmarks.json
├── render/              # Render presets
│   └── preset.json
├── scene/
│   └── state.json       # Canonical TV16 scene review state
└── assets/              # Additional resources
    ├── hdri/
    │   └── environment.hdr
    └── overlays/
        └── ortho.png
```

## Python API

### Creating Bundles

```python
from forge3d.bundle import (
    CameraBookmark,
    RasterOverlaySpec,
    ReviewLayer,
    SceneBaseState,
    SceneState,
    SceneVariant,
    save_bundle,
)

# Save a scene bundle
bundle_path = save_bundle(
    "my_scene",
    name="Mountain Terrain",
    dem_path="terrain/mt_fuji.tif",
    scene_state=SceneState(
        base=SceneBaseState(
            preset={"exposure": 1.5, "z_scale": 2.0},
            camera_bookmarks=[
                CameraBookmark(
                    name="summit_view",
                    eye=(1000.0, 2000.0, 1500.0),
                    target=(0.0, 0.0, 0.0),
                    fov_deg=45.0,
                )
            ],
            raster_overlays=[
                RasterOverlaySpec(
                    name="ortho",
                    path="assets/imagery/ortho.png",
                    opacity=0.8,
                )
            ],
        ),
        review_layers=[
            ReviewLayer(id="annotations", labels=[{
                "kind": "point",
                "text": "Summit",
                "world_pos": [0.0, 0.0, 3776.0],
            }]),
        ],
        variants=[
            SceneVariant(id="review", active_layer_ids=["annotations"]),
        ],
        active_variant_id="review",
    ),
)
```

### Loading Bundles

```python
from forge3d.bundle import load_bundle, is_bundle

# Check if path is a valid bundle
if is_bundle("my_scene.forge3d"):
    # Load the bundle
    bundle = load_bundle("my_scene.forge3d", verify_checksums=True)
    
    print(f"Bundle: {bundle.manifest.name}")
    print(f"Version: {bundle.manifest.version}")
    print(f"DEM: {bundle.dem_path}")
    print(f"Preset: {bundle.preset}")
    
    print(bundle.scene_state.active_variant_id)
    print(bundle.effective_scene_state().preset)
    print([layer.id for layer in bundle.list_review_layers()])
```

## Data Structures

### BundleManifest

The manifest contains metadata and checksums for all bundle files.

```python
@dataclass
class BundleManifest:
    version: int               # Schema version (currently 2)
    name: str                  # Human-readable bundle name
    created_at: str            # ISO 8601 timestamp
    description: Optional[str] # Optional description
    checksums: Dict[str, str]  # SHA-256 checksums for files
    terrain: Optional[TerrainMeta]
    camera_bookmarks: List[CameraBookmark]
    preset: Optional[Dict[str, Any]]
```

### TerrainMeta

```python
@dataclass
class TerrainMeta:
    dem_path: str              # Relative path within bundle
    crs: Optional[str]         # Coordinate reference system
    domain: Optional[Tuple[float, float]]  # Elevation [min, max]
    colormap: Optional[str]    # Colormap name or path
```

### CameraBookmark

```python
@dataclass
class CameraBookmark:
    name: str                  # Bookmark name
    eye: Tuple[float, float, float]    # Camera position
    target: Tuple[float, float, float] # Look-at target
    up: Tuple[float, float, float]     # Up vector (default: 0,1,0)
    fov_deg: float             # Field of view in degrees
```

### Scene Review State

`scene/state.json` stores the canonical TV16 registry:

```python
@dataclass
class SceneState:
    base: SceneBaseState
    review_layers: List[ReviewLayer]
    variants: List[SceneVariant]
    active_variant_id: Optional[str]
```

`LoadedBundle` exposes helper methods for variant and layer resolution:

```python
bundle.list_variants()
bundle.list_review_layers()
bundle.get_active_variant_id()
bundle.apply_variant("review")
bundle.set_review_layer_visible("annotations", True)
bundle.effective_scene_state()
```

### LoadedBundle

```python
@dataclass
class LoadedBundle:
    path: Path                 # Bundle directory path
    manifest: BundleManifest   # Parsed manifest
    dem_path: Optional[Path]   # Resolved DEM file path
    overlays: Optional[List[Dict]]  # Loaded overlay data
    labels: Optional[List[Dict]]    # Loaded label data
    preset: Optional[Dict]     # Render preset configuration
    hdr_path: Optional[Path]   # Resolved base HDR file path
    scene_state: SceneState    # Canonical TV16 review registry
```

## IPC Commands

The viewer supports bundle operations via IPC:

### SaveBundle

```python
from forge3d.viewer_ipc import send_ipc

# Save current scene to bundle
response = send_ipc(sock, {
    "cmd": "SaveBundle",
    "path": "output/my_scene.forge3d",
    "name": "My Scene"
})
```

### ViewerHandle.load_bundle

```python
with forge3d.open_viewer_async() as viewer:
    bundle = viewer.load_bundle("scenes/my_scene.forge3d", variant_id="review")
    print(viewer.get_active_scene_variant())
```

### LoadBundle

```python
# Load a bundle into the viewer
response = send_ipc(sock, {
    "cmd": "LoadBundle",
    "path": "scenes/my_scene.forge3d"
})
```

## Checksum Verification

Bundles include SHA-256 checksums for all included files. Loading with `verify_checksums=True` validates file integrity:

```python
try:
    bundle = load_bundle("my_scene.forge3d", verify_checksums=True)
except ValueError as e:
    print(f"Bundle corrupted: {e}")
```

## Version Compatibility

- **Version 2**: Canonical `scene/state.json` bundle registry
- **Version 1**: Legacy bundles still load and synthesize an empty review registry

Forward compatibility: Bundles with higher version numbers are rejected. Backward compatibility: Future versions will support loading older bundles.

## Best Practices

1. **Always include checksums**: Use `save_bundle()` which computes them automatically
2. **Use relative paths**: All paths in manifest are relative to bundle root
3. **Include camera bookmarks**: Helps users reproduce exact viewpoints
4. **Document presets**: Include descriptive names and comments in preset JSON
