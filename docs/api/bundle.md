# Scene Bundle API

The Scene Bundle (`.forge3d`) format provides portable, reproducible scene packages.

## Overview

A `.forge3d` bundle is a directory containing:

```
my_scene.forge3d/
├── manifest.json        # Version, checksums, metadata
├── terrain/             # DEM files
│   └── dem.tif
├── overlays/            # Vector data and labels
│   ├── vectors.geojson
│   └── labels.json
├── camera/              # Camera bookmarks
│   └── bookmarks.json
├── render/              # Render presets
│   └── preset.json
└── assets/              # Additional resources
    └── hdri/
        └── environment.hdr
```

## Python API

### Creating Bundles

```python
from forge3d.bundle import save_bundle, CameraBookmark

# Save a scene bundle
bundle_path = save_bundle(
    "my_scene",
    name="Mountain Terrain",
    dem_path="terrain/mt_fuji.tif",
    colormap_name="terrain",
    domain=(0.0, 3776.0),
    crs="EPSG:32654",
    preset={
        "exposure": 1.5,
        "z_scale": 2.0,
        "colormap_strength": 0.8,
    },
    camera_bookmarks=[
        CameraBookmark(
            name="summit_view",
            eye=(1000.0, 2000.0, 1500.0),
            target=(0.0, 0.0, 0.0),
            fov_deg=45.0,
        )
    ],
    hdr_path="assets/hdri/evening_sky.hdr",
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
    
    # Access camera bookmarks
    for bookmark in bundle.manifest.camera_bookmarks:
        print(f"  Camera: {bookmark.name} at {bookmark.eye}")
```

## Data Structures

### BundleManifest

The manifest contains metadata and checksums for all bundle files.

```python
@dataclass
class BundleManifest:
    version: int               # Schema version (currently 1)
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
    hdr_path: Optional[Path]   # Resolved HDR file path
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

- **Version 1**: Current schema (introduced in v1.11.0)

Forward compatibility: Bundles with higher version numbers are rejected. Backward compatibility: Future versions will support loading older bundles.

## Best Practices

1. **Always include checksums**: Use `save_bundle()` which computes them automatically
2. **Use relative paths**: All paths in manifest are relative to bundle root
3. **Include camera bookmarks**: Helps users reproduce exact viewpoints
4. **Document presets**: Include descriptive names and comments in preset JSON
