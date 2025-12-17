# Interactive Viewer

> **Status:** ✅ Implemented
>
> The forge3d interactive viewer provides a windowed application for exploring 3D scenes and terrain in real-time.

## Overview

The interactive viewer supports both general 3D scene viewing and terrain DEM visualization with orbit camera controls, real-time parameter adjustments, and snapshot capture.

## Quick Start

### Interactive Terrain Viewer

```bash
# Build the viewer binary
cargo build --release --bin interactive_viewer

# Launch with a DEM file (4K window by default)
python examples/terrain_viewer_interactive.py --dem assets/dem_rainier.tif

# Enable PBR mode with sun controls
python examples/terrain_viewer_interactive.py --dem assets/dem_rainier.tif \
    --pbr --sun-azimuth 135 --sun-elevation 45 --sun-intensity 1.0

# Full PBR with HDR environment map
python examples/terrain_viewer_interactive.py --dem assets/dem_rainier.tif \
    --pbr --hdr assets/snow_field_4k.hdr --exposure 1.0 --msaa 8
```

### Automatic Snapshot Mode

```bash
# Render terrain to PNG and exit
python examples/terrain_viewer_interactive.py --dem path/to/dem.tif --snapshot output.png

# High-resolution snapshot (up to 16K supported)
python examples/terrain_viewer_interactive.py --dem path/to/dem.tif \
    --snapshot output.png --width 8192 --height 8192
```

## Window Controls

| Input | Action |
|-------|--------|
| **Mouse drag** | Orbit camera around terrain |
| **Scroll wheel** | Zoom in/out |
| **W/S** or **↑/↓** | Tilt camera up/down |
| **A/D** or **←/→** | Rotate camera left/right |
| **Q/E** | Zoom out/in |
| **Shift** | Faster movement |
| **Escape** | Close viewer |

## Terminal Commands

While the viewer window is open, you can type commands in the terminal to adjust parameters in real-time (similar to rayshader's `render_camera`):

### Setting Multiple Parameters

```bash
# Camera controls
set phi=45 theta=60 radius=2000 fov=55

# Lighting
set sun_az=135 sun_el=45 intensity=1.5 ambient=0.3

# Terrain rendering
set zscale=2.0 shadow=0.5

# Background color (RGB 0-1)
set background=0.2,0.3,0.5

# Water (level in elevation units + color)
set water=1500 water_color=0.1,0.3,0.5

# Combine any parameters in one command
set phi=90 theta=30 zscale=1.5 sun_el=60 ambient=0.4
```

### PBR Mode Commands

```bash
pbr on                        # Enable PBR rendering
pbr off                       # Return to legacy mode
pbr exposure=2.0              # Adjust exposure
pbr shadows=pcss ibl=1.5      # Shadow technique + IBL intensity
```

### Snapshots (up to 16K)

```bash
snap output.png               # Snapshot at window size
snap output.png 3840x2160     # Snapshot at 4K
snap output.png 7680x4320     # Snapshot at 8K
snap output.png 16384x16384   # Snapshot at 16K (max)
```

### Other Commands

```bash
params          # Show current parameters
quit            # Close viewer
```

## Available Parameters

| Parameter | Description | Range/Default |
|-----------|-------------|---------------|
| `phi` | Camera azimuth | degrees (default: 135°) |
| `theta` | Camera elevation | 5-85° (default: 45°) |
| `radius` | Camera distance | 100-50000 (auto from DEM) |
| `fov` | Field of view | 10-120° (default: 55°) |
| `sun_az` | Sun azimuth | degrees (default: 135°) |
| `sun_el` | Sun elevation | -90 to 90° (default: 35°) |
| `intensity` | Sun intensity | ≥0 (default: 1.0) |
| `exposure` | PBR exposure | 0.1-5.0 (default: 1.0) |
| `ibl` | IBL intensity | ≥0 (default: 1.0) |
| `shadows` | Shadow technique | none/hard/pcf/pcss |
| `ambient` | Ambient light | 0-1 (default: 0.3) |
| `zscale` | Vertical exaggeration | ≥0.01 (default: 0.3) |
| `shadow` | Shadow intensity | 0-1 (default: 0.5) |
| `background` | Sky color RGB | 0-1,0-1,0-1 |
| `water` | Water level | elevation units |
| `water_color` | Water RGB | 0-1,0-1,0-1 |

## IPC Protocol

The viewer communicates via JSON over TCP. You can control it programmatically:

```python
import json
import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(("127.0.0.1", port))

# Send command
cmd = {"cmd": "set_terrain", "phi": 45, "theta": 60, "zscale": 1.5}
sock.sendall((json.dumps(cmd) + "\n").encode())

# Receive response
response = json.loads(sock.recv(4096).decode())
```

### Available IPC Commands

- `load_terrain` — Load DEM file: `{"cmd": "load_terrain", "path": "/path/to/dem.tif"}`
- `set_terrain` — Set parameters: `{"cmd": "set_terrain", "phi": 45, "zscale": 2.0, ...}`
- `set_terrain_camera` — Camera only: `{"cmd": "set_terrain_camera", "phi_deg": 45, ...}`
- `set_terrain_sun` — Sun only: `{"cmd": "set_terrain_sun", "azimuth_deg": 135, ...}`
- `get_terrain_params` — Get current params: `{"cmd": "get_terrain_params"}`
- `snapshot` — Capture frame: `{"cmd": "snapshot", "path": "/path/to/output.png"}`
- `close` — Close viewer: `{"cmd": "close"}`

## Platform Support

- ✅ **macOS** — Metal backend
- ✅ **Linux** — Vulkan backend (X11 and Wayland)
- ✅ **Windows** — DX12/Vulkan backend

## Implementation Details

The terrain viewer uses a standalone WGSL shader with:
- **4K default window** (3840×2160)
- **Snapshots up to 16K** (16384×16384, 270 megapixels max)
- Height-based terrain colormap (green valleys → brown slopes → white peaks)
- Real-time normal calculation via screen-space derivatives
- Configurable z-scale for vertical exaggeration
- Diffuse lighting with shadow intensity control
- Optional water plane with specular highlights

**PBR Mode** adds:
- Blinn-Phong specular + diffuse lighting
- Real-time CSM shadows (hard/pcf/pcss)
- ACES filmic tonemapping
- Configurable sun azimuth, elevation, intensity
- HDR environment map support for IBL

### Source Files

- `src/viewer/viewer_terrain.rs` — Terrain viewer implementation
- `src/viewer/mod.rs` — Main viewer with IPC handling
- `src/viewer/ipc.rs` — IPC protocol definitions
- `examples/terrain_viewer_interactive.py` — Python launcher script

## See Also

- [Terrain Rendering](../terrain_rendering.rst) — Offscreen terrain rendering API
- [Terrain Demo Quickstart](../examples/terrain_demo_quickstart.rst) — Programmatic terrain rendering
