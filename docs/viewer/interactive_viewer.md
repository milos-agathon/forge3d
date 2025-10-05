# Interactive Viewer

> **Status:** Planned for Workstream I, Task I1
>
> **Implementation:** Not yet available
>
> This page documents the planned interactive windowed viewer for real-time scene exploration.

## Overview

The forge3d interactive viewer will provide a windowed application for exploring 3D scenes in real-time with camera controls and visualization settings.

## Planned Features

### Camera Controls

#### Orbit Camera Mode
- **Mouse drag** — Rotate camera around target
- **Mouse scroll** — Zoom in/out
- **Shift + drag** — Pan camera laterally

#### FPS Camera Mode
- **WASD** — Move camera forward/left/backward/right
- **Mouse look** — Rotate view direction
- **Q/E** — Move up/down
- **Shift** — Increase movement speed

### Performance

- **Target:** 60 FPS at 1080p on mid-tier GPU (GTX 1060 / M1)
- **V-Sync:** Configurable
- **DPI Scaling:** Automatic high-DPI display support

### Windowing

- **Windowed mode:** Resizable window with title bar
- **Fullscreen mode:** (planned)
- **Multi-monitor:** (planned)

## Planned API

### Rust API (planned)

```rust
use forge3d::viewer::{Viewer, CameraMode};

fn main() {
    let mut viewer = Viewer::new(1024, 768, "forge3d Viewer").unwrap();

    // Set camera mode
    viewer.set_camera_mode(CameraMode::Orbit);
    viewer.set_orbit_target([0.0, 0.0, 0.0]);
    viewer.set_orbit_distance(10.0);

    // Run event loop
    viewer.run();
}
```

### Python API (planned)

```python
import forge3d as f3d

# Create interactive viewer
viewer = f3d.create_interactive_viewer(
    width=1024,
    height=768,
    scene=my_scene,
    camera_mode="orbit"  # or "fps"
)

# Configure camera
viewer.set_orbit_target([0.0, 0.0, 0.0])
viewer.set_orbit_distance(10.0)

# Run (blocks until window closed)
viewer.run()
```

## Implementation Roadmap

As outlined in the [Workstream I Audit Report](../../reports/audit_I.md):

### Phase 1: Core Viewer (~3-5 days)

1. **`src/viewer/mod.rs`** — winit integration
   - EventLoop and Window setup
   - Surface and swapchain configuration
   - 60 FPS render loop
   - DPI scaling support

2. **`src/viewer/camera_controller.rs`** — Camera controls
   - Orbit camera implementation
   - FPS camera implementation
   - Input event handling

3. **Example:** `examples/interactive_viewer.rs` or `examples/interactive_viewer_demo.py`

4. **Tests:** Smoke tests and camera delta validation

### Phase 2: Python Bindings (optional, ~1-2 days)

- PyO3 bindings in `python/forge3d/viewer.py`
- `create_interactive_viewer()` function
- Python example demonstrating usage

### Phase 3: Documentation & Polish (~1 day)

- Update this page with actual API
- Add tutorial and examples
- Platform-specific notes

## Alternatives (Current Workarounds)

Until the interactive viewer is implemented, use these alternatives:

### 1. Offscreen Rendering Loop

```python
import forge3d as f3d

# Manual camera animation
for i in range(360):
    angle = i * (3.14159 / 180.0)
    eye = [10 * cos(angle), 5, 10 * sin(angle)]

    # Render frame
    rgba = render_with_camera(eye=eye, target=[0,0,0], up=[0,1,0])
    f3d.save_png_deterministic(f"frame_{i:04d}.png", rgba)
```

### 2. External Viewers

Export geometry and view in external tools:
- **Blender** — Use `f3d.export_obj()` or `f3d.export_gltf()`
- **MeshLab** — For point clouds and meshes

### 3. Jupyter Notebooks

Use inline visualization for static frames:

```python
from forge3d.helpers.ipython_display import display_offscreen

display_offscreen(800, 600, scene=my_scene, title="View 1")
```

## Platform Support

When implemented, the interactive viewer will support:

- ✅ **Windows** — DX12 backend (primary target)
- ✅ **Linux** — Vulkan backend (X11 and Wayland)
- ✅ **macOS** — Metal backend

### DPI Scaling

High-DPI displays (Retina, 4K) will be handled automatically via winit's scale_factor API.

## Dependencies

The interactive viewer will require:

- **winit** — Windowing and input (already in [Cargo.toml](../../Cargo.toml#L50))
- **wgpu** — GPU surface creation

No additional runtime dependencies beyond forge3d's existing stack.

## See Also

- [Screenshot Controls](screenshot_controls.md) — Capture frames from viewer (F12 hotkey when viewer implemented)
- [Offscreen Rendering](../offscreen/index.md) — Headless alternative
- [Workstream I Audit Report](../../reports/audit_I.md) — Implementation status
