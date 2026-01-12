# Order-Independent Transparency (OIT)

P0.1/M1 feature for correct rendering of overlapping transparent surfaces.

## Overview

Order-Independent Transparency (OIT) solves the classic problem of rendering overlapping
transparent surfaces without requiring depth sorting. This is essential for:

- **Water surfaces** with reflections and refractions
- **Volumetric effects** like fog and clouds
- **Vector overlays** on terrain (lines, polygons)
- **Glass and translucent materials**

## Quick Start

```python
import forge3d as f3d

# Create scene
sess = f3d.Session(width=1920, height=1080)
scene = f3d.Scene(sess)

# Enable OIT with automatic mode selection
scene.enable_oit()  # or scene.enable_oit("auto")

# Render
frame = scene.render()
```

## API Reference

### Scene Methods

#### `enable_oit(mode='auto')`

Enable Order-Independent Transparency for correct rendering of overlapping transparent surfaces.

**Parameters:**
- `mode` (str, optional): Transparency mode selection
  - `'auto'` (default): Automatically selects dual-source blending if hardware supports it, otherwise falls back to WBOIT
  - `'wboit'`: Force Weighted Blended OIT (works on all hardware)
  - `'dual_source'`: Force dual-source blending (requires hardware support)
  - `'standard'` or `'disabled'`: Disable OIT, use standard alpha blending

**Example:**
```python
scene.enable_oit()           # automatic mode selection
scene.enable_oit('wboit')    # force weighted-blended OIT
scene.enable_oit('standard') # disable OIT
```

#### `disable_oit()`

Disable Order-Independent Transparency, reverting to standard alpha blending.

```python
scene.disable_oit()
```

#### `is_oit_enabled()`

Check if Order-Independent Transparency is currently enabled.

**Returns:** `bool` - True if OIT is enabled

```python
if scene.is_oit_enabled():
    print("OIT is active")
```

#### `get_oit_mode()`

Get current OIT mode as a string.

**Returns:** `str` - One of `'auto'`, `'wboit'`, `'dual_source'`, or `'disabled'`

```python
mode = scene.get_oit_mode()
print(f"Current OIT mode: {mode}")
```

## CLI Usage

All examples support the `--oit` flag:

```bash
# Enable automatic OIT mode
python examples/terrain_demo.py --oit auto

# Force WBOIT mode
python examples/terrain_viewer_interactive.py --oit wboit

# Disable OIT explicitly
python examples/fuji_labels_demo.py --oit off
```

**Available modes:** `auto`, `wboit`, `dual_source`, `off`

## Viewer IPC

For interactive viewer examples using IPC:

```python
from forge3d.viewer_ipc import send_ipc, set_oit_enabled, get_oit_mode

# Enable OIT via IPC
set_oit_enabled(sock, enabled=True, mode="auto")

# Query current mode
response = get_oit_mode(sock)
print(f"OIT mode: {response.get('mode')}")
```

## Technical Details

### WBOIT (Weighted Blended OIT)

The default fallback mode uses weighted blended order-independent transparency:

- Works on all hardware with standard alpha blending
- Single-pass rendering with accumulation buffers
- Weights based on depth and alpha for correct blending
- Suitable for most use cases with moderate transparency

### Dual-Source Blending

When hardware supports it, dual-source blending provides higher quality:

- Uses two color outputs per fragment
- Better handling of high-opacity transparent surfaces
- Requires `dual_source_blending` WebGPU feature

### Performance Considerations

- OIT adds some GPU overhead for the compositing pass
- WBOIT uses additional render targets for accumulation
- For opaque-only scenes, disable OIT for best performance
- Memory usage is bounded by the 512 MiB GPU budget

## Preset Support

OIT can be configured via JSON presets:

```json
{
  "oit": "auto"
}
```

Valid values: `"auto"`, `"wboit"`, `"dual_source"`, `"off"`
