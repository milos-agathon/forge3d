# Mount Fuji Labels Demo

This example demonstrates the forge3d labeling system with terrain visualization, showcasing features from Plan 1 (MVP), Plan 2 (Cartographic Rules), and Plan 3 (Premium Typography).

## Overview

The `fuji_labels_demo.py` example loads a Mount Fuji terrain DEM and overlays place name labels from OpenStreetMap data stored in a GeoPackage file. It demonstrates the full range of labeling capabilities available in forge3d.

## Features Demonstrated

### Plan 1: MVP Label Features
- **MSDF Font Atlas**: Pre-generated signed distance field font atlas for crisp text at any size
- **Screen-space Placement**: Labels anchored to world positions, rendered in screen space
- **Grid-based Collision**: Prevents overlapping labels
- **Depth Occlusion**: Labels behind terrain are hidden/faded
- **Halo Rendering**: White outline around dark text for readability

### Plan 2: Cartographic Rules Engine
- **Priority System**: Higher priority labels are placed first (based on elevation)
- **Scale-dependent Visibility**: Labels appear/disappear based on zoom level (`min_zoom`, `max_zoom`)
- **R-tree Collision Detection**: Efficient collision detection for many labels
- **Leader Lines**: Offset labels with connector lines to anchor points
- **Horizon Fade**: Labels near the horizon fade smoothly
- **Rotated Labels**: Text at any angle using `rotation` parameter
- **Label Styling**: Underline and small-caps text styles

### Plan 3: Premium Typography
- **Typography Controls**: Letter-spacing (`tracking`), kerning, word spacing, line height
- **Declutter Algorithms**: Greedy (fast) or simulated annealing (better results)
- **Callout Boxes**: Labeled boxes with rounded corners, pointers, and customizable styling
- **Curved Text**: Labels along polyline paths (for contours, rivers, roads)
- **Line Labels**: Text following paths with configurable placement and repetition

## Example Demonstrations

The demo includes the following comprehensive examples:

### 1. Basic Place Name Labels
All place names from OpenStreetMap data with:
- Elevation-based priority (higher peaks = higher priority)
- Scale-dependent visibility (high peaks always visible, lower landmarks only when zoomed)
- Leader lines for offset labels
- Horizon fade for distant labels

### 2. Advanced Label Styling
- **Underlined Labels**: Major landmarks like "FIFTH STATION" with underline style
- **Small Caps Labels**: Geographic features like "Crater Lake" in small capitals
- **Rotated Labels**: Ridge and valley names at various angles (45°, -30°, 90°)
- **Custom Colors**: Different text colors for different feature types

### 3. Callout Boxes
- **Summit Callout**: Main peak with golden border, multi-line text
- **Information Callout**: Visitor center details with blue border
- **Warning Callout**: Hazard warnings with yellow background and red text

### 4. Line and Curved Labels
- **Hiking Trail**: "Yoshida Trail" following a simulated path along terrain
- **Contour Line**: "3000m CONTOUR" curved along elevation contour
- **Path Following**: Text naturally following terrain features

### 5. Horizon Fade Demonstration
- Multiple distant peak labels with varying fade angles (5°, 10°, 15°)
- Shows how labels gracefully disappear near the horizon

### 6. Scale-Dependent Visibility
- Detail labels only visible at zoom levels 2-10
- Overview labels always visible
- Demonstrates multi-scale cartographic design

### 7. Priority System
- Three overlapping labels (HIGH, MED, LOW priority)
- Shows collision resolution based on priority values
- Color-coded by priority level

### 8. Typography Settings
- Global tracking (letter-spacing) set to 0.02
- Kerning enabled for better text spacing
- Line height set to 1.2 for multi-line callouts

## Usage

### Interactive Viewing
```bash
python examples/fuji_labels_demo.py
```

### With Preset
```bash
python examples/fuji_labels_demo.py --preset high_quality
```

### Take Snapshot
```bash
python examples/fuji_labels_demo.py --snapshot fuji_labels.png
```

### Custom Options
```bash
python examples/fuji_labels_demo.py --width 1920 --height 1080 --pbr --shadows pcss
```

## Requirements

- `assets/tif/Mount_Fuji_30m.tif` - Terrain DEM
- `assets/gpkg/Mount_Fuji_places.gpkg` - OSM place names
- `assets/fonts/default_atlas.png` - Font atlas image
- `assets/fonts/default_atlas.json` - Font atlas metrics

## Label API Reference

### Basic Labels (Plan 1)
```python
from forge3d.viewer_ipc import add_label, load_label_atlas

# Load font atlas first
load_label_atlas(sock, "path/to/atlas.png", "path/to/atlas.json")

# Add a simple label
add_label(
    sock,
    text="Mountain Peak",
    world_pos=(x, y, z),
    size=16.0,
    color=(0.1, 0.1, 0.1, 1.0),
    halo_color=(1.0, 1.0, 1.0, 0.8),
    halo_width=1.5,
)
```

### Priority and Zoom (Plan 2)
```python
add_label(
    sock,
    text="Capital City",
    world_pos=(x, y, z),
    priority=100,          # Higher = more important
    min_zoom=0.5,          # Visible from zoom 0.5
    max_zoom=10.0,         # Hidden above zoom 10
    horizon_fade_angle=10.0,  # Fade near horizon
)
```

### Offset Labels with Leader Lines (Plan 2)
```python
add_label(
    sock,
    text="Summit: 3776m",
    world_pos=(x, y, z),
    offset=(30.0, -20.0),  # Screen-space offset in pixels
    leader=True,           # Show connector line
)
```

### Line Labels (Plan 2)
```python
from forge3d.viewer_ipc import add_line_label

add_line_label(
    sock,
    text="River Name",
    polyline=[(x1, y1, z1), (x2, y2, z2), ...],
    placement="along",     # "center" or "along"
    repeat_distance=200.0, # Repeat every 200px
)
```

### Typography Settings (Plan 3)
```python
from forge3d.viewer_ipc import set_label_typography

set_label_typography(
    sock,
    tracking=0.05,      # Letter-spacing (0.0 = normal)
    kerning=True,       # Enable kerning adjustments
    line_height=1.2,    # Line height multiplier
    word_spacing=1.0,   # Word spacing multiplier
)
```

### Declutter Algorithm (Plan 3)
```python
from forge3d.viewer_ipc import set_declutter_algorithm

# Fast greedy algorithm
set_declutter_algorithm(sock, algorithm="greedy")

# Better results with simulated annealing
set_declutter_algorithm(
    sock, 
    algorithm="annealing",
    seed=42,              # For reproducibility
    max_iterations=1000,
)
```

### Callout Boxes (Plan 3)
```python
from forge3d.viewer_ipc import add_callout

add_callout(
    sock,
    text="Multi-line\nCallout Text",
    anchor=(x, y, z),              # World position for pointer
    offset=(0.0, -50.0),           # Screen offset from anchor
    background_color=(1.0, 1.0, 1.0, 0.95),
    border_color=(0.2, 0.2, 0.2, 1.0),
    border_width=1.0,
    corner_radius=4.0,
    padding=8.0,
    text_size=14.0,
    text_color=(0.1, 0.1, 0.1, 1.0),
)
```

### Curved Labels (Plan 3)
```python
from forge3d.viewer_ipc import add_curved_label

# Curved label along a terrain contour
contour_points = []
center_lon, center_lat = 138.73, 35.355
radius_deg = 0.015
for i in range(12):
    angle = (i / 11.0) * math.pi * 1.5
    lon = center_lon + radius_deg * math.cos(angle)
    lat = center_lat + radius_deg * math.sin(angle)
    x, z, _ = world_to_terrain_local(lon, lat)
    contour_points.append((x, elevation, z))

add_curved_label(
    sock,
    text="3000m CONTOUR",
    polyline=contour_points,
    size=14.0,
    color=(0.4, 0.2, 0.0, 1.0),
    halo_color=(1.0, 1.0, 1.0, 0.85),
    halo_width=1.5,
    priority=90,
    tracking=0.05,         # Wider letter-spacing
    center_on_path=True,   # Center text on path
)
```

### Rotated Labels (Plan 2)
```python
import math

# Add label rotated at 45 degrees
angle_rad = math.radians(45)
add_label(
    sock,
    text="Ridge NE",
    world_pos=(x, y, z),
    size=14.0,
    color=(0.3, 0.0, 0.3, 1.0),
    halo_color=(1.0, 1.0, 1.0, 0.85),
    halo_width=1.5,
    rotation=angle_rad,  # Angle in radians
)
```

### Styled Labels (Plan 2)
```python
# Underlined label for major landmark
add_label(
    sock,
    text="FIFTH STATION",
    world_pos=(x, y, z),
    size=24.0,
    color=(0.0, 0.2, 0.5, 1.0),
    halo_color=(1.0, 1.0, 1.0, 0.95),
    halo_width=3.0,
    priority=150,
    underline=True,  # Underlined style
)

# Small caps label for geographic features
add_label(
    sock,
    text="Crater Lake",
    world_pos=(x, y, z),
    size=16.0,
    color=(0.0, 0.4, 0.6, 1.0),
    halo_color=(1.0, 1.0, 1.0, 0.9),
    halo_width=2.0,
    small_caps=True,  # Small caps style
)
```

## Window Controls

| Control | Action |
|---------|--------|
| Mouse drag | Orbit camera |
| Scroll wheel | Zoom in/out |
| W/S or ↑/↓ | Tilt camera up/down |
| A/D or ←/→ | Rotate camera left/right |
| Q/E | Zoom out/in |

## Label Features via IPC and Terminal Commands

### IPC Workflow

The forge3d labeling system uses **Inter-Process Communication (IPC)** rather than direct CLI flags. The viewer runs as a separate process and receives label commands via:

1. **Python IPC API** (recommended) - Used by `fuji_labels_demo.py`
2. **Direct socket communication** - For custom integrations
3. **Terminal commands** (limited) - During interactive viewing

### Why IPC Instead of CLI Flags?

Labels are **dynamic and interactive** - you typically want to add, remove, and modify them during runtime based on:
- Loaded terrain/mesh data
- User interactions
- Real-time calculations
- External data sources (databases, APIs, GeoPackages)

CLI flags work for **static configuration** (window size, FOV), but labels require **bidirectional communication** and **state management**.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Python Script (examples/fuji_labels_demo.py)               │
│  - Loads terrain/data                                        │
│  - Calculates label positions                                │
│  - Sends IPC commands via socket                             │
└──────────────────────┬──────────────────────────────────────┘
                       │ TCP Socket (localhost:port)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  forge3d Viewer Process (interactive_viewer)                │
│  - Renders terrain/meshes                                    │
│  - Receives IPC commands                                     │
│  - Updates labels dynamically                                │
│  - Responds with status                                      │
└─────────────────────────────────────────────────────────────┘
```

### Running with IPC Mode

Start the viewer in IPC mode:
```bash
# Start viewer with IPC server
forge3d_viewer --ipc-port 9999

# Or with Python helper
python examples/fuji_labels_demo.py
```

The Python script automatically:
1. Starts the viewer with IPC enabled
2. Waits for viewer to be ready
3. Sends label commands via socket
4. Handles responses

### Terminal Commands

When running interactively, you can use these terminal commands:

**General Viewer Commands:**
```bash
set phi=45 theta=60 radius=2000 fov=55
set sun_az=135 sun_el=45 intensity=1.5
snap output.png [1920x1080]
pbr on/off
quit
```

**Label System Commands** (limited terminal support):
```bash
# Note: Most label features require Python IPC API
# Terminal commands are primarily for debugging

# Clear all labels
:clear-labels

# Enable/disable labels
:labels on
:labels off

# Set zoom level (affects scale-dependent visibility)
:label-zoom 2.5

# Set max visible labels
:label-max 50
```

### Complete IPC Command Reference

For full control, use the Python IPC API. Here's what gets sent over the socket:

#### Basic Label
```python
# Python API call
add_label(sock, text="Peak", world_pos=(x, y, z), size=16.0)

# Equivalent IPC JSON (sent internally)
{
    "cmd": "add_label",
    "text": "Peak",
    "world_pos": [x, y, z],
    "size": 16.0,
    "color": [0.1, 0.1, 0.1, 1.0],
    "halo_color": [1.0, 1.0, 1.0, 0.9],
    "halo_width": 2.0
}
```

#### Styled Label with Plan 2 Features
```python
# Python API
add_label(
    sock, text="Summit", world_pos=(x, y, z),
    size=20.0, priority=100, min_zoom=0.5, max_zoom=10.0,
    rotation=0.785,  # 45 degrees in radians
    underline=True, offset=(30.0, -20.0), leader=True,
    horizon_fade_angle=10.0
)

# IPC JSON
{
    "cmd": "add_label",
    "text": "Summit",
    "world_pos": [x, y, z],
    "size": 20.0,
    "priority": 100,
    "min_zoom": 0.5,
    "max_zoom": 10.0,
    "rotation": 0.785,
    "underline": true,
    "offset": [30.0, -20.0],
    "leader": true,
    "horizon_fade_angle": 10.0
}
```

#### Callout (Plan 3)
```python
# Python API
add_callout(
    sock, text="Summit\n3776m", anchor=(x, y, z),
    offset=(0.0, -60.0),
    background_color=(1.0, 1.0, 1.0, 0.9),
    border_color=(0.8, 0.5, 0.0, 1.0),
    border_width=2.0, corner_radius=6.0, padding=10.0
)

# IPC JSON
{
    "cmd": "add_callout",
    "text": "Summit\\n3776m",
    "anchor": [x, y, z],
    "offset": [0.0, -60.0],
    "background_color": [1.0, 1.0, 1.0, 0.9],
    "border_color": [0.8, 0.5, 0.0, 1.0],
    "border_width": 2.0,
    "corner_radius": 6.0,
    "padding": 10.0
}
```

#### Curved Label (Plan 3)
```python
# Python API
add_curved_label(
    sock, text="CONTOUR 3000m",
    polyline=[(x1,y1,z1), (x2,y2,z2), ...],
    size=14.0, tracking=0.05, center_on_path=True
)

# IPC JSON
{
    "cmd": "add_curved_label",
    "text": "CONTOUR 3000m",
    "polyline": [[x1,y1,z1], [x2,y2,z2], ...],
    "size": 14.0,
    "tracking": 0.05,
    "center_on_path": true
}
```

#### Typography Settings (Plan 3)
```python
# Python API
set_label_typography(sock, tracking=0.02, kerning=True, line_height=1.2)

# IPC JSON
{
    "cmd": "set_label_typography",
    "tracking": 0.02,
    "kerning": true,
    "line_height": 1.2
}
```

#### Declutter Algorithm (Plan 3)
```python
# Python API
set_declutter_algorithm(sock, algorithm="greedy", seed=42)

# IPC JSON
{
    "cmd": "set_declutter_algorithm",
    "algorithm": "greedy",
    "seed": 42
}
```

### Custom IPC Integration

For non-Python applications, connect directly to the IPC socket:

```bash
# Start viewer
forge3d_viewer --ipc-port 9999 &

# Send command via netcat (example)
echo '{"cmd":"add_label","text":"Test","world_pos":[0,100,0]}' | nc localhost 9999
```

**Response format:**
```json
{
    "ok": true,
    "id": 12345  // Label ID for later removal
}
```

### Why Not CLI Flags for Labels?

Imagine trying to add 50+ labels with CLI flags:
```bash
# ❌ This would be terrible UX
forge3d_viewer \
  --label1 "Peak 1,100.5,200.3,150.2,16,..." \
  --label2 "Peak 2,105.2,198.7,148.9,16,..." \
  --label3 "Valley,110.8,205.1,142.3,14,..." \
  # ... 47 more labels
```

Instead, use Python for data-driven labeling:
```python
# ✅ Clean, readable, dynamic
for peak in peaks_dataframe.itertuples():
    add_label(
        sock,
        text=peak.name,
        world_pos=(peak.x, peak.y, peak.z),
        size=calculate_size(peak.elevation),
        priority=peak.elevation // 100
    )
```

## Advanced Use Cases

### Multi-Callout Information System
```python
# Create an information system with different callout styles
callout_types = [
    {
        "name": "Summit",
        "bg": (1.0, 1.0, 0.95, 0.95),
        "border": (0.8, 0.5, 0.0, 1.0),
        "text_color": (0.0, 0.0, 0.0, 1.0),
    },
    {
        "name": "Info",
        "bg": (1.0, 1.0, 1.0, 0.92),
        "border": (0.0, 0.5, 0.8, 1.0),
        "text_color": (0.0, 0.0, 0.0, 1.0),
    },
    {
        "name": "Warning",
        "bg": (1.0, 1.0, 0.0, 0.88),
        "border": (0.8, 0.0, 0.0, 1.0),
        "text_color": (0.5, 0.0, 0.0, 1.0),
    },
]

for callout_type in callout_types:
    add_callout(
        sock,
        text=f"{callout_type['name']}\nDetails here",
        anchor=(x, y, z),
        offset=(0.0, -60.0),
        background_color=callout_type["bg"],
        border_color=callout_type["border"],
        text_color=callout_type["text_color"],
        border_width=2.0,
        corner_radius=6.0,
        padding=10.0,
    )
```

### Hiking Trail Visualization
```python
# Create a simulated hiking trail with line label
trail_points = []
for i in range(8):
    t = i / 7.0
    # Winding path calculation
    lon = base_lon + 0.03 * t + 0.005 * math.sin(t * math.pi * 2)
    lat = base_lat + 0.04 * t + 0.003 * math.cos(t * math.pi * 3)
    elev = (1500.0 + 800.0 * t) * Z_SCALE
    x, z, _ = world_to_terrain_local(lon, lat)
    trail_points.append((x, elev, z))

add_line_label(
    sock,
    text="Yoshida Trail",
    polyline=trail_points,
    size=16.0,
    color=(0.6, 0.3, 0.0, 1.0),  # Brown
    halo_color=(1.0, 1.0, 1.0, 0.9),
    halo_width=2.0,
    priority=120,
    placement="along",
)
```

### Priority-Based Collision Resolution
```python
# Demonstrate label collision with priority
overlap_positions = [
    (x, y, z),
    (x + 20, y, z),
    (x + 40, y, z),
]
priorities = [200, 100, 50]  # HIGH, MEDIUM, LOW

for pos, priority, name in zip(overlap_positions, priorities, ["HIGH", "MED", "LOW"]):
    add_label(
        sock,
        text=f"{name} Priority",
        world_pos=pos,
        priority=priority,
        color=(1.0 - priority/200.0, 0.0, priority/200.0, 1.0),
    )
# Result: Only HIGH priority label will be visible due to collision
```

### Scale-Dependent Detail Levels
```python
# Overview labels (always visible)
add_label(sock, text="Major Peak", world_pos=pos1, 
          priority=100, min_zoom=0.0, max_zoom=100.0)

# Intermediate detail (zoom 1-10)
add_label(sock, text="Ridge Name", world_pos=pos2,
          priority=80, min_zoom=1.0, max_zoom=10.0)

# Fine detail (zoom 2+)
add_label(sock, text="Small Feature", world_pos=pos3,
          priority=60, min_zoom=2.0, max_zoom=20.0)
```

### Horizon Fade for Atmospheric Perspective
```python
# Labels fade naturally as they approach horizon
distances = [(0.5, 15.0), (1.0, 10.0), (1.5, 5.0)]

for distance_factor, fade_angle in distances:
    far_x, far_z, _ = world_to_terrain_local(
        base_lon + distance_factor * 0.05,
        base_lat + distance_factor * 0.03
    )
    add_label(
        sock,
        text=f"Distant Peak",
        world_pos=(far_x, far_y, far_z),
        horizon_fade_angle=fade_angle,  # Smaller = more aggressive fade
    )
```

## Architecture

The labeling system is implemented in Rust with a Python API:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Python API                               │
│  add_label(), add_callout(), add_curved_label(), etc.          │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LabelManager (Rust)                          │
│  - Label storage and lifecycle                                  │
│  - World-to-screen projection                                   │
│  - R-tree collision detection                                   │
│  - Priority sorting and scale filtering                         │
│  - Horizon fade calculation                                     │
│  - Curved path layout (Plan 3)                                  │
│  - Declutter algorithms (Plan 3)                                │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│               TextOverlayRenderer                               │
│  - MSDF atlas texture                                           │
│  - Instance buffer for glyphs                                   │
│  - Fragment shader with halo support                            │
│  - Typography controls (tracking, kerning)                      │
└─────────────────────────────────────────────────────────────────┘
```

## GPU Resource Usage

| Resource | Format | Size | Notes |
|----------|--------|------|-------|
| MSDF Atlas | Rgba8Unorm | ~16 MiB | Per font family |
| Instance Buffer | Dynamic | ~160 KiB | Grows with label count |
| Collision R-tree | CPU | ~500 KiB | Label bounding boxes |

Total VRAM impact: ~17 MiB per font

## See Also

- [Label API Reference](../api/labels.md)
- [Text Rendering](../api/text_overlay.md)
- [Typography Guide](../api/typography.md)
