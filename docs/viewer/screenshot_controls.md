# Screenshot & Record Controls

forge3d provides tools for capturing screenshots and recording frame sequences, with support for embedding camera and exposure metadata.

## Quick Start

### Screenshot with Metadata

```python
import forge3d as f3d
from forge3d.helpers.offscreen import save_png_with_exif, render_offscreen_rgba

# Render a frame
rgba = render_offscreen_rgba(800, 600, seed=42)

# Define metadata
metadata = {
    "camera": {
        "eye": [10.0, 20.0, 30.0],
        "target": [0.0, 0.0, 0.0],
        "up": [0.0, 1.0, 0.0],
        "fov_deg": 45.0
    },
    "exposure": {
        "mode": "ACES",
        "stops": 0.5,
        "gamma": 2.2
    }
}

# Save with metadata
save_png_with_exif("screenshot.png", rgba, metadata)
```

### Record Frame Sequence

```python
from forge3d.helpers.frame_dump import FrameDumper

dumper = FrameDumper(output_dir="frames", prefix="render")
dumper.start_recording()

for i in range(100):
    rgba = render_frame(i)  # Your rendering function
    dumper.capture_frame(rgba)

frame_count = dumper.stop_recording()
print(f"Recorded {frame_count} frames")
# Output: frames/render_0000.png, frames/render_0001.png, ..., frames/render_0099.png
```

## Metadata Schema

### Camera Metadata

| Field | Type | Description |
|-------|------|-------------|
| `eye` | list[float] | Camera position (x, y, z) |
| `target` | list[float] | Look-at target (x, y, z) |
| `up` | list[float] | Up vector (x, y, z) |
| `fov_deg` | float | Field of view in degrees |

### Exposure Metadata

| Field | Type | Description |
|-------|------|-------------|
| `mode` | str | Tone-mapping mode (e.g., "ACES", "Reinhard", "Hable") |
| `stops` | float | Exposure adjustment in stops |
| `gamma` | float | Gamma correction value |

### General Metadata

| Field | Type | Description |
|-------|------|-------------|
| `description` | str | Free-form description |
| `software` | str | Software name and version |

## Metadata Storage

Metadata is embedded in PNG files as **text chunks** (tEXt/zTXt), which are standard PNG features. This approach:

- ✅ Maintains PNG compatibility with all viewers
- ✅ Does not affect image data or quality
- ✅ Can be extracted with standard tools
- ✅ Negligible file size impact (~100-200 bytes)

### Extracting Metadata

**Using PIL/Pillow (Python):**

```python
from PIL import Image

img = Image.open("screenshot.png")
if hasattr(img, 'text'):
    for key, value in img.text.items():
        print(f"{key}: {value}")
```

**Using exiftool (Command Line):**

```bash
exiftool screenshot.png | grep forge3d
```

## Frame Dumper API

### `FrameDumper` Class

```python
class FrameDumper:
    def __init__(self, output_dir="frames", prefix="frame"):
        """Initialize frame dumper.

        Args:
            output_dir: Directory for output frames (created if missing)
            prefix: Filename prefix
        """

    def start_recording(self) -> None:
        """Start recording frames."""

    def capture_frame(self, rgba: np.ndarray) -> Path:
        """Capture a single frame.

        Args:
            rgba: RGBA array (H, W, 4) uint8 or float32

        Returns:
            Path to saved frame
        """

    def stop_recording(self) -> int:
        """Stop recording.

        Returns:
            Number of frames captured
        """

    def get_frame_count(self) -> int:
        """Get current frame count."""

    def is_recording(self) -> bool:
        """Check if currently recording."""
```

### Convenience Function

```python
def dump_frame_sequence(
    frames: list[np.ndarray],
    output_dir="frames",
    prefix="frame"
) -> int:
    """Dump a list of frames to disk.

    Args:
        frames: List of RGBA arrays
        output_dir: Output directory
        prefix: Filename prefix

    Returns:
        Number of frames written
    """
```

## Interactive Hotkeys

**Note:** Interactive hotkeys require the windowed viewer (Workstream I1). When the interactive viewer is available:

- **F12** — Capture screenshot with timestamp
- **R** — Start/stop frame recording

## Examples

See:
- [examples/screenshot_demo.py](../../examples/screenshot_demo.py) — Demonstrates EXIF and frame dump
- [examples/viewer_offscreen_demo.py](../../examples/viewer_offscreen_demo.py) — Basic offscreen rendering

## Best Practices

### Deterministic Output

When determinism is required (e.g., regression testing), use `save_png_deterministic()` **without** metadata:

```python
from forge3d.helpers.offscreen import save_png_deterministic

save_png_deterministic("test.png", rgba)  # No metadata = deterministic
```

When EXIF is disabled (metadata=None), `save_png_with_exif()` produces deterministic output identical to `save_png_deterministic()`.

### Frame Sequence Organization

Organize large frame sequences by scene or shot:

```python
dumper = FrameDumper(output_dir="renders/shot01", prefix="frame")
```

### Memory Management

For long recordings, consider:

- Recording to SSD for fast I/O
- Limiting resolution (e.g., 1080p preview instead of 4K)
- Using batch processing with frame ranges

## Limitations

- Metadata is stored as **text**, not binary EXIF
- Some tools may not display PNG text chunks by default
- Metadata is **not** preserved when converting PNG to other formats (JPEG, WebP, etc.)

## See Also

- [Offscreen Rendering](../offscreen/index.md)
- [Interactive Viewer](interactive_viewer.md) (for hotkeys, when implemented)
