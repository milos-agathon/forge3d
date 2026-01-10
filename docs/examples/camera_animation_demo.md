# Camera Animation Demo

**Feature C: Camera Path + Keyframe Animation + MP4 Export**

This example demonstrates the camera animation system for creating smooth, keyframe-based camera flyovers of terrain. The system supports real-time preview in the interactive viewer and offline rendering to PNG sequences or MP4 video files.

## Overview

The camera animation feature provides:

- **Keyframe-based animation**: Define camera positions at specific times
- **Cubic Hermite interpolation**: Smooth Catmull-Rom spline interpolation between keyframes
- **Interactive preview**: Real-time animation playback in the viewer
- **Frame export**: Render animation frames to PNG sequences
- **MP4 encoding**: Direct export to MP4 video using ffmpeg
- **Flexible camera control**: Control azimuth (phi), elevation (theta), radius, and FOV

## Prerequisites

### Required
- Python 3.8+
- forge3d extension module installed (`maturin develop --release`)
- Interactive viewer binary built (`cargo build --release --bin interactive_viewer`)
- DEM terrain file (e.g., `assets/tif/Mount_Fuji_30m.tif`)

### Optional (for MP4 export)
- **ffmpeg**: Required for MP4 video encoding
  - macOS: `brew install ffmpeg`
  - Linux: `apt-get install ffmpeg`
  - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

## Quick Start

### Interactive Preview

Preview an orbit animation in the interactive viewer:

```bash
python examples/camera_animation_demo.py
```

This will:
1. Create a 10-second orbit animation with 5 keyframes
2. Launch the interactive viewer with Mount Fuji terrain
3. Play the animation in a continuous loop
4. Close with Ctrl+C or by closing the window

### Export to PNG Frames

Export animation frames to a directory:

```bash
python examples/camera_animation_demo.py --export ./frames --fps 30
```

Output:
- `frames/frame_0000.png` through `frames/frame_0299.png` (for 10s @ 30fps)

### Export to MP4 Video

Export animation directly to MP4:

```bash
python examples/camera_animation_demo.py --export ./frames --mp4 --fps 30
```

Output:
- PNG frames in `./frames/`
- MP4 video: `./frames.mp4`

### Export MP4 with Cleanup

Export MP4 and delete PNG frames:

```bash
python examples/camera_animation_demo.py \
  --export ./frames \
  --mp4 \
  --mp4-output mount_fuji_orbit.mp4 \
  --cleanup-frames \
  --fps 60 \
  --width 3840 \
  --height 2160
```

Output:
- 4K MP4 video at 60fps: `mount_fuji_orbit.mp4`
- PNG frames automatically deleted

## Animation Types

The demo includes three pre-built animation types:

### 1. Orbit (Default)

360° rotation around the terrain at varying elevation and distance.

```bash
python examples/camera_animation_demo.py --animation orbit
```

**Characteristics:**
- Smooth circular path around the center
- Elevation varies from 35° to 50°
- Radius varies from 2200 to 2800 meters
- FOV fixed at 55°
- Duration: 10 seconds (default)

### 2. Flyover

Dramatic zoom and pan sequence with multiple viewpoints.

```bash
python examples/camera_animation_demo.py --animation flyover
```

**Characteristics:**
- Starts with wide establishing shot (radius 5000m)
- Zooms in for close-up detail (radius 1500m)
- Sweeps across terrain from multiple angles
- Elevation varies from 25° to 55°
- FOV changes from 45° to 65° for dramatic effect
- Duration: 15 seconds (default)

### 3. Sunrise

Slow sunrise reveal with gradual elevation change.

```bash
python examples/camera_animation_demo.py --animation sunrise
```

**Characteristics:**
- Low starting elevation (15°) simulating sunrise
- Gradually rises to 45° elevation
- Full 360° rotation during ascent
- Constant radius (3000m) and FOV (50°)
- Duration: 20 seconds (default)

## Command-Line Interface

### Basic Options

```
--dem PATH              Path to DEM file (default: assets/tif/dem_rainier.tif)
--animation TYPE        Animation type: orbit, flyover, sunrise (default: orbit)
--duration SECONDS      Animation duration in seconds (default: 10)
--z-scale FLOAT         Terrain height exaggeration (default: 0.15)
--preview-only          Print interpolation preview without launching viewer
```

### Lighting Options

```
--static-sun            Disable dynamic sun movement (sun stays fixed)
--sun-offset DEGREES    Sun offset angle from camera in degrees (default: 120)
--sun-intensity FLOAT   Sun intensity multiplier (default: 1.0)
```

**Dynamic Lighting (Default):**
By default, the sun direction automatically updates as the camera moves, creating dramatic lighting changes throughout the animation. The sun is positioned at an offset angle from the camera (default 120°) for side/back lighting that reveals terrain features.

**Static Lighting:**
Use `--static-sun` to keep the sun in a fixed position throughout the animation.

### Export Options

```
--export DIR            Export frames to directory (instead of interactive preview)
--fps INT               Frames per second for export (default: 30)
--width INT             Output width in pixels (default: 1920)
--height INT            Output height in pixels (default: 1080)
```

### MP4 Options

```
--mp4                   Create MP4 video after frame export (requires ffmpeg)
--mp4-output FILE       MP4 output path (default: <export_dir>.mp4)
--mp4-quality PRESET    Quality preset: high, medium, low (default: high)
--cleanup-frames        Delete PNG frames after MP4 creation
```

## Usage Examples

### 1. Preview Different Terrains

```bash
# Preview Mount Fuji
python examples/camera_animation_demo.py --dem assets/tif/Mount_Fuji_30m.tif

# Preview Rainier
python examples/camera_animation_demo.py --dem assets/tif/dem_rainier.tif
```

### 2. Test Interpolation Without Viewer

```bash
python examples/camera_animation_demo.py --preview-only
```

Output:
```
Animation Preview: 5 keyframes, 10.0s duration
============================================================
At 2 fps: 21 frames

t= 0.00s | phi=   0.00° theta=45.00° radius= 2500.0 fov= 55.0°
t= 0.50s | phi=  12.24° theta=43.24° radius= 2444.8 fov= 55.0°
...
```

### 3. High-Quality 4K Export

```bash
python examples/camera_animation_demo.py \
  --export ./4k_export \
  --mp4 \
  --fps 60 \
  --width 3840 \
  --height 2160 \
  --mp4-quality high \
  --animation flyover \
  --duration 15
```

### 4. Quick Draft Preview Export

```bash
python examples/camera_animation_demo.py \
  --export ./draft \
  --mp4 \
  --fps 15 \
  --width 1280 \
  --height 720 \
  --mp4-quality low \
  --cleanup-frames
```

### 5. Custom Duration and Terrain Scale

```bash
python examples/camera_animation_demo.py \
  --animation sunrise \
  --duration 30 \
  --z-scale 0.25 \
  --export ./sunrise_export \
  --mp4
```

### 6. Control Lighting

**Front lighting (sun behind camera):**
```bash
python examples/camera_animation_demo.py \
  --export ./front_lit \
  --mp4 \
  --sun-offset 180
```

**Side lighting (dramatic shadows):**
```bash
python examples/camera_animation_demo.py \
  --export ./side_lit \
  --mp4 \
  --sun-offset 90
```

**Static lighting (no sun movement):**
```bash
python examples/camera_animation_demo.py \
  --export ./static_lit \
  --mp4 \
  --static-sun
```

**High-intensity backlighting:**
```bash
python examples/camera_animation_demo.py \
  --export ./backlit \
  --mp4 \
  --sun-offset 150 \
  --sun-intensity 1.5
```

## Python API Reference

### CameraAnimation Class

```python
from forge3d import CameraAnimation

# Create animation
anim = CameraAnimation()

# Add keyframes (time, phi, theta, radius, fov)
anim.add_keyframe(0.0, 0.0, 45.0, 2500.0, 55.0)
anim.add_keyframe(5.0, 180.0, 35.0, 2000.0, 55.0)
anim.add_keyframe(10.0, 360.0, 45.0, 2500.0, 55.0)

# Evaluate at specific time
state = anim.evaluate(3.5)  # Returns CameraState
print(f"phi={state.phi_deg}, theta={state.theta_deg}")

# Get frame count for given fps
total_frames = anim.get_frame_count(30)  # 300 frames for 10s @ 30fps

# Get animation duration
duration = anim.get_duration()  # 10.0 seconds
```

### CameraState Class

```python
# Returned by anim.evaluate(time)
state = anim.evaluate(2.5)

# Access camera parameters
phi = state.phi_deg      # Azimuth angle (degrees)
theta = state.theta_deg  # Elevation angle (degrees)
radius = state.radius    # Distance from target (meters)
fov = state.fov_deg      # Field of view (degrees)
```

### Creating Custom Animations

```python
from forge3d import CameraAnimation

def create_custom_animation(duration: float) -> CameraAnimation:
    """Create a custom camera animation."""
    anim = CameraAnimation()
    
    # Define keyframes
    keyframes = [
        # time, phi, theta, radius, fov
        (0.0, 0.0, 30.0, 3000.0, 60.0),
        (duration * 0.25, 90.0, 40.0, 2500.0, 55.0),
        (duration * 0.50, 180.0, 45.0, 2000.0, 50.0),
        (duration * 0.75, 270.0, 40.0, 2500.0, 55.0),
        (duration, 360.0, 30.0, 3000.0, 60.0),
    ]
    
    for t, phi, theta, r, fov in keyframes:
        anim.add_keyframe(t, phi, theta, r, fov)
    
    return anim
```

## MP4 Export Details

### Quality Presets

The `--mp4-quality` flag controls the H.264 CRF (Constant Rate Factor):

| Preset | CRF | File Size | Quality | Use Case |
|--------|-----|-----------|---------|----------|
| `high` | 18 | Large | Excellent | Final output, archival |
| `medium` | 23 | Moderate | Good | General use, web |
| `low` | 28 | Small | Acceptable | Previews, drafts |

**CRF Scale**: Lower values = higher quality + larger files (0-51 scale, 18-28 recommended)

### Encoding Settings

MP4 files are encoded with:
- **Codec**: H.264 (libx264)
- **Pixel Format**: yuv420p (universal compatibility)
- **Preset**: medium (balanced speed/compression)
- **Web Optimization**: `+faststart` flag for streaming

### Example Output Sizes

For a 10-second animation at 1920×1080 @ 30fps:

- **High quality** (CRF=18): ~25-35 MB
- **Medium quality** (CRF=23): ~12-18 MB
- **Low quality** (CRF=28): ~6-10 MB

Actual sizes vary based on terrain complexity and motion.

## Frame Export Details

### Frame Naming

Frames are exported with zero-padded sequential numbers:

```
frame_0000.png
frame_0001.png
frame_0002.png
...
frame_0299.png  (for 10s @ 30fps)
```

### Frame Count Calculation

```
total_frames = ceil(duration * fps)
```

Examples:
- 10s @ 30fps = 300 frames
- 15s @ 60fps = 900 frames
- 5s @ 24fps = 120 frames

### Export Performance

Typical frame export times (1920×1080, Mount Fuji terrain):

- **Frame rendering**: ~0.3-0.5 seconds per frame
- **10s @ 30fps (300 frames)**: ~2-3 minutes
- **15s @ 60fps (900 frames)**: ~7-10 minutes

MP4 encoding adds:
- **30fps, 300 frames**: ~5-15 seconds
- **60fps, 900 frames**: ~15-30 seconds

## Troubleshooting

### Interactive Viewer Issues

**Problem**: Purple screen in viewer, animation doesn't start

**Solution**: The terrain needs time to load. The script includes an 8-second warmup delay. If you still see a purple screen:
1. Ensure the DEM file path is correct
2. Check viewer logs for terrain loading errors
3. Try a smaller DEM file first

**Problem**: Animation is choppy or stutters

**Solution**: The viewer runs at 30fps interpolation. For smoother playback:
- Use a lower duration (less time between keyframes)
- Add more intermediate keyframes
- Ensure system has sufficient GPU resources

### Export Issues

**Problem**: Frame export hangs or times out

**Solution**:
1. Check if viewer process started (look for `FORGE3D_VIEWER_READY` message)
2. Increase socket timeout in `export_animation_frames()` if needed
3. Test with a shorter animation first (`--duration 3`)

**Problem**: Frames are black or corrupted

**Solution**:
1. Ensure terrain loaded before rendering (warmup delay in place)
2. Check snapshot wait time (default 0.3s per frame)
3. Verify DEM file is valid

### MP4 Encoding Issues

**Problem**: `ERROR: ffmpeg not found`

**Solution**: Install ffmpeg:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Check installation
ffmpeg -version
```

**Problem**: `ffmpeg failed with code 1`

**Solution**:
1. Check if PNG frames exist in export directory
2. Verify frame naming pattern: `frame_0000.png`, `frame_0001.png`, etc.
3. Try encoding manually:
   ```bash
   ffmpeg -framerate 30 -i frames/frame_%04d.png -c:v libx264 -pix_fmt yuv420p test.mp4
   ```

**Problem**: MP4 file is too large

**Solution**:
1. Use lower quality preset: `--mp4-quality medium` or `--mp4-quality low`
2. Reduce resolution: `--width 1280 --height 720`
3. Lower framerate: `--fps 24`

**Problem**: MP4 won't play in browser/mobile

**Solution**: Ensure using yuv420p pixel format (default). Some players require this for H.264 compatibility.

## Advanced Usage

### Batch Export Multiple Animations

```bash
# Create a shell script
for anim_type in orbit flyover sunrise; do
  python examples/camera_animation_demo.py \
    --animation $anim_type \
    --export ./export_${anim_type} \
    --mp4 \
    --mp4-output ${anim_type}_animation.mp4 \
    --cleanup-frames
done
```

### Custom Python Script

```python
from pathlib import Path
from forge3d import CameraAnimation

def main():
    # Create custom animation
    anim = CameraAnimation()
    
    # Dramatic zoom-in
    anim.add_keyframe(0.0, 45.0, 60.0, 5000.0, 70.0)   # Wide shot
    anim.add_keyframe(5.0, 90.0, 45.0, 2000.0, 50.0)   # Mid shot
    anim.add_keyframe(10.0, 135.0, 30.0, 800.0, 35.0)  # Close-up
    
    # Export using the demo functions
    from examples.camera_animation_demo import export_animation_frames
    
    export_animation_frames(
        dem_path=Path("assets/tif/Mount_Fuji_30m.tif"),
        anim=anim,
        output_dir=Path("./custom_export"),
        fps=30,
        width=1920,
        height=1080,
        z_scale=0.15,
        create_mp4=True,
        mp4_quality="high",
        cleanup_frames=True,
    )

if __name__ == "__main__":
    main()
```

### Integrating with Other Tools

Export frames and use external tools:

```bash
# Export frames
python examples/camera_animation_demo.py --export ./frames --fps 30

# Create GIF (requires ImageMagick)
convert -delay 3.33 -loop 0 frames/frame_*.png animation.gif

# Create WebM (alternative to MP4)
ffmpeg -framerate 30 -i frames/frame_%04d.png -c:v libvpx-vp9 -crf 30 animation.webm

# Create image sequence for Adobe After Effects
# (frames already in correct format)
```

## Performance Tips

### For Faster Export

1. **Lower resolution**: Use `--width 1280 --height 720` for previews
2. **Reduce fps**: Use `--fps 24` instead of 30 or 60
3. **Shorter duration**: Test with `--duration 3` before full export
4. **Lower terrain resolution**: Use smaller/downsampled DEM files

### For Higher Quality

1. **Increase resolution**: Use `--width 3840 --height 2160` for 4K
2. **Higher framerate**: Use `--fps 60` for smooth motion
3. **High quality preset**: Use `--mp4-quality high`
4. **More keyframes**: Add intermediate keyframes for smoother paths

### For Smaller Files

1. **Use medium/low quality**: `--mp4-quality medium`
2. **Lower resolution**: `--width 1280 --height 720`
3. **Reduce fps**: `--fps 24`
4. **Shorter animations**: Reduce `--duration`

## Technical Details

### Interpolation Algorithm

The system uses **Cubic Hermite (Catmull-Rom) spline interpolation** with:
- Automatic tangent calculation from neighboring keyframes
- Smooth C1 continuity between segments
- Time-based clamping to keyframe boundaries

### Camera Coordinate System

- **phi (azimuth)**: Horizontal rotation, 0° = North, 90° = East, clockwise
- **theta (elevation)**: Vertical angle from horizon, 0° = level, 90° = straight up
- **radius**: Distance from terrain center in meters
- **fov**: Vertical field of view in degrees (typical range: 35-70°)

### Dynamic Lighting System

**Sun Direction Calculation:**
```python
sun_azimuth = (camera_phi + sun_offset) % 360.0
sun_elevation = 45.0 + 10.0 * ((camera_phi % 180.0) / 180.0 - 0.5)
```

The sun follows the camera with a configurable offset angle:
- **0°**: Sun directly in front of camera (flat lighting)
- **90°**: Sun to the side (strong shadows)
- **120°** (default): Side/back lighting (dramatic, reveals features)
- **180°**: Sun directly behind camera (silhouette/rim lighting)

Sun elevation varies slightly (40-50°) based on camera position to create more dynamic lighting as the camera moves around the terrain.

**Lighting Scenarios:**

| Sun Offset | Effect | Best For |
|------------|--------|----------|
| 0-30° | Flat, minimal shadows | Scientific visualization |
| 60-90° | Strong side shadows | Feature emphasis |
| 110-130° | Dramatic back/side lighting | Cinematic quality (default) |
| 150-180° | Rim/silhouette lighting | Dramatic reveals |

### IPC Communication

The demo uses the interactive viewer's IPC (Inter-Process Communication) protocol:
- **Connection**: TCP socket to viewer
- **Commands**: JSON-formatted requests
- **Response**: JSON-formatted acknowledgments
- **Timeout**: 30 seconds for frame rendering

### Viewer Startup

The viewer requires initialization time:
1. Shader compilation (~1-2 seconds)
2. Terrain loading (~2-5 seconds)
3. Initial rendering (~1 second)

Total warmup: 4-8 seconds (handled automatically by demo script)

## Related Documentation

- [Terrain Demo Quickstart](terrain_demo_quickstart.rst) - Basic terrain visualization
- [PBM+POM Viewer](pbm_pom_viewer.md) - Advanced terrain rendering features
- [API Reference](../api/api_reference.rst) - Full Python API documentation

## Changelog

### v1.9.1 (January 2026)
- ✅ Initial camera animation implementation (Feature C Plan 1 MVP)
- ✅ MP4 export with ffmpeg integration
- ✅ Three pre-built animation types (orbit, flyover, sunrise)
- ✅ Interactive viewer integration with IPC
- ✅ Frame export to PNG sequences
- ✅ **Dynamic sun lighting** - Sun direction automatically follows camera for dramatic lighting changes
- ✅ Configurable sun offset and intensity
- ✅ Static sun mode for fixed lighting

## License

Copyright © 2026 forge3d contributors. Licensed under the project license.
