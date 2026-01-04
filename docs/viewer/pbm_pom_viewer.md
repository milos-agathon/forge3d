# PBR Terrain Viewer - User Guide

**Status: ✅ Complete - All Phases Implemented and Tested**

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [CLI Reference](#cli-reference)
4. [Interactive Commands](#interactive-commands)
5. [Advanced Examples](#advanced-examples)
6. [API Reference](#api-reference)
7. [Technical Details](#technical-details)
8. [Testing](#testing)
9. [Implementation](#implementation)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The interactive terrain viewer now supports an enhanced **PBR rendering mode** with improved lighting, materials, and visual quality. This mode is **opt-in** - the legacy simple shader remains the default for backward compatibility.

### Feature Comparison

| Feature | Legacy Mode | PBR Mode |
|---------|-------------|----------|
| Lighting | Lambertian diffuse | Blinn-Phong specular + diffuse |
| Materials | Height-based colormap | Height + slope based with roughness |
| Shadows | Fake shadow intensity | Real-time CSM shadows (hard/pcf/pcss) |
| Heightfield AO | None | Ray-traced terrain AO (opt-in) |
| Sun Visibility | None | Ray-traced terrain self-shadowing (opt-in) |
| Tonemapping | None | ACES filmic curve |
| Exposure | Fixed | Configurable (0.1 - 5.0) |
| Sun Control | Fixed | Configurable azimuth, elevation, intensity |
| Window Size | 1280×720 | 4K default (3840×2160) |
| Snapshot | Up to 4K | Up to 16K (16384×16384) |

---

## Quick Start

### Build the Viewer

```bash
cargo build --release --bin interactive_viewer
```

### Basic Usage

```bash
# Legacy mode (unchanged default behavior)
python examples/terrain_viewer_interactive.py --dem assets/tif/dem_rainier.tif

# Window size (default 4K)
python examples/terrain_viewer_interactive.py --dem assets/tif/dem_rainier.tif --width 1920 --height 1080

# PBR mode (enhanced rendering)
python examples/terrain_viewer_interactive.py --dem assets/tif/dem_rainier.tif --pbr

# PBR with custom exposure
python examples/terrain_viewer_interactive.py --dem assets/tif/dem_rainier.tif --pbr --exposure 0.2
```

---

## CLI Reference

### Window & Output

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--dem` | Path | Required | GeoTIFF DEM file |
| `--width` | Int | 3840 | Window width (4K default) |
| `--height` | Int | 2160 | Window height (4K default) |
| `--snapshot` | Path | - | Take snapshot and exit |

### PBR Rendering

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--pbr` | Flag | Off | Enable PBR rendering mode |
| `--exposure` | Float | 1.0 | ACES tonemapping exposure (0.1-5.0) |
| `--normal-strength` | Float | 1.0 | Terrain normal amplification |
| `--ibl-intensity` | Float | 1.0 | Ambient/IBL lighting intensity |
| `--hdr` | Path | - | HDR environment map for IBL |
| `--shadows` | String | "pcss" | Shadow technique (none/hard/pcf/pcss) |
| `--shadow-map-res` | Int | 2048 | Shadow map resolution |
| `--msaa` | Int | 1 | MSAA samples (1, 4, or 8) |

### Sun Lighting

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--sun-azimuth` | Float | 135.0 | Sun azimuth angle in degrees |
| `--sun-elevation` | Float | 35.0 | Sun elevation angle in degrees |
| `--sun-intensity` | Float | 1.0 | Sun light intensity multiplier |

### Heightfield AO

Ambient occlusion computed by ray-marching the heightfield in multiple directions, producing soft darkening in concave terrain regions (valleys, crevices).

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--height-ao` | Flag | Off | Enable heightfield ray-traced AO |
| `--height-ao-directions` | Int | 6 | Ray directions around horizon [4-16] |
| `--height-ao-steps` | Int | 16 | Ray march steps per direction [8-64] |
| `--height-ao-distance` | Float | 200.0 | Max ray distance in world units |
| `--height-ao-strength` | Float | 1.0 | AO darkening intensity [0.0-2.0] |
| `--height-ao-resolution` | Float | 0.5 | Texture resolution scale [0.1-1.0] |

### Sun Visibility

Terrain self-shadowing computed by ray-marching along the sun direction, producing shadows where terrain occludes direct sunlight.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--sun-vis` | Flag | Off | Enable heightfield sun visibility |
| `--sun-vis-mode` | String | "soft" | Shadow mode: "hard" or "soft" |
| `--sun-vis-samples` | Int | 4 | Jittered samples for soft shadows [1-16] |
| `--sun-vis-steps` | Int | 24 | Ray march steps toward sun [8-64] |
| `--sun-vis-distance` | Float | 400.0 | Max ray distance in world units |
| `--sun-vis-softness` | Float | 1.0 | Soft shadow penumbra size [0.0-4.0] |
| `--sun-vis-bias` | Float | 0.01 | Self-shadowing bias |
| `--sun-vis-resolution` | Float | 0.5 | Texture resolution scale [0.1-1.0] |

**Note:** Sun visibility combines multiplicatively with CSM shadows when both are enabled.

### Material Layering

Terrain material layers: snow, rock, wetness based on altitude and slope.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--snow` | Flag | Off | Enable snow layer based on altitude and slope |
| `--snow-altitude` | Float | 2500.0 | Minimum altitude for snow in world units |
| `--snow-blend` | Float | 200.0 | Altitude blend range for snow transition |
| `--snow-slope` | Float | 45.0 | Maximum slope for snow accumulation in degrees |
| `--rock` | Flag | Off | Enable exposed rock layer on steep slopes |
| `--rock-slope` | Float | 45.0 | Minimum slope for exposed rock in degrees |
| `--wetness` | Flag | Off | Enable wetness effect in low areas |
| `--wetness-strength` | Float | 0.3 | Wetness darkening strength [0.0-1.0] |

### Vector Overlays

Depth-correct vector overlays with halos.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--overlay-depth` | Flag | Off | Enable depth testing for vector overlays |
| `--overlay-depth-bias` | Float | 0.001 | Depth bias to prevent z-fighting |
| `--overlay-halo` | Flag | Off | Enable halo/outline around vector overlays |
| `--overlay-halo-width` | Float | 2.0 | Halo width in pixels |
| `--overlay-halo-color` | String | "0,0,0,0.5" | Halo color as R,G,B,A |

### Tonemapping

HDR tonemapping and color grading.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--tonemap` | Choice | "aces" | Tonemap operator (reinhard/reinhard_extended/aces/uncharted2/exposure) |
| `--tonemap-white-point` | Float | 4.0 | White point for extended operators |
| `--white-balance` | Flag | Off | Enable white balance adjustment |
| `--temperature` | Float | 6500.0 | Color temperature in Kelvin [2000-12000] |
| `--tint` | Float | 0.0 | Green-magenta tint [-1.0 to 1.0] |

---

## Interactive Commands

### Camera & Terrain

```
> set phi=45 theta=60 radius=2000 fov=55    # Camera position
> set zscale=2.0 shadow=0.5                  # Terrain params
> set background=0.2,0.3,0.5                 # Background color
> set water=1500 water_color=0.1,0.3,0.5     # Water settings
```

### Sun Lighting

```
> set sun_az=180 sun_el=30 intensity=1.2    # Adjust sun at runtime
> sun 135 45 1.0                             # Legacy: azimuth elevation intensity
```

### PBR Mode

```
> pbr on                          # Enable PBR mode
> pbr off                         # Disable PBR mode (return to legacy)
> pbr exposure=2.0                # Change exposure
> pbr shadows=pcss ibl=1.5        # Shadow technique + IBL
> pbr normal=1.5 exposure=1.2     # Change multiple params
```

### Snapshots (up to 16K resolution)

```
> snap output.png                 # Snapshot at window size
> snap output.png 1920x1080       # Snapshot at 1080p
> snap output.png 3840x2160       # Snapshot at 4K
> snap output.png 7680x4320       # Snapshot at 8K
> snap output.png 16384x16384     # Snapshot at 16K (max)
```

---

## Advanced Examples

### Full PBR Configuration

```bash
python examples/terrain_viewer_interactive.py --dem assets/tif/dem_rainier.tif --pbr --exposure 0.2 --hdr assets/hdri/snow_field_4k.hdr --normal-strength 0.5 --ibl-intensity 0.1 --sun-azimuth 90 --sun-elevation 45 --sun-intensity 0.8 --msaa 8
```

### Heightfield Effects

```bash
# Heightfield AO (darkens valleys/crevices)
python examples/terrain_viewer_interactive.py --dem assets/tif/dem_rainier.tif --pbr --height-ao --height-ao-strength 1.0

# Sun visibility (terrain self-shadowing)
python examples/terrain_viewer_interactive.py --dem assets/tif/dem_rainier.tif --pbr --sun-vis --sun-vis-mode soft --sun-elevation 15

# Both terrain effects combined
python examples/terrain_viewer_interactive.py --dem assets/tif/dem_rainier.tif --pbr --height-ao --sun-vis --exposure 1.2
```

### Material & Stylization

```bash
# Alpine scene with snow and rock
python examples/terrain_viewer_interactive.py --dem assets/tif/dem_rainier.tif --pbr --snow --snow-altitude 2500 --rock --rock-slope 45 --tonemap aces

# Cinematic warm render
python examples/terrain_viewer_interactive.py --dem assets/tif/dem_rainier.tif --pbr --tonemap uncharted2 --white-balance --temperature 5500 --exposure 1.2 --tonemap uncharted2 --white-balance --temperature 5500 --exposure 1.2

# All features enabled
python examples/terrain_viewer_interactive.py --dem assets/tif/dem_rainier.tif --pbr --height-ao --sun-vis --snow --rock --wetness --overlay-depth --overlay-halo --tonemap aces --white-balance
```

### Presets

```bash
# Alpine preset
python examples/terrain_viewer_interactive.py --dem assets/tif/dem_rainier.tif --preset alpine

# Cinematic preset  
python examples/terrain_viewer_interactive.py --dem assets/tif/dem_rainier.tif --preset cinematic

# High quality preset
python examples/terrain_viewer_interactive.py --dem assets/tif/dem_rainier.tif --preset high_quality --width 800 --height 800
```

### Post-Processing Effects

The following examples demonstrate the post-processing pipeline.

#### Barrel Distortion + Chromatic Aberration

```bash
# Barrel distortion (center expands, edges compress)
python examples/terrain_viewer_interactive.py --dem assets/tif/dem_rainier.tif --pbr --lens-effects --lens-distortion 0.15

# Chromatic aberration (RGB fringing at edges)
python examples/terrain_viewer_interactive.py --dem assets/tif/dem_rainier.tif --pbr --lens-effects --lens-ca 0.03

# Combined lens effects with vignette
python examples/terrain_viewer_interactive.py --dem assets/tif/dem_rainier.tif --pbr --lens-effects --lens-distortion 0.15 --lens-ca 0.03 --lens-vignette 0.3

# Vintage camera look
python examples/terrain_viewer_interactive.py --dem assets/tif/dem_rainier.tif --pbr --lens-effects --lens-distortion 0.1 --lens-ca 0.02 --lens-vignette 0.5 --tonemap uncharted2
```

#### Depth of Field (Separable Blur)

```bash
# Shallow depth of field (wide aperture, blurry background)
python examples/terrain_viewer_interactive.py --dem assets/tif/dem_rainier.tif --pbr --dof --dof-f-stop 2.8 --dof-focus-distance 300 --dof-quality high

# Deep depth of field (narrow aperture, everything sharp)
python examples/terrain_viewer_interactive.py --dem assets/tif/dem_rainier.tif --pbr --dof --dof-f-stop 16 --dof-focus-distance 500

# Tilt-shift miniature effect (diagonal focus plane)
python examples/terrain_viewer_interactive.py --dem assets/tif/dem_rainier.tif --pbr --dof --dof-f-stop 2.8 --dof-focus-distance 200 --dof-tilt-pitch 30

# Tilt-shift with yaw rotation
python examples/terrain_viewer_interactive.py --dem assets/tif/dem_rainier.tif --pbr --dof --dof-f-stop 2.8 --dof-focus-distance 250 --dof-tilt-pitch 15 --dof-tilt-yaw 45
```

#### Motion Blur (Temporal Accumulation)

```bash
# Camera pan motion blur (horizontal sweep)
python examples/terrain_viewer_interactive.py --dem assets/tif/dem_rainier.tif --pbr --motion-blur --mb-samples 16 --mb-shutter-angle 180 --mb-cam-phi-delta 5

# Camera tilt motion blur (vertical sweep)
python examples/terrain_viewer_interactive.py --dem assets/tif/dem_rainier.tif --pbr --motion-blur --mb-samples 16 --mb-shutter-angle 180 --mb-cam-theta-delta 3

# Zoom blur (dolly in/out)
python examples/terrain_viewer_interactive.py --dem assets/tif/dem_rainier.tif --pbr --motion-blur --mb-samples 16 --mb-shutter-angle 180 --mb-cam-radius-delta -50

# High-quality motion blur (32 samples)
python examples/terrain_viewer_interactive.py --dem assets/tif/dem_rainier.tif --pbr --motion-blur --mb-samples 32 --mb-shutter-angle 270 --mb-cam-phi-delta 10
```

#### Volumetric Fog + Light Shafts

```bash
# Uniform fog
python examples/terrain_viewer_interactive.py --dem assets/tif/dem_rainier.tif --pbr --volumetrics --vol-mode uniform --vol-density 0.02

# Height-based fog (denser near ground)
python examples/terrain_viewer_interactive.py --dem assets/tif/dem_rainier.tif --pbr --volumetrics --vol-mode height --vol-density 0.015

# God rays (volumetric light shafts)
python examples/terrain_viewer_interactive.py --dem assets/tif/dem_rainier.tif --pbr --volumetrics --vol-mode height --vol-density 0.02 --vol-light-shafts --vol-shaft-intensity 2.0 --sun-elevation 15

# Atmospheric haze with scattering
python examples/terrain_viewer_interactive.py --dem assets/tif/dem_rainier.tif --pbr --volumetrics --vol-density 0.01 --vol-scattering 0.8 --vol-absorption 0.1

# Half-resolution volumetrics (performance mode)
python examples/terrain_viewer_interactive.py --dem assets/tif/dem_rainier.tif --pbr --volumetrics --vol-density 0.02 --vol-half-res
```

#### Combined Post-Processing Scenes

```bash
# Cinematic sunrise with fog and god rays
python examples/terrain_viewer_interactive.py --dem assets/tif/dem_rainier.tif --pbr --volumetrics --vol-mode height --vol-density 0.02 --vol-light-shafts --sun-elevation 10 --sun-azimuth 90 --lens-vignette 0.3 --tonemap aces

# Dreamy landscape (DoF + fog + soft lighting)
python examples/terrain_viewer_interactive.py --dem assets/tif/dem_rainier.tif --pbr --dof --dof-f-stop 2.8 --dof-focus-distance 400 --volumetrics --vol-density 0.01 --lens-vignette 0.2 --exposure 1.2

# Action shot (motion blur + lens effects)
python examples/terrain_viewer_interactive.py --dem assets/tif/dem_rainier.tif --pbr --motion-blur --mb-samples 16 --mb-cam-phi-delta 8 --lens-effects --lens-ca 0.01 --lens-vignette 0.2

# Full post-processing stack
python examples/terrain_viewer_interactive.py --dem assets/tif/dem_rainier.tif --pbr --dof --dof-f-stop 4.0 --dof-focus-distance 300 --lens-effects --lens-distortion 0.05 --lens-ca 0.01 --lens-vignette 0.25 --volumetrics --vol-mode height --vol-density 0.01 --tonemap aces --exposure 1.1
```

---

## API Reference

### IPC Command: `set_terrain_pbr`

Configure PBR rendering via IPC JSON:

```json
{
    "cmd": "set_terrain_pbr",
    "enabled": true,
    "exposure": 1.5,
    "normal_strength": 1.2,
    "ibl_intensity": 1.0,
    "shadow_technique": "pcss"
}
```

All fields except `cmd` are optional - only specified fields are updated.

### Rust Struct: `ViewerTerrainPbrConfig`

```rust
pub struct ViewerTerrainPbrConfig {
    pub enabled: bool,           // Default: false (legacy mode)
    pub exposure: f32,           // Default: 1.0
    pub normal_strength: f32,    // Default: 1.0
    pub ibl_intensity: f32,      // Default: 1.0
    pub shadow_technique: String, // Default: "none"
    pub shadow_map_res: u32,     // Default: 2048
    pub hdr_path: Option<PathBuf>,
    pub msaa: u32,               // Default: 1
}
```

---

## Technical Details

### PBR Shader Features

The PBR shader (`src/viewer/terrain/shader_pbr.rs`) provides:

#### Lighting Model
- **Blinn-Phong specular** with roughness-based shininess
- **Soft self-shadows** via normal·light smoothstep
- **Sky-gradient ambient** approximating hemisphere lighting

#### Materials
- **Height-based zones**: vegetation → rock → snow transitions
- **Slope-based rock exposure**: steep slopes reveal rock texture
- **Roughness variation**: snow is glossy, rock is matte

#### Post-Processing
- **ACES tonemapping** for HDR to LDR conversion
- **Gamma correction** (linear → sRGB)
- **Configurable exposure** for brightness control

### Known Limitations

- **No POM displacement**: Parallax occlusion mapping not implemented in viewer shader
- **16K snapshots require significant GPU memory**: ~1GB+ VRAM for render target alone

These features exist in the full `TerrainRenderer` and may be integrated in future phases.

---

## Testing

### Run the Test Suite

```bash
pytest tests/test_terrain_viewer_pbr.py -v
```

### Test Coverage

| Test | Validates |
|------|-----------|
| `test_legacy_mode_renders` | Legacy mode still works |
| `test_pbr_mode_enables` | IPC command accepted |
| `test_pbr_produces_different_output` | PBR visually differs from legacy |
| `test_pbr_exposure_affects_output` | Exposure parameter works |
| `test_pbr_can_be_disabled` | Toggle back to legacy |
| `test_height_ao_compute_pipeline_metal_compat` | Height AO R32Float/Metal compatibility |
| `test_sun_visibility_compute_pipeline_metal_compat` | Sun vis R32Float/Metal compatibility |
| `test_both_heightfield_effects_combined` | Combined heightfield effects |

---

## Implementation

### Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | IPC protocol + ViewerCmd extension | ✅ Complete |
| 2 | ViewerTerrainPbrConfig struct | ✅ Complete |
| 3 | Python CLI args + IPC commands | ✅ Complete |
| 4 | PBR shader + pipeline integration | ✅ Complete |
| 5 | End-to-end testing (5/5 passed) | ✅ Complete |
| 6 | Documentation | ✅ Complete |

### Files Modified

#### Backend (Rust)

| File | Change |
|------|--------|
| `src/viewer/terrain/shader_pbr.rs` | **NEW** - PBR WGSL shader |
| `src/viewer/terrain/pbr_renderer.rs` | **NEW** - Config struct |
| `src/viewer/terrain/mod.rs` | Module exports |
| `src/viewer/terrain/scene.rs` | PBR pipeline init |
| `src/viewer/terrain/render.rs` | Dual render path |
| `src/viewer/viewer_enums.rs` | `ViewerCmd::SetTerrainPbr` |
| `src/viewer/ipc/protocol.rs` | IPC request mapping |
| `src/viewer/cmd/handler.rs` | Command handler |

#### Frontend (Python)

| File | Change |
|------|--------|
| `examples/terrain_viewer_interactive.py` | CLI args + IPC commands |
| `tests/test_terrain_viewer_pbr.py` | **NEW** - Integration tests |

---

## Troubleshooting

### PBR mode looks the same as legacy

- Ensure `--pbr` flag is passed on command line
- Check console for `[terrain] PBR pipeline initialized` message
- Try adjusting exposure: `--pbr --exposure 2.0`

### Shader compilation error

- Run `cargo build --release --bin interactive_viewer` to rebuild
- Check for WGSL syntax errors in `src/viewer/terrain/shader_pbr.rs`

### Tests fail with "viewer binary not found"

- Build the viewer: `cargo build --release --bin interactive_viewer`
