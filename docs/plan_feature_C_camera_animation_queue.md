# Feature C: Camera Path + Keyframe Animation + Offline Render Queue

## Investigation Summary

### Camera State Storage

**Location:** `src/viewer/terrain/scene.rs::ViewerTerrainData`
```rust
pub cam_radius: f32,      // Distance from target
pub cam_phi_deg: f32,     // Azimuth (horizontal rotation)
pub cam_theta_deg: f32,   // Elevation (vertical angle)
pub cam_fov_deg: f32,     // Field of view
```

**Camera math:** `src/camera/mod.rs`
- `camera_look_at()` — view matrix from eye/target/up
- `camera_perspective()` — projection matrix
- `camera_view_proj()` — combined view-projection
- Orbit camera: phi/theta/radius → eye position computed in `src/viewer/terrain/render.rs:148-150`

### Sun/Lighting State

**Location:** `src/viewer/terrain/scene.rs::ViewerTerrainData`
```rust
pub sun_azimuth_deg: f32,
pub sun_elevation_deg: f32,
pub sun_intensity: f32,
pub ambient: f32,
```

### Render-to-Texture / Snapshot

**Snapshot:** `src/viewer/render/snapshot.rs` (if exists) or inline in render loop
- Renders to offscreen texture
- Uses `AsyncReadbackHandle` for PNG export

**Existing readback:** `src/core/async_readback.rs`
- Double-buffered async texture readback
- Supports RGBA8 and other formats

### Determinism Considerations

1. **Accumulation AA:** `src/terrain/accumulation.rs` — frame accumulation for AA
2. **Random sampling:** Some effects use frame index as seed
3. **Async GPU:** wgpu command submission is async; must sync for deterministic output

---

## Plan 1: MVP — Keyframe Camera + Offline Frame Export

### 1. Goal
Define camera keyframes with cubic interpolation, export PNG frames via existing render path. No UI; Python-driven animation definition.

### 2. Scope and Non-Goals

**In scope:**
- Camera keyframes: position (phi, theta, radius, fov) + time
- Cubic Hermite interpolation between keyframes
- Offline rendering: iterate frames, render each, save PNG
- Deterministic output (fixed seed per frame)

**Non-goals:**
- Sun/time-of-day animation
- Overlay visibility animation
- Timeline UI in viewer
- Video encoding (handled externally via ffmpeg)

### 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Python API                                  │
│  anim = CameraAnimation()                                       │
│  anim.add_keyframe(t=0, phi=0, theta=45, radius=1000, fov=60)  │
│  anim.add_keyframe(t=5, phi=180, theta=30, ...)                │
│  viewer.render_animation(anim, output_dir, fps=30)             │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                 AnimationController (Rust)                      │
│  - Keyframe storage                                             │
│  - Cubic Hermite interpolation                                  │
│  - Frame iteration (t = frame / fps)                            │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│              OfflineRenderQueue                                 │
│  - Set camera state from interpolated values                    │
│  - Render frame (sync)                                          │
│  - Readback → PNG save                                          │
│  - Progress callback                                            │
└─────────────────────────────────────────────────────────────────┘
```

**Reused:** Existing render loop, `AsyncReadbackHandle`, camera state
**New:** `CameraAnimation`, `AnimationController`, `OfflineRenderQueue`

### 4. Integration Points

| Action | File | Symbol |
|--------|------|--------|
| New | `src/animation/mod.rs` | `CameraAnimation`, `CameraKeyframe` |
| New | `src/animation/interpolation.rs` | `cubic_hermite()`, `evaluate_at()` |
| New | `src/animation/render_queue.rs` | `OfflineRenderQueue`, frame loop |
| Modify | `src/viewer/terrain/scene.rs` | Add `set_camera_state()` method |
| Modify | `src/viewer/render/main_loop.rs` | Support headless render mode |
| PyO3 | `src/lib.rs` | `CameraAnimation`, `render_animation()` |

### 5. GPU Resources and Formats

| Resource | Format | Size @ 1080p | Size @ 4K | Notes |
|----------|--------|--------------|-----------|-------|
| Render Target | Rgba8Unorm | 8.3 MiB | 33 MiB | Existing |
| Readback Buffer | MAP_READ | 8.3 MiB | 33 MiB | Per-frame readback |
| PNG Output | Disk | ~500 KiB/frame | ~2 MiB/frame | Compressed |

**No additional VRAM** — uses existing render targets

### 6. Shader Changes (WGSL)

**None required** — animation is camera state manipulation only.

**Determinism:** Set `frame_index` uniform to animation frame number for reproducible sampling:
```rust
// In render loop
uniforms.frame_index = animation_frame;  // Not wall-clock frame
```

### 7. User-Facing API

**Python:**
```python
from forge3d import CameraAnimation

# Define animation
anim = CameraAnimation()
anim.add_keyframe(time=0.0, phi=0, theta=45, radius=5000, fov=60)
anim.add_keyframe(time=2.5, phi=90, theta=30, radius=3000, fov=60)
anim.add_keyframe(time=5.0, phi=180, theta=45, radius=5000, fov=60)

# Render to frames
viewer.render_animation(
    animation=anim,
    output_dir="./frames",
    fps=30,
    width=1920,
    height=1080,
    progress_callback=lambda frame, total: print(f"{frame}/{total}")
)

# Optional: invoke ffmpeg
import subprocess
subprocess.run([
    "ffmpeg", "-framerate", "30",
    "-i", "./frames/frame_%04d.png",
    "-c:v", "libx264", "-pix_fmt", "yuv420p",
    "output.mp4"
])
```

**Config keys:**
- `animation.default_fps` (int, default: `30`)
- `animation.accumulation_frames` (int, default: `1`) — for AA

### 8. Quality & Determinism

- **Interpolation:** Cubic Hermite provides smooth acceleration/deceleration
- **Determinism:** Fixed `frame_index` seed; no time-based randomness
- **Frame consistency:** Each frame rendered independently; no temporal artifacts
- **Failure modes:**
  - Disk full → stop with error, report last successful frame
  - Keyframe gap → linear interpolation warning

### 9. Validation & Tests

**Test:** `tests/test_animation_mvp.py`
```python
def test_keyframe_interpolation():
    """Interpolation produces correct intermediate values."""
    anim = CameraAnimation()
    anim.add_keyframe(time=0.0, phi=0, theta=45, radius=1000, fov=60)
    anim.add_keyframe(time=1.0, phi=90, theta=45, radius=1000, fov=60)
    
    state = anim.evaluate(0.5)
    assert abs(state.phi - 45.0) < 0.1  # Midpoint
    
def test_frame_export():
    """Frames exported to disk with correct naming."""
    anim = CameraAnimation()
    anim.add_keyframe(time=0.0, phi=0, ...)
    anim.add_keyframe(time=0.1, phi=10, ...)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        viewer.render_animation(anim, tmpdir, fps=30)
        frames = list(Path(tmpdir).glob("*.png"))
        assert len(frames) == 4  # 0.1s * 30fps + 1

def test_determinism():
    """Same animation produces identical frames."""
    anim = CameraAnimation()
    anim.add_keyframe(...)
    
    viewer.render_animation(anim, "./run1", fps=30)
    viewer.render_animation(anim, "./run2", fps=30)
    
    for f1, f2 in zip(sorted(Path("./run1").glob("*.png")), 
                      sorted(Path("./run2").glob("*.png"))):
        assert file_hash(f1) == file_hash(f2)
```

**Commands:**
```bash
python -m pytest tests/test_animation_mvp.py -v
```

### 10. Milestones & Deliverables

| # | Name | Files | Deliverables | Acceptance | Risks |
|---|------|-------|--------------|------------|-------|
| M1 | Keyframe Storage | `src/animation/mod.rs` | `CameraKeyframe` struct, storage | Keyframes stored | None |
| M2 | Cubic Interpolation | `src/animation/interpolation.rs` | `evaluate_at()` | Smooth interpolation | Tangent computation |
| M3 | Render Loop | `src/animation/render_queue.rs` | Frame iteration | All frames rendered | Memory per frame |
| M4 | PNG Export | `src/animation/render_queue.rs` | Async save | Files on disk | I/O bottleneck |
| M5 | Determinism | `src/viewer/render/main_loop.rs` | Fixed frame seed | Identical reruns | Async timing |
| M6 | Python API | `src/lib.rs` | `CameraAnimation` class | Test passes | API ergonomics |

---

## Plan 2: Standard — Timeline UI + Render Queue with Progress

### 1. Goal
Add timeline scrubbing UI in viewer, keyframe editing, render queue with progress reporting, and deterministic sampling settings for production renders.

### 2. Scope and Non-Goals

**In scope:**
- Timeline bar in viewer (egui or custom)
- Keyframe markers, drag-to-edit
- Playback controls (play/pause/scrub)
- Render queue with progress bar
- Sun/time-of-day keyframing
- Deterministic accumulation (multi-sample AA)

**Non-goals:**
- Complex easing curves (beyond cubic)
- Overlay visibility keyframing
- Video encoding in Rust
- Resumable renders

### 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Timeline UI (egui)                          │
│  [|====●====●====●====|------>]  [▶] [⏸] [⏹]                   │
│   0s   2s   4s   6s   8s       time                             │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                 AnimationTimeline                               │
│  - Keyframe tracks (camera, sun)                                │
│  - Current time cursor                                          │
│  - Playback state machine                                       │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│              ProductionRenderQueue                              │
│  - Multi-sample accumulation per frame                          │
│  - Progress callback with ETA                                   │
│  - Cancel support                                               │
└─────────────────────────────────────────────────────────────────┘
```

**Reused:** Plan 1 interpolation, render queue base
**New:** Timeline UI, keyframe tracks, accumulation rendering

### 4. Integration Points

| Action | File | Symbol |
|--------|------|--------|
| New | `src/animation/timeline.rs` | `AnimationTimeline`, `Track<T>` |
| New | `src/animation/tracks/camera.rs` | `CameraTrack` |
| New | `src/animation/tracks/sun.rs` | `SunTrack` |
| New | `src/viewer/ui/timeline.rs` | `TimelineWidget` (egui) |
| Modify | `src/viewer/hud.rs` | Add timeline to HUD |
| New | `src/animation/production_queue.rs` | Accumulation render |

### 5. GPU Resources and Formats

| Resource | Format | Size @ 1080p | Size @ 4K | Notes |
|----------|--------|--------------|-----------|-------|
| Accumulation Buffer | Rgba32Float | 33 MiB | 132 MiB | High-precision accumulation |
| Sample Counter | R32Uint | 8.3 MiB | 33 MiB | Per-pixel sample count |

**Additional VRAM:** ~41 MiB @ 1080p for accumulation

### 6. Shader Changes (WGSL)

**Accumulation shader** (may already exist in `src/terrain/accumulation.rs`):
```wgsl
@compute @workgroup_size(8, 8)
fn cs_accumulate(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let coord = gid.xy;
    let current = textureLoad(current_frame, coord, 0);
    let accum = textureLoad(accumulation_buffer, coord, 0);
    let count = textureLoad(sample_count, coord, 0).r;
    
    let new_accum = accum + current;
    let new_count = count + 1.0;
    
    textureStore(accumulation_buffer, coord, new_accum);
    textureStore(sample_count, coord, vec4(new_count, 0.0, 0.0, 0.0));
    
    // Final output = accum / count
    let final_color = new_accum / new_count;
    textureStore(output, coord, final_color);
}
```

### 7. User-Facing API

**Python:**
```python
# Create timeline with multiple tracks
timeline = AnimationTimeline(duration=10.0)

# Camera track
timeline.camera.add_keyframe(t=0, phi=0, theta=45, radius=5000)
timeline.camera.add_keyframe(t=5, phi=180, theta=30, radius=3000)
timeline.camera.add_keyframe(t=10, phi=360, theta=45, radius=5000)

# Sun track (time-of-day)
timeline.sun.add_keyframe(t=0, azimuth=90, elevation=30)  # Morning
timeline.sun.add_keyframe(t=5, azimuth=180, elevation=60)  # Noon
timeline.sun.add_keyframe(t=10, azimuth=270, elevation=30)  # Evening

# Show timeline UI
viewer.show_timeline(timeline)

# Production render with accumulation
viewer.render_production(
    timeline=timeline,
    output_dir="./frames",
    fps=30,
    samples_per_frame=16,  # Accumulation samples
    progress_callback=lambda p: print(f"{p.frame}/{p.total} ETA: {p.eta}")
)
```

**Viewer commands (IPC):**
```
timeline play
timeline pause
timeline seek 5.0
timeline add_keyframe camera 5.0 phi=90
```

### 8. Quality & Determinism

- **Accumulation:** N samples per frame, averaged for high-quality AA
- **Sample seeds:** Frame index × sample index for unique seeds
- **Playback:** Real-time preview uses single sample; production uses N
- **Cancel support:** Check cancel flag between samples

### 9. Validation & Tests

**Test:** `tests/test_animation_standard.py`
```python
def test_timeline_playback():
    """Timeline playback updates camera in real-time."""
    timeline = AnimationTimeline(duration=1.0)
    timeline.camera.add_keyframe(t=0, phi=0, ...)
    timeline.camera.add_keyframe(t=1, phi=90, ...)
    
    viewer.show_timeline(timeline)
    viewer.play_timeline()
    time.sleep(0.5)
    
    state = viewer.get_camera_state()
    assert 40 < state.phi < 50  # Approximately midpoint

def test_sun_track():
    """Sun keyframes interpolate correctly."""
    timeline = AnimationTimeline(duration=2.0)
    timeline.sun.add_keyframe(t=0, azimuth=0, elevation=30)
    timeline.sun.add_keyframe(t=2, azimuth=180, elevation=60)
    
    state = timeline.evaluate(1.0)
    assert abs(state.sun_azimuth - 90) < 1

def test_accumulation_quality():
    """More samples produces smoother output."""
    timeline = create_simple_timeline()
    
    viewer.render_production(timeline, "./1sample", samples_per_frame=1)
    viewer.render_production(timeline, "./16sample", samples_per_frame=16)
    
    # 16-sample should have less noise
    noise_1 = measure_edge_noise("./1sample/frame_0000.png")
    noise_16 = measure_edge_noise("./16sample/frame_0000.png")
    assert noise_16 < noise_1 * 0.5
```

### 10. Milestones & Deliverables

| # | Name | Files | Deliverables | Acceptance | Risks |
|---|------|-------|--------------|------------|-------|
| M1 | Track System | `src/animation/timeline.rs` | Generic `Track<T>` | Multiple tracks work | Type complexity |
| M2 | Sun Track | `src/animation/tracks/sun.rs` | Sun keyframes | Lighting animates | Color interpolation |
| M3 | Timeline UI | `src/viewer/ui/timeline.rs` | egui widget | Scrubbing works | egui integration |
| M4 | Playback Engine | `src/animation/playback.rs` | Play/pause/seek | Real-time preview | Frame timing |
| M5 | Accumulation | `src/animation/production_queue.rs` | Multi-sample | Quality improves | Memory for buffer |
| M6 | Progress/Cancel | `src/animation/production_queue.rs` | ETA, cancel | Responsive UI | Async cancel |
| M7 | IPC Commands | `src/viewer/ipc/protocol.rs` | Timeline commands | Python control | Command parsing |

---

## Plan 3: Premium — Scene Scripting + Batch Jobs + Resumable Renders

### 1. Goal
Full animation system with scriptable scene parameters (camera, sun, overlays, labels), batch job queue, resumable renders, and optional video assembly integration.

### 2. Scope and Non-Goals

**In scope:**
- Scene scripting language (JSON or YAML-based)
- Animated parameters: camera, sun, overlay opacity, label visibility, fog
- Batch job queue with multiple animations
- Resumable renders (checkpoint to disk)
- Optional ffmpeg integration for video assembly
- Render presets (draft, preview, production)

**Non-goals:**
- Node-based animation graph
- Physics-based camera (smooth follow, etc.)
- Real-time video streaming

### 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                  Scene Script (JSON/YAML)                       │
│  scenes:                                                        │
│    - name: "flyover"                                            │
│      duration: 30                                               │
│      tracks:                                                    │
│        camera: [{t: 0, ...}, {t: 15, ...}, {t: 30, ...}]       │
│        sun: [{t: 0, azimuth: 90}, {t: 30, azimuth: 270}]       │
│        fog: [{t: 0, density: 0}, {t: 15, density: 0.01}]       │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                  SceneScriptEngine                              │
│  - Parse script                                                 │
│  - Validate tracks                                              │
│  - Generate AnimationTimeline                                   │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                  BatchJobQueue                                  │
│  - Multiple animations queued                                   │
│  - Priority scheduling                                          │
│  - Checkpoint/resume per job                                    │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│              ResumableRenderJob                                 │
│  - Checkpoint file (last completed frame)                       │
│  - Resume from checkpoint                                       │
│  - Optional video assembly                                      │
└─────────────────────────────────────────────────────────────────┘
```

### 4. Integration Points

| Action | File | Symbol |
|--------|------|--------|
| New | `src/animation/script.rs` | `SceneScript`, `parse_script()` |
| New | `src/animation/tracks/fog.rs` | `FogTrack` |
| New | `src/animation/tracks/overlay.rs` | `OverlayTrack` (opacity, visibility) |
| New | `src/animation/tracks/label.rs` | `LabelTrack` (density, visibility) |
| New | `src/animation/batch.rs` | `BatchJobQueue`, `RenderJob` |
| New | `src/animation/checkpoint.rs` | `Checkpoint`, save/load |
| New | `src/animation/video.rs` | `VideoAssembler` (ffmpeg wrapper) |

### 5. GPU Resources and Formats

Same as Plan 2, plus:

| Resource | Format | Size | Notes |
|----------|--------|------|-------|
| Checkpoint File | JSON + frame index | ~1 KiB | On disk |
| Job Queue State | Memory | ~10 KiB/job | In-memory |

### 6. Shader Changes (WGSL)

**No new shaders** — fog, overlay opacity already controllable via uniforms.

**Animated uniforms:**
- `fog_uniforms.fog_density` — animatable
- `overlay_uniforms.opacity` — per-layer animatable
- `label_uniforms.density` — label density factor

### 7. User-Facing API

**Scene script (YAML):**
```yaml
# animation_script.yaml
name: "Mt. Rainier Flyover"
duration: 30.0
fps: 30
output:
  directory: "./renders/rainier"
  format: "png"
  video: true  # Assemble video after

tracks:
  camera:
    - time: 0
      phi: 0
      theta: 60
      radius: 10000
      fov: 60
    - time: 15
      phi: 180
      theta: 30
      radius: 5000
      fov: 45
    - time: 30
      phi: 360
      theta: 60
      radius: 10000
      fov: 60

  sun:
    - time: 0
      azimuth: 90
      elevation: 20
      intensity: 0.8
    - time: 30
      azimuth: 270
      elevation: 45
      intensity: 1.0

  fog:
    - time: 0
      density: 0.0
    - time: 15
      density: 0.005
    - time: 30
      density: 0.0

  overlay:
    layer: "roads"
    keyframes:
      - time: 0
        opacity: 0.0
      - time: 5
        opacity: 1.0
      - time: 25
        opacity: 1.0
      - time: 30
        opacity: 0.0

presets:
  draft:
    width: 960
    height: 540
    samples: 1
  preview:
    width: 1920
    height: 1080
    samples: 4
  production:
    width: 3840
    height: 2160
    samples: 32
```

**Python:**
```python
from forge3d.animation import SceneScript, BatchQueue

# Load and validate script
script = SceneScript.load("animation_script.yaml")
script.validate()

# Render with preset
script.render(preset="production", progress=print_progress)

# Batch queue
queue = BatchQueue()
queue.add(script1, priority=1)
queue.add(script2, priority=2)
queue.add(script3, priority=1)

# Start batch processing
queue.start(
    workers=1,  # Single GPU
    on_job_complete=lambda job: print(f"Completed: {job.name}"),
    on_error=lambda job, err: print(f"Failed: {job.name}: {err}")
)

# Resume interrupted render
script.render(preset="production", resume=True)  # Reads checkpoint
```

**CLI (optional):**
```bash
forge3d-render animation_script.yaml --preset production --resume
forge3d-render batch_manifest.yaml --workers 1
```

### 8. Quality & Determinism

- **Scripted reproducibility:** Same script + preset = identical output
- **Checkpoint format:** JSON with frame index, RNG state
- **Resume accuracy:** Restarts from last incomplete frame
- **Video assembly:** ffmpeg invoked with exact parameters

### 9. Validation & Tests

**Test:** `tests/test_animation_premium.py`
```python
def test_script_parsing():
    """Script parses and validates correctly."""
    script = SceneScript.load("test_script.yaml")
    assert script.duration == 10.0
    assert len(script.tracks["camera"]) == 3
    assert script.tracks["fog"][0].density == 0.0

def test_fog_animation():
    """Fog density animates over time."""
    script = create_fog_animation_script()
    
    # Render single frame at t=15 (peak fog)
    viewer.render_frame_at(script, t=15.0, output="fog_peak.png")
    
    # Check fog is visible
    img = load_image("fog_peak.png")
    assert image_has_fog(img)  # Reduced contrast, blue shift

def test_batch_queue():
    """Batch queue processes multiple jobs."""
    queue = BatchQueue()
    queue.add(script1)
    queue.add(script2)
    
    completed = []
    queue.start(on_job_complete=lambda j: completed.append(j.name))
    queue.wait()
    
    assert set(completed) == {"script1", "script2"}

def test_resume_render():
    """Interrupted render resumes from checkpoint."""
    script = create_long_animation()
    
    # Simulate interruption at frame 50
    script.render(preset="draft", max_frames=50)
    
    # Resume
    script.render(preset="draft", resume=True)
    
    # All frames should exist
    frames = list(Path(script.output_dir).glob("*.png"))
    assert len(frames) == script.total_frames

def test_video_assembly():
    """Video assembled from frames."""
    script = create_short_animation()
    script.render(preset="draft")
    
    assert Path(script.output_dir / "output.mp4").exists()
    # Check video duration matches
    duration = get_video_duration(script.output_dir / "output.mp4")
    assert abs(duration - script.duration) < 0.1
```

### 10. Milestones & Deliverables

| # | Name | Files | Deliverables | Acceptance | Risks |
|---|------|-------|--------------|------------|-------|
| M1 | Script Parser | `src/animation/script.rs` | YAML/JSON parsing | Scripts load | Schema validation |
| M2 | Fog Track | `src/animation/tracks/fog.rs` | Fog animation | Fog changes over time | Uniform binding |
| M3 | Overlay Track | `src/animation/tracks/overlay.rs` | Opacity animation | Layers fade | Per-layer state |
| M4 | Batch Queue | `src/animation/batch.rs` | Job queue | Multiple jobs run | Priority scheduling |
| M5 | Checkpointing | `src/animation/checkpoint.rs` | Save/load state | Resume works | State serialization |
| M6 | Video Assembly | `src/animation/video.rs` | ffmpeg wrapper | MP4 created | ffmpeg availability |
| M7 | Render Presets | `src/animation/presets.rs` | Draft/preview/prod | Quality matches | Preset validation |

---

## Determinism Strategy

### Sources of Non-Determinism

1. **Frame timing:** Wall-clock vs animation time
2. **Random sampling:** TAA, DoF, SSAO jitter
3. **Async GPU:** Command buffer submission order
4. **Floating-point:** GPU FP precision varies

### Mitigation

1. **Fixed frame index:** Use `animation_frame` not `wall_clock_frame` for seeds
2. **Explicit seeds:** `seed = animation_frame * 1000 + sample_index`
3. **Sync rendering:** `device.poll(wgpu::Maintain::Wait)` between frames
4. **Accumulation reset:** Clear accumulation buffer at each new frame

```rust
fn render_frame_deterministic(frame: u32, samples: u32) {
    // Reset accumulation
    clear_accumulation_buffer();
    
    for sample in 0..samples {
        // Deterministic seed
        let seed = frame * 10000 + sample;
        set_jitter_seed(seed);
        
        // Render and accumulate
        render_sample();
        accumulate();
    }
    
    // Finalize
    divide_by_sample_count();
    
    // Sync before readback
    device.poll(wgpu::Maintain::Wait);
}
```

---

## Recommendation

**Start with Plan 1 (MVP)** for basic animation export (~1 week). This enables simple flyover videos with Python scripting. Upgrade to Plan 2 for interactive timeline UI and production quality (accumulation). Plan 3 is for automated rendering pipelines (batch processing, overnight renders).

**Key consideration:** Video encoding is best handled by ffmpeg externally — Rust video encoding adds significant complexity with minimal benefit.
