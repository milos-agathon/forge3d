# Khumbu Smooth Orbit Light Render Design

Date: 2026-05-05

## Goal

Update `examples/khumbu_icefall_sentinel_timelapse.py` so `examples/out/khumbu_icefall_sentinel_timelapse/khumbu_icefall_sentinel_timelapse.mp4` has the same kind of smooth motion as the reference `3Drot.mp4`: a continuous camera orbit around a 3D terrain slab while dates progress over time.

The selected target is a continuous orbit at the existing export settings: 24 fps, 16 seconds, and 1600x1000 by default. The visual style should also move closer to the reference video by making the scene lighter: brighter background, less heavy shadowing, brighter terrain grade, and a clearer white/light-gray presentation while keeping enough contrast to read relief.

## Current Behavior

The current Khumbu renderer renders only unique Sentinel date keyframes with a fixed camera progress value. It then grounds those still images, blends date transitions in 2D, labels each frame, and encodes the resulting PNG sequence.

This produces date changes but not smooth camera motion. The exported video has 384 frames at 24 fps, but most frames are repeated or blended still views from the same camera angle.

## Proposed Behavior

Render each output frame through the viewer with a camera progress value derived from the frame index. The frame plan remains responsible for date selection, blend alpha, label date, and label opacity. The renderer changes from keyframe snapshots to full-frame snapshots so camera motion is present on every encoded frame.

For frames with a date transition, render both camera-matched source dates at the same camera progress and blend those two images after grounding. This preserves the existing local blend behavior without introducing parallax mismatch between dates.

The orbit should be gentle and presentation-focused:

- Use a continuous normalized progress over the full frame count.
- Drive `_terrain_state()` with a smooth/eased orbit phase instead of a fixed `0.5`.
- Keep the terrain framed as a floating slab throughout the movement.
- Avoid abrupt direction changes or loop closure jumps within the single 16 second clip.

## Light Render Style

The lighter reference look should be implemented by tuning constants already owned by the example rather than changing shared renderer behavior.

Expected adjustments:

- Use a white or near-white background instead of the current gray background.
- Lighten the base slab and contact shadow treatment so the slab still reads as 3D without looking heavy.
- Increase terrain exposure modestly and reduce overly deep shadows.
- Keep PBR relief readable by retaining ambient occlusion and directional light, but with less aggressive contrast.
- Keep labels legible on the lighter scene by preserving the existing badge structure unless tests or screenshots show it overpowering the frame.

## Architecture

The change stays within the Khumbu example and its tests.

Primary units:

- `build_frame_plan()` continues to define date cadence and transition metadata.
- A new small helper should compute camera progress for a frame index and total frame count.
- `render_frames()` should render the required per-frame source images with matching camera progress.
- `_terrain_state()` and `_terrain_pbr_state()` should remain the central places for camera and lighting parameters.
- `_ground_frame()` and label helpers continue to own post-processing.

No Rust viewer API changes are expected. The existing `viewer.send_ipc(_terrain_state(...))`, `viewer.load_overlay(...)`, and `viewer.snapshot(...)` path is sufficient.

## Data Flow

1. Build the frame plan from selected Sentinel scenes.
2. Prepare the DEM and overlay images as before.
3. For each output frame:
   - Compute normalized camera progress from the output frame index.
   - Render the primary scene at that camera progress.
   - If `blend_scene` exists, render the peer scene at the same camera progress.
   - Ground the rendered image or images.
   - Blend transition images when needed.
   - Add the date label.
   - Save `frame_%04d.png`.
4. Copy the midpoint frame to the preview path.
5. Encode the MP4 as before.

## Caching And Performance

This intentionally increases render cost because it snapshots every output frame instead of only unique date keyframes. At the default 384 frames, transition frames may require two snapshots, so the implementation should avoid unnecessary extra viewer initialization and keep all frame rendering inside one viewer session.

Disk cleanup should remain deterministic by clearing prior `frame_*.png` files and internal raw frame files before writing a new sequence.

## Error Handling

Existing dependency and ffmpeg errors remain unchanged. New helper validation should handle empty or one-frame plans without division by zero. Rendering should continue to fail loudly if viewer snapshot output is unavailable, matching current behavior.

## Testing

Focused tests should cover:

- Camera progress generation is bounded, deterministic, and smooth over multiple frames.
- `render_frames()` sends changing terrain progress values across output frames.
- Transition peer frames use the same camera progress as their matching primary frame.
- Frame count remains equal to the frame plan length.
- Light style constants and terrain state reflect the brighter render target.

The existing test that asserts a fixed camera across keyframes should be replaced because fixed camera behavior is now intentionally removed.

## Out Of Scope

- Changing Sentinel scene selection.
- Changing DEM preparation or overlay download logic.
- Adding a new Rust animation API.
- Matching the reference video's exact 10 fps or 11 second duration.
- Replacing the existing label design unless it conflicts with the lighter render after verification.
