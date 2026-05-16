# Vancouver AIS Showcase Video Design

## Goal

Create a self-contained synthetic Vancouver-style AIS flyover video using forge3d. The output should match the supplied reference video's format and visual language: portrait 720x900, 30 fps, about 28 seconds, silent MP4, oblique coastal terrain, glowing AIS tracks, small labels, date/time HUD, and attribution text.

The demo must not depend on external terrain, basemap, or AIS downloads. It should be reproducible from repository code and generated data.

## Chosen Approach

Use a hybrid render/compositor pipeline:

1. forge3d renders the 3D map base from a generated coastal heightmap and generated albedo overlay.
2. Python/Pillow composites the stylized AIS and HUD layers on each snapshot.
3. ffmpeg encodes a silent H.264 MP4 from the rendered frame sequence.

This follows the existing example pattern used by `examples/pnoa_river_showcase_video.py`, while tailoring the generated data and post layer to the Vancouver AIS reference.

## Scene Design

The synthetic map should evoke Vancouver rather than reproduce real geography. The generated scene includes:

- a broad harbor/strait basin
- river-like and fjord-like water corridors
- islands and peninsulas
- a dense city-grid texture in the flatter lower portion
- steep green-gray mountain ridges in the upper portion
- muted satellite/map-like terrain coloring

The camera starts over the city/harbor and eases northward and upward toward mountain inlets. Motion should be smooth and slow, with the same oblique-map feeling as the reference.

## AIS And HUD Design

Synthetic vessel paths are deterministic. The overlay should include:

- cyan harbor traffic routes
- blue long-haul tracks through inlets and channels
- magenta port/activity clusters
- track reveal over time
- brighter moving vessel heads
- small rounded white vessel labels
- subtle place labels
- top-center date/time HUD progressing from `01 Apr 2026` to `02 Apr 2026`
- lower-left attribution/credit text
- glow, haze, and vignette post-processing

The output is silent. No audio stream is required.

## Script Interface

Add an example script, expected path:

```text
examples/vancouver_ais_showcase_video.py
```

CLI defaults:

- `--output`: `examples/out/vancouver_ais_showcase/vancouver_ais_showcase.mp4`
- `--preview`: preview PNG next to the output
- `--preview-only`: render one frame and skip MP4 encoding
- `--fps`: default `30`
- `--duration`: default `27.6` seconds
- `--width`: default `720`
- `--height`: default `900`

The script should print the DEM/preview/MP4 paths it creates, matching the style of existing video examples.

## Components

- Synthetic terrain generator: produces a float32 heightmap and a companion RGBA albedo overlay.
- Scene configuration builder: computes camera, target, sun, z-scale, and render settings from the generated terrain.
- AIS generator: returns deterministic paths, labels, colors, and timing metadata.
- Frame compositor: takes a forge3d snapshot and overlays tracks, vessels, labels, HUD, glow, and vignette.
- Encoder helper: invokes ffmpeg for silent H.264 MP4 output.
- Preview/full render orchestration: runs a forge3d viewer session, snapshots frames, composites them, and encodes when requested.

## Testing

Tests should avoid requiring a GPU or launching the viewer. Cover:

- deterministic terrain and AIS generation for a fixed seed
- frame plan timing and HUD date progression
- preview-only behavior using a fake viewer
- ffmpeg command construction for silent MP4 output
- basic image compositor behavior on synthetic input

Full video rendering remains an example/runtime path because it depends on the viewer binary, WebGPU, and ffmpeg.

## Non-Goals

- Real Vancouver DEM, basemap, or AIS ingestion
- Exact geographic correctness
- Audio generation
- Pixel-golden comparison against the reference video
- Full use of forge3d vector/label IPC for every stylized overlay element

## Open Decisions

No open decisions remain from brainstorming. The selected scope is a self-contained synthetic Vancouver-style demo using the hybrid forge3d snapshot plus Python compositor approach.
