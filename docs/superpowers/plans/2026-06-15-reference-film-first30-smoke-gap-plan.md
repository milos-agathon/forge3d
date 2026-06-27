# Reference Film First-30s Smoke Gap Plan

**Date:** 2026-06-15  
**Scope:** Gaps found by checking every frame in the first 30 seconds of `rapidsave.com_oc_heat_and_smoke_the_recordbreaking_2023-pezptq0xr7ub1.mp4`.
**Primary file:** `examples/california_cigar_smoke_demo.py`
**Requirements source:** `docs/superpowers/specs/2026-06-12-cigar-smoke-source-wisp-requirements.md`

## Evidence

- Reference metadata: 1920x1080 H.264, 30 fps; first 30 seconds = 900 frames.
- Every-frame extraction was generated at 480 px analysis width.
- Summary artifacts retained in `examples/out/california_cigar_smoke/reference_first30_every_frame_audit/`:
  - `reference_first30_all_900_frames_micro_contact.jpg`
  - `reference_first30_half_second_contact.jpg`
  - `generated_first30_half_second_contact.jpg`
  - `first30_frame_metrics.json`

Key measured gap:

- Reference median smoke-mask coverage: about `0.062`.
- Current generated median smoke-mask coverage: about `0.116`, roughly `1.9x` the reference.
- Reference p90 frame-to-frame luma delta: about `1.94`.
- Current generated p90 frame-to-frame luma delta: about `0.90`, less than half the reference.

## Exact-Match Constraint

Do not treat parameter tuning as a path to `100%` resemblance. A procedural smoke system can become visually similar, but it cannot be identical to the first 900 encoded reference frames unless the reference smoke itself is decoded, isolated, represented as frame-indexed data, and replayed or reconstructed with bounded per-frame error.

There must be two explicitly separate targets:

- `reference-exact-smoke`: a reference-derived smoke playback/reconstruction mode whose purpose is identical first-30s resemblance to the smoke effect.
- `reference-procedural-smoke`: a generative approximation mode that may borrow the reference's event timing, coverage envelope, and plume morphology, but must not be described as 100% identical.

All tasks below are required for the exact-match target. The procedural target can reuse the analysis products, but it is not sufficient for the user's stated 100% resemblance goal.

## Exact-Match Task Plan

### A. Lock The Reference Artifact

- Compute and persist SHA-256 for `rapidsave.com_oc_heat_and_smoke_the_recordbreaking_2023-pezptq0xr7ub1.mp4`.
- Persist `ffprobe` metadata for the exact source video: codec, width, height, frame rate, duration, frame count, pixel format, color tags, and bit rate.
- Create `examples/out/california_cigar_smoke/reference_first30_every_frame_audit/reference_exact_manifest.json`.
- Manifest fields must include:
  - `reference_video_path`
  - `reference_sha256`
  - `start_frame: 0`
  - `frame_count: 900`
  - `fps: 30`
  - `width: 1920`
  - `height: 1080`
  - `decode_command`
  - `color_policy`
  - `generated_at`
  - `artifact_schema_version`
- Acceptance: fail if the video SHA-256, resolution, fps, or frame count differs from the manifest.

### B. Decode Native Reference Frames Losslessly

- Decode frames 0-899 at native 1920x1080 into a lossless cache, not scaled JPEG.
- Required cache path: `examples/.cache/california_cigar_smoke/reference_exact/frames/ref_%04d.png`.
- Use deterministic ffmpeg flags and record them in the manifest.
- Generate a native-resolution frame list with per-frame SHA-256 values.
- Acceptance: exactly 900 PNG frames exist; every frame has dimensions 1920x1080; every frame hash is recorded; re-decoding produces identical hashes.

### C. Build Non-Smoke Exclusion Masks

- Create a static UI/text exclusion mask for:
  - top-left attribution/source label
  - top-right date label
  - lower-left burned-area label
  - any frame-edge letterbox or compression border
- Create a dynamic fire/core exclusion mask for orange/yellow/white hot pixels and their immediate bloom support.
- Create a coastline/water/background stability mask so the persistent west-coast blue gradient is not classified as smoke.
- Save masks under `examples/.cache/california_cigar_smoke/reference_exact/masks/`.
- Required mask artifacts:
  - `ui_exclusion_mask.png`
  - `static_background_valid_mask.png`
  - `fire_mask_%04d.png`
  - `candidate_smoke_domain_%04d.png`
- Acceptance: every extracted smoke matte must prove that UI/date/area text and fire hot cores are excluded from smoke statistics and from smoke playback alpha.

### D. Derive A Clean Reference Background

- Estimate a smoke-free background plate at 1920x1080 using the first 900 frames and exclusion masks.
- Use temporal dark-percentile/background modeling per pixel, not a single first frame, because the reference already contains smoke in early frames.
- Save:
  - `reference_background_clean.png`
  - `reference_background_confidence.png`
  - `background_residual_audit_sheet.jpg`
- Acceptance: subtracting `reference_background_clean.png` from low-smoke frames must not produce persistent false smoke over ocean gradient, terrain relief, labels, or lakes.

### E. Extract Per-Frame Smoke RGBA

- For each frame 0-899, compute a smoke alpha matte and observed smoke color layer at native resolution.
- Use low-saturation positive luminance residuals from the clean background, temporal consistency, local texture, and fire/UI exclusions.
- Preserve bright smoke pulses. Do not clamp all smoke to low alpha; reference peak events contain localized high-opacity white/gray smoke.
- Save:
  - `smoke_alpha_%04d.png` as 8-bit alpha
  - `smoke_rgb_%04d.png` as observed unpremultiplied RGB
  - `smoke_rgba_%04d.png` as premultiplied/replay-ready RGBA
  - `smoke_confidence_%04d.png`
  - `smoke_overlay_audit_%04d.jpg`
- Acceptance: the extracted smoke layer must reconstruct the reference smoke over the clean background with low error in all 900 frames before it is used by the renderer.

### F. Manual Correction Queue For Low-Confidence Frames

- Automatically flag frames where any of these are true:
  - smoke reconstruction MAE exceeds the acceptance threshold
  - UI/fire leakage appears in the smoke matte
  - large transparent holes appear inside dense smoke
  - false smoke appears over stable water/background regions
  - event-boundary frames lose smoke continuity
- Save `manual_correction_queue.json` with frame index, reason, metric values, and thumbnail path.
- Provide corrected matte support:
  - `smoke_alpha_corrected_%04d.png`
  - `smoke_rgb_corrected_%04d.png`
  - `correction_notes.json`
- Acceptance: exact-match mode cannot be accepted until the correction queue is empty or every queued frame has an explicit approved correction note.

### G. Transcribe Reference Event Timing

- Detect and save smoke event boundaries from all 900 frames, using frame-difference spikes, smoke-coverage peaks/troughs, centroid jumps, and date-label changes.
- Required output: `reference_smoke_events.json`.
- Each event must include:
  - `event_id`
  - `start_frame`
  - `peak_frame`
  - `end_frame`
  - `date_label`
  - `coverage_peak`
  - `centroid_path`
  - `dominant_axis_degrees`
  - `notes`
- Seed known spike clusters from the all-frame audit: approximately 5-7s, 13-15s, 24-26s, and 27s, then refine from native-frame metrics.
- Acceptance: event boundaries must explain the first-30s smoke cadence; a continuously drifting field must fail this task.

### H. Implement Reference Smoke Playback In The Demo

- Add a dedicated render preset, e.g. `reference-exact-smoke`.
- Add CLI options:
  - `--reference-smoke-cache`
  - `--reference-smoke-mode exact|procedural`
  - `--reference-smoke-start-frame`
  - `--reference-smoke-frame-count`
- Implement `reference_smoke_rgba(frame_index, output_size, cache_dir)` in `examples/california_cigar_smoke_demo.py`.
- In exact mode, composite smoke as a screen-space layer after terrain/background but before fire/text unless the audit proves another order matches the reference better.
- Do not warp exact reference smoke through the local map plate. Native reference smoke is already screen-space and must not be distorted by `warp_map_layer_to_plate()`.
- Acceptance: exact mode uses `smoke_rgba_%04d.png` directly for the corresponding frame index, with deterministic nearest-frame mapping at 30 fps.

### I. Match Reference Geography And Labels For Exact Mode

- Exact smoke playback must use a North America/reference-scale background and label semantics, not the California August Complex map.
- The top-right date label must match the decoded reference date for each frame range.
- The lower-left burned-area label must match the decoded reference scale and units, including the transition from `k ha` to `M ha`.
- Active fire points must appear/disappear at the reference geography scale; they may be reconstructed from the reference fire mask for exact mode.
- Acceptance: all 900 generated exact-mode frames must have the same date/area semantics and continent-scale composition as the reference before smoke similarity is scored.

### J. Add Exact Smoke Metrics

- Add all-frame metrics under `reference_exact_smoke_gate_report`.
- Required per-frame metrics:
  - `smoke_mask_iou`
  - `smoke_alpha_mae`
  - `smoke_rgb_mae`
  - `smoke_coverage_absolute_error`
  - `smoke_centroid_error_px`
  - `smoke_principal_axis_error_degrees`
  - `frame_delta_error`
  - `event_id_match`
  - `ui_leakage_fraction`
  - `fire_leakage_fraction`
  - `background_false_positive_fraction`
- Required sequence metrics:
  - `median_smoke_mask_iou`
  - `minimum_smoke_mask_iou`
  - `median_alpha_mae`
  - `maximum_alpha_mae`
  - `coverage_curve_correlation`
  - `frame_delta_curve_correlation`
  - `event_boundary_frame_error_max`
  - `all_frame_count`
- Acceptance thresholds for exact mode:
  - `all_frame_count == 900`
  - `minimum_smoke_mask_iou >= 0.985`
  - `median_smoke_mask_iou >= 0.995`
  - `maximum_alpha_mae <= 3.0` on the smoke union
  - `median_alpha_mae <= 1.0` on the smoke union
  - `maximum_smoke_rgb_mae <= 4.0` on smoke pixels
  - `median_smoke_rgb_mae <= 1.5` on smoke pixels
  - `maximum_smoke_centroid_error_px <= 2.0`
  - `coverage_curve_correlation >= 0.995`
  - `frame_delta_curve_correlation >= 0.990`
  - `event_boundary_frame_error_max <= 1`
  - `ui_leakage_fraction == 0.0`
  - `fire_leakage_fraction <= 0.002`
  - `background_false_positive_fraction <= 0.002`

### K. Add Exact Audit Artifacts

- Save all artifacts in `examples/out/california_cigar_smoke/reference_exact_smoke_audit/`.
- Required artifacts:
  - `reference_exact_manifest.json`
  - `reference_smoke_events.json`
  - `reference_exact_smoke_gate_report.json`
  - `reference_exact_all_900_frames_micro_contact.jpg`
  - `generated_exact_all_900_frames_micro_contact.jpg`
  - `smoke_difference_all_900_frames_micro_contact.jpg`
  - `reference_exact_half_second_contact.jpg`
  - `generated_exact_half_second_contact.jpg`
  - `smoke_difference_half_second_contact.jpg`
  - `worst_50_smoke_error_frames/`
- Acceptance: a reviewer must be able to inspect the worst 50 frames without re-running extraction.

### L. Keep Procedural Work Honest

- If implementing a procedural approximation after exact playback, it must use the exact audit as target data.
- It must fit event timing, coverage curves, centroid paths, and plume-axis changes from `reference_smoke_events.json`.
- It must remain labeled as approximate unless it passes the exact metrics above.
- Acceptance: any procedural-only candidate that fails exact metrics cannot be described as 100% resemblance, even if it looks close in sparse contact sheets.

## Missing Plan Items

### 1. Reference-2023 Continent Mode

- Add a mode distinct from the local August Complex/California target.
- Use a North America map extent comparable to the reference, not the regional California crop.
- Use 2023 date labels and burned-area scale matching the reference first 30 seconds, which reaches about `12.3 M ha`.
- Treat the current August Complex 2020 mode as a separate local demo, not the reference-film match.

### 2. Daily Or Keyframed Smoke-State Driver

- Implement reference-film smoke as a sequence of daily/event smoke states.
- Support abrupt or eased plume-state replacement at daily cadence.
- Preserve temporal interpolation within a day, but allow large smoke fields to appear, curl, detach, and be replaced.
- Add all-frame gates for frame-difference spikes, smoke-coverage peaks/troughs, and component persistence.

### 3. Observed-Smoke Primitive

- Make observed or observed-like smoke masks a first-class source, not a future optional replacement.
- Accept NOAA HMS smoke polygons, HRRR-Smoke rasters, satellite aerosol/smoke products, or curated reference-derived masks.
- Use procedural advection and texture for interpolation/detail, not as the only source of truth.

### 4. Per-Event Regional Plumes

- Replace the single persistent regional texture field with independent plume events.
- Each plume event needs birth/death time, wind vector, curl/roll-up, opacity envelope, and dissipation.
- Permit multiple simultaneous plumes with different directions and scales.
- Keep source-attached wisps for local fire clusters, but do not force broad regional smoke to attach to every fire point.

### 5. Anti-Contour Acceptance Gates

- Add metrics that reject contour/isoline smoke bands, nested rings, and smooth topographic-looking level sets.
- The current axis-band and texture gates are insufficient because the generated candidate can still read as contour smoke.
- Include visual audit frames around detected contour/ring peaks.

### 6. Fire Distribution And Decoupling

- Add continent-scale active-fire cluster appearance/disappearance over time.
- Allow many fire points to have no visible plume in a frame.
- Allow transported smoke fields to exist away from visible hot cores.
- Keep fire and smoke correlated at event scale, not one-to-one at every point.

### 7. Every-Frame Acceptance Artifact

- Generate all 900 first-30s reference frames and generated frames at analysis resolution for each candidate.
- Save a micro contact sheet with every frame.
- Save 0.5s or 0.2s larger contact sheets.
- Save frame-difference spike frames and metrics JSON.
- Do not accept reference-film mode from sparse fixed-time stills alone.

## Initial Implementation Order

1. Add `reference-exact-smoke` as a separate preset from the local August Complex and procedural reference-film presets.
2. Implement native 1920x1080 lossless reference extraction and the exact manifest.
3. Build UI/fire/background exclusion masks.
4. Derive the clean background and per-frame smoke RGBA cache.
5. Add the manual correction queue and resolve all flagged frames.
6. Transcribe smoke event timing into `reference_smoke_events.json`.
7. Implement screen-space exact smoke playback in `examples/california_cigar_smoke_demo.py`.
8. Match reference geography/date/area/fire semantics for exact mode.
9. Add exact all-frame smoke metrics and audit artifacts.
10. Render the 30-second exact candidate and compare all 900 frames.
11. Only after exact mode is passing, implement or tune `reference-procedural-smoke` from the same event/matte data.

## Acceptance

The next implementation pass should not be considered exact-reference complete until:

- The candidate uses `reference-exact-smoke` and the locked manifest.
- The candidate uses reference-scale geography, 2023 timeline, and reference label semantics.
- All 900 first-30s frames are audited at native resolution.
- The exact smoke gate report passes every threshold in Task J.
- The smoke-difference contact sheets show no visible systematic leakage, missing plume mass, contour artifacts, text leakage, or fire leakage.
- The worst 50 smoke-error frames are reviewed and either corrected or explicitly accepted with metric evidence.
- Any separate procedural candidate is labeled approximate unless it passes the same exact metrics.
