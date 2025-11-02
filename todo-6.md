

# P6 implementation plan: priorities, estimates, and order

Below is a concrete execution plan to take P6 to “fully implemented,” with ordering, estimates, dependencies, and exit criteria. Citations refer to files you already have (e.g., [src/shaders/sky.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/sky.wgsl:0:0-0:0), [src/shaders/volumetric.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/volumetric.wgsl:0:0-0:0), [src/lighting/types.rs](cci:7://file:///Users/mpopovic3/forge3d/src/lighting/types.rs:0:0-0:0), [examples/terrain_demo.py](cci:7://file:///Users/mpopovic3/forge3d/examples/terrain_demo.py:0:0-0:0), [python/forge3d/config.py](cci:7://file:///Users/mpopovic3/forge3d/python/forge3d/config.py:0:0-0:0), [docs/rendering_options.rst](cci:7://file:///Users/mpopovic3/forge3d/docs/rendering_options.rst:0:0-0:0)).

## Milestone 1 — Sky pass (user-visible first)

- __P6-01: SkyRenderer pass wiring__ (High, 1.5–2.5 days)
  - Load [src/shaders/sky.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/sky.wgsl:0:0-0:0); create UBO for [SkySettings](cci:2://file:///Users/mpopovic3/forge3d/src/lighting/types.rs:984:0-1003:1) (see [src/lighting/types.rs](cci:7://file:///Users/mpopovic3/forge3d/src/lighting/types.rs:0:0-0:0)).
  - Bind camera uniforms (align with `CameraUniforms` used in other shaders).
  - Choose sun direction from first directional light; fallback to `SkySettings.sun_direction`.
  - Render before opaques (background) via fullscreen triangle or compute.
  - Exit criteria: with `--sky preetham|hosek-wilkie`, a sky appears and respects turbidity/ground_albedo; no regressions.

- __P6-02: Sky calibration & tonemapping__ (High, 0.5–1 day)
  - Calibrate `sun_intensity` vs `exposure`; avoid ad‑hoc multipliers.
  - Confirm midday and low-sun exposures look plausible.
  - Exit criteria: reasonable defaults for mid‑day and sunset with simple tonemap in [sky.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/sky.wgsl:0:0-0:0).

- __P6-03: Accurate Hosek–Wilkie coefficients__ (Medium, 1–2 days)
  - Replace the simplified coefficients in [sky.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/sky.wgsl:0:0-0:0) with the published RGB datasets or a LUT fit.
  - Sanity validation across turbidity ∈ [1,10], sun elevations ∈ [0°, 90°].
  - Exit criteria: visible quality improvement near horizon/sunset; model parity with reference curves.

Notes
- You can ship P6 visually with P6‑01/02, then upgrade to accurate HW in P6‑03. HW is medium risk (dataset licensing/format).

## Milestone 2 — Volumetric fog ray marching

- __P6-06: Expand config schema (Py + Rust)__ (High, 0.5–1 day)
  - Extend [python/forge3d/config.py](cci:7://file:///Users/mpopovic3/forge3d/python/forge3d/config.py:0:0-0:0) and [src/render/params.rs](cci:7://file:///Users/mpopovic3/forge3d/src/render/params.rs:0:0-0:0) to include all shader parameters needed by [src/shaders/volumetric.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/volumetric.wgsl:0:0-0:0):
    - `height_falloff`, `max_steps`, `start_distance`, `max_distance`, `absorption`, `scattering_color`, `ambient_color`, `temporal_alpha`, `use_shadows`, `jitter_strength`.
  - Map CLI `--volumetric 'density=...,phase=hg,g=...,max_steps=...,start=...,max=...,shadows=true,temporal=...’`.
  - Exit criteria: [load_renderer_config()](cci:1://file:///Users/mpopovic3/forge3d/python/forge3d/config.py:613:0-630:14) returns a complete `RendererConfig.atmosphere.volumetric` that can drive the shader.

- __P6-04: Fog ray-march compute pass integration__ (High, 2–3 days)
  - Load [src/shaders/volumetric.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/volumetric.wgsl:0:0-0:0); UBO upload from [VolumetricSettings](cci:2://file:///Users/mpopovic3/forge3d/src/lighting/types.rs:1102:0-1142:1).
  - Bind scene depth, shadow map + comparison sampler, and a history texture (for temporal reprojection).
  - Dispatch `cs_volumetric` at viewport size; write RGBA16F fog.
  - Exit criteria: fog layer appears and thickens with distance/density; god-rays visible when occluded (with shadows on).

- __P6-05: Composition + temporal stability__ (High, 1–2 days)
  - Composite fog into main HDR color (Beer–Lambert or premultiplied alpha).
  - Implement ping-pong history and frame-jitter sequencing.
  - Exit criteria: turning `temporal_alpha` up visibly stabilizes fog; no ghosting with static camera.

Notes
- Dependencies: shadow map availability (directional light). Ensure a shadow matrix is bound (CSM/PCF path).
- Camera uniforms across shaders should be consistent. If there’s drift, unify during P6‑04 (or include in P6‑10 polish).

## Milestone 3 — Viewer UX + acceptance

- __P6-07: Viewer toggles/HUD__ (Medium, 1 day)
  - Add interactive commands:
    - `:sky off|preetham|hosek-wilkie`, `:sky-turbidity <f>`, `:sky-ground <f>`
    - `:fog on|off`, `:fog-density <f>`, `:fog-g <f>`, `:fog-steps <u32>`, `:fog-shadow on|off`, `:fog-temporal <0..0.9>`
  - Display current settings in a small overlay (e.g., top-left).
  - Exit criteria: single-terminal workflow preserved; you can live-tune sky and fog without restart.

- __P6-08: Acceptance tests & goldens__ (High, 1–2 days)
  - Sky tests: golden images for midday and low sun; verify horizon color shift.
  - Fog tests: scene with an occluder; assert non-zero in-scatter behind occluder when shadows enabled.
  - Temporal test: run N frames with jitter; assert variance reduces with `temporal_alpha`.
  - Wire into CI; store goldens under `tests/golden/` and document in `docs/`.
  - Exit criteria: repeatable CI checks that map to the “Acceptance” in [p6.md](cci:7://file:///Users/mpopovic3/forge3d/p6.md:0:0-0:0).

## Milestone 4 — Froxelized volumetrics (optimization)

- __P6-09: Froxel pipeline__ (Medium, 3–4 days)
  - Implement `cs_build_froxels` to populate 3D fog (RGBA16F) and `cs_apply_froxels` to integrate along view rays.
  - Add a config toggle: `volumetric.mode = raymarch|froxels`; provide low/med/high presets.
  - Exit criteria: froxel path achieves similar look at lower cost for mid/high distances.

## Milestone 5 — Performance and polish

- __P6-10: Perf & polish__ (Medium, 1–2 days)
  - Half-res fog + bilateral upsample option; step heuristics based on depth range.
  - Unify `CameraUniforms` across WGSL ([bundles.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/bundles.wgsl:0:0-0:0), [gbuffer.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/gbuffer.wgsl:0:0-0:0), [sky.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/sky.wgsl:0:0-0:0), [volumetric.wgsl](cci:7://file:///Users/mpopovic3/forge3d/src/shaders/volumetric.wgsl:0:0-0:0)) if needed.
  - Update [docs/rendering_options.rst](cci:7://file:///Users/mpopovic3/forge3d/docs/rendering_options.rst:0:0-0:0) with full parameter set and tuning tips.
  - Ensure build is warning-free; address clippy where touched.

---

## Recommended implementation order

1) P6‑01 SkyRenderer wiring → P6‑02 calibration → P6‑03 accurate HW  
2) P6‑06 config expansion → P6‑04 fog compute → P6‑05 composition/temporal  
3) P6‑07 viewer toggles → P6‑08 acceptance tests/goldens  
4) P6‑09 froxels (optional optimization)  
5) P6‑10 perf/polish/docs

Rationale
- Sky first gives immediate visible value and is isolated (no depth/shadows).
- Volumetric raymarch next enables acceptance (“beams visible”, “temporal stability”).
- Viewer toggles satisfy the interactive workflow requirement.
- Froxels are an optimization you can land after baseline value ships.
- Perf/polish brings consistency and docs readiness.

## Estimates summary (single engineer; add 30–40% overlap for review/CI)

- P6‑01: 1.5–2.5 days
- P6‑02: 0.5–1 day
- P6‑03: 1–2 days
- P6‑06: 0.5–1 day
- P6‑04: 2–3 days
- P6‑05: 1–2 days
- P6‑07: 1 day
- P6‑08: 1–2 days
- P6‑09: 3–4 days
- P6‑10: 1–2 days

Base “ship” without froxels: ~8–12 days  
With froxels and polish: +4–6 days → ~12–18 days

Parallelization suggestions
- One engineer does P6‑01/02 while another expands config (P6‑06).
- P6‑03 (HW coefficients) can be done in parallel with fog wiring.
- Viewer toggles (P6‑07) can start once sky pass is in.
- Tests (P6‑08) trail each feature by 0.5 day.

## Risks and mitigations

- __Hosek–Wilkie datasets__: Ensure license/embedding approach (LUT vs computation). Mitigation: ship Preetham + “approx HW” first; switch to LUTs after review.
- __Shadow dependencies__: God-rays require directional shadow map + matrix. Mitigation: clear errors when missing; auto-disable shadows in fog.
- __Uniform mismatches__: Multiple `CameraUniforms` flavors across shaders. Mitigation: unify layouts early (P6‑04 or P6‑10).
- __Performance__: Fog step counts can spike. Mitigation: presets, half-res option, step capping, early-outs.

## Deliverables per milestone

- M1: Sky visible with Preetham/HW, controllable via config.  
- M2: Volumetric fog with god-rays and temporal reprojection; correct composition.  
- M3: Interactive viewer controls; acceptance tests in CI.  
- M4: Optional froxel mode and quality presets.  
- M5: Performance switches, unified uniforms, updated docs, warning-free build.

I’ve also organized these tasks in a working todo list with priorities and statuses so we can track progress as we implement. If you want, I can start by drafting the Rust SkyRenderer scaffolding and the UBO upload path for [SkySettings](cci:2://file:///Users/mpopovic3/forge3d/src/lighting/types.rs:984:0-1003:1).

