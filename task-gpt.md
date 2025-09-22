# GPT-5-CODEX-HIGH TASK
# Title: Workstream B — Realtime PBR & Lighting (implement & wire missing/partial items)
# Mode: Implementor • Make small, atomic commits • Keep diffs tight • Follow steps in order

REPO
- Root: .
- Create branch: wsB-pbr-lighting-integration
- Paths to use only:
  - src/
  - shaders/
  - python/forge3d/
  - examples/
  - docs/
  - tests/

CONSTRAINTS
- Platforms: win_amd64, linux_x86_64, macos_universal2
- GPU budget: ≤512 MiB host-visible heap (prefer low-res textures, ≤4× MSAA defaults)
- Toolchain: cmake≥3.24, cargo/rustc, PyO3, VMA
- APIs: WebGPU/WGSL primary; design compatible with Vulkan 1.2
- Docs: Sphinx
- Safety: do NOT touch build/, dist/, target/, _build/, .venv/, *.png/*.bin artifacts; no blind search/replace

WORKSTREAM B TASKS (scope only)
- B1  MSAA & Super-Sampling
- B2  Tone-Mapping & Exposure v2
- B3  SSAO Pass
- B4  Cascaded Shadow Maps
- B5  Planar Reflections (Realtime)
- B6  Depth of Field (Realtime)
- B7  Cloud Shade Overlay
- B8  Render Clouds (Realtime)
- B10 Ground Plane (Raster)
- B11 Water Surface Color Toggle
- B12 Soft Light Radius (Raster)
- B13 Point & Spot Lights (Realtime)
- B14 Rect Area Lights (LTC) — verify only
- B15 Image-Based Lighting (IBL)
- B16 Dual-source blending OIT (Realtime)
- B17 Depth-clip control for CSM

PLAN — EXECUTE IN ORDER
0) Scaffolding (tests/docs)
   - Add pytest markers “pbr” and “lighting”.
   - Create Sphinx stubs for PBR & Lighting in docs/api/.
   - Commit: “WS-B: test/docs scaffolding”

1) B2 — Tone-Mapping & Exposure v2
   - Add shaders/tone_map.wgsl implementing ACES, Reinhard, Hable; include unit curve helpers.
   - Wire post-pass into src/pipeline/pbr.rs (or equivalent render graph).
   - Python API:
     - python/forge3d/pbr.py: set_tone_mapping(mode: Literal["aces","reinhard","hable"])
     - python/forge3d/lighting.py: set_exposure_stops(stops: float)
   - Tests: tests/test_b2_tonemap.py (curve sanity, deterministic toggles)
   - Example: examples/pbr_spheres.py (cycle modes & exposure)
   - Commit: “WS-B: tone-mapping & exposure v2 (ACES/Reinhard/Hable)”

2) B1 — MSAA & resolve
   - Enable multisampled color/depth; resolve to presentable texture.
   - Python: python/forge3d/viewer.py set_msaa(samples: int {1,2,4,8})
   - Tests: tests/test_b1_msaa.py (edge metric improves vs 1×, no artifacts)
   - Commit: “WS-B: MSAA targets + resolve + API”

3) B3 — SSAO Pass
   - Ensure G-buffer normals/depth available; add shaders/ssao.wgsl + bilateral blur.
   - Integrate offscreen → composite; expose radius/intensity API.
   - Tests: tests/test_b3_ssao.py (depth cue present; perf at 1080p marker)
   - Example: examples/ssao_demo.py
   - Commit: “WS-B: SSAO pass + bilateral blur + API”

4) B4 — Cascaded Shadow Maps (CSM)
   - Implement 3–4 cascades; PCF/EVSM kernels in shaders/csm.wgsl.
   - Split shadow map build vs sampling; integrate with PBR lights.
   - Tests: tests/test_b4_csm.py (no peter-panning; stable motion)
   - Example: examples/csm_demo.py
   - Commit: “WS-B: CSM (3–4 cascades) with PCF/EVSM”

5) B5 — Planar Reflections
   - Clip-plane render-to-texture; roughness-aware blur.
   - shaders/planar_reflections.wgsl; viewer toggle/API.
   - Tests: tests/test_b5_reflections.py (correct reflection; ≤15% frame cost)
   - Example: examples/reflective_plane_demo.py
   - Commit: “WS-B: planar reflections + roughness blur”

6) B6 — Realtime DOF
   - shaders/dof.wgsl: circle-of-confusion + near/far gather.
   - Camera API: aperture, focus_distance.
   - Tests: tests/test_b6_dof.py (bokeh size matches aperture; no halos)
   - Example: examples/dof_demo.py
   - Commit: “WS-B: realtime DOF + camera params”

7) B7 — Cloud Shade Overlay
   - 2D texture modulation over terrain; density/speed uniforms.
   - Tests: tests/test_b7_cloudshade.py (no banding; modulates irradiance)
   - Commit: “WS-B: cloud shade overlay pass”

8) B8 — Realtime Clouds
   - shaders/clouds.wgsl: billboard/volumetric-lite path; IBL-aware scatter approx.
   - Tests: tests/test_b8_clouds.py (60 FPS 1080p on low-VRAM preset)
   - Example: examples/clouds_demo.py
   - Commit: “WS-B: realtime clouds (lite) + IBL tie-in”

9) B10 — Ground Plane (Raster)
   - Simple raster ground plane (grid/albedo); viewer toggle.
   - Tests: tests/test_b10_groundplane.py (no z-fighting; below geometry)
   - Commit: “WS-B: raster ground plane + toggle”

10) B11 — Water Surface Color Toggle
    - Pipeline uniform for water albedo/hue; Python setter.
    - Tests: tests/test_b11_water_toggle.py (predictable tint; round-trip)
    - Commit: “WS-B: water surface color toggle”

11) B12 — Soft Light Radius (Raster)
    - Add radius parameter + falloff function in raster light shader path.
    - Tests: tests/test_b12_softlight.py (visible softening; raster path stable)
    - Commit: “WS-B: raster soft light radius control”

12) B13 — Point & Spot Lights
    - Complete per-light buffers, shadow toggles, penumbra shaping.
    - Tests: tests/test_b13_point_spot.py (correct illum; shadow toggles verified)
    - Commit: “WS-B: point/spot lights polish”

13) B15 — IBL polish
    - Generate irradiance/specular prefilter + BRDF LUT; verify rough/spec.
    - Tests: tests/test_b15_ibl.py
    - Docs: docs/api/lighting.md sections for IBL pipeline & assets cache
    - Commit: “WS-B: IBL prefilter + BRDF LUT + docs”

14) B16 — Dual-source blending OIT
    - Enable dual-source where supported; keep WBOIT fallback; runtime switch.
    - Example: examples/oit_dual_source_demo.py (+ reference)
    - Tests: tests/test_b16_oit_dual_source.py (ΔE ≤ 2 vs reference; stable FPS)
    - Commit: “WS-B: dual-source OIT + WBOIT fallback + example”

15) B17 — Depth-clip control for CSM (depends on B4)
    - Support unclippedDepth when available; retune cascades; regressions tests.
    - Tests: tests/test_b17_depthclip.py (artifact removal; no regressions)
    - Commit: “WS-B: CSM depth-clip control + tuning”

16) B14 — Rect Area Lights (LTC) — verify-only
    - Run example(s); confirm visual parity; no code changes.
    - Commit: “WS-B: verify LTC rect area lights (no changes)”

DOCS (finalize)
- docs/api/pbr.md, docs/api/lighting.md: how-to per feature, low-VRAM presets, toggles.
- Commit: “WS-B: PBR/lighting docs (Sphinx)”

EXAMPLES (finalize)
- Ensure runnable examples listed below exist & use ≤512 MiB presets:
  - examples/pbr_spheres.py
  - examples/advanced_terrain_shadows_pbr.py
  - examples/ssao_demo.py
  - examples/csm_demo.py
  - examples/reflective_plane_demo.py
  - examples/dof_demo.py
  - examples/clouds_demo.py
  - examples/oit_dual_source_demo.py
- Commit: “WS-B: runnable examples (low-VRAM presets)”

TEST MATRIX / ACCEPTANCE CRITERIA (must pass)
- B1: Edge metric improves ≥20% vs 1×; no resolve artifacts.
- B2: Tone-map curves unit tests pass; exposure stop control deterministic.
- B3: SSAO visible depth cue; 1080p/60 marker on low-VRAM preset.
- B4: No peter-panning; stable during motion.
- B5: Correct reflection; ≤15% frame cost at 1080p preset.
- B6: Bokeh size matches aperture; no haloing.
- B7: Cloud shade modulates irradiance without banding.
- B8: Clouds 60 FPS at 1080p on preset.
- B10: Ground plane under geometry; z-fighting guarded.
- B11: Water tint toggles predictably; scene round-trip intact.
- B12: Radius control softens falloff; raster path stable.
- B13: Point/spot illumination correct; shadow toggles verified.
- B14: Visual parity with prior LTC output (no regressions).
- B15: Roughness/specular behavior correct; BRDF LUT generated.
- B16: ΔE ≤ 2 vs dual-source reference; FPS stable at 1080p.
- B17: CSM clipping artifacts removed on supported GPUs; no regressions.

FILES TO CREATE/TOUCH (non-exhaustive)
- shaders/: tone_map.wgsl, ssao.wgsl, csm.wgsl, planar_reflections.wgsl, dof.wgsl, clouds.wgsl
- python/forge3d/: viewer.py, pbr.py, lighting.py (new setters/toggles)
- examples/: pbr_spheres.py, advanced_terrain_shadows_pbr.py, ssao_demo.py, csm_demo.py, reflective_plane_demo.py, dof_demo.py, clouds_demo.py, oit_dual_source_demo.py
- docs/api/: pbr.md, lighting.md
- tests/: test_b1_msaa.py, test_b2_tonemap.py, test_b3_ssao.py, test_b4_csm.py, test_b5_reflections.py, test_b6_dof.py, test_b7_cloudshade.py, test_b8_clouds.py, test_b10_groundplane.py, test_b11_water_toggle.py, test_b12_softlight.py, test_b13_point_spot.py, test_b15_ibl.py, test_b16_oit_dual_source.py, test_b17_depthclip.py

RUNBOOK (use locally in this order)
- git checkout -b wsB-pbr-lighting-integration
- cargo fmt --check
- cargo clippy --all-targets --all-features -D warnings
- cargo test -q
- pytest -q
- pytest -k "pbr or lighting" -v
- sphinx-build -b html docs _build/html
- maturin build --release
- python examples/pbr_spheres.py
- python examples/advanced_terrain_shadows_pbr.py

DONE WHEN
- All tests above pass on Windows/Linux/macOS runners.
- Examples render headless within ≤512 MiB host-visible heap.
- ΔE and FPS ACs met; docs build clean; APIs surfaced for all toggles.

GIT HYGIENE
- One feature per commit (code + tests + minimal docs tweak).
- Clear messages prefixed “WS-B: …”.
- No vendored binaries or large assets.

OUTPUT
- Provide final summary of changed files, passing test counts, and a short demo log for examples (first/last 3 lines).