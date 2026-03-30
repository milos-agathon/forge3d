# Terrain Remaining Implementation Plan

**Date:** 2026-03-30
**Primary source:** `docs/plans/2026-03-16-terrain-viz-epics.md`
**Audit basis:** direct repo inspection of `examples/`, `python/forge3d/`, `src/terrain/`, `src/viewer/terrain/`, `docs/terrain/`, `tests/`, and targeted validation in the current worktree.

This document replaces the earlier stale conclusion that the main remaining work was "restore missing demos." That was true for an earlier snapshot, but it is no longer true in the current worktree. Most claimed implemented-terrain epics now have real runtime and example files again. The remaining work is narrower and more precise:

1. fix the implemented epics that still have broken or misleading public surfaces
2. standardize the demo/example substrate where it is inconsistent
3. only then spend time on the genuinely unimplemented terrain backlog

---

## 1. Validation Summary

### 1.1 What was validated

- Targeted terrain-epic regression suite:
  - `pytest tests/test_terrain_sky_parity.py tests/test_aov.py tests/test_terrain_scatter.py tests/test_terrain_tv4_material_variation.py tests/test_terrain_probes.py tests/test_terrain_tv6_heterogeneous_volumetrics.py tests/test_terrain_tv10_subsurface_materials.py tests/test_tv12_offline_quality.py tests/test_tv12_offline_architecture.py tests/test_terrain_tv13_lod_pipeline.py tests/test_camera_rigs.py tests/test_terrain_camera_rigs_demo.py tests/test_tv20_virtual_texturing.py tests/test_terrain_tv21_blending.py tests/test_tv22_scatter_wind.py -q`
  - Result: `191 passed, 2 skipped`
- Example-backed and golden-adjacent validation:
  - `pytest tests/test_terrain_tv4_demo.py tests/test_terrain_tv6_heterogeneous_volumetrics.py tests/test_terrain_tv10_demo.py tests/test_terrain_tv21_demo.py tests/test_terrain_tv24_demo.py tests/test_terrain_visual_goldens.py tests/test_terrain_tv10_goldens.py -q`
  - Result: `1 failed, 4 passed, 2 skipped`
  - The failing test was `tests/test_terrain_tv10_demo.py`
- Direct demo smoke runs:
  - Passed: `TV2`, `TV4`, `TV5`, `TV12`, `TV13`, `TV17`, `TV20`, `TV22`
  - Failed: `TV3` under reduced `--max-dem-size`

### 1.2 Completion scale used below

| Level | Meaning |
|---|---|
| **95-100%** | Runtime, docs, tests, and demo surface are aligned enough to treat the epic as effectively complete. |
| **80-94%** | Core implementation is real, but meaningful quality debt remains. |
| **60-79%** | Core runtime exists, but shipped surface drift or scope caveats make "Implemented" too strong without qualification. |

---

## 2. Implemented Foundations Audit

This section covers every epic listed in the `Implemented foundations` table in `docs/plans/2026-03-16-terrain-viz-epics.md`.

| Epic | Completion | Current status | Findings | Required follow-up |
|---|---:|---|---|---|
| **TV1 - Terrain Atmosphere Path Parity** | **97%** | Strongest shipped terrain epic. Runtime, docs, example, and parity tests are aligned. | No material defect found in this audit. | Keep current parity coverage intact. |
| **TV2 - Terrain Output and Compositing Foundation** | **88%** | AOV and EXR runtime are real and green. Demo smoke passed. | The demo's `_terrain_span()` helper uses raw raster resolution directly, so EPSG:4326 DEMs produce absurd spans like `0.05` and the scene quietly falls back to minimum camera/z-scale heuristics. The feature works, but the demo math is wrong. | Replace the local span heuristic with the guarded helper pattern already used by TV3/TV13/TV22, or make it CRS-aware. Recheck framing and printed diagnostics after that fix. |
| **TV3 - Terrain Scatter and Population** | **84%** | Runtime tests are green. | The public demo is brittle. `python examples/terrain_tv3_scatter_demo.py --width 640 --height 360 --max-dem-size 384` failed because `seeded_random_transforms()` could place only `46` of `96` requested hero instances after filtering. This makes the exposed downsampling knob unsafe and explains why TV3 is not in the example-sanity CI matrix. | Make the demo adapt requested counts, spacing, or filters to reduced terrain size, then add a demo smoke test or CI lane. |
| **TV4 - Terrain Material Variation Upgrade** | **86%** | Runtime tests and demo smoke passed. | Same demo-span bug as TV2. The TV4 demo printed `Terrain span: 0.03` on an EPSG:4326 Mount Fuji DEM, which is not a defensible scene span and is only hidden by downstream clamps. | Standardize TV4 onto the guarded span helper used by TV3/TV13/TV22. |
| **TV5 - Terrain Local Probe Lighting** | **96%** | Diffuse and reflection probe runtime, docs, demo smoke, and tests are aligned. | No actionable defect found in this audit. | Keep probe and reflection-probe coverage healthy. |
| **TV6 - Heterogeneous Terrain Volumetrics** | **95%** | Runtime and example-backed coverage are aligned. | No actionable defect found in this audit. | Keep viewer/reporting coverage healthy. |
| **TV10 - Terrain Subsurface Materials** | **68%** | Core native decode and shader path are real, but the public surface is not clean. | The real-DEM demo is broken right now. `tests/test_terrain_tv10_demo.py` fails because `examples/terrain_tv10_subsurface_demo.py` still passes dead kwargs like `snow_subsurface_color` even though the shipped API is `snow_subsurface_tint`. `docs/terrain/subsurface.md` repeats the dead `*_subsurface_color` names and also documents the wrong defaults; the actual defaults are zero-strength neutral tints, not non-zero snow/wetness SSS. `tests/test_terrain_tv10_goldens.py` is also stale behind its dedicated-golden gate and would hit the same dead kwargs when enabled. TV10 also shares the same broken `_terrain_span()` helper used by TV2/TV4/TV12. | Rename all public references to `*_subsurface_tint`, correct the documented defaults, repair the demo, update the dedicated-golden test, and switch the demo to the guarded span helper. TV10 should not stay in the "cleanly implemented" bucket until this is fixed. |
| **TV12 - Terrain Offline Render Quality** | **84%** | Public API, runtime tests, architecture-doc checks, and demo smoke all passed. | The TV12 demo has the same broken `_terrain_span()` logic as TV2/TV4/TV10 and printed `Terrain span: 0.05` for the Rainier DEM. Separately, the shipped implementation is still CPU-readback-heavy by design; it is functional, but that remains the main architecture ceiling for the epic. | Fix the demo span helper now. Leave GPU-resident accumulation/convergence as a separate improvement unless product goals require it. |
| **TV13 - Terrain Population LOD Pipeline** | **93%** | Simplification, auto-LOD, HLOD, tests, and demo smoke all worked. | No correctness bug found. Small smoke settings can make HLOD look unimpressive or even worse in raw draw-count terms, but that is an example-quality issue, not a broken runtime. | Optional: retune the demo defaults if it should prove HLOD payoff more clearly. Not a blocker. |
| **TV17 - Terrain Camera Rig Toolkit** | **95%** | Tests passed and the export-path demo smoke passed. | No actionable defect found in this audit. | Keep current regression and demo coverage. |
| **TV20 - Terrain Material Virtual Texturing** | **78%** if the epic means full terrain VT, **93%** if it means shipped v1 scope | Real shipped v1. Tests and demo smoke passed. | The native runtime is intentionally albedo-only. `normal` and `mask` families are accepted by the Python contract for forward compatibility but are not decoded or sampled natively. Residency feedback also still depends on CPU readback. This is not a hidden bug; it is a scope boundary that must stay explicit. | Decide whether "Implemented" means "v1 shipped albedo paging" or "full terrain material VT." If the latter, normal/mask family decode, sampling, and stats still remain. |
| **TV21 - Terrain-Mesh Blending and Contact Integration** | **95%** | Runtime, tests, and demo smoke are aligned. | No actionable defect found in this audit. | Keep demo and regression coverage. |
| **TV22 - Scatter Wind Animation** | **83%** | Runtime tests and demo smoke passed. | The epic is functionally shipped, but it is built on an acknowledged substrate defect. `docs/terrain/scatter-wind-animation.md` explicitly records accepted limitations including render-path-dependent phase and a translation/basis-frame mismatch because TV22 works around a pre-existing scatter instance-packing issue. That means TV22 is shipped, but not fully clean. | Keep TV22 marked as implemented in shipped scope, but track scatter instance-packing cleanup as the dependency for fully closing the epic. |

### 2.1 Bottom line on implemented foundations

- **Effectively complete:** `TV1`, `TV5`, `TV6`, `TV13`, `TV17`, `TV21`
- **Implemented with meaningful quality debt:** `TV2`, `TV3`, `TV4`, `TV12`, `TV22`
- **Implemented but still publicly broken/misaligned:** `TV10`
- **Implemented only when read as shipped v1 scope:** `TV20`

---

## 3. Cross-Epic Defects And Quality Debt

These are the real remaining implementation items for the already-shipped column.

| Priority | Area | Problem | Affected epics | Requirement |
|---|---|---|---|---|
| **P0** | TV10 public-surface drift | The runtime uses `*_subsurface_tint`, but the demo, docs, and dedicated-golden test still use dead `*_subsurface_color` names and wrong defaults. | `TV10` | Repair demo/docs/golden suite and rerun example-backed validation. |
| **P0** | Inconsistent terrain-span helpers across demos | Some demos guard against implausibly small world spans from geographic-degree rasters; others do not. This yields nonsense spans like `0.03` or `0.05` and hides the mistake behind minimum camera/z-scale clamps. | `TV2`, `TV4`, `TV10`, `TV12` now; `TV3`, `TV13`, `TV22`, and `TV24` already use the safer pattern | Extract and reuse one guarded helper, or make span calculation explicitly CRS-aware. |
| **P1** | TV3 demo robustness | The public downsampling control can invalidate hardcoded seeded placement counts and crash the demo. | `TV3` | Scale transform requests to terrain size or relax filters automatically when the user downsamples. |
| **P1** | Scope labeling for shipped v1 work | Some implemented epics are clean only if their scope caveats remain explicit. | `TV20`, `TV22`, `TV12` | Keep docs and planning language honest about what is shipped vs. what is deferred. |
| **P2** | Example-sanity coverage is uneven | The example CI matrix currently covers only a small subset of the implemented terrain demos. TV3's public-demo failure escaped because there was no direct example lane. | Especially `TV3`, but also the broader terrain demo surface | After fixing the demo substrate, expand example smoke coverage selectively. |

---

## 4. Dependencies And Blockers

| Dependency | Why it matters | Blocks |
|---|---|---|
| **Shared demo-span helper cleanup** | Until the repo uses one consistent terrain-span rule, several demos will keep reporting bogus scale and relying on hidden clamps. | Clean closure of `TV2`, `TV4`, `TV10`, and `TV12` |
| **Scatter instance-packing cleanup** | TV22 explicitly works around a pre-existing packing/basis mismatch. | A truly clean close on `TV22`; possibly future scatter-adjacent work |
| **Native normal/mask VT path** | The Python contract already accepts more than albedo, but the runtime does not. | Calling `TV20` fully complete in the broad sense |
| **GPU-resident offline accumulation/convergence** | TV12 is functionally real today, but CPU readback remains the main architecture ceiling. | Calling `TV12` renderer-grade complete rather than functionally shipped |

---

## 5. Revised Execution Order

The next work should not start with new terrain epics. It should start by cleaning the already-shipped column.

1. **Fix TV10 first.**
   Repair `examples/terrain_tv10_subsurface_demo.py`, `docs/terrain/subsurface.md`, and `tests/test_terrain_tv10_goldens.py` so the shipped API, demo, docs, and gated golden lane all agree on `*_subsurface_tint`.
2. **Standardize terrain-span helpers next.**
   Lift the guarded helper pattern already present in TV3/TV13/TV22 into the remaining demos that still use raw geographic-degree span.
3. **Make the TV3 demo robust.**
   The demo must survive its own exposed `--max-dem-size` knob before it can be treated as a reliable shipped example or promoted into CI example coverage.
4. **Then decide how hard to close the scope-limited epics.**
   `TV20` and `TV22` are shippable today, but only with their current caveats kept explicit.
5. **Only after that move back to the true backlog.**
   The core unimplemented terrain epics are still `TV16`, then `TV18`, then `TV7`.

---

## 6. Net-New Terrain Epics Still Not Implemented

After the implemented-foundations cleanup above, the genuinely unbuilt terrain backlog is still:

### 6.1 Core backlog

- **TV16 - Terrain Scene Variants and Review Layers**
- **TV18 - Terrain Shot Queue and Bounded Timeline**
- **TV7 - Weather Particle Foundation**

### 6.2 Conditional or deferred backlog

- **TV8 - Coastal / Hydrology Water Upgrade**
- **TV9 - OCIO Color-Managed Terrain Output**
- **TV11 - Page-Based Terrain Shadowing**
- **TV14 - Terrain Flow and Trajectory Visualization**
- **TV15 - Compute Tessellation for Terrain**
- **TV19 - Collaborative Terrain Review**
- **TV23 - Terrain Temporal Upscaling and Upscaled Viewer Path**

---

## 7. Practical Conclusion

The repo no longer has a "missing terrain demos everywhere" problem. That diagnosis is stale.

The actual remaining work is:

1. **repair the implemented column where it is still publicly inconsistent**
2. **standardize the demo substrate where helpers drifted**
3. **only then continue with `TV16`, `TV18`, and `TV7`**

That is the current terrain remaining implementation plan.
