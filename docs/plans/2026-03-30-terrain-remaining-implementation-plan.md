# Terrain Remaining Implementation Plan

**Date:** 2026-03-30  
**Sources:** `docs/plans/2026-03-16-terrain-viz-epics.md`, direct repo audit of `src/`, `python/forge3d/`, `tests/`, `examples/`, and targeted terrain regression runs in the current worktree.  
**Purpose:** Record the work that still must be implemented after the 2026-03-30 terrain-foundations re-audit. This document distinguishes between:

1. work needed to make already-claimed "implemented" epics actually complete in the current worktree
2. net-new terrain epics that are still genuinely unimplemented

---

## 1. Completion Rule

A terrain epic is not fully complete in this repo until all of the following are true:

1. The runtime/API path exists and works.
2. The docs describe the real API and real behavior.
3. Any claimed example/demo script exists and runs, or all claims about that example are removed.
4. Tests match the current surface and do not point at deleted assets.
5. CI does not reference missing example scripts.

The current worktree clears item 1 for many shipped epics, but fails items 2-5 in several places.

---

## 2. P0 Repo-Completion Work For Already-Implemented Epics

These items come before any new terrain epic. They are required to make the current "Implemented foundations" table defensible.

| Priority | Area | What still must be implemented | Done when |
|---|---|---|---|
| **P0** | Example/demo recovery or cleanup | Restore the missing terrain example scripts, or deliberately replace them and update every doc/test/CI reference. Affected claims currently cover TV2, TV3, TV4, TV5, TV6, TV10, TV12, TV13, TV20, TV21, and TV22. | No terrain docs/tests/CI entries point at deleted `examples/terrain_tv*_demo.py` files. |
| **P0** | Demo-regression repair | Fix the currently broken example-driven tests by restoring the expected scripts or rewriting the tests around the new example locations. | The demo tests for TV4, TV6, TV10, TV21, and TV24 pass again, or are intentionally replaced with equivalent coverage. |
| **P0** | CI matrix repair | Remove or replace CI entries that still point at deleted example scripts. | CI example lanes only reference scripts that exist in `examples/`. |
| **P0** | TV10 docs/API alignment | Update TV10 docs to use the real `*_subsurface_tint` API, correct the default values, and remove algorithm claims that the shader does not implement. | The TV10 doc snippet is copy-paste correct and the prose matches the actual shader/data model. |

---

## 3. Epic-By-Epic Remaining Work

This section is the concrete delta from "runtime exists" to "epic is actually complete."

| Epic | Current reality | What still must be implemented |
|---|---|---|
| **TV1 - Terrain Atmosphere Path Parity** | Strongest shipped foundation. Runtime, example, and parity tests are aligned. | No material implementation gap found in this audit. Keep current coverage healthy. |
| **TV2 - Terrain Output and Compositing Foundation** | Core AOV and EXR functionality is real. | Restore or replace the missing TV2 example script and repair the CI/example-story around it. |
| **TV3 - Terrain Scatter and Population** | Scatter runtime looks real and tested. | Restore or replace the missing TV3 demo so the docs stop advertising a nonexistent bundled example. |
| **TV4 - Terrain Material Variation Upgrade** | Core runtime/tests are present. | Restore or replace the missing TV4 demo, repair its demo test, and repair the CI example lane that still points at the deleted script. |
| **TV5 - Terrain Local Probe Lighting** | Probe lighting runtime is present, including reflection probes. | Restore or replace the missing probe-lighting demo and the missing reflection-probe demo, or remove the corresponding docs/demo-test claims. |
| **TV6 - Heterogeneous Terrain Volumetrics** | Runtime/settings path is present. | Restore or replace the missing real-DEM volumetrics demo so the example-driven coverage matches the shipped claim again. |
| **TV10 - Terrain Subsurface Materials** | Native decode and shader path are present. | Restore or replace the missing TV10 demo and fix the docs so they describe the real API, real defaults, and real shader behavior. |
| **TV12 - Terrain Offline Render Quality** | Public API and runtime path are present. | Restore or replace the missing TV12 demo. If the goal is a more complete renderer-grade offline path, move accumulation/convergence off the current CPU-readback-heavy approach. |
| **TV13 - Terrain Population LOD Pipeline** | Simplification, auto-LOD, and HLOD runtime are present. | Restore or replace the missing TV13 end-to-end demo so the docs and shipped claim stay honest. |
| **TV17 - Terrain Camera Rig Toolkit** | Best-aligned shipped epic besides TV1. | No material implementation gap found in this audit. Keep the current demo/tests intact. |
| **Terrain Material Virtual Texturing** | Good shipped v1. Current native runtime pages albedo only. | Restore or replace the missing TV20 demo. If this epic is meant to be complete beyond v1 scope, add native normal/mask family decode, residency, and sampling rather than leaving them as contract placeholders. |
| **TV21 - Terrain-Mesh Blending and Contact Integration** | Runtime path is present and tested. | Restore or replace the missing TV21 real-DEM example and repair its example smoke test. |
| **TV22 - Scatter Wind Animation** | Runtime/viewer path is present. | Restore or replace the missing TV22 demo. If this epic is meant to be fully closed rather than v1-shipped, fix the underlying scatter instance-packing issue that TV22 currently works around. |

---

## 4. Net-New Terrain Epics Still Not Implemented

These are the actual terrain epics that remain unbuilt after the implemented-foundations audit.

### 4.1 Core backlog

| Epic | Why it still must be implemented | Minimum definition of done |
|---|---|---|
| **TV16 - Terrain Scene Variants and Review Layers** | Bundles persist terrain metadata, presets, and bookmarks, but there is still no real terrain review-state model. | Named terrain variants, grouped review-layer visibility, atomic list/query/apply APIs, and persistence/state-isolation tests. |
| **TV18 - Terrain Shot Queue and Bounded Timeline** | Forge3D can render one terrain sequence, but still lacks a terrain delivery workflow for multi-shot output. | Serializable shot manifest, bounded terrain-only track set, pass-aware multi-shot rendering, stable output layout, and resume semantics. |
| **TV7 - Weather Particle Foundation** | TV6 volumetrics do not replace true terrain weather particles. | GPU spawn/update/render for terrain weather, terrain-aware collision/kill behavior, and a narrow preset set covering rain, snow, dust, and ash. |

### 4.2 Deferred or conditional work

These are still genuinely unimplemented, but they are not the first priority after the repo-completion work above:

- **TV11 - Page-Based Terrain Shadowing**
- **TV8 - Coastal / Hydrology Water Upgrade**
- **TV9 - OCIO Color-Managed Terrain Output**
- **TV14 - Terrain Flow and Trajectory Visualization**
- **TV15 - Compute Tessellation for Terrain**
- **TV19 - Collaborative Terrain Review**
- **TV23 - Terrain Temporal Upscaling and Upscaled Viewer Path**

---

## 5. Recommended Execution Order

1. **Repair the shipped surface first.** Restore or replace missing example scripts, fix demo tests, and clean up CI/doc references.
2. **Fix TV10 documentation next.** It is the one implemented epic whose public doc is actively wrong enough to break copy-paste usage.
3. **Then build TV16.** Review-state management is the next real missing terrain workflow primitive.
4. **Then build TV18.** Shot queuing depends on a clearer terrain review/variant model.
5. **Then build TV7.** Weather particles matter, but they are less foundational than the review/delivery backlog.
6. **Only after that, decide whether to expand scope-limited shipped work.** The main candidates are TV20 normal/mask VT support, a more GPU-resident TV12 pipeline, and the substrate cleanup behind TV22.

---

## 6. Bottom Line

Forge3D does not primarily need more terrain feature ideation right now. It needs two things:

1. cleanup work that makes the already-claimed terrain foundations actually complete in the current repo
2. the next real backlog epics: TV16, TV18, and then TV7

That is the remaining implementation plan.
