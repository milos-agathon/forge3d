## Terrain Rendering Improvement Plan v3 (strict, preservation-first)

This plan is execution-ready: every phase has specific work items, validation, and artifacts. Do not advance to the next phase until the current one meets its exit criteria.

### 0) Non-negotiable rules
- Preserve all currently correct outputs (terrain color, POM, IBL paths, water mask debug 100/101/102, PBR debug 4-6). Any untouched path must remain bitwise identical.
- Keep changes minimal and reversible; prefer feature flags and config toggles over structural refactors.
- Every phase must capture before/after evidence (images + JSON) using the same assets, camera paths, and seeds.
- Deterministic only: pin seeds, config, CLI args, and device selection; no stochastic sampling.
- No new file formats; only extend existing JSON/log schemas as noted below.
- Any constant/threshold change must include a one-sentence rationale and proof of no regression to preserved results.

### 1) Baseline snapshot (no code changes)
- Reproduce current outputs: terrain render, water mask debug 100/101/102, PBR debug 4-6, and `terrain_high_roughness.png`.
- Use the canonical render command (document CLI, asset paths, seeds, GPU/driver, git SHA).
- Store under `reports/terrain/baseline/`:
  - `baseline_run.log` (full console).
  - `baseline_summary.json` with md5 of each PNG, GPU/driver, CLI args, git rev, timestamp.
  - PNGs used for later diffs.

### 2) Phase P1 - Cascaded Shadows (2-3 days, critical)
**Objective:** Enable CSM path with deterministic, artifact-free shadows.

**Constraints:** No changes to non-shadow PBR math. If CSM unavailable, fall back to NoopShadow with visually identical output.

**Work:**  
1) Single source of truth for `TERRAIN_USE_SHADOWS` in `src/shaders/terrain_pbr_pom.wgsl`.  
2) Bind real CSM resources in `TerrainRenderer` (group 3); default to one cascade if params missing. Validate layouts.  
3) Cascade splits config (default `[50, 200, 800, 3000]`) behind a toggle; defaults must match baseline.  
4) Optional PCSS light-size param; default is prior hard-shadow behavior.  
5) Compile-time debug overlay for cascade boundaries toggled without code edits.

**Validation:**  
- Produce `phase_p1.png` and `phase_p1_diff.png` vs baseline; report SSIM.  
- Inspect shadow maps for acne/peter-panning and cliff light leaks.  
- Log final shadow config used.

**Deliverables:** `reports/terrain/p1_run.log`, `reports/terrain/p1_result.json` (shadow config, SSIM, pass/fail), `phase_p1.png`.

### 3) Phase P2 - Atmospheric Depth (1-2 days, critical)
**Objective:** Add deterministic height-based fog with tunable density/inscatter.

**Constraints:** No change to tonemap/IBL. Fog disabled by default to preserve baseline.

**Work:**  
1) Add fog uniform struct + bind group entries; defaults yield baseline when `fog_density=0`.  
2) Implement `apply_atmospheric_fog` after PBR and before tonemap; respect camera/world height.  
3) CLI params: `--fog-density`, `--fog-height-falloff`, `--fog-inscatter` (all default 0).

**Validation:**  
- `phase_p2.png` and `phase_p2_diff.png` vs P1 with fog off (should match).  
- With fog on, verify distant fade/valley emphasis; log params.

**Deliverables:** `reports/terrain/p2_run.log`, `reports/terrain/p2_result.json` (params, SSIM no-fog vs P1, fog-on notes), updated render.

### 4) Phase P3 - Normal Anti-Aliasing Fix (2-3 days, high)
**Objective:** Restore specular detail without flakes by replacing the roughness floor hack.

**Constraints:** Water branch unchanged. No change to height/POM sampling order.

**Work:**  
1) Generate normal-variance mipchain at heightmap upload; document texture format/layout.  
2) Shader: sample normal + variance and apply Toksvig (or equivalent) to specular roughness only; diffuse uses base roughness.  
3) Lower roughness floor to 0.25 for land, 0.02 for water; clamp for stability.

**Validation:**  
- Debug modes 23-25 unchanged.  
- `phase_p3.png` vs P2 shows restored specular without sparkles in PBR energy debug (mode 12/17).  
- `p3_result.json` includes energy histogram deltas and roughness floor confirmation.

**Deliverables:** `reports/terrain/p3_run.log`, `reports/terrain/p3_result.json`, `phase_p3.png`.

### 5) Phase P4 - Water Planar Reflections (2-3 days, high; after P1)
**Objective:** Add planar reflections with deterministic camera mirroring.

**Constraints:** Do not alter water mask or depth attenuation. Reflection pass is half-res and clipped below water plane.

**Work:**  
1) Add reflection render pass with mirrored camera + clip plane; output view + sampler at group 6.  
2) Shader: sample reflection with wave-based distortion; Fresnel mix with underwater color.  
3) Shore attenuation: reduce wave intensity near land (distance from water mask).

**Validation:**  
- `phase_p4.png` and `phase_p4_diff.png` vs P3; reflections visible on calm water, no land bleeding.  
- Log reflection resolution, clip plane, wave params.

**Deliverables:** `reports/terrain/p4_run.log`, `reports/terrain/p4_result.json` (reflection stats, SSIM vs P3), updated render.

### 6) Phase P5 - Ambient Occlusion Enhancement (1 day, medium)
**Objective:** Verify SSAO contribution and add heightmap AO fallback.

**Constraints:** No change to normal computation. SSAO debug mode stays intact.

**Work:**  
1) Add debug mode 28 outputting raw SSAO buffer.  
2) Precompute coarse horizon AO from the heightmap at upload; bind as optional multiplier (default weight 0 = no-op).

**Validation:**  
- `phase_p5.png` vs P4 with AO off (identical).  
- With AO on, valleys darken without crushing; log AO enable flag and weight.  
- `p5_result.json` confirms SSAO presence and AO fallback path.

**Deliverables:** `reports/terrain/p5_run.log`, `reports/terrain/p5_result.json`, updated render.

### 7) Phase P6 - Micro-Detail (1-2 days, medium; after P3)
**Objective:** Add close-range surface detail without LOD popping.

**Constraints:** Detail normals fade by distance; no change to base triplanar weights.

**Work:**
1) Triplanar detail normal sampling (2 m repeat) blended via RNM with distance fade.
2) Procedural albedo brightness noise (+/- 10%) using stable world-space coordinates.

**Validation:**
- `phase_p6.png` and `phase_p6_diff.png` vs P5 (detail off) prove isolation.
- Check for shimmer during camera motion; log fade distances.

**Deliverables:** `reports/terrain/p6_run.log`, `reports/terrain/p6_result.json`, updated render.

### 8) Implementation schedule (10-14 days total)
- P1: 2-3 days (critical)  
- P2: 1-2 days (critical)  
- P3: 2-3 days (high)  
- P4: 2-3 days (high; after P1)  
- P5: 1 day (medium)  
- P6: 1-2 days (medium; after P3)

### 9) Verification protocol (per phase)
1) Run `python examples/terrain_demo.py` (pinned assets/args/seeds) to produce `phase_P.png`. Document CLI, assets, seeds, GPU/driver, git SHA.  
2) Run `python scripts/compare_images.py baseline phase_P --ssim` to emit `phase_P_diff.png` and SSIM.  
3) Store logs and JSON under `reports/terrain/phase_P/` (one directory per phase).  
4) Record git SHA, GPU/driver, seeds in each `*_result.json`.

### 10) Final proofpack
- After P6, rerun full render with all features on.  
- Produce `reports/terrain/proofpack_summary_final.json`:
  ```json
  {
    "status": "PASS",
    "phases": ["p1", "p2", "p3", "p4", "p5", "p6"],
    "assets": ["dem path", "hdri path"],
    "ssim_vs_baseline": 0.0,
    "changed_files": ["..."],
    "notes": ["no regressions to preserved outputs"]
  }
  ```
- Include `phase_final.png` and its diff vs baseline under `reports/terrain/`.
