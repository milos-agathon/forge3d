## Terrain Rendering Improvement Plan v3 (strict, preservation-first)

### 0) Hard rules (non‑negotiable)
1. Preserve all currently passing outputs and visuals not called out as failing. No regressions to already-correct behaviors (terrain color, POM behavior, IBL paths, water mask debug modes 100/101/102).
2. All changes must be minimal-surface-area and reversible; prefer config toggles over broad refactors.
3. Every phase must capture before/after evidence (images + JSON) and reuse the same input assets, camera paths, and seeds.
4. Deterministic runs only: pin seeds, config, and CLI parameters; no nondeterministic sampling.
5. No new file formats; only extend existing JSON/log outputs as listed.
6. If a threshold/constant is adjusted, justify with a single-sentence rationale and demonstrate no regression to preserved results.

### 1) Baseline snapshot (do not modify code)
- Reproduce current outputs for: terrain render, water mask debug 100/101/102, PBR debug 4–6, existing “terrain_high_roughness.png”.
- Save to `reports/terrain/baseline/`:
  - `baseline_run.log` (full console)
  - `baseline_summary.json` capturing: render hashes (md5 of png), GPU/driver info, CLI args, git rev, timestamp.

### 2) Phase P1 — Cascaded Shadows (2–3 days, critical)
**Objective:** Enable CSM path with deterministic, artifact-free shadows.

**Constraints:**  
- Do not alter non-shadow PBR math.  
- If CSM unavailable, fall back to NoopShadow without visual difference to current baseline.

**Implementation requirements:**  
1. Flip `TERRAIN_USE_SHADOWS` to true in `src/shaders/terrain_pbr_pom.wgsl` with a single source of truth.  
2. Bind real CSM resources in `TerrainRenderer` (group 3) with validated layouts; default to 1 cascade if params missing.  
3. Provide cascade splits tuned for terrain (e.g., `[50, 200, 800, 3000]`) behind config with defaults matching baseline.  
4. Add optional PCSS light-size param; default to prior behavior (hard shadows) to preserve baseline if not set.  
5. Add debug overlay for cascade boundaries (compile-time flag) that can be toggled without code changes.

**Validation:**  
- New images vs baseline: `phase_p1.png`, `phase_p1_diff.png` (SSIM reported).  
- Shadow map visualization: ensure no acne/peter-panning on flat areas, no light leaks on cliffs.  
- Log shadow config used.

**Deliverables:**  
- `reports/terrain/p1_run.log`, `p1_result.json` (shadow config, SSIM vs baseline, pass/fail).  
- Updated shadow-enabled render `phase_p1.png`.

### 3) Phase P2 — Atmospheric Depth (1–2 days, critical)
**Objective:** Add deterministic height-based fog with tunable density/inscatter.

**Constraints:**  
- No change to existing tonemap/IBL.  
- Fog must be disabled by default to preserve baseline when toggled off.

**Implementation requirements:**  
1. Add fog uniforms struct and bind group entries; defaults produce baseline-equivalent output when `fog_density=0`.  
2. Implement `apply_atmospheric_fog` in shader after PBR, before tonemap; honor cam/world height.  
3. CLI params `--fog-density`, `--fog-height-falloff`, `--fog-inscatter`; default 0 for no-op.

**Validation:**  
- `phase_p2.png`, `phase_p2_diff.png` vs P1 when fog disabled (should match).  
- With fog enabled, verify distant fade and valley emphasis; log parameters.

**Deliverables:**  
- `reports/terrain/p2_run.log`, `p2_result.json` (params, SSIM no-fog vs P1, fog-on sanity notes).

### 4) Phase P3 — Normal Anti-Aliasing Fix (2–3 days, high)
**Objective:** Restore specular detail without flakes; replace roughness floor hack.

**Constraints:**  
- Keep water branch unchanged.  
- No change to height/POM sampling order.

**Implementation requirements:**  
1. Generate normal-variance mipchain at heightmap upload; write to texture (format + layout documented).  
2. Shader: sample normal+variance, apply Toksvig (or equivalent) to specular roughness only; diffuse stays base roughness.  
3. Lower roughness floor to ≤0.25 for land, ≤0.02 for water; guard with clamp for stability.

**Validation:**  
- Debug modes 23–25 must still behave per spec.  
- Compare `phase_p3.png` vs P2: specular restored, no sparkles in PBR energy debug (mode 12/17).  
- `p3_result.json`: energy histogram deltas, roughness floor confirmation.

**Deliverables:**  
- `reports/terrain/p3_run.log`, `p3_result.json`, updated render `phase_p3.png`.

### 5) Phase P4 — Water Planar Reflections (2–3 days, high; depends on P1)
**Objective:** Add planar reflections with deterministic camera mirroring.

**Constraints:**  
- Do not alter water mask or depth attenuation.  
- Reflection pass must be half-res and clipped below water plane.

**Implementation requirements:**  
1. Add reflection render pass with mirrored camera and clip plane; output view + sampler @ group(6).  
2. Shader: sample reflection with wave-based distortion; Fresnel mix with underwater color.  
3. Shore attenuation: reduce wave intensity near land (grad based on water mask distance).

**Validation:**  
- `phase_p4.png`, `phase_p4_diff.png` vs P3; confirm reflections visible on calm water, no land bleeding.  
- Log reflection resolution, clip plane, wave params.

**Deliverables:**  
- `reports/terrain/p4_run.log`, `p4_result.json` (reflection stats, SSIM vs P3), updated render.

### 6) Phase P5 — Ambient Occlusion Enhancement (1 day, medium)
**Objective:** Verify SSAO contribution; add heightmap AO fallback.

**Constraints:**  
- No change to normal computation.  
- SSAO debug mode must remain intact.

**Implementation requirements:**  
1. Add debug mode 28 outputting raw SSAO buffer.  
2. Precompute coarse horizon AO from heightmap at upload; bind as optional multiplier (weight default 0 = no-op).

**Validation:**  
- `phase_p5.png` vs P4 with AO off (identical).  
- With AO on, valleys darker without crushing; log AO enable flag and weight.  
- `p5_result.json`: SSAO presence, AO fallback path success.

**Deliverables:**  
- `reports/terrain/p5_run.log`, `p5_result.json`, updated render.

### 7) Phase P6 — Micro-Detail (1–2 days, medium; depends on P3)
**Objective:** Add close-range surface detail without LOD popping.

**Constraints:**  
- Detail normals must fade by distance; no change to base triplanar weights.

**Implementation requirements:**  
1. Triplanar detail normal sampling (2 m repeat) blended via RNM with distance fade.  
2. Procedural albedo brightness noise (±10%) with stable world-space coordinates.

**Validation:**  
- `phase_p6.png`, `phase_p6_diff.png` vs P5 (detail off) to prove isolation.  
- Check for shimmer with camera motion; log fade distances.

**Deliverables:**  
- `reports/terrain/p6_run.log`, `p6_result.json`, updated render.

### 8) Implementation Schedule (10–14 days total)
- P1: 2–3 days (critical)
- P2: 1–2 days (critical)
- P3: 2–3 days (high)
- P4: 2–3 days (high; after P1)
- P5: 1 day (medium)
- P6: 1–2 days (medium; after P3)

### 9) Verification Protocol (per phase)
1. Run `python examples/terrain_demo.py` with pinned assets/args (documented in log) to produce `phase_P.png`.
2. Run `python scripts/compare_images.py baseline phase_P --ssim` to produce `phase_P_diff.png` and SSIM.
3. Store logs and JSON summaries under `reports/terrain/phase_P/`.
4. Record git SHA, GPU/driver, seeds.

### 10) Final proofpack
- After P6: rerun full render with all features on; produce `reports/terrain/proofpack_summary_final.json`:
  ```json
  {
    "status": "PASS",
    "phases": ["p1","p2","p3","p4","p5","p6"],
    "assets": ["dem path", "hdri path"],
    "ssim_vs_baseline": 0.0,
    "changed_files": ["..."],
    "notes": ["no regressions to preserved outputs"]
  }
  ```
- Include final render `phase_final.png` and diff vs baseline.

---