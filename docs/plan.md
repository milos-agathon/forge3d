Milestone A — Make debug modes provably distinct (unblocker)

Why

Your log + images show:
	•	Mode 0 differs ✅
	•	Modes 23–27 are pixel-identical ❌ (so your new diagnostic modes aren’t actually diagnosing)

This means the shader is almost certainly doing something like if (debug_mode >= 23) { return <same viz>; } or a switch with a fallthrough/default that catches 23–27.

Work
	1.	Hard-wire sentinel colors first (to validate branching):
	•	23 → pure red
	•	24 → pure green
	•	25 → pure blue
	•	26 → grayscale ramp from computed LOD
	•	27 → grayscale ramp from normal_blend
	2.	Once branching is verified, swap sentinel returns back to the real visualizations.

Deliverables
	•	reports/flake_diag/mode23_sentinel.png … mode27_sentinel.png (must be obviously different)
	•	Updated tests/test_flake_diagnosis.py:
	•	asserts mode23 != mode24 != … != mode27 (pairwise non-equality)
	•	keeps your “non-uniform” checks, but adds “not identical to other modes”
	•	reports/flake_diag/debug_grid_v2.png (6-up grid, visibly distinct)

Acceptance: pixel-equality between any two of {23..27} must be false.

⸻

Milestone B — Validate the actual flake root cause with the now-working modes

Why

Right now your “interpretation” is plausible, but mode 26/27 aren’t actually showing LOD/blend yet (they’re identical to 24/25), so we can’t trust the narrative until A is done.

Work (after A)

Run the same camera + settings and capture:
	•	Mode 23 (No Specular) → confirm flakes disappear = spec aliasing
	•	Mode 24 (No Height Normal) → confirm whether height-normal bandwidth is the driver
	•	Mode 25 (ddxddy normal) → sanity check “ground truth” stability
	•	Mode 26 (Height LOD) → verify LOD is a smooth field (no banding/tiling)
	•	Mode 27 (Normal Blend) → confirm blend follows your intended fade curve spatially

Deliverables
	•	reports/flake_diag/modes_23_27_<scene>.png (individual)
	•	reports/flake_diag/debug_grid_<scene>.png (grid)
	•	reports/flake_diag/flake_readout.json containing:
	•	sparkle metric / energy metric for baseline vs (23,24)
	•	% pixels above threshold in specular-only view (if you have it)

Acceptance: Mode 26/27 must visually encode LOD/blend (not look like “generic smooth normal”).

⸻

Milestone C — Lock the LOD-aware Sobel fix into a regression-proof implementation

Why

Your stated “root cause” is exactly the classic bug: implicit LOD height sampling + level-0 texel offsets. You implemented the right class of fix (consistent LOD across 9 taps), but we need to make it robust and measurable.

Work
	1.	Compute LOD from UV derivatives, not world position:
	•	rho = max(length(dpdx(uv) * texDims), length(dpdy(uv) * texDims))
	•	lod = clamp(log2(rho), 0, maxMip)
	2.	Scale Sobel offsets to the chosen mip:
	•	texel_uv = (1.0 / texDims) * exp2(lod) (or equivalent)
	3.	Ensure all 9 taps use textureSampleLevel(..., lod) and offsets derived from that mip’s texel size.

Deliverables
	•	Code: calculate_normal_lod_aware() uses one lod, one texel scale, nine SampleLevels.
	•	reports/flake_diag/normal_compare.png:
	•	baseline (old), fixed (lod-aware), ddxddy
	•	reports/flake_diag/normal_diff_heatmap.png (fixed vs ddxddy)

Acceptance: flakes are reduced without “popping”; normal field should not show mip-grid discontinuities when orbiting.

⸻

Milestone D — Tune the minification fade so it doesn’t over-smooth

Why

Your “fixed vs ground truth” montage shows a huge qualitative difference: ddxddy is extremely smooth; LOD-aware Sobel still has structured variation. That can be fine, but you want controlled bandlimit, not accidental blur or residual contouring.

Work
	1.	Make lod_lo/lod_hi/blend curve configurable (even if only via constants/env for now).
	2.	Consider a smoother curve (e.g., smoothstep) instead of linear to reduce threshold-y transitions.
	3.	Add a “near detail preserved” check:
	•	render same shot at two distances
	•	verify near retains high-frequency detail relative to far

Deliverables
	•	reports/flake_diag/normal_blend_curve_v2.png
	•	reports/flake_diag/orbit_sequence/ (N frames or 4 keyframes) showing no popping
	•	Updated docs/plan.md section: bandlimit/fade policy + recommended defaults

Acceptance: far field stable; near field still has terrain micro-shape; orbit has no visible popping rings.

⸻

Milestone E — CI-proof pack (so this never regresses)

Work
	•	Add a dedicated test scene + camera preset (your 256×256 harness is perfect).
	•	In CI:
	1.	Render modes 0, 23–27
	2.	Assert:
	•	23–27 are pairwise non-identical
	•	sparkle/energy metric improvements hold vs baseline
	•	mode26 has sufficient dynamic range (e.g., p95 - p05 > epsilon)

Deliverables
	•	tests/test_flake_diagnosis.py upgraded with:
	•	pairwise image difference assertions
	•	metric persistence thresholds
	•	reports/flake_diag/latest/ artifact bundle written by CI (optional but ideal)

⸻

Immediate next action (what I would do first)

Milestone A. Until 23–27 are truly distinct, every conclusion drawn from them is shaky. Your current output already proves the issue: modes 23–27 are identical pixel-for-pixel, so the shader isn’t branching per-mode the way you think.
