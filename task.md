You are working in the forge3d repo on P5 – SSR.
Focus on Milestone M2 – baseline glossy SSR trace / shade / fallback and validation, using the existing harnesses and meta writer.

1. Objective

Implement correct screen-space specular reflections for the glossy spheres scene and make all M2 validations pass, so that:
	•	p5_ssr_glossy_reference = forward-only reference (no SSR).
	•	p5_ssr_glossy_spheres = same frame plus a horizontal reflection stripe banded around the spheres, with no extra geometry or artifacts.
	•	p5_ssr_thickness_ablation = clear before/after comparison where undershoot is visible without thickness and gone with thickness.
	•	p5_meta.json reports meaningful SSR metrics and sets ssr_status: "SHADE_READY" when thresholds are met.

2. Files you MUST inspect and edit

Rust:
	•	src/passes/ssr.rs
	•	src/p5/ssr.rs (scene / harness glue, if present)
	•	src/viewer/mod.rs (for capture helpers, but minimize changes here)
	•	src/p5/meta.rs (status + metric analysis)

WGSL shaders:
	•	src/shaders/ssr/trace.wgsl
	•	src/shaders/ssr/shade.wgsl
	•	src/shaders/ssr/fallback_env.wgsl

You may read but should not heavily modify:
	•	Existing SSGI / temporal passes (for pattern reference only).
	•	P5 harnesses:
	•	examples/p5_ssr_glossy.rs
	•	examples/p5_ssr_thickness_ablation.rs

Do not change CLI / meta file names or output paths.

3. Required behaviour – SSR trace & shade (M2)

3.1 Trace – trace.wgsl

Implement a correct screen-space ray march using the existing G-buffer:
	•	Inputs per pixel (from G-buffer):
	•	View-space position P
	•	View-space normal N
	•	Roughness r (0–1) – for M2, you can treat all as perfect mirror; roughness weighting comes in P5.3, but you still need r for metrics per sphere.
	•	Compute view direction V (from camera to P in view space).
	•	Compute reflected direction R = reflect(-V, N) in view space.
	•	Project P and P + t * R into clip space and then into NDC / screen space. Use the existing camera / reprojection helpers used by SSGI.
	•	March along R in screen space using:
	•	--ssr-max-steps (max steps)
	•	--ssr-thickness (acceptable thickness / depth tolerance, in view-space units)
	•	Depth test: consider a hit when P_ray.z is behind the reconstructed depth by less than thickness and in front by more than a small epsilon (avoid self-intersections).
	•	Stop marching when:
	•	You detect a hit (record hit UV and depth).
	•	You leave the viewport (miss).
	•	You exceed max_steps (miss).

Outputs from trace:
	•	For each pixel, write to an SSR trace buffer:
	•	hit_uv (vec2) in 0–1 normalized screen coordinates (only valid if hit).
	•	hit_flag (0 or 1).
	•	Optional: number of steps taken (for avg_steps / total_steps).

Critical: The trace must never directly write color into the main color buffer – only hit data into its own buffer.

3.2 Shade – shade.wgsl

Use the trace output to shade only the original pixels:
	•	If hit_flag == 1:
	•	Sample the pre-tonemap lighting buffer at hit_uv (the buffer that contains forward specular/diffuse lighting; this is already wired as lit_output or similar).
	•	Compute a mirror-like reflection term from that sample: for M2, treat r ~ 0 (glossy spheres) and simply use the sampled color as the reflection contribution.
	•	Combine with the existing forward shading:
	•	color_ssr = color_forward + reflection_weight * sample_color
	•	For M2 you can choose reflection_weight = 1.0 on spheres; later milestones will modulate this by Fresnel and roughness.
	•	If hit_flag == 0:
	•	Do nothing in shade; fallback is handled in fallback_env.wgsl.

Important constraints:
	•	Shade must not re-project or draw the hit surface geometry itself.
	•	No new quads or strips should appear in p5_ssr_glossy_spheres that are not present in the reference. The only visible difference should be a reflection band wrapping around the spheres.

3.3 Fallback – fallback_env.wgsl

For pixels with no SSR hit:
	•	Sample the existing IBL / environment reflection using the same reflection vector R (converted to world space if that’s what the IBL expects).
	•	Add that IBL reflection to the forward color where hit_flag == 0.
	•	Respect existing roughness if the BRDF uses it, but do not add new roughness logic beyond what’s already there; P5.3 will introduce proper cone filtering.

4. Required metrics & thresholds (M2)

Update / use the analysis in src/p5/meta.rs so that p5_meta.json matches these thresholds when the implementation is correct:
	1.	Hit-rate & steps
	•	ssr.hit_rate must be in [0.05, 0.8].
	•	Current value (~0.001) is far too low.
	•	ssr.avg_steps should be > 0 and < ssr_max_steps.
	•	ssr.total_steps should equal the sum of steps over all rays.
	2.	Miss / IBL behaviour
	•	For miss pixels, compare SSR output vs a pure IBL reference:
	•	ssr.max_delta_e_miss ≤ 2.0
	•	ssr.min_rgb_miss ≥ 2 / 255.0` (~0.0078)
	•	This guarantees:
	•	No obvious color mismatch between miss regions and IBL.
	•	No black holes in miss regions.
	3.	Edge streaks
	•	ssr.edge_streaks.num_streaks_gt1px must be 0.
	•	Implement detection as follows:
	•	Compute a binary mask of pixels where SSR contribution is large (e.g., luminance difference vs reference > threshold).
	•	Run a simple connected-component scan along depth discontinuities (where depth gradient exceeds a threshold).
	•	Count any contiguous bright segment wider than 1 pixel as a streak.
	•	Store the count in edge_streaks.num_streaks_gt1px.
	4.	Stripe contrast monotonicity
	•	ssr.stripe_contrast must be a 9-element array, one per sphere ordered by increasing roughness r.
	•	For each sphere:
	•	Extract a small horizontal band ROI that intersects the stripe reflection on that sphere.
	•	Compute stripe contrast as (max_luma - min_luma) in that band, normalized to [0,1].
	•	Enforce:
	•	stripe_contrast[0] > stripe_contrast[1] > ... > stripe_contrast[8] (strictly monotonic decreasing).
	•	stripe_contrast[0] - stripe_contrast[8] ≥ 0.1 (a meaningful visual difference between shiniest and roughest).
	•	If monotonicity or range checks fail, set ssr.status / ssr_status to "SSR_TRACE_FAIL" and include a reason string if there is a field for it.
	5.	Thickness ablation metric
	•	Use p5_ssr_thickness_ablation outputs (thickness off/on) to compute:
	•	ssr.thickness_ablation.undershoot_before
	•	ssr.thickness_ablation.undershoot_after
	•	Suggested algorithm:
	•	Identify a vertical strip beneath the spheres / reflective geometry where floor reflections appear.
	•	For each image (before vs after), compute:
	•	A measure of “undershoot” = average luminance of bright reflection pixels below the true contact edge minus luminance at the contact edge.
	•	Expected behaviour:
	•	undershoot_before > 0.02 (visible undershoot)
	•	undershoot_after ≤ 0.005 (essentially gone).
	•	The ablation PNG should visually show a clear difference: left half with bright stripe leaking below contact, right half with stripe clamped correctly.
	6.	Status field
	•	In p5_meta.json:
	•	When all thresholds above pass, set:
	•	ssr.status = "SHADE_READY"
	•	ssr_status = "SHADE_READY" (top-level alias)
	•	Otherwise, keep or set:
	•	ssr.status = "SSR_TRACE_FAIL"
	•	ssr_status = "SSR_TRACE_FAIL"
	•	Do not fudge the thresholds just to pass; fix the implementation.

5. Visual acceptance – what the images must look like

After your changes, running:
	•	cargo run --release --example p5_ssr_glossy
	•	cargo run --release --example p5_ssr_thickness_ablation

must produce:
	1.	p5_ssr_glossy_reference vs p5_ssr_glossy_spheres
	•	Same camera, same eight/nine spheres row.
	•	Reference: spheres lit by environment, no reflection stripe.
	•	SSR: each sphere has a horizontal reflection stripe band around it (like a light bar reflected on a glossy ball).
	•	No slanted quad or extra geometry anywhere in the frame.
	•	Background / floor unchanged except for correct SSR contributions (if any).
	2.	p5_ssr_thickness_ablation
	•	Side-by-side: left = thickness off, right = thickness on.
	•	Left: visible undershoot / reflection leakage beneath contact edges.
	•	Right: leakage removed; reflections snugly terminate at geometry boundaries.
	•	This visual difference must line up with the numeric undershoot_before / undershoot_after metrics.

6. Final checklist for Codex

When you are done, ensure:
	•	trace.wgsl writes only hit/miss data; no color writes.
	•	shade.wgsl reads hit data, samples pre-tonemap buffer at hit UV, and composites into the original pixel, not at the hit location.
	•	fallback_env.wgsl adds IBL reflection only where hit_flag == 0.
	•	p5_meta hit-rate, miss metrics, stripe_contrast, edge_streaks, and thickness_ablation are all computed, not stubbed.
	•	Running both P5 examples regenerates:
	•	reports/p5/p5_ssr_glossy_reference.*
	•	reports/p5/p5_ssr_glossy_spheres.*
	•	reports/p5/p5_ssr_thickness_ablation.*
	•	reports/p5/p5_meta.json with ssr_status: "SHADE_READY".

Use this as your implementation spec for completing M2.