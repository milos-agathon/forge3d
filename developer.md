Below is a step-by-step roadmap with the first milestone fully specified (inputs, tasks, checks, and exact deliverables). After you ship M1’s images, send them over and I’ll audit.

M1 — Dielectric Cook-Torrance GGX (direct light) ✅ one scene, end-to-end

Goal: Produce physically plausible, gamma-corrected PNGs for a gray dielectric sphere lit by a single directional light, with roughness sweep and component debugs (NDF, G, spec radiance). This fixes the core issues (exposure/gamma, roughness binding, denominator stability, correct NDF/G/DFG math).

Scope (what’s in / out)
	•	In: GGX microfacet BRDF, Schlick Fresnel, Smith GGX G, stable denominator, linear→sRGB output, per-tile roughness binding, debug views.
	•	Out (later milestones): Principled energy compensation, metals, IBL/LUTs, tone-mapping variants, performance work.

Scene + parameters (hardcoded for the gallery)
	•	Geometry: UV sphere, radius 1.
	•	Camera: view dir V = normalize(0, 0, 1) (screen-space forward).
	•	Normal: from sphere.
	•	Light: directional L = normalize(0.5, 0.5, 1).
	•	Dielectric: F0 = 0.04 (RGB scalar); diffuse = baseColor / π, baseColor = 0.5 gray.
	•	Roughness sweep (tiles): r ∈ {0.10, 0.30, 0.50, 0.70, 0.90}, α = r².
	•	Radiance: Li = 3.0 (to keep mid-tones visible).
	•	Output: sRGB-encoded PNG (no tone map for M1).

Shader math (reference)
	•	Half-vector: H = normalize(L + V).
	•	Dot terms: NL = saturate(dot(N,L)), NV = saturate(dot(N,V)), NH = saturate(dot(N,H)), VH = saturate(dot(V,H)).
	•	GGX NDF:
a = α
a2 = a*a
den = (NH*NH)*(a2-1.0)+1.0
D = a2 / (π * den*den)
	•	Smith GGX G1:
k = (a + 1)^2 / 8  (or use GGX exact form; pick one and use consistently)
G1(x) = x / (x*(1.0 - k) + k) with x ∈ {NL,NV}
	•	Joint G: G = G1(NL) * G1(NV)
	•	Schlick Fresnel: F = F0 + (1-F0) * pow(1 - VH, 5)
	•	Specular BRDF:
den_spec = max(4.0 * NL * NV, 1e-4)
f_spec = (D * F * G) / den_spec
	•	Diffuse BRDF: f_diff = (baseColor / π)
	•	Radiance: Lo = (f_spec + f_diff) * Li * NL
	•	Encode: srgb = linear_to_srgb(Lo) (per channel with sRGB OETF).

Deliverables
	1.	m1_brdf_gallery_ggx.png — 1 row × 5 tiles (r values) showing full shaded result with titles like GGX  r=0.30  α=0.0900.
	2.	m1_debug_ndf.png — same layout, visualizing D(N,H) remapped for visibility: D / max_D(α) (no lighting).
	3.	m1_debug_g.png — G(N,L,V) grayscale.
	4.	m1_debug_spec_radiance.png — (D*F*G/den_spec) * Li * NL (i.e., radiance contribution of the specular lobe only).
	5.	m1_meta.json — the exact numeric inputs used per tile (r, α, Li, F0), and a hash of the GPU constants per draw (to prove bindings changed).

Naming: keep exactly these names so audits stay frictionless.

Acceptance tests (must pass)
	•	A1 – Visibility: No gallery tile is “nearly black.” Mean sRGB luminance per tile ∈ [75, 140] for Li=3.0.
	•	A2 – Roughness monotonicity: In m1_debug_spec_radiance.png, the area above a fixed threshold (e.g., linear radiance > 0.5) increases as r increases.
	•	A3 – NDF shape: m1_debug_ndf.png shows a peaked lobe that widens with r; not a flat disc.
	•	A4 – No blow-ups: In linear space, no NaNs/Infs and no values > 32 before encode. (Log violations to console.)
	•	A5 – Binding sanity: m1_meta.json shows different α per tile and the on-frame captions match those values.

Tasks (do in order)
	•	T1. Linear→sRGB encode utility and ensure all PNGs pass through it (no tone map).
	•	T2. Robust dot/epsilon helpers (saturate, shared EPS=1e-4), used everywhere.
	•	T3. Implement GGX D/G/F exactly as above (pick Smith k-form or exact, but be consistent).
	•	T4. Stable denominator den_spec = max(4*NL*NV, 1e-4) used in a single function.
	•	T5. Dielectric path (F0=0.04, diffuse = base/π).
	•	T6. Per-tile params via push constants (roughness, alpha, F0, baseColor, Li). Avoid UBO dynamic offset issues.
	•	T7. Debug view switch (full, NDF, G, spec-radiance).
	•	T8. Gallery generator that renders the 5 tiles into a single atlas with on-frame captions.
	•	T9. Meta dump of bound constants per tile.
	•	T10. Sanity script to compute mean luminance and threshold area (can be CPU-side; print to console).

Submission checklist (quick self-audit)
	•	PNGs look mid-gray, not crushed.
	•	Highlights widen & dim as r increases.
	•	NDF debug shows a proper lobe.
	•	Spec radiance has no pin-prick clips; smooth lobe.
	•	m1_meta.json exists and values match captions.

⸻

M2 — Disney Principled (Burley) with energy compensation (next after M1)

Goal: Add Principled diffuse (Burley), specular energy compensation, and match GGX at r=0.5 within ±5% mean luminance after compensation.

Deliverables: m2_brdf_gallery_principled.png, m2_debug_energy.png, updated meta.json.

Exit checks: mean luminance at high roughness slightly higher than GGX; metals still pending.

⸻

M3 — Metal workflow (RGB F0, zero diffuse)

Goal: Switchable dielectric/metal; verify diffuse=0 for metals, F0 = baseColor RGB.

Deliverables: paired GGX/Principled metal galleries.

Exit checks: No diffuse leakage; spec color matches baseColor.

⸻

M4 — Preintegrated IBL (DFG LUT + env sampling)

Goal: Split-sum IBL (Lambertian irradiance + specular prefilter + DFG LUT).

Deliverables: environment-lit galleries, dfg_lut.png.

Exit checks: energy-conserving behavior across r; no seam artifacts.

⸻

M5 — Color management & tone mapping

Goal: ACES and Reinhard toggles; linear pipeline remains testable.

Deliverables: side-by-side tone map comparison, m5_tonemap_compare.png.

⸻

M6 — Validation harness

Goal: CPU reference GGX at 32×32 probe points; compare GPU within 1e-3 RMS.

Deliverables: CSV diffs, CI step.

⸻

M7 — Extensions & performance

Goal: Clearcoat, sheen (Charlie), and a basic perf pass (batching, UBO layout, barriers).
