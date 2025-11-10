P5 — Screen-Space Effects (SSAO/GTAO, SSGI, SSR)

Milestone plan with precise deliverables, file touch-points, commands, and acceptance checks.

⸻

P5.0 – Foundation: G-Buffer, Framegraph hooks, toggles

Scope
	•	Add a post-lit framegraph branch gi_passes that is entirely optional and no-op when disabled.
	•	Introduce a consistent G-Buffer for screen-space passes: depth (with mip chain), view-space normals, roughness/metallic, albedo, and (optional) motion vectors for temporal filters.

Implementation
	•	WGSL
	•	src/shaders/gbuffer/pack.wgsl – writes:
	•	RG16F normal_vs (oct-encoded),
	•	R8G8 roughness_metallic,
	•	optional RGBA8 albedo,
	•	depth in default depth target.
	•	src/shaders/gbuffer/common.wgsl – helpers: oct decode, view-pos reconstruct from depth.
	•	src/shaders/hzb_build.wgsl – build hierarchical Z (min-depth) mip chain.
	•	Rust
	•	src/passes/gbuffer.rs – render pass producing the above targets.
	•	src/passes/hzb.rs – compute mips for depth.
	•	src/framegraph.rs – add_gi_passes() node after main lighting, before tonemap.
	•	src/cli/args.rs – --gi <ssao|ssgi|ssr> with sane defaults.
	•	examples/interactive_viewer.rs – route :gi .../params to runtime state.

Deliverables
	•	Build succeeds; viewer runs with :viz normal|depth|material|gi.
	•	Artifacts (PNG, RGB, opaque) in reports/p5/:
	•	p5_gbuffer_normals.png, p5_gbuffer_depth_mips.png (grid of 5 mips), p5_gbuffer_material.png.
	•	reports/p5/p5_meta.json with render target formats, sizes, and sha256 of shader sets.

Acceptance
	•	Depth mip 0..4 present and monotone (per-mip min(depth) non-decreasing).
	•	Normals visualize correctly (no NaNs; unit length within ±1e-3 after decode).
	•	Toggling :gi off produces a frame bit-identical to pre-P5 baseline.

⸻

P5.1 – SSAO/GTAO + bilateral blur + (opt) temporal

Scope
	•	Implement SSAO first, then switchable GTAO kernel (same interface).
	•	Edge-aware bilateral blur (2-pass separable).
	•	Optional temporal accumulation using motion vectors (if available); otherwise box temporal.

Implementation
	•	WGSL
	•	src/shaders/ssao/ssao.wgsl – view-space hemisphere sampling (blue-noise seeded).
	•	src/shaders/ssao/gtao.wgsl – horizon-based term per Burley; shares common includes.
	•	src/shaders/filters/bilateral_separable.wgsl – depth+normal weights.
	•	src/shaders/temporal/resolve_ao.wgsl – history reprojection + clamp to neighborhood mins/max.
	•	Rust
	•	src/passes/ssao.rs – dispatch AO, blur, (opt) temporal, composite to gi.ao.
	•	Parameters in GiParams: ssao_radius, ssao_intensity, ssao_technique.
	•	Viewer commands: :ssao-technique, :ssao-radius, :ssao-intensity, :ssao-composite on|off, :ssao-mul.

Deliverables
	•	reports/p5/p5_ssao_cornell.png – Cornell box, AO on vs off split-view.
	•	reports/p5/p5_ssao_params_grid.png – radius × intensity grid (3×3).
	•	reports/p5/p5_meta.json updated with AO kernel, sample count, timings (ms).

Acceptance
	•	Crease darkening: mean luminance in corner ROI at least 10% lower than flat wall ROI with AO on; ≤2% delta when AO off.
	•	Bilateral blur removes ≥70% of high-freq AO noise (FFT energy test) with ≤2% edge leakage across depth discontinuities.
	•	AO toggle does not change specular highlights (verify max specular pixel ±1/255).

⸻

P5.2 – SSGI (half-res tracing, fallback to IBL) + temporal + upsample

Scope
	•	Half-resolution ray-march in view-space along bent normal; gather diffuse radiance from previous frame color + IBL fallback.
	•	Temporal accumulation with clamping; edge-aware upsample to full-res.

Implementation
	•	WGSL
	•	src/shaders/ssgi/trace.wgsl – sphere-traced steps (:ssgi-steps, :ssgi-radius), hierarchical depth test via HZB.
	•	src/shaders/ssgi/shade.wgsl – sample previous-frame color buffer (ping-pong) + diffuse IBL fallback when miss.
	•	src/shaders/ssgi/resolve_temporal.wgsl – alpha blend with neighborhood clamp.
	•	src/shaders/filters/edge_aware_upsample.wgsl – depth/normal guided upsample to full-res.
	•	Rust
	•	src/passes/ssgi.rs – manages half-res targets, history, and upsample.
	•	Hot params mapped to CLI / viewer (steps, radius, half, temporal-alpha, upsample sigmas).

Deliverables
	•	reports/p5/p5_ssgi_cornell.png – shows visible red/green bounce between Cornell box walls (reference albedos set in scene).
	•	reports/p5/p5_ssgi_temporal_compare.png – single-frame vs 16-frame accumulated.
	•	Updated p5_meta.json with miss ratio, avg steps, accumulation alpha, perf (ms).

Acceptance
	•	Wall bounce: red-wall ROI gets +5–12% luminance increase on adjacent neutral wall with SSGI on (vs off), saturating below ACES clamp.
	•	Stability: frame-to-frame SSIM ≥ 0.95 after 8 frames with camera static.
	•	Fallback check: when :ssgi-steps 0, result equals diffuse IBL (ΔE ≤ 1.0).

⸻

P5.3 – SSR (HZB-assisted or thickness test) + Fresnel/roughness weighting

Scope
	•	Screen-space specular reflections with hierarchical Z or thickness approach; glossy weighting via roughness; fallback to env reflection.

Implementation
	•	WGSL
	•	src/shaders/ssr/trace.wgsl – reflect view ray about normal; screen-space march with HZB; --ssr-max-steps, --ssr-thickness.
	•	src/shaders/ssr/shade.wgsl – sample current color (or pre-tonemap spec buffer) and blend with Fresnel Schlick and roughness‐based cone.
	•	src/shaders/ssr/fallback_env.wgsl – IBL reflection used when miss or outside frustum.
	•	Rust
	•	src/passes/ssr.rs – targets/history; composition order before tonemap, after direct specular.

Deliverables
	•	reports/p5/p5_ssr_glossy_spheres.png – row of spheres (r=0.1→0.9) reflecting a bright stripe and sky.
	•	reports/p5/p5_ssr_thickness_ablation.png – shows undershoot artifacts without thickness vs fixed with thickness.
	•	Meta: hit-rate %, avg steps, miss→IBL ratio, perf (ms).

Acceptance
	•	Reflectivity scales with roughness: mean reflected stripe contrast drops monotonically (r=0.1..0.9).
	•	Fallback: pixels tagged “miss” visually match IBL reflection (ΔE ≤ 2) and no black holes (min RGB ≥ 2/255).
	•	Edges: no >1-px bright streaks at depth discontinuities after resolve.

⸻

P5.4 – Composition & Ordering + Quality knobs

Scope
	•	Centralize composition of AO, SSGI, SSR with physically-sane weights:
	•	AO multiplies diffuse only.
	•	SSGI adds to diffuse (clamped to energy budget).
	•	SSR replaces/lerps specular by roughness/Fresnel.

Implementation
	•	WGSL
	•	src/shaders/gi/composite.wgsl – single compute kernel blending gi.ao, gi.ssgi, gi.ssr into the lighting buffer.
	•	Rust
	•	src/passes/gi.rs – orchestrates sub-passes; toggles and parameter propagation.

Deliverables
	•	reports/p5/p5_gi_stack_ablation.png – 4 columns: baseline, +AO, +AO+SSGI, +AO+SSGI+SSR.
	•	p5_meta.json includes composition order, weights, and final timings.

Acceptance
	•	Composition never raises pixel energy above baseline+IBL by >5% (histogram cap).
	•	Turning each effect on/off affects only its intended component (diffuse vs spec) within ±1/255 tolerance.

⸻

P5.5 – CLI & Viewer integration (end-to-end)

Scope
	•	Wire all CLI flags and viewer commands, including :viz gi mode that shows a 3-channel debug (AO, SSGI, SSR) or selectable single channel.

Implementation
	•	Rust
	•	src/cli/args.rs – --gi ssao|ssgi|ssr, --ssao-*, --ssgi-*, --ssr-*.
	•	examples/interactive_viewer.rs – :gi, :viz gi, :snapshot, depth range tools.

Deliverables
	•	Golden command scripts in scripts/p5_golden.sh|.bat that produce all artifacts.
	•	tests/golden/ PNGs + tests/test_p5_cli.rs (parses, sets, round-trips state).

Acceptance
	•	All flags map 1:1 to viewer commands; querying state echoes the last set values.
	•	Snapshots generated as opaque PNG (no alpha), deterministic hashes across runs with the same seed.

⸻

P5.6 – Performance pass & budgets

Targets (RTX 3060 / 1080p)
	•	SSAO/GTAO: ≤1.6 ms
	•	SSGI (half-res): ≤2.8 ms (8–16 steps)
	•	SSR (HZB): ≤2.2 ms (max 32 steps)
	•	HZB build: ≤0.5 ms
	•	Bilateral/Temporal/Composite combined: ≤1.2 ms

Implementation
	•	Integrate simple GPU timer queries; record per-pass ms in p5_meta.json.

Acceptance
	•	Each pass at or below budget in the Cornell scene; otherwise report regression with deltas.

⸻

P5.7 – Automated acceptance tests

Tests (pytest + image metrics)
	•	tests/test_p5_ssao.py – ROI luminance deltas (crease vs flat) ≥10%.
	•	tests/test_p5_ssgi.py – Cornell bounce + SSIM temporal stability ≥0.95 after 8 frames.
	•	tests/test_p5_ssr.py – roughness-contrast monotonicity and no black holes.
	•	tests/test_p5_cli.rs – CLI round-trip.

Artifacts
	•	On pass, write reports/p5/p5_PASS.txt with hashed metrics; on fail, emit side-by-side diffs.

⸻

P5.8 – Documentation & examples

Docs
	•	docs/p5_gi_passes.md – diagrams (G-Buffer, HZB, ray-march), parameter tables, troubleshooting (light leaks, speckle, banding).
	•	examples/gi_showcase.rs – minimal scene toggling AO/SSGI/SSR with on-screen perf HUD.

Acceptance
	•	Doc includes before/after imagery, parameter min/max, and per-pass pitfalls.
	•	Example builds and runs on Windows/macOS/Linux.

⸻

Quick-Start Commands

# Build viewer
cargo run --release --example interactive_viewer -- \
  --ibl assets/snow_field_4k.hdr --gi ssao:on --ssao-radius 0.5 --ssao-intensity 1.0

# SSGI demo (half-res, temporal)
cargo run --release --example interactive_viewer -- \
  --gi ssgi:on --ssgi-half on --ssgi-steps 12 --ssgi-radius 0.8 --ssgi-temporal-alpha 0.1

# SSR demo
cargo run --release --example interactive_viewer -- \
  --gi ssr:on --ssr-max-steps 24 --ssr-thickness 0.2


⸻

File Map (new/modified)

src/
  passes/
    gbuffer.rs
    hzb.rs
    ssao.rs
    ssgi.rs
    ssr.rs
    gi.rs
  cli/args.rs        (extend)
  framegraph.rs      (add_gi_passes)
  ...
src/shaders/
  gbuffer/{pack.wgsl,common.wgsl}
  hzb_build.wgsl
  ssao/{ssao.wgsl,gtao.wgsl}
  ssgi/{trace.wgsl,shade.wgsl,resolve_temporal.wgsl}
  ssr/{trace.wgsl,shade.wgsl,fallback_env.wgsl}
  filters/{bilateral_separable.wgsl,edge_aware_upsample.wgsl}
  temporal/resolve_ao.wgsl
  gi/composite.wgsl
examples/
  interactive_viewer.rs  (commands)
  gi_showcase.rs
scripts/
  p5_golden.sh|.bat
reports/p5/               (artifacts, meta)
tests/
  test_p5_ssao.py
  test_p5_ssgi.py
  test_p5_ssr.py
  test_p5_cli.rs
docs/
  p5_gi_passes.md


⸻

Definition of Done (P5 overall)
	•	All milestones’ Acceptance sections pass on CI for the Cornell and Glossy Spheres scenes.
	•	Artifacts exist and are PNG, opaque; p5_meta.json contains formats, timings, and metrics.
	•	Toggling each GI effect on/off is hot (no rebuild), and the pipeline is bit-identical with all GI off compared to pre-P5 baseline.