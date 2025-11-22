// src/shaders/gi/composite.wgsl
// P5.4: Screen-space GI composition (design-only for Milestone 2).
//
// This file currently contains only commentary describing the intended composition
// math for AO, SSGI, and SSR. No executable WGSL is present yet; the actual
// compute kernel will be implemented in Milestone 3.
//
// Terminology and inputs (conceptual)
// -----------------------------------
// We conceptually split the baseline lighting (before any GI) into:
//   L_diffuse_base : direct diffuse + diffuse IBL
//   L_spec_base    : direct specular + specular IBL
//   L_emissive     : emissive term (unaffected by AO/SSGI/SSR)
//   L_baseline     : L_diffuse_base + L_spec_base + L_emissive
//
// The GI passes produce:
//   ao       : scalar ambient occlusion from SSAO/GTAO in [0,1]
//              (1 = fully visible, 0 = fully occluded).
//   L_ssgi   : diffuse GI radiance from SSGI (RGB, linear HDR, diffuse-only).
//   L_spec_ssr_raw : specular reflection from SSR + environment (RGB, HDR).
//   ssr_alpha      : alpha channel from SSR final buffer, used as hit mask:
//                    > 0 for valid SSR surface hits, 0 for env-only misses.
//
// The composite pass is controlled by a uniform struct `GiCompositeParams`
// (see src/passes/gi.rs for the Rust-side description) with fields:
//   ao_enable, ssgi_enable, ssr_enable : toggles (0/1 on GPU).
//   ao_weight, ssgi_weight, ssr_weight : quality/strength knobs in [0,1].
//   energy_cap                         : scalar cap, expected 1.05.
//
// The final composed lighting per pixel is:
//   L_final = L_diffuse_gi + L_spec_final + L_emissive
// with the constraints that:
//   * AO and SSGI only affect L_diffuse_* (diffuse component).
//   * SSR only affects L_spec_*          (specular component).
//   * luminance(L_final) <= energy_cap * luminance(L_baseline)
//     for almost all pixels, preventing GI from adding more than ~5% energy
//     relative to baseline+IBL.
//
// 1. AO (diffuse only)
// --------------------
// AO is applied as a scalar multiplier on the diffuse base lighting only.
// We introduce an "effective AO" factor that respects the AO toggle and weight:
//
//   ao_raw in [0,1] := sampled AO texture value (1 = no occlusion).
//   ao_weight in [0,1] := quality knob (1 = full AO, 0 = disabled via weight).
//   ao_enable in {0,1} := hard toggle from uniform.
//
// First compute a weighted AO factor between 1.0 (no AO) and ao_raw:
//   ao_weighted = mix(1.0, ao_raw, ao_weight)
//               = 1.0 * (1 - ao_weight) + ao_raw * ao_weight.
//
// Then gate AO entirely based on the toggle:
//   effective_ao = select(1.0, ao_weighted, ao_enable == 1)
//
// Finally apply AO only to the diffuse base term:
//   L_diffuse_ao = L_diffuse_base * effective_ao
//   L_spec_base  remains unchanged here.
//   L_emissive   remains unchanged here.
//
// This guarantees that enabling/disabling AO only modulates the diffuse
// component (up to numerical noise), never specular or emissive.
//
// 2. SSGI (diffuse additive, energy-bounded)
// -----------------------------------------
// SSGI adds diffuse bounce light on top of the AO-darkened diffuse, but we must
// respect a 5% energy budget relative to the baseline+IBL frame.
//
// Inputs and knobs:
//   L_ssgi      : RGB diffuse GI radiance (linear HDR) from SSGI.
//   ssgi_weight : [0,1] quality knob for GI intensity.
//   ssgi_enable : {0,1} hard toggle.
//   energy_cap  : scalar >= 1.0, default 1.05.
//
// Step 1: compute the raw additional diffuse contribution from SSGI:
//   L_ssgi_scaled = ssgi_weight * L_ssgi
//   L_ssgi_masked = (ssgi_enable == 1) ? L_ssgi_scaled : vec3(0.0)
//
//   L_diffuse_gi_raw = L_diffuse_ao + L_ssgi_masked
//
// Step 2: define luminance helper (linear RGB):
//   luminance(c) = dot(c, vec3(0.2126, 0.7152, 0.0722))
//
// Step 3: derive the maximum allowed luminance *after* GI, expressed as a
//         multiple of the baseline+IBL luminance:
//   L_baseline    = L_diffuse_base + L_spec_base + L_emissive
//   Y_baseline    = luminance(L_baseline)
//   Y_cap_total   = energy_cap * Y_baseline
//
// We also define the state just after AO, before GI:
//   L_after_ao    = L_diffuse_ao + L_spec_base + L_emissive
//   Y_after_ao    = luminance(L_after_ao)
//
// The budget available for GI in luminance space is:
//   Y_budget_gi   = max(Y_cap_total - Y_after_ao, 0.0)
//
// Step 4: measure how much luminance the raw SSGI contribution would add:
//   L_extra       = L_ssgi_masked
//   Y_extra       = luminance(L_extra)
//
// Step 5: compute a scale factor that only affects the extra GI term, leaving
//         the baseline (including AO) untouched:
//   eps           = 1e-4
//   scale_extra   =
//       if Y_extra <= eps then 1.0
//       else clamp(Y_budget_gi / Y_extra, 0.0, 1.0)
//
//   L_diffuse_gi  = L_diffuse_ao + scale_extra * L_extra
//
// This scaling scheme:
//   * Preserves the hue of the GI contribution (uniform scalar on RGB).
//   * Only scales down the *excess* GI when it would push the pixel above the
//     allowed energy_cap; the baseline (direct + IBL + AO) is not dimmed.
//   * If there is no remaining budget (Y_budget_gi <= 0), GI is effectively
//     clamped to zero for that pixel.
//
// After this stage, the intermediate lighting is:
//   L_after_gi = L_diffuse_gi + L_spec_base + L_emissive
// with luminance bounded by:
//   luminance(L_after_gi) <= energy_cap * luminance(L_baseline)
// (up to numerical precision and any QA tolerance on histograms).
//
// 3. SSR (specular replacement / lerp)
// ------------------------------------
// SSR replaces or blends the specular component using roughness and Fresnel,
// and it must only affect the specular term.
//
// Inputs (per pixel):
//   L_spec_base    : baseline specular (direct + IBL) from PBR shading.
//   L_spec_ssr_raw : specular reflection from SSR+env buffer (rgb).
//   ssr_alpha      : alpha from SSR buffer (>0 for surface hits, 0 for env-only).
//   roughness      : perceptual roughness in [0,1] from G-buffer.
//   F0             : base reflectance at normal incidence (from metallic / albedo).
//   V, N           : view and normal vectors for Fresnel.
//   ssr_weight     : user weight [0,1].
//   ssr_enable     : toggle {0,1}.
//
// Step 1: derive a Fresnel factor k_fresnel consistent with the existing BRDF:
//   k_fresnel = fresnel_schlick(dot(N, V), F0)   // RGB or scalar; we use scalar
//              ≈ mix(1.0, F0_scalar, (1 - dot(N, V))^5)
//
// Step 2: define a roughness-dependent fade-out so that very rough surfaces
//         rely mostly on the original specular (SSR suppressed):
//   n        : small integer in [2,4], controls how quickly SSR fades with roughness.
//   k_rough  = (1.0 - roughness)^n
//
// Step 3: combine Fresnel, roughness, user weight, toggle, and hit mask into a
//         single blend factor k_ssr in [0,1]:
//
//   hit      = step(0.0, ssr_alpha)   // 1.0 for SSR surface hits, 0.0 for env-only.
//   w_toggle = (ssr_enable == 1) ? 1.0 : 0.0
//
//   k_ssr    = clamp(hit * w_toggle * ssr_weight * k_fresnel * k_rough, 0.0, 1.0)
//
// For env-only misses (hit = 0), k_ssr = 0 and we keep L_spec_base; this avoids
// double-counting IBL when the SSR buffer contains only an environment fallback.
//
// Step 4: mix baseline specular with SSR specular using k_ssr:
//   L_spec_final = mix(L_spec_base, L_spec_ssr_raw, k_ssr)
//
// By construction, SSR only modifies the specular term, leaving the diffuse and
// emissive components untouched.
//
// 4. Final composition and component isolation
// -------------------------------------------
// After applying AO, SSGI, and SSR as above, the final HDR lighting per pixel is:
//
//   L_final = L_diffuse_gi + L_spec_final + L_emissive
//
// with the intended properties:
//   * AO path (ao_enable/ao_weight) only changes L_diffuse_*.
//   * SSGI path (ssgi_enable/ssgi_weight) only adds to L_diffuse_* and is
//     energy-bounded against baseline+IBL via `energy_cap`.
//   * SSR path (ssr_enable/ssr_weight) only replaces/lerps L_spec_*, controlled
//     by Fresnel, roughness, and valid SSR hits.
//
// The Milestone 3 implementation of this file will turn these formulas into a
// single WGSL compute entry point that reads the baseline lighting / GI buffers
// and writes the final lighting buffer.
//
// Below is the first implementation of that compute kernel. It closely follows
// the design above but currently treats the entire baseline lighting as the
// diffuse component; L_spec_base and L_emissive are reserved for later
// refinement when the pipeline provides separated components.

struct GiCompositeParams {
    ao_enable:   u32,
    ssgi_enable: u32,
    ssr_enable:  u32,
    _pad0:       u32,
    ao_weight:   f32,
    ssgi_weight: f32,
    ssr_weight:  f32,
    energy_cap:  f32,
};

@group(0) @binding(0) var baseline_lighting: texture_2d<f32>;
@group(0) @binding(1) var diffuse_base_texture: texture_2d<f32>;
@group(0) @binding(2) var spec_base_texture: texture_2d<f32>;
@group(0) @binding(3) var ao_texture: texture_2d<f32>;
@group(0) @binding(4) var ssgi_texture: texture_2d<f32>;
@group(0) @binding(5) var ssr_texture: texture_2d<f32>;
@group(0) @binding(6) var normal_texture: texture_2d<f32>;
@group(0) @binding(7) var material_texture: texture_2d<f32>;
@group(0) @binding(8) var output_lighting: texture_storage_2d<rgba16float, write>;
@group(0) @binding(9) var<uniform> gi_params: GiCompositeParams;

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn fresnel_schlick_scalar(cos_theta: f32, f0: f32) -> f32 {
    let clamped = clamp(cos_theta, 0.0, 1.0);
    return f0 + (1.0 - f0) * pow(1.0 - clamped, 5.0);
}

@compute @workgroup_size(8, 8, 1)
fn cs_gi_composite(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel = gid.xy;
    let dims = textureDimensions(baseline_lighting);
    if (pixel.x >= dims.x || pixel.y >= dims.y) {
        return;
    }

    // Load baseline lighting and separated diffuse/spec components.
    let base_sample = textureLoad(baseline_lighting, pixel, 0);
    let L_baseline = base_sample.rgb;
    let diffuse_sample = textureLoad(diffuse_base_texture, pixel, 0);
    let spec_sample = textureLoad(spec_base_texture, pixel, 0);
    var L_diffuse_base = diffuse_sample.rgb;
    var L_spec_base = spec_sample.rgb;
    // Emissive is reserved for future pipelines that provide a separate term.
    var L_emissive = vec3<f32>(0.0);

    // --- AO (diffuse only) ---
    var ao = textureLoad(ao_texture, pixel, 0).r;
    ao = clamp(ao, 0.0, 1.0);
    let ao_weight = clamp(gi_params.ao_weight, 0.0, 1.0);
    let ao_weighted = mix(1.0, ao, ao_weight);
    let ao_enabled = gi_params.ao_enable != 0u;
    let effective_ao = select(1.0, ao_weighted, ao_enabled);
    let L_diffuse_ao = L_diffuse_base * effective_ao;

    // --- SSGI (diffuse additive, energy-bounded) ---
    let ssgi_rgb = textureLoad(ssgi_texture, pixel, 0).rgb;
    let ssgi_weight = clamp(gi_params.ssgi_weight, 0.0, 1.0);
    let L_ssgi_scaled = ssgi_rgb * ssgi_weight;
    var L_ssgi_masked = vec3<f32>(0.0);
    if (gi_params.ssgi_enable != 0u) {
        L_ssgi_masked = L_ssgi_scaled;
    }

    let L_diffuse_gi_raw = L_diffuse_ao + L_ssgi_masked;

    // Energy budget relative to baseline+IBL.
    let energy_cap = max(gi_params.energy_cap, 1.0);
    let L_after_ao = L_diffuse_ao + L_spec_base + L_emissive;
    let Y_after_ao = luminance(L_after_ao);
    let L_baseline_total = L_diffuse_base + L_spec_base + L_emissive;
    let Y_baseline = luminance(L_baseline_total);
    let Y_cap_total = energy_cap * Y_baseline;
    let Y_budget_gi = max(Y_cap_total - Y_after_ao, 0.0);

    let L_extra = L_ssgi_masked;
    let Y_extra = luminance(L_extra);

    var scale_extra = 1.0;
    let eps = 1e-4;
    if (Y_extra > eps) {
        scale_extra = clamp(Y_budget_gi / Y_extra, 0.0, 1.0);
    }

    let L_diffuse_gi = L_diffuse_ao + L_extra * scale_extra;

    // --- SSR (specular replacement / lerp) ---
    let ssr_sample = textureLoad(ssr_texture, pixel, 0);
    let L_spec_ssr_raw = ssr_sample.rgb;
    let ssr_alpha = ssr_sample.a;

    // Material properties from G-buffer for F0 and roughness.
    let normal_sample = textureLoad(normal_texture, pixel, 0);
    let roughness = clamp(normal_sample.w, 0.0, 1.0);
    let material_sample = textureLoad(material_texture, pixel, 0);
    let albedo = material_sample.rgb;
    let metallic = clamp(material_sample.a, 0.0, 1.0);

    let dielectric_f0 = vec3<f32>(0.04, 0.04, 0.04);
    let f0_rgb = mix(dielectric_f0, albedo, vec3<f32>(metallic));
    let f0_scalar = clamp((f0_rgb.r + f0_rgb.g + f0_rgb.b) / 3.0, 0.02, 0.98);

    // View-agnostic Fresnel approximation: cos_theta ≈ 1.
    let k_fresnel = fresnel_schlick_scalar(1.0, f0_scalar);
    let n_power: f32 = 2.0;
    let k_rough = pow(max(1.0 - roughness, 0.0), n_power);

    let hit = select(0.0, 1.0, ssr_alpha > 0.0);
    let w_toggle = select(0.0, 1.0, gi_params.ssr_enable != 0u);
    let ssr_weight = clamp(gi_params.ssr_weight, 0.0, 1.0);
    let k_ssr = clamp(hit * w_toggle * ssr_weight * k_fresnel * k_rough, 0.0, 1.0);

    let L_spec_final = mix(L_spec_base, L_spec_ssr_raw, k_ssr);

    // Final composition
    var L_final = L_diffuse_gi + L_spec_final + L_emissive;
    L_final = max(L_final, vec3<f32>(0.0));

    // Basic NaN / Inf defense: if any channel is NaN or extremely large, zero it out.
    if (any(L_final != L_final) || any(L_final > vec3<f32>(1e9)) || any(L_final < vec3<f32>(-1e9))) {
        L_final = vec3<f32>(0.0);
    }

    textureStore(output_lighting, pixel, vec4<f32>(L_final, base_sample.a));
}
