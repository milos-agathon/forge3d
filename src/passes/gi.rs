// src/passes/gi.rs
// P5.4: GI composition orchestration (design-only for Milestone 2).
//
// This file currently serves as documentation for the planned GI composite pass
// and its uniform parameters. No Rust code is implemented yet; wiring of the
// actual pass will happen in later milestones.
//
// High-level role of GiPass (future work)
// --------------------------------------
// A future `GiPass` type in this module will:
//   * Own or reference the existing AO, SSGI, and SSR renderers.
//   * Ensure those passes produce well-defined intermediate textures:
//       - ao_resolved      : R32Float AO scalar in [0,1].
//       - ssgi_resolved    : Rgba16Float diffuse GI radiance (rgb, HDR).
//       - ssr_final        : Rgba16Float specular reflection (rgb, HDR) with
//                            alpha used as a hit mask (a > 0 for surface hits,
//                            a = 0 for env-only).
//       - baseline_lighting: HDR buffer containing L_diffuse_base + L_spec_base
//                            (+ emissive) *before* AO/SSGI/SSR.
//   * Bind these textures, plus the G-buffer and a uniform buffer, into a single
//     compute pipeline using `src/shaders/gi/composite.wgsl`.
//   * Dispatch the composite kernel once per frame to produce the final lighting
//     buffer used by the P5 harness and viewer.
//
// Planned GI composite params uniform
// -----------------------------------
// The composite kernel is controlled by a small uniform buffer that encodes
// toggles, quality knobs, and the global energy cap. On the Rust side this will
// be represented as a struct like:
//
//   pub struct GiCompositeParams {
//       // Feature toggles (exposed to CLI / viewer; mapped to 0/1 on GPU).
//       pub ao_enable: bool;   // AO multiplier on diffuse (Milestone 2: design only).
//       pub ssgi_enable: bool; // SSGI diffuse GI term.
//       pub ssr_enable: bool;  // SSR specular replacement / lerp.
//
//       // Quality knobs / strengths in [0,1].
//       pub ao_weight: f32;    // 0 = no AO effect, 1 = full AO; interpolates in-between.
//       pub ssgi_weight: f32;  // Scales diffuse GI radiance before energy capping.
//       pub ssr_weight: f32;   // Scales SSR blend factor (in addition to Fresnel/roughness).
//
//       // Global energy budget relative to baseline+IBL.
//       // The composite pass enforces:
//       //   luminance(L_final) <= energy_cap * luminance(L_baseline)
//       // for almost all pixels, by scaling only the *excess* SSGI contribution.
//       pub energy_cap: f32;   // Expected default: 1.05 (5% over baseline+IBL).
//   }
//
// GPU-side (WGSL) representation
// ------------------------------
// To be compatible with WGSL std140-like layout rules, the booleans will be
// encoded as `u32` values (0 = false, 1 = true) and the struct will be padded
// to 16-byte alignment. The corresponding WGSL struct will look like:
//
//   struct GiCompositeParams {
//       ao_enable:   u32;  // 0 or 1
//       ssgi_enable: u32;  // 0 or 1
//       ssr_enable:  u32;  // 0 or 1
//       _pad0:       u32;  // padding to 16 bytes
//
//       ao_weight:   f32;  // [0,1]
//       ssgi_weight: f32;  // [0,1]
//       ssr_weight:  f32;  // [0,1]
//       energy_cap:  f32;  // >= 1.0, typically 1.05
//   };
//
// The composite shader (see src/shaders/gi/composite.wgsl) will read this
// uniform and apply the formulas documented there:
//   * AO multiplies diffuse only using (ao_enable, ao_weight).
//   * SSGI adds to diffuse only using (ssgi_enable, ssgi_weight) and scales its
//     extra contribution so that per-pixel luminance stays within `energy_cap`
//     times the baseline+IBL luminance.
//   * SSR replaces/lerps specular only using (ssr_enable, ssr_weight) combined
//     with roughness and Fresnel, and uses the SSR alpha channel to distinguish
//     true surface hits from env-only fallbacks.
//
// Later milestones will turn this design into an actual `GiPass` implementation
// that owns the compute pipeline, uploads `GiCompositeParams` each frame, and
// wires the AO/SSGI/SSR textures into the composite dispatch.
