// src/viewer/p5/gi_verification.rs
// P5.4 GI verification methods
// Split from gi.rs as part of the viewer refactoring

use anyhow::Context;
use serde_json::json;

use crate::core::screen_space_effects::ScreenSpaceEffect as SSE;

use super::super::image_analysis::read_texture_rgba16_to_rgb_f32;
use super::super::Viewer;

impl Viewer {
    pub(crate) fn compute_p54_gi_verification(&mut self) -> anyhow::Result<()> {
        let (ao_orig, ssgi_orig, ssr_orig) = {
            let gi = self.gi.as_ref().context("GI manager not available")?;
            (
                gi.is_enabled(SSE::SSAO),
                gi.is_enabled(SSE::SSGI),
                gi.is_enabled(SSE::SSR),
            )
        };
        let ao_weight_orig = self.gi_ao_weight;
        let ssgi_weight_orig = self.gi_ssgi_weight;
        let ssr_weight_orig = self.gi_ssr_weight;
        let ssr_enable_orig = self.ssr_params.ssr_enable;
        let dims = (self.config.width.max(1), self.config.height.max(1));

        // Baseline: all GI effects disabled
        {
            let gi = self.gi.as_mut().context("GI manager not available")?;
            gi.disable_effect(SSE::SSAO);
            gi.disable_effect(SSE::SSGI);
            gi.disable_effect(SSE::SSR);
        }
        self.ssr_params.set_enabled(false);
        self.sync_ssr_params_to_gi();
        self.reexecute_gi(None)?;
        let baseline_hdr =
            read_texture_rgba16_to_rgb_f32(&self.device, &self.queue, &self.gi_output_hdr, dims)?;
        let baseline_diffuse = read_texture_rgba16_to_rgb_f32(
            &self.device,
            &self.queue,
            &self.gi_baseline_diffuse_hdr,
            dims,
        )?;
        let baseline_spec = read_texture_rgba16_to_rgb_f32(
            &self.device,
            &self.queue,
            &self.gi_baseline_spec_hdr,
            dims,
        )?;

        // AO only
        {
            let gi = self.gi.as_mut().context("GI manager not available")?;
            gi.enable_effect(&self.device, SSE::SSAO)?;
            gi.disable_effect(SSE::SSGI);
            gi.disable_effect(SSE::SSR);
        }
        self.ssr_params.set_enabled(false);
        self.sync_ssr_params_to_gi();
        self.reexecute_gi(None)?;
        let ao_hdr =
            read_texture_rgba16_to_rgb_f32(&self.device, &self.queue, &self.gi_output_hdr, dims)?;

        // AO + SSGI
        {
            let gi = self.gi.as_mut().context("GI manager not available")?;
            gi.enable_effect(&self.device, SSE::SSAO)?;
            gi.enable_effect(&self.device, SSE::SSGI)?;
            gi.disable_effect(SSE::SSR);
        }
        self.ssr_params.set_enabled(false);
        self.sync_ssr_params_to_gi();
        self.reexecute_gi(None)?;
        let ao_ssgi_hdr =
            read_texture_rgba16_to_rgb_f32(&self.device, &self.queue, &self.gi_output_hdr, dims)?;

        // AO + SSGI + SSR
        {
            let gi = self.gi.as_mut().context("GI manager not available")?;
            gi.enable_effect(&self.device, SSE::SSAO)?;
            gi.enable_effect(&self.device, SSE::SSGI)?;
            gi.enable_effect(&self.device, SSE::SSR)?;
        }
        self.ssr_params.set_enabled(true);
        self.sync_ssr_params_to_gi();
        self.reexecute_gi(None)?;
        let ao_ssgi_ssr_hdr =
            read_texture_rgba16_to_rgb_f32(&self.device, &self.queue, &self.gi_output_hdr, dims)?;

        // Restore original GI state
        self.restore_gi_verification_state(ao_orig, ssgi_orig, ssr_orig)?;
        self.gi_ao_weight = ao_weight_orig;
        self.gi_ssgi_weight = ssgi_weight_orig;
        self.gi_ssr_weight = ssr_weight_orig;
        self.ssr_params.set_enabled(ssr_enable_orig);
        self.sync_ssr_params_to_gi();
        self.reexecute_gi(None)?;

        // Compute and write verification metrics
        self.compute_gi_verification_metrics(
            &baseline_hdr,
            &baseline_diffuse,
            &baseline_spec,
            &ao_hdr,
            &ao_ssgi_hdr,
            &ao_ssgi_ssr_hdr,
            dims,
        )
    }

    fn restore_gi_verification_state(
        &mut self,
        ao: bool,
        ssgi: bool,
        ssr: bool,
    ) -> anyhow::Result<()> {
        let gi = self.gi.as_mut().context("GI manager not available")?;
        if ao {
            gi.enable_effect(&self.device, SSE::SSAO)?;
        } else {
            gi.disable_effect(SSE::SSAO);
        }
        if ssgi {
            gi.enable_effect(&self.device, SSE::SSGI)?;
        } else {
            gi.disable_effect(SSE::SSGI);
        }
        if ssr {
            gi.enable_effect(&self.device, SSE::SSR)?;
        } else {
            gi.disable_effect(SSE::SSR);
        }
        Ok(())
    }

    fn compute_gi_verification_metrics(
        &mut self,
        baseline_hdr: &[[f32; 3]],
        baseline_diffuse: &[[f32; 3]],
        baseline_spec: &[[f32; 3]],
        ao_hdr: &[[f32; 3]],
        ao_ssgi_hdr: &[[f32; 3]],
        ao_ssgi_ssr_hdr: &[[f32; 3]],
        dims: (u32, u32),
    ) -> anyhow::Result<()> {
        let count = (dims.0 as usize) * (dims.1 as usize);

        // Energy check in HDR
        let mut max_luminance_ratio = 0.0f32;
        let mut luminance_violations = 0u64;
        let mut luminance_samples = 0u64;
        let eps_base = 1e-6f32;

        for i in 0..count {
            let [br, bg, bb] = baseline_hdr[i];
            let [gr, gg, gb] = ao_ssgi_ssr_hdr[i];
            let yb = 0.2126 * br + 0.7152 * bg + 0.0722 * bb;
            let ya = 0.2126 * gr + 0.7152 * gg + 0.0722 * gb;
            if yb > eps_base {
                let ratio = ya / yb.max(eps_base);
                if ratio.is_finite() {
                    luminance_samples += 1;
                    if ratio > max_luminance_ratio {
                        max_luminance_ratio = ratio;
                    }
                    if ratio > 1.05 + 1e-4 {
                        luminance_violations += 1;
                    }
                }
            }
        }

        let violation_fraction = if luminance_samples > 0 {
            luminance_violations as f32 / luminance_samples as f32
        } else {
            0.0
        };

        // Component isolation metrics in HDR
        let mut max_diffuse_delta_ao = 0.0f32;
        let mut max_diffuse_delta_ssgi = 0.0f32;
        let mut max_spec_delta_ssr = 0.0f32;
        let mut max_unintended_diffuse_delta_ssr = 0.0f32;
        let mut max_unintended_spec_delta_ao = 0.0f32;
        let mut max_unintended_spec_delta_ssgi = 0.0f32;

        for i in 0..count {
            let [bd_r, bd_g, bd_b] = baseline_diffuse[i];
            let [bs_r, bs_g, bs_b] = baseline_spec[i];
            let [ar, ag, ab] = ao_hdr[i];
            let [sgi_r, sgi_g, sgi_b] = ao_ssgi_hdr[i];
            let [sr, sg, sb] = ao_ssgi_ssr_hdr[i];

            // AO: compare baseline vs AO using separated diffuse/spec
            let diffuse_ao_r = ar - bs_r;
            let diffuse_ao_g = ag - bs_g;
            let diffuse_ao_b = ab - bs_b;
            let d_ao_max = (diffuse_ao_r - bd_r)
                .abs()
                .max((diffuse_ao_g - bd_g).abs().max((diffuse_ao_b - bd_b).abs()));
            if d_ao_max > max_diffuse_delta_ao {
                max_diffuse_delta_ao = d_ao_max;
            }

            let spec_ao_r = ar - diffuse_ao_r;
            let spec_ao_g = ag - diffuse_ao_g;
            let spec_ao_b = ab - diffuse_ao_b;
            let d_spec_ao_max = (spec_ao_r - bs_r)
                .abs()
                .max((spec_ao_g - bs_g).abs().max((spec_ao_b - bs_b).abs()));
            if d_spec_ao_max > max_unintended_spec_delta_ao {
                max_unintended_spec_delta_ao = d_spec_ao_max;
            }

            // SSGI: compare AO vs AO+SSGI
            let diffuse_ssgi_r = sgi_r - bs_r;
            let diffuse_ssgi_g = sgi_g - bs_g;
            let diffuse_ssgi_b = sgi_b - bs_b;
            let d_ssgi_max = (diffuse_ssgi_r - diffuse_ao_r).abs().max(
                (diffuse_ssgi_g - diffuse_ao_g)
                    .abs()
                    .max((diffuse_ssgi_b - diffuse_ao_b).abs()),
            );
            if d_ssgi_max > max_diffuse_delta_ssgi {
                max_diffuse_delta_ssgi = d_ssgi_max;
            }

            let spec_ssgi_r = sgi_r - diffuse_ssgi_r;
            let spec_ssgi_g = sgi_g - diffuse_ssgi_g;
            let spec_ssgi_b = sgi_b - diffuse_ssgi_b;
            let d_spec_ssgi_max = (spec_ssgi_r - spec_ao_r).abs().max(
                (spec_ssgi_g - spec_ao_g)
                    .abs()
                    .max((spec_ssgi_b - spec_ao_b).abs()),
            );
            if d_spec_ssgi_max > max_unintended_spec_delta_ssgi {
                max_unintended_spec_delta_ssgi = d_spec_ssgi_max;
            }

            // SSR: compare AO+SSGI vs AO+SSGI+SSR
            let spec_ssr_r = sr - diffuse_ssgi_r;
            let spec_ssr_g = sg - diffuse_ssgi_g;
            let spec_ssr_b = sb - diffuse_ssgi_b;
            let spec_mag = (spec_ssr_r - bs_r)
                .abs()
                .max((spec_ssr_g - bs_g).abs().max((spec_ssr_b - bs_b).abs()));
            if spec_mag > max_spec_delta_ssr {
                max_spec_delta_ssr = spec_mag;
            }

            let diffuse_with_ssr_r = sr - spec_ssr_r;
            let diffuse_with_ssr_g = sg - spec_ssr_g;
            let diffuse_with_ssr_b = sb - spec_ssr_b;
            let diff_max = (diffuse_with_ssr_r - diffuse_ssgi_r).abs().max(
                (diffuse_with_ssr_g - diffuse_ssgi_g)
                    .abs()
                    .max((diffuse_with_ssr_b - diffuse_ssgi_b).abs()),
            );
            if diff_max > max_unintended_diffuse_delta_ssr {
                max_unintended_diffuse_delta_ssr = diff_max;
            }
        }

        let max_unintended_component_delta = max_unintended_diffuse_delta_ssr
            .max(max_unintended_spec_delta_ao.max(max_unintended_spec_delta_ssgi));
        let tolerance = 1.0f32 / 255.0f32;

        self.write_p5_meta(|meta| {
            meta.insert("gi_verification".to_string(), json!({
                "luminance": { "max_ratio": max_luminance_ratio, "violation_count": luminance_violations, "violation_fraction": violation_fraction },
                "component_isolation": {
                    "ao": { "max_diffuse_delta": max_diffuse_delta_ao, "max_unintended_spec_delta": max_unintended_spec_delta_ao },
                    "ssgi": { "max_diffuse_delta": max_diffuse_delta_ssgi, "max_unintended_spec_delta": max_unintended_spec_delta_ssgi },
                    "ssr": { "max_spec_delta": max_spec_delta_ssr, "max_unintended_diffuse_delta": max_unintended_diffuse_delta_ssr },
                },
                "max_unintended_component_delta": max_unintended_component_delta,
                "tolerance_1_over_255": tolerance,
            }));
        })?;
        Ok(())
    }
}
