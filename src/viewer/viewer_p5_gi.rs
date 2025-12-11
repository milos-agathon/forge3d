// src/viewer/viewer_p5_gi.rs
// P5.4 GI stack ablation and verification methods
// RELEVANT FILES: src/viewer/mod.rs

use anyhow::Context;
use serde_json::json;
use std::fs;
use std::path::Path;

use crate::core::screen_space_effects::ScreenSpaceEffect as SSE;
use crate::util::image_write;

use super::image_analysis::read_texture_rgba16_to_rgb_f32;
use super::Viewer;

impl Viewer {
    pub(crate) fn capture_p54_gi_stack_ablation(&mut self) -> anyhow::Result<()> {
        let out_dir = Path::new("reports/p5");
        fs::create_dir_all(out_dir)?;

        let capture_w = self.config.width.max(1);
        let capture_h = self.config.height.max(1);

        let (ao_orig, ssgi_orig, ssr_orig) = {
            let gi = self.gi.as_ref().context("GI manager not available")?;
            (gi.is_enabled(SSE::SSAO), gi.is_enabled(SSE::SSGI), gi.is_enabled(SSE::SSR))
        };
        let ao_weight_orig = self.gi_ao_weight;
        let ssgi_weight_orig = self.gi_ssgi_weight;
        let ssr_weight_orig = self.gi_ssr_weight;
        let ssr_enable_orig = self.ssr_params.ssr_enable;

        // 1) Baseline: all GI effects off
        {
            let gi = self.gi.as_mut().context("GI manager not available")?;
            gi.disable_effect(SSE::SSAO);
            gi.disable_effect(SSE::SSGI);
            gi.disable_effect(SSE::SSR);
        }
        self.ssr_params.set_enabled(false);
        self.sync_ssr_params_to_gi();
        self.reexecute_gi(None)?;
        let baseline_bytes = self.capture_gi_output_tonemapped_rgba8()?;

        // 2) AO only
        {
            let gi = self.gi.as_mut().context("GI manager not available")?;
            gi.enable_effect(&self.device, SSE::SSAO)?;
            gi.disable_effect(SSE::SSGI);
            gi.disable_effect(SSE::SSR);
        }
        self.ssr_params.set_enabled(false);
        self.sync_ssr_params_to_gi();
        self.reexecute_gi(None)?;
        let ao_bytes = self.capture_gi_output_tonemapped_rgba8()?;

        // 3) AO + SSGI
        {
            let gi = self.gi.as_mut().context("GI manager not available")?;
            gi.enable_effect(&self.device, SSE::SSAO)?;
            gi.enable_effect(&self.device, SSE::SSGI)?;
            gi.disable_effect(SSE::SSR);
        }
        self.ssr_params.set_enabled(false);
        self.sync_ssr_params_to_gi();
        self.reexecute_gi(None)?;
        let ao_ssgi_bytes = self.capture_gi_output_tonemapped_rgba8()?;

        // 4) AO + SSGI + SSR
        {
            let gi = self.gi.as_mut().context("GI manager not available")?;
            gi.enable_effect(&self.device, SSE::SSAO)?;
            gi.enable_effect(&self.device, SSE::SSGI)?;
            gi.enable_effect(&self.device, SSE::SSR)?;
        }
        self.ssr_params.set_enabled(true);
        self.sync_ssr_params_to_gi();
        self.reexecute_gi(None)?;
        let ao_ssgi_ssr_bytes = self.capture_gi_output_tonemapped_rgba8()?;

        // Assemble 4-column ablation image
        let out_w = capture_w * 4;
        let out_h = capture_h;
        let mut combined = vec![0u8; (out_w * out_h * 4) as usize];
        let row_stride = (capture_w as usize) * 4;
        for y in 0..(capture_h as usize) {
            let dst_off = y * (out_w as usize) * 4;
            let src_off = y * row_stride;
            combined[dst_off..dst_off + row_stride].copy_from_slice(&baseline_bytes[src_off..src_off + row_stride]);
            combined[dst_off + row_stride..dst_off + row_stride * 2].copy_from_slice(&ao_bytes[src_off..src_off + row_stride]);
            combined[dst_off + row_stride * 2..dst_off + row_stride * 3].copy_from_slice(&ao_ssgi_bytes[src_off..src_off + row_stride]);
            combined[dst_off + row_stride * 3..dst_off + row_stride * 4].copy_from_slice(&ao_ssgi_ssr_bytes[src_off..src_off + row_stride]);
        }

        let out_path = out_dir.join("p5_gi_stack_ablation.png");
        image_write::write_png_rgba8_small(&out_path, &combined, out_w, out_h)?;
        println!("[P5] Wrote {}", out_path.display());

        // Record GI composition parameters and timings into p5_meta.json
        {
            let gi = self.gi.as_ref().context("GI manager not available")?;
            let ao_enable = gi.is_enabled(SSE::SSAO);
            let ssgi_enable = gi.is_enabled(SSE::SSGI);
            let ssr_enable = gi.is_enabled(SSE::SSR) && self.ssr_params.ssr_enable;

            let (ao_kernel_ms, ao_blur_ms, ao_temporal_ms) = gi.ssao_timings_ms().unwrap_or((0.0, 0.0, 0.0));
            let ao_total_ms = ao_kernel_ms + ao_blur_ms + ao_temporal_ms;

            let (ssgi_trace_ms, ssgi_shade_ms, ssgi_temporal_ms, ssgi_upsample_ms) = gi.ssgi_timings_ms().unwrap_or((0.0, 0.0, 0.0, 0.0));
            let ssgi_total_ms = ssgi_trace_ms + ssgi_shade_ms + ssgi_temporal_ms + ssgi_upsample_ms;

            let (ssr_trace_ms, ssr_shade_ms, ssr_fallback_ms) = gi.ssr_timings_ms().unwrap_or((0.0, 0.0, 0.0));
            let ssr_total_ms = ssr_trace_ms + ssr_shade_ms + ssr_fallback_ms;

            let composite_ms = self.gi_pass.as_ref().map(|p| p.composite_ms()).unwrap_or(0.0);
            let hzb_ms = gi.hzb_ms();

            let gpu_hzb_ms = self.gi_gpu_hzb_ms;
            let gpu_ssao_ms = self.gi_gpu_ssao_ms;
            let gpu_ssgi_ms = self.gi_gpu_ssgi_ms;
            let gpu_ssr_ms = self.gi_gpu_ssr_ms;
            let gpu_composite_ms = self.gi_gpu_composite_ms;

            let hzb_measured_ms = if gpu_hzb_ms > 0.0 { gpu_hzb_ms } else { hzb_ms };
            let ssao_measured_ms = if gpu_ssao_ms > 0.0 { gpu_ssao_ms } else { ao_total_ms };
            let ssgi_measured_ms = if gpu_ssgi_ms > 0.0 { gpu_ssgi_ms } else { ssgi_total_ms };
            let ssr_measured_ms = if gpu_ssr_ms > 0.0 { gpu_ssr_ms } else { ssr_total_ms };
            let composite_measured_ms = if gpu_composite_ms > 0.0 { gpu_composite_ms } else { composite_ms };

            // P5.6 performance budgets (RTX 3060 / 1080p)
            let ssao_budget_ms: f32 = 1.6;
            let ssgi_budget_ms: f32 = 2.8;
            let ssr_budget_ms: f32 = 2.2;
            let hzb_budget_ms: f32 = 0.5;
            let btc_budget_ms: f32 = 1.2;

            let ssao_delta_ms = ssao_measured_ms - ssao_budget_ms;
            let ssgi_delta_ms = ssgi_measured_ms - ssgi_budget_ms;
            let ssr_delta_ms = ssr_measured_ms - ssr_budget_ms;
            let hzb_delta_ms = hzb_measured_ms - hzb_budget_ms;

            let btc_measured_ms = ao_blur_ms + ao_temporal_ms + composite_measured_ms;
            let btc_delta_ms = btc_measured_ms - btc_budget_ms;

            let p56_status = if hzb_measured_ms <= hzb_budget_ms
                && ssao_measured_ms <= ssao_budget_ms
                && ssgi_measured_ms <= ssgi_budget_ms
                && ssr_measured_ms <= ssr_budget_ms
                && btc_measured_ms <= btc_budget_ms
            { "OK" } else { "REGRESSION" };

            let gpu_timing_supported = self.gi_timing.as_ref().map(|t| t.is_supported()).unwrap_or(false);

            self.write_p5_meta(|meta| {
                meta.insert("gi_composition".to_string(), json!({
                    "order": ["baseline", "ao", "ssgi", "ssr"],
                    "weights": { "ao_weight": self.gi_ao_weight, "ssgi_weight": self.gi_ssgi_weight, "ssr_weight": self.gi_ssr_weight },
                    "toggles": { "ao_enable": ao_enable, "ssgi_enable": ssgi_enable, "ssr_enable": ssr_enable },
                    "timings_ms": { "ao": ao_total_ms, "ssgi": ssgi_total_ms, "ssr": ssr_total_ms, "composite": composite_ms, "hzb": hzb_ms },
                    "gpu_ms": { "hzb": gpu_hzb_ms, "ssao": gpu_ssao_ms, "ssgi": gpu_ssgi_ms, "ssr": gpu_ssr_ms, "composite": gpu_composite_ms },
                    "gpu_timing": { "supported": gpu_timing_supported },
                    "perf_budgets": {
                        "hzb": { "budget_ms": hzb_budget_ms, "measured_ms": hzb_measured_ms, "delta_ms": hzb_delta_ms, "within_budget": hzb_measured_ms <= hzb_budget_ms },
                        "ssao": { "budget_ms": ssao_budget_ms, "measured_ms": ssao_measured_ms, "delta_ms": ssao_delta_ms, "within_budget": ssao_measured_ms <= ssao_budget_ms },
                        "ssgi": { "budget_ms": ssgi_budget_ms, "measured_ms": ssgi_measured_ms, "delta_ms": ssgi_delta_ms, "within_budget": ssgi_measured_ms <= ssgi_budget_ms },
                        "ssr": { "budget_ms": ssr_budget_ms, "measured_ms": ssr_measured_ms, "delta_ms": ssr_delta_ms, "within_budget": ssr_measured_ms <= ssr_budget_ms },
                        "bilateral_temporal_composite": { "budget_ms": btc_budget_ms, "measured_ms": btc_measured_ms, "delta_ms": btc_delta_ms, "within_budget": btc_measured_ms <= btc_budget_ms },
                    },
                }));
                meta.insert("p56_status".to_string(), json!(p56_status));
            })?;
        }

        // Restore original GI state and re-render
        {
            let gi = self.gi.as_mut().context("GI manager not available")?;
            if ao_orig { gi.enable_effect(&self.device, SSE::SSAO)?; } else { gi.disable_effect(SSE::SSAO); }
            if ssgi_orig { gi.enable_effect(&self.device, SSE::SSGI)?; } else { gi.disable_effect(SSE::SSGI); }
            if ssr_orig { gi.enable_effect(&self.device, SSE::SSR)?; } else { gi.disable_effect(SSE::SSR); }
        }
        self.gi_ao_weight = ao_weight_orig;
        self.gi_ssgi_weight = ssgi_weight_orig;
        self.gi_ssr_weight = ssr_weight_orig;
        self.ssr_params.set_enabled(ssr_enable_orig);
        self.sync_ssr_params_to_gi();
        self.reexecute_gi(None)?;

        self.compute_p54_gi_verification()?;
        Ok(())
    }

    pub(crate) fn compute_p54_gi_verification(&mut self) -> anyhow::Result<()> {
        let (ao_orig, ssgi_orig, ssr_orig) = {
            let gi = self.gi.as_ref().context("GI manager not available")?;
            (gi.is_enabled(SSE::SSAO), gi.is_enabled(SSE::SSGI), gi.is_enabled(SSE::SSR))
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
        let baseline_hdr = read_texture_rgba16_to_rgb_f32(&self.device, &self.queue, &self.gi_output_hdr, dims)?;
        let baseline_diffuse = read_texture_rgba16_to_rgb_f32(&self.device, &self.queue, &self.gi_baseline_diffuse_hdr, dims)?;
        let baseline_spec = read_texture_rgba16_to_rgb_f32(&self.device, &self.queue, &self.gi_baseline_spec_hdr, dims)?;

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
        let ao_hdr = read_texture_rgba16_to_rgb_f32(&self.device, &self.queue, &self.gi_output_hdr, dims)?;

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
        let ao_ssgi_hdr = read_texture_rgba16_to_rgb_f32(&self.device, &self.queue, &self.gi_output_hdr, dims)?;

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
        let ao_ssgi_ssr_hdr = read_texture_rgba16_to_rgb_f32(&self.device, &self.queue, &self.gi_output_hdr, dims)?;

        // Restore original GI state
        {
            let gi = self.gi.as_mut().context("GI manager not available")?;
            if ao_orig { gi.enable_effect(&self.device, SSE::SSAO)?; } else { gi.disable_effect(SSE::SSAO); }
            if ssgi_orig { gi.enable_effect(&self.device, SSE::SSGI)?; } else { gi.disable_effect(SSE::SSGI); }
            if ssr_orig { gi.enable_effect(&self.device, SSE::SSR)?; } else { gi.disable_effect(SSE::SSR); }
        }
        self.gi_ao_weight = ao_weight_orig;
        self.gi_ssgi_weight = ssgi_weight_orig;
        self.gi_ssr_weight = ssr_weight_orig;
        self.ssr_params.set_enabled(ssr_enable_orig);
        self.sync_ssr_params_to_gi();
        self.reexecute_gi(None)?;

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
                    if ratio > max_luminance_ratio { max_luminance_ratio = ratio; }
                    if ratio > 1.05 + 1e-4 { luminance_violations += 1; }
                }
            }
        }

        let violation_fraction = if luminance_samples > 0 { luminance_violations as f32 / luminance_samples as f32 } else { 0.0 };

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
            let d_ao_r = (diffuse_ao_r - bd_r).abs();
            let d_ao_g = (diffuse_ao_g - bd_g).abs();
            let d_ao_b = (diffuse_ao_b - bd_b).abs();
            let d_ao_max = d_ao_r.max(d_ao_g.max(d_ao_b));
            if d_ao_max > max_diffuse_delta_ao { max_diffuse_delta_ao = d_ao_max; }

            let spec_ao_r = ar - diffuse_ao_r;
            let spec_ao_g = ag - diffuse_ao_g;
            let spec_ao_b = ab - diffuse_ao_b;
            let d_spec_ao_r = (spec_ao_r - bs_r).abs();
            let d_spec_ao_g = (spec_ao_g - bs_g).abs();
            let d_spec_ao_b = (spec_ao_b - bs_b).abs();
            let d_spec_ao_max = d_spec_ao_r.max(d_spec_ao_g.max(d_spec_ao_b));
            if d_spec_ao_max > max_unintended_spec_delta_ao { max_unintended_spec_delta_ao = d_spec_ao_max; }

            // SSGI: compare AO vs AO+SSGI
            let diffuse_ssgi_r = sgi_r - bs_r;
            let diffuse_ssgi_g = sgi_g - bs_g;
            let diffuse_ssgi_b = sgi_b - bs_b;
            let d_ssgi_r = (diffuse_ssgi_r - diffuse_ao_r).abs();
            let d_ssgi_g = (diffuse_ssgi_g - diffuse_ao_g).abs();
            let d_ssgi_b = (diffuse_ssgi_b - diffuse_ao_b).abs();
            let d_ssgi_max = d_ssgi_r.max(d_ssgi_g.max(d_ssgi_b));
            if d_ssgi_max > max_diffuse_delta_ssgi { max_diffuse_delta_ssgi = d_ssgi_max; }

            let spec_ssgi_r = sgi_r - diffuse_ssgi_r;
            let spec_ssgi_g = sgi_g - diffuse_ssgi_g;
            let spec_ssgi_b = sgi_b - diffuse_ssgi_b;
            let d_spec_ssgi_r = (spec_ssgi_r - spec_ao_r).abs();
            let d_spec_ssgi_g = (spec_ssgi_g - spec_ao_g).abs();
            let d_spec_ssgi_b = (spec_ssgi_b - spec_ao_b).abs();
            let d_spec_ssgi_max = d_spec_ssgi_r.max(d_spec_ssgi_g.max(d_spec_ssgi_b));
            if d_spec_ssgi_max > max_unintended_spec_delta_ssgi { max_unintended_spec_delta_ssgi = d_spec_ssgi_max; }

            // SSR: compare AO+SSGI vs AO+SSGI+SSR
            let spec_ssr_r = sr - diffuse_ssgi_r;
            let spec_ssr_g = sg - diffuse_ssgi_g;
            let spec_ssr_b = sb - diffuse_ssgi_b;
            let spec_delta_r = (spec_ssr_r - bs_r).abs();
            let spec_delta_g = (spec_ssr_g - bs_g).abs();
            let spec_delta_b = (spec_ssr_b - bs_b).abs();
            let spec_mag = spec_delta_r.max(spec_delta_g.max(spec_delta_b));
            if spec_mag > max_spec_delta_ssr { max_spec_delta_ssr = spec_mag; }

            let diffuse_with_ssr_r = sr - spec_ssr_r;
            let diffuse_with_ssr_g = sg - spec_ssr_g;
            let diffuse_with_ssr_b = sb - spec_ssr_b;
            let diff_r = (diffuse_with_ssr_r - diffuse_ssgi_r).abs();
            let diff_g = (diffuse_with_ssr_g - diffuse_ssgi_g).abs();
            let diff_b = (diffuse_with_ssr_b - diffuse_ssgi_b).abs();
            let diff_max = diff_r.max(diff_g.max(diff_b));
            if diff_max > max_unintended_diffuse_delta_ssr { max_unintended_diffuse_delta_ssr = diff_max; }
        }

        let max_unintended_component_delta = max_unintended_diffuse_delta_ssr.max(max_unintended_spec_delta_ao.max(max_unintended_spec_delta_ssgi));
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
