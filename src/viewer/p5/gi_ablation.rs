// src/viewer/p5/gi_ablation.rs
// P5.4 GI stack ablation capture methods
// Split from gi.rs as part of the viewer refactoring

use anyhow::Context;
use serde_json::json;
use std::fs;
use std::path::Path;

use crate::core::screen_space_effects::ScreenSpaceEffect as SSE;
use crate::util::image_write;

use super::super::Viewer;

impl Viewer {
    pub(crate) fn capture_p54_gi_stack_ablation(&mut self) -> anyhow::Result<()> {
        let out_dir = Path::new("reports/p5");
        fs::create_dir_all(out_dir)?;

        let capture_w = self.config.width.max(1);
        let capture_h = self.config.height.max(1);

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
            combined[dst_off..dst_off + row_stride]
                .copy_from_slice(&baseline_bytes[src_off..src_off + row_stride]);
            combined[dst_off + row_stride..dst_off + row_stride * 2]
                .copy_from_slice(&ao_bytes[src_off..src_off + row_stride]);
            combined[dst_off + row_stride * 2..dst_off + row_stride * 3]
                .copy_from_slice(&ao_ssgi_bytes[src_off..src_off + row_stride]);
            combined[dst_off + row_stride * 3..dst_off + row_stride * 4]
                .copy_from_slice(&ao_ssgi_ssr_bytes[src_off..src_off + row_stride]);
        }

        let out_path = out_dir.join("p5_gi_stack_ablation.png");
        image_write::write_png_rgba8_small(&out_path, &combined, out_w, out_h)?;
        println!("[P5] Wrote {}", out_path.display());

        // Record GI composition parameters and timings into p5_meta.json
        self.write_gi_composition_meta()?;

        // Restore original GI state and re-render
        self.restore_gi_state(ao_orig, ssgi_orig, ssr_orig)?;
        self.gi_ao_weight = ao_weight_orig;
        self.gi_ssgi_weight = ssgi_weight_orig;
        self.gi_ssr_weight = ssr_weight_orig;
        self.ssr_params.set_enabled(ssr_enable_orig);
        self.sync_ssr_params_to_gi();
        self.reexecute_gi(None)?;

        self.compute_p54_gi_verification()?;
        Ok(())
    }

    fn write_gi_composition_meta(&mut self) -> anyhow::Result<()> {
        let gi = self.gi.as_ref().context("GI manager not available")?;
        let ao_enable = gi.is_enabled(SSE::SSAO);
        let ssgi_enable = gi.is_enabled(SSE::SSGI);
        let ssr_enable = gi.is_enabled(SSE::SSR) && self.ssr_params.ssr_enable;

        let (ao_kernel_ms, ao_blur_ms, ao_temporal_ms) =
            gi.ssao_timings_ms().unwrap_or((0.0, 0.0, 0.0));
        let ao_total_ms = ao_kernel_ms + ao_blur_ms + ao_temporal_ms;

        let (ssgi_trace_ms, ssgi_shade_ms, ssgi_temporal_ms, ssgi_upsample_ms) =
            gi.ssgi_timings_ms().unwrap_or((0.0, 0.0, 0.0, 0.0));
        let ssgi_total_ms = ssgi_trace_ms + ssgi_shade_ms + ssgi_temporal_ms + ssgi_upsample_ms;

        let (ssr_trace_ms, ssr_shade_ms, ssr_fallback_ms) =
            gi.ssr_timings_ms().unwrap_or((0.0, 0.0, 0.0));
        let ssr_total_ms = ssr_trace_ms + ssr_shade_ms + ssr_fallback_ms;

        let composite_ms = self
            .gi_pass
            .as_ref()
            .map(|p| p.composite_ms())
            .unwrap_or(0.0);
        let hzb_ms = gi.hzb_ms();

        let gpu_hzb_ms = self.gi_gpu_hzb_ms;
        let gpu_ssao_ms = self.gi_gpu_ssao_ms;
        let gpu_ssgi_ms = self.gi_gpu_ssgi_ms;
        let gpu_ssr_ms = self.gi_gpu_ssr_ms;
        let gpu_composite_ms = self.gi_gpu_composite_ms;

        let hzb_measured_ms = if gpu_hzb_ms > 0.0 { gpu_hzb_ms } else { hzb_ms };
        let ssao_measured_ms = if gpu_ssao_ms > 0.0 {
            gpu_ssao_ms
        } else {
            ao_total_ms
        };
        let ssgi_measured_ms = if gpu_ssgi_ms > 0.0 {
            gpu_ssgi_ms
        } else {
            ssgi_total_ms
        };
        let ssr_measured_ms = if gpu_ssr_ms > 0.0 {
            gpu_ssr_ms
        } else {
            ssr_total_ms
        };
        let composite_measured_ms = if gpu_composite_ms > 0.0 {
            gpu_composite_ms
        } else {
            composite_ms
        };

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
        {
            "OK"
        } else {
            "REGRESSION"
        };

        let gpu_timing_supported = self
            .gi_timing
            .as_ref()
            .map(|t| t.is_supported())
            .unwrap_or(false);

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
        Ok(())
    }

    fn restore_gi_state(&mut self, ao: bool, ssgi: bool, ssr: bool) -> anyhow::Result<()> {
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
}
