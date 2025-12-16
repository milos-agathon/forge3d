// src/viewer/p5/ssgi_cornell.rs
// P5.2 SSGI Cornell capture methods
// Split from core.rs as part of the viewer refactoring

use anyhow::Context;
use half::f16;
use serde_json::json;
use std::fs;
use std::path::Path;

use crate::core::screen_space_effects::ScreenSpaceEffect as SSE;

use super::super::image_analysis::compute_max_delta_e;
use super::super::viewer_constants::{
    P52_MAX_MEGAPIXELS, P5_SSGI_CORNELL_WARMUP_FRAMES, P5_SSGI_DIFFUSE_SCALE,
};
use super::super::viewer_image_utils::downscale_rgba8_bilinear;
use super::super::Viewer;

impl Viewer {
    pub(crate) fn capture_p52_ssgi_cornell(&mut self) -> anyhow::Result<()> {
        let out_dir = Path::new("reports/p5");
        fs::create_dir_all(out_dir)?;

        let state = self.setup_p51_cornell_scene()?;
        self.render_geometry_to_gbuffer_once()?;

        let capture_w = self.config.width.max(1);
        let capture_h = self.config.height.max(1);
        let capture_is_srgb = matches!(
            self.config.format,
            wgpu::TextureFormat::Rgba8UnormSrgb | wgpu::TextureFormat::Bgra8UnormSrgb
        );

        let was_enabled = {
            let gi = self.gi.as_ref().context("GI manager not available")?;
            gi.is_enabled(SSE::SSGI)
        };
        if !was_enabled {
            let gi = self.gi.as_mut().context("GI manager not available")?;
            gi.enable_effect(&self.device, SSE::SSGI)?;
        }

        let original_settings = {
            let gi = self.gi.as_ref().context("GI manager not available")?;
            gi.ssgi_settings().context("SSGI settings unavailable")?
        };

        {
            let gi = self.gi.as_mut().context("GI manager not available")?;
            gi.disable_effect(SSE::SSGI);
        }
        self.reexecute_gi(None)?;
        let off_bytes = self.capture_material_rgba8()?;

        {
            let gi = self.gi.as_mut().context("GI manager not available")?;
            gi.enable_effect(&self.device, SSE::SSGI)?;
            gi.update_ssgi_settings(&self.queue, |s| {
                *s = original_settings;
            });
            gi.ssgi_reset_history(&self.device, &self.queue)?;
            gi.set_ssgi_composite_intensity(&self.queue, P5_SSGI_DIFFUSE_SCALE);
        }
        for _ in 0..P5_SSGI_CORNELL_WARMUP_FRAMES {
            self.reexecute_gi(None)?;
        }
        let on_bytes = self.capture_material_rgba8()?;

        // Combine split image and write
        let combined = self.combine_ssgi_split_image(&off_bytes, &on_bytes, capture_w, capture_h);
        self.write_ssgi_cornell_image(&combined, capture_w * 2, capture_h, out_dir)?;

        // Metrics
        let (miss_ratio, avg_steps) = self.compute_ssgi_miss_metrics(&original_settings)?;
        let (trace_ms, shade_ms, temporal_ms, upsample_ms) = {
            let gi = self.gi.as_ref().context("GI manager not available")?;
            gi.ssgi_timings_ms().unwrap_or((0.0, 0.0, 0.0, 0.0))
        };

        // Wall bounce measurement
        let (bounce_red_pct, bounce_green_pct) =
            self.compute_wall_bounce(&off_bytes, &on_bytes, capture_w, capture_h, capture_is_srgb);

        // ΔE fallback test
        let max_delta_e = self.compute_ssgi_delta_e_fallback(&original_settings)?;

        // Restore settings
        {
            let gi = self.gi.as_mut().context("GI manager not available")?;
            gi.update_ssgi_settings(&self.queue, |s| {
                *s = original_settings;
            });
            gi.set_ssgi_composite_intensity(&self.queue, 1.0);
            if !was_enabled {
                gi.disable_effect(SSE::SSGI);
            }
        }

        self.write_p5_meta(|meta| {
            meta.insert(
                "ssgi".to_string(),
                json!({
                    "miss_ratio": miss_ratio,
                    "avg_steps": avg_steps,
                    "accumulation_alpha": original_settings.temporal_alpha,
                    "perf_ms": {
                        "trace_ms": trace_ms,
                        "shade_ms": shade_ms,
                        "temporal_ms": temporal_ms,
                        "upsample_ms": upsample_ms,
                        "total_ssgi_ms": trace_ms + shade_ms + temporal_ms + upsample_ms,
                    },
                    "max_delta_e": max_delta_e,
                }),
            );
            meta.insert(
                "ssgi_bounce".to_string(),
                json!({
                    "red_pct": bounce_red_pct,
                    "green_pct": bounce_green_pct,
                }),
            );
        })?;

        self.restore_p51_cornell_scene(state);
        Ok(())
    }

    fn combine_ssgi_split_image(
        &self,
        off_bytes: &[u8],
        on_bytes: &[u8],
        capture_w: u32,
        capture_h: u32,
    ) -> Vec<u8> {
        let out_w = capture_w * 2;
        let out_h = capture_h;
        let mut combined = vec![0u8; (out_w * out_h * 4) as usize];
        let row_stride = (capture_w as usize) * 4;
        for y in 0..(capture_h as usize) {
            let dst_off = y * (out_w as usize) * 4;
            let src_off = y * row_stride;
            combined[dst_off..dst_off + row_stride]
                .copy_from_slice(&off_bytes[src_off..src_off + row_stride]);
            combined[dst_off + row_stride..dst_off + row_stride * 2]
                .copy_from_slice(&on_bytes[src_off..src_off + row_stride]);
        }
        combined
    }

    fn write_ssgi_cornell_image(
        &self,
        combined: &[u8],
        out_w: u32,
        out_h: u32,
        out_dir: &Path,
    ) -> anyhow::Result<()> {
        let write_buf: Vec<u8>;
        let (final_w, final_h, data_ref): (u32, u32, &[u8]) = {
            let px = (out_w as u64 as f64) * (out_h as u64 as f64);
            let max_px = (P52_MAX_MEGAPIXELS * 1_000_000.0) as f64;
            if px > max_px {
                let scale = (max_px / px).sqrt().clamp(0.0, 1.0);
                let dw = (out_w as f64 * scale).floor().max(1.0) as u32;
                let dh = (out_h as f64 * scale).floor().max(1.0) as u32;
                write_buf = downscale_rgba8_bilinear(combined, out_w, out_h, dw, dh);
                (dw, dh, &write_buf)
            } else {
                (out_w, out_h, combined)
            }
        };
        crate::util::image_write::write_png_rgba8_small(
            &out_dir.join("p5_ssgi_cornell.png"),
            data_ref,
            final_w,
            final_h,
        )?;
        if final_w != out_w || final_h != out_h {
            println!(
                "[P5.2] downscaled SSGI Cornell capture to {}x{} (from {}x{})",
                final_w, final_h, out_w, out_h
            );
        }
        println!("[P5] Wrote reports/p5/p5_ssgi_cornell.png");
        Ok(())
    }

    fn compute_ssgi_miss_metrics(
        &self,
        settings: &crate::core::screen_space_effects::SsgiSettings,
    ) -> anyhow::Result<(f32, f32)> {
        let (hit_bytes, dims) = self.read_ssgi_hit_bytes()?;
        let step_len = {
            let steps = settings.num_steps.max(1) as f32;
            settings.step_size.max(settings.radius / steps)
        };
        let mut miss = 0u64;
        let mut hit = 0u64;
        let mut step_acc = 0.0f64;
        for i in 0..(dims.0 * dims.1) as usize {
            let off = i * 8;
            let dist = f16::from_le_bytes([hit_bytes[off + 4], hit_bytes[off + 5]]).to_f32();
            let mask = f16::from_le_bytes([hit_bytes[off + 6], hit_bytes[off + 7]]).to_f32();
            if mask >= 0.5 {
                hit += 1;
                let steps = if step_len > 0.0 { dist / step_len } else { 0.0 };
                step_acc += steps as f64;
            } else {
                miss += 1;
            }
        }
        let total = (dims.0 as u64) * (dims.1 as u64);
        let miss_ratio = if total > 0 {
            miss as f32 / total as f32
        } else {
            0.0
        };
        let avg_steps = if hit > 0 {
            (step_acc / hit as f64) as f32
        } else {
            0.0
        };
        Ok((miss_ratio, avg_steps))
    }

    fn compute_wall_bounce(
        &self,
        off_bytes: &[u8],
        on_bytes: &[u8],
        capture_w: u32,
        capture_h: u32,
        capture_is_srgb: bool,
    ) -> (f32, f32) {
        const ROI_R_NEUTRAL_BASE: (u32, u32, u32, u32) = (750, 360, 930, 560);
        const ROI_G_NEUTRAL_BASE: (u32, u32, u32, u32) = (950, 360, 1130, 560);
        const BASE_WIDTH: u32 = 1920;
        const BASE_HEIGHT: u32 = 1080;

        let scale_roi = |base: (u32, u32, u32, u32)| -> (u32, u32, u32, u32) {
            let (x0, y0, x1, y1) = base;
            (
                (x0 as f32 * capture_w as f32 / BASE_WIDTH as f32) as u32,
                (y0 as f32 * capture_h as f32 / BASE_HEIGHT as f32) as u32,
                (x1 as f32 * capture_w as f32 / BASE_WIDTH as f32) as u32,
                (y1 as f32 * capture_h as f32 / BASE_HEIGHT as f32) as u32,
            )
        };

        let roi_red = scale_roi(ROI_R_NEUTRAL_BASE);
        let roi_green = scale_roi(ROI_G_NEUTRAL_BASE);

        let compute_roi_luminance = |bytes: &[u8], roi: (u32, u32, u32, u32)| -> f32 {
            let (x0, y0, x1, y1) = roi;
            let mut sum_luma = 0.0f64;
            let mut count = 0u32;
            let to_linear = |c: u8| -> f32 {
                let v = c as f32 / 255.0;
                if capture_is_srgb {
                    if v <= 0.04045 {
                        v / 12.92
                    } else {
                        ((v + 0.055) / 1.055).powf(2.4)
                    }
                } else {
                    v
                }
            };
            for y in y0.min(capture_h)..y1.min(capture_h) {
                for x in x0.min(capture_w)..x1.min(capture_w) {
                    let idx = ((y * capture_w + x) * 4) as usize;
                    if idx + 3 < bytes.len() {
                        let r = to_linear(bytes[idx]);
                        let g = to_linear(bytes[idx + 1]);
                        let b = to_linear(bytes[idx + 2]);
                        sum_luma += (0.2126 * r + 0.7152 * g + 0.0722 * b) as f64;
                        count += 1;
                    }
                }
            }
            if count > 0 {
                (sum_luma / count as f64) as f32
            } else {
                0.0
            }
        };

        let l_r_off = compute_roi_luminance(off_bytes, roi_red);
        let l_r_on = compute_roi_luminance(on_bytes, roi_red);
        let l_g_off = compute_roi_luminance(off_bytes, roi_green);
        let l_g_on = compute_roi_luminance(on_bytes, roi_green);

        let bounce_red_pct = (l_r_on - l_r_off) / l_r_off.max(1e-6);
        let bounce_green_pct = (l_g_on - l_g_off) / l_g_off.max(1e-6);
        (bounce_red_pct, bounce_green_pct)
    }

    fn compute_ssgi_delta_e_fallback(
        &mut self,
        original_settings: &crate::core::screen_space_effects::SsgiSettings,
    ) -> anyhow::Result<f32> {
        {
            let gi = self.gi.as_mut().context("GI manager not available")?;
            gi.enable_effect(&self.device, SSE::SSGI)?;
            gi.update_ssgi_settings(&self.queue, |s| {
                s.num_steps = 0;
                s.step_size = original_settings.step_size;
                s.temporal_alpha = 0.0;
                s.intensity = 1.0;
            });
            gi.ssgi_reset_history(&self.device, &self.queue)?;
        }
        self.reexecute_gi(None)?;
        let first_bytes = self.read_ssgi_filtered_bytes()?.0;

        self.reexecute_gi(None)?;
        let second_bytes = self.read_ssgi_filtered_bytes()?.0;

        Ok(compute_max_delta_e(&second_bytes, &first_bytes))
    }
}
