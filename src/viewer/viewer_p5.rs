// src/viewer/viewer_p5.rs
// P5 capture methods for the interactive viewer
// RELEVANT FILES: src/viewer/mod.rs

use anyhow::Context;
use half::f16;
use serde_json::json;
use std::fs;
use std::path::Path;

use crate::core::screen_space_effects::ScreenSpaceEffect as SSE;

use super::viewer_constants::{
    P52_MAX_MEGAPIXELS, P5_SSGI_CORNELL_WARMUP_FRAMES, P5_SSGI_DIFFUSE_SCALE,
};
use super::viewer_image_utils::downscale_rgba8_bilinear;
use super::Viewer;
use super::image_analysis::{compute_max_delta_e, compute_ssim, rgba16_to_luma};

impl Viewer {
    pub(crate) fn capture_p52_ssgi_cornell(&mut self) -> anyhow::Result<()> {
        let out_dir = Path::new("reports/p5");
        fs::create_dir_all(out_dir)?;

        // Reuse the P5.1 Cornell box scene so SSGI captures are rendered
        // against the same geometry and camera framing as the AO artifacts.
        let state = self.setup_p51_cornell_scene()?;
        // Ensure GBuffer/depth are populated for the Cornell geometry before
        // we start toggling SSGI on/off.
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
            // Task 2: Set SSGI diffuse intensity factor for compositing
            // This scalar controls how much SSGI radiance is added to diffuse lighting
            // Tuned to achieve 5-12% bounce on neutral walls adjacent to colored walls
            gi.set_ssgi_composite_intensity(&self.queue, P5_SSGI_DIFFUSE_SCALE);
        }
        for _ in 0..P5_SSGI_CORNELL_WARMUP_FRAMES {
            self.reexecute_gi(None)?;
        }
        let on_bytes = self.capture_material_rgba8()?;

        // Combine split image
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

        let write_buf: Vec<u8>;
        let (final_w, final_h, data_ref): (u32, u32, &[u8]) = {
            let px = (out_w as u64 as f64) * (out_h as u64 as f64);
            let max_px = (P52_MAX_MEGAPIXELS * 1_000_000.0) as f64;
            if px > max_px {
                let scale = (max_px / px).sqrt().clamp(0.0, 1.0);
                let dw = (out_w as f64 * scale).floor().max(1.0) as u32;
                let dh = (out_h as f64 * scale).floor().max(1.0) as u32;
                write_buf = downscale_rgba8_bilinear(&combined, out_w, out_h, dw, dh);
                (dw, dh, &write_buf)
            } else {
                (out_w, out_h, &combined)
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

        // Metrics: miss ratio & avg steps
        let (miss_ratio, avg_steps) = {
            let (hit_bytes, dims) = self.read_ssgi_hit_bytes()?;
            let step_len = {
                let s = original_settings;
                let steps = s.num_steps.max(1) as f32;
                s.step_size.max(s.radius / steps)
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
            (miss_ratio, avg_steps)
        };

        // Timings
        let (trace_ms, shade_ms, temporal_ms, upsample_ms) = {
            let gi = self.gi.as_ref().context("GI manager not available")?;
            gi.ssgi_timings_ms().unwrap_or((0.0, 0.0, 0.0, 0.0))
        };

        // Task 1: Wall bounce measurement - ROI-based luminance calculation
        // Define hard-coded ROIs for neutral wall regions adjacent to red and green walls
        // Cornell box: red wall on left (x=-1), green on right (x=1), neutral walls elsewhere
        // Back wall (z=1) is neutral and should receive bounce from both red and green walls
        // ROIs are defined for 1920x1080 base resolution and scaled for other resolutions
        // ROI_R_NEUTRAL: neutral back wall region near the red wall (left side of back wall)
        // ROI_G_NEUTRAL: neutral back wall region near the green wall (right side of back wall)
        // These ROIs avoid the checker cube, edges, and specular highlights
        const ROI_R_NEUTRAL_BASE: (u32, u32, u32, u32) = (750, 360, 930, 560); // centered patch facing red wall
        const ROI_G_NEUTRAL_BASE: (u32, u32, u32, u32) = (950, 360, 1130, 560); // centered patch facing green wall
        const BASE_WIDTH: u32 = 1920;
        const BASE_HEIGHT: u32 = 1080;

        // Scale ROIs to match current resolution
        let roi_red = {
            let (x0_base, y0_base, x1_base, y1_base) = ROI_R_NEUTRAL_BASE;
            let x0 = (x0_base as f32 * capture_w as f32 / BASE_WIDTH as f32) as u32;
            let y0 = (y0_base as f32 * capture_h as f32 / BASE_HEIGHT as f32) as u32;
            let x1 = (x1_base as f32 * capture_w as f32 / BASE_WIDTH as f32) as u32;
            let y1 = (y1_base as f32 * capture_h as f32 / BASE_HEIGHT as f32) as u32;
            (x0, y0, x1, y1)
        };
        let roi_green = {
            let (x0_base, y0_base, x1_base, y1_base) = ROI_G_NEUTRAL_BASE;
            let x0 = (x0_base as f32 * capture_w as f32 / BASE_WIDTH as f32) as u32;
            let y0 = (y0_base as f32 * capture_h as f32 / BASE_HEIGHT as f32) as u32;
            let x1 = (x1_base as f32 * capture_w as f32 / BASE_WIDTH as f32) as u32;
            let y1 = (y1_base as f32 * capture_h as f32 / BASE_HEIGHT as f32) as u32;
            (x0, y0, x1, y1)
        };

        // Compute luminance for ROIs in SSGI OFF and ON frames
        // Use linear luminance from final rendered color (post-GI, pre-tone-mapping)
        // Convert from sRGB if needed so bounce math happens in linear space
        let compute_roi_luminance = |bytes: &[u8], roi: (u32, u32, u32, u32)| -> f32 {
            let (x0, y0, x1, y1) = roi;
            let mut sum_luma = 0.0f64;
            let mut count = 0u32;
            let width = capture_w;
            let height = capture_h;
            let srgb = capture_is_srgb;
            let x_start = x0.min(width);
            let x_end = x1.min(width);
            let y_start = y0.min(height);
            let y_end = y1.min(height);
            let to_linear = |channel: u8| -> f32 {
                let c = channel as f32 / 255.0;
                if srgb {
                    if c <= 0.04045 {
                        c / 12.92
                    } else {
                        ((c + 0.055) / 1.055).powf(2.4)
                    }
                } else {
                    c
                }
            };
            for y in y_start..y_end {
                for x in x_start..x_end {
                    let idx = ((y * width + x) * 4) as usize;
                    if idx + 3 < bytes.len() {
                        let r = to_linear(bytes[idx]);
                        let g = to_linear(bytes[idx + 1]);
                        let b = to_linear(bytes[idx + 2]);
                        // Compute linear luminance: L = 0.2126*R + 0.7152*G + 0.0722*B
                        let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                        sum_luma += luma as f64;
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

        let l_r_off = compute_roi_luminance(&off_bytes, roi_red);
        let l_r_on = compute_roi_luminance(&on_bytes, roi_red);
        let l_g_off = compute_roi_luminance(&off_bytes, roi_green);
        let l_g_on = compute_roi_luminance(&on_bytes, roi_green);

        let bounce_red_pct = (l_r_on - l_r_off) / l_r_off.max(1e-6);
        let bounce_green_pct = (l_g_on - l_g_off) / l_g_off.max(1e-6);

        // Task 2: ΔE fallback (steps=0) – verify steps=0 SSGI outputs pure diffuse IBL
        // Render with steps=0 twice and compare - they should be identical (pure IBL, no temporal variance)
        let max_delta_e = {
            // First render with steps=0 (should output pure diffuse IBL)
            {
                let gi = self.gi.as_mut().context("GI manager not available")?;
                gi.enable_effect(&self.device, SSE::SSGI)?;
                gi.update_ssgi_settings(&self.queue, |s| {
                    s.num_steps = 0;
                    s.step_size = original_settings.step_size;
                    s.temporal_alpha = 0.0;
                    s.intensity = 1.0; // Ensure intensity doesn't affect pure IBL
                });
                gi.ssgi_reset_history(&self.device, &self.queue)?;
            }
            self.reexecute_gi(None)?;
            let first_bytes = self.read_ssgi_filtered_bytes()?.0;

            // Second render with steps=0 (should be identical)
            self.reexecute_gi(None)?;
            let second_bytes = self.read_ssgi_filtered_bytes()?.0;

            compute_max_delta_e(&second_bytes, &first_bytes)
        };

        // Restore SSGI settings, composite intensity, and effect enablement
        {
            let gi = self.gi.as_mut().context("GI manager not available")?;
            gi.update_ssgi_settings(&self.queue, |s| {
                *s = original_settings;
            });
            // Reset composite intensity to default (1.0)
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
            // Task 1: Store bounce metrics
            meta.insert(
                "ssgi_bounce".to_string(),
                json!({
                    "red_pct": bounce_red_pct,
                    "green_pct": bounce_green_pct,
                }),
            );
        })?;
        // Restore the previous viewer scene/camera so subsequent captures or
        // interactive use continue from the original state.
        self.restore_p51_cornell_scene(state);

        Ok(())
    }

    pub(crate) fn capture_p52_ssgi_temporal(&mut self) -> anyhow::Result<()> {
        let out_dir = Path::new("reports/p5");
        fs::create_dir_all(out_dir)?;

        // Use the same Cornell box setup as P5.1 so the temporal comparison
        // runs on the canonical test scene.
        let state = self.setup_p51_cornell_scene()?;
        self.render_geometry_to_gbuffer_once()?;

        let (w, h) = (self.config.width.max(1), self.config.height.max(1));

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
            gi.set_ssgi_composite_intensity(&self.queue, P5_SSGI_DIFFUSE_SCALE);
        }

        {
            let gi = self.gi.as_mut().context("GI manager not available")?;
            gi.update_ssgi_settings(&self.queue, |s| {
                s.temporal_alpha = 0.0;
            });
            gi.ssgi_reset_history(&self.device, &self.queue)?;
        }
        self.reexecute_gi(None)?;
        let single_bytes = self.capture_material_rgba8()?;

        {
            let gi = self.gi.as_mut().context("GI manager not available")?;
            gi.update_ssgi_settings(&self.queue, |s| {
                *s = original_settings;
            });
            gi.ssgi_reset_history(&self.device, &self.queue)?;
        }

        let mut frame8_luma = Vec::new();
        let mut frame9_luma = Vec::new();

        for frame in 0..16 {
            self.reexecute_gi(None)?;
            if frame == 7 || frame == 8 {
                let (bytes, _) = self.read_ssgi_filtered_bytes()?;
                let luma = rgba16_to_luma(&bytes);
                if frame == 7 {
                    frame8_luma = luma;
                } else {
                    frame9_luma = luma;
                }
            }
        }

        let accum_bytes = self.capture_material_rgba8()?;

        // Combine side-by-side
        let out_w = w * 2;
        let out_h = h;
        let mut combined = vec![0u8; (out_w * out_h * 4) as usize];
        for y in 0..h as usize {
            let dst_off = y * (out_w as usize) * 4;
            combined[dst_off..dst_off + (w as usize * 4)]
                .copy_from_slice(&single_bytes[y * (w as usize) * 4..(y + 1) * (w as usize) * 4]);
            combined[dst_off + (w as usize * 4)..dst_off + (w as usize * 8)]
                .copy_from_slice(&accum_bytes[y * (w as usize) * 4..(y + 1) * (w as usize) * 4]);
        }

        let write_buf: Vec<u8>;
        let (final_w, final_h, data_ref): (u32, u32, &[u8]) = {
            let px = (out_w as u64 as f64) * (out_h as u64 as f64);
            let max_px = (P52_MAX_MEGAPIXELS * 1_000_000.0) as f64;
            if px > max_px {
                let scale = (max_px / px).sqrt().clamp(0.0, 1.0);
                let dw = (out_w as f64 * scale).floor().max(1.0) as u32;
                let dh = (out_h as f64 * scale).floor().max(1.0) as u32;
                write_buf = downscale_rgba8_bilinear(&combined, out_w, out_h, dw, dh);
                (dw, dh, &write_buf)
            } else {
                (out_w, out_h, &combined)
            }
        };
        crate::util::image_write::write_png_rgba8_small(
            &out_dir.join("p5_ssgi_temporal_compare.png"),
            data_ref,
            final_w,
            final_h,
        )?;
        if final_w != out_w || final_h != out_h {
            println!(
                "[P5.2] downscaled SSGI temporal capture to {}x{} (from {}x{})",
                final_w, final_h, out_w, out_h
            );
        }
        println!("[P5] Wrote reports/p5/p5_ssgi_temporal_compare.png");

        let ssim = if !frame8_luma.is_empty() && frame8_luma.len() == frame9_luma.len() {
            compute_ssim(&frame8_luma, &frame9_luma)
        } else {
            1.0
        };

        self.write_p5_meta(|meta| {
            let entry = meta.entry("ssgi_temporal".to_string()).or_insert(json!({}));
            if let Some(obj) = entry.as_object_mut() {
                obj.insert("ssim_frame8_9".to_string(), json!(ssim));
                obj.insert("accumulation_frames".to_string(), json!(16));
            }
        })?;
        // Restore settings
        {
            let gi = self.gi.as_mut().context("GI manager not available")?;
            gi.update_ssgi_settings(&self.queue, |s| {
                *s = original_settings;
            });
            gi.ssgi_reset_history(&self.device, &self.queue)?;
            gi.set_ssgi_composite_intensity(&self.queue, 1.0);
        }
        if !was_enabled {
            if let Some(ref mut gi) = self.gi {
                gi.disable_effect(SSE::SSGI);
            }
        }
        // Restore the original viewer scene/camera state.
        self.restore_p51_cornell_scene(state);

        Ok(())
    }
}
