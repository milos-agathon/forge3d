// src/viewer/p5/ssgi_temporal.rs
// P5.2 SSGI temporal comparison capture methods
// Split from core.rs as part of the viewer refactoring

use anyhow::Context;
use serde_json::json;
use std::fs;
use std::path::Path;

use crate::core::screen_space_effects::ScreenSpaceEffect as SSE;

use super::super::image_analysis::{compute_ssim, rgba16_to_luma};
use super::super::viewer_constants::{P52_MAX_MEGAPIXELS, P5_SSGI_DIFFUSE_SCALE};
use super::super::viewer_image_utils::downscale_rgba8_bilinear;
use super::super::Viewer;

impl Viewer {
    pub(crate) fn capture_p52_ssgi_temporal(&mut self) -> anyhow::Result<()> {
        let out_dir = Path::new("reports/p5");
        fs::create_dir_all(out_dir)?;

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

        self.restore_p51_cornell_scene(state);
        Ok(())
    }
}
