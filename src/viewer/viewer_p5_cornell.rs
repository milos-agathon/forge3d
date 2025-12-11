// src/viewer/viewer_p5_cornell.rs
// P5.1 Cornell box SSAO capture methods
// RELEVANT FILES: src/viewer/mod.rs, src/viewer/viewer_p5.rs

use anyhow::Context;
use serde_json::json;
use std::fs;
use std::path::Path;

use super::viewer_constants::P51_MAX_MEGAPIXELS;
use super::viewer_image_utils::downscale_rgba8_bilinear;
use super::viewer_render_helpers::render_view_to_rgba8_ex;
use super::Viewer;

impl Viewer {
    pub(crate) fn capture_p51_cornell_with_scene(&mut self) -> anyhow::Result<()> {
        let state = self.setup_p51_cornell_scene()?;
        // Render Cornell geometry into the GI GBuffer and depth, then let the
        // SSAO path rebuild its AO AOVs from this scene before capturing.
        self.render_geometry_to_gbuffer_once()?;
        let result = self.capture_p51_cornell_split();
        self.restore_p51_cornell_scene(state);
        result
    }

    pub(crate) fn capture_p51_cornell_split(&mut self) -> anyhow::Result<()> {
        let out_dir = Path::new("reports/p5");
        fs::create_dir_all(out_dir)?;
        // Ensure SSAO has up-to-date buffers for the current view before capturing ON/OFF.
        if let Some(technique) = self.gi.as_ref().map(|gi| gi.ssao_settings().technique) {
            self.reexecute_ssao_with(technique)?;
        }
        let (off_bytes, on_bytes, w, h) = {
            let gi = self.gi.as_ref().context("GI manager not available")?;
            let (w, h) = gi.gbuffer().dimensions();
            let far = self.viz_depth_max_override.unwrap_or(self.view_config.zfar);
            let off_bytes = self.with_comp_pipeline(|comp_pl, comp_bgl| {
                let fog_view = if self.fog_enabled {
                    &self.fog_output_view
                } else {
                    &self.fog_zero_view
                };
                render_view_to_rgba8_ex(
                    &self.device,
                    &self.queue,
                    comp_pl,
                    comp_bgl,
                    &self.sky_output_view,
                    &gi.gbuffer().depth_view,
                    fog_view,
                    self.config.format,
                    self.config.width,
                    self.config.height,
                    far,
                    &gi.gbuffer().material_view,
                    0,
                )
            })?;
            let on_view = gi
                .material_with_ao_view()
                .unwrap_or(&gi.gbuffer().material_view);
            let on_bytes = self.with_comp_pipeline(|comp_pl, comp_bgl| {
                let fog_view = if self.fog_enabled {
                    &self.fog_output_view
                } else {
                    &self.fog_zero_view
                };
                render_view_to_rgba8_ex(
                    &self.device,
                    &self.queue,
                    comp_pl,
                    comp_bgl,
                    &self.sky_output_view,
                    &gi.gbuffer().depth_view,
                    fog_view,
                    self.config.format,
                    self.config.width,
                    self.config.height,
                    far,
                    on_view,
                    0,
                )
            })?;
            (off_bytes, on_bytes, w, h)
        };
        // Assemble side-by-side
        let mut out = vec![0u8; (w * 2 * h * 4) as usize];
        for y in 0..h as usize {
            let row_off = &off_bytes[(y * (w as usize) * 4)..((y + 1) * (w as usize) * 4)];
            let row_on = &on_bytes[(y * (w as usize) * 4)..((y + 1) * (w as usize) * 4)];
            let dst = &mut out[(y * (2 * w as usize) * 4)..((y + 1) * (2 * w as usize) * 4)];
            dst[..(w as usize * 4)].copy_from_slice(row_off);
            dst[(w as usize * 4)..].copy_from_slice(row_on);
        }
        // Downscale if the composite exceeds the pixel budget
        let out_w = w * 2;
        let out_h = h;
        let max_px = (P51_MAX_MEGAPIXELS * 1_000_000.0) as f64;
        let px = (out_w as u64 as f64) * (out_h as u64 as f64);
        let write_buf: Vec<u8>;
        let (final_w, final_h, data_ref): (u32, u32, &[u8]) = if px > max_px {
            let scale = (max_px / px).sqrt().clamp(0.0, 1.0);
            let dw = (out_w as f64 * scale).floor().max(1.0) as u32;
            let dh = (out_h as f64 * scale).floor().max(1.0) as u32;
            write_buf = downscale_rgba8_bilinear(&out, out_w, out_h, dw, dh);
            (dw, dh, &write_buf)
        } else {
            (out_w, out_h, &out)
        };
        crate::util::image_write::write_png_rgba8_small(
            &out_dir.join("p5_ssao_cornell.png"),
            data_ref,
            final_w,
            final_h,
        )?;
        if final_w != out_w || final_h != out_h {
            println!(
                "[P5.1] downscaled Cornell capture to {}x{} (from {}x{})",
                final_w, final_h, out_w, out_h
            );
        }
        println!("[P5] Wrote reports/p5/p5_ssao_cornell.png");

        // Derive specular preservation metric from OFF vs ON split
        // Use top-1% brightest pixels by luma in OFF image then compare delta with ON
        let mut off_lumas: Vec<f32> = Vec::with_capacity((w * h) as usize);
        let mut on_lumas: Vec<f32> = Vec::with_capacity((w * h) as usize);
        for y in 0..h as usize {
            for x in 0..w as usize {
                let i = (y * w as usize + x) * 4;
                let lo = 0.2126 * (off_bytes[i] as f32)
                    + 0.7152 * (off_bytes[i + 1] as f32)
                    + 0.0722 * (off_bytes[i + 2] as f32);
                let ln = 0.2126 * (on_bytes[i] as f32)
                    + 0.7152 * (on_bytes[i + 1] as f32)
                    + 0.0722 * (on_bytes[i + 2] as f32);
                off_lumas.push(lo / 255.0);
                on_lumas.push(ln / 255.0);
            }
        }
        let mut idxs: Vec<usize> = (0..off_lumas.len()).collect();
        idxs.sort_by(|&a, &b| {
            off_lumas[b]
                .partial_cmp(&off_lumas[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let top_n = (off_lumas.len() as f32 * 0.01).ceil() as usize;
        let top_n = top_n.max(1).min(off_lumas.len());
        let mut sum_off = 0.0;
        let mut sum_on = 0.0;
        for k in 0..top_n {
            let i = idxs[k];
            sum_off += off_lumas[i];
            sum_on += on_lumas[i];
        }
        let mean_off = sum_off / top_n as f32;
        let mean_on = sum_on / top_n as f32;
        let spec_delta = (mean_on - mean_off).abs();
        let specular_preservation = if spec_delta <= 0.01 { "PASS" } else { "FAIL" };
        self.write_p5_meta(|meta| {
            meta.insert(
                "specular_preservation".to_string(),
                json!(format!(
                    "{} (delta={:.4})",
                    specular_preservation, spec_delta
                )),
            );
        })?;
        Ok(())
    }

    pub(crate) fn reexecute_ssao_with(&mut self, technique: u32) -> anyhow::Result<()> {
        if let Some(ref mut gi) = self.gi {
            // Ensure SSAO is enabled so AO AOVs are allocated
            if !gi.is_enabled(crate::core::screen_space_effects::ScreenSpaceEffect::SSAO) {
                let _ = gi.enable_effect(
                    &self.device,
                    crate::core::screen_space_effects::ScreenSpaceEffect::SSAO,
                );
            }
            gi.update_ssao_settings(&self.queue, |s| {
                s.technique = technique;
            });
            // Rebuild HZB and execute
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("p51.ssao.reexec"),
                });
            gi.build_hzb(&self.device, &mut enc, self.z_view.as_ref().unwrap(), false);
            let _ = gi.execute(&self.device, &mut enc, None, None);
            self.queue.submit(std::iter::once(enc.finish()));
        }
        Ok(())
    }
}
