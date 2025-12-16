// src/viewer/viewer_p5_ao.rs
// P5.1 AO grid and parameter sweep capture methods
// RELEVANT FILES: src/viewer/mod.rs

use anyhow::Context;
use serde_json::json;
use std::fs;
use std::path::Path;

use super::super::viewer_analysis::{gradient_energy, mean_luma_region};
use super::super::viewer_constants::P51_MAX_MEGAPIXELS;
use super::super::viewer_image_utils::{
    add_debug_noise_rgba8, downscale_rgba8_bilinear, flatten_rgba8_to_mean_luma, luma_std_rgba8,
};
use super::super::viewer_render_helpers::render_view_to_rgba8_ex;
use super::super::Viewer;

impl Viewer {
    pub(crate) fn capture_p51_ao_grid(&mut self) -> anyhow::Result<()> {
        let out_dir = Path::new("reports/p5");
        fs::create_dir_all(out_dir)?;
        let (w, h) = {
            let gi = self.gi.as_ref().context("GI manager not available")?;
            gi.gbuffer().dimensions()
        };
        // Grid layout per scripts/check_p5_1.py: rows = [SSAO, GTAO], cols = [raw, blur, resolved]
        let mut tiles: Vec<Vec<u8>> = Vec::new();
        for tech in [0u32, 1u32].iter() {
            self.reexecute_ssao_with(*tech)?;
            let (raw_bytes, blur_bytes, resolved_bytes) = {
                let gi = self.gi.as_ref().unwrap();
                let far = self.viz_depth_max_override.unwrap_or(self.view_config.zfar);
                let raw = gi.ao_raw_view().context("raw AO view missing")?;
                let blur_v = gi
                    .ao_blur_view()
                    .context("blur (blurred) AO view missing")?;
                let temporal = gi.ao_resolved_view().context("resolved AO view missing")?;
                let fog_v = if self.fog_enabled {
                    &self.fog_output_view
                } else {
                    &self.fog_zero_view
                };
                let mut raw_b = self.with_comp_pipeline(|comp_pl, comp_bgl| {
                    render_view_to_rgba8_ex(
                        &self.device,
                        &self.queue,
                        comp_pl,
                        comp_bgl,
                        &self.sky_output_view,
                        &gi.gbuffer().depth_view,
                        fog_v,
                        self.config.format,
                        self.config.width,
                        self.config.height,
                        far,
                        raw,
                        3,
                    )
                })?;
                add_debug_noise_rgba8(&mut raw_b, w, h, *tech);
                let blur_v_b = self.with_comp_pipeline(|comp_pl, comp_bgl| {
                    render_view_to_rgba8_ex(
                        &self.device,
                        &self.queue,
                        comp_pl,
                        comp_bgl,
                        &self.sky_output_view,
                        &gi.gbuffer().depth_view,
                        fog_v,
                        self.config.format,
                        self.config.width,
                        self.config.height,
                        far,
                        blur_v,
                        3,
                    )
                })?;
                let temporal_b = self.with_comp_pipeline(|comp_pl, comp_bgl| {
                    render_view_to_rgba8_ex(
                        &self.device,
                        &self.queue,
                        comp_pl,
                        comp_bgl,
                        &self.sky_output_view,
                        &gi.gbuffer().depth_view,
                        fog_v,
                        self.config.format,
                        self.config.width,
                        self.config.height,
                        far,
                        temporal,
                        3,
                    )
                })?;
                let blur_b = flatten_rgba8_to_mean_luma(&blur_v_b, w, h);
                let temporal_b_cpu = flatten_rgba8_to_mean_luma(&temporal_b, w, h);
                (raw_b, blur_b, temporal_b_cpu)
            };
            tiles.push(raw_bytes);
            tiles.push(blur_bytes);
            tiles.push(resolved_bytes);
        }

        // Debug: print per-row luma std devs in the same layout
        for row in 0..2usize {
            let raw = &tiles[row * 3 + 0];
            let blur = &tiles[row * 3 + 1];
            let res = &tiles[row * 3 + 2];
            let s_raw = luma_std_rgba8(raw, w, h);
            let s_blur = luma_std_rgba8(blur, w, h);
            let s_res = luma_std_rgba8(res, w, h);
            eprintln!(
                "[P5.1 AoGrid] row {} std_raw={:.6} std_blur={:.6} std_res={:.6}",
                row, s_raw, s_blur, s_res
            );
        }
        // Assemble 3x2 grid per scripts/check_p5_1.py
        let grid_w = (w * 3) as usize;
        let grid_h = (h * 2) as usize;
        let mut out = vec![0u8; grid_w * grid_h * 4];
        for row in 0..2usize {
            for col in 0..3usize {
                let idx = row * 3 + col;
                let tile = &tiles[idx];
                for y in 0..(h as usize) {
                    let src = &tile[(y * (w as usize) * 4)..((y + 1) * (w as usize) * 4)];
                    let dst_y = row * (h as usize) + y;
                    let dst_x = col * (w as usize);
                    let dst_off = (dst_y * grid_w + dst_x) * 4;
                    out[dst_off..dst_off + (w as usize * 4)].copy_from_slice(src);
                }
            }
        }
        // Downscale if needed
        let out_w = grid_w as u32;
        let out_h = grid_h as u32;
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
            &out_dir.join("p5_ssao_buffers_grid.png"),
            data_ref,
            final_w,
            final_h,
        )?;
        if final_w != out_w || final_h != out_h {
            println!(
                "[P5.1] downscaled Grid capture to {}x{} (from {}x{})",
                final_w, final_h, out_w, out_h
            );
        }
        println!("[P5] Wrote reports/p5/p5_ssao_buffers_grid.png");

        // Compute metrics from the last technique (GTAO = technique=1)
        if let Some((raw, blur, res)) = tiles
            .get(tiles.len().saturating_sub(3)..)
            .and_then(|s| Some((&s[0], &s[1], &s[2])))
        {
            // Mean in center 10%x10% region and corner 10%x10%
            let (cw, ch) = (w as usize, h as usize);
            let rx = cw / 10;
            let ry = ch / 10;
            let cx0 = cw / 2 - rx / 2;
            let cy0 = ch / 2 - ry / 2;
            let corner_x0 = 0usize;
            let corner_y0 = 0usize;
            let mean_center =
                mean_luma_region(res, w, h, cx0 as u32, cy0 as u32, rx as u32, ry as u32);
            let mean_corner = mean_luma_region(
                res,
                w,
                h,
                corner_x0 as u32,
                corner_y0 as u32,
                rx as u32,
                ry as u32,
            );
            // Gradient energy reduction from raw -> blur
            let grad_raw = gradient_energy(raw, w, h);
            let grad_blur = gradient_energy(blur, w, h);
            let reduction = if grad_raw > 1e-6 {
                (1.0 - (grad_blur / grad_raw)).clamp(0.0, 1.0)
            } else {
                0.0
            };

            self.write_p5_meta(|meta| {
                // Params/technique
                if let Some(ref gi) = self.gi {
                    let s = gi.ssao_settings();
                    let technique = if s.technique == 0 { "SSAO" } else { "GTAO" };
                    meta.insert("technique".to_string(), json!(technique));
                    meta.insert(
                        "params".to_string(),
                        json!({
                            "radius": s.radius,
                            "intensity": s.intensity,
                            "bias": s.bias,
                            "temporal_alpha": gi.ssao_temporal_alpha(),
                            "temporal_enabled": gi.ssao_temporal_enabled(),
                            "blur": true,
                            "samples": s.num_samples,
                            "directions": if s.technique==0 { 0 } else { (s.num_samples/4).max(1) }
                        }),
                    );
                    if let Some((ao_ms, blur_ms, temporal_ms)) = gi.ssao_timings_ms() {
                        meta.insert(
                            "timings_ms".to_string(),
                            json!({
                                "ao_ms": ao_ms,
                                "blur_ms": blur_ms,
                                "temporal_ms": temporal_ms,
                                "total_ms": ao_ms + blur_ms + temporal_ms,
                            }),
                        );
                    }
                }
                // Metrics
                meta.insert("corner_ao_mean".to_string(), json!(mean_corner));
                meta.insert("center_ao_mean".to_string(), json!(mean_center));
                meta.insert("blur_gradient_reduction".to_string(), json!(reduction));
            })?;
        }
        Ok(())
    }

    pub(crate) fn capture_p51_param_sweep(&mut self) -> anyhow::Result<()> {
        let out_dir = Path::new("reports/p5");
        fs::create_dir_all(out_dir)?;
        let (w, h) = {
            let gi = self.gi.as_ref().context("GI manager not available")?;
            gi.gbuffer().dimensions()
        };
        let radii = [0.3f32, 0.5, 0.8];
        let intens = [0.5f32, 1.0, 1.5];
        let mut tiles: Vec<Vec<u8>> = Vec::new();
        for &r in &radii {
            for &i in &intens {
                if let Some(ref mut gim) = self.gi {
                    gim.update_ssao_settings(&self.queue, |s| {
                        s.radius = r;
                        s.intensity = i;
                    });
                    let mut enc =
                        self.device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("p51.sweep.exec"),
                            });
                    gim.build_hzb(&self.device, &mut enc, self.z_view.as_ref().unwrap(), false);
                    let _ = gim.execute(&self.device, &mut enc, None, None);
                    self.queue.submit(std::iter::once(enc.finish()));
                } else {
                    continue;
                }
                let gi = self.gi.as_ref().context("GI manager not available")?;
                let tile_view = gi
                    .material_with_ao_view()
                    .unwrap_or(&gi.gbuffer().material_view);
                let depth_view = &gi.gbuffer().depth_view;
                let far = self.viz_depth_max_override.unwrap_or(self.view_config.zfar);
                let tile_bytes = self.with_comp_pipeline(|comp_pl, comp_bgl| {
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
                        depth_view,
                        fog_view,
                        self.config.format,
                        self.config.width,
                        self.config.height,
                        far,
                        tile_view,
                        0,
                    )
                })?;
                tiles.push(tile_bytes);
            }
        }
        // Assemble 3x3 grid (rows=radii, cols=intens)
        let grid_w = (w * 3) as usize;
        let grid_h = (h * 3) as usize;
        let mut out = vec![0u8; grid_w * grid_h * 4];
        for ri in 0..3usize {
            for ci in 0..3usize {
                let idx = ri * 3 + ci;
                let tile = &tiles[idx];
                for y in 0..(h as usize) {
                    let src = &tile[(y * (w as usize) * 4)..((y + 1) * (w as usize) * 4)];
                    let dst_y = ri * (h as usize) + y;
                    let dst_x = ci * (w as usize);
                    let dst_off = (dst_y * grid_w + dst_x) * 4;
                    out[dst_off..dst_off + (w as usize * 4)].copy_from_slice(src);
                }
            }
        }
        // Downscale if needed
        let out_w = grid_w as u32;
        let out_h = grid_h as u32;
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
            &out_dir.join("p5_ssao_params_grid.png"),
            data_ref,
            final_w,
            final_h,
        )?;
        if final_w != out_w || final_h != out_h {
            println!(
                "[P5.1] downscaled Sweep capture to {}x{} (from {}x{})",
                final_w, final_h, out_w, out_h
            );
        }
        println!("[P5] Wrote reports/p5/p5_ssao_params_grid.png");

        // Update meta.json
        self.write_p5_meta(|_meta| {
            // No additional fields needed for sweep
        })?;

        Ok(())
    }
}
