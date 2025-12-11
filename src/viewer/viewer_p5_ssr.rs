// src/viewer/viewer_p5_ssr.rs
// P5.3 SSR glossy capture methods
// RELEVANT FILES: src/viewer/mod.rs

use anyhow::{bail, Context};
use half::f16;
use serde_json::json;
use std::fs;
use std::path::Path;

use crate::core::screen_space_effects::{ScreenSpaceEffect as SSE, SsrStats};
use crate::p5::meta::{self as p5_meta, build_ssr_meta, SsrMetaInput};
use crate::p5::ssr::{self, SsrScenePreset};
use crate::p5::ssr_analysis;
use crate::renderer::readback::read_texture_tight;
use crate::util::image_write;

use super::viewer_render_helpers::render_view_to_rgba8_ex;
use super::Viewer;

fn mean_abs_diff(a: &[u8], b: &[u8]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let sum: u64 = a.iter().zip(b.iter()).map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs() as u64).sum();
    sum as f32 / a.len() as f32
}

fn srgb_triplet_to_linear(rgb: &[u8]) -> [f32; 3] {
    let to_lin = |c: u8| {
        let v = c as f32 / 255.0;
        if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
    };
    [to_lin(rgb[0]), to_lin(rgb[1]), to_lin(rgb[2])]
}

fn delta_e_lab(a: [f32; 3], b: [f32; 3]) -> f32 {
    fn to_xyz(rgb: [f32; 3]) -> [f32; 3] {
        let r = rgb[0]; let g = rgb[1]; let b = rgb[2];
        [0.4124*r + 0.3576*g + 0.1805*b, 0.2126*r + 0.7152*g + 0.0722*b, 0.0193*r + 0.1192*g + 0.9505*b]
    }
    fn to_lab(xyz: [f32; 3]) -> [f32; 3] {
        let f = |t: f32| if t > 0.008856 { t.powf(1.0/3.0) } else { 7.787*t + 16.0/116.0 };
        let (x, y, z) = (xyz[0]/0.95047, xyz[1], xyz[2]/1.08883);
        [116.0*f(y) - 16.0, 500.0*(f(x) - f(y)), 200.0*(f(y) - f(z))]
    }
    let la = to_lab(to_xyz(a));
    let lb = to_lab(to_xyz(b));
    ((la[0]-lb[0]).powi(2) + (la[1]-lb[1]).powi(2) + (la[2]-lb[2]).powi(2)).sqrt()
}

impl Viewer {
    pub(crate) fn capture_p53_ssr_glossy(&mut self) -> anyhow::Result<()> {
        const SSR_REF_NAME: &str = "p5_ssr_glossy_reference.png";
        let out_dir = Path::new("reports/p5");
        fs::create_dir_all(out_dir)?;

        if !self.ssr_scene_loaded {
            self.apply_ssr_scene_preset()?;
        }

        let mut ssr_stats = SsrStats::new();

        {
            if let Some(ref mut gi_mgr) = self.gi {
                if !gi_mgr.is_enabled(SSE::SSR) {
                    gi_mgr.enable_effect(&self.device, SSE::SSR)?;
                }
            } else {
                bail!("GI manager not available");
            }
            self.sync_ssr_params_to_gi();
        }

        let capture_w = self.config.width.max(1);
        let capture_h = self.config.height.max(1);
        let original_ssr_enable = self.ssr_params.ssr_enable;
        let (reference_bytes, ssr_bytes) = {
            let far = self.viz_depth_max_override.unwrap_or(self.view_config.zfar);

            self.ssr_params.set_enabled(false);
            self.sync_ssr_params_to_gi();
            self.reexecute_gi(None)?;
            let reference_bytes = {
                let gi = self.gi.as_ref().context("GI manager not available")?;
                self.with_comp_pipeline(|comp_pl, comp_bgl| {
                    let fog_view = if self.fog_enabled { &self.fog_output_view } else { &self.fog_zero_view };
                    render_view_to_rgba8_ex(&self.device, &self.queue, comp_pl, comp_bgl,
                        &self.sky_output_view, &gi.gbuffer().depth_view, fog_view,
                        self.config.format, capture_w, capture_h, far, &gi.gbuffer().material_view, 0)
                })?
            };

            self.ssr_params.set_enabled(true);
            self.sync_ssr_params_to_gi();
            self.reexecute_gi(Some(&mut ssr_stats))?;
            if let Some(ref mut gi_mgr) = self.gi {
                gi_mgr.collect_ssr_stats(&self.device, &self.queue, &mut ssr_stats).context("collect SSR stats")?;
            } else {
                bail!("GI manager not available");
            }

            // Capture lit buffer for debugging stripe intensity
            {
                let gi = self.gi.as_ref().context("GI manager not available")?;
                let capture_view = |slf: &Self, view: &wgpu::TextureView, label: &str| -> anyhow::Result<()> {
                    let bytes = slf.with_comp_pipeline(|comp_pl, comp_bgl| {
                        let fog_view = if slf.fog_enabled { &slf.fog_output_view } else { &slf.fog_zero_view };
                        render_view_to_rgba8_ex(&slf.device, &slf.queue, comp_pl, comp_bgl,
                            &slf.sky_output_view, &gi.gbuffer().depth_view, fog_view,
                            slf.config.format, capture_w, capture_h, far, view, 0)
                    })?;
                    image_write::write_png_rgba8_small(&out_dir.join(label), &bytes, capture_w, capture_h)?;
                    Ok(())
                };
                capture_view(self, &self.lit_output_view, "p5_ssr_glossy_lit.png")?;
                if let Some(view) = gi.ssr_spec_view() { capture_view(self, view, "p5_ssr_glossy_spec.png")?; }
                if let Some(view) = gi.ssr_final_view() { capture_view(self, view, "p5_ssr_glossy_final.png")?; }
            }

            let ssr_bytes = {
                let gi = self.gi.as_ref().context("GI manager not available")?;
                let ssr_view = gi.material_with_ssr_view().unwrap_or(&gi.gbuffer().material_view);
                self.with_comp_pipeline(|comp_pl, comp_bgl| {
                    let fog_view = if self.fog_enabled { &self.fog_output_view } else { &self.fog_zero_view };
                    render_view_to_rgba8_ex(&self.device, &self.queue, comp_pl, comp_bgl,
                        &self.sky_output_view, &gi.gbuffer().depth_view, fog_view,
                        self.config.format, capture_w, capture_h, far, ssr_view, 0)
                })?
            };
            (reference_bytes, ssr_bytes)
        };

        if original_ssr_enable != self.ssr_params.ssr_enable {
            self.ssr_params.set_enabled(original_ssr_enable);
            self.sync_ssr_params_to_gi();
            self.reexecute_gi(None)?;
        }

        let ssr_path = out_dir.join(ssr::DEFAULT_OUTPUT_NAME);
        image_write::write_png_rgba8_small(&ssr_path, &ssr_bytes, capture_w, capture_h)?;
        println!("[P5] Wrote {}", ssr_path.display());

        let ref_path = out_dir.join(SSR_REF_NAME);
        image_write::write_png_rgba8_small(&ref_path, &reference_bytes, capture_w, capture_h)?;
        println!("[P5] Wrote {}", ref_path.display());

        let mut stripe_contrast = [0.0f32; 9];
        let mut stripe_contrast_reference: Option<[f32; 9]> = None;
        match ssr_analysis::analyze_stripe_contrast(&ref_path, &ssr_path) {
            Ok(summary) => {
                stripe_contrast = summary.ssr;
                stripe_contrast_reference = Some(summary.reference);
            }
            Err(err) => {
                eprintln!("[P5.3] analyze_stripe_contrast failed ({}); falling back", err);
                let preset = match self.ssr_scene_preset.clone() {
                    Some(p) => p,
                    None => SsrScenePreset::load_or_default("assets/p5/p5_ssr_scene.json")?,
                };
                let bands = crate::p5::ssr_analysis::analyze_single_image_contrast(&preset, &ssr_bytes, capture_w, capture_h);
                for (i, v) in bands.into_iter().take(9).enumerate() { stripe_contrast[i] = v; }
            }
        }
        let edge_streaks = ssr_analysis::count_edge_streaks(&reference_bytes, &ssr_bytes, capture_w, capture_h);
        let mean_diff = mean_abs_diff(&reference_bytes, &ssr_bytes);

        let (mut min_rgb_miss, mut max_delta_e_miss) = (f32::INFINITY, 0.0f32);
        if let Some(ref gi) = self.gi {
            if let (Some(hit_tex), Some(ssr_tex)) = (gi.ssr_hit_texture(), gi.ssr_output_texture()) {
                let hit_bytes = read_texture_tight(&self.device, &self.queue, hit_tex, (capture_w, capture_h), wgpu::TextureFormat::Rgba16Float).context("read SSR hit texture")?;
                let ssr_lin_bytes = read_texture_tight(&self.device, &self.queue, ssr_tex, (capture_w, capture_h), wgpu::TextureFormat::Rgba16Float).context("read SSR filtered texture")?;
                let pixel_count = (capture_w as usize) * (capture_h as usize);
                for i in 0..pixel_count {
                    let hb = &hit_bytes[i * 8..i * 8 + 8];
                    let hit_mask = f16::from_le_bytes([hb[6], hb[7]]).to_f32();
                    if hit_mask < 0.5 {
                        let sb = &ssr_lin_bytes[i * 8..i * 8 + 8];
                        let r = f16::from_le_bytes([sb[0], sb[1]]).to_f32();
                        let g = f16::from_le_bytes([sb[2], sb[3]]).to_f32();
                        let b = f16::from_le_bytes([sb[4], sb[5]]).to_f32();
                        let local_min = r.min(g).min(b);
                        if local_min < min_rgb_miss { min_rgb_miss = local_min; }
                        let idx8 = i * 4;
                        if idx8 + 3 < ssr_bytes.len() && idx8 + 3 < reference_bytes.len() {
                            let ssr_rgb = srgb_triplet_to_linear(&ssr_bytes[idx8..idx8 + 3]);
                            let ref_rgb = srgb_triplet_to_linear(&reference_bytes[idx8..idx8 + 3]);
                            let de = delta_e_lab(ssr_rgb, ref_rgb);
                            if de > max_delta_e_miss { max_delta_e_miss = de; }
                        }
                    }
                }
            }
        }
        if !min_rgb_miss.is_finite() { min_rgb_miss = 0.0; }

        println!("[P5.3] SSR params -> enable: {}, max_steps: {}, thickness: {:.3}", true, self.ssr_params.ssr_max_steps, self.ssr_params.ssr_thickness);
        println!("[P5.3] SSR metrics -> hit_rate {:.3}, avg_steps {:.2}, diff {:.4}", ssr_stats.hit_rate(), ssr_stats.avg_steps(), mean_diff);

        let ssr_meta = build_ssr_meta(SsrMetaInput {
            stats: Some(&ssr_stats), stripe_contrast: Some(&stripe_contrast), stripe_contrast_reference: stripe_contrast_reference.as_ref(),
            mean_abs_diff: mean_diff, edge_streaks_gt1px: edge_streaks, max_delta_e_miss, min_rgb_miss,
        });
        println!("[P5.3] SSR status -> {}", ssr_meta.status);
        p5_meta::write_p5_meta(out_dir, |meta| { meta.insert("ssr".to_string(), ssr_meta.value.clone()); })?;
        Ok(())
    }

    pub(crate) fn capture_p53_ssr_thickness_ablation(&mut self) -> anyhow::Result<()> {
        const OUTPUT_NAME: &str = "p5_ssr_thickness_ablation.png";
        let out_dir = Path::new("reports/p5");
        fs::create_dir_all(out_dir)?;

        {
            if let Some(ref mut gi_mgr) = self.gi {
                if !gi_mgr.is_enabled(SSE::SSR) {
                    gi_mgr.enable_effect(&self.device, SSE::SSR)?;
                }
            } else {
                bail!("GI manager not available");
            }
            self.sync_ssr_params_to_gi();
        }

        let _far = self.viz_depth_max_override.unwrap_or(self.view_config.zfar);
        let capture_w = self.config.width.max(1);
        let capture_h = self.config.height.max(1);
        let original_thickness = self.ssr_params.ssr_thickness;
        let original_enable = self.ssr_params.ssr_enable;

        // 1) Reference (SSR disabled)
        self.ssr_params.set_enabled(false);
        self.sync_ssr_params_to_gi();
        self.reexecute_gi(None)?;
        let reference_bytes = self.capture_gi_output_tonemapped_rgba8()?;

        // 2) SSR enabled, thin thickness variant
        self.ssr_params.set_enabled(true);
        self.ssr_params.set_thickness(0.0);
        self.sync_ssr_params_to_gi();
        self.reexecute_gi(None)?;
        let _unused_off_bytes = self.capture_gi_output_tonemapped_rgba8()?;

        // 3) SSR enabled, restored thickness
        let restored_thickness = if original_thickness <= 0.0 { 0.08 } else { original_thickness };
        let thin_thickness = (restored_thickness * 0.15).max(0.005);

        self.ssr_params.set_thickness(thin_thickness);
        self.sync_ssr_params_to_gi();
        self.reexecute_gi(None)?;
        let off_bytes = self.capture_gi_output_tonemapped_rgba8()?;

        self.ssr_params.set_thickness(restored_thickness);
        self.sync_ssr_params_to_gi();
        self.reexecute_gi(None)?;
        let on_bytes = self.capture_gi_output_tonemapped_rgba8()?;

        self.ssr_params.set_thickness(original_thickness);
        self.ssr_params.set_enabled(original_enable);
        self.sync_ssr_params_to_gi();
        self.reexecute_gi(None)?;

        let out_w = capture_w * 2;
        let out_h = capture_h;
        let mut composed = vec![0u8; (out_w * out_h * 4) as usize];
        let row_bytes = (capture_w as usize) * 4;
        for y in 0..(capture_h as usize) {
            let dst_off = y * row_bytes * 2;
            let src_off = y * row_bytes;
            composed[dst_off..dst_off + row_bytes].copy_from_slice(&off_bytes[src_off..src_off + row_bytes]);
            composed[dst_off + row_bytes..dst_off + row_bytes * 2].copy_from_slice(&on_bytes[src_off..src_off + row_bytes]);
        }

        let out_path = out_dir.join(OUTPUT_NAME);
        image_write::write_png_rgba8_small(&out_path, &composed, out_w, out_h)?;
        let streaks_off = ssr_analysis::count_edge_streaks(&reference_bytes, &off_bytes, capture_w, capture_h);
        let streaks_on = ssr_analysis::count_edge_streaks(&reference_bytes, &on_bytes, capture_w, capture_h);
        println!("[P5] Wrote {} (thickness thin {:.3} | baseline {:.3})", out_path.display(), thin_thickness, restored_thickness);
        println!("[P5.3] Edge streak counts -> off: {} | on: {}", streaks_off, streaks_on);

        let preset = match self.ssr_scene_preset.clone() {
            Some(p) => p,
            None => SsrScenePreset::load_or_default("assets/p5/p5_ssr_scene.json")?,
        };
        let (undershoot_before, undershoot_after) = ssr_analysis::compute_undershoot_metric(
            &preset, &reference_bytes, &on_bytes, &off_bytes, capture_w, capture_h,
        );
        println!("[P5.3] Thickness undershoot metrics -> before: {:.6}, after: {:.6}", undershoot_before, undershoot_after);
        p5_meta::write_p5_meta(out_dir, |meta| {
            let ssr_entry = meta.entry("ssr".to_string()).or_insert(serde_json::json!({}));
            if let Some(obj) = ssr_entry.as_object_mut() {
                p5_meta::patch_thickness_ablation(obj, undershoot_before, undershoot_after);
            }
        })?;
        Ok(())
    }
}
