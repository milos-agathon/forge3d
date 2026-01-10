// src/viewer/p5/ssr_glossy.rs
// P5.3 SSR glossy capture methods
// Split from ssr.rs as part of the viewer refactoring

use anyhow::{bail, Context};
use half::f16;
use std::fs;
use std::path::Path;

use crate::core::screen_space_effects::{ScreenSpaceEffect as SSE, SsrStats};
use crate::p5::meta::{self as p5_meta, build_ssr_meta, SsrMetaInput};
use crate::p5::ssr::{self, SsrScenePreset};
use crate::p5::ssr_analysis;
use crate::renderer::readback::read_texture_tight;
use crate::util::image_write;

use super::super::viewer_render_helpers::render_view_to_rgba8_ex;
use super::super::Viewer;
use super::ssr_helpers::{delta_e_lab, mean_abs_diff, srgb_triplet_to_linear};

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

        let (reference_bytes, ssr_bytes) =
            self.capture_ssr_reference_and_result(capture_w, capture_h, &mut ssr_stats)?;

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

        let (stripe_contrast, stripe_contrast_reference) = self
            .analyze_ssr_stripe_contrast(&ref_path, &ssr_path, &ssr_bytes, capture_w, capture_h)?;

        let edge_streaks =
            ssr_analysis::count_edge_streaks(&reference_bytes, &ssr_bytes, capture_w, capture_h);
        let mean_diff = mean_abs_diff(&reference_bytes, &ssr_bytes);

        let (min_rgb_miss, max_delta_e_miss) =
            self.compute_ssr_miss_metrics(&ssr_bytes, &reference_bytes, capture_w, capture_h)?;

        println!(
            "[P5.3] SSR params -> enable: {}, max_steps: {}, thickness: {:.3}",
            true, self.ssr_params.ssr_max_steps, self.ssr_params.ssr_thickness
        );
        println!(
            "[P5.3] SSR metrics -> hit_rate {:.3}, avg_steps {:.2}, diff {:.4}",
            ssr_stats.hit_rate(),
            ssr_stats.avg_steps(),
            mean_diff
        );

        let ssr_meta = build_ssr_meta(SsrMetaInput {
            stats: Some(&ssr_stats),
            stripe_contrast: Some(&stripe_contrast),
            stripe_contrast_reference: stripe_contrast_reference.as_ref(),
            mean_abs_diff: mean_diff,
            edge_streaks_gt1px: edge_streaks,
            max_delta_e_miss,
            min_rgb_miss,
        });
        println!("[P5.3] SSR status -> {}", ssr_meta.status);
        p5_meta::write_p5_meta(out_dir, |meta| {
            meta.insert("ssr".to_string(), ssr_meta.value.clone());
        })?;
        Ok(())
    }

    fn capture_ssr_reference_and_result(
        &mut self,
        capture_w: u32,
        capture_h: u32,
        ssr_stats: &mut SsrStats,
    ) -> anyhow::Result<(Vec<u8>, Vec<u8>)> {
        let far = self.viz_depth_max_override.unwrap_or(self.view_config.zfar);
        let out_dir = Path::new("reports/p5");

        self.ssr_params.set_enabled(false);
        self.sync_ssr_params_to_gi();
        self.reexecute_gi(None)?;
        let reference_bytes = {
            let gi = self.gi.as_ref().context("GI manager not available")?;
            self.with_comp_pipeline(|comp_pl, comp_bgl| {
                let fog_view = if self.fog_enabled {
                    &self.fog_output_view
                } else {
                    &self.fog_zero_view
                };
                render_view_to_rgba8_ex(super::super::viewer_render_helpers::RenderViewArgs {
                    device: &self.device,
                    queue: &self.queue,
                    comp_pl,
                    comp_bgl,
                    sky_view: &self.sky_output_view,
                    depth_view: &gi.gbuffer().depth_view,
                    fog_view,
                    surface_format: self.config.format,
                    width: capture_w,
                    height: capture_h,
                    far,
                    src_view: &gi.gbuffer().material_view,
                    mode: 0,
                })
            })?
        };

        self.ssr_params.set_enabled(true);
        self.sync_ssr_params_to_gi();
        self.reexecute_gi(Some(ssr_stats))?;
        if let Some(ref mut gi_mgr) = self.gi {
            gi_mgr
                .collect_ssr_stats(&self.device, &self.queue, ssr_stats)
                .context("collect SSR stats")?;
        } else {
            bail!("GI manager not available");
        }

        // Capture debug images
        {
            let gi = self.gi.as_ref().context("GI manager not available")?;
            let capture_view =
                |slf: &Self, view: &wgpu::TextureView, label: &str| -> anyhow::Result<()> {
                    let bytes = slf.with_comp_pipeline(|comp_pl, comp_bgl| {
                        let fog_view = if slf.fog_enabled {
                            &slf.fog_output_view
                        } else {
                            &slf.fog_zero_view
                        };
                        render_view_to_rgba8_ex(super::super::viewer_render_helpers::RenderViewArgs {
                            device: &slf.device,
                            queue: &slf.queue,
                            comp_pl,
                            comp_bgl,
                            sky_view: &slf.sky_output_view,
                            depth_view: &gi.gbuffer().depth_view,
                            fog_view,
                            surface_format: slf.config.format,
                            width: capture_w,
                            height: capture_h,
                            far,
                            src_view: view,
                            mode: 0,
                        })
                    })?;
                    image_write::write_png_rgba8_small(
                        &out_dir.join(label),
                        &bytes,
                        capture_w,
                        capture_h,
                    )?;
                    Ok(())
                };
            capture_view(self, &self.lit_output_view, "p5_ssr_glossy_lit.png")?;
            if let Some(view) = gi.ssr_spec_view() {
                capture_view(self, view, "p5_ssr_glossy_spec.png")?;
            }
            if let Some(view) = gi.ssr_final_view() {
                capture_view(self, view, "p5_ssr_glossy_final.png")?;
            }
        }

        let ssr_bytes = {
            let gi = self.gi.as_ref().context("GI manager not available")?;
            let ssr_view = gi
                .material_with_ssr_view()
                .unwrap_or(&gi.gbuffer().material_view);
            self.with_comp_pipeline(|comp_pl, comp_bgl| {
                let fog_view = if self.fog_enabled {
                    &self.fog_output_view
                } else {
                    &self.fog_zero_view
                };
                render_view_to_rgba8_ex(super::super::viewer_render_helpers::RenderViewArgs {
                    device: &self.device,
                    queue: &self.queue,
                    comp_pl,
                    comp_bgl,
                    sky_view: &self.sky_output_view,
                    depth_view: &gi.gbuffer().depth_view,
                    fog_view,
                    surface_format: self.config.format,
                    width: capture_w,
                    height: capture_h,
                    far,
                    src_view: ssr_view,
                    mode: 0,
                })
            })?
        };
        Ok((reference_bytes, ssr_bytes))
    }

    fn analyze_ssr_stripe_contrast(
        &self,
        ref_path: &Path,
        ssr_path: &Path,
        ssr_bytes: &[u8],
        capture_w: u32,
        capture_h: u32,
    ) -> anyhow::Result<([f32; 9], Option<[f32; 9]>)> {
        let mut stripe_contrast = [0.0f32; 9];
        let mut stripe_contrast_reference: Option<[f32; 9]> = None;
        match ssr_analysis::analyze_stripe_contrast(ref_path, ssr_path) {
            Ok(summary) => {
                stripe_contrast = summary.ssr;
                stripe_contrast_reference = Some(summary.reference);
            }
            Err(err) => {
                eprintln!(
                    "[P5.3] analyze_stripe_contrast failed ({}); falling back",
                    err
                );
                let preset = match self.ssr_scene_preset.clone() {
                    Some(p) => p,
                    None => SsrScenePreset::load_or_default("assets/p5/p5_ssr_scene.json")?,
                };
                let bands = crate::p5::ssr_analysis::analyze_single_image_contrast(
                    &preset, ssr_bytes, capture_w, capture_h,
                );
                for (i, v) in bands.into_iter().take(9).enumerate() {
                    stripe_contrast[i] = v;
                }
            }
        }
        Ok((stripe_contrast, stripe_contrast_reference))
    }

    fn compute_ssr_miss_metrics(
        &self,
        ssr_bytes: &[u8],
        reference_bytes: &[u8],
        capture_w: u32,
        capture_h: u32,
    ) -> anyhow::Result<(f32, f32)> {
        let mut min_rgb_miss = f32::INFINITY;
        let mut max_delta_e_miss = 0.0f32;
        if let Some(ref gi) = self.gi {
            if let (Some(hit_tex), Some(ssr_tex)) = (gi.ssr_hit_texture(), gi.ssr_output_texture())
            {
                let hit_bytes = read_texture_tight(
                    &self.device,
                    &self.queue,
                    hit_tex,
                    (capture_w, capture_h),
                    wgpu::TextureFormat::Rgba16Float,
                )
                .context("read SSR hit texture")?;
                let ssr_lin_bytes = read_texture_tight(
                    &self.device,
                    &self.queue,
                    ssr_tex,
                    (capture_w, capture_h),
                    wgpu::TextureFormat::Rgba16Float,
                )
                .context("read SSR filtered texture")?;
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
                        if local_min < min_rgb_miss {
                            min_rgb_miss = local_min;
                        }
                        let idx8 = i * 4;
                        if idx8 + 3 < ssr_bytes.len() && idx8 + 3 < reference_bytes.len() {
                            let ssr_rgb = srgb_triplet_to_linear(&ssr_bytes[idx8..idx8 + 3]);
                            let ref_rgb = srgb_triplet_to_linear(&reference_bytes[idx8..idx8 + 3]);
                            let de = delta_e_lab(ssr_rgb, ref_rgb);
                            if de > max_delta_e_miss {
                                max_delta_e_miss = de;
                            }
                        }
                    }
                }
            }
        }
        if !min_rgb_miss.is_finite() {
            min_rgb_miss = 0.0;
        }
        Ok((min_rgb_miss, max_delta_e_miss))
    }
}
