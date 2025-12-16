// src/viewer/p5/ssr_thickness.rs
// P5.3 SSR thickness ablation capture methods
// Split from ssr.rs as part of the viewer refactoring

use anyhow::bail;
use std::fs;
use std::path::Path;

use crate::core::screen_space_effects::ScreenSpaceEffect as SSE;
use crate::p5::meta as p5_meta;
use crate::p5::ssr::SsrScenePreset;
use crate::p5::ssr_analysis;
use crate::util::image_write;

use super::super::Viewer;

impl Viewer {
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
        let restored_thickness = if original_thickness <= 0.0 {
            0.08
        } else {
            original_thickness
        };
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
            composed[dst_off..dst_off + row_bytes]
                .copy_from_slice(&off_bytes[src_off..src_off + row_bytes]);
            composed[dst_off + row_bytes..dst_off + row_bytes * 2]
                .copy_from_slice(&on_bytes[src_off..src_off + row_bytes]);
        }

        let out_path = out_dir.join(OUTPUT_NAME);
        image_write::write_png_rgba8_small(&out_path, &composed, out_w, out_h)?;
        let streaks_off =
            ssr_analysis::count_edge_streaks(&reference_bytes, &off_bytes, capture_w, capture_h);
        let streaks_on =
            ssr_analysis::count_edge_streaks(&reference_bytes, &on_bytes, capture_w, capture_h);
        println!(
            "[P5] Wrote {} (thickness thin {:.3} | baseline {:.3})",
            out_path.display(),
            thin_thickness,
            restored_thickness
        );
        println!(
            "[P5.3] Edge streak counts -> off: {} | on: {}",
            streaks_off, streaks_on
        );

        let preset = match self.ssr_scene_preset.clone() {
            Some(p) => p,
            None => SsrScenePreset::load_or_default("assets/p5/p5_ssr_scene.json")?,
        };
        let (undershoot_before, undershoot_after) = ssr_analysis::compute_undershoot_metric(
            &preset,
            &reference_bytes,
            &on_bytes,
            &off_bytes,
            capture_w,
            capture_h,
        );
        println!(
            "[P5.3] Thickness undershoot metrics -> before: {:.6}, after: {:.6}",
            undershoot_before, undershoot_after
        );
        p5_meta::write_p5_meta(out_dir, |meta| {
            let ssr_entry = meta
                .entry("ssr".to_string())
                .or_insert(serde_json::json!({}));
            if let Some(obj) = ssr_entry.as_object_mut() {
                p5_meta::patch_thickness_ablation(obj, undershoot_before, undershoot_after);
            }
        })?;
        Ok(())
    }
}
