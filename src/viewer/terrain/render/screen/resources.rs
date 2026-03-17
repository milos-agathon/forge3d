use super::ScreenRenderFlags;
use crate::viewer::terrain::ViewerTerrainScene;

impl ViewerTerrainScene {
    pub(super) fn prepare_screen_resources(
        &mut self,
        width: u32,
        height: u32,
    ) -> ScreenRenderFlags {
        self.ensure_depth(width, height);

        if self.pbr_config.enabled && self.pbr_pipeline.is_none() {
            if let Err(e) = self.init_pbr_pipeline(self.surface_format) {
                eprintln!("[render] Failed to initialize PBR pipeline: {}", e);
            }
        }

        let use_pbr = self.pbr_config.enabled && self.pbr_pipeline.is_some();
        let needs_dof = self.pbr_config.dof.enabled;
        let needs_post_process = self.pbr_config.lens_effects.enabled
            && (self.pbr_config.lens_effects.distortion.abs() > 0.001
                || self.pbr_config.lens_effects.chromatic_aberration > 0.001
                || self.pbr_config.lens_effects.vignette_strength > 0.001);
        let needs_volumetrics =
            self.pbr_config.volumetrics.enabled && self.pbr_config.volumetrics.density > 0.0001;
        let needs_denoise = self.pbr_config.denoise.enabled;
        let needs_dof_scratch = needs_volumetrics && needs_post_process && !needs_dof;

        if (needs_post_process || needs_volumetrics) && self.post_process.is_none() {
            self.init_post_process();
        }
        if (needs_dof || needs_dof_scratch) && self.dof_pass.is_none() {
            self.init_dof_pass();
        }
        if needs_volumetrics && self.volumetrics_pass.is_none() {
            self.init_volumetrics_pass();
        }
        if needs_denoise && self.denoise_pass.is_none() {
            self.init_denoise_pass();
        }

        if needs_denoise {
            if let Some(ref mut denoise) = self.denoise_pass {
                let _ = denoise.get_input_view(width, height);
            }
        }
        if needs_dof || needs_dof_scratch {
            if let Some(ref mut dof) = self.dof_pass {
                let _ = dof.get_input_view(width, height, self.surface_format);
            }
        }

        let has_vector_overlays_early = self
            .vector_overlay_stack
            .as_ref()
            .map(|s| s.visible_layer_count() > 0)
            .unwrap_or(false);
        if has_vector_overlays_early && self.oit_enabled {
            if self.wboit_compose_bind_group.is_none() || self.wboit_size != (width, height) {
                self.init_wboit(width, height);
            }
        }

        if needs_post_process || needs_volumetrics {
            if let Some(ref mut pp) = self.post_process {
                let _ = pp.get_intermediate_view(width, height, self.surface_format);
            }
        }

        ScreenRenderFlags {
            use_pbr,
            needs_dof,
            needs_post_process,
            needs_volumetrics,
            needs_denoise,
        }
    }
}
