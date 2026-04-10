use super::SnapshotRenderState;
use crate::viewer::terrain::dof;
use crate::viewer::terrain::ViewerTerrainScene;

impl ViewerTerrainScene {
    pub(super) fn apply_snapshot_effects(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        depth_view: &wgpu::TextureView,
        color_tex: wgpu::Texture,
        color_view: wgpu::TextureView,
        state: &SnapshotRenderState,
    ) -> wgpu::Texture {
        let mut out_tex = color_tex;
        let mut out_view = color_view;

        let needs_volumetrics = self.pbr_config.volumetrics.is_effectively_enabled();
        if needs_volumetrics {
            if self.volumetrics_pass.is_none() {
                self.init_volumetrics_pass();
            }

            let (vol_output_tex, vol_output_view) = self.create_snapshot_color_target(
                "terrain_viewer.snapshot_vol_output",
                target_format,
                width,
                height,
            );
            if let Some(ref mut vol_pass) = self.volumetrics_pass {
                let terrain = self.terrain.as_ref().unwrap();
                let cam_radius = terrain.cam_radius;
                let terrain_sun_intensity = terrain.sun_intensity;

                vol_pass.apply(
                    encoder,
                    &self.queue,
                    &out_view,
                    depth_view,
                    &terrain.heightmap_view,
                    &terrain.heightmap,
                    terrain.dimensions,
                    terrain.revision,
                    &vol_output_view,
                    width,
                    height,
                    state.view_proj.inverse().to_cols_array_2d(),
                    [state.eye.x, state.eye.y, state.eye.z],
                    1.0,
                    cam_radius * 10.0,
                    [state.sun_dir.x, state.sun_dir.y, state.sun_dir.z],
                    terrain_sun_intensity,
                    [
                        state.terrain_width,
                        terrain.domain.0,
                        state.shader_z_scale,
                        state.h_range,
                    ],
                    &self.pbr_config.volumetrics,
                );

                out_tex = vol_output_tex;
                out_view = vol_output_view;
            }
        }

        let needs_dof = self.pbr_config.dof.enabled;
        if needs_dof {
            if self.dof_pass.is_none() {
                self.init_dof_pass();
            }

            let (dof_output_tex, dof_output_view) = self.create_snapshot_color_target(
                "terrain_viewer.snapshot_dof_output",
                target_format,
                width,
                height,
            );
            if let Some(ref mut dof) = self.dof_pass {
                let _ = dof.get_input_view(width, height, target_format);
                let cam_radius = self
                    .terrain
                    .as_ref()
                    .map(|t| t.cam_radius)
                    .unwrap_or(2000.0);
                let dof_cfg = dof::DofConfig {
                    focus_distance: self.pbr_config.dof.focus_distance,
                    f_stop: self.pbr_config.dof.f_stop,
                    focal_length: self.pbr_config.dof.focal_length,
                    quality: self.pbr_config.dof.quality,
                    max_blur_radius: self.pbr_config.dof.max_blur_radius,
                    blur_strength: self.pbr_config.dof.blur_strength,
                    tilt_pitch: self.pbr_config.dof.tilt_pitch,
                    tilt_yaw: self.pbr_config.dof.tilt_yaw,
                };

                dof.apply(
                    encoder,
                    &self.queue,
                    &out_view,
                    depth_view,
                    &dof_output_view,
                    width,
                    height,
                    target_format,
                    &dof_cfg,
                    1.0,
                    cam_radius * 10.0,
                );

                out_tex = dof_output_tex;
                out_view = dof_output_view;
            }
        }

        let needs_post_process = self.pbr_config.lens_effects.enabled
            && (self.pbr_config.lens_effects.distortion.abs() > 0.001
                || self.pbr_config.lens_effects.chromatic_aberration > 0.001
                || self.pbr_config.lens_effects.vignette_strength > 0.001);
        if needs_post_process {
            if self.post_process.is_none() {
                self.init_post_process();
            }

            let (lens_output_tex, lens_output_view) = self.create_snapshot_color_target(
                "terrain_viewer.snapshot_lens_output",
                target_format,
                width,
                height,
            );
            if let Some(ref mut pp) = self.post_process {
                let lens = &self.pbr_config.lens_effects;

                pp.apply_from_input(
                    encoder,
                    &self.queue,
                    &out_view,
                    &lens_output_view,
                    width,
                    height,
                    lens.distortion,
                    lens.chromatic_aberration,
                    lens.vignette_strength,
                    lens.vignette_radius,
                    lens.vignette_softness,
                );
                return lens_output_tex;
            }
        }

        out_tex
    }
}
