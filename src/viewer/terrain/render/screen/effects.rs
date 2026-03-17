use super::{ScreenRenderFlags, ScreenRenderState};
use crate::viewer::terrain::dof;
use crate::viewer::terrain::post_process::PostProcessPass;
use crate::viewer::terrain::ViewerTerrainScene;

impl ViewerTerrainScene {
    pub(super) fn apply_screen_effects(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        width: u32,
        height: u32,
        flags: &ScreenRenderFlags,
        state: &ScreenRenderState,
    ) {
        if flags.needs_denoise {
            let (iterations, sigma_color) = {
                let config = &self.pbr_config.denoise;
                (config.iterations, config.sigma_color)
            };

            let ViewerTerrainScene {
                denoise_pass,
                post_process,
                dof_pass,
                depth_view,
                queue,
                device,
                surface_format,
                ..
            } = self;

            if let Some(denoise) = denoise_pass.as_mut() {
                let depth_view = depth_view.as_ref().unwrap();
                denoise.apply(encoder, depth_view, iterations, sigma_color);

                let denoise_result = denoise
                    .get_last_result_view(iterations)
                    .unwrap_or(denoise.view_a.as_ref().unwrap());

                if post_process.is_none() {
                    *post_process = Some(PostProcessPass::new(device.clone(), *surface_format));
                }

                let post_process = post_process.as_mut().unwrap();
                let mut intermediate_view = None;
                let next_target = if flags.needs_volumetrics {
                    intermediate_view = post_process.intermediate_view.take();
                    intermediate_view.as_ref().unwrap()
                } else if flags.needs_dof {
                    dof_pass.as_ref().unwrap().input_view.as_ref().unwrap()
                } else if flags.needs_post_process {
                    intermediate_view = post_process.intermediate_view.take();
                    intermediate_view.as_ref().unwrap()
                } else {
                    view
                };

                post_process.apply_from_input(
                    encoder,
                    queue,
                    denoise_result,
                    next_target,
                    width,
                    height,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                );

                if let Some(view) = intermediate_view {
                    post_process.intermediate_view = Some(view);
                }
            }
        }

        if flags.needs_volumetrics {
            if let Some(ref vol_pass) = self.volumetrics_pass {
                let terrain = self.terrain.as_ref().unwrap();
                let depth_view = self.depth_view.as_ref().unwrap();
                let color_input = self
                    .post_process
                    .as_ref()
                    .unwrap()
                    .intermediate_view
                    .as_ref()
                    .unwrap();
                let vol_output = if flags.needs_dof || flags.needs_post_process {
                    self.dof_pass.as_ref().unwrap().input_view.as_ref().unwrap()
                } else {
                    view
                };

                vol_pass.apply(
                    encoder,
                    &self.queue,
                    color_input,
                    depth_view,
                    &terrain.heightmap_view,
                    vol_output,
                    width,
                    height,
                    state.view_proj.inverse().to_cols_array_2d(),
                    [state.eye.x, state.eye.y, state.eye.z],
                    1.0,
                    state.cam_radius * 10.0,
                    [state.sun_dir.x, state.sun_dir.y, state.sun_dir.z],
                    terrain.sun_intensity,
                    [
                        state.terrain_width,
                        terrain.domain.0,
                        state.shader_z_scale,
                        state.h_range,
                    ],
                    &self.pbr_config.volumetrics,
                );
            }
        }

        if flags.needs_dof {
            let dof_output = if flags.needs_post_process {
                self.post_process
                    .as_ref()
                    .unwrap()
                    .intermediate_view
                    .as_ref()
                    .unwrap()
            } else {
                view
            };

            if let Some(ref mut dof) = self.dof_pass {
                let depth_view = self.depth_view.as_ref().unwrap();
                let dof_cfg = dof::DofConfig {
                    enabled: true,
                    focus_distance: self.pbr_config.dof.focus_distance,
                    f_stop: self.pbr_config.dof.f_stop,
                    focal_length: self.pbr_config.dof.focal_length,
                    quality: self.pbr_config.dof.quality,
                    max_blur_radius: self.pbr_config.dof.max_blur_radius,
                    blur_strength: self.pbr_config.dof.blur_strength,
                    tilt_pitch: self.pbr_config.dof.tilt_pitch,
                    tilt_yaw: self.pbr_config.dof.tilt_yaw,
                };
                dof.apply_from_input(
                    encoder,
                    &self.queue,
                    depth_view,
                    dof_output,
                    width,
                    height,
                    self.surface_format,
                    &dof_cfg,
                    1.0,
                    state.cam_radius * 10.0,
                );
            }
        }

        if flags.needs_post_process {
            let external_input = if !flags.needs_dof && flags.needs_volumetrics {
                self.dof_pass
                    .as_ref()
                    .and_then(|dof| dof.input_view.as_ref())
            } else {
                None
            };

            if let Some(ref mut pp) = self.post_process {
                let lens = &self.pbr_config.lens_effects;
                if let Some(input_view) = external_input {
                    pp.apply_from_input(
                        encoder,
                        &self.queue,
                        input_view,
                        view,
                        width,
                        height,
                        lens.distortion,
                        lens.chromatic_aberration,
                        lens.vignette_strength,
                        lens.vignette_radius,
                        lens.vignette_softness,
                    );
                } else {
                    pp.apply(
                        encoder,
                        &self.queue,
                        view,
                        width,
                        height,
                        lens.distortion,
                        lens.chromatic_aberration,
                        lens.vignette_strength,
                        lens.vignette_radius,
                        lens.vignette_softness,
                    );
                }
            }
        }
    }
}
