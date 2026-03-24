use super::*;

mod execute;
mod setup;

pub(in crate::terrain::renderer) use setup::{
    PreparedMaterials, RenderTargets, UploadedHeightInputs,
};

impl TerrainScene {
    pub(crate) fn render_internal(
        &mut self,
        material_set: &crate::render::material_set::MaterialSet,
        env_maps: &crate::lighting::ibl_wrapper::IBL,
        params: &crate::terrain::render_params::TerrainRenderParams,
        heightmap: PyReadonlyArray2<f32>,
        water_mask: Option<PyReadonlyArray2<f32>>,
        time_seconds: f32,
    ) -> Result<crate::Frame> {
        let decoded = params.decoded();
        self.prepare_frame_lighting(decoded)?;

        let height_inputs =
            self.upload_height_inputs(heightmap, water_mask, params.terrain_data_revision)?;
        let probe_world_span = if params.camera_mode.to_lowercase() == "mesh" {
            params.terrain_span.max(1e-3)
        } else {
            1.0
        };
        super::probes::prepare_probes(
            self,
            &decoded.probes,
            probe_world_span,
            &height_inputs.heightmap_data,
            (height_inputs.width, height_inputs.height),
            params.z_scale,
            height_inputs.terrain_data_hash,
        );
        super::probes::prepare_reflection_probes(
            self,
            &decoded.reflection_probes,
            material_set,
            env_maps,
            params,
            decoded,
            probe_world_span,
            &height_inputs.heightmap_data,
            (height_inputs.width, height_inputs.height),
            params.z_scale,
            height_inputs.terrain_data_hash,
        );
        let materials = self.prepare_material_context(material_set, params, decoded)?;

        let uniforms = self.build_uniforms(
            params,
            decoded,
            height_inputs.width as f32,
            height_inputs.height as f32,
        )?;
        let uniform_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("terrain.uniform_buffer"),
                contents: bytemuck::cast_slice(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let ibl_bind_group = self.prepare_ibl_bind_group(env_maps)?;
        let lut_texture_uploaded = if params.height_curve_mode.as_str() == "lut" {
            params
                .height_curve_lut
                .as_ref()
                .map(|lut| self.upload_height_curve_lut(lut.as_ref().as_slice()))
                .transpose()?
        } else {
            None
        };

        let requested_msaa = params.msaa_samples.max(1);
        let effective_msaa =
            select_effective_msaa(requested_msaa, self.color_format, &self.adapter);
        if effective_msaa != requested_msaa {
            log::warn!(
                "MSAA: requested {} not supported for {:?}; using {}",
                requested_msaa,
                self.color_format,
                effective_msaa
            );
        }

        self.ensure_pipeline_sample_count(effective_msaa)?;
        let render_targets = self.create_render_targets(params, requested_msaa, effective_msaa)?;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("terrain.encoder"),
            });

        let height_ao_computed = self.compute_height_ao_pass(
            &mut encoder,
            &height_inputs.heightmap_view,
            render_targets.internal_width,
            render_targets.internal_height,
            height_inputs.width,
            height_inputs.height,
            params,
            decoded,
        )?;
        let sun_vis_computed = self.compute_sun_visibility_pass(
            &mut encoder,
            &height_inputs.heightmap_view,
            render_targets.internal_width,
            render_targets.internal_height,
            height_inputs.width,
            height_inputs.height,
            params,
            decoded,
        )?;

        let shadow_setup = self.prepare_shadow_setup(
            &mut encoder,
            params,
            decoded,
            &height_inputs.heightmap_view,
        )?;
        let shadow_bind_group = shadow_setup
            .shadow_bind_group
            .as_ref()
            .unwrap_or(&self.noop_shadow.bind_group);
        let sky_texture = self.render_sky_texture(
            &mut encoder,
            decoded,
            shadow_setup.view_matrix,
            shadow_setup.proj_matrix,
            shadow_setup.eye,
            render_targets.internal_width,
            render_targets.internal_height,
        )?;
        let sky_view = sky_texture
            .as_ref()
            .map(|(_, view)| view)
            .unwrap_or(&self.sky_fallback_view);

        let height_curve_view = lut_texture_uploaded
            .as_ref()
            .map(|(_, view)| view as &wgpu::TextureView)
            .unwrap_or(&self.height_curve_identity_view);

        let pass_bind_groups = self.create_terrain_pass_bind_groups(
            &uniform_buffer,
            &height_inputs.heightmap_view,
            materials.material_view(),
            materials.material_sampler(),
            &materials.shading_buffer,
            materials.colormap_view(),
            materials.colormap_sampler(),
            &materials.overlay_buffer,
            height_curve_view,
            height_inputs.water_mask_view_uploaded.as_ref(),
            sky_view,
            height_ao_computed,
            sun_vis_computed,
            decoded,
            shadow_setup.height_min,
            shadow_setup.height_exag,
            shadow_setup.eye.y,
        )?;

        let water_reflection_bind_group = self.prepare_water_reflection_bind_group(
            &mut encoder,
            params,
            decoded,
            render_targets.internal_width,
            render_targets.internal_height,
            shadow_setup.eye,
            shadow_setup.view_matrix,
            shadow_setup.proj_matrix,
            &height_inputs.heightmap_view,
            materials.material_view(),
            materials.material_sampler(),
            &materials.shading_buffer,
            materials.colormap_view(),
            materials.colormap_sampler(),
            &materials.overlay_buffer,
            height_curve_view,
            height_inputs.water_mask_view_uploaded.as_ref(),
            height_ao_computed,
            sun_vis_computed,
            &ibl_bind_group,
            shadow_bind_group,
            &pass_bind_groups.fog,
            &pass_bind_groups.material_layer,
        )?;

        if let Some((_, background_view)) = sky_texture.as_ref() {
            self.blit_background_texture(&mut encoder, &render_targets, background_view)?;
        }

        self.run_main_pass(
            &mut encoder,
            params,
            &render_targets,
            &pass_bind_groups.main,
            &ibl_bind_group,
            shadow_bind_group,
            &pass_bind_groups.fog,
            &water_reflection_bind_group,
            &pass_bind_groups.material_layer,
            sky_texture.is_some(),
        )?;

        #[cfg(feature = "enable-gpu-instancing")]
        {
            let scatter_state = self.build_scatter_render_state(
                params,
                decoded,
                height_inputs.width,
                height_inputs.height,
                shadow_setup.view_matrix,
                shadow_setup.proj_matrix,
                shadow_setup.eye,
                time_seconds,
            );
            self.render_scatter_pass(
                &mut encoder,
                &render_targets,
                &height_inputs.heightmap_view,
                &scatter_state,
            )?;
        }

        let (final_texture, final_width, final_height) =
            self.resolve_output(&mut encoder, params, decoded, render_targets)?;
        self.queue.submit(Some(encoder.finish()));

        Ok(crate::Frame::new(
            self.device.clone(),
            self.queue.clone(),
            final_texture,
            final_width,
            final_height,
            self.color_format,
        ))
    }
}
