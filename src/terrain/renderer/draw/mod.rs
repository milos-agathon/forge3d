use super::*;
use crate::core::resource_tracker::tracked_create_buffer_init;

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
        let (certificate_capture, _allocation_scope) =
            self.begin_certificate_capture("terrain.render_internal");
        let mut timing = self.take_render_timing();
        let decoded = params.decoded();
        self.prepare_frame_lighting(decoded)?;

        let height_inputs =
            self.upload_height_inputs(heightmap, water_mask, params.terrain_data_revision)?;
        self.prepare_geometry(params)?;
        let probe_world_span = if is_mesh_camera_mode(&params.camera_mode)
            || is_clipmap_camera_mode(&params.camera_mode)
        {
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
        )?;
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
        )?;
        let materials = self.prepare_material_context(material_set, params, decoded)?;

        let uniforms = self.build_uniforms(
            params,
            decoded,
            height_inputs.width as f32,
            height_inputs.height as f32,
        )?;
        super::runtime_contract::record_observation(
            "terrain.render_internal",
            &uniforms,
            &materials.shading_uniforms,
            &materials.overlay_binding.uniform,
            &height_inputs.heightmap_data,
            height_inputs.width,
            height_inputs.height,
        )
        .map_err(anyhow::Error::msg)?;
        let uniform_buffer = tracked_create_buffer_init(
            self.device.as_ref(),
            &wgpu::util::BufferInitDescriptor {
                label: Some("terrain.uniform_buffer"),
                contents: bytemuck::cast_slice(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        )?;

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

        self.ensure_pipeline_sample_count(
            effective_msaa,
            is_clipmap_camera_mode(&params.camera_mode),
        )?;
        let render_targets = self.create_render_targets(params, requested_msaa, effective_msaa)?;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("terrain.encoder"),
            });
        let vt_scope = ts_begin(&mut timing, &mut encoder, "terrain.material_vt");
        let material_vt_ready = self.prepare_material_vt_frame(
            &mut encoder,
            params,
            decoded,
            materials.gpu_materials.layer_count,
            render_targets.internal_width,
            render_targets.internal_height,
        )?;
        ts_end(&mut timing, &mut encoder, vt_scope, 0);

        let ao_scope = ts_begin(&mut timing, &mut encoder, "terrain.height_ao");
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
        ts_end(&mut timing, &mut encoder, ao_scope, 0);

        let sun_vis_scope = ts_begin(&mut timing, &mut encoder, "terrain.sun_visibility");
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
        ts_end(&mut timing, &mut encoder, sun_vis_scope, 0);

        let shadow_scope = ts_begin(&mut timing, &mut encoder, "terrain.shadow");
        let shadow_setup = self.prepare_shadow_setup(
            &mut encoder,
            params,
            decoded,
            &height_inputs.heightmap_view,
            height_inputs.width,
            height_inputs.height,
        )?;
        ts_end(&mut timing, &mut encoder, shadow_scope, 0);
        let shadow_bind_group = shadow_setup
            .shadow_bind_group
            .as_ref()
            .unwrap_or(&self.noop_shadow.bind_group);
        let sky_scope = ts_begin(&mut timing, &mut encoder, "terrain.sky");
        let sky_texture = self.render_sky_texture(
            &mut encoder,
            decoded,
            shadow_setup.view_matrix,
            shadow_setup.proj_matrix,
            shadow_setup.eye,
            render_targets.internal_width,
            render_targets.internal_height,
        )?;
        ts_end(&mut timing, &mut encoder, sky_scope, 0);
        let sky_view = sky_texture
            .as_ref()
            .map(|(_, view)| view)
            .unwrap_or(&self.sky_fallback_view);

        let height_curve_view = lut_texture_uploaded
            .as_ref()
            .map(|(_, view)| view as &wgpu::TextureView)
            .unwrap_or(&self.height_curve_identity_view);

        // When height streaming is active the mosaic supplies the geometry /
        // shading height source; coarse passes above (AO, sun visibility,
        // shadows) keep the per-render overview heightmap.
        let main_height_view = self.main_pass_height_view(&height_inputs.heightmap_view);
        let pass_bind_groups = self.create_terrain_pass_bind_groups(
            &uniform_buffer,
            main_height_view,
            materials.material_view(),
            materials.material_sampler(),
            materials.material_normal_view(),
            materials.material_roughness_view(),
            materials.material_mask_view(),
            materials.material_map_sampler(),
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
            material_vt_ready,
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
            main_height_view,
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
            let bg_scope = ts_begin(&mut timing, &mut encoder, "terrain.background");
            self.blit_background_texture(&mut encoder, &render_targets, background_view)?;
            ts_end(&mut timing, &mut encoder, bg_scope, 1);
        }

        let main_scope = ts_begin(&mut timing, &mut encoder, "terrain.main");
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
        ts_end(&mut timing, &mut encoder, main_scope, 1);

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
                shadow_setup.shadow_bind_group.as_ref(),
                &scatter_state,
            )?;
        }

        let resolve_scope = ts_begin(&mut timing, &mut encoder, "terrain.resolve");
        let (final_texture, final_width, final_height) =
            self.resolve_output(&mut encoder, params, decoded, render_targets)?;
        ts_end(&mut timing, &mut encoder, resolve_scope, 1);
        self.stage_material_vt_feedback_readback(&mut encoder)?;
        if let Some(t) = timing.as_mut() {
            t.resolve_queries(&mut encoder);
        }
        self.queue.submit(Some(encoder.finish()));
        self.finish_material_vt_frame()?;

        // Live GPU-pass timings for the certificate: read back the resolved
        // timestamps, record each pass, and freeze the render capture.
        self.record_render_timings(&mut timing);
        self.store_render_timing(timing);
        self.finish_certificate_capture(certificate_capture);

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
