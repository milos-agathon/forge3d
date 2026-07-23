use super::setup::RenderTargets;
use super::*;
use crate::core::resource_tracker::TrackedTexture;

impl TerrainScene {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::terrain::renderer) fn encode_forward_pass(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        params: &crate::terrain::render_params::TerrainRenderParams,
        decoded: &crate::terrain::render_params::DecodedTerrainSettings,
        height_inputs: &UploadedHeightInputs,
        materials: &PreparedMaterials,
        uniform_buffer: &Arc<crate::core::resource_tracker::TrackedBuffer>,
        ibl_bind_group: &wgpu::BindGroup,
        height_curve_view: &wgpu::TextureView,
        render_targets: &RenderTargets,
        shadow_setup: &crate::terrain::renderer::shadows::ShadowSetup,
        material_vt_ready: bool,
        height_ao_computed: bool,
        sun_vis_computed: bool,
        time_seconds: f32,
        timing: &mut Option<crate::core::gpu_timing::GpuTimingManager>,
    ) -> Result<()> {
        let shadow_bind_group = shadow_setup
            .shadow_bind_group
            .as_ref()
            .unwrap_or(&self.noop_shadow.bind_group);
        let sky_scope = ts_begin(timing, encoder, "terrain.sky");
        let sky_texture = self.render_sky_texture(
            encoder,
            decoded,
            shadow_setup.view_matrix,
            shadow_setup.proj_matrix,
            shadow_setup.eye,
            render_targets.internal_width,
            render_targets.internal_height,
        )?;
        ts_end(timing, encoder, sky_scope, 0);
        let sky_view = sky_texture
            .as_ref()
            .map(|(_, view)| view)
            .unwrap_or(&self.sky_fallback_view);
        let main_height_view = self.main_pass_height_view(&height_inputs.heightmap_view);
        let pass_bind_groups = self.create_terrain_pass_bind_groups(
            uniform_buffer,
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
            encoder,
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
            ibl_bind_group,
            shadow_bind_group,
            &pass_bind_groups.fog,
            &pass_bind_groups.material_layer,
        )?;
        if let Some((_, background_view)) = sky_texture.as_ref() {
            let scope = ts_begin(timing, encoder, "terrain.background");
            self.blit_background_texture(encoder, render_targets, background_view)?;
            ts_end(timing, encoder, scope, 1);
        }
        let main_scope = ts_begin(timing, encoder, "terrain.main");
        self.run_main_pass(
            encoder,
            params,
            render_targets,
            &pass_bind_groups.main,
            ibl_bind_group,
            shadow_bind_group,
            &pass_bind_groups.fog,
            &water_reflection_bind_group,
            &pass_bind_groups.material_layer,
            sky_texture.is_some(),
        )?;
        ts_end(timing, encoder, main_scope, 1);

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
                encoder,
                render_targets,
                &height_inputs.heightmap_view,
                shadow_setup.shadow_bind_group.as_ref(),
                &scatter_state,
            )?;
        }
        #[cfg(not(feature = "enable-gpu-instancing"))]
        let _ = time_seconds;
        Ok(())
    }

    pub(in crate::terrain::renderer) fn blit_background_texture(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        render_targets: &RenderTargets,
        source_view: &wgpu::TextureView,
    ) -> Result<()> {
        let blit_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain.background.blit.bind_group"),
            layout: &self.blit_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler_linear),
                },
            ],
        });

        let color_view = render_targets
            .msaa_view
            .as_ref()
            .unwrap_or(&render_targets.internal_view);
        let resolve_target = if render_targets.msaa_view.is_some() {
            Some(&render_targets.internal_view)
        } else {
            None
        };

        let msaa_pipeline = if render_targets.sample_count > 1 {
            Some(Self::create_depth_blit_pipeline(
                self.device.as_ref(),
                &self.blit_bind_group_layout,
                self.color_format,
                render_targets.sample_count,
            ))
        } else {
            None
        };
        let blit_pipeline = msaa_pipeline
            .as_ref()
            .unwrap_or(&self.background_blit_pipeline);

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("terrain.background.blit_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    resolve_target,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.1,
                            b: 0.15,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &render_targets.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            crate::core::shader_registry::record_shader_use(if render_targets.sample_count > 1 {
                "terrain.blit.depth.shader"
            } else {
                "terrain.blit.shader"
            });
            pass.set_pipeline(blit_pipeline);
            pass.set_bind_group(0, &blit_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_main_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        params: &crate::terrain::render_params::TerrainRenderParams,
        render_targets: &RenderTargets,
        bind_group: &wgpu::BindGroup,
        ibl_bind_group: &wgpu::BindGroup,
        shadow_bind_group: &wgpu::BindGroup,
        fog_bind_group: &wgpu::BindGroup,
        water_reflection_bind_group: &wgpu::BindGroup,
        material_layer_bind_group: &wgpu::BindGroup,
        preserve_background: bool,
    ) -> Result<()> {
        let pipeline_cache = self
            .pipeline
            .lock()
            .map_err(|_| anyhow!("TerrainRenderer pipeline mutex poisoned"))?;

        let color_view = render_targets
            .msaa_view
            .as_ref()
            .unwrap_or(&render_targets.internal_view);
        let resolve_target = if render_targets.msaa_view.is_some() {
            Some(&render_targets.internal_view)
        } else {
            None
        };

        let light_buffer_guard = self
            .light_buffer
            .lock()
            .map_err(|_| anyhow!("Light buffer mutex poisoned"))?;
        let light_bind_group = light_buffer_guard
            .bind_group()
            .expect("LightBuffer should always provide a bind group");

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("terrain.render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    resolve_target,
                    ops: wgpu::Operations {
                        load: if preserve_background {
                            wgpu::LoadOp::Load
                        } else {
                            wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.1,
                                g: 0.1,
                                b: 0.15,
                                a: 1.0,
                            })
                        },
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &render_targets.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: if preserve_background {
                            wgpu::LoadOp::Load
                        } else {
                            wgpu::LoadOp::Clear(1.0)
                        },
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            let geometry = self.geometry_provider()?;
            crate::core::shader_registry::record_shader_use(if geometry.is_clipmap() {
                "terrain_pbr_pom.clipmap.shader"
            } else {
                "terrain_pbr_pom.shader"
            });
            pass.set_pipeline(if geometry.is_clipmap() {
                pipeline_cache.clipmap_pipeline.as_ref().ok_or_else(|| {
                    anyhow!("clipmap pipeline not initialized for clipmap geometry")
                })?
            } else {
                &pipeline_cache.pipeline
            });
            pass.set_bind_group(0, bind_group, &[]);
            pass.set_bind_group(1, light_bind_group, &[]);
            pass.set_bind_group(2, ibl_bind_group, &[]);
            pass.set_bind_group(3, shadow_bind_group, &[]);
            pass.set_bind_group(4, fog_bind_group, &[]);
            pass.set_bind_group(5, water_reflection_bind_group, &[]);
            pass.set_bind_group(6, material_layer_bind_group, &[]);
            geometry.draw(&mut pass);
        }

        let _ = params;
        Ok(())
    }

    pub(in crate::terrain::renderer) fn resolve_output(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        params: &crate::terrain::render_params::TerrainRenderParams,
        decoded: &crate::terrain::render_params::DecodedTerrainSettings,
        render_targets: &RenderTargets,
    ) -> Result<(Arc<TrackedTexture>, u32, u32)> {
        if !render_targets.needs_scaling {
            return Ok((
                render_targets.resolved_texture.clone(),
                render_targets.out_width,
                render_targets.out_height,
            ));
        }
        let sampling = &decoded.sampling;
        let blit_sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("terrain.blit.sampler"),
            address_mode_u: Self::map_address_mode(sampling.address_u),
            address_mode_v: Self::map_address_mode(sampling.address_v),
            address_mode_w: Self::map_address_mode(sampling.address_w),
            mag_filter: Self::map_filter_mode(sampling.mag_filter),
            min_filter: Self::map_filter_mode(sampling.min_filter),
            mipmap_filter: Self::map_filter_mode(sampling.mip_filter),
            anisotropy_clamp: sampling.anisotropy as u16,
            ..Default::default()
        });
        let blit_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain.blit.bind_group"),
            layout: &self.blit_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&render_targets.internal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&blit_sampler),
                },
            ],
        });

        {
            let mut blit_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("terrain.blit_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &render_targets.resolved_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            crate::core::shader_registry::record_shader_use("terrain.blit.shader");
            blit_pass.set_pipeline(&self.blit_pipeline);
            blit_pass.set_bind_group(0, &blit_bind_group, &[]);
            blit_pass.draw(0..3, 0..1);
        }

        let _ = params;
        Ok((
            render_targets.resolved_texture.clone(),
            render_targets.out_width,
            render_targets.out_height,
        ))
    }
}
