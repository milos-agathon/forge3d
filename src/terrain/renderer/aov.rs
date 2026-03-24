use super::draw::RenderTargets;
use super::*;
use crate::terrain::render_params;

const TERRAIN_AOV_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

struct AovAttachmentTarget {
    internal_texture: wgpu::Texture,
    internal_view: wgpu::TextureView,
    _msaa_texture: Option<wgpu::Texture>,
    msaa_view: Option<wgpu::TextureView>,
}

struct TerrainAovTargets {
    albedo: AovAttachmentTarget,
    normal: AovAttachmentTarget,
    depth: AovAttachmentTarget,
}

impl TerrainScene {
    fn ensure_aov_pipeline_sample_count(&self, effective_msaa: u32) -> Result<()> {
        let mut aov_pipeline = self
            .aov_pipeline
            .lock()
            .map_err(|_| anyhow!("TerrainRenderer AOV pipeline mutex poisoned"))?;
        let mut sample_count = self
            .aov_pipeline_sample_count
            .lock()
            .map_err(|_| anyhow!("TerrainRenderer AOV sample count mutex poisoned"))?;

        if aov_pipeline.is_none() || *sample_count != effective_msaa {
            let light_buffer = self
                .light_buffer
                .lock()
                .map_err(|_| anyhow!("Light buffer mutex poisoned"))?;
            *aov_pipeline = Some(Self::create_aov_render_pipeline(
                self.device.as_ref(),
                &self.bind_group_layout,
                light_buffer.bind_group_layout(),
                &self.ibl_bind_group_layout,
                &self.shadow_bind_group_layout,
                &self.fog_bind_group_layout,
                &self.water_reflection_bind_group_layout,
                &self.material_layer_bind_group_layout,
                self.color_format,
                effective_msaa,
            ));
            *sample_count = effective_msaa;
        }

        Ok(())
    }

    fn create_aov_attachment_target(
        &self,
        label: &str,
        width: u32,
        height: u32,
        sample_count: u32,
    ) -> AovAttachmentTarget {
        let internal_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TERRAIN_AOV_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let internal_view = internal_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let msaa_texture = if sample_count > 1 {
            Some(self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("{label}.msaa")),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count,
                dimension: wgpu::TextureDimension::D2,
                format: TERRAIN_AOV_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            }))
        } else {
            None
        };
        let msaa_view = msaa_texture
            .as_ref()
            .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()));

        AovAttachmentTarget {
            internal_texture,
            internal_view,
            _msaa_texture: msaa_texture,
            msaa_view,
        }
    }

    fn create_aov_render_targets(
        &self,
        width: u32,
        height: u32,
        sample_count: u32,
    ) -> TerrainAovTargets {
        TerrainAovTargets {
            albedo: self.create_aov_attachment_target(
                "terrain.aov.albedo",
                width,
                height,
                sample_count,
            ),
            normal: self.create_aov_attachment_target(
                "terrain.aov.normal",
                width,
                height,
                sample_count,
            ),
            depth: self.create_aov_attachment_target(
                "terrain.aov.depth",
                width,
                height,
                sample_count,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_main_pass_with_aov(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        params: &crate::terrain::render_params::TerrainRenderParams,
        render_targets: &RenderTargets,
        aov_targets: &TerrainAovTargets,
        bind_group: &wgpu::BindGroup,
        ibl_bind_group: &wgpu::BindGroup,
        shadow_bind_group: &wgpu::BindGroup,
        fog_bind_group: &wgpu::BindGroup,
        water_reflection_bind_group: &wgpu::BindGroup,
        material_layer_bind_group: &wgpu::BindGroup,
        preserve_background: bool,
    ) -> Result<()> {
        let aov_pipeline_guard = self
            .aov_pipeline
            .lock()
            .map_err(|_| anyhow!("TerrainRenderer AOV pipeline mutex poisoned"))?;
        let pipeline = aov_pipeline_guard
            .as_ref()
            .ok_or_else(|| anyhow!("terrain AOV pipeline not initialized"))?;

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

        let albedo_view = aov_targets
            .albedo
            .msaa_view
            .as_ref()
            .unwrap_or(&aov_targets.albedo.internal_view);
        let albedo_resolve = if aov_targets.albedo.msaa_view.is_some() {
            Some(&aov_targets.albedo.internal_view)
        } else {
            None
        };

        let normal_view = aov_targets
            .normal
            .msaa_view
            .as_ref()
            .unwrap_or(&aov_targets.normal.internal_view);
        let normal_resolve = if aov_targets.normal.msaa_view.is_some() {
            Some(&aov_targets.normal.internal_view)
        } else {
            None
        };

        let depth_view = aov_targets
            .depth
            .msaa_view
            .as_ref()
            .unwrap_or(&aov_targets.depth.internal_view);
        let depth_resolve = if aov_targets.depth.msaa_view.is_some() {
            Some(&aov_targets.depth.internal_view)
        } else {
            None
        };

        let color_attachments = [
            Some(wgpu::RenderPassColorAttachment {
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
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: albedo_view,
                resolve_target: albedo_resolve,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: normal_view,
                resolve_target: normal_resolve,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: depth_view,
                resolve_target: depth_resolve,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            }),
        ];

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("terrain.render_pass.aov"),
                color_attachments: &color_attachments,
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

            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.set_bind_group(1, light_bind_group, &[]);
            pass.set_bind_group(2, ibl_bind_group, &[]);
            pass.set_bind_group(3, shadow_bind_group, &[]);
            pass.set_bind_group(4, fog_bind_group, &[]);
            pass.set_bind_group(5, water_reflection_bind_group, &[]);
            pass.set_bind_group(6, material_layer_bind_group, &[]);

            let vertex_count = if params.camera_mode.to_lowercase() == "mesh" {
                let grid_size: u32 = 512;
                6 * (grid_size - 1) * (grid_size - 1)
            } else {
                3
            };
            pass.draw(0..vertex_count, 0..1);
        }

        Ok(())
    }

    fn resolve_aux_output(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        decoded: &crate::terrain::render_params::DecodedTerrainSettings,
        internal_texture: wgpu::Texture,
        internal_view: wgpu::TextureView,
        out_width: u32,
        out_height: u32,
        needs_scaling: bool,
        renormalize_normals: bool,
        label: &str,
    ) -> Result<wgpu::Texture> {
        if !needs_scaling {
            return Ok(internal_texture);
        }

        let output_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: out_width,
                height: out_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TERRAIN_AOV_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let output_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let sampling = &decoded.sampling;
        let blit_sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("terrain.aov.blit.sampler"),
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
            label: Some("terrain.aov.blit.bind_group"),
            layout: &self.blit_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&internal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&blit_sampler),
                },
            ],
        });
        let blit_pipeline = if renormalize_normals {
            &self.normal_blit_pipeline
        } else {
            &self.aov_blit_pipeline
        };

        {
            let mut blit_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("terrain.aov.blit_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &output_view,
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
            blit_pass.set_pipeline(blit_pipeline);
            blit_pass.set_bind_group(0, &blit_bind_group, &[]);
            blit_pass.draw(0..3, 0..1);
        }

        Ok(output_texture)
    }

    /// Internal render method with populated terrain AOV capture.
    /// Returns (beauty_frame, aov_frame) tuple.
    pub(crate) fn render_internal_with_aov(
        &mut self,
        material_set: &crate::render::material_set::MaterialSet,
        env_maps: &crate::lighting::ibl_wrapper::IBL,
        params: &render_params::TerrainRenderParams,
        heightmap: numpy::PyReadonlyArray2<'_, f32>,
        water_mask: Option<numpy::PyReadonlyArray2<'_, f32>>,
        time_seconds: f32,
    ) -> Result<(crate::Frame, crate::AovFrame)> {
        let decoded = params.decoded();
        self.prepare_frame_lighting(decoded)?;

        let height_inputs = self.upload_height_inputs(heightmap, water_mask)?;
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
        );
        super::probes::prepare_reflection_probes(
            self,
            &decoded.reflection_probes,
            probe_world_span,
            &height_inputs.heightmap_data,
            (height_inputs.width, height_inputs.height),
            params.z_scale,
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
        self.ensure_aov_pipeline_sample_count(effective_msaa)?;
        let render_targets = self.create_render_targets(params, requested_msaa, effective_msaa)?;
        let aov_targets = self.create_aov_render_targets(
            render_targets.internal_width,
            render_targets.internal_height,
            effective_msaa,
        );

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("terrain.encoder.aov"),
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

        self.run_main_pass_with_aov(
            &mut encoder,
            params,
            &render_targets,
            &aov_targets,
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

        let needs_scaling = render_targets.needs_scaling;
        let (final_texture, final_width, final_height) =
            self.resolve_output(&mut encoder, params, decoded, render_targets)?;

        let albedo_texture = self.resolve_aux_output(
            &mut encoder,
            decoded,
            aov_targets.albedo.internal_texture,
            aov_targets.albedo.internal_view,
            final_width,
            final_height,
            needs_scaling,
            false,
            "terrain.aov.albedo.resolved",
        )?;
        let normal_texture = self.resolve_aux_output(
            &mut encoder,
            decoded,
            aov_targets.normal.internal_texture,
            aov_targets.normal.internal_view,
            final_width,
            final_height,
            needs_scaling,
            true,
            "terrain.aov.normal.resolved",
        )?;
        let depth_texture = self.resolve_aux_output(
            &mut encoder,
            decoded,
            aov_targets.depth.internal_texture,
            aov_targets.depth.internal_view,
            final_width,
            final_height,
            needs_scaling,
            false,
            "terrain.aov.depth.resolved",
        )?;

        self.queue.submit(Some(encoder.finish()));

        let aov_config = &decoded.aov;
        let aov_frame = crate::AovFrame::new(
            self.device.clone(),
            self.queue.clone(),
            if aov_config.albedo {
                Some(albedo_texture)
            } else {
                None
            },
            if aov_config.normal {
                Some(normal_texture)
            } else {
                None
            },
            if aov_config.depth {
                Some(depth_texture)
            } else {
                None
            },
            final_width,
            final_height,
        );

        let beauty_frame = crate::Frame::new(
            self.device.clone(),
            self.queue.clone(),
            final_texture,
            final_width,
            final_height,
            self.color_format,
        );

        Ok((beauty_frame, aov_frame))
    }

    pub(super) fn log_color_debug(
        &self,
        _params: &render_params::TerrainRenderParams,
        binding: &OverlayBinding,
    ) {
        let debug_mode = binding.uniform.params1[1] as i32;
        let albedo_mode = match binding.uniform.params1[2] as i32 {
            0 => "material",
            1 => "colormap",
            2 => "mix",
            _ => "unknown",
        };
        let blend_mode = match binding.uniform.params1[0] as i32 {
            0 => "Replace",
            1 => "Alpha",
            2 => "Multiply",
            3 => "Additive",
            _ => "unknown",
        };

        log::info!(target: "color.debug", "╔══════════════════════════════════════════════════");
        log::info!(target: "color.debug", "║ Color Configuration Debug");
        log::info!(target: "color.debug", "╠══════════════════════════════════════════════════");
        log::info!(target: "color.debug", "║ Domain: [{}, {}]", binding.uniform.params0[0],
            binding.uniform.params0[0] + 1.0 / binding.uniform.params0[1].max(1e-6));
        log::info!(target: "color.debug", "║ Overlay Strength: {}", binding.uniform.params0[2]);
        log::info!(target: "color.debug", "║ Colormap Strength: {}", binding.uniform.params1[3]);
        log::info!(target: "color.debug", "║ Albedo Mode: {}", albedo_mode);
        log::info!(target: "color.debug", "║ Blend Mode: {}", blend_mode);
        log::info!(target: "color.debug", "║ Debug Mode: {}", debug_mode);
        log::info!(target: "color.debug", "║ Gamma: {}", binding.uniform.params2[0]);
        log::info!(target: "color.debug", "║ Roughness Mult: {}", binding.uniform.params2[1]);
        log::info!(target: "color.debug", "║ Spec AA Enabled: {}", binding.uniform.params2[2]);

        if binding.lut.is_some() {
            log::info!(target: "color.debug", "╠══════════════════════════════════════════════════");
            log::info!(target: "color.debug", "║ LUT Samples:");
            log::info!(target: "color.debug", "║   t=0.0 probe ready");
            log::info!(target: "color.debug", "║   t=0.5 probe ready");
            log::info!(target: "color.debug", "║   t=1.0 probe ready");
            log::info!(target: "color.debug", "║ LUT texture bound: yes");
        } else {
            log::info!(target: "color.debug", "║ LUT texture bound: no");
        }
        log::info!(target: "color.debug", "╚══════════════════════════════════════════════════");
    }
}
