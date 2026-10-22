use super::*;

pub(super) struct FroxelState {
    resources: crate::viewer::init::FogResources,
    // Separate storage/sample bindings avoid simultaneous read/write use of
    // the real froxel volume, even with an explicit shared pipeline layout.
    dummy: TrackedTexture,
    csm: TrackedBuffer,
}

impl FroxelState {
    pub(super) fn new(
        scene: &TerrainScene,
        params: &crate::terrain::render_params::TerrainRenderParams,
        frame_layout: &wgpu::BindGroupLayout,
    ) -> Result<Self> {
        let source = super::shaders::froxel_source()?;
        let resources = crate::viewer::init::create_shared_fog_resources(
            &scene.device,
            params.size_px.0,
            params.size_px.1,
            &source,
            frame_layout,
        )?;
        let dummy = tracked_create_texture(
            &scene.device,
            &wgpu::TextureDescriptor {
                label: Some("concordia.froxel.unused"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        )?;
        // FogCsmUniforms: header(20 floats), four 20-float cascades, tail(8).
        let csm = tracked_create_buffer_init(
            &scene.device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("concordia.froxel.csm"),
                contents: bytemuck::cast_slice(&[0.0f32; 108]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        )?;
        Ok(Self {
            resources,
            dummy,
            csm,
        })
    }

    pub(super) fn encode(
        &mut self,
        scene: &TerrainScene,
        encoder: &mut wgpu::CommandEncoder,
        params: &crate::terrain::render_params::TerrainRenderParams,
        depth: &wgpu::TextureView,
        frame_group: &wgpu::BindGroup,
        frame_index: u32,
        camera_matrices: (glam::Vec3, glam::Mat4, glam::Mat4),
    ) -> Result<()> {
        let fog = &self.resources;
        let decoded = params.decoded();
        let vol = &decoded.volumetrics;
        let (eye, view, proj) = camera_matrices;
        let inv_proj = proj.inverse();
        let near_h = inv_proj * glam::Vec4::new(0.0, 0.0, 0.0, 1.0);
        let far_h = inv_proj * glam::Vec4::new(0.0, 0.0, 1.0, 1.0);
        let near = (-near_h.z / near_h.w).max(f32::MIN_POSITIVE);
        let far = (-far_h.z / far_h.w).max(near);
        let camera = crate::viewer::viewer_types::FogCameraUniforms {
            view: view.to_cols_array_2d(),
            proj: proj.to_cols_array_2d(),
            inv_view: view.inverse().to_cols_array_2d(),
            inv_proj: inv_proj.to_cols_array_2d(),
            view_proj: (proj * view).to_cols_array_2d(),
            eye_position: eye.to_array(),
            near,
            far,
            ..Zeroable::zeroed()
        };
        let uniforms = crate::viewer::viewer_types::VolumetricUniformsStd140 {
            density: vol.density,
            height_falloff: if vol.mode
                == crate::terrain::render_params::VolumetricsModeNative::Uniform
            {
                0.0
            } else {
                vol.height_falloff
            },
            phase_g: vol.phase_g,
            max_steps: vol.shaft_samples,
            start_distance: near,
            max_distance: far,
            scattering_color: decoded.light.color.map(|value| value * vol.scattering),
            absorption: vol.absorption,
            sun_direction: decoded.light.direction,
            sun_intensity: decoded.light.intensity
                * if vol.light_shafts {
                    vol.shaft_intensity
                } else {
                    1.0
                },
            ambient_color: decoded.fog.inscatter.map(|value| value * vol.scattering),
            temporal_alpha: 0.0,
            use_shadows: u32::from(vol.use_shadows && decoded.shadow.enabled),
            frame_index,
            _pad0: vol.base_height.to_bits(),
            ..Zeroable::zeroed()
        };
        scene
            .queue
            .write_buffer(&fog.fog_camera, 0, bytemuck::bytes_of(&camera));
        scene
            .queue
            .write_buffer(&fog.fog_params, 0, bytemuck::bytes_of(&uniforms));
        let source_csm = &scene.csm_renderer.uniforms;
        let mut csm = Vec::with_capacity(108);
        csm.extend(source_csm.light_direction);
        csm.extend(source_csm.light_view);
        for cascade in &source_csm.cascades {
            csm.extend(cascade.light_view_proj.iter().flatten().copied());
            csm.extend([
                cascade.far_distance,
                cascade.near_distance,
                cascade.texel_size,
                0.0,
            ]);
        }
        csm.extend([
            f32::from_bits(source_csm.cascade_count),
            f32::from_bits(source_csm.pcf_kernel_size),
            source_csm.depth_bias,
            source_csm.slope_bias,
            source_csm.shadow_map_size,
            0.0,
            0.0,
            0.0,
        ]);
        scene
            .queue
            .write_buffer(&self.csm, 0, bytemuck::cast_slice(&csm));
        let bg0 = scene.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("concordia.froxel.inputs"),
            layout: &fog.fog_bgl0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: fog.fog_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: fog.fog_camera.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(depth),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&fog.fog_depth_sampler),
                },
            ],
        });
        let shadow = scene.csm_renderer.shadow_texture_view();
        let bg1 = scene.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("concordia.froxel.shadows"),
            layout: &fog.fog_bgl1,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&shadow),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&scene.csm_renderer.shadow_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.csm.as_entire_binding(),
                },
            ],
        });
        let bg2 = scene.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("concordia.froxel.output"),
            layout: &fog.fog_bgl2,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&fog.fog_output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&fog.fog_history_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&fog.fog_history_sampler),
                },
            ],
        });
        let dummy = self.dummy.create_view(&Default::default());
        let make_bg3 = |storage: &wgpu::TextureView, sampled: &wgpu::TextureView| {
            scene.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("concordia.froxel.volume"),
                layout: &fog.fog_bgl3,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(storage),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(sampled),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&fog.froxel_sampler),
                    },
                ],
            })
        };
        let build = make_bg3(&fog.froxel_view, &dummy);
        let apply = make_bg3(&dummy, &fog.froxel_view);
        for (pipeline, volume, groups) in [
            (
                &fog.froxel_build_pipeline,
                &build,
                (
                    fog._froxel_tex.width().div_ceil(4),
                    fog._froxel_tex.height().div_ceil(4),
                    fog._froxel_tex.depth_or_array_layers().div_ceil(4),
                ),
            ),
            (
                &fog.froxel_apply_pipeline,
                &apply,
                (
                    params.size_px.0.div_ceil(8),
                    params.size_px.1.div_ceil(8),
                    1,
                ),
            ),
        ] {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("concordia.froxel"),
                timestamp_writes: None,
            });
            crate::core::shader_registry::record_shader_use("concordia.froxel.shader");
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bg0, &[]);
            pass.set_bind_group(1, &bg1, &[]);
            pass.set_bind_group(2, &bg2, &[]);
            pass.set_bind_group(3, volume, &[]);
            pass.set_bind_group(4, frame_group, &[]);
            pass.dispatch_workgroups(groups.0, groups.1, groups.2);
        }
        Ok(())
    }

    pub(super) fn resource_sizes(&self) -> serde_json::Value {
        serde_json::json!({"volume":texture_extent(&self.resources._froxel_tex),
            "apply":texture_extent(&self.resources.fog_output)})
    }

    pub(super) fn view(&self) -> &wgpu::TextureView {
        &self.resources.fog_output_view
    }
}
