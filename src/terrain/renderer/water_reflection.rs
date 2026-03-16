use super::*;
use crate::terrain::render_params;

pub(super) struct WaterReflectionInitResources {
    pub(super) water_reflection_uniform_buffer: wgpu::Buffer,
    pub(super) water_reflection_texture: wgpu::Texture,
    pub(super) water_reflection_view: wgpu::TextureView,
    pub(super) water_reflection_sampler: wgpu::Sampler,
    pub(super) water_reflection_depth_texture: wgpu::Texture,
    pub(super) water_reflection_depth_view: wgpu::TextureView,
    pub(super) water_reflection_size: (u32, u32),
    pub(super) water_reflection_fallback_view: wgpu::TextureView,
}

pub(super) fn create_water_reflection_init_resources(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    color_format: wgpu::TextureFormat,
) -> WaterReflectionInitResources {
    let water_reflection_uniform_buffer =
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("terrain.water_reflection.uniform_buffer"),
            contents: bytemuck::bytes_of(&WaterReflectionUniforms::disabled()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

    let water_reflection_resolution = 512u32;
    let water_reflection_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("terrain.water_reflection.texture"),
        size: wgpu::Extent3d {
            width: water_reflection_resolution,
            height: water_reflection_resolution,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: color_format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let water_reflection_view =
        water_reflection_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let water_reflection_depth_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("terrain.water_reflection.depth"),
        size: wgpu::Extent3d {
            width: water_reflection_resolution,
            height: water_reflection_resolution,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let water_reflection_depth_view =
        water_reflection_depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let water_reflection_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("terrain.water_reflection.sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    let water_reflection_fallback_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("terrain.water_reflection.fallback"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: color_format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &water_reflection_fallback_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &[0u8; 4],
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    let water_reflection_fallback_view =
        water_reflection_fallback_texture.create_view(&wgpu::TextureViewDescriptor::default());

    WaterReflectionInitResources {
        water_reflection_uniform_buffer,
        water_reflection_texture,
        water_reflection_view,
        water_reflection_sampler,
        water_reflection_depth_texture,
        water_reflection_depth_view,
        water_reflection_size: (water_reflection_resolution, water_reflection_resolution),
        water_reflection_fallback_view,
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct WaterReflectionUniforms {
    pub(super) reflection_view_proj: [[f32; 4]; 4],
    pub(super) water_plane: [f32; 4],
    pub(super) reflection_params: [f32; 4],
    pub(super) camera_world_pos: [f32; 4],
    pub(super) enable_flags: [f32; 4],
}

impl WaterReflectionUniforms {
    pub(super) fn disabled() -> Self {
        Self {
            reflection_view_proj: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            water_plane: [0.0, 1.0, 0.0, 0.0],
            reflection_params: [0.8, 5.0, 0.02, 0.3],
            camera_world_pos: [0.0, 0.0, 0.0, 1.0],
            enable_flags: [0.0, 0.0, 0.5, 0.0],
        }
    }

    pub(super) fn enabled_main_pass(
        reflection_view_proj: [[f32; 4]; 4],
        water_plane_height: f32,
        camera_pos: [f32; 3],
        intensity: f32,
        fresnel_power: f32,
        wave_strength: f32,
        shore_atten_width: f32,
        resolution_scale: f32,
    ) -> Self {
        Self {
            reflection_view_proj,
            water_plane: [0.0, 0.0, 1.0, -water_plane_height],
            reflection_params: [intensity, fresnel_power, wave_strength, shore_atten_width],
            camera_world_pos: [camera_pos[0], camera_pos[1], camera_pos[2], 1.0],
            enable_flags: [1.0, 0.0, resolution_scale, 0.0],
        }
    }

    pub(super) fn for_reflection_pass(water_plane_height: f32) -> Self {
        Self {
            reflection_view_proj: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            water_plane: [0.0, 0.0, 1.0, -water_plane_height],
            reflection_params: [0.0, 0.0, 0.0, 0.0],
            camera_world_pos: [0.0, 0.0, 0.0, 1.0],
            enable_flags: [0.0, 1.0, 0.0, 0.0],
        }
    }
}

pub(super) fn compute_mirrored_view_matrix(
    view_matrix: [[f32; 4]; 4],
    plane_height: f32,
) -> [[f32; 4]; 4] {
    let reflect = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 2.0 * plane_height],
        [0.0, 0.0, 0.0, 1.0],
    ];
    mul_mat4(view_matrix, reflect)
}

pub(super) fn mul_mat4(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut result = [[0.0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    result
}

impl TerrainScene {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prepare_water_reflection_bind_group(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        params: &render_params::TerrainRenderParams,
        decoded: &render_params::DecodedTerrainSettings,
        internal_width: u32,
        internal_height: u32,
        eye: glam::Vec3,
        view_matrix: glam::Mat4,
        proj_matrix: glam::Mat4,
        heightmap_view: &wgpu::TextureView,
        material_view: &wgpu::TextureView,
        material_sampler: &wgpu::Sampler,
        shading_buffer: &wgpu::Buffer,
        colormap_view: &wgpu::TextureView,
        colormap_sampler: &wgpu::Sampler,
        overlay_buffer: &wgpu::Buffer,
        height_curve_view: &wgpu::TextureView,
        water_mask_view_uploaded: Option<&wgpu::TextureView>,
        ibl_bind_group: &wgpu::BindGroup,
        shadow_bind_group: &wgpu::BindGroup,
        fog_bind_group: &wgpu::BindGroup,
    ) -> Result<wgpu::BindGroup> {
        let reflection_settings = &decoded.reflection;
        self.ensure_reflection_texture_size(internal_width, internal_height)?;

        if reflection_settings.enabled {
            let mirrored_view = {
                let view_arr: [[f32; 4]; 4] = view_matrix.to_cols_array_2d();
                let mirrored_arr =
                    compute_mirrored_view_matrix(view_arr, reflection_settings.water_plane_height);
                glam::Mat4::from_cols_array_2d(&mirrored_arr)
            };

            let reflection_uniforms =
                Self::build_uniforms_with_matrices(params, decoded, mirrored_view, proj_matrix);
            let reflection_uniform_buffer =
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("terrain.reflection.uniform_buffer"),
                        contents: bytemuck::cast_slice(&reflection_uniforms),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

            let reflection_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("terrain.reflection.bind_group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: reflection_uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(heightmap_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.sampler_linear),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(material_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(material_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: shading_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::TextureView(colormap_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::Sampler(colormap_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: overlay_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: wgpu::BindingResource::TextureView(height_curve_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: wgpu::BindingResource::Sampler(&self.height_curve_lut_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 11,
                        resource: wgpu::BindingResource::TextureView(
                            water_mask_view_uploaded.unwrap_or(&self.water_mask_fallback_view),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 12,
                        resource: wgpu::BindingResource::TextureView(
                            self.ao_debug_view
                                .as_ref()
                                .or(self.coarse_ao_view.as_ref())
                                .unwrap_or(&self.ao_debug_fallback_view),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 13,
                        resource: wgpu::BindingResource::Sampler(&self.ao_debug_sampler),
                    },
                ],
            });

            let reflection_pass_water_uniforms = WaterReflectionUniforms::for_reflection_pass(
                reflection_settings.water_plane_height,
            );
            let reflection_pass_water_uniform_buffer =
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("terrain.reflection_pass.water_uniform_buffer"),
                        contents: bytemuck::bytes_of(&reflection_pass_water_uniforms),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

            let reflection_view_guard = self
                .water_reflection_view
                .lock()
                .map_err(|_| anyhow!("water_reflection_view mutex poisoned"))?;
            let reflection_pass_water_bind_group =
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("terrain.reflection_pass.water_bind_group"),
                    layout: &self.water_reflection_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: reflection_pass_water_uniform_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(
                                &self.water_reflection_fallback_view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(
                                &self.water_reflection_sampler,
                            ),
                        },
                    ],
                });

            let light_buffer_guard = self
                .light_buffer
                .lock()
                .map_err(|_| anyhow!("Light buffer mutex poisoned"))?;
            let light_bind_group = light_buffer_guard
                .bind_group()
                .expect("LightBuffer should always provide a bind group");

            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("terrain.reflection_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &*reflection_view_guard,
                        resolve_target: None,
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
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                pass.set_pipeline(&self.water_reflection_pipeline);
                pass.set_bind_group(0, &reflection_bind_group, &[]);
                pass.set_bind_group(1, light_bind_group, &[]);
                pass.set_bind_group(2, ibl_bind_group, &[]);
                pass.set_bind_group(3, shadow_bind_group, &[]);
                pass.set_bind_group(4, fog_bind_group, &[]);
                pass.set_bind_group(5, &reflection_pass_water_bind_group, &[]);
                let vertex_count = if params.camera_mode.to_lowercase() == "mesh" {
                    let grid_size: u32 = 512;
                    6 * (grid_size - 1) * (grid_size - 1)
                } else {
                    3
                };
                pass.draw(0..vertex_count, 0..1);
            }

            drop(light_buffer_guard);
            drop(reflection_view_guard);

            log::info!(
                target: "terrain.water_reflection",
                "P4: Rendered reflection pass at {}x{} (plane_height={:.2})",
                internal_width / 2,
                internal_height / 2,
                reflection_settings.water_plane_height
            );
        }

        let water_reflection_uniforms = if reflection_settings.enabled {
            let view_arr: [[f32; 4]; 4] = view_matrix.to_cols_array_2d();
            let mirrored_view =
                compute_mirrored_view_matrix(view_arr, reflection_settings.water_plane_height);
            let proj_arr: [[f32; 4]; 4] = proj_matrix.to_cols_array_2d();
            let reflection_view_proj = mul_mat4(proj_arr, mirrored_view);
            WaterReflectionUniforms::enabled_main_pass(
                reflection_view_proj,
                reflection_settings.water_plane_height,
                eye.to_array(),
                reflection_settings.intensity,
                reflection_settings.fresnel_power,
                reflection_settings.wave_strength,
                reflection_settings.shore_atten_width,
                0.5,
            )
        } else {
            WaterReflectionUniforms::disabled()
        };
        self.queue.write_buffer(
            &self.water_reflection_uniform_buffer,
            0,
            bytemuck::bytes_of(&water_reflection_uniforms),
        );

        let reflection_view_guard = self
            .water_reflection_view
            .lock()
            .map_err(|_| anyhow!("water_reflection_view mutex poisoned"))?;
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain.water_reflection.bind_group"),
            layout: &self.water_reflection_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.water_reflection_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&*reflection_view_guard),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.water_reflection_sampler),
                },
            ],
        });
        drop(reflection_view_guard);

        Ok(bind_group)
    }
}
