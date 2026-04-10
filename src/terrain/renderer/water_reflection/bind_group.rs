use super::uniforms::{compute_mirrored_view_matrix, mul_mat4, WaterReflectionUniforms};
use super::*;

impl TerrainScene {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::terrain::renderer) fn prepare_water_reflection_bind_group(
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
        height_ao_computed: bool,
        sun_vis_computed: bool,
        ibl_bind_group: &wgpu::BindGroup,
        shadow_bind_group: &wgpu::BindGroup,
        fog_bind_group: &wgpu::BindGroup,
        material_layer_bind_group: &wgpu::BindGroup,
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

            let reflection_bind_group = self.create_terrain_main_bind_group(
                "terrain.reflection.bind_group",
                &reflection_uniform_buffer,
                heightmap_view,
                material_view,
                material_sampler,
                shading_buffer,
                colormap_view,
                colormap_sampler,
                overlay_buffer,
                height_curve_view,
                water_mask_view_uploaded,
                height_ao_computed,
                sun_vis_computed,
            )?;

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
            let reflection_depth_guard = self
                .water_reflection_depth_view
                .lock()
                .map_err(|_| anyhow!("water_reflection_depth_view mutex poisoned"))?;
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
                        view: &reflection_view_guard,
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
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &reflection_depth_guard,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
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
                pass.set_bind_group(6, material_layer_bind_group, &[]);
                let vertex_count = if params.camera_mode.to_lowercase() == "mesh" {
                    let grid_size: u32 = 512;
                    6 * (grid_size - 1) * (grid_size - 1)
                } else {
                    3
                };
                pass.draw(0..vertex_count, 0..1);
            }

            drop(light_buffer_guard);
            drop(reflection_depth_guard);
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
                    resource: wgpu::BindingResource::TextureView(&reflection_view_guard),
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
