use super::*;

pub(super) struct AtmosphereInitResources {
    pub(super) sky_bind_group_layout0: wgpu::BindGroupLayout,
    pub(super) sky_bind_group_layout1: wgpu::BindGroupLayout,
    pub(super) sky_pipeline: wgpu::ComputePipeline,
    pub(super) sky_fallback_texture: wgpu::Texture,
    pub(super) sky_fallback_view: wgpu::TextureView,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TerrainSkyUniforms {
    sun_direction_turbidity: [f32; 4],
    ground_albedo_sun_size_sun_intensity_exposure: [f32; 4],
    model_pad: [u32; 4],
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TerrainSkyCameraUniforms {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    inv_view: [[f32; 4]; 4],
    inv_proj: [[f32; 4]; 4],
    eye_position: [f32; 3],
    _pad0: f32,
}

pub(super) fn create_atmosphere_init_resources(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> AtmosphereInitResources {
    let sky_bind_group_layout0 =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain.sky.bgl0"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

    let sky_bind_group_layout1 =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain.sky.bgl1"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

    let sky_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("terrain.sky.shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/sky.wgsl").into()),
    });

    let sky_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("terrain.sky.pipeline_layout"),
        bind_group_layouts: &[&sky_bind_group_layout0, &sky_bind_group_layout1],
        push_constant_ranges: &[],
    });

    let sky_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("terrain.sky.pipeline"),
        layout: Some(&sky_pipeline_layout),
        module: &sky_shader,
        entry_point: "cs_render_sky",
    });

    let sky_fallback_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("terrain.sky.fallback"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &sky_fallback_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &[26, 26, 38, 255],
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
    let sky_fallback_view =
        sky_fallback_texture.create_view(&wgpu::TextureViewDescriptor::default());

    AtmosphereInitResources {
        sky_bind_group_layout0,
        sky_bind_group_layout1,
        sky_pipeline,
        sky_fallback_texture,
        sky_fallback_view,
    }
}

impl TerrainScene {
    pub(super) fn render_sky_texture(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        decoded: &crate::terrain::render_params::DecodedTerrainSettings,
        view_matrix: glam::Mat4,
        proj_matrix: glam::Mat4,
        eye: glam::Vec3,
        width: u32,
        height: u32,
    ) -> Result<Option<(wgpu::Texture, wgpu::TextureView)>> {
        if !decoded.sky.enabled || width == 0 || height == 0 {
            return Ok(None);
        }

        let sky_uniforms = TerrainSkyUniforms {
            // sky.wgsl is authored in a Y-up frame while terrain lighting is Z-up.
            // Swizzle the decoded terrain light so the sky disk still tracks the
            // terrain sun direction on screen.
            sun_direction_turbidity: [
                decoded.light.direction[0],
                decoded.light.direction[2],
                decoded.light.direction[1],
                decoded.sky.turbidity.clamp(1.0, 10.0),
            ],
            ground_albedo_sun_size_sun_intensity_exposure: [
                decoded.sky.ground_albedo.clamp(0.0, 1.0),
                decoded.sky.sun_size.max(0.0),
                decoded.sky.sun_intensity.max(0.0),
                decoded.sky.sky_exposure.max(0.0),
            ],
            model_pad: [1, 0, 0, 0],
        };
        let sky_params = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("terrain.sky.params"),
                contents: bytemuck::bytes_of(&sky_uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let sky_camera_uniforms = TerrainSkyCameraUniforms {
            view: view_matrix.to_cols_array_2d(),
            proj: proj_matrix.to_cols_array_2d(),
            inv_view: view_matrix.inverse().to_cols_array_2d(),
            inv_proj: proj_matrix.inverse().to_cols_array_2d(),
            eye_position: eye.to_array(),
            _pad0: 0.0,
        };
        let sky_camera = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("terrain.sky.camera"),
                contents: bytemuck::bytes_of(&sky_camera_uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let sky_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain.sky.output"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let sky_view = sky_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let sky_bg0 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain.sky.bg0"),
            layout: &self.sky_bind_group_layout0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sky_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&sky_view),
                },
            ],
        });
        let sky_bg1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain.sky.bg1"),
            layout: &self.sky_bind_group_layout1,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: sky_camera.as_entire_binding(),
            }],
        });

        let gx = (width + 7) / 8;
        let gy = (height + 7) / 8;
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("terrain.sky.compute"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.sky_pipeline);
            cpass.set_bind_group(0, &sky_bg0, &[]);
            cpass.set_bind_group(1, &sky_bg1, &[]);
            cpass.dispatch_workgroups(gx, gy, 1);
        }

        Ok(Some((sky_texture, sky_view)))
    }
}
