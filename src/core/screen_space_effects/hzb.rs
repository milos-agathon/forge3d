use super::*;

/// Hierarchical Z-Buffer (min-depth pyramid) for accelerated occlusion queries
pub struct HzbPyramid {
    pub(crate) tex: Texture,
    pub(crate) mip_count: u32,
    width: u32,
    height: u32,
    // Compute pipelines and layouts
    bgl_copy: BindGroupLayout,
    bgl_down: BindGroupLayout,
    pipe_copy: ComputePipeline,
    pipe_down: ComputePipeline,
}

impl HzbPyramid {
    pub(crate) fn new(device: &Device, width: u32, height: u32) -> RenderResult<Self> {
        use crate::core::mipmap::calculate_mip_levels;
        let mip_count = calculate_mip_levels(width, height).max(1);
        // HZB is a float color texture (R32Float) with mip chain
        let tex = device.create_texture(&TextureDescriptor {
            label: Some("p5.hzb.pyramid"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R32Float,
            usage: TextureUsages::TEXTURE_BINDING
                | TextureUsages::STORAGE_BINDING
                | TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("p5.hzb.build.shader"),
            source: ShaderSource::Wgsl(include_str!("../../shaders/hzb_build.wgsl").into()),
        });

        // Group 0: depth copy (depth texture -> r32f storage). We use textureLoad on depth (no sampler).
        let bgl_copy = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("p5.hzb.bgl.copy"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Depth,
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::R32Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        // Group 1: downsample (r32f -> r32f) with reversed_z uniform
        let bgl_down = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("p5.hzb.bgl.down"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::R32Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Use separate pipeline layouts per entry to keep validation simple
        let pl_copy = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("p5.hzb.pl.copy"),
            bind_group_layouts: &[&bgl_copy],
            push_constant_ranges: &[],
        });
        let pl_down = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("p5.hzb.pl.down"),
            bind_group_layouts: &[&bgl_down],
            push_constant_ranges: &[],
        });
        let pipe_copy = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("p5.hzb.pipe.copy"),
            layout: Some(&pl_copy),
            module: &shader,
            entry_point: "cs_copy",
        });
        let pipe_down = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("p5.hzb.pipe.down"),
            layout: Some(&pl_down),
            module: &shader,
            entry_point: "cs_downsample",
        });

        Ok(Self {
            tex,
            mip_count,
            width,
            height,
            bgl_copy,
            bgl_down,
            pipe_copy,
            pipe_down,
        })
    }

    #[allow(dead_code)]
    fn ensure_size(&mut self, device: &Device, width: u32, height: u32) {
        if self.width == width && self.height == height {
            return;
        }
        // Recreate texture with new size
        *self = Self::new(device, width, height).expect("HzbPyramid::new should not fail");
    }

    /// Build HZB from a source DEPTH view (mip 0), limiting to `levels` mips (including level 0).
    /// Produces a pyramid in `self.tex` up to the requested number of levels.
    pub fn build_n(
        &self,
        device: &Device,
        encoder: &mut CommandEncoder,
        src_depth: &TextureView,
        levels: u32,
        reversed_z: bool,
    ) {
        // Copy depth -> HZB level 0
        let dst0 = self.tex.create_view(&TextureViewDescriptor {
            label: Some("p5.hzb.mip0"),
            format: None,
            dimension: Some(TextureViewDimension::D2),
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
        });
        let bg_copy = device.create_bind_group(&BindGroupDescriptor {
            label: Some("p5.hzb.bg.copy"),
            layout: &self.bgl_copy,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(src_depth),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&dst0),
                },
            ],
        });
        let mut pass0 = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("p5.hzb.pass.copy"),
            timestamp_writes: None,
        });
        pass0.set_pipeline(&self.pipe_copy);
        pass0.set_bind_group(0, &bg_copy, &[]);
        let gx0 = (self.width + 7) / 8;
        let gy0 = (self.height + 7) / 8;
        pass0.dispatch_workgroups(gx0, gy0, 1);
        drop(pass0);

        // Downsample chain up to requested levels
        let build_to = levels.min(self.mip_count).saturating_sub(1);
        let mut level_w = self.width;
        let mut level_h = self.height;
        // Create uniform buffer for reversed_z flag
        let reversed_z_val: u32 = if reversed_z { 1 } else { 0 };
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("p5.hzb.params"),
            contents: bytemuck::cast_slice(&[reversed_z_val]),
            usage: BufferUsages::UNIFORM,
        });
        for level in 1..=build_to {
            let src_view = self.tex.create_view(&TextureViewDescriptor {
                label: Some("p5.hzb.src.prev"),
                format: None,
                dimension: Some(TextureViewDimension::D2),
                aspect: TextureAspect::All,
                base_mip_level: level - 1,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: Some(1),
            });
            let dst_view = self.tex.create_view(&TextureViewDescriptor {
                label: Some("p5.hzb.dst.curr"),
                format: None,
                dimension: Some(TextureViewDimension::D2),
                aspect: TextureAspect::All,
                base_mip_level: level,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: Some(1),
            });
            let bg_down = device.create_bind_group(&BindGroupDescriptor {
                label: Some("p5.hzb.bg.down"),
                layout: &self.bgl_down,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(&src_view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(&dst_view),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("p5.hzb.pass.down"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipe_down);
            pass.set_bind_group(0, &bg_down, &[]);
            level_w = (level_w / 2).max(1);
            level_h = (level_h / 2).max(1);
            let gx = (level_w + 7) / 8;
            let gy = (level_h + 7) / 8;
            pass.dispatch_workgroups(gx, gy, 1);
            drop(pass);
        }
    }

    /// Build HZB from a source DEPTH view (mip 0). Produces a full pyramid in self.tex
    pub(crate) fn build(
        &self,
        device: &Device,
        encoder: &mut CommandEncoder,
        src_depth: &TextureView,
        reversed_z: bool,
    ) {
        self.build_n(device, encoder, src_depth, self.mip_count, reversed_z);
    }

    #[allow(dead_code)]
    pub(crate) fn texture_view(&self) -> TextureView {
        self.tex.create_view(&TextureViewDescriptor::default())
    }
}
