// src/shadows/blur_pass.rs
// P0.2/M3: Separable Gaussian blur pass for VSM/EVSM/MSM moment maps
// Applies two-pass blur (horizontal then vertical) to smooth moment statistics

use bytemuck::{Pod, Zeroable};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferBindingType,
    BufferDescriptor, BufferUsages, ComputePipeline, ComputePipelineDescriptor, Device,
    Extent3d, PipelineLayoutDescriptor, Queue, ShaderModuleDescriptor, ShaderSource,
    ShaderStages, StorageTextureAccess, Texture, TextureDescriptor, TextureDimension,
    TextureFormat, TextureSampleType, TextureUsages, TextureView, TextureViewDescriptor,
    TextureViewDimension,
};

/// Parameters for shadow blur pass
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct BlurParams {
    direction: [f32; 2],    // (1,0) for horizontal, (0,1) for vertical
    kernel_radius: u32,
    cascade_count: u32,
    texture_size: u32,
    _padding: [u32; 3],
}

/// Shadow blur pass for VSM/EVSM/MSM moment maps
pub struct ShadowBlurPass {
    pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
    params_buffer: Buffer,
    // Intermediate texture for two-pass blur
    intermediate_texture: Option<Texture>,
    intermediate_view: Option<TextureView>,
    current_size: u32,
    current_cascades: u32,
}

impl ShadowBlurPass {
    /// Create a new shadow blur pass
    pub fn new(device: &Device) -> Self {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("shadow_blur_shader"),
            source: ShaderSource::Wgsl(include_str!("../shaders/shadow_blur.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("shadow_blur_bind_group_layout"),
            entries: &[
                // Input texture (binding 0)
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                // Output texture (binding 1)
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba16Float,
                        view_dimension: TextureViewDimension::D2Array,
                    },
                    count: None,
                },
                // Parameters uniform (binding 2)
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

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("shadow_blur_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("shadow_blur_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "cs_blur",
        });

        let params_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("shadow_blur_params"),
            size: std::mem::size_of::<BlurParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            bind_group_layout,
            params_buffer,
            intermediate_texture: None,
            intermediate_view: None,
            current_size: 0,
            current_cascades: 0,
        }
    }

    /// Ensure intermediate texture is allocated with correct size
    fn ensure_intermediate_texture(&mut self, device: &Device, size: u32, cascades: u32) {
        if self.current_size == size && self.current_cascades == cascades {
            return;
        }

        let texture = device.create_texture(&TextureDescriptor {
            label: Some("shadow_blur_intermediate"),
            size: Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: cascades,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });

        let view = texture.create_view(&TextureViewDescriptor {
            label: Some("shadow_blur_intermediate_view"),
            format: Some(TextureFormat::Rgba16Float),
            dimension: Some(TextureViewDimension::D2Array),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(cascades),
        });

        self.intermediate_texture = Some(texture);
        self.intermediate_view = Some(view);
        self.current_size = size;
        self.current_cascades = cascades;
    }

    /// Execute two-pass separable Gaussian blur on moment maps
    pub fn execute(
        &mut self,
        device: &Device,
        queue: &Queue,
        encoder: &mut wgpu::CommandEncoder,
        moment_view: &TextureView,
        moment_texture: &Texture,
        cascade_count: u32,
        shadow_map_size: u32,
        kernel_radius: u32,
    ) {
        // Ensure intermediate texture exists
        self.ensure_intermediate_texture(device, shadow_map_size, cascade_count);

        let intermediate_view = self.intermediate_view.as_ref().unwrap();

        // Pass 1: Horizontal blur (moment -> intermediate)
        self.execute_pass(
            device,
            queue,
            encoder,
            moment_view,
            intermediate_view,
            [1.0, 0.0], // Horizontal
            kernel_radius,
            cascade_count,
            shadow_map_size,
            "shadow_blur_horizontal",
        );

        // Create output view for vertical pass
        let output_view = moment_texture.create_view(&TextureViewDescriptor {
            label: Some("shadow_blur_output_view"),
            format: Some(TextureFormat::Rgba16Float),
            dimension: Some(TextureViewDimension::D2Array),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(cascade_count),
        });

        // Pass 2: Vertical blur (intermediate -> moment)
        self.execute_pass(
            device,
            queue,
            encoder,
            intermediate_view,
            &output_view,
            [0.0, 1.0], // Vertical
            kernel_radius,
            cascade_count,
            shadow_map_size,
            "shadow_blur_vertical",
        );
    }

    fn execute_pass(
        &self,
        device: &Device,
        queue: &Queue,
        encoder: &mut wgpu::CommandEncoder,
        input_view: &TextureView,
        output_view: &TextureView,
        direction: [f32; 2],
        kernel_radius: u32,
        cascade_count: u32,
        texture_size: u32,
        label: &str,
    ) {
        // Update parameters
        let params = BlurParams {
            direction,
            kernel_radius,
            cascade_count,
            texture_size,
            _padding: [0; 3],
        };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[params]));

        // Create bind group
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some(label),
            layout: &self.bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(input_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(output_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: self.params_buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch compute shader
        let workgroup_size = 8;
        let dispatch_x = (texture_size + workgroup_size - 1) / workgroup_size;
        let dispatch_y = (texture_size + workgroup_size - 1) / workgroup_size;

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, cascade_count);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blur_params_size() {
        assert_eq!(
            std::mem::size_of::<BlurParams>(),
            32,
            "BlurParams must be 32 bytes (aligned for GPU)"
        );
    }
}
