// src/shadows/moment_pass.rs
// Moment generation pass for VSM/EVSM/MSM shadow techniques
// Converts depth maps into moment statistics via compute shader

use bytemuck::{Pod, Zeroable};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferBindingType,
    BufferDescriptor, BufferUsages, ComputePipeline, ComputePipelineDescriptor, Device,
    PipelineLayoutDescriptor, Queue, ShaderModuleDescriptor, ShaderSource, ShaderStages,
    StorageTextureAccess, Texture, TextureFormat, TextureSampleType, TextureView,
    TextureViewDimension,
};

use crate::lighting::types::ShadowTechnique;

/// Parameters for moment generation
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct MomentGenParams {
    technique: u32,
    cascade_count: u32,
    evsm_positive_exp: f32,
    evsm_negative_exp: f32,
    shadow_map_size: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
    // vec3<u32> in WGSL requires 16-byte alignment
    _padding3: [u32; 3],
    _padding4: u32,
}

/// Moment generation compute pass
pub struct MomentGenerationPass {
    pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
    params_buffer: Buffer,
    bind_group: Option<BindGroup>,
}

impl MomentGenerationPass {
    /// Create a new moment generation pass
    pub fn new(device: &Device) -> Self {
        // Load shader
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("moment_generation_shader"),
            source: ShaderSource::Wgsl(include_str!("../shaders/moment_generation.wgsl").into()),
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("moment_gen_bind_group_layout"),
            entries: &[
                // Depth texture input (binding 0)
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Depth,
                        view_dimension: TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                // Moment texture output (binding 1)
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

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("moment_gen_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create compute pipeline
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("moment_gen_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        // Create params buffer
        let params_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("moment_gen_params"),
            size: std::mem::size_of::<MomentGenParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            bind_group_layout,
            params_buffer,
            bind_group: None,
        }
    }

    /// Prepare bind group for rendering
    pub fn prepare_bind_group(
        &mut self,
        device: &Device,
        depth_view: &TextureView,
        moment_view: &TextureView,
    ) {
        self.bind_group = Some(device.create_bind_group(&BindGroupDescriptor {
            label: Some("moment_gen_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(depth_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(moment_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: self.params_buffer.as_entire_binding(),
                },
            ],
        }));
    }

    /// Update parameters and execute the compute pass
    pub fn execute(
        &self,
        queue: &Queue,
        encoder: &mut wgpu::CommandEncoder,
        technique: ShadowTechnique,
        cascade_count: u32,
        shadow_map_size: u32,
        evsm_positive_exp: f32,
        evsm_negative_exp: f32,
    ) {
        // Update parameters
        let params = MomentGenParams {
            technique: technique.as_u32(),
            cascade_count,
            evsm_positive_exp,
            evsm_negative_exp,
            shadow_map_size,
            _padding0: 0,
            _padding1: 0,
            _padding2: 0,
            _padding3: [0; 3],
            _padding4: 0,
        };

        queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[params]));

        // Execute compute pass
        let bind_group = self
            .bind_group
            .as_ref()
            .expect("Bind group must be prepared before execution");

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("moment_generation_pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, bind_group, &[]);

        // Dispatch compute shader (8x8 workgroup size)
        let workgroup_size = 8;
        let dispatch_x = (shadow_map_size + workgroup_size - 1) / workgroup_size;
        let dispatch_y = (shadow_map_size + workgroup_size - 1) / workgroup_size;
        let dispatch_z = cascade_count;

        compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, dispatch_z);
    }
}

/// Helper to create a storage texture view for moment generation output
pub fn create_moment_storage_view(moment_texture: &Texture, cascade_count: u32) -> TextureView {
    moment_texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("moment_storage_view"),
        format: Some(TextureFormat::Rgba16Float),
        dimension: Some(TextureViewDimension::D2Array),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: Some(1),
        base_array_layer: 0,
        array_layer_count: Some(cascade_count),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moment_pass_creation() {
        let device = crate::gpu::create_device_for_test();
        let _pass = MomentGenerationPass::new(&device);
        // Just verify it constructs without panicking
    }

    #[test]
    fn test_moment_params_size() {
        // Verify struct is properly aligned for GPU
        // WGSL vec3<u32> requires 16-byte alignment, making total size 48 bytes
        assert_eq!(
            std::mem::size_of::<MomentGenParams>(),
            48,
            "MomentGenParams must be 48 bytes (aligned for WGSL vec3)"
        );
    }
}
