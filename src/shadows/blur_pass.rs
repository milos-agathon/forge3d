// src/shadows/blur_pass.rs
// P0.2/M3: Separable Gaussian blur pass for VSM/EVSM/MSM moment maps
// Applies two-pass blur (horizontal then vertical) to smooth moment statistics

use crate::core::error::RenderResult;
use crate::core::resource_tracker::{
    tracked_create_buffer, tracked_create_texture, TrackedBuffer, TrackedTexture,
};
use bytemuck::{Pod, Zeroable};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, BufferBindingType, BufferDescriptor,
    BufferUsages, ComputePipeline, ComputePipelineDescriptor, Device, Extent3d,
    PipelineLayoutDescriptor, Queue, ShaderStages, StorageTextureAccess, TextureDescriptor,
    TextureDimension, TextureFormat, TextureSampleType, TextureUsages, TextureView,
    TextureViewDescriptor, TextureViewDimension,
};

pub const DEFAULT_MOMENT_BLUR_RADIUS: u32 = 3;

/// Parameters for shadow blur pass
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct BlurParams {
    direction: [f32; 2], // (1,0) for horizontal, (0,1) for vertical
    kernel_radius: u32,
    cascade_count: u32,
    texture_size: u32,
    _padding: [u32; 7],
}

/// Shadow blur pass for VSM/EVSM/MSM moment maps
pub struct ShadowBlurPass {
    pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
    params_buffers: [TrackedBuffer; 2],
    // Intermediate texture for two-pass blur
    intermediate_texture: Option<TrackedTexture>,
    intermediate_view: Option<TextureView>,
    current_size: u32,
    current_cascades: u32,
}

impl ShadowBlurPass {
    /// Create a new shadow blur pass
    pub fn new(device: &Device) -> RenderResult<Self> {
        let shader = crate::core::shader_registry::create_labeled_shader_module(
            device,
            "shadow_blur_shader",
            include_str!("../shaders/shadow_blur.wgsl"),
        );

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

        let pipeline = crate::core::shader_registry::create_compute_pipeline_scoped(
            device,
            &ComputePipelineDescriptor {
                label: Some("shadow_blur_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "cs_blur",
            },
        );

        let create_params_buffer = |label| {
            tracked_create_buffer(
                device,
                &BufferDescriptor {
                    label: Some(label),
                    size: std::mem::size_of::<BlurParams>() as u64,
                    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                },
            )
        };
        let params_buffers = [
            create_params_buffer("shadow_blur_horizontal_params")?,
            create_params_buffer("shadow_blur_vertical_params")?,
        ];

        Ok(Self {
            pipeline,
            bind_group_layout,
            params_buffers,
            intermediate_texture: None,
            intermediate_view: None,
            current_size: 0,
            current_cascades: 0,
        })
    }

    /// Ensure intermediate texture is allocated with correct size
    fn ensure_intermediate_texture(
        &mut self,
        device: &Device,
        size: u32,
        cascades: u32,
    ) -> RenderResult<()> {
        if self.current_size == size && self.current_cascades == cascades {
            return Ok(());
        }

        let texture = tracked_create_texture(
            device,
            &TextureDescriptor {
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
            },
        )?;

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
        Ok(())
    }

    /// Execute two-pass separable Gaussian blur on moment maps
    pub fn execute(
        &mut self,
        device: &Device,
        queue: &Queue,
        encoder: &mut wgpu::CommandEncoder,
        moment_view: &TextureView,
        moment_texture: &wgpu::Texture,
        cascade_count: u32,
        shadow_map_size: u32,
        kernel_radius: u32,
    ) -> RenderResult<()> {
        // Ensure intermediate texture exists
        self.ensure_intermediate_texture(device, shadow_map_size, cascade_count)?;

        let intermediate_view = self.intermediate_view.as_ref().unwrap();

        // Pass 1: Horizontal blur (moment -> intermediate)
        self.execute_pass(
            device,
            queue,
            encoder,
            moment_view,
            intermediate_view,
            [1.0, 0.0], // Horizontal
            &self.params_buffers[0],
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
            &self.params_buffers[1],
            kernel_radius,
            cascade_count,
            shadow_map_size,
            "shadow_blur_vertical",
        );

        Ok(())
    }

    fn execute_pass(
        &self,
        device: &Device,
        queue: &Queue,
        encoder: &mut wgpu::CommandEncoder,
        input_view: &TextureView,
        output_view: &TextureView,
        direction: [f32; 2],
        params_buffer: &TrackedBuffer,
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
            _padding: [0; 7],
        };
        queue.write_buffer(params_buffer, 0, bytemuck::cast_slice(&[params]));

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
                    resource: params_buffer.as_entire_binding(),
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
            48,
            "BlurParams must match WGSL's 48-byte uniform layout"
        );
    }

    #[test]
    fn intermediate_texture_is_cached_and_resized_with_the_atlas() {
        let context = crate::core::gpu::try_ctx().expect("GPU context");
        let device = &context.device;
        let mut blur = ShadowBlurPass::new(device).expect("blur pass");

        blur.ensure_intermediate_texture(device, 9, 1)
            .expect("initial intermediate");
        let initial_id = blur
            .intermediate_texture
            .as_ref()
            .expect("intermediate texture")
            .ledger_id();

        blur.ensure_intermediate_texture(device, 9, 1)
            .expect("cached intermediate");
        assert_eq!(
            blur.intermediate_texture
                .as_ref()
                .expect("intermediate texture")
                .ledger_id(),
            initial_id,
            "unchanged atlas dimensions must reuse the intermediate texture"
        );

        blur.ensure_intermediate_texture(device, 13, 2)
            .expect("resized intermediate");
        assert_ne!(
            blur.intermediate_texture
                .as_ref()
                .expect("intermediate texture")
                .ledger_id(),
            initial_id,
            "atlas resize must recreate the intermediate texture"
        );
        assert_eq!((blur.current_size, blur.current_cascades), (13, 2));
    }

    #[test]
    fn separable_blur_filters_both_axes() {
        let context = crate::core::gpu::try_ctx().expect("GPU context");
        let device = &context.device;
        let queue = &context.queue;
        let format_features = context
            .adapter
            .get_texture_format_features(TextureFormat::Rgba16Float);
        assert!(
            format_features
                .allowed_usages
                .contains(TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING),
            "adapter cannot sample and storage-write Rgba16Float"
        );

        let size = 9u32;
        let texture = tracked_create_texture(
            device,
            &TextureDescriptor {
                label: Some("shadow_blur_test_moments"),
                size: Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba16Float,
                usage: TextureUsages::TEXTURE_BINDING
                    | TextureUsages::STORAGE_BINDING
                    | TextureUsages::COPY_DST
                    | TextureUsages::COPY_SRC,
                view_formats: &[],
            },
        )
        .expect("moment texture");
        let view = texture.create_view(&TextureViewDescriptor {
            dimension: Some(TextureViewDimension::D2Array),
            array_layer_count: Some(1),
            ..Default::default()
        });

        let mut values = vec![0.0f32; (size * size * 4) as usize];
        let center = ((size / 2 * size + size / 2) * 4) as usize;
        values[center..center + 4].fill(1.0);
        let bytes = values
            .into_iter()
            .flat_map(|value| half::f16::from_f32(value).to_le_bytes())
            .collect::<Vec<_>>();
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &bytes,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(size * 8),
                rows_per_image: Some(size),
            },
            Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
        );

        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut blur = ShadowBlurPass::new(device).expect("blur pass");
        device.poll(wgpu::Maintain::Wait);
        if let Some(error) = pollster::block_on(device.pop_error_scope()) {
            panic!("blur pipeline raised a GPU validation error: {error}");
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("shadow_blur_test"),
        });
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        blur.execute(device, queue, &mut encoder, &view, &texture, 1, size, 2)
            .expect("blur execute");
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);
        if let Some(error) = pollster::block_on(device.pop_error_scope()) {
            panic!("blur dispatch raised a GPU validation error: {error}");
        }

        let output = crate::core::hdr::read_hdr_texture(
            device,
            queue,
            &texture,
            size,
            size,
            TextureFormat::Rgba16Float,
        )
        .expect("read blurred moments");
        let sample = |x: u32, y: u32| output[((y * size + x) * 4) as usize];
        let center_value = sample(size / 2, size / 2);
        let horizontal = sample(size / 2 + 1, size / 2);
        let vertical = sample(size / 2, size / 2 + 1);
        println!(
            "blur impulse center={center_value}, horizontal={horizontal}, vertical={vertical}"
        );
        assert!(
            horizontal > 0.08,
            "horizontal pass was not applied: {horizontal}"
        );
        assert!(vertical > 0.08, "vertical pass was not applied: {vertical}");
        assert!(
            (horizontal - vertical).abs() < 0.01,
            "separable blur is anisotropic: horizontal={horizontal}, vertical={vertical}"
        );
    }
}
