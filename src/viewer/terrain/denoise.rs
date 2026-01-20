// src/viewer/terrain/denoise.rs

use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct DenoiseUniforms {
    width: f32,
    height: f32,
    step_width: f32,
    sigma_color: f32,
    sigma_normal: f32,
    sigma_depth: f32,
    padding: [f32; 2],
}

pub struct DenoisePass {
    device: Arc<wgpu::Device>,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    
    // Ping-pong textures
    pub texture_a: Option<wgpu::Texture>,
    pub view_a: Option<wgpu::TextureView>,
    pub texture_b: Option<wgpu::Texture>,
    pub view_b: Option<wgpu::TextureView>,
    
    width: u32,
    height: u32,
}

impl DenoisePass {
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("denoise_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/denoise_atrous.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("denoise_bgl"),
            entries: &[
                // Input texture
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Output storage
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // Depth texture
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("denoise_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("denoise_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        Self {
            device,
            bind_group_layout,
            pipeline,
            texture_a: None,
            view_a: None,
            texture_b: None,
            view_b: None,
            width: 0,
            height: 0,
        }
    }

    pub fn ensure_resources(&mut self, width: u32, height: u32) {
        if self.width == width && self.height == height && self.texture_a.is_some() {
            return;
        }

        self.width = width;
        self.height = height;

        let desc = wgpu::TextureDescriptor {
            label: Some("denoise_pingpong"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        };

        let tex_a = self.device.create_texture(&desc);
        let tex_b = self.device.create_texture(&desc);

        self.view_a = Some(tex_a.create_view(&wgpu::TextureViewDescriptor::default()));
        self.view_b = Some(tex_b.create_view(&wgpu::TextureViewDescriptor::default()));
        
        self.texture_a = Some(tex_a);
        self.texture_b = Some(tex_b);
    }
    
    pub fn get_input_view(&mut self, width: u32, height: u32) -> &wgpu::TextureView {
        self.ensure_resources(width, height);
        self.view_a.as_ref().unwrap()
    }
    
    pub fn apply(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        depth_view: &wgpu::TextureView,
        iterations: u32,
        sigma_color: f32,
    ) {
        if iterations == 0 {
            return;
        }
        
        let width = self.width;
        let height = self.height;
        if width == 0 { return; } 
        
        // Pass 0: Input (view_a) -> B
        let mut source = self.view_a.as_ref().unwrap();
        let mut dest_storage = self.view_b.as_ref().unwrap();
        
        for i in 0..iterations {
            let step_width = 1.0 * (1 << i) as f32;
            
            let uniforms = DenoiseUniforms {
                width: width as f32,
                height: height as f32,
                step_width,
                sigma_color,
                sigma_normal: 0.1,
                sigma_depth: 0.1,
                padding: [0.0; 2],
            };
            
            let temp_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
                label: Some("temp_denoise_uniform"),
                contents: bytemuck::cast_slice(&[uniforms]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("denoise_bg"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(source) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(dest_storage) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(depth_view) },
                    wgpu::BindGroupEntry { binding: 3, resource: temp_buffer.as_entire_binding() },
                ],
            });
            
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("denoise_pass"), timestamp_writes: None });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups((width + 7) / 8, (height + 7) / 8, 1);
            drop(cpass);
            
            // Setup for next pass
            if i % 2 == 0 {
                // Wrote to B. Next read B, write A.
                source = self.view_b.as_ref().unwrap();
                dest_storage = self.view_a.as_ref().unwrap();
            } else {
                // Wrote to A. Next read A, write B.
                source = self.view_a.as_ref().unwrap();
                dest_storage = self.view_b.as_ref().unwrap();
            }
        }
    }
    
    // Helper to get the result view to blit from
    pub fn get_last_result_view(&self, iterations: u32) -> Option<&wgpu::TextureView> {
         if iterations == 0 { return None; }
         if iterations % 2 != 0 {
             // Odd: Last write was to B
             self.view_b.as_ref()
         } else {
             // Even: Last write was to A
             self.view_a.as_ref()
         }
    }
}
