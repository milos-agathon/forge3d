//! GI composition orchestration for P5.4.
//!
//! This module provides the `GiPass` type that owns compute pipelines for
//! `gi/composite.wgsl` and `gi/debug.wgsl`, orchestrating AO, SSGI, and SSR
//! texture composition into the final lighting buffer.

mod bind_groups;
mod params;

pub use params::{GiCompositeParams, GiCompositeParamsStd140};

use crate::core::error::RenderResult;
use crate::core::gpu_timing::GpuTimingManager;
use wgpu::{
    util::DeviceExt, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindingResource, Buffer,
    BufferUsages, CommandEncoder, ComputePassDescriptor, ComputePipeline,
    ComputePipelineDescriptor, Device, ShaderModuleDescriptor, ShaderSource, TextureView,
};

pub struct GiPass {
    pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
    debug_pipeline: ComputePipeline,
    debug_bind_group_layout: BindGroupLayout,
    params_buffer: Buffer,
    width: u32,
    height: u32,
    params: GiCompositeParams,
    last_composite_ms: f32,
    last_debug_ms: f32,
}

impl GiPass {
    pub fn new(device: &Device, width: u32, height: u32) -> RenderResult<Self> {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("p5.gi.composite"),
            source: ShaderSource::Wgsl(include_str!("../../shaders/gi/composite.wgsl").into()),
        });
        let debug_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("p5.gi.debug"),
            source: ShaderSource::Wgsl(include_str!("../../shaders/gi/debug.wgsl").into()),
        });

        let bind_group_layout = bind_groups::create_composite_bind_group_layout(device);
        let pipeline = create_compute_pipeline(device, &shader, &bind_group_layout, "composite");

        let debug_bind_group_layout = bind_groups::create_debug_bind_group_layout(device);
        let debug_pipeline =
            create_compute_pipeline(device, &debug_shader, &debug_bind_group_layout, "debug");

        let params = GiCompositeParams::default();
        let params_std: GiCompositeParamsStd140 = params.into();
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("p5.gi.composite.params"),
            contents: bytemuck::bytes_of(&params_std),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        Ok(Self {
            pipeline,
            bind_group_layout,
            debug_pipeline,
            debug_bind_group_layout,
            params_buffer,
            width,
            height,
            params,
            last_composite_ms: 0.0,
            last_debug_ms: 0.0,
        })
    }

    pub fn params(&self) -> &GiCompositeParams {
        &self.params
    }

    pub fn composite_ms(&self) -> f32 {
        self.last_composite_ms
    }

    pub fn debug_ms(&self) -> f32 {
        self.last_debug_ms
    }

    pub fn update_params<F: FnOnce(&mut GiCompositeParams)>(&mut self, queue: &wgpu::Queue, f: F) {
        f(&mut self.params);
        let std140: GiCompositeParamsStd140 = self.params.into();
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&std140));
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        baseline_lighting: &TextureView,
        diffuse_view: &TextureView,
        spec_view: &TextureView,
        ao_view: &TextureView,
        ssgi_view: &TextureView,
        ssr_view: &TextureView,
        normal_view: &TextureView,
        material_view: &TextureView,
        output_view: &TextureView,
        mut timing: Option<&mut GpuTimingManager>,
    ) -> RenderResult<()> {
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("p5.gi.composite.bg"),
            layout: &self.bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(baseline_lighting),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(diffuse_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(spec_view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(ao_view),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::TextureView(ssgi_view),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: BindingResource::TextureView(ssr_view),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: BindingResource::TextureView(normal_view),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: BindingResource::TextureView(material_view),
                },
                BindGroupEntry {
                    binding: 8,
                    resource: BindingResource::TextureView(output_view),
                },
                BindGroupEntry {
                    binding: 9,
                    resource: self.params_buffer.as_entire_binding(),
                },
            ],
        });

        let t0 = std::time::Instant::now();
        let timing_scope = timing
            .as_deref_mut()
            .map(|t| t.begin_scope(encoder, "p5.composite"));

        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("p5.gi.composite.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((self.width + 7) / 8, (self.height + 7) / 8, 1);
        drop(pass);

        if let Some(scope_id) = timing_scope {
            if let Some(t) = timing.as_deref_mut() {
                t.end_scope(encoder, scope_id);
            }
        }

        self.last_composite_ms = t0.elapsed().as_secs_f32() * 1000.0;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute_debug(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        ao_view: &TextureView,
        ssgi_view: &TextureView,
        ssr_view: &TextureView,
        debug_output_view: &TextureView,
    ) -> RenderResult<()> {
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("p5.gi.debug.bg"),
            layout: &self.debug_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(ao_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(ssgi_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(ssr_view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(debug_output_view),
                },
            ],
        });

        let t0 = std::time::Instant::now();
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("p5.gi.debug.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.debug_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((self.width + 7) / 8, (self.height + 7) / 8, 1);
        drop(pass);
        self.last_debug_ms = t0.elapsed().as_secs_f32() * 1000.0;
        Ok(())
    }
}

fn create_compute_pipeline(
    device: &Device,
    shader: &wgpu::ShaderModule,
    bind_group_layout: &BindGroupLayout,
    kind: &str,
) -> ComputePipeline {
    let entry_point = match kind {
        "composite" => "cs_gi_composite",
        "debug" => "cs_gi_debug",
        _ => "cs_gi_composite",
    };
    device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some(&format!("p5.gi.{}.pipeline", kind)),
        layout: Some(
            &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("p5.gi.{}.layout", kind)),
                bind_group_layouts: &[bind_group_layout],
                push_constant_ranges: &[],
            }),
        ),
        module: shader,
        entry_point,
    })
}
