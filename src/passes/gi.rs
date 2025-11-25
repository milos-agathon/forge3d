// src/passes/gi.rs
// P5.4: GI composition orchestration (design-only for Milestone 2).
//
// This file currently serves as documentation for the planned GI composite pass
// and its uniform parameters. No Rust code is implemented yet; wiring of the
// actual pass will happen in later milestones.
//
// High-level role of GiPass (future work)
// --------------------------------------
// A future `GiPass` type in this module will:
//   * Own or reference the existing AO, SSGI, and SSR renderers.
//   * Ensure those passes produce well-defined intermediate textures:
//       - ao_resolved      : R32Float AO scalar in [0,1].
//       - ssgi_resolved    : Rgba16Float diffuse GI radiance (rgb, HDR).
//       - ssr_final        : Rgba16Float specular reflection (rgb, HDR) with
//                            alpha used as a hit mask (a > 0 for surface hits,
//                            a = 0 for env-only).
//       - baseline_lighting: HDR buffer containing L_diffuse_base + L_spec_base
//                            (+ emissive) *before* AO/SSGI/SSR.
//   * Bind these textures, plus the G-buffer and a uniform buffer, into a single
//     compute pipeline using `src/shaders/gi/composite.wgsl`.
//   * Dispatch the composite kernel once per frame to produce the final lighting
//     buffer used by the P5 harness and viewer.
//
// Planned GI composite params uniform
// -----------------------------------
// The composite kernel is controlled by a small uniform buffer that encodes
// toggles, quality knobs, and the global energy cap. On the Rust side this will
// be represented as a struct like:
//
//   pub struct GiCompositeParams {
//       // Feature toggles (exposed to CLI / viewer; mapped to 0/1 on GPU).
//       pub ao_enable: bool;   // AO multiplier on diffuse (Milestone 2: design only).
//       pub ssgi_enable: bool; // SSGI diffuse GI term.
//       pub ssr_enable: bool;  // SSR specular replacement / lerp.
//
//       // Quality knobs / strengths in [0,1].
//       pub ao_weight: f32;    // 0 = no AO effect, 1 = full AO; interpolates in-between.
//       pub ssgi_weight: f32;  // Scales diffuse GI radiance before energy capping.
//       pub ssr_weight: f32;   // Scales SSR blend factor (in addition to Fresnel/roughness).
//
//       // Global energy budget relative to baseline+IBL.
//       // The composite pass enforces:
//       //   luminance(L_final) <= energy_cap * luminance(L_baseline)
//       // for almost all pixels, by scaling only the *excess* SSGI contribution.
//       pub energy_cap: f32;   // Expected default: 1.05 (5% over baseline+IBL).
//   }
//
// GPU-side (WGSL) representation
// ------------------------------
// To be compatible with WGSL std140-like layout rules, the booleans will be
// encoded as `u32` values (0 = false, 1 = true) and the struct will be padded
// to 16-byte alignment. The corresponding WGSL struct will look like:
//
//   struct GiCompositeParams {
//       ao_enable:   u32;  // 0 or 1
//       ssgi_enable: u32;  // 0 or 1
//       ssr_enable:  u32;  // 0 or 1
//       _pad0:       u32;  // padding to 16 bytes
//
//       ao_weight:   f32;  // [0,1]
//       ssgi_weight: f32;  // [0,1]
//       ssr_weight:  f32;  // [0,1]
//       energy_cap:  f32;  // >= 1.0, typically 1.05
//   };
//
// The composite shader (see src/shaders/gi/composite.wgsl) will read this
// uniform and apply the formulas documented there:
//   * AO multiplies diffuse only using (ao_enable, ao_weight).
//   * SSGI adds to diffuse only using (ssgi_enable, ssgi_weight) and scales its
//     extra contribution so that per-pixel luminance stays within `energy_cap`
//     times the baseline+IBL luminance.
//   * SSR replaces/lerps specular only using (ssr_enable, ssr_weight) combined
//     with roughness and Fresnel, and uses the SSR alpha channel to distinguish
//     true surface hits from env-only fallbacks.
//
// Later milestones will turn this design into an actual `GiPass` implementation
// that owns the compute pipeline, uploads `GiCompositeParams` each frame, and
// wires the AO/SSGI/SSR textures into the composite dispatch.
//
// Below is the first concrete `GiPass` implementation. It owns a compute
// pipeline for `gi/composite.wgsl` and a uniform buffer for `GiCompositeParams`.
// High-level orchestration (deciding when to run AO/SSGI/SSR and which textures
// to pass in) will be wired up by the P5 harness and viewer code.

use crate::core::gpu_timing::GpuTimingManager;
use crate::error::RenderResult;
use wgpu::{
    util::DeviceExt,
    BindGroupDescriptor,
    BindGroupEntry,
    BindGroupLayout,
    BindGroupLayoutDescriptor,
    BindGroupLayoutEntry,
    BindingResource,
    BindingType,
    Buffer,
    BufferBindingType,
    BufferUsages,
    CommandEncoder,
    ComputePassDescriptor,
    ComputePipeline,
    ComputePipelineDescriptor,
    Device,
    ShaderModuleDescriptor,
    ShaderSource,
    ShaderStages,
    StorageTextureAccess,
    TextureSampleType,
    TextureFormat,
    TextureView,
    TextureViewDimension,
};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GiCompositeParamsStd140 {
    ao_enable: u32,
    ssgi_enable: u32,
    ssr_enable: u32,
    _pad0: u32,
    ao_weight: f32,
    ssgi_weight: f32,
    ssr_weight: f32,
    energy_cap: f32,
}

/// CPU-side representation of GI composite controls.
#[derive(Clone, Copy)]
pub struct GiCompositeParams {
    pub ao_enable: bool,
    pub ssgi_enable: bool,
    pub ssr_enable: bool,
    pub ao_weight: f32,
    pub ssgi_weight: f32,
    pub ssr_weight: f32,
    pub energy_cap: f32,
}

impl From<GiCompositeParams> for GiCompositeParamsStd140 {
    fn from(p: GiCompositeParams) -> Self {
        Self {
            ao_enable: if p.ao_enable { 1 } else { 0 },
            ssgi_enable: if p.ssgi_enable { 1 } else { 0 },
            ssr_enable: if p.ssr_enable { 1 } else { 0 },
            _pad0: 0,
            ao_weight: p.ao_weight,
            ssgi_weight: p.ssgi_weight,
            ssr_weight: p.ssr_weight,
            energy_cap: p.energy_cap,
        }
    }
}

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
            source: ShaderSource::Wgsl(include_str!("../shaders/gi/composite.wgsl").into()),
        });

        let debug_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("p5.gi.debug"),
            source: ShaderSource::Wgsl(include_str!("../shaders/gi/debug.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("p5.gi.composite.bgl"),
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
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 6,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 7,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 8,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba16Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 9,
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

        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("p5.gi.composite.pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("p5.gi.composite.layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &shader,
            entry_point: "cs_gi_composite",
        });

        let debug_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("p5.gi.debug.bgl"),
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
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba16Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let debug_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("p5.gi.debug.pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("p5.gi.debug.layout"),
                bind_group_layouts: &[&debug_bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &debug_shader,
            entry_point: "cs_gi_debug",
        });

        let params = GiCompositeParams {
            ao_enable: true,
            ssgi_enable: true,
            ssr_enable: true,
            ao_weight: 1.0,
            ssgi_weight: 1.0,
            ssr_weight: 1.0,
            energy_cap: 1.05,
        };
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

        let timing_scope = if let Some(timer) = timing.as_deref_mut() {
            Some(timer.begin_scope(encoder, "p5.composite"))
        } else {
            None
        };

        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("p5.gi.composite.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let gx = (self.width + 7) / 8;
        let gy = (self.height + 7) / 8;
        pass.dispatch_workgroups(gx, gy, 1);
        drop(pass);

        if let Some(scope_id) = timing_scope {
            if let Some(timer) = timing.as_deref_mut() {
                timer.end_scope(encoder, scope_id);
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
        let gx = (self.width + 7) / 8;
        let gy = (self.height + 7) / 8;
        pass.dispatch_workgroups(gx, gy, 1);
        drop(pass);
        self.last_debug_ms = t0.elapsed().as_secs_f32() * 1000.0;
        Ok(())
    }
}
