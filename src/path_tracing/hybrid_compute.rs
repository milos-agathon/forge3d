// src/path_tracing/hybrid_compute.rs
// Hybrid path tracer combining SDF raymarching with mesh BVH traversal
// Extends the existing path tracing compute pipeline with hybrid scene support

use std::path::PathBuf;
use std::collections::HashSet;

use bytemuck::{Pod, Zeroable};
use half::f16;
use wgpu::util::DeviceExt;

use crate::error::RenderError;
use crate::gpu::{align_copy_bpr, ctx};
use crate::formats::hdr::load_hdr;
use crate::io::tex_upload::create_texture_2d;
use crate::path_tracing::aov::{AovFrames, AovKind};
use crate::path_tracing::compute::{Sphere, Uniforms, TileParams};
use crate::path_tracing::tile_dispatch::TileIterator;
use crate::path_tracing::memory_governor::{MemoryGovernor, MemoryBudget};
use crate::path_tracing::oidn_runner::{OidnMode, TiledOidnConfig, denoise_tiled};
use crate::sdf::HybridScene;
use std::borrow::Cow;

/// Load the hybrid kernel WGSL and expand simple `#include "..."` directives.
///
/// WGSL has no preprocessor, so we inline our shader snippets. This function
/// repeatedly expands known includes and removes duplicate include directives to
/// avoid parser errors and symbol redefinitions.
fn load_hybrid_kernel_src() -> String {
    // Deterministic assembly to avoid duplicate definitions
    // 1) Core SDF primitives (types and basic evaluators)
    let sdf_primitives = include_str!("../shaders/sdf_primitives.wgsl");

    // 2) CSG operations (strip its own includes)
    let sdf_operations_raw = include_str!("../shaders/sdf_operations.wgsl");
    let sdf_operations = sdf_operations_raw
        .lines()
        .filter(|l| !l.trim_start().starts_with("#include"))
        .collect::<Vec<_>>()
        .join("\n");

    // 3) Sampling utilities (Sobol and CMJ)
    let sobol = include_str!("../shaders/sampling/sobol.wgsl");
    let cmj = include_str!("../shaders/sampling/cmj.wgsl");

    // 3b) HDRI sampling utilities
    let hdri_common = include_str!("../../shaders/common/hdri.wgsl");

    // 4) Hybrid traversal utilities (strip includes)
    let hybrid_traversal_raw = include_str!("../shaders/hybrid_traversal.wgsl");
    let hybrid_traversal = hybrid_traversal_raw
        .lines()
        .filter(|l| !l.trim_start().starts_with("#include"))
        .collect::<Vec<_>>()
        .join("\n");

    // 5) Kernel entry (strip includes)
    let kernel_raw = include_str!("../shaders/hybrid_kernel.wgsl");
    let kernel = kernel_raw
        .lines()
        .filter(|l| !l.trim_start().starts_with("#include"))
        .collect::<Vec<_>>()
        .join("\n");

    // Concatenate in dependency order
    [
        sdf_primitives,
        &sdf_operations,
        sobol,
        cmj,
        hdri_common,
        &hybrid_traversal,
        &kernel,
    ]
    .join("\n")
}

/// Additional uniforms for hybrid traversal
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct HybridUniforms {
    pub sdf_primitive_count: u32,
    pub sdf_node_count: u32,
    pub mesh_vertex_count: u32,
    pub mesh_index_count: u32,
    pub mesh_bvh_node_count: u32,
    pub mesh_bvh_root_index: u32,
    pub traversal_mode: u32, // 0 = hybrid, 1 = SDF only, 2 = mesh only
    pub _pad: [u32; 1],
}

/// Traversal mode for hybrid rendering
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TraversalMode {
    Hybrid = 0,
    SdfOnly = 1,
    MeshOnly = 2,
}

impl Default for TraversalMode {
    fn default() -> Self {
        Self::Hybrid
    }
}

/// Hybrid path tracer parameters
#[derive(Clone, Debug)]
pub struct HybridTracerParams {
    /// Base path tracer uniforms
    pub base_uniforms: Uniforms,
    /// Traversal mode
    pub traversal_mode: TraversalMode,
    /// Early exit distance for optimization
    pub early_exit_distance: f32,
    /// Shadow softness factor
    pub shadow_softness: f32,
    /// Force true resolution rendering (no downscaling)
    pub rt_true_res: bool,
    /// Allow upscaling fallback for legacy compatibility
    pub rt_allow_upscale: bool,
    /// Tile size for large resolution renders
    pub tile_size: (u32, u32),
    /// Memory budget for auto-tuning tile size
    pub memory_budget: Option<MemoryBudget>,
    /// OIDN execution mode
    pub oidn_mode: OidnMode,
    /// Total triangle count for geometry budget estimation
    pub total_triangles: usize,
    /// Enable progress reporting
    pub show_progress: bool,
    /// Samples per pixel (total)
    pub rt_spp: u32,
    /// Samples per batch (for progressive rendering)
    pub rt_batch_spp: u32,
    /// Debug mode: 0=normal, 1=colored grid diagnostic, 2=green sentinel
    pub debug_mode: u32,
    /// Optional HDRI path for environment lighting
    pub hdri_path: Option<PathBuf>,
}

impl Default for HybridTracerParams {
    fn default() -> Self {
        Self {
            base_uniforms: Uniforms {
                width: 512,
                height: 512,
                frame_index: 0,
                aov_flags: 0,
                cam_origin: [0.0, 0.0, 0.0],
                cam_fov_y: std::f32::consts::PI / 4.0,
                cam_right: [1.0, 0.0, 0.0],
                cam_aspect: 1.0,
                cam_up: [0.0, 1.0, 0.0],
                cam_exposure: 1.0,
                cam_forward: [0.0, 0.0, -1.0],
                seed_hi: 12345,
                seed_lo: 0,
                sampling_mode: 0,
                lighting_type: 0,
                lighting_intensity: 1.0,
                lighting_azimuth: 5.497787,
                lighting_elevation: 0.785398,
                shadow_intensity: 1.0,
                hdri_intensity: 0.0,
                hdri_rotation: 0.0,
                hdri_exposure: 1.0,
                hdri_enabled: 0.0,
                _pad_end: [0, 0, 0],
            },
            traversal_mode: TraversalMode::Hybrid,
            early_exit_distance: 0.01,
            shadow_softness: 4.0,
            rt_true_res: true,  // Default: enforce true resolution
            rt_allow_upscale: false,  // Default: no upscaling fallback
            tile_size: (512, 512),  // Default tile size for large resolutions
            memory_budget: None,  // Default: no memory governor
            oidn_mode: OidnMode::Final,  // Default: denoise entire image at once
            total_triangles: 1_000_000,  // Default triangle count estimate
            show_progress: true,  // Default: show progress
            rt_spp: 64,  // Default: 64 samples per pixel
            rt_batch_spp: 8,  // Default: 8 samples per batch
            debug_mode: 0,  // Default: normal rendering
            hdri_path: None,
        }
    }
}

/// Hybrid path tracer implementation
pub struct HybridPathTracer {
    /// Shader module for hybrid traversal
    #[allow(dead_code)]
    shader: wgpu::ShaderModule,
    /// Bind group layouts
    layouts: HybridBindGroupLayouts,
    /// Pipeline layout
    #[allow(dead_code)]
    pipeline_layout: wgpu::PipelineLayout,
    /// Compute pipeline
    pipeline: wgpu::ComputePipeline,
}

/// Bind group layouts for hybrid path tracer
struct HybridBindGroupLayouts {
    uniforms: wgpu::BindGroupLayout, // Group 0: camera uniforms
    scene: wgpu::BindGroupLayout,    // Group 1: spheres + legacy mesh buffers
    accum: wgpu::BindGroupLayout,    // Group 2: accumulation buffer
    output: wgpu::BindGroupLayout,   // Group 3: primary output texture + AOVs
}

impl HybridPathTracer {
    /// Create a new hybrid path tracer
    pub fn new() -> Result<Self, RenderError> {
        let device = &ctx().device;

        // Load hybrid kernel shader (expand simple includes)
        let shader_src = load_hybrid_kernel_src();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hybrid-pt-kernel"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        // Create bind group layouts
        let layouts = HybridBindGroupLayouts {
            uniforms: Self::create_uniforms_layout(device),
            scene: Self::create_scene_layout(device),
            accum: Self::create_accum_layout(device),
            output: Self::create_output_layout(device),
        };

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hybrid-pt-pipeline-layout"),
            bind_group_layouts: &[
                &layouts.uniforms,
                &layouts.scene,
                &layouts.accum,
                &layouts.output,
            ],
            push_constant_ranges: &[],
        });

        // Create compute pipeline
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("hybrid-pt-compute"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        Ok(Self {
            shader,
            layouts,
            pipeline_layout,
            pipeline,
        })
    }

    /// Render using hybrid traversal with TRUE-RES tiled rendering for large resolutions
    pub fn render(
        &self,
        width: u32,
        height: u32,
        spheres: &[Sphere],
        hybrid_scene: &HybridScene,
        params: HybridTracerParams,
    ) -> Result<Vec<u8>, RenderError> {
        eprintln!("[HYBRID-RENDER] Starting render {}x{}, mesh_buffers={}", 
            width, height, hybrid_scene.mesh_buffers.is_some());
        
        // TRUE-RES MODE: Enforce no internal downscaling unless explicitly allowed
        if params.rt_true_res {
            eprintln!("[TRUE-RES MODE] Rendering at native {}x{} (no downscaling)", width, height);
            
            // Assert that uniforms match requested resolution
            if params.base_uniforms.width != width || params.base_uniforms.height != height {
                return Err(RenderError::Render(format!(
                    "TRUE-RES MODE violation: uniforms {}x{} != requested {}x{}",
                    params.base_uniforms.width, params.base_uniforms.height, width, height
                )));
            }
            
            // Use tiled rendering for large resolutions
            let tile_threshold = 640;
            if width > tile_threshold || height > tile_threshold {
                eprintln!("[TRUE-RES MODE] Using tiled rendering with {}x{} tiles",
                    params.tile_size.0, params.tile_size.1);
                return self.render_tiled_true_res(width, height, spheres, hybrid_scene, params);
            }
        } else if !params.rt_allow_upscale {
            return Err(RenderError::Render(
                "Downscaling disabled. Enable with rt_allow_upscale=true or use rt_true_res=true".into()
            ));
        } else {
            // Legacy upscaling path (opt-in only)
            eprintln!("[LEGACY MODE] Using downscaled rendering (rt_allow_upscale=true)");
            return self.render_downscaled(width, height, spheres, hybrid_scene, params);
        }
        
        // Direct render for small resolutions
        self.render_single_tile(width, height, spheres, hybrid_scene, params, None)
    }
    
    /// Render a single tile (full image or tile from tiling path)
    fn render_single_tile(
        &self,
        width: u32,
        height: u32,
        spheres: &[Sphere],
        hybrid_scene: &HybridScene,
        params: HybridTracerParams,
        tile_params_override: Option<&TileParams>,  // If Some, use this instead of creating new ones
    ) -> Result<Vec<u8>, RenderError> {
        let device = &ctx().device;
        let queue = &ctx().queue;

        // Prepare hybrid uniforms
        // Prefer the uploaded mesh buffer metadata if available; otherwise fall back to BVH handle
        let bvh_node_count = if let Some(mesh) = &hybrid_scene.mesh_buffers {
            mesh.bvh_node_count
        } else {
            hybrid_scene
                .bvh
                .as_ref()
                .map_or(0, |b| match &b.backend {
                    crate::accel::BvhBackend::Cpu(cpu) => cpu.nodes.len() as u32,
                    crate::accel::BvhBackend::Gpu(gpu) => gpu.node_count,
                })
        };
        
        let (mesh_vtx_count, mesh_idx_count) = if let Some(mesh) = &hybrid_scene.mesh_buffers {
            (mesh.vertex_count, mesh.index_count)
        } else {
            (hybrid_scene.vertices.len() as u32, hybrid_scene.indices.len() as u32)
        };

        // Determine BVH root index: CPU SAH builder appends root at the end; GPU/LBVH usually uses 0
        let bvh_root_index: u32 = if let Some(bvh) = &hybrid_scene.bvh {
            match &bvh.backend {
                crate::accel::BvhBackend::Cpu(cpu) => cpu.nodes.len().saturating_sub(1) as u32,
                crate::accel::BvhBackend::Gpu(_gpu) => 0u32,
            }
        } else {
            0u32
        };

        let hybrid_uniforms = HybridUniforms {
            sdf_primitive_count: hybrid_scene.sdf_scene.primitive_count() as u32,
            sdf_node_count: hybrid_scene.sdf_scene.node_count() as u32,
            mesh_vertex_count: mesh_vtx_count,
            mesh_index_count: mesh_idx_count,
            mesh_bvh_node_count: bvh_node_count,
            mesh_bvh_root_index: bvh_root_index,
            traversal_mode: params.traversal_mode as u32,
            _pad: [0],
        };
        
        eprintln!("[PT-GPU] Hybrid uniforms: verts={}, indices={}, bvh_nodes={}, root={}, mode={}",
            hybrid_uniforms.mesh_vertex_count,
            hybrid_uniforms.mesh_index_count,
            hybrid_uniforms.mesh_bvh_node_count,
            hybrid_uniforms.mesh_bvh_root_index,
            hybrid_uniforms.traversal_mode);

        // Prepare base uniforms (write to buffer below after potential HDRI toggles)
        let mut base_uniforms = params.base_uniforms;

        let hybrid_ubo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hybrid-pt-hybrid-ubo"),
            contents: bytemuck::bytes_of(&hybrid_uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // -------------------------------------------------------------------------------------
        // HDRI environment setup (texture, sampler, enable flag)
        // -------------------------------------------------------------------------------------
        let hdri_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("hybrid-pt-hdri-sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let (_hdri_texture, hdri_view) = if base_uniforms.hdri_enabled > 0.5 {
            match Self::load_hdri_texture(device, queue, &params) {
                Ok(pair) => pair,
                Err(e) => {
                    eprintln!("[PT-GPU] HDRI load failed: {}", e);
                    base_uniforms.hdri_enabled = 0.0;
                    Self::fallback_hdri_texture(device, queue)
                }
            }
        } else {
            Self::fallback_hdri_texture(device, queue)
        };

        // Create buffers for base uniforms (after potential HDRI adjustments)
        let base_ubo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hybrid-pt-base-ubo"),
            contents: bytemuck::bytes_of(&base_uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Scene buffer (spheres)
        let scene_bytes: Cow<[u8]> = if spheres.is_empty() {
            Cow::Owned(vec![0u8; std::mem::size_of::<Sphere>()])
        } else {
            Cow::Borrowed(bytemuck::cast_slice(spheres))
        };
        let scene_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hybrid-pt-scene"),
            contents: scene_bytes.as_ref(),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // TileParams buffer (for all renders, tiled or not)
        let tile_params_data = if let Some(override_params) = tile_params_override {
            // Use provided TileParams (from tiled render path)
            eprintln!("[TileParams] Using override: origin=({},{}) size={}x{} img={}x{} spp_batch={} debug_mode={}",
                override_params.tile_origin_size[0], override_params.tile_origin_size[1],
                override_params.tile_origin_size[2], override_params.tile_origin_size[3],
                override_params.img_spp_counts[0], override_params.img_spp_counts[1],
                override_params.img_spp_counts[2], override_params.seeds_misc[1]);
            *override_params
        } else {
            // Create default TileParams for full image
            let spp_batch = params.rt_batch_spp.max(1);
            eprintln!("[TileParams] Single-tile: origin=(0,0) size={}x{} img={}x{} spp_batch={} debug_mode={} seed={}",
                width, height, width, height, spp_batch, params.debug_mode, params.base_uniforms.seed_hi);
            TileParams {
                tile_origin_size: [0, 0, width, height],
                img_spp_counts: [width, height, spp_batch, 0],
                seeds_misc: [params.base_uniforms.seed_hi, params.debug_mode, 0, 0],
                _pad0: [0; 32],
                _pad1: [0; 20],
            }
        };
        let tile_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("TileParams"),
            contents: bytemuck::bytes_of(&tile_params_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Accumulation buffer
        let accum_size = (width as u64) * (height as u64) * 16; // vec4<f32>
        let accum_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hybrid-pt-accum"),
            size: accum_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Output texture
        let out_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("hybrid-pt-out-tex"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let out_view = out_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // AOV textures
        let aovs_all = [
            AovKind::Albedo,
            AovKind::Normal,
            AovKind::Depth,
            AovKind::Direct,
            AovKind::Indirect,
            AovKind::Emission,
            AovKind::Visibility,
        ];
        let aov_frames = AovFrames::new(device, width, height, &aovs_all);
        let aov_views: Vec<wgpu::TextureView> = aovs_all
            .iter()
            .map(|k| {
                aov_frames
                    .get_texture(*k)
                    .unwrap()
                    .create_view(&wgpu::TextureViewDescriptor::default())
            })
            .collect();

        // Create bind groups
        let bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hybrid-pt-bg0"),
            layout: &self.layouts.uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: base_ubo.as_entire_binding(),
            }],
        });

        // Scene + Hybrid bind group (Group 1)
        let mut bg1_entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: scene_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: hybrid_ubo.as_entire_binding(),
            },
        ];

        // Add mesh buffer entries at bindings 2..4
        let mesh_entries = hybrid_scene.get_mesh_bind_entries();
        eprintln!("[PT-GPU] Binding {} mesh buffer entries", mesh_entries.len());
        bg1_entries.extend(
            mesh_entries
                .into_iter()
                .enumerate()
                .map(|(i, mut entry)| {
                    entry.binding = (i + 2) as u32;
                    eprintln!("[PT-GPU] Binding mesh buffer {} to slot {}", i, entry.binding);
                    entry
                }),
        );

        // HDRI bindings (texture and sampler) occupy slots 5 and 6 respectively
        bg1_entries.push(wgpu::BindGroupEntry {
            binding: 5,
            resource: wgpu::BindingResource::TextureView(&hdri_view),
        });
        bg1_entries.push(wgpu::BindGroupEntry {
            binding: 6,
            resource: wgpu::BindingResource::Sampler(&hdri_sampler),
        });

        let bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hybrid-pt-bg1"),
            layout: &self.layouts.scene,
            entries: &bg1_entries,
        });

        // Group 2: TileParams + accum buffer
        let bg2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hybrid-pt-bg2-tile-params"),
            layout: &self.layouts.accum,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tile_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: accum_buf.as_entire_binding(),
                },
            ],
        });

        // Output + AOVs bind group (Group 3)
        let mut bg3_entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&out_view),
        }];
        // Map AOVs to bindings 1..7 in the same group
        for (i, view) in aov_views.iter().enumerate() {
            bg3_entries.push(wgpu::BindGroupEntry {
                binding: (i as u32) + 1,
                resource: wgpu::BindingResource::TextureView(view),
            });
        }
        let bg3 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hybrid-pt-bg3"),
            layout: &self.layouts.output,
            entries: &bg3_entries,
        });

        // Removed separate AOV (bg4) and Hybrid (bg5) groups; merged into bg3 and bg1 respectively.

        // Dispatch compute shader
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hybrid-pt-encoder"),
        });

        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hybrid-pt-cpass"),
                ..Default::default()
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bg0, &[]);
            cpass.set_bind_group(1, &bg1, &[]);
            cpass.set_bind_group(2, &bg2, &[]);
            cpass.set_bind_group(3, &bg3, &[]);

            let gx = (width + 7) / 8;
            let gy = (height + 7) / 8;
            cpass.dispatch_workgroups(gx, gy, 1);
        }

        // Copy texture to buffer for readback
        let row_bytes = width * 8; // rgba16f = 8 bytes per pixel
        let padded_bpr = align_copy_bpr(row_bytes);
        let read_size = (padded_bpr as u64) * (height as u64);
        let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hybrid-pt-read"),
            size: read_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        enc.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &out_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &read_buf,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bpr),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        queue.submit([enc.finish()]);
        device.poll(wgpu::Maintain::Wait);

        // Read back and convert to RGBA8
        let slice = read_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|_| RenderError::Readback("map_async channel closed".into()))?
            .map_err(|e| RenderError::Readback(format!("MapAsync failed: {:?}", e)))?;

        let data = slice.get_mapped_range();
        let mut out = vec![0u8; (width as usize) * (height as usize) * 4];
        let src_stride = padded_bpr as usize;
        let dst_stride = (width as usize) * 4;

        for y in 0..(height as usize) {
            let row = &data[y * src_stride..y * src_stride + (width as usize) * 8];
            for x in 0..(width as usize) {
                let o = x * 8;
                let r = f16::from_bits(u16::from_le_bytes([row[o + 0], row[o + 1]])).to_f32();
                let g = f16::from_bits(u16::from_le_bytes([row[o + 2], row[o + 3]])).to_f32();
                let b = f16::from_bits(u16::from_le_bytes([row[o + 4], row[o + 5]])).to_f32();

                let ix = y * dst_stride + x * 4;
                out[ix + 0] = (r.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                out[ix + 1] = (g.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                out[ix + 2] = (b.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                out[ix + 3] = 255u8;
            }
        }

        drop(data);
        read_buf.unmap();
        
        // Check if output is valid (not all same color)
        let first_pixel = (out[0] as u32) | ((out[1] as u32) << 8) | ((out[2] as u32) << 16);
        let all_same = out.chunks(4).all(|p| {
            let px = (p[0] as u32) | ((p[1] as u32) << 8) | ((p[2] as u32) << 16);
            px == first_pixel
        });
        
        if all_same {
            eprintln!("[HYBRID-RENDER] WARNING: Output is solid color (likely render failed)");
        } else {
            eprintln!("[HYBRID-RENDER] Render succeeded with varied colors");
        }
        
        Ok(out)
    }
    
    /// Render large images using true-resolution tiling (no downscaling)
    /// Integrates memory governor, buffer pooling, progress reporting, and OOM recovery
    fn render_tiled_true_res(
        &self,
        width: u32,
        height: u32,
        spheres: &[Sphere],
        hybrid_scene: &HybridScene,
        params: HybridTracerParams,
    ) -> Result<Vec<u8>, RenderError> {
        // Use memory governor to compute optimal tile size if budget is provided
        let (tile_w, tile_h) = if let Some(ref budget) = params.memory_budget {
            let governor = MemoryGovernor::new(budget.clone());
            let enable_oidn = params.oidn_mode != OidnMode::Off;
            
            match governor.compute_tile_size(width, height, params.total_triangles, enable_oidn) {
                Ok((tw, th, footprint)) => {
                    eprintln!("[MemoryGovernor] Computed optimal tile size: {}x{}", tw, th);
                    footprint.print_table();
                    (tw, th)
                }
                Err(e) => {
                    return Err(RenderError::Render(format!("Memory governor failed: {}", e)));
                }
            }
        } else {
            params.tile_size
        };
        
        let mut current_tile_size = (tile_w, tile_h);
        let mut retry_count = 0;
        const MAX_RETRIES: usize = 2;
        
        // OOM recovery loop
        loop {
            match self.render_tiled_inner(
                width, height, spheres, hybrid_scene, &params, current_tile_size
            ) {
                Ok(output) => return Ok(output),
                Err(e) => {
                    let err_str = format!("{:?}", e);
                    if err_str.contains("OutOfMemory") || err_str.contains("OOM") {
                        retry_count += 1;
                        if retry_count > MAX_RETRIES {
                            return Err(RenderError::Render(
                                format!("OOM persists after {} retries with tile reduction", MAX_RETRIES)
                            ));
                        }
                        
                        // Halve tile size and retry
                        current_tile_size.0 = (current_tile_size.0 / 2).max(96);
                        current_tile_size.1 = (current_tile_size.1 / 2).max(96);
                        eprintln!("[OOM Recovery] Reducing tile size to {}x{} (retry {}/{})",
                            current_tile_size.0, current_tile_size.1, retry_count, MAX_RETRIES);
                    } else {
                        return Err(e);
                    }
                }
            }
        }
    }
    
    /// Inner rendering function with fixed tile size
    fn render_tiled_inner(
        &self,
        width: u32,
        height: u32,
        spheres: &[Sphere],
        hybrid_scene: &HybridScene,
        params: &HybridTracerParams,
        tile_size: (u32, u32),
    ) -> Result<Vec<u8>, RenderError> {
        let (tile_w, tile_h) = tile_size;
        let tiles: Vec<_> = TileIterator::new(width, height, tile_w, tile_h).collect();
        let total_tiles = tiles.len();
        
        eprintln!("[TRUE-RES TILED] Rendering {}x{} as {} tiles ({}x{} each)",
            width, height, total_tiles, tile_w, tile_h);
        
        // Allocate final output buffer
        let mut output = vec![0u8; (width as usize) * (height as usize) * 4];
        
        // Track timing for ETA
        let start_time = std::time::Instant::now();
        
        for (tile_idx, tile) in tiles.iter().enumerate() {
            let tile_start = std::time::Instant::now();
            
            if params.show_progress {
                // Estimate ETA
                let progress = (tile_idx as f64) / (total_tiles as f64);
                let elapsed = start_time.elapsed().as_secs_f64();
                let eta_secs = if tile_idx > 0 {
                    elapsed / progress * (1.0 - progress)
                } else {
                    0.0
                };
                
                eprintln!("[TRUE-RES TILED] Tile {}/{} ({:.1}%): origin=({},{}) size={}x{} | ETA: {:.0}s",
                    tile_idx + 1, total_tiles, progress * 100.0,
                    tile.x, tile.y, tile.width, tile.height, eta_secs);
            }
            
            // Create TileParams for this tile
            let spp_batch = params.rt_batch_spp.max(1); // Safety: ensure at least 1
            assert!(spp_batch > 0, "[CRITICAL] spp_batch must be > 0");
            assert!(tile.width > 0 && tile.height > 0, "[CRITICAL] Tile dimensions must be > 0");
            
            let tile_params_data = TileParams {
                tile_origin_size: [tile.x, tile.y, tile.width, tile.height],
                img_spp_counts: [width, height, spp_batch, 0],  // Use actual SPP batch
                seeds_misc: [params.base_uniforms.seed_hi, params.debug_mode, 0, 0],  // debug_mode from params
                _pad0: [0; 32],
                _pad1: [0; 20],
            };
            
            // Calculate workgroup dispatch
            let wg_x = (tile.width + 7) / 8;
            let wg_y = (tile.height + 7) / 8;
            
            eprintln!("[TileParams] Tile {}: origin=({},{}) size={}x{} img={}x{} spp_batch={} debug_mode={} workgroups=({},{}) seed={}",
                tile_idx + 1, tile.x, tile.y, tile.width, tile.height, width, height, 
                spp_batch, params.debug_mode, wg_x, wg_y, params.base_uniforms.seed_hi);
            
            // Render this tile at full resolution
            let tile_data = self.render_tile_with_params(
                tile.width,
                tile.height,
                spheres,
                hybrid_scene,
                params,
                &tile_params_data,
            )?;
            // P0 telemetry: compute basic stats for this tile
            if params.show_progress {
                let mut sum_r: u64 = 0;
                let mut sum_g: u64 = 0;
                let mut sum_b: u64 = 0;
                let mut pure_green: u64 = 0;
                let mut g_dominant: u64 = 0;
                let mut uniq: HashSet<u32> = HashSet::with_capacity((tile.width as usize * tile.height as usize).min(131072));
                let mut first5: Vec<String> = Vec::with_capacity(5);
                let mut counted = 0usize;
                for (pi, p) in tile_data.chunks(4).enumerate() {
                    let r = p[0] as u32;
                    let g = p[1] as u32;
                    let b = p[2] as u32;
                    sum_r += r as u64;
                    sum_g += g as u64;
                    sum_b += b as u64;
                    let key = r | (g << 8) | (b << 16);
                    uniq.insert(key);
                    if r == 0 && b == 0 && g > 0 { pure_green += 1; }
                    if (g as i32) > (r as i32 + 10) && (g as i32) > (b as i32 + 10) { g_dominant += 1; }
                    if pi < 5 {
                        first5.push(format!("#{:02X}{:02X}{:02X}", r as u8, g as u8, b as u8));
                    }
                    counted += 1;
                }
                let total = (tile.width as u64) * (tile.height as u64);
                let mean_r = (sum_r as f64) / (total as f64);
                let mean_g = (sum_g as f64) / (total as f64);
                let mean_b = (sum_b as f64) / (total as f64);
                let pg_pct = 100.0 * (pure_green as f64) / (total as f64);
                let gd_pct = 100.0 * (g_dominant as f64) / (total as f64);
                
                // CRITICAL VALIDATION: Check for solid black or solid color tiles
                let is_solid_color = uniq.len() == 1;
                let is_nearly_black = mean_r < 1.0 && mean_g < 1.0 && mean_b < 1.0;
                
                if is_solid_color || is_nearly_black {
                    eprintln!(
                        "[HYBRID-RENDER] WARNING: Tile {} is solid color! origin=({},{}) size={}x{} unique_colors={} meanRGB=({:.1},{:.1},{:.1})",
                        tile_idx + 1,
                        tile.x,
                        tile.y,
                        tile.width,
                        tile.height,
                        uniq.len(),
                        mean_r, mean_g, mean_b
                    );
                }
                
                eprintln!(
                    "[TileStats] {}: origin=({},{}) size={}x{} debug_mode={} unique_colors={} meanRGB=({:.1},{:.1},{:.1}) pure_green={:.2}% g_dominant={:.2}% first5={}",
                    tile_idx + 1,
                    tile.x,
                    tile.y,
                    tile.width,
                    tile.height,
                    params.debug_mode,
                    uniq.len(),
                    mean_r, mean_g, mean_b,
                    pg_pct, gd_pct,
                    first5.join(",")
                );
            }
            
            // Copy tile into final output at correct position
            for ty in 0..tile.height {
                let src_offset = (ty as usize) * (tile.width as usize) * 4;
                let dst_offset = ((tile.y + ty) as usize) * (width as usize) * 4 + (tile.x as usize) * 4;
                let row_bytes = (tile.width as usize) * 4;
                
                output[dst_offset..dst_offset + row_bytes]
                    .copy_from_slice(&tile_data[src_offset..src_offset + row_bytes]);
            }
            
            let tile_elapsed = tile_start.elapsed().as_secs_f64();
            if params.show_progress {
                eprintln!("[TRUE-RES TILED] Tile {} complete in {:.1}s", tile_idx + 1, tile_elapsed);
            }
        }
        
        let total_elapsed = start_time.elapsed().as_secs_f64();
        eprintln!("[TRUE-RES TILED] Complete: {} tiles in {:.1}s", total_tiles, total_elapsed);
        
        // Apply OIDN denoising if requested
        if params.oidn_mode != OidnMode::Off {
            eprintln!("[TRUE-RES TILED] Applying OIDN denoising (mode: {:?})...", params.oidn_mode);
            let oidn_config = TiledOidnConfig {
                mode: params.oidn_mode,
                overlap: 32,
                strength: 0.8, // TODO: make configurable
            };
            
            match denoise_tiled(&output, width, height, tile_w, tile_h, &oidn_config) {
                Ok(denoised) => {
                    eprintln!("[TRUE-RES TILED] Denoising complete");
                    return Ok(denoised);
                }
                Err(e) => {
                    eprintln!("[TRUE-RES TILED] WARNING: Denoising failed: {}. Using raw output.", e);
                }
            }
        }
        
        Ok(output)
    }
    
    /// Render a single tile with TileParams for true-resolution tiling
    fn render_tile_with_params(
        &self,
        width: u32,
        height: u32,
        spheres: &[Sphere],
        hybrid_scene: &HybridScene,
        params: &HybridTracerParams,
        tile_params: &TileParams,
    ) -> Result<Vec<u8>, RenderError> {
        // CRITICAL: width/height here are TILE dimensions
        // The full image dimensions are in tile_params.img_spp_counts[0..1]
        // Base uniforms MUST use full image dimensions for camera ray generation!
        
        let full_img_w = tile_params.img_spp_counts[0];
        let full_img_h = tile_params.img_spp_counts[1];
        
        eprintln!("[render_tile_with_params] tile={}x{}, full_img={}x{}, origin=({},{})",
            width, height, full_img_w, full_img_h,
            tile_params.tile_origin_size[0], tile_params.tile_origin_size[1]);
        
        // Pass params with FULL image dimensions in uniforms for correct camera rays
        // The tile-specific information is in TileParams, not in base uniforms
        let mut modified_params = params.clone();
        modified_params.base_uniforms.width = full_img_w;
        modified_params.base_uniforms.height = full_img_h;
        
        // render_single_tile will create output texture with TILE dimensions
        // and use TileParams for coordinate mapping
        self.render_single_tile(width, height, spheres, hybrid_scene, modified_params, Some(tile_params))
    }
    
    /// Render large images by downscaling then upsampling (LEGACY, opt-in only)
    fn render_downscaled(
        &self,
        width: u32,
        height: u32,
        spheres: &[Sphere],
        hybrid_scene: &HybridScene,
        params: HybridTracerParams,
    ) -> Result<Vec<u8>, RenderError> {
        // Calculate safe resolution maintaining aspect ratio (stay well below 640px limit)
        let max_safe = 580u32;
        let scale = (width.max(height) as f32 / max_safe as f32).max(1.0);
        let render_w = (width as f32 / scale) as u32;
        let render_h = (height as f32 / scale) as u32;
        
        eprintln!("[HYBRID-RENDER] Downscaling {}x{} to {}x{} (scale {:.2}x)",
            width, height, render_w, render_h, scale);
        
        // Update uniforms to match the actual render resolution
        let mut render_params = params.clone();
        render_params.base_uniforms.width = render_w;
        render_params.base_uniforms.height = render_h;
        
        // Render at safe resolution with corrected uniforms
        let downscaled = self.render_single_tile(
            render_w,
            render_h,
            spheres,
            hybrid_scene,
            render_params,
            None,  // No TileParams override for downscaled path
        )?;
        
        // Bilinear upscale to target resolution
        let upscaled = self.upscale_bilinear(&downscaled, render_w, render_h, width, height);
        
        eprintln!("[HYBRID-RENDER] Upscaled {}x{} to {}x{}",
            render_w, render_h, width, height);
        
        Ok(upscaled)
    }
    
    /// Bilinear upscale from source to target resolution
    /// Uses row-by-row processing for better memory behavior
    fn upscale_bilinear(
        &self,
        src: &[u8],
        src_w: u32,
        src_h: u32,
        dst_w: u32,
        dst_h: u32,
    ) -> Vec<u8> {
        eprintln!("[UPSCALE] Allocating {}x{} = {} MB", dst_w, dst_h, (dst_w * dst_h * 4) as f64 / 1e6);
        let mut dst = Vec::with_capacity((dst_w * dst_h * 4) as usize);
        
        let x_ratio = (src_w - 1) as f32 / dst_w as f32;
        let y_ratio = (src_h - 1) as f32 / dst_h as f32;
        
        // Process row by row
        for dy in 0..dst_h {
            if dy % 256 == 0 {
                eprintln!("[UPSCALE] Processing row {}/{}", dy, dst_h);
            }
            
            let sy = dy as f32 * y_ratio;
            let sy0 = (sy.floor() as u32).min(src_h - 1);
            let sy1 = (sy0 + 1).min(src_h - 1);
            let fy = sy - sy0 as f32;
            
            for dx in 0..dst_w {
                let sx = dx as f32 * x_ratio;
                let sx0 = (sx.floor() as u32).min(src_w - 1);
                let sx1 = (sx0 + 1).min(src_w - 1);
                let fx = sx - sx0 as f32;
                
                let i00 = (sy0 * src_w + sx0) as usize * 4;
                let i10 = (sy0 * src_w + sx1) as usize * 4;
                let i01 = (sy1 * src_w + sx0) as usize * 4;
                let i11 = (sy1 * src_w + sx1) as usize * 4;
                
                // Bilinear interpolation for each channel
                for c in 0..4 {
                    let v00 = src[i00 + c] as f32;
                    let v10 = src[i10 + c] as f32;
                    let v01 = src[i01 + c] as f32;
                    let v11 = src[i11 + c] as f32;
                    
                    let v0 = v00 * (1.0 - fx) + v10 * fx;
                    let v1 = v01 * (1.0 - fx) + v11 * fx;
                    let v = v0 * (1.0 - fy) + v1 * fy;
                    
                    dst.push(v.clamp(0.0, 255.0) as u8);
                }
            }
        }
        
        eprintln!("[UPSCALE] Complete: {} bytes", dst.len());
        dst
    }

    // Bind group layout creation methods
    fn create_uniforms_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hybrid-pt-bgl0-uniforms"),
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
        })
    }

    fn create_scene_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hybrid-pt-bgl1-scene"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        })
    }

    fn load_hdri_texture(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        params: &HybridTracerParams,
    ) -> Result<(wgpu::Texture, wgpu::TextureView), RenderError> {
        let path = params.hdri_path.as_ref().ok_or_else(|| {
            RenderError::upload("HDRI enabled but no hdri_path provided")
        })?;

        let hdr = load_hdr(path).map_err(|e| {
            RenderError::upload(format!("Failed to load HDRI '{}': {}", path.display(), e))
        })?;

        let texture = create_texture_2d(
            device,
            "hybrid-pt-hdri",
            hdr.width,
            hdr.height,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );

        let rgba = hdr.to_rgba();
        let mut rgba16 = Vec::with_capacity(rgba.len() * 2);
        for &v in &rgba {
            rgba16.extend_from_slice(&f16::from_f32(v).to_le_bytes());
        }

        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &rgba16,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(hdr.width * 8),
                rows_per_image: Some(hdr.height),
            },
            wgpu::Extent3d {
                width: hdr.width,
                height: hdr.height,
                depth_or_array_layers: 1,
            },
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Ok((texture, view))
    }

    fn fallback_hdri_texture(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = create_texture_2d(
            device,
            "hybrid-pt-hdri-fallback",
            1,
            1,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );

        let pixel = [
            f16::from_f32(0.5),
            f16::from_f32(0.5),
            f16::from_f32(0.5),
            f16::from_f32(1.0),
        ];
        let mut bytes = Vec::with_capacity(8);
        for p in &pixel {
            bytes.extend_from_slice(&p.to_le_bytes());
        }

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
                bytes_per_row: Some(8),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn create_accum_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hybrid-pt-bgl2-tile-params"),
            entries: &[
                // Binding 0: TileParams uniform
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
                // Binding 1: Accumulation buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    fn create_output_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hybrid-pt-bgl3-out"),
            entries: &[
                // out_tex at binding 0
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // AOVs binding 1..7
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        })
    }

    fn create_aov_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hybrid-pt-bgl4-aovs"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        })
    }

    fn create_hybrid_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hybrid-pt-bgl5-hybrid"),
            entries: &[
                // Hybrid uniforms
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
                // SDF primitives
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // SDF nodes
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Mesh vertices
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Mesh indices
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // BVH nodes
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }
}
