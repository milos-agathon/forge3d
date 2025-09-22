// src/path_tracing/wavefront/mod.rs
// Wavefront Path Tracer: Main scheduler and orchestration
// Queue-based GPU path tracing with persistent threads and stream compaction

pub mod pipeline;
pub mod queues;

use crate::path_tracing::restir::{
    create_debug_aov_buffer, create_diag_flags_buffer, create_reservoir_buffer,
    create_restir_gbuffer, create_restir_gbuffer_pos, empty_alias_entries_buffer,
    empty_light_samples_buffer,
};
use crate::path_tracing::TracerParams;
use crate::scene::Scene;
use glam::Mat4;
use pipeline::WavefrontPipelines;
use queues::*;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use wgpu::{BindGroup, Buffer, CommandEncoder, Device, Queue};

// Constants from task specification
const MAX_DEPTH: u32 = 8;
const WORKGROUP_SIZE: u32 = 256;
const QUEUE_CAPACITY_SCALE: u32 = 4; // capacity = width * height * scale

/// Wavefront path tracer scheduler
/// Orchestrates queue-based path tracing on GPU with multiple stages
pub struct WavefrontScheduler {
    device: Arc<Device>,
    queue: Arc<Queue>,
    pipelines: WavefrontPipelines,
    queue_buffers: QueueBuffers,
    width: u32,
    height: u32,
    frame_index: u32,
    // Feature toggle: temporarily disable ReSTIR passes to focus on core path
    restir_enabled: bool,
    // Fine-grained toggle: enable/disable spatial pass separately
    restir_spatial_enabled: bool,
    restir_reservoirs: Buffer,
    restir_light_samples: Buffer,
    restir_alias_entries: Buffer,
    restir_light_probs: Buffer,
    restir_prev: Buffer,
    restir_out: Buffer,
    restir_diag_flags: Buffer,
    restir_debug_aov: Buffer,
    restir_gbuffer: Buffer,
    restir_gbuffer_pos: Buffer,
    restir_settings: Buffer,
    restir_gbuffer_mat: Buffer,
    instances_buffer: Buffer,
    blas_descs: Buffer,
    // Stored minimal scene bind group for ReSTIR spatial (lights + G-buffers)
    restir_scene_spatial_bind_group: Option<BindGroup>,
    // SVGF guidance AOVs written by shading at primary hit (threaded for future SVGF stages)
    aov_albedo: Buffer,
    aov_depth: Buffer,
    aov_normal: Buffer,
    // Participating media (single scatter HG) parameters uniform
    medium_params: Buffer,
    // Hair curve segments buffer (world-space, read-only in shaders)
    hair_segments: Buffer,
}

impl WavefrontScheduler {
    /// Create new wavefront scheduler
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        width: u32,
        height: u32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let pipelines = WavefrontPipelines::new(&device)?;
        let queue_capacity = width * height * QUEUE_CAPACITY_SCALE;
        let queue_buffers = QueueBuffers::new(&device, queue_capacity)?;
        let restir_reservoirs =
            create_reservoir_buffer(&device, (width as usize) * (height as usize));
        let restir_prev = create_reservoir_buffer(&device, (width as usize) * (height as usize));
        let restir_out = create_reservoir_buffer(&device, (width as usize) * (height as usize));
        // Initialize light samples buffer with one padded element to satisfy binding size
        let restir_light_samples = crate::path_tracing::restir::create_light_samples_buffer(
            &device,
            &[crate::path_tracing::restir::LightSample {
                position: [0.0; 3],
                light_index: 0,
                direction: [0.0; 3],
                intensity: 0.0,
                light_type: 0,
                params: [0.0; 3],
                _pad: [0; 4],
            }],
        );
        let restir_alias_entries = empty_alias_entries_buffer(&device);
        let restir_light_probs = crate::path_tracing::restir::empty_light_probs_buffer(&device);
        let restir_diag_flags =
            create_diag_flags_buffer(&device, (width as usize) * (height as usize));
        let restir_debug_aov =
            create_debug_aov_buffer(&device, (width as usize) * (height as usize));
        let restir_gbuffer = create_restir_gbuffer(&device, (width as usize) * (height as usize));
        let restir_gbuffer_pos =
            create_restir_gbuffer_pos(&device, (width as usize) * (height as usize));
        // RestirSettings uniform defaults:
        // [debug_aov_mode=0, qmc_mode=1 (Sobol), adaptive_threshold=0.25f, pad]
        let settings_init: [u32; 4] = [0, 1, f32::to_bits(0.25f32), 0];
        let restir_settings = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("restir-settings-uniform"),
            contents: bytemuck::cast_slice(&settings_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        // Material ID G-buffer (u32 per pixel)
        let mat_zero: Vec<u32> = vec![0u32; (width as usize) * (height as usize)];
        let restir_gbuffer_mat = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("restir-gbuffer-mat"),
            contents: bytemuck::cast_slice(&mat_zero),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        // Create a minimal instances buffer with one identity instance to satisfy binding
        // Layout matches accel::instancing::InstanceData and WGSL Instance
        let ident: [f32; 16] = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let inst0 = crate::accel::instancing::InstanceData {
            transform: ident,
            inv_transform: ident,
            blas_index: 0,
            material_id: 0,
            _padding: [0; 2],
        };
        let instances_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("instances-buffer-initial"),
            contents: bytemuck::cast_slice(&[inst0]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Create a default BLAS descriptor table with one empty entry (index 0)
        let default_desc = crate::path_tracing::mesh::BlasDesc {
            node_offset: 0,
            node_count: 0,
            tri_offset: 0,
            tri_count: 0,
            vtx_offset: 0,
            vtx_count: 0,
            _pad0: 0,
            _pad1: 0,
        };
        let blas_descs = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("blas-descs-default"),
            contents: bytemuck::cast_slice(&[default_desc]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Allocate SVGF guidance AOV buffers (RGBA32F per pixel)
        let px_count = (width as usize) * (height as usize);
        let aov_bytes: u64 = (px_count * std::mem::size_of::<[f32; 4]>()) as u64;
        let aov_albedo = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("aov-albedo"),
            size: aov_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let aov_depth = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("aov-depth"),
            size: aov_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let aov_normal = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("aov-normal"),
            size: aov_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Medium params uniform (g, sigma_t, density, enable)
        let medium_init: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
        let medium_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("medium-params-uniform"),
            contents: bytemuck::cast_slice(&medium_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Hair segments default buffer (1 dummy)
        let dummy_seg: [u32; 4] = [0; 4];
        let hair_segments = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hair-segments"),
            contents: bytemuck::cast_slice(&dummy_seg),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        Ok(Self {
            device,
            queue,
            pipelines,
            queue_buffers,
            width,
            height,
            frame_index: 0,
            restir_enabled: false,
            restir_spatial_enabled: false,
            restir_reservoirs,
            restir_light_samples,
            restir_alias_entries,
            restir_light_probs,
            restir_prev,
            restir_out,
            restir_diag_flags,
            restir_debug_aov,
            restir_gbuffer,
            restir_gbuffer_pos,
            restir_settings,
            restir_gbuffer_mat,
            instances_buffer,
            blas_descs,
            restir_scene_spatial_bind_group: None,
            aov_albedo,
            aov_depth,
            aov_normal,
            medium_params,
            hair_segments,
        })
    }

    /// Enable/disable ReSTIR passes (init/temporal/spatial)
    pub fn set_restir_enabled(&mut self, enabled: bool) {
        self.restir_enabled = enabled;
    }

    /// Enable/disable ReSTIR spatial pass independently
    pub fn set_restir_spatial_enabled(&mut self, enabled: bool) {
        self.restir_spatial_enabled = enabled;
    }

    /// Create ReSTIR spatial bind group (Group 2 for spatial pass)
    fn create_restir_spatial_bind_group(&self) -> Result<BindGroup, Box<dyn std::error::Error>> {
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("restir-spatial-bind-group"),
            layout: &self.pipelines.restir_spatial_bind_group_layout,
            entries: &[
                // in: current prev (temporal result)
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.restir_prev.as_entire_binding(),
                },
                // out: spatial output buffer
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.restir_out.as_entire_binding(),
                },
            ],
        });
        Ok(bind_group)
    }

    /// Initialize and store the minimal scene bind group used by ReSTIR spatial (Group 1)
    pub fn init_restir_scene_spatial_bind_group(
        &mut self,
        area_lights: &Buffer,
        directional_lights: &Buffer,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("restir-scene-spatial-bind-group"),
            layout: &self.pipelines.restir_scene_spatial_bind_group_layout,
            entries: &[
                // 4: area lights (RO)
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: area_lights.as_entire_binding(),
                },
                // 5: directional lights (RO)
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: directional_lights.as_entire_binding(),
                },
                // 10: G-buffer normal/roughness (RO)
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: self.restir_gbuffer.as_entire_binding(),
                },
                // 11: G-buffer position (RO)
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: self.restir_gbuffer_pos.as_entire_binding(),
                },
            ],
        });
        self.restir_scene_spatial_bind_group = Some(bg);
        Ok(())
    }

    /// Resize for new dimensions
    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), Box<dyn std::error::Error>> {
        self.width = width;
        self.height = height;
        let queue_capacity = width * height * QUEUE_CAPACITY_SCALE;
        self.queue_buffers = QueueBuffers::new(&self.device, queue_capacity)?;
        self.restir_reservoirs =
            create_reservoir_buffer(&self.device, (width as usize) * (height as usize));
        self.restir_light_samples = empty_light_samples_buffer(&self.device);
        self.restir_alias_entries = empty_alias_entries_buffer(&self.device);
        self.restir_light_probs =
            crate::path_tracing::restir::empty_light_probs_buffer(&self.device);
        self.restir_prev =
            create_reservoir_buffer(&self.device, (width as usize) * (height as usize));
        self.restir_out =
            create_reservoir_buffer(&self.device, (width as usize) * (height as usize));
        self.restir_diag_flags =
            create_diag_flags_buffer(&self.device, (width as usize) * (height as usize));
        self.restir_debug_aov =
            create_debug_aov_buffer(&self.device, (width as usize) * (height as usize));
        self.restir_gbuffer =
            create_restir_gbuffer(&self.device, (width as usize) * (height as usize));
        self.restir_gbuffer_pos =
            create_restir_gbuffer_pos(&self.device, (width as usize) * (height as usize));
        // Recreate mat-id buffer
        let mat_zero: Vec<u32> = vec![0u32; (width as usize) * (height as usize)];
        self.restir_gbuffer_mat =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("restir-gbuffer-mat"),
                    contents: bytemuck::cast_slice(&mat_zero),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                });
        // Recreate SVGF AOV buffers
        let px_count = (width as usize) * (height as usize);
        let aov_bytes: u64 = (px_count * std::mem::size_of::<[f32; 4]>()) as u64;
        self.aov_albedo = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("aov-albedo"),
            size: aov_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.aov_depth = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("aov-depth"),
            size: aov_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.aov_normal = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("aov-normal"),
            size: aov_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Recreate medium params to keep alignment (copy previous if needed)
        let medium_init: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
        self.medium_params = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("medium-params-uniform"),
                contents: bytemuck::cast_slice(&medium_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        // Keep hair_segments as-is on resize
        Ok(())
    }

    /// Dispatch ReSTIR spatial reuse stage (neighborhood reuse)
    fn dispatch_restir_spatial(
        &self,
        encoder: &mut CommandEncoder,
        uniforms_buffer: &Buffer,
        _scene_bind_group: &BindGroup,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let uniforms_bind_group = self.create_uniforms_bind_group(uniforms_buffer)?;
        let spatial_bind_group = self.create_restir_spatial_bind_group()?;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("restir-spatial-pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipelines.restir_spatial);
        pass.set_bind_group(0, &uniforms_bind_group, &[]);
        // Spatial pipeline layout: 0=uniforms, 1=scene(minimal), 2=spatial
        if let Some(ref scene_spatial_bg) = self.restir_scene_spatial_bind_group {
            pass.set_bind_group(1, scene_spatial_bg, &[]);
        } else {
            // If not initialized, skip spatial to avoid invalid binding
            return Ok(());
        }
        pass.set_bind_group(2, &spatial_bind_group, &[]);

        let num_pixels = self.width * self.height;
        let workgroups = (num_pixels + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        pass.dispatch_workgroups(workgroups, 1, 1);

        Ok(())
    }

    /// Dispatch ReSTIR temporal reuse stage (combines prev + curr into out)
    fn dispatch_restir_temporal(
        &self,
        encoder: &mut CommandEncoder,
        uniforms_buffer: &Buffer,
        scene_bind_group: &BindGroup,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let uniforms_bind_group = self.create_uniforms_bind_group(uniforms_buffer)?;
        let temporal_bind_group = self.create_restir_temporal_bind_group()?;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("restir-temporal-pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipelines.restir_temporal);
        pass.set_bind_group(0, &uniforms_bind_group, &[]);
        pass.set_bind_group(1, scene_bind_group, &[]);
        pass.set_bind_group(2, &temporal_bind_group, &[]);

        let num_pixels = self.width * self.height;
        let workgroups = (num_pixels + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        pass.dispatch_workgroups(workgroups, 1, 1);

        Ok(())
    }

    /// Update ReSTIR light data buffers (samples and alias entries)
    pub fn set_restir_light_data(&mut self, light_samples: Buffer, alias_entries: Buffer) {
        self.restir_light_samples = light_samples;
        self.restir_alias_entries = alias_entries;
    }

    /// Update per-light probabilities buffer (binding 3 in ReSTIR layout)
    pub fn set_restir_light_probs(&mut self, light_probs: Buffer) {
        self.restir_light_probs = light_probs;
    }

    /// Dispatch shadow stage
    fn dispatch_shadow(
        &self,
        encoder: &mut CommandEncoder,
        uniforms_buffer: &Buffer,
        scene_bind_group: &BindGroup,
        accum_bind_group: &BindGroup,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let uniforms_bind_group = self.create_uniforms_bind_group(uniforms_buffer)?;
        let queue_bind_group = self.queue_buffers.create_shadow_bind_group(&self.device)?;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("shadow-pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipelines.shadow);
        pass.set_bind_group(0, &uniforms_bind_group, &[]);
        pass.set_bind_group(1, scene_bind_group, &[]);
        pass.set_bind_group(2, &queue_bind_group, &[]);
        pass.set_bind_group(3, accum_bind_group, &[]);

        let active_capacity = self.queue_buffers.capacity;
        let workgroups = (active_capacity + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        pass.dispatch_workgroups(workgroups, 1, 1);

        Ok(())
    }

    /// Execute one frame of wavefront path tracing
    pub fn render_frame(
        &mut self,
        _scene: &Scene,
        _params: &TracerParams,
        _accum_buffer: &Buffer,
        uniforms_buffer: &Buffer,
        scene_bind_group: &BindGroup,
        accum_bind_group: &BindGroup,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wavefront-frame"),
            });

        // Reset queue counters for new frame
        self.queue_buffers.reset_counters(&self.queue, &mut encoder);

        // Optionally run ReSTIR passes (disabled by default)
        if self.restir_enabled {
            // ReSTIR Init: initialize per-pixel reservoirs (A8 stub)
            self.dispatch_restir_init(&mut encoder, uniforms_buffer, scene_bind_group)?;

            // ReSTIR Temporal: combine last frame with current (MVP stub)
            // Note: do not swap here; keep prev as input (RO) and out as output (RW) for spatial
            self.dispatch_restir_temporal(&mut encoder, uniforms_buffer, scene_bind_group)?;

            // ReSTIR Spatial: neighborhood reuse on temporal result (optional)
            if self.restir_spatial_enabled {
                self.dispatch_restir_spatial(&mut encoder, uniforms_buffer, scene_bind_group)?;
                // Swap again so prev holds spatial result for shading
                {
                    use std::mem::swap;
                    swap(&mut (self.restir_prev), &mut (self.restir_out));
                }
            }
        }

        // Continue with core wavefront path tracing passes

        // Stage 1: Ray generation - generate primary rays
        self.dispatch_raygen(
            &mut encoder,
            uniforms_buffer,
            scene_bind_group,
            accum_bind_group,
        )?;

        // Wavefront loop: iterate until all rays terminated
        let max_iterations = MAX_DEPTH * 2; // Safety limit
        for iteration in 0..max_iterations {
            // Check if any rays remain
            let ray_count =
                self.queue_buffers
                    .get_active_ray_count(&self.device, &self.queue, &mut encoder)?;
            if ray_count == 0 {
                break;
            }

            // Stage 2: Intersection - test rays against scene
            self.dispatch_intersect(&mut encoder, uniforms_buffer, scene_bind_group)?;

            // Stage 3: Shading - evaluate BRDF and generate scatter rays
            self.dispatch_shade(
                &mut encoder,
                uniforms_buffer,
                scene_bind_group,
                accum_bind_group,
            )?;

            // Stage 4: Shadow - test NEE visibility and accumulate contributions
            self.dispatch_shadow(
                &mut encoder,
                uniforms_buffer,
                scene_bind_group,
                accum_bind_group,
            )?;

            // Stage 5: Scatter - convert scatter rays back to regular rays
            self.dispatch_scatter(
                &mut encoder,
                uniforms_buffer,
                scene_bind_group,
                accum_bind_group,
            )?;

            // Stage 6: Compaction - remove terminated rays (optional, every N iterations)
            if iteration % 2 == 0 && iteration > 0 {
                self.dispatch_compact(&mut encoder)?;
            }
        }

        // Submit all GPU work
        let command_buffer = encoder.finish();
        self.queue.submit(std::iter::once(command_buffer));

        self.frame_index += 1;
        Ok(())
    }

    /// Dispatch ray generation stage
    fn dispatch_raygen(
        &self,
        encoder: &mut CommandEncoder,
        uniforms_buffer: &Buffer,
        scene_bind_group: &BindGroup,
        accum_bind_group: &BindGroup,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let uniforms_bind_group = self.create_uniforms_bind_group(uniforms_buffer)?;
        let queue_bind_group = self.queue_buffers.create_raygen_bind_group(&self.device)?;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("raygen-pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipelines.raygen);
        pass.set_bind_group(0, &uniforms_bind_group, &[]);
        pass.set_bind_group(1, scene_bind_group, &[]);
        pass.set_bind_group(2, &queue_bind_group, &[]);
        pass.set_bind_group(3, accum_bind_group, &[]);

        let num_pixels = self.width * self.height;
        let workgroups = (num_pixels + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        pass.dispatch_workgroups(workgroups, 1, 1);

        Ok(())
    }

    /// Dispatch intersection stage
    fn dispatch_intersect(
        &self,
        encoder: &mut CommandEncoder,
        uniforms_buffer: &Buffer,
        scene_bind_group: &BindGroup,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let uniforms_bind_group = self.create_uniforms_bind_group(uniforms_buffer)?;
        let queue_bind_group = self
            .queue_buffers
            .create_intersect_bind_group(&self.device)?;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("intersect-pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipelines.intersect);
        pass.set_bind_group(0, &uniforms_bind_group, &[]);
        pass.set_bind_group(1, scene_bind_group, &[]);
        pass.set_bind_group(2, &queue_bind_group, &[]);

        // Persistent threads - dispatch more threads than rays for load balancing
        let active_capacity = self.queue_buffers.capacity;
        let workgroups = (active_capacity + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        pass.dispatch_workgroups(workgroups, 1, 1);

        Ok(())
    }

    /// Dispatch shading stage
    fn dispatch_shade(
        &self,
        encoder: &mut CommandEncoder,
        uniforms_buffer: &Buffer,
        scene_bind_group: &BindGroup,
        accum_bind_group: &BindGroup,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let uniforms_bind_group = self.create_uniforms_bind_group(uniforms_buffer)?;
        let queue_bind_group = self.queue_buffers.create_shade_bind_group(&self.device)?;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("shade-pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipelines.shade);
        pass.set_bind_group(0, &uniforms_bind_group, &[]);
        pass.set_bind_group(1, scene_bind_group, &[]);
        pass.set_bind_group(2, &queue_bind_group, &[]);
        pass.set_bind_group(3, accum_bind_group, &[]);

        let active_capacity = self.queue_buffers.capacity;
        let workgroups = (active_capacity + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        pass.dispatch_workgroups(workgroups, 1, 1);

        Ok(())
    }

    /// Dispatch scatter stage
    fn dispatch_scatter(
        &self,
        encoder: &mut CommandEncoder,
        uniforms_buffer: &Buffer,
        scene_bind_group: &BindGroup,
        accum_bind_group: &BindGroup,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let uniforms_bind_group = self.create_uniforms_bind_group(uniforms_buffer)?;
        let queue_bind_group = self.queue_buffers.create_scatter_bind_group(&self.device)?;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("scatter-pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipelines.scatter);
        pass.set_bind_group(0, &uniforms_bind_group, &[]);
        pass.set_bind_group(1, scene_bind_group, &[]);
        pass.set_bind_group(2, &queue_bind_group, &[]);
        pass.set_bind_group(3, accum_bind_group, &[]);

        let active_capacity = self.queue_buffers.capacity;
        let workgroups = (active_capacity + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        pass.dispatch_workgroups(workgroups, 1, 1);

        Ok(())
    }

    /// Dispatch compaction stage
    fn dispatch_compact(
        &self,
        encoder: &mut CommandEncoder,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let queue_bind_group = self.queue_buffers.create_compact_bind_group(&self.device)?;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compact-pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipelines.compact);
        pass.set_bind_group(2, &queue_bind_group, &[]);

        // Use simple compaction for now (single workgroup)
        pass.dispatch_workgroups(1, 1, 1);

        Ok(())
    }

    /// Create uniforms bind group
    fn create_uniforms_bind_group(
        &self,
        uniforms_buffer: &Buffer,
    ) -> Result<BindGroup, Box<dyn std::error::Error>> {
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("uniforms-bind-group"),
            layout: &self.pipelines.uniforms_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniforms_buffer.as_entire_binding(),
            }],
        });
        Ok(bind_group)
    }

    /// Create ReSTIR bind group (Group 2 for restir init stage)
    fn create_restir_bind_group(&self) -> Result<BindGroup, Box<dyn std::error::Error>> {
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("restir-bind-group"),
            layout: &self.pipelines.restir_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.restir_reservoirs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.restir_light_samples.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.restir_alias_entries.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.restir_light_probs.as_entire_binding(),
                },
            ],
        });
        Ok(bind_group)
    }

    /// Create ReSTIR temporal bind group (Group 2 for temporal pass)
    fn create_restir_temporal_bind_group(&self) -> Result<BindGroup, Box<dyn std::error::Error>> {
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("restir-temporal-bind-group"),
            layout: &self.pipelines.restir_temporal_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.restir_prev.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.restir_reservoirs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.restir_out.as_entire_binding(),
                },
            ],
        });
        Ok(bind_group)
    }

    /// Dispatch ReSTIR initial reservoir population stage
    fn dispatch_restir_init(
        &self,
        encoder: &mut CommandEncoder,
        uniforms_buffer: &Buffer,
        scene_bind_group: &BindGroup,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let uniforms_bind_group = self.create_uniforms_bind_group(uniforms_buffer)?;
        let restir_bind_group = self.create_restir_bind_group()?;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("restir-init-pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipelines.restir_init);
        pass.set_bind_group(0, &uniforms_bind_group, &[]);
        pass.set_bind_group(1, scene_bind_group, &[]);
        pass.set_bind_group(2, &restir_bind_group, &[]);

        let num_pixels = self.width * self.height;
        let workgroups = (num_pixels + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        pass.dispatch_workgroups(workgroups, 1, 1);

        Ok(())
    }

    /// Get current frame index for deterministic seeding
    pub fn frame_index(&self) -> u32 {
        self.frame_index
    }

    /// Reset frame index
    pub fn reset_frame_index(&mut self) {
        self.frame_index = 0;
    }

    /// Get number of pixels for current frame size (width*height)
    pub fn aov_pixel_count(&self) -> usize {
        (self.width as usize) * (self.height as usize)
    }

    /// Copy the full RGBA32F depth AOV buffer into dst
    ///
    /// The destination buffer must be at least 16 bytes * pixel_count in size and have COPY_DST usage.
    pub fn copy_aov_depth_to(&self, encoder: &mut wgpu::CommandEncoder, dst: &wgpu::Buffer) {
        let bytes = (self.aov_pixel_count() * core::mem::size_of::<[f32; 4]>()) as u64;
        encoder.copy_buffer_to_buffer(&self.aov_depth, 0, dst, 0, bytes);
    }

    /// Copy the full RGBA32F albedo AOV buffer into dst
    pub fn copy_aov_albedo_to(&self, encoder: &mut wgpu::CommandEncoder, dst: &wgpu::Buffer) {
        let bytes = (self.aov_pixel_count() * core::mem::size_of::<[f32; 4]>()) as u64;
        encoder.copy_buffer_to_buffer(&self.aov_albedo, 0, dst, 0, bytes);
    }

    /// Copy the full RGBA32F normal AOV buffer into dst
    pub fn copy_aov_normal_to(&self, encoder: &mut wgpu::CommandEncoder, dst: &wgpu::Buffer) {
        let bytes = (self.aov_pixel_count() * core::mem::size_of::<[f32; 4]>()) as u64;
        encoder.copy_buffer_to_buffer(&self.aov_normal, 0, dst, 0, bytes);
    }

    /// Getters for AOV buffers (read-only references)
    pub fn aov_depth_buffer(&self) -> &Buffer {
        &self.aov_depth
    }
    pub fn aov_normal_buffer(&self) -> &Buffer {
        &self.aov_normal
    }
    pub fn aov_albedo_buffer(&self) -> &Buffer {
        &self.aov_albedo
    }

    /// Create scene bind group (Group 1) binding spheres/materials and mesh buffers
    pub fn create_scene_bind_group(
        &self,
        spheres_buffer: &Buffer,
        mesh_vertices: &Buffer,
        mesh_indices: &Buffer,
        mesh_bvh: &Buffer,
        area_lights: &Buffer,
        directional_lights: &Buffer,
        object_importance: &Buffer,
    ) -> Result<BindGroup, Box<dyn std::error::Error>> {
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wavefront-scene-bind-group"),
            layout: &self.pipelines.scene_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: spheres_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: mesh_vertices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: mesh_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: mesh_bvh.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: area_lights.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: directional_lights.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: object_importance.as_entire_binding(),
                },
                // ReSTIR reservoirs (temporal/spatial result) for shading
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.restir_prev.as_entire_binding(),
                },
                // Diagnostics flags buffer (read-only view in shading)
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: self.restir_diag_flags.as_entire_binding(),
                },
                // Debug AOV buffer (read-only in shading)
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: self.restir_debug_aov.as_entire_binding(),
                },
                // ReSTIR G-buffer (normal.xyz, roughness) (RW in shading)
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: self.restir_gbuffer.as_entire_binding(),
                },
                // ReSTIR G-buffer position (RW in shading)
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: self.restir_gbuffer_pos.as_entire_binding(),
                },
                // ReSTIR settings (uniforms/toggles)
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: self.restir_settings.as_entire_binding(),
                },
                // ReSTIR material-id buffer (RW in shading)
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: self.restir_gbuffer_mat.as_entire_binding(),
                },
                // Instances buffer (read-only)
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: self.instances_buffer.as_entire_binding(),
                },
                // BLAS descriptor table (read-only)
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: self.blas_descs.as_entire_binding(),
                },
                // AOV albedo (RW by shading at primary hit)
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: self.aov_albedo.as_entire_binding(),
                },
                // AOV depth (RW by shading at primary hit)
                wgpu::BindGroupEntry {
                    binding: 17,
                    resource: self.aov_depth.as_entire_binding(),
                },
                // AOV normal (RW by shading at primary hit)
                wgpu::BindGroupEntry {
                    binding: 18,
                    resource: self.aov_normal.as_entire_binding(),
                },
                // Medium params (uniform)
                wgpu::BindGroupEntry {
                    binding: 19,
                    resource: self.medium_params.as_entire_binding(),
                },
                // Hair segments (RO)
                wgpu::BindGroupEntry {
                    binding: 20,
                    resource: self.hair_segments.as_entire_binding(),
                },
            ],
        });
        Ok(bind_group)
    }

    /// Set participating media parameters: (g, sigma_t, density, enable_flag)
    pub fn set_medium_params(&self, g: f32, sigma_t: f32, density: f32, enable: bool) {
        let val: [f32; 4] = [g, sigma_t, density, if enable { 1.0 } else { 0.0 }];
        self.queue
            .write_buffer(&self.medium_params, 0, bytemuck::cast_slice(&val));
    }

    /// Replace hair segments buffer (must be STORAGE | COPY_DST)
    pub fn set_hair_segments_buffer(&mut self, buffer: Buffer) {
        self.hair_segments = buffer;
    }

    /// Toggle preview of ReSTIR debug AOV in shading
    pub fn set_restir_debug_aov_mode(&self, enabled: bool) {
        let val: [u32; 4] = [if enabled { 1 } else { 0 }, 0, 0, 0];
        self.queue
            .write_buffer(&self.restir_settings, 0, bytemuck::cast_slice(&val));
    }

    /// Set QMC mode (0=off/default, 1=sobol) in RestirSettings
    pub fn set_qmc_mode(&self, mode: u32) {
        let mode_u32: [u32; 1] = [mode];
        // offset 4 bytes (second u32)
        self.queue
            .write_buffer(&self.restir_settings, 4, bytemuck::cast_slice(&mode_u32));
    }

    /// Set adaptive Russian-roulette threshold (packed as f32 in u32 bits)
    pub fn set_adaptive_rr_threshold(&self, threshold: f32) {
        let bits: u32 = threshold.to_bits();
        let arr: [u32; 1] = [bits];
        // offset 8 bytes (third u32)
        self.queue
            .write_buffer(&self.restir_settings, 8, bytemuck::cast_slice(&arr));
    }

    /// Set adaptive SPP clamp (third u32) for raygen when using QMC (P6)
    /// If limit == 0, the shader will use uniforms.spp as-is.
    pub fn set_adaptive_spp_limit(&self, limit: u32) {
        let arr: [u32; 1] = [limit];
        self.queue
            .write_buffer(&self.restir_settings, 8, bytemuck::cast_slice(&arr));
    }

    /// Set/replace the instances buffer used for TLAS-style instancing (binding 14)
    pub fn set_instances_buffer(&mut self, buffer: Buffer) {
        self.instances_buffer = buffer;
    }

    /// Upload a list of object-to-world transforms as instances (binding 14).
    /// Each instance stores object_to_world and world_to_object (inverse) matrices.
    /// NOTE: After calling this, recreate the scene bind group via create_scene_bind_group()
    /// so the new buffer is picked up by the shader dispatch.
    pub fn upload_instances(&mut self, transforms: &[Mat4]) {
        if transforms.is_empty() {
            return;
        }
        // Back-compat: build InstanceData with default blas_index=0, material_id=0
        let mut inst: Vec<crate::accel::instancing::InstanceData> =
            Vec::with_capacity(transforms.len());
        for m in transforms {
            let inv = m.inverse();
            inst.push(crate::accel::instancing::InstanceData {
                transform: m.to_cols_array(),
                inv_transform: inv.to_cols_array(),
                blas_index: 0,
                material_id: 0,
                _padding: [0; 2],
            });
        }
        self.upload_instances_data(&inst);
    }

    /// Upload instances with explicit material/blAS indices.
    /// Convenience wrapper that accepts tuples of (transform, blas_index, material_id).
    /// NOTE: After calling this, recreate the scene bind group via create_scene_bind_group().
    pub fn upload_instances_with_meta(&mut self, items: &[(Mat4, u32, u32)]) {
        if items.is_empty() {
            return;
        }
        let mut inst: Vec<crate::accel::instancing::InstanceData> = Vec::with_capacity(items.len());
        for (m, blas_index, material_id) in items.iter().copied() {
            let inv = m.inverse();
            inst.push(crate::accel::instancing::InstanceData {
                transform: m.to_cols_array(),
                inv_transform: inv.to_cols_array(),
                blas_index,
                material_id,
                _padding: [0; 2],
            });
        }
        self.upload_instances_data(&inst);
    }

    /// Upload pre-built InstanceData array (matches WGSL Instance layout)
    /// NOTE: After calling this, recreate the scene bind group via create_scene_bind_group().
    pub fn upload_instances_data(&mut self, instances: &[crate::accel::instancing::InstanceData]) {
        if instances.is_empty() {
            return;
        }
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("instances-buffer"),
                contents: bytemuck::cast_slice(instances),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        self.instances_buffer = buffer;
    }

    /// Set/replace the BLAS descriptor table buffer (binding 15)
    pub fn set_blas_descs_buffer(&mut self, buffer: Buffer) {
        self.blas_descs = buffer;
    }

    /// Create accumulation bind group (Group 3) for the HDR accumulation buffer
    pub fn create_accum_bind_group(&self, accum_buffer: &Buffer) -> BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("accum-bind-group"),
            layout: &self.pipelines.accum_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: accum_buffer.as_entire_binding(),
            }],
        })
    }

    /// Dispatch AO-from-AOVs compute pass into ao_out buffer (vec4<f32> per pixel; AO in .x)
    pub fn dispatch_ao_from_aovs(
        &self,
        encoder: &mut CommandEncoder,
        samples: u32,
        intensity: f32,
        bias: f32,
        seed: u32,
        ao_out: &Buffer,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Build AO uniforms [width, height, samples, intensity, bias, seed, pad]
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct AOUniforms {
            width: u32,
            height: u32,
            samples: u32,
            intensity: f32,
            bias: f32,
            seed: u32,
            _pad0: u32,
        }
        let u = AOUniforms {
            width: self.width,
            height: self.height,
            samples,
            intensity,
            bias,
            seed,
            _pad0: 0,
        };
        let ubuf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ao-uniforms"),
                contents: bytemuck::bytes_of(&u),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ao-bind-group"),
            layout: &self.pipelines.ao_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.aov_depth.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.aov_normal.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: ao_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: ubuf.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ao-from-aovs-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipelines.ao_compute);
        pass.set_bind_group(0, &bg, &[]);
        let wg_x = (self.width + 15) / 16;
        let wg_y = (self.height + 15) / 16;
        pass.dispatch_workgroups(wg_x, wg_y, 1);
        Ok(())
    }

    /// Render a frame using internal pipelines without requiring a Scene/TracerParams
    pub fn render_frame_simple(
        &mut self,
        uniforms_buffer: &Buffer,
        scene_bind_group: &BindGroup,
        accum_bind_group: &BindGroup,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wavefront-frame-simple"),
            });

        // Reset queue counters for new frame
        self.queue_buffers.reset_counters(&self.queue, &mut encoder);

        // Optionally run ReSTIR passes (disabled by default)
        if self.restir_enabled {
            self.dispatch_restir_init(&mut encoder, uniforms_buffer, scene_bind_group)?;
            // Temporal stage; do not swap yet to avoid binding conflicts in spatial
            self.dispatch_restir_temporal(&mut encoder, uniforms_buffer, scene_bind_group)?;
            if self.restir_spatial_enabled {
                self.dispatch_restir_spatial(&mut encoder, uniforms_buffer, scene_bind_group)?;
                {
                    use std::mem::swap;
                    swap(&mut (self.restir_prev), &mut (self.restir_out));
                }
            }
        }

        // Raygen
        self.dispatch_raygen(
            &mut encoder,
            uniforms_buffer,
            scene_bind_group,
            accum_bind_group,
        )?;

        // Wavefront loop: execute at least one iteration so primary rays get processed.
        let max_iterations = MAX_DEPTH * 2;
        let mut did_any = false;
        for iteration in 0..max_iterations {
            let ray_count =
                self.queue_buffers
                    .get_active_ray_count(&self.device, &self.queue, &mut encoder)?;
            if ray_count == 0 && did_any {
                break;
            }

            self.dispatch_intersect(&mut encoder, uniforms_buffer, scene_bind_group)?;
            self.dispatch_shade(
                &mut encoder,
                uniforms_buffer,
                scene_bind_group,
                accum_bind_group,
            )?;
            self.dispatch_shadow(
                &mut encoder,
                uniforms_buffer,
                scene_bind_group,
                accum_bind_group,
            )?;
            self.dispatch_scatter(
                &mut encoder,
                uniforms_buffer,
                scene_bind_group,
                accum_bind_group,
            )?;

            did_any = true;
            if iteration % 2 == 0 && iteration > 0 {
                self.dispatch_compact(&mut encoder)?;
            }
        }

        let command_buffer = encoder.finish();
        self.queue.submit(std::iter::once(command_buffer));
        self.frame_index += 1;
        Ok(())
    }
}
