// src/path_tracing/wavefront/mod.rs
// Wavefront Path Tracer: Main scheduler and orchestration
// Queue-based GPU path tracing with persistent threads and stream compaction

pub mod pipeline;
pub mod queues;

use crate::path_tracing::TracerParams;
use crate::scene::Scene;
use pipeline::WavefrontPipelines;
use queues::*;
use std::sync::Arc;
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

        Ok(Self {
            device,
            queue,
            pipelines,
            queue_buffers,
            width,
            height,
            frame_index: 0,
        })
    }

    /// Resize for new dimensions
    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), Box<dyn std::error::Error>> {
        self.width = width;
        self.height = height;
        let queue_capacity = width * height * QUEUE_CAPACITY_SCALE;
        self.queue_buffers = QueueBuffers::new(&self.device, queue_capacity)?;
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

            // Stage 4: Scatter - convert scatter rays back to regular rays
            self.dispatch_scatter(&mut encoder, uniforms_buffer, accum_bind_group)?;

            // Stage 5: Compaction - remove terminated rays (optional, every N iterations)
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

    /// Get current frame index for deterministic seeding
    pub fn frame_index(&self) -> u32 {
        self.frame_index
    }

    /// Reset frame index
    pub fn reset_frame_index(&mut self) {
        self.frame_index = 0;
    }
}
