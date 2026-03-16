// src/path_tracing/wavefront/mod.rs
// Wavefront Path Tracer: Main scheduler and orchestration
// Queue-based GPU path tracing with persistent threads and stream compaction

pub mod pipeline;
pub mod queues;

mod aov;
mod bindings;
mod control;
mod dispatch;
mod instances;
mod render;
mod restir;

use crate::path_tracing::restir::{
    create_debug_aov_buffer, create_diag_flags_buffer, create_reservoir_buffer,
    create_restir_gbuffer, create_restir_gbuffer_pos, empty_alias_entries_buffer,
    empty_light_samples_buffer,
};
use crate::scene::Scene;
use glam::Mat4;
use pipeline::WavefrontPipelines;
use queues::*;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use wgpu::{BindGroup, Buffer, CommandEncoder, Device, Queue};

const MAX_DEPTH: u32 = 8;
const WORKGROUP_SIZE: u32 = 256;
const QUEUE_CAPACITY_SCALE: u32 = 4;

pub struct WavefrontScheduler {
    device: Arc<Device>,
    queue: Arc<Queue>,
    pipelines: WavefrontPipelines,
    queue_buffers: QueueBuffers,
    width: u32,
    height: u32,
    frame_index: u32,
    restir_enabled: bool,
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
    restir_scene_spatial_bind_group: Option<BindGroup>,
    aov_albedo: Buffer,
    aov_depth: Buffer,
    aov_normal: Buffer,
    medium_params: Buffer,
    hair_segments: Buffer,
}

impl WavefrontScheduler {
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
        let settings_init: [u32; 4] = [0, 1, f32::to_bits(0.25f32), 0];
        let restir_settings = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("restir-settings-uniform"),
            contents: bytemuck::cast_slice(&settings_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let mat_zero: Vec<u32> = vec![0u32; (width as usize) * (height as usize)];
        let restir_gbuffer_mat = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("restir-gbuffer-mat"),
            contents: bytemuck::cast_slice(&mat_zero),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

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

        let medium_init: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
        let medium_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("medium-params-uniform"),
            contents: bytemuck::cast_slice(&medium_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

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
}
