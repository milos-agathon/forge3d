// src/accel/lbvh_gpu.rs
// GPU LBVH builder orchestrating WGSL compute pipelines for Morton code generation, radix sort, and BVH linking.
// This file exists to implement GPU-accelerated BVH construction using WGSL compute shaders with memory budget compliance.
// RELEVANT FILES:src/shaders/lbvh_*.wgsl,src/accel/types.rs,src/accel/mod.rs

use crate::accel::types::{Aabb, BuildOptions, BuildStats, BvhHandle, BvhNode, Triangle};
use crate::accel::{BvhBackend, GpuBvhData};
use anyhow::Result;
use bytemuck::{cast_slice, Pod, Zeroable};
use std::sync::Arc;
use std::time::Instant;
use wgpu::util::DeviceExt;
use wgpu::{Buffer, BufferUsages, ComputePipeline, Device, Queue};

mod buffers;
mod build;
mod morton;
mod refit;
mod setup;
mod sort;
mod sort_bitonic;
mod topology;

use buffers::GpuBuffers;

#[allow(dead_code)]
/// Uniforms for Morton code generation
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct MortonUniforms {
    prim_count: u32,
    frame_index: u32,
    _pad0: u32,
    _pad1: u32,
    world_min: [f32; 3],
    _pad2: f32,
    world_extent: [f32; 3],
    _pad3: f32,
}

#[allow(dead_code)]
/// Uniforms for radix sort passes
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct SortUniforms {
    prim_count: u32,
    pass_shift: u32,
    _pad0: u32,
    _pad1: u32,
}

#[allow(dead_code)]
/// Uniforms for BVH linking
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct LinkUniforms {
    prim_count: u32,
    node_count: u32,
    _pad0: u32,
    _pad1: u32,
}

#[allow(dead_code)]
/// GPU BVH builder with WGSL compute pipelines
pub struct GpuBvhBuilder {
    device: Arc<Device>,
    queue: Arc<Queue>,

    // Compute pipelines
    morton_pipeline: ComputePipeline,
    sort_count_pipeline: ComputePipeline,
    sort_scan_pipeline: ComputePipeline,
    sort_scatter_pipeline: ComputePipeline,
    sort_clear_pipeline: ComputePipeline,
    sort_bitonic_pipeline: ComputePipeline,
    link_nodes_pipeline: ComputePipeline,
    init_leaves_pipeline: ComputePipeline,
    refit_leaves_pipeline: ComputePipeline,
    refit_internal_pipeline: ComputePipeline,
    refit_iterative_pipeline: ComputePipeline,
}
