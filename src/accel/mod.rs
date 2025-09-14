// src/accel/mod.rs
// Acceleration structures module for forge3d - GPU-first BVH construction with CPU fallback.
// This module provides LBVH GPU construction, SAH CPU fallback, and unified API for path tracing integration.
// RELEVANT FILES:src/shaders/lbvh_*.wgsl,src/path_tracing/accel.rs,python/forge3d/path_tracing.py

pub mod types;
pub mod lbvh_gpu;
pub mod sah_cpu;

pub use types::{Aabb, BvhNode, Triangle, BuildOptions, BvhHandle};
pub use lbvh_gpu::GpuBvhBuilder;
pub use sah_cpu::CpuSahBuilder;

use anyhow::{Result, Context};
use wgpu::{Device, Queue};
use std::sync::Arc;

/// GPU context option for BVH building
#[derive(Clone)]
pub enum GpuContext {
    Available {
        device: Arc<Device>,
        queue: Arc<Queue>,
    },
    NotAvailable,
}

/// Unified BVH builder that selects GPU when available, otherwise CPU
pub struct BvhBuilder {
    gpu_builder: Option<GpuBvhBuilder>,
    cpu_builder: CpuSahBuilder,
}

impl BvhBuilder {
    /// Create a new BVH builder with optional GPU support
    pub fn new(gpu_context: GpuContext) -> Result<Self> {
        let gpu_builder = match gpu_context {
            GpuContext::Available { device, queue } => {
                Some(GpuBvhBuilder::new(device, queue)?)
            }
            GpuContext::NotAvailable => None,
        };

        let cpu_builder = CpuSahBuilder::new();

        Ok(Self {
            gpu_builder,
            cpu_builder,
        })
    }

    /// Build BVH from triangles using GPU when available, otherwise CPU
    pub fn build(&mut self, triangles: &[Triangle], options: &BuildOptions) -> Result<BvhHandle> {
        if let Some(ref mut gpu_builder) = self.gpu_builder {
            // Try GPU first
            match gpu_builder.build(triangles, options) {
                Ok(handle) => return Ok(handle),
                Err(e) => {
                    log::warn!("GPU BVH build failed, falling back to CPU: {}", e);
                }
            }
        }

        // Fallback to CPU
        self.cpu_builder.build(triangles, options)
            .context("CPU BVH build failed")
    }

    /// Refit existing BVH with updated triangle data
    pub fn refit(&mut self, handle: &mut BvhHandle, triangles: &[Triangle]) -> Result<()> {
        match &handle.backend {
            BvhBackend::Gpu(_) => {
                if let Some(ref mut gpu_builder) = self.gpu_builder {
                    gpu_builder.refit(handle, triangles)
                        .context("GPU BVH refit failed")
                } else {
                    anyhow::bail!("Cannot refit GPU BVH without GPU context")
                }
            }
            BvhBackend::Cpu(_) => {
                self.cpu_builder.refit(handle, triangles)
                    .context("CPU BVH refit failed")
            }
        }
    }

    /// Get information about available backends
    pub fn backend_info(&self) -> String {
        match &self.gpu_builder {
            Some(_) => "GPU + CPU".to_string(),
            None => "CPU only".to_string(),
        }
    }
}

/// Backend-specific BVH data
#[derive(Debug)]
pub enum BvhBackend {
    Gpu(GpuBvhData),
    Cpu(CpuBvhData),
}

/// GPU BVH data (buffers and metadata)
#[derive(Debug)]
pub struct GpuBvhData {
    pub nodes_buffer: wgpu::Buffer,
    pub indices_buffer: wgpu::Buffer,
    pub node_count: u32,
    pub primitive_count: u32,
    pub world_aabb: Aabb,
}

/// CPU BVH data (vectors and metadata)
#[derive(Debug)]
pub struct CpuBvhData {
    pub nodes: Vec<BvhNode>,
    pub indices: Vec<u32>,
    pub world_aabb: Aabb,
}

/// Convenience functions for building BVH with different backends
pub fn build_bvh(
    triangles: &[Triangle], 
    options: &BuildOptions, 
    gpu_context: GpuContext
) -> Result<BvhHandle> {
    let mut builder = BvhBuilder::new(gpu_context)?;
    builder.build(triangles, options)
}

pub fn refit_bvh(
    builder: &mut BvhBuilder,
    handle: &mut BvhHandle, 
    triangles: &[Triangle]
) -> Result<()> {
    builder.refit(handle, triangles)
}