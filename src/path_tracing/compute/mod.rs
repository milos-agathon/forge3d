// src/path_tracing/compute.rs
// Minimal GPU compute path tracer implementation (A1) using WGSL kernel and wgpu.
// This exists to create the compute pipeline, allocate buffers/textures, dispatch, and read back RGBA and AOVs.
// RELEVANT FILES:src/path_tracing/mod.rs,src/shaders/pt_kernel.wgsl,python/forge3d/path_tracing.py,src/lib.rs

use std::num::NonZeroU32;

use half::f16;
use wgpu::util::DeviceExt;

use crate::core::error::RenderError;
use crate::core::gpu::{align_copy_bpr, ctx};
use crate::path_tracing::aov::{AovFrames, AovKind};
use crate::path_tracing::mesh::create_empty_mesh_buffers;

pub use super::compute_types::{Sphere, Uniforms};

mod dispatch;
mod readback;
mod render;
mod setup;

struct DispatchResources {
    pipeline: wgpu::ComputePipeline,
    bg0: wgpu::BindGroup,
    bg1: wgpu::BindGroup,
    bg2: wgpu::BindGroup,
    bg3: wgpu::BindGroup,
    bg4: wgpu::BindGroup,
    out_tex: wgpu::Texture,
    aov_frames: AovFrames,
}

pub struct PathTracerGPU;
