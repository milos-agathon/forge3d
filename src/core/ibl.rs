// src/core/ibl.rs
// Image-based lighting precompute pipeline and runtime bindings
// Provides compute-based irradiance/specular integration with disk caching
// RELEVANT FILES: src/ibl_wrapper.rs, src/shaders/ibl_equirect.wgsl, src/shaders/ibl_prefilter.wgsl, src/shaders/ibl_brdf.wgsl, src/shaders/lighting.wgsl, src/terrain_renderer.rs

use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use bytemuck::{Pod, Zeroable};
use half::f16;
use log::{info, warn};
use serde::{Deserialize, Serialize};
use serde_json;
use sha2::{Digest, Sha256};
use wgpu::util::DeviceExt;

const CUBE_FACE_COUNT: u32 = 6;
const CACHE_MAGIC: &[u8; 8] = b"IBLCACHE";
const CACHE_VERSION: u32 = 1;
const COPY_ALIGNMENT: usize = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;

/// Resize HDR image data using simple box filtering (nearest neighbor for downsampling)
fn resize_hdr_data(
    src_data: &[f32],
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    channels: usize,
) -> Vec<f32> {
    let src_w = src_width as usize;
    let src_h = src_height as usize;
    let dst_w = dst_width as usize;
    let dst_h = dst_height as usize;

    let mut dst_data = Vec::with_capacity(dst_w * dst_h * channels);

    for y in 0..dst_h {
        for x in 0..dst_w {
            // Map destination pixel to source coordinates
            let src_x = (x * src_w) / dst_w;
            let src_y = (y * src_h) / dst_h;
            let src_idx = (src_y * src_w + src_x) * channels;

            // Copy pixel channels
            for c in 0..channels {
                dst_data.push(src_data[src_idx + c]);
            }
        }
    }

    dst_data
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IBLQuality {
    Low,
    Medium,
    High,
    Ultra,
}

impl Default for IBLQuality {
    fn default() -> Self {
        Self::Medium
    }
}

impl IBLQuality {
    pub fn irradiance_size(self) -> u32 {
        match self {
            Self::Low => 64,
            Self::Medium => 128,
            Self::High => 256,
            Self::Ultra => 256,
        }
    }

    pub fn specular_size(self) -> u32 {
        match self {
            Self::Low => 128,
            Self::Medium => 256,
            Self::High => 512,
            Self::Ultra => 1024,
        }
    }

    pub fn specular_mip_levels(self) -> u32 {
        match self {
            Self::Low => 5,
            Self::Medium => 6,
            Self::High => 7,
            Self::Ultra => 8,
        }
    }

    pub fn brdf_size(self) -> u32 {
        512
    }

    pub fn base_environment_size(self) -> u32 {
        self.specular_size()
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PrefilterUniforms {
    env_size: u32,
    src_width: u32,
    src_height: u32,
    face_count: u32,
    mip_level: u32,
    max_mip_levels: u32,
    sample_count: u32,
    brdf_size: u32,
    roughness: f32,
    intensity: f32,
    pad0: f32,
    pad1: f32,
}

impl PrefilterUniforms {
    fn new(base_resolution: u32, quality: IBLQuality) -> Self {
        Self {
            env_size: base_resolution,
            src_width: 0,
            src_height: 0,
            face_count: CUBE_FACE_COUNT,
            mip_level: 0,
            max_mip_levels: quality.specular_mip_levels(),
            sample_count: 1024,
            brdf_size: quality.brdf_size(),
            roughness: 0.0,
            intensity: 1.0,
            pad0: 0.0,
            pad1: 0.0,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct IblCacheMetadata {
    version: u32,
    hdr_path: String,
    hdr_width: u32,
    hdr_height: u32,
    quality: String,
    base_resolution: u32,
    irradiance_size: u32,
    specular_size: u32,
    specular_mips: u32,
    brdf_size: u32,
    created_unix_secs: u64,
    sha256: String,
}

#[derive(Debug, Clone)]
struct IblCacheConfig {
    dir: PathBuf,
    hdr_path: String,
    hdr_width: u32,
    hdr_height: u32,
    cache_key: Option<String>,
}

fn align_to(value: usize, alignment: usize) -> usize {
    ((value + alignment - 1) / alignment) * alignment
}

fn pad_image_rows(data: &[u8], width: u32, height: u32, bytes_per_pixel: usize) -> (Vec<u8>, u32) {
    let tight_bpr = bytes_per_pixel * width as usize;
    let padded_bpr = align_to(tight_bpr, COPY_ALIGNMENT);
    if padded_bpr == tight_bpr {
        return (data.to_vec(), tight_bpr as u32);
    }

    let mut padded = vec![0u8; padded_bpr * height as usize];
    for row in 0..height as usize {
        let src = row * tight_bpr;
        let dst = row * padded_bpr;
        padded[dst..dst + tight_bpr].copy_from_slice(&data[src..src + tight_bpr]);
    }

    (padded, padded_bpr as u32)
}

fn strip_image_padding(padded: &[u8], width: u32, height: u32, bytes_per_pixel: usize) -> Vec<u8> {
    let tight_bpr = bytes_per_pixel * width as usize;
    let padded_bpr = align_to(tight_bpr, COPY_ALIGNMENT);
    if padded_bpr == tight_bpr {
        return padded.to_vec();
    }

    let mut tight = vec![0u8; tight_bpr * height as usize];
    for row in 0..height as usize {
        let src = row * padded_bpr;
        let dst = row * tight_bpr;
        tight[dst..dst + tight_bpr].copy_from_slice(&padded[src..src + tight_bpr]);
    }

    tight
}

pub struct IBLRenderer {
    quality: IBLQuality,
    base_resolution: u32,

    equirect_layout: wgpu::BindGroupLayout,
    convolve_layout: wgpu::BindGroupLayout,
    brdf_layout: wgpu::BindGroupLayout,
    pbr_layout: wgpu::BindGroupLayout,

    equirect_pipeline: wgpu::ComputePipeline,
    irradiance_pipeline: wgpu::ComputePipeline,
    specular_pipeline: wgpu::ComputePipeline,
    brdf_pipeline: wgpu::ComputePipeline,

    uniforms: PrefilterUniforms,
    uniform_buffer: wgpu::Buffer,

    environment_equirect: Option<wgpu::Texture>,
    environment_cubemap: Option<wgpu::Texture>,
    environment_view: Option<wgpu::TextureView>,
    irradiance_map: Option<wgpu::Texture>,
    irradiance_view: Option<wgpu::TextureView>,
    specular_map: Option<wgpu::Texture>,
    specular_view: Option<wgpu::TextureView>,
    brdf_lut: Option<wgpu::Texture>,
    brdf_view: Option<wgpu::TextureView>,

    // M7: Optional overrides for budget fitting
    specular_size_override: Option<u32>,
    irradiance_size_override: Option<u32>,
    brdf_size_override: Option<u32>,

    env_sampler: wgpu::Sampler,
    equirect_sampler: wgpu::Sampler,
    cache: Option<IblCacheConfig>,
    pbr_bind_group: Option<wgpu::BindGroup>,
    is_initialized: bool,
}

mod brdf_lut;
mod cache;
mod constructor;
mod environment;
mod image_io;
mod irradiance;
mod prefilter;
mod runtime;

fn read_blob(reader: &mut BufReader<File>) -> Result<Vec<u8>, String> {
    let mut len_bytes = [0u8; 8];
    reader
        .read_exact(&mut len_bytes)
        .map_err(|e| format!("Failed to read blob length: {e}"))?;
    let len = u64::from_le_bytes(len_bytes) as usize;
    let mut buf = vec![0u8; len];
    reader
        .read_exact(&mut buf)
        .map_err(|e| format!("Failed to read blob: {e}"))?;
    Ok(buf)
}

fn write_blob(writer: &mut BufWriter<File>, data: &[u8]) -> Result<(), String> {
    writer
        .write_all(&(data.len() as u64).to_le_bytes())
        .map_err(|e| format!("Failed to write blob length: {e}"))?;
    writer
        .write_all(data)
        .map_err(|e| format!("Failed to write blob: {e}"))?;
    Ok(())
}

fn cubemap_data_len(base_size: u32, mip_levels: u32, bytes_per_pixel: usize) -> usize {
    let mut total = 0usize;
    for mip in 0..mip_levels {
        let size = (base_size >> mip).max(1);
        total += bytes_per_pixel * (size * size * CUBE_FACE_COUNT) as usize;
    }
    total
}
