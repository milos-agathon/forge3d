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
impl IBLRenderer {
    pub fn new(device: &wgpu::Device, quality: IBLQuality) -> Self {
        let shader_equirect = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ibl.precompute.shader.equirect"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/ibl_equirect.wgsl").into()),
        });
        let shader_prefilter = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ibl.precompute.shader.prefilter"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/ibl_prefilter.wgsl").into()),
        });
        let shader_brdf = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ibl.precompute.shader.brdf"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/ibl_brdf.wgsl").into()),
        });

        let equirect_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ibl.precompute.equirect.layout"),
            entries: &[
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
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                    },
                    count: None,
                },
            ],
        });

        let convolve_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ibl.precompute.convolve.layout"),
            entries: &[
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
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                    },
                    count: None,
                },
            ],
        });

        let brdf_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ibl.precompute.brdf.layout"),
            entries: &[
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
            ],
        });

        let pbr_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ibl.runtime.pbr.layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let equirect_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ibl.precompute.pipeline.equirect"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("ibl.precompute.layout.equirect"),
                    bind_group_layouts: &[&equirect_layout],
                    push_constant_ranges: &[],
                }),
            ),
            module: &shader_equirect,
            entry_point: "cs_equirect_to_cubemap",
        });

        let irradiance_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("ibl.precompute.pipeline.irradiance"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("ibl.precompute.layout.irradiance"),
                        bind_group_layouts: &[&convolve_layout],
                        push_constant_ranges: &[],
                    }),
                ),
                module: &shader_prefilter,
                entry_point: "cs_irradiance_convolve",
            });

        let specular_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ibl.precompute.pipeline.specular"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("ibl.precompute.layout.specular"),
                    bind_group_layouts: &[&convolve_layout],
                    push_constant_ranges: &[],
                }),
            ),
            module: &shader_prefilter,
            entry_point: "cs_specular_prefilter",
        });

        let brdf_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ibl.precompute.pipeline.brdf"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("ibl.precompute.layout.brdf"),
                    bind_group_layouts: &[&brdf_layout],
                    push_constant_ranges: &[],
                }),
            ),
            module: &shader_brdf,
            entry_point: "cs_brdf_lut",
        });

        let base_resolution = quality.base_environment_size();
        let uniforms = PrefilterUniforms::new(base_resolution, quality);
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ibl.prefilter.uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let env_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ibl.runtime.sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            lod_min_clamp: 0.0,
            lod_max_clamp: 16.0,
            ..Default::default()
        });

        // Separate sampler for equirectangular sampling: Repeat on U to avoid horizontal seam
        let equirect_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ibl.precompute.equirect.sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            lod_min_clamp: 0.0,
            lod_max_clamp: 16.0,
            ..Default::default()
        });

        Self {
            quality,
            base_resolution,
            equirect_layout,
            convolve_layout,
            brdf_layout,
            pbr_layout,
            equirect_pipeline,
            irradiance_pipeline,
            specular_pipeline,
            brdf_pipeline,
            uniforms,
            uniform_buffer,
            environment_equirect: None,
            environment_cubemap: None,
            environment_view: None,
            irradiance_map: None,
            irradiance_view: None,
            specular_map: None,
            specular_view: None,
            brdf_lut: None,
            brdf_view: None,
            specular_size_override: None,
            irradiance_size_override: None,
            brdf_size_override: None,
            env_sampler,
            equirect_sampler,
            cache: None,
            pbr_bind_group: None,
            is_initialized: false,
        }
    }
    pub fn set_base_resolution(&mut self, base_resolution: u32) {
        let safe = base_resolution.max(16);
        self.base_resolution = safe;
        self.uniforms.env_size = safe;
        self.is_initialized = false;
        self.invalidate_cache_key();
    }

    pub fn configure_cache<P: AsRef<Path>>(
        &mut self,
        dir: P,
        hdr_path: &Path,
    ) -> Result<(), String> {
        let canonical = hdr_path
            .canonicalize()
            .unwrap_or_else(|_| hdr_path.to_path_buf());
        let dir = dir.as_ref();
        if !dir.exists() {
            fs::create_dir_all(dir).map_err(|e| format!("Failed to create cache dir: {e}"))?;
        }
        self.cache = Some(IblCacheConfig {
            dir: dir.to_path_buf(),
            hdr_path: canonical.to_string_lossy().into_owned(),
            hdr_width: 0,
            hdr_height: 0,
            cache_key: None,
        });
        Ok(())
    }

    pub fn load_environment_map(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        hdr_data: &[f32],
        width: u32,
        height: u32,
    ) -> Result<(), String> {
        if width == 0 || height == 0 {
            return Err("HDR image dimensions must be positive".into());
        }
        let pixel_count = (width as usize) * (height as usize);
        if hdr_data.len() != pixel_count * 3 && hdr_data.len() != pixel_count * 4 {
            return Err(format!(
                "HDR data length {} does not match width*height*{{3|4}}",
                hdr_data.len()
            ));
        }
        let channel_count = if hdr_data.len() == pixel_count * 4 {
            4
        } else {
            3
        };

        // Clamp dimensions to GPU limits
        let max_dim = device.limits().max_texture_dimension_2d;
        let (target_width, target_height) = if width > max_dim || height > max_dim {
            let scale = (max_dim as f32 / width.max(height) as f32).min(1.0);
            let new_width = (width as f32 * scale) as u32;
            let new_height = (height as f32 * scale) as u32;
            if new_width != width || new_height != height {
                warn!(
                    "HDR image {}x{} exceeds GPU limit {}, resizing to {}x{}",
                    width, height, max_dim, new_width, new_height
                );
            }
            // Ensure both dimensions are within limits
            (
                new_width.max(1).min(max_dim),
                new_height.max(1).min(max_dim),
            )
        } else {
            // Even if original dimensions are within limits, ensure they don't exceed
            (width.min(max_dim), height.min(max_dim))
        };

        // Resize HDR data if needed
        let (resized_data, resized_width, resized_height) =
            if target_width != width || target_height != height {
                let resized = resize_hdr_data(
                    hdr_data,
                    width,
                    height,
                    target_width,
                    target_height,
                    channel_count,
                );
                (resized, target_width, target_height)
            } else {
                (hdr_data.to_vec(), width, height)
            };

        let resized_pixel_count = (resized_width as usize) * (resized_height as usize);
        let mut texels = Vec::with_capacity(resized_pixel_count * 4);
        for idx in 0..resized_pixel_count {
            let src = idx * channel_count;
            texels.push(f16::from_f32(resized_data[src]).to_bits());
            texels.push(f16::from_f32(resized_data[src + 1]).to_bits());
            texels.push(f16::from_f32(resized_data[src + 2]).to_bits());
            let alpha = if channel_count == 4 {
                resized_data[src + 3]
            } else {
                1.0
            };
            texels.push(f16::from_f32(alpha).to_bits());
        }

        // Final safety check: ensure dimensions never exceed device limits
        // This should not be necessary if resize logic works correctly, but serves as a safeguard
        let max_dim_final = device.limits().max_texture_dimension_2d;
        let final_width = resized_width.min(max_dim_final).max(1);
        let final_height = resized_height.min(max_dim_final).max(1);

        // If we need to clamp further, we need to resize the data again
        let (padded, bpr) = if final_width != resized_width || final_height != resized_height {
            warn!(
                "CRITICAL: Resized dimensions {}x{} still exceed device limit {}! Clamping to {}x{}.",
                resized_width, resized_height, max_dim_final, final_width, final_height
            );
            // Resize the data to the final clamped dimensions
            let clamped_data = resize_hdr_data(
                &resized_data,
                resized_width,
                resized_height,
                final_width,
                final_height,
                4,
            );
            // Convert to f16 and pad
            let clamped_pixel_count = (final_width as usize) * (final_height as usize);
            let mut clamped_texels = Vec::with_capacity(clamped_pixel_count * 4);
            for idx in 0..clamped_pixel_count {
                let src = idx * 4;
                clamped_texels.push(f16::from_f32(clamped_data[src]).to_bits());
                clamped_texels.push(f16::from_f32(clamped_data[src + 1]).to_bits());
                clamped_texels.push(f16::from_f32(clamped_data[src + 2]).to_bits());
                clamped_texels.push(f16::from_f32(clamped_data[src + 3]).to_bits());
            }
            let clamped_raw_bytes = bytemuck::cast_slice(&clamped_texels);
            pad_image_rows(clamped_raw_bytes, final_width, final_height, 8)
        } else {
            let raw_bytes = bytemuck::cast_slice(&texels);
            pad_image_rows(raw_bytes, resized_width, resized_height, 8)
        };

        let equirect = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ibl.environment.equirect"),
            size: wgpu::Extent3d {
                width: final_width,
                height: final_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &equirect,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &padded,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(bpr),
                rows_per_image: Some(final_height),
            },
            wgpu::Extent3d {
                width: final_width,
                height: final_height,
                depth_or_array_layers: 1,
            },
        );

        let env_size = self.base_resolution;
        let cubemap = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ibl.environment.cubemap"),
            size: wgpu::Extent3d {
                width: env_size,
                height: env_size,
                depth_or_array_layers: CUBE_FACE_COUNT,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let cubemap_view = cubemap.create_view(&wgpu::TextureViewDescriptor {
            label: Some("ibl.environment.cubemap.view"),
            format: Some(wgpu::TextureFormat::Rgba16Float),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(CUBE_FACE_COUNT),
        });

        self.uniforms.src_width = final_width;
        self.uniforms.src_height = final_height;
        self.uniforms.env_size = env_size;
        self.uniforms.face_count = CUBE_FACE_COUNT;
        self.uniforms.mip_level = 0;
        self.uniforms.roughness = 0.0;
        self.write_uniforms(queue);

        let storage_view = cubemap.create_view(&wgpu::TextureViewDescriptor {
            label: Some("ibl.environment.cubemap.storage"),
            format: Some(wgpu::TextureFormat::Rgba16Float),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(CUBE_FACE_COUNT),
        });

        let equirect_view = equirect.create_view(&wgpu::TextureViewDescriptor {
            label: Some("ibl.environment.equirect.view"),
            ..Default::default()
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ibl.precompute.equirect.bind_group"),
            layout: &self.equirect_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&equirect_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.equirect_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&storage_view),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ibl.precompute.encoder.equirect"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ibl.precompute.pass.equirect"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.equirect_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let work = 8u32;
            let groups_x = (env_size + work - 1) / work;
            let groups_y = (env_size + work - 1) / work;
            pass.dispatch_workgroups(groups_x, groups_y, CUBE_FACE_COUNT);
        }

        queue.submit(Some(encoder.finish()));

        self.environment_equirect = Some(equirect);
        self.environment_cubemap = Some(cubemap);
        self.environment_view = Some(cubemap_view);
        self.invalidate_cache_key();
        if let Some(ref mut cfg) = self.cache {
            cfg.hdr_width = width;
            cfg.hdr_height = height;
        }
        self.is_initialized = false;
        Ok(())
    }
    pub fn initialize(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<(), String> {
        if self.environment_cubemap.is_none() {
            self.create_default_environment(device, queue)?;
        }

        if self.try_load_cache(device, queue)? {
            self.create_pbr_bind_group(device);
            self.is_initialized = true;
            return Ok(());
        }

        // Cache miss - building IBL resources
        info!(
            "IBL cache miss: building irradiance_{}.cube, prefilter_mips.cube, brdf_{}.png",
            self.quality.irradiance_size(),
            self.quality.brdf_size()
        );
        self.generate_irradiance_map(device, queue)?;
        self.generate_specular_map(device, queue)?;
        self.generate_brdf_lut(device, queue)?;

        self.create_pbr_bind_group(device);
        self.is_initialized = true;

        if let Err(err) = self.write_cache(device, queue) {
            warn!("Failed to write IBL cache: {err}");
        }

        Ok(())
    }
    pub fn generate_irradiance_map(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<(), String> {
        let env_view = self
            .environment_view
            .as_ref()
            .ok_or("Environment cube not available")?;

        let size = self
            .irradiance_size_override
            .unwrap_or(self.quality.irradiance_size());
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ibl.irradiance.cubemap"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: CUBE_FACE_COUNT,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let cube_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("ibl.irradiance.cubemap.view"),
            format: Some(wgpu::TextureFormat::Rgba16Float),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(CUBE_FACE_COUNT),
        });

        let storage_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("ibl.irradiance.storage.view"),
            format: Some(wgpu::TextureFormat::Rgba16Float),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(CUBE_FACE_COUNT),
        });

        self.uniforms.env_size = size;
        // Irradiance uses fixed 128 samples (handled in shader, but set for consistency)
        self.uniforms.sample_count = 128;
        self.uniforms.mip_level = 0;
        self.uniforms.roughness = 0.0;
        self.write_uniforms(queue);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ibl.irradiance.bind_group"),
            layout: &self.convolve_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(env_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.env_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&storage_view),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ibl.irradiance.encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ibl.irradiance.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.irradiance_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let work = 8;
            let groups = (size + work - 1) / work;
            pass.dispatch_workgroups(groups, groups, CUBE_FACE_COUNT);
        }

        queue.submit(Some(encoder.finish()));

        self.irradiance_map = Some(texture);
        self.irradiance_view = Some(cube_view);
        Ok(())
    }
    pub fn generate_specular_map(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<(), String> {
        let env_view = self
            .environment_view
            .as_ref()
            .ok_or("Environment cube not available")?;

        let size = self
            .specular_size_override
            .unwrap_or(self.quality.specular_size());
        let mip_levels = self
            .uniforms
            .max_mip_levels
            .min(self.quality.specular_mip_levels())
            .max(1);

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ibl.specular.cubemap"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: CUBE_FACE_COUNT,
            },
            mip_level_count: mip_levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let cube_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("ibl.specular.cubemap.view"),
            format: Some(wgpu::TextureFormat::Rgba16Float),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(mip_levels),
            base_array_layer: 0,
            array_layer_count: Some(CUBE_FACE_COUNT),
        });

        self.uniforms.max_mip_levels = mip_levels;
        self.write_uniforms(queue);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ibl.specular.encoder"),
        });

        for mip in 0..mip_levels {
            let mip_size = (size >> mip).max(1);
            self.uniforms.env_size = mip_size;
            self.uniforms.mip_level = mip;
            // Spec: mip0=1024, mip1=512, mip2=256, ... min 64
            let sample_count = (1024u32 >> mip).max(64);
            self.uniforms.sample_count = sample_count;
            // Roughness mapping: mip = roughness^2 * (mipCount-1)
            // For prefilter, we use: roughness = sqrt(mip / (mipCount-1))
            self.uniforms.roughness = if mip_levels > 1 {
                (mip as f32 / ((mip_levels - 1) as f32)).sqrt()
            } else {
                0.0
            };
            self.write_uniforms(queue);

            let storage_view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("ibl.specular.storage.mip{mip}")),
                format: Some(wgpu::TextureFormat::Rgba16Float),
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: mip,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: Some(CUBE_FACE_COUNT),
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("ibl.specular.bind_group.mip{mip}")),
                layout: &self.convolve_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(env_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.env_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&storage_view),
                    },
                ],
            });

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("ibl.specular.pass.mip{mip}")),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.specular_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                let work = 8;
                let groups = (mip_size + work - 1) / work;
                pass.dispatch_workgroups(groups, groups, CUBE_FACE_COUNT);
            }
        }

        queue.submit(Some(encoder.finish()));

        self.specular_map = Some(texture);
        self.specular_view = Some(cube_view);
        Ok(())
    }
    pub fn generate_brdf_lut(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<(), String> {
        let size = self
            .brdf_size_override
            .unwrap_or(self.quality.brdf_size())
            .max(16);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ibl.brdf.lut"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("ibl.brdf.lut.view"),
            ..Default::default()
        });

        self.uniforms.brdf_size = size;
        // Use fixed sample count for deterministic BRDF LUT (spec requirement: no random seeds)
        self.uniforms.sample_count = 1024;
        self.write_uniforms(queue);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ibl.brdf.bind_group"),
            layout: &self.brdf_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ibl.brdf.encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ibl.brdf.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.brdf_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let work = 8;
            let groups = (size + work - 1) / work;
            pass.dispatch_workgroups(groups, groups, 1);
        }

        queue.submit(Some(encoder.finish()));

        self.brdf_lut = Some(texture);
        self.brdf_view = Some(view);
        Ok(())
    }

    // M7: Override helpers for budget fitting
    pub fn override_specular_mip_levels(&mut self, levels: u32) {
        let lv = levels.max(1);
        self.uniforms.max_mip_levels = lv;
        self.is_initialized = false;
        self.invalidate_cache_key();
    }

    pub fn override_specular_face_size(&mut self, size: u32) {
        self.specular_size_override = Some(size.max(32));
        self.is_initialized = false;
        self.invalidate_cache_key();
    }

    pub fn override_irradiance_size(&mut self, size: u32) {
        self.irradiance_size_override = Some(size.max(32));
        self.is_initialized = false;
        self.invalidate_cache_key();
    }

    pub fn override_brdf_size(&mut self, size: u32) {
        let s = size.max(16);
        self.brdf_size_override = Some(s);
        self.uniforms.brdf_size = s;
        self.is_initialized = false;
        self.invalidate_cache_key();
    }
    pub fn set_quality(&mut self, quality: IBLQuality) {
        if self.quality == quality {
            return;
        }
        self.quality = quality;
        self.base_resolution = quality.base_environment_size();
        self.uniforms.env_size = self.base_resolution;
        self.uniforms.max_mip_levels = quality.specular_mip_levels();
        self.uniforms.brdf_size = quality.brdf_size();
        self.is_initialized = false;
        self.invalidate_cache_key();
    }

    pub fn quality(&self) -> IBLQuality {
        self.quality
    }

    pub fn textures(
        &self,
    ) -> (
        Option<&wgpu::Texture>,
        Option<&wgpu::Texture>,
        Option<&wgpu::Texture>,
    ) {
        (
            self.irradiance_map.as_ref(),
            self.specular_map.as_ref(),
            self.brdf_lut.as_ref(),
        )
    }

    pub fn pbr_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.pbr_layout
    }

    pub fn pbr_bind_group(&self) -> Option<&wgpu::BindGroup> {
        self.pbr_bind_group.as_ref()
    }

    pub fn sampler(&self) -> &wgpu::Sampler {
        &self.env_sampler
    }

    pub fn is_initialized(&self) -> bool {
        self.is_initialized
    }

    fn write_uniforms(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&self.uniforms));
    }

    fn create_pbr_bind_group(&mut self, device: &wgpu::Device) {
        if let (Some(spec), Some(irr), Some(brdf)) =
            (&self.specular_view, &self.irradiance_view, &self.brdf_view)
        {
            self.pbr_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ibl.runtime.pbr.bind_group"),
                layout: &self.pbr_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(spec),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(irr),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.env_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(brdf),
                    },
                ],
            }));
        }
    }

    fn create_default_environment(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<(), String> {
        let width = 16;
        let height = 8;
        let mut data = Vec::with_capacity((width * height * 3) as usize);
        for y in 0..height {
            let v = y as f32 / (height - 1) as f32;
            for x in 0..width {
                let u = x as f32 / (width - 1) as f32;
                let color = [0.1 + 0.9 * u, 0.1 + 0.5 * (1.0 - v), 0.3 + 0.7 * v];
                data.extend_from_slice(&color);
            }
        }
        self.load_environment_map(device, queue, &data, width, height)
    }
    fn cache_key(&self) -> Option<String> {
        self.cache.as_ref().and_then(|cfg| cfg.cache_key.clone())
    }

    fn invalidate_cache_key(&mut self) {
        if let Some(ref mut cfg) = self.cache {
            cfg.cache_key = None;
        }
    }

    fn ensure_cache_key(&mut self) {
        if let Some(ref mut cfg) = self.cache {
            if cfg.cache_key.is_some() || cfg.hdr_width == 0 || cfg.hdr_height == 0 {
                return;
            }
            let mut hasher = Sha256::new();
            hasher.update(&cfg.hdr_path);
            hasher.update(cfg.hdr_width.to_le_bytes());
            hasher.update(cfg.hdr_height.to_le_bytes());
            hasher.update(self.base_resolution.to_le_bytes());
            hasher.update(self.quality.irradiance_size().to_le_bytes());
            hasher.update(self.quality.specular_size().to_le_bytes());
            hasher.update(self.quality.specular_mip_levels().to_le_bytes());
            hasher.update(self.quality.brdf_size().to_le_bytes());
            cfg.cache_key = Some(format!("{:x}", hasher.finalize()));
        }
    }

    fn cache_path(&mut self) -> Option<PathBuf> {
        self.ensure_cache_key();
        self.cache.as_ref().and_then(|cfg| {
            cfg.cache_key
                .as_ref()
                .map(|key| cfg.dir.join(format!("{key}.iblcache")))
        })
    }
    fn try_load_cache(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<bool, String> {
        let path = match self.cache_path() {
            Some(path) => path,
            None => return Ok(false),
        };
        if !path.exists() {
            return Ok(false);
        }

        let mut reader = BufReader::new(
            File::open(&path)
                .map_err(|e| format!("Failed to open IBL cache '{}': {e}", path.display()))?,
        );

        let mut magic = [0u8; 8];
        reader
            .read_exact(&mut magic)
            .map_err(|e| format!("Failed to read IBL cache magic: {e}"))?;
        if &magic != CACHE_MAGIC {
            return Err(format!("Invalid IBL cache magic in '{}'", path.display()));
        }

        let mut version_bytes = [0u8; 4];
        reader
            .read_exact(&mut version_bytes)
            .map_err(|e| format!("Failed to read IBL cache version: {e}"))?;
        let version = u32::from_le_bytes(version_bytes);
        if version != CACHE_VERSION {
            return Ok(false);
        }

        let mut meta_len_bytes = [0u8; 4];
        reader
            .read_exact(&mut meta_len_bytes)
            .map_err(|e| format!("Failed to read IBL cache metadata length: {e}"))?;
        let meta_len = u32::from_le_bytes(meta_len_bytes) as usize;
        let mut meta_buf = vec![0u8; meta_len];
        reader
            .read_exact(&mut meta_buf)
            .map_err(|e| format!("Failed to read IBL cache metadata: {e}"))?;
        let metadata: IblCacheMetadata = serde_json::from_slice(&meta_buf)
            .map_err(|e| format!("Failed to parse IBL cache metadata: {e}"))?;

        if metadata.hdr_width == 0
            || metadata.hdr_height == 0
            || metadata.irradiance_size != self.quality.irradiance_size()
            || metadata.specular_size != self.quality.specular_size()
            || metadata.specular_mips != self.quality.specular_mip_levels()
            || metadata.brdf_size != self.quality.brdf_size()
        {
            return Ok(false);
        }

        // Validate sha256 hash (spec requirement: graceful invalidate on mismatch)
        self.ensure_cache_key();
        if let Some(expected_key) = self.cache_key() {
            if metadata.sha256 != expected_key {
                warn!(
                    "IBL cache sha256 mismatch: expected {}, got {}",
                    expected_key, metadata.sha256
                );
                return Ok(false);
            }
        }

        let irradiance_bytes = read_blob(&mut reader)?;
        let specular_bytes = read_blob(&mut reader)?;
        let brdf_bytes = read_blob(&mut reader)?;

        let (irr_tex, irr_view) = Self::upload_cubemap(
            device,
            queue,
            metadata.irradiance_size,
            1,
            &irradiance_bytes,
        )?;
        self.irradiance_map = Some(irr_tex);
        self.irradiance_view = Some(irr_view);
        let (spec_tex, spec_view) = Self::upload_cubemap(
            device,
            queue,
            metadata.specular_size,
            metadata.specular_mips,
            &specular_bytes,
        )?;
        self.specular_map = Some(spec_tex);
        self.specular_view = Some(spec_view);
        let (brdf_tex, brdf_view) = Self::upload_2d(
            device,
            queue,
            metadata.brdf_size,
            metadata.brdf_size,
            &brdf_bytes,
        )?;
        self.brdf_lut = Some(brdf_tex);
        self.brdf_view = Some(brdf_view);

        let cache_size_mib = (irradiance_bytes.len() + specular_bytes.len() + brdf_bytes.len())
            as f32
            / (1024.0 * 1024.0);
        info!(
            "IBL cache hit: '{}' ({:.2} MiB) - irradiance_{}.cube, prefilter_mips.cube, brdf_{}.png",
            path.display(),
            cache_size_mib,
            metadata.irradiance_size,
            metadata.brdf_size
        );
        Ok(true)
    }
    fn write_cache(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<(), String> {
        let path = match self.cache_path() {
            Some(path) => path,
            None => return Ok(()),
        };
        let irradiance_tex = self
            .irradiance_map
            .as_ref()
            .ok_or("Irradiance texture unavailable")?;
        let specular_tex = self
            .specular_map
            .as_ref()
            .ok_or("Specular texture unavailable")?;
        let brdf_tex = self
            .brdf_lut
            .as_ref()
            .ok_or("BRDF LUT texture unavailable")?;

        let metadata = IblCacheMetadata {
            version: CACHE_VERSION,
            hdr_path: self
                .cache
                .as_ref()
                .map(|cfg| cfg.hdr_path.clone())
                .unwrap_or_default(),
            hdr_width: self.cache.as_ref().map(|c| c.hdr_width).unwrap_or(0),
            hdr_height: self.cache.as_ref().map(|c| c.hdr_height).unwrap_or(0),
            quality: format!("{:?}", self.quality),
            base_resolution: self.base_resolution,
            irradiance_size: self.quality.irradiance_size(),
            specular_size: self.quality.specular_size(),
            specular_mips: self.quality.specular_mip_levels(),
            brdf_size: self.quality.brdf_size(),
            created_unix_secs: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            sha256: self
                .cache
                .as_ref()
                .and_then(|cfg| cfg.cache_key.clone())
                .unwrap_or_default(),
        };

        let irradiance_bytes = self.download_cubemap(
            device,
            queue,
            irradiance_tex,
            self.quality.irradiance_size(),
            1,
        )?;
        let specular_bytes = self.download_cubemap(
            device,
            queue,
            specular_tex,
            self.quality.specular_size(),
            self.quality.specular_mip_levels(),
        )?;
        let brdf_bytes = self.download_2d(
            device,
            queue,
            brdf_tex,
            self.quality.brdf_size(),
            self.quality.brdf_size(),
        )?;

        let mut writer = BufWriter::new(
            File::create(&path)
                .map_err(|e| format!("Failed to create IBL cache '{}': {e}", path.display()))?,
        );

        writer
            .write_all(CACHE_MAGIC)
            .map_err(|e| format!("Failed to write IBL cache magic: {e}"))?;
        writer
            .write_all(&CACHE_VERSION.to_le_bytes())
            .map_err(|e| format!("Failed to write IBL cache version: {e}"))?;

        let meta_json = serde_json::to_vec(&metadata)
            .map_err(|e| format!("Failed to serialise metadata: {e}"))?;
        writer
            .write_all(&(meta_json.len() as u32).to_le_bytes())
            .map_err(|e| format!("Failed to write metadata length: {e}"))?;
        writer
            .write_all(&meta_json)
            .map_err(|e| format!("Failed to write metadata: {e}"))?;

        write_blob(&mut writer, &irradiance_bytes)?;
        write_blob(&mut writer, &specular_bytes)?;
        write_blob(&mut writer, &brdf_bytes)?;
        writer
            .flush()
            .map_err(|e| format!("Failed to flush cache: {e}"))?;

        info!(
            "Wrote IBL cache '{}' ({:.2} MiB)",
            path.display(),
            (irradiance_bytes.len() + specular_bytes.len() + brdf_bytes.len()) as f32
                / (1024.0 * 1024.0)
        );
        Ok(())
    }
    fn upload_cubemap(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        base_size: u32,
        mip_levels: u32,
        bytes: &[u8],
    ) -> Result<(wgpu::Texture, wgpu::TextureView), String> {
        let expected_len = cubemap_data_len(base_size, mip_levels, 8);
        if bytes.len() != expected_len {
            return Err("IBL cache cubemap payload size mismatch".into());
        }

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ibl.cache.cubemap"),
            size: wgpu::Extent3d {
                width: base_size,
                height: base_size,
                depth_or_array_layers: CUBE_FACE_COUNT,
            },
            mip_level_count: mip_levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let mut offset = 0usize;
        for mip in 0..mip_levels {
            let mip_size = (base_size >> mip).max(1);
            let stride = (mip_size * mip_size * 8) as usize;
            for face in 0..CUBE_FACE_COUNT {
                let slice = &bytes[offset..offset + stride];
                offset += stride;
                let (padded, bpr) = pad_image_rows(slice, mip_size, mip_size, 8);
                queue.write_texture(
                    wgpu::ImageCopyTexture {
                        texture: &texture,
                        mip_level: mip,
                        origin: wgpu::Origin3d {
                            x: 0,
                            y: 0,
                            z: face,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    &padded,
                    wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(bpr),
                        rows_per_image: Some(mip_size),
                    },
                    wgpu::Extent3d {
                        width: mip_size,
                        height: mip_size,
                        depth_or_array_layers: 1,
                    },
                );
            }
        }

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("ibl.cache.cubemap.view"),
            format: Some(wgpu::TextureFormat::Rgba16Float),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(mip_levels),
            base_array_layer: 0,
            array_layer_count: Some(CUBE_FACE_COUNT),
        });

        Ok((texture, view))
    }
    fn upload_2d(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        bytes: &[u8],
    ) -> Result<(wgpu::Texture, wgpu::TextureView), String> {
        let expected = (width * height * 8) as usize;
        if bytes.len() != expected {
            return Err("IBL cache BRDF payload size mismatch".into());
        }

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ibl.cache.brdf"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let (padded, bpr) = pad_image_rows(bytes, width, height, 8);
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &padded,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(bpr),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("ibl.cache.brdf.view"),
            ..Default::default()
        });
        Ok((texture, view))
    }
    fn download_cubemap(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        base_size: u32,
        mip_levels: u32,
    ) -> Result<Vec<u8>, String> {
        let bytes_per_pixel = 8usize;
        let total_len = cubemap_data_len(base_size, mip_levels, bytes_per_pixel);
        let mut result = Vec::with_capacity(total_len);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ibl.download.cubemap.encoder"),
        });

        let mut buffer_slices = Vec::new();

        for mip in 0..mip_levels {
            let mip_size = (base_size >> mip).max(1);
            let padded_row = align_to(bytes_per_pixel * mip_size as usize, COPY_ALIGNMENT);
            let padded_face = padded_row * mip_size as usize;
            let padded_mip = padded_face * CUBE_FACE_COUNT as usize;

            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("ibl.download.cubemap.buffer.mip{mip}")),
                size: padded_mip as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            for face in 0..CUBE_FACE_COUNT {
                encoder.copy_texture_to_buffer(
                    wgpu::ImageCopyTexture {
                        texture,
                        mip_level: mip,
                        origin: wgpu::Origin3d {
                            x: 0,
                            y: 0,
                            z: face,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::ImageCopyBuffer {
                        buffer: &buffer,
                        layout: wgpu::ImageDataLayout {
                            offset: (face as usize * padded_face) as u64,
                            bytes_per_row: Some(padded_row as u32),
                            rows_per_image: Some(mip_size),
                        },
                    },
                    wgpu::Extent3d {
                        width: mip_size,
                        height: mip_size,
                        depth_or_array_layers: 1,
                    },
                );
            }

            buffer_slices.push((buffer, padded_row, mip_size));
        }

        queue.submit(Some(encoder.finish()));

        for (buffer, _, _) in buffer_slices.iter() {
            buffer.slice(..).map_async(wgpu::MapMode::Read, |_| ());
        }
        device.poll(wgpu::Maintain::Wait);

        for (buffer, padded_row, mip_size) in buffer_slices.iter() {
            let data = buffer.slice(..).get_mapped_range();
            for face in 0..CUBE_FACE_COUNT as usize {
                let face_offset = face * (*padded_row as usize * *mip_size as usize);
                let face_slice =
                    &data[face_offset..face_offset + (*padded_row as usize * *mip_size as usize)];
                let tight = strip_image_padding(face_slice, *mip_size, *mip_size, 8);
                result.extend_from_slice(&tight);
            }
            drop(data);
            buffer.unmap();
        }

        Ok(result)
    }
    fn download_2d(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>, String> {
        let bytes_per_pixel = 8usize;
        let padded_row = align_to(bytes_per_pixel * width as usize, COPY_ALIGNMENT);
        let padded_total = padded_row * height as usize;

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ibl.download.brdf.buffer"),
            size: padded_total as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ibl.download.brdf.encoder"),
        });

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row as u32),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        queue.submit(Some(encoder.finish()));
        buffer.slice(..).map_async(wgpu::MapMode::Read, |_| ());
        device.poll(wgpu::Maintain::Wait);

        let data = buffer.slice(..).get_mapped_range();
        let tight = strip_image_padding(&data, width, height, bytes_per_pixel);
        buffer.unmap();
        Ok(tight)
    }
}
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
