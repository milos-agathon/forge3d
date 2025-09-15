// A2-BEGIN:terrain-module
#![allow(dead_code)]

// T11-BEGIN:terrain-mesh-mod
pub mod mesh;
pub use mesh::{make_grid, GridMesh, GridVertex, Indices};
// T11-END:terrain-mesh-mod

// T33-BEGIN:terrain-mod
pub mod pipeline;
pub use pipeline::TerrainPipeline;
// T33-END:terrain-mod

// B11-BEGIN:tiling-mod
pub mod tiling;
pub use tiling::{
    CacheStats, Frustum, QuadTreeNode, TileBounds, TileCache, TileData, TileId, TilingSystem,
};
// B11-END:tiling-mod

// B12-BEGIN:lod-mod
pub mod lod;
pub use lod::{
    calculate_triangle_reduction, screen_space_error, select_lod_for_tile, LodConfig,
    ScreenSpaceError,
};
// B12-END:lod-mod

// B13/B14-BEGIN:analysis-mod
pub mod analysis;
pub use analysis::{
    contour_extract, slope_aspect_compute, ContourPolyline, ContourResult, SlopeAspect,
};
// B13/B14-END:analysis-mod

use numpy::IntoPyArray;
use pyo3::prelude::*;
use std::num::NonZeroU32;
use wgpu::util::DeviceExt;

// B16-BEGIN: Light data structures
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PointLight {
    pub position: [f32; 3],
    pub _pad0: f32,
    pub color: [f32; 3],
    pub intensity: f32,
    pub radius: f32,
    pub _pad1: [f32; 3],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SpotLight {
    pub position: [f32; 3],
    pub _pad0: f32,
    pub direction: [f32; 3],
    pub _pad1: f32,
    pub color: [f32; 3],
    pub intensity: f32,
    pub radius: f32,
    pub inner_cone: f32,
    pub outer_cone: f32,
    pub _pad2: f32,
}

impl Default for PointLight {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            _pad0: 0.0,
            color: [1.0, 1.0, 1.0],
            intensity: 1.0,
            radius: 10.0,
            _pad1: [0.0; 3],
        }
    }
}

impl Default for SpotLight {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            _pad0: 0.0,
            direction: [0.0, -1.0, 0.0], // Point downward
            _pad1: 0.0,
            color: [1.0, 1.0, 1.0],
            intensity: 1.0,
            radius: 10.0,
            inner_cone: 0.2, // ~11.5 degrees
            outer_cone: 0.4, // ~23 degrees
            _pad2: 0.0,
        }
    }
}
// B16-END: Light data structures

// T33-BEGIN:colormap-imports
use crate::colormap::{
    decode_png_rgba8, map_name_to_type, to_linear_u8_rgba, ColormapType, SUPPORTED,
};
// T33-END:colormap-imports

// B15-BEGIN:memory-integration
use crate::core::memory_tracker::{global_tracker, is_host_visible_usage};
// B15-END:memory-integration

// ---------- Colormaps ----------

pub struct ColormapLUT {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    pub format: wgpu::TextureFormat,
}

impl ColormapLUT {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        adapter: &wgpu::Adapter,
        which: ColormapType,
    ) -> Result<(Self, &'static str), Box<dyn std::error::Error>> {
        let name = match which {
            ColormapType::Viridis => "viridis",
            ColormapType::Magma => "magma",
            ColormapType::Terrain => "terrain",
        };

        // R2a: Runtime format selection
        let force_unorm = std::env::var_os("VF_FORCE_LUT_UNORM").is_some();
        let srgb_ok = adapter
            .get_texture_format_features(wgpu::TextureFormat::Rgba8UnormSrgb)
            .allowed_usages
            .contains(wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
        let use_srgb = !force_unorm && srgb_ok;

        let (format, format_name, palette) = if use_srgb {
            // Use sRGB format with PNG bytes as-is
            let palette = decode_png_rgba8(name)
                .map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))?;
            (
                wgpu::TextureFormat::Rgba8UnormSrgb,
                "Rgba8UnormSrgb",
                palette,
            )
        } else {
            // Use UNORM format with CPU-linearized bytes
            let srgb_palette = decode_png_rgba8(name)
                .map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))?;
            let palette = to_linear_u8_rgba(&srgb_palette);
            (wgpu::TextureFormat::Rgba8Unorm, "Rgba8Unorm", palette)
        };

        // 256×1 RGBA8
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("colormap-lut"),
            size: wgpu::Extent3d {
                width: 256,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &palette,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(NonZeroU32::new(256 * 4).unwrap().into()),
                rows_per_image: Some(NonZeroU32::new(1).unwrap().into()),
            },
            wgpu::Extent3d {
                width: 256,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("colormap-lut-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        Ok((
            Self {
                texture: tex,
                view,
                sampler,
                format,
            },
            format_name,
        ))
    }

    /// Create a multi-palette LUT supporting runtime palette selection
    /// Creates a 256×N texture where N is the number of palettes
    pub fn new_multi_palette(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        adapter: &wgpu::Adapter,
        palette_names: &[&str],
    ) -> Result<(Self, &'static str), Box<dyn std::error::Error>> {
        if palette_names.is_empty() {
            return Err("At least one palette must be specified".into());
        }

        // R2a: Runtime format selection
        let force_unorm = std::env::var_os("VF_FORCE_LUT_UNORM").is_some();
        let srgb_ok = adapter
            .get_texture_format_features(wgpu::TextureFormat::Rgba8UnormSrgb)
            .allowed_usages
            .contains(wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
        let use_srgb = !force_unorm && srgb_ok;

        let height = palette_names.len() as u32;
        let format = if use_srgb {
            wgpu::TextureFormat::Rgba8UnormSrgb
        } else {
            wgpu::TextureFormat::Rgba8Unorm
        };
        let format_name = if use_srgb {
            "Rgba8UnormSrgb"
        } else {
            "Rgba8Unorm"
        };

        // Create 256×N texture for N palettes
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("colormap-lut-multi"),
            size: wgpu::Extent3d {
                width: 256,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Create combined palette data
        let mut combined_data = Vec::with_capacity(256 * height as usize * 4);

        for &palette_name in palette_names {
            let _palette_type = map_name_to_type(palette_name)?;

            let palette_data = if use_srgb {
                // Use sRGB format with PNG bytes as-is
                decode_png_rgba8(palette_name).map_err(|e| {
                    Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, e))
                })?
            } else {
                // Use UNORM format with CPU-linearized bytes
                let srgb_palette = decode_png_rgba8(palette_name).map_err(|e| {
                    Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, e))
                })?;
                to_linear_u8_rgba(&srgb_palette)
            };

            if palette_data.len() != 256 * 4 {
                return Err(format!(
                    "Invalid palette size for {}: expected 1024 bytes, got {}",
                    palette_name,
                    palette_data.len()
                )
                .into());
            }

            combined_data.extend_from_slice(&palette_data);
        }

        // Upload all palette rows at once
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &combined_data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(NonZeroU32::new(256 * 4).unwrap().into()),
                rows_per_image: Some(NonZeroU32::new(height).unwrap().into()),
            },
            wgpu::Extent3d {
                width: 256,
                height,
                depth_or_array_layers: 1,
            },
        );

        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("colormap-lut-multi-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Ok((
            Self {
                texture: tex,
                view,
                sampler,
                format,
            },
            format_name,
        ))
    }

    /// Get the number of palette rows in this LUT
    pub fn palette_count(&self) -> u32 {
        self.texture.height()
    }
}

// ---------- Uniforms (std140-compatible, 176 bytes to match WGSL) ----------

// Updated TerrainUniforms to match WGSL shader layout (176 bytes)
#[repr(C, align(16))]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TerrainUniforms {
    pub view: [[f32; 4]; 4],          // 64 B
    pub proj: [[f32; 4]; 4],          // 64 B
    pub sun_exposure: [f32; 4],       // (sun_dir.x, sun_dir.y, sun_dir.z, exposure) (16 B)
    pub spacing_h_exag_pad: [f32; 4], // (dx, dy, h_exag, palette_index) (16 B)
    pub _pad_tail: [f32; 4],          // keep total 176 bytes (16 B)
}

// Compile-time size and alignment checks
const _: () = assert!(::std::mem::size_of::<TerrainUniforms>() == 176);
const _: () = assert!(::std::mem::align_of::<TerrainUniforms>() == 16);

impl TerrainUniforms {
    /// Note: `h_range = max - min` (pass a single range, not min/max separately)
    pub fn new(
        view: glam::Mat4,
        proj: glam::Mat4,
        sun_dir: glam::Vec3,
        exposure: f32,
        spacing: f32,
        h_range: f32,
        exaggeration: f32,
    ) -> Self {
        Self {
            view: view.to_cols_array_2d(),
            proj: proj.to_cols_array_2d(),
            sun_exposure: [sun_dir.x, sun_dir.y, sun_dir.z, exposure],
            spacing_h_exag_pad: [spacing, h_range, exaggeration, 0.0], // Default palette_index = 0
            _pad_tail: [0.0; 4], // Initialize tail padding to zero
        }
    }

    /// Create TerrainUniforms with explicit palette selection
    pub fn new_with_palette(
        view: glam::Mat4,
        proj: glam::Mat4,
        sun_dir: glam::Vec3,
        exposure: f32,
        spacing: f32,
        h_range: f32,
        exaggeration: f32,
        palette_index: u32,
    ) -> Self {
        Self {
            view: view.to_cols_array_2d(),
            proj: proj.to_cols_array_2d(),
            sun_exposure: [sun_dir.x, sun_dir.y, sun_dir.z, exposure],
            spacing_h_exag_pad: [spacing, h_range, exaggeration, palette_index as f32],
            _pad_tail: [0.0; 4], // Initialize tail padding to zero
        }
    }

    pub fn from_mvp_legacy(
        mvp: glam::Mat4,
        light: glam::Vec3,
        h_min: f32,
        h_max: f32,
        exaggeration: f32,
    ) -> Self {
        let view = glam::Mat4::IDENTITY;
        let h_range = h_max - h_min;
        Self::new(view, mvp, light, 1.0, 1.0, h_range, exaggeration)
    }

    pub fn for_rendering(
        view_matrix: glam::Mat4,
        proj_matrix: glam::Mat4,
        sun_direction: glam::Vec3,
        exposure: f32,
        terrain_spacing: f32,
        height_range: f32,
        height_exaggeration: f32,
    ) -> Self {
        Self::new(
            view_matrix,
            proj_matrix,
            sun_direction,
            exposure,
            terrain_spacing,
            height_range,
            height_exaggeration,
        )
    }

    /// T31: Pack the first 44 floats in the expected T31 lanes order for debug tests
    /// Layout: [0..15]=view, [16..31]=proj, [32..35]=sun_exposure, [36..39]=spacing/h_range/exag/0, [40..43]=pad
    /// Matrices are stored in column-major format (compatible with WGSL/GPU layout)
    pub fn to_debug_lanes_44(&self) -> [f32; 44] {
        let mut lanes = [0.0f32; 44];

        // [0..15] = view matrix (16 floats, keep in column-major format as stored)
        for col in 0..4 {
            for row in 0..4 {
                lanes[col * 4 + row] = self.view[col][row];
            }
        }

        // [16..31] = proj matrix (16 floats, keep in column-major format as stored)
        for col in 0..4 {
            for row in 0..4 {
                lanes[16 + col * 4 + row] = self.proj[col][row];
            }
        }

        // [32..35] = sun_exposure (4 floats: sun_dir.xyz, exposure)
        lanes[32..36].copy_from_slice(&self.sun_exposure);

        // [36..39] = spacing_h_exag_pad (4 floats: spacing, h_range, exaggeration, 0)
        lanes[36..40].copy_from_slice(&self.spacing_h_exag_pad);

        // [40..43] = _pad_tail (4 floats, zeroed)
        lanes[40..44].copy_from_slice(&self._pad_tail);

        lanes
    }
}

// T2.1 Global state
#[derive(Debug, Clone)]
pub struct Globals {
    pub sun_dir: glam::Vec3,
    pub exposure: f32,
    pub spacing: f32,
    pub h_min: f32,
    pub h_max: f32,
    pub exaggeration: f32,
    pub view_world_position: glam::Vec3,
    pub palette_index: u32, // L2: Palette selection index
}

impl Default for Globals {
    fn default() -> Self {
        Self {
            sun_dir: glam::Vec3::new(0.5, 0.8, 0.6).normalize(),
            exposure: 1.0,
            spacing: 1.0,
            // choose a sane range matching our analytic spike heights (~±0.5)
            h_min: -0.5,
            h_max: 0.5,
            exaggeration: 1.0,
            view_world_position: glam::Vec3::new(0.0, 0.0, 5.0), // Default camera position
            palette_index: 0,                                    // Default to first palette
        }
    }
}

impl Globals {
    pub fn to_uniforms(&self, view: glam::Mat4, proj: glam::Mat4) -> TerrainUniforms {
        let h_range = self.h_max - self.h_min;

        TerrainUniforms::new_with_palette(
            view,
            proj,
            self.sun_dir,
            self.exposure,
            self.spacing,
            h_range,
            self.exaggeration,
            self.palette_index,
        )
    }
}

// ---------- Render spike object used by tests ----------

const TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

#[pyclass(module = "_vulkan_forge", name = "TerrainSpike")]
pub struct TerrainSpike {
    width: u32,
    height: u32,
    grid: u32,

    device: wgpu::Device,
    queue: wgpu::Queue,

    // T33-BEGIN:tp-and-bgs
    tp: crate::terrain::pipeline::TerrainPipeline,
    bg0_globals: wgpu::BindGroup,
    bg1_height: wgpu::BindGroup,
    bg2_lut: wgpu::BindGroup,
    // T33-END:tp-and-bgs
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    nidx: u32,

    ubo: wgpu::Buffer,
    colormap_lut: ColormapLUT,
    lut_format: &'static str,

    color: wgpu::Texture,
    color_view: wgpu::TextureView,

    globals: Globals,
    last_uniforms: TerrainUniforms,

    // T33: optional height texture state
    height_view: Option<wgpu::TextureView>,
    height_sampler: Option<wgpu::Sampler>,

    // B11: Tiling system for large DEMs
    tiling_system: Option<TilingSystem>,
}

#[pymethods]
impl TerrainSpike {
    #[new]
    #[pyo3(text_signature = "(width, height, grid=128, colormap='viridis')")]
    pub fn new(
        width: u32,
        height: u32,
        grid: Option<u32>,
        colormap: Option<String>,
    ) -> PyResult<Self> {
        let grid = grid.unwrap_or(128).max(2);

        let colormap_name = colormap.as_deref().unwrap_or("viridis");

        // Validate colormap against central SUPPORTED list
        if !SUPPORTED.contains(&colormap_name) {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Unknown colormap '{}'. Supported: {}",
                colormap_name,
                SUPPORTED.join(", ")
            )));
        }

        let which = map_name_to_type(colormap_name).map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Unknown colormap '{}'. Supported: {}",
                colormap_name,
                SUPPORTED.join(", ")
            ))
        })?;

        // Instance/adapter/device
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No suitable GPU adapter"))?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("terrain-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        ))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Offscreen color + depth
        let color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain-color"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TEXTURE_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let color_view = color.create_view(&Default::default());

        // Shader + pipeline - using T33 shared pipeline

        // T33-BEGIN:remove-local-ubo-layout
        // Removed local, conflated layout; using shared tp.{bgl_globals,bgl_height,bgl_lut}
        // T33-END:remove-local-ubo-layout

        // T33-BEGIN:terrainspike-use-t33
        // Use shared T33 pipeline
        let tp = crate::terrain::pipeline::TerrainPipeline::create(&device, TEXTURE_FORMAT);
        // T33-END:terrainspike-use-t33

        // Mesh + uniforms
        let (vbuf, ibuf, nidx) = build_grid_xyuv(&device, grid);
        let (view, proj, light) = build_view_matrices(width, height);

        let mut globals = Globals::default();
        // R4: Seed globals.sun_dir from computed light
        globals.sun_dir = light;
        // Use globals (with h_min/h_max) -> h_range is computed inside to_uniforms()
        let uniforms = globals.to_uniforms(view, proj);

        let ubo_usage = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
        let uniform_size = std::mem::size_of::<TerrainUniforms>() as u64;

        // Runtime debug assertion to ensure uniform buffer matches WGSL expectations
        debug_assert_eq!(
            uniform_size, 176,
            "Uniform buffer size {} doesn't match WGSL expectation {}",
            uniform_size, 176
        );

        let ubo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("terrain-ubo"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: ubo_usage,
        });

        // B15: Track UBO allocation (not host-visible)
        let tracker = global_tracker();
        tracker.track_buffer_allocation(uniform_size, is_host_visible_usage(ubo_usage));

        let (lut, lut_format) = ColormapLUT::new(&device, &queue, &adapter, which)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // T33-BEGIN:bg1-height-dummy
        // Provide a tiny dummy height if the spike has none yet (keeps validation clean)
        let (hview, hsamp) = {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("dummy-height-r32f"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::bytes_of(&0.0f32),
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(std::num::NonZeroU32::new(4).unwrap().into()),
                    rows_per_image: Some(std::num::NonZeroU32::new(1).unwrap().into()),
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
            let view = tex.create_view(&Default::default());
            // T33-BEGIN:height-sampler-doc
            // NOTE on height sampling:
            // The height texture is R32Float and bound with:
            //   - sample_type = Float { filterable: false }
            //   - sampler     = SamplerBindingType::NonFiltering
            // Many backends disallow linear filtering on 32-bit float textures,
            // so we must use a *non-filtering* sampler (nearest). The sampler
            // descriptor uses NEAREST modes, matching the NonFiltering binding.
            // T33-END:height-sampler-doc
            let samp = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("dummy-height-sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                mipmap_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            });
            (view, samp)
        };
        // T33-END:bg1-height-dummy

        // T33-BEGIN:bg-build-and-cache
        // Build bind groups once from the created pipeline layouts
        let bg0_globals = tp.make_bg_globals(&device, &ubo);
        let bg1_height = tp.make_bg_height(&device, &hview, &hsamp);
        let bg2_lut = tp.make_bg_lut(&device, &lut.view, &lut.sampler);
        // T33-END:bg-build-and-cache

        Ok(Self {
            width,
            height,
            grid,
            device,
            queue,
            // T33-BEGIN:store-tp-and-bgs
            tp,
            bg0_globals,
            bg1_height,
            bg2_lut,
            // T33-END:store-tp-and-bgs
            vbuf,
            ibuf,
            nidx,
            ubo,
            colormap_lut: lut,
            lut_format,
            color,
            color_view,
            globals,
            last_uniforms: uniforms,
            height_view: Some(hview),
            height_sampler: Some(hsamp),
            // B11: Initialize tiling system as None (enabled via separate method)
            tiling_system: None,
        })
    }

    #[pyo3(text_signature = "($self, path)")]
    pub fn render_png(&mut self, path: String) -> PyResult<()> {
        // Encode pass
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("terrain-encoder"),
            });
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("terrain-rp"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.02,
                            g: 0.02,
                            b: 0.03,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            rp.set_pipeline(&self.tp.pipeline);
            // T33-BEGIN:set-bgs-0-1-2
            rp.set_bind_group(0, &self.bg0_globals, &[]);
            rp.set_bind_group(1, &self.bg1_height, &[]);
            rp.set_bind_group(2, &self.bg2_lut, &[]);
            // T33-END:set-bgs-0-1-2
            rp.set_vertex_buffer(0, self.vbuf.slice(..));
            rp.set_index_buffer(self.ibuf.slice(..), wgpu::IndexFormat::Uint32);
            rp.draw_indexed(0..self.nidx, 0, 0..1);
        }
        self.queue.submit(Some(encoder.finish()));

        // Readback → PNG
        let bytes_per_pixel = 4u32;
        let unpadded_bpr = self.width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bpr = ((unpadded_bpr + align - 1) / align) * align;

        let buf_size = (padded_bpr * self.height) as wgpu::BufferAddress;

        // B15: Check memory budget before creating host-visible readback buffer
        let tracker = global_tracker();
        tracker.check_budget(buf_size).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Memory budget exceeded during terrain readback: {}",
                e
            ))
        })?;

        let usage = wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ;
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain-readback"),
            size: buf_size,
            usage,
            mapped_at_creation: false,
        });

        // B15: Track allocation (host-visible)
        tracker.track_buffer_allocation(buf_size, is_host_visible_usage(usage));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("copy-encoder"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.color,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &readback,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(NonZeroU32::new(padded_bpr).unwrap().into()),
                    rows_per_image: Some(NonZeroU32::new(self.height).unwrap().into()),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();

        let mut pixels = Vec::with_capacity((unpadded_bpr * self.height) as usize);
        for row in 0..self.height {
            let start = (row * padded_bpr) as usize;
            let end = start + unpadded_bpr as usize;
            pixels.extend_from_slice(&data[start..end]);
        }
        drop(data);
        readback.unmap();

        // B15: Free allocation after use
        tracker.free_buffer_allocation(buf_size, is_host_visible_usage(usage));

        let img = image::RgbaImage::from_raw(self.width, self.height, pixels)
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Invalid image buffer"))?;
        img.save(path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn debug_lut_format(&self) -> &'static str {
        self.lut_format
    }

    #[pyo3(text_signature = "($self, eye, target, up, fovy_deg, znear, zfar)")]
    pub fn set_camera_look_at(
        &mut self,
        eye: (f32, f32, f32),
        target: (f32, f32, f32),
        up: (f32, f32, f32),
        fovy_deg: f32,
        znear: f32,
        zfar: f32,
    ) -> PyResult<()> {
        use crate::camera;

        // Compute aspect ratio from current framebuffer dimensions
        let aspect = self.width as f32 / self.height as f32;

        // Validate parameters using camera module validators
        let eye_vec = glam::Vec3::new(eye.0, eye.1, eye.2);
        let target_vec = glam::Vec3::new(target.0, target.1, target.2);
        let up_vec = glam::Vec3::new(up.0, up.1, up.2);

        camera::validate_camera_params(eye_vec, target_vec, up_vec, fovy_deg, znear, zfar)?;

        // Compute view and projection matrices
        let view = glam::Mat4::look_at_rh(eye_vec, target_vec, up_vec);
        let fovy_rad = fovy_deg.to_radians();
        let proj = camera::perspective_wgpu(fovy_rad, aspect, znear, zfar);

        // Build new uniforms using existing globals
        let uniforms = self.globals.to_uniforms(view, proj);

        // Write to UBO
        self.queue
            .write_buffer(&self.ubo, 0, bytemuck::bytes_of(&uniforms));

        // Store for debugging
        self.last_uniforms = uniforms;

        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn debug_uniforms_f32<'py>(
        &self,
        py: pyo3::Python<'py>,
    ) -> pyo3::PyResult<pyo3::Bound<'py, numpy::PyArray1<f32>>> {
        // Return only the first 44 floats (176 bytes) for T31 compatibility:
        // [0..15]=view, [16..31]=proj, [32..35]=sun_exposure, [36..39]=spacing/h_range/exag/0, [40..43]=pad
        let uniform_lanes = self.last_uniforms.to_debug_lanes_44();
        Ok(numpy::PyArray1::from_slice_bound(py, &uniform_lanes))
    }

    // B15: Expose memory metrics to Python
    #[pyo3(text_signature = "($self)")]
    pub fn get_memory_metrics<'py>(
        &self,
        py: pyo3::Python<'py>,
    ) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::types::PyDict>> {
        let tracker = global_tracker();
        let metrics = tracker.get_metrics();

        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("buffer_count", metrics.buffer_count)?;
        dict.set_item("texture_count", metrics.texture_count)?;
        dict.set_item("buffer_bytes", metrics.buffer_bytes)?;
        dict.set_item("texture_bytes", metrics.texture_bytes)?;
        dict.set_item("host_visible_bytes", metrics.host_visible_bytes)?;
        dict.set_item("total_bytes", metrics.total_bytes)?;
        dict.set_item("limit_bytes", metrics.limit_bytes)?;
        dict.set_item("within_budget", metrics.within_budget)?;
        dict.set_item("utilization_ratio", metrics.utilization_ratio)?;

        Ok(dict)
    }

    // B11: Enable tiled DEM system
    #[pyo3(
        text_signature = "($self, bounds_min_x, bounds_min_y, bounds_max_x, bounds_max_y, cache_capacity=4, max_lod=4)"
    )]
    pub fn enable_tiling(
        &mut self,
        bounds_min_x: f32,
        bounds_min_y: f32,
        bounds_max_x: f32,
        bounds_max_y: f32,
        cache_capacity: Option<usize>,
        max_lod: Option<u32>,
    ) -> PyResult<()> {
        use glam::Vec2;

        let root_bounds = TileBounds::new(
            Vec2::new(bounds_min_x, bounds_min_y),
            Vec2::new(bounds_max_x, bounds_max_y),
        );

        let capacity = cache_capacity.unwrap_or(4);
        let max_lod = max_lod.unwrap_or(4);
        let tile_size = Vec2::new(1000.0, 1000.0); // Default 1km tiles

        let tiling_system = TilingSystem::new(root_bounds, capacity, max_lod, tile_size);
        self.tiling_system = Some(tiling_system);

        Ok(())
    }

    // B11: Naming shim for deliverable requirement - forwards to enable_tiling()
    #[pyo3(
        text_signature = "($self, bounds_min_x, bounds_min_y, bounds_max_x, bounds_max_y, cache_capacity=4, max_lod=4)"
    )]
    pub fn set_height_tiled(
        &mut self,
        bounds_min_x: f32,
        bounds_min_y: f32,
        bounds_max_x: f32,
        bounds_max_y: f32,
        cache_capacity: Option<usize>,
        max_lod: Option<u32>,
    ) -> PyResult<()> {
        // Forward to existing implementation
        self.enable_tiling(
            bounds_min_x,
            bounds_min_y,
            bounds_max_x,
            bounds_max_y,
            cache_capacity,
            max_lod,
        )
    }

    // B11: Get visible tiles for a camera position
    #[pyo3(
        text_signature = "($self, camera_pos, camera_dir, fov_deg=45.0, aspect=1.0, near=0.1, far=1000.0)"
    )]
    pub fn get_visible_tiles(
        &mut self,
        camera_pos: (f32, f32, f32),
        camera_dir: (f32, f32, f32),
        fov_deg: Option<f32>,
        aspect: Option<f32>,
        near: Option<f32>,
        far: Option<f32>,
    ) -> PyResult<Vec<(u32, u32, u32)>> {
        use glam::Vec3;

        let tiling_system = self.tiling_system.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "Tiling system not enabled. Call enable_tiling() first.",
            )
        })?;

        let frustum = Frustum::new(
            Vec3::new(camera_pos.0, camera_pos.1, camera_pos.2),
            Vec3::new(camera_dir.0, camera_dir.1, camera_dir.2).normalize(),
            fov_deg.unwrap_or(45.0).to_radians(),
            aspect.unwrap_or(1.0),
            near.unwrap_or(0.1),
            far.unwrap_or(1000.0),
        );

        let visible_tiles = tiling_system.get_visible_tiles(&frustum);

        // Convert TileId to Python-friendly tuples (lod, x, y)
        let result: Vec<(u32, u32, u32)> = visible_tiles
            .into_iter()
            .map(|tile_id| (tile_id.lod, tile_id.x, tile_id.y))
            .collect();

        Ok(result)
    }

    // B11: Load a specific tile
    #[pyo3(text_signature = "($self, lod, x, y)")]
    pub fn load_tile(&mut self, lod: u32, x: u32, y: u32) -> PyResult<()> {
        let tiling_system = self.tiling_system.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "Tiling system not enabled. Call enable_tiling() first.",
            )
        })?;

        let tile_id = TileId::new(lod, x, y);
        tiling_system.load_tile(tile_id).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to load tile: {}", e))
        })?;

        Ok(())
    }

    // B11: Get cache statistics
    #[pyo3(text_signature = "($self)")]
    pub fn get_cache_stats<'py>(
        &self,
        py: pyo3::Python<'py>,
    ) -> PyResult<pyo3::Bound<'py, pyo3::types::PyDict>> {
        let tiling_system = self.tiling_system.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "Tiling system not enabled. Call enable_tiling() first.",
            )
        })?;

        let stats = tiling_system.get_cache_stats();
        let dict = pyo3::types::PyDict::new_bound(py);

        dict.set_item("capacity", stats.capacity)?;
        dict.set_item("current_size", stats.current_size)?;
        dict.set_item("memory_usage_bytes", stats.memory_usage_bytes)?;

        Ok(dict)
    }

    // B11: Stream and load visible tiles for a camera
    #[pyo3(
        text_signature = "($self, camera_pos, camera_dir, fov_deg=45.0, aspect=1.0, near=0.1, far=1000.0)"
    )]
    pub fn stream_visible_tiles(
        &mut self,
        camera_pos: (f32, f32, f32),
        camera_dir: (f32, f32, f32),
        fov_deg: Option<f32>,
        aspect: Option<f32>,
        near: Option<f32>,
        far: Option<f32>,
    ) -> PyResult<Vec<(u32, u32, u32)>> {
        // Get visible tiles
        let visible_tiles =
            self.get_visible_tiles(camera_pos, camera_dir, fov_deg, aspect, near, far)?;

        // Load each visible tile
        for (lod, x, y) in &visible_tiles {
            if let Err(e) = self.load_tile(*lod, *x, *y) {
                // Log error but continue with other tiles
                eprintln!(
                    "Warning: Failed to load tile ({}, {}, {}): {}",
                    lod, x, y, e
                );
            }
        }

        Ok(visible_tiles)
    }

    // B12: Calculate screen-space error for a tile
    #[pyo3(
        text_signature = "($self, tile_lod, tile_x, tile_y, camera_pos, camera_target, camera_up, fov_deg=45.0, viewport_width=1024, viewport_height=768, pixel_error_budget=2.0)"
    )]
    pub fn calculate_screen_space_error(
        &self,
        tile_lod: u32,
        tile_x: u32,
        tile_y: u32,
        camera_pos: (f32, f32, f32),
        camera_target: (f32, f32, f32),
        camera_up: (f32, f32, f32),
        fov_deg: Option<f32>,
        viewport_width: Option<u32>,
        viewport_height: Option<u32>,
        pixel_error_budget: Option<f32>,
    ) -> PyResult<(f32, f32, bool)> {
        use crate::terrain::lod::{screen_space_error, LodConfig};
        use glam::Vec3;

        let tiling_system = self.tiling_system.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "Tiling system not enabled. Call enable_tiling() first.",
            )
        })?;

        let tile_id = TileId::new(tile_lod, tile_x, tile_y);
        let root_bounds = &tiling_system.root_bounds;
        let tile_bounds =
            QuadTreeNode::calculate_bounds(root_bounds, tile_id, tiling_system.tile_size);

        let eye = Vec3::new(camera_pos.0, camera_pos.1, camera_pos.2);
        let target = Vec3::new(camera_target.0, camera_target.1, camera_target.2);
        let up = Vec3::new(camera_up.0, camera_up.1, camera_up.2);

        let view = crate::terrain::lod::create_view_matrix(eye, target, up);
        let fov_rad = fov_deg.unwrap_or(45.0).to_radians();
        let vp_width = viewport_width.unwrap_or(1024);
        let vp_height = viewport_height.unwrap_or(768);
        let aspect = vp_width as f32 / vp_height as f32;
        let proj = crate::terrain::lod::create_projection_matrix(fov_rad, aspect, 0.1, 1000.0);

        let config = LodConfig::new(
            pixel_error_budget.unwrap_or(2.0),
            vp_width,
            vp_height,
            fov_rad,
        );

        let sse = screen_space_error(&tile_bounds, tile_id, eye, view, proj, &config);

        Ok((sse.edge_length_pixels, sse.error_pixels, sse.within_budget))
    }

    // B12: Select appropriate LOD for a tile based on screen-space error
    #[pyo3(
        text_signature = "($self, base_tile_lod, base_tile_x, base_tile_y, camera_pos, camera_target, camera_up, fov_deg=45.0, viewport_width=1024, viewport_height=768, pixel_error_budget=2.0, max_lod=4)"
    )]
    pub fn select_lod_for_tile(
        &self,
        base_tile_lod: u32,
        base_tile_x: u32,
        base_tile_y: u32,
        camera_pos: (f32, f32, f32),
        camera_target: (f32, f32, f32),
        camera_up: (f32, f32, f32),
        fov_deg: Option<f32>,
        viewport_width: Option<u32>,
        viewport_height: Option<u32>,
        pixel_error_budget: Option<f32>,
        max_lod: Option<u32>,
    ) -> PyResult<(u32, u32, u32)> {
        use crate::terrain::lod::{select_lod_for_tile, LodConfig};
        use glam::Vec3;

        let tiling_system = self.tiling_system.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "Tiling system not enabled. Call enable_tiling() first.",
            )
        })?;

        let base_tile_id = TileId::new(base_tile_lod, base_tile_x, base_tile_y);
        let root_bounds = &tiling_system.root_bounds;
        let tile_bounds =
            QuadTreeNode::calculate_bounds(root_bounds, base_tile_id, tiling_system.tile_size);

        let eye = Vec3::new(camera_pos.0, camera_pos.1, camera_pos.2);
        let target = Vec3::new(camera_target.0, camera_target.1, camera_target.2);
        let up = Vec3::new(camera_up.0, camera_up.1, camera_up.2);

        let view = crate::terrain::lod::create_view_matrix(eye, target, up);
        let fov_rad = fov_deg.unwrap_or(45.0).to_radians();
        let vp_width = viewport_width.unwrap_or(1024);
        let vp_height = viewport_height.unwrap_or(768);
        let aspect = vp_width as f32 / vp_height as f32;
        let proj = crate::terrain::lod::create_projection_matrix(fov_rad, aspect, 0.1, 1000.0);

        let config = LodConfig::new(
            pixel_error_budget.unwrap_or(2.0),
            vp_width,
            vp_height,
            fov_rad,
        );

        let selected_tile = select_lod_for_tile(
            &tile_bounds,
            base_tile_id,
            eye,
            view,
            proj,
            &config,
            max_lod.unwrap_or(4),
        );

        Ok((selected_tile.lod, selected_tile.x, selected_tile.y))
    }

    // B12: Calculate triangle count reduction for LOD comparison
    #[pyo3(text_signature = "($self, full_res_tiles, lod_tiles, base_triangles_per_tile=1000)")]
    pub fn calculate_triangle_reduction(
        &self,
        full_res_tiles: Vec<(u32, u32, u32)>,
        lod_tiles: Vec<(u32, u32, u32)>,
        base_triangles_per_tile: Option<u32>,
    ) -> PyResult<f32> {
        use crate::terrain::lod::calculate_triangle_reduction;

        let full_res: Vec<TileId> = full_res_tiles
            .into_iter()
            .map(|(lod, x, y)| TileId::new(lod, x, y))
            .collect();

        let lod: Vec<TileId> = lod_tiles
            .into_iter()
            .map(|(lod, x, y)| TileId::new(lod, x, y))
            .collect();

        let base_triangles = base_triangles_per_tile.unwrap_or(1000);
        let reduction = calculate_triangle_reduction(&full_res, &lod, base_triangles);

        Ok(reduction)
    }

    // B13: Compute slope and aspect for height field
    #[pyo3(text_signature = "($self, heights, width, height, dx=1.0, dy=1.0)")]
    pub fn slope_aspect_compute<'py>(
        &self,
        py: pyo3::Python<'py>,
        heights: numpy::PyReadonlyArray1<f32>,
        width: u32,
        height: u32,
        dx: Option<f32>,
        dy: Option<f32>,
    ) -> pyo3::PyResult<(
        pyo3::Bound<'py, numpy::PyArray1<f32>>,
        pyo3::Bound<'py, numpy::PyArray1<f32>>,
    )> {
        use crate::terrain::analysis::slope_aspect_compute;
        use numpy::PyUntypedArrayMethods;

        // Validate input array
        if !heights.is_c_contiguous() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Heights array must be C-contiguous. Use np.ascontiguousarray().",
            ));
        }

        let heights_slice = heights.as_slice()?;
        let expected_len = (width * height) as usize;

        if heights_slice.len() != expected_len {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Heights array length {} does not match dimensions {}x{}={}",
                heights_slice.len(),
                width,
                height,
                expected_len
            )));
        }

        let dx = dx.unwrap_or(1.0);
        let dy = dy.unwrap_or(1.0);

        // Compute slope and aspect
        let result = slope_aspect_compute(heights_slice, width as usize, height as usize, dx, dy)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        // Extract slopes and aspects into separate arrays
        let mut slopes = Vec::with_capacity(result.len());
        let mut aspects = Vec::with_capacity(result.len());

        for sa in result {
            slopes.push(sa.slope_deg);
            aspects.push(sa.aspect_deg);
        }

        // Convert to NumPy arrays
        let slopes_arr = ndarray::Array1::from_vec(slopes);
        let aspects_arr = ndarray::Array1::from_vec(aspects);

        Ok((
            slopes_arr.into_pyarray_bound(py),
            aspects_arr.into_pyarray_bound(py),
        ))
    }

    // B14: Extract contour lines from height field
    #[pyo3(signature = (heights, width, height, /, dx=1.0, dy=1.0, *, levels))]
    pub fn contour_extract<'py>(
        &self,
        py: pyo3::Python<'py>,
        heights: numpy::PyReadonlyArray1<f32>,
        width: u32,
        height: u32,
        dx: Option<f32>,
        dy: Option<f32>,
        levels: Vec<f32>,
    ) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::types::PyDict>> {
        use crate::terrain::analysis::contour_extract;
        use numpy::PyUntypedArrayMethods;

        // Validate input array
        if !heights.is_c_contiguous() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Heights array must be C-contiguous. Use np.ascontiguousarray().",
            ));
        }

        let heights_slice = heights.as_slice()?;
        let expected_len = (width * height) as usize;

        if heights_slice.len() != expected_len {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Heights array length {} does not match dimensions {}x{}={}",
                heights_slice.len(),
                width,
                height,
                expected_len
            )));
        }

        if levels.is_empty() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "At least one contour level must be specified",
            ));
        }

        let dx = dx.unwrap_or(1.0);
        let dy = dy.unwrap_or(1.0);

        // Extract contours
        let result = contour_extract(
            heights_slice,
            width as usize,
            height as usize,
            dx,
            dy,
            &levels,
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        // Build Python result dictionary
        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("polyline_count", result.polyline_count)?;
        dict.set_item("total_points", result.total_points)?;

        // Convert polylines to Python format
        let polylines_list = pyo3::types::PyList::empty_bound(py);
        for polyline in result.polylines {
            let polyline_dict = pyo3::types::PyDict::new_bound(py);
            polyline_dict.set_item("level", polyline.level)?;

            // Convert points to NumPy array (Nx2)
            let points_flat: Vec<f32> = polyline
                .points
                .iter()
                .flat_map(|(x, y)| vec![*x, *y])
                .collect();

            if !points_flat.is_empty() {
                let points_arr =
                    ndarray::Array2::from_shape_vec((polyline.points.len(), 2), points_flat)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

                polyline_dict.set_item("points", points_arr.into_pyarray_bound(py))?;
            } else {
                // Empty points array (0x2)
                let empty_arr = ndarray::Array2::<f32>::zeros((0, 2));
                polyline_dict.set_item("points", empty_arr.into_pyarray_bound(py))?;
            }

            polylines_list.append(polyline_dict)?;
        }

        dict.set_item("polylines", polylines_list)?;

        Ok(dict)
    }
}

// ---------- Geometry (analytic spike) ----------

// T33-BEGIN:build-grid-xyuv
/// Minimal grid that matches T3.1/T3.3 vertex layout: interleaved [x, z, u, v] (Float32x4) => 16-byte stride.
fn build_grid_xyuv(device: &wgpu::Device, n: u32) -> (wgpu::Buffer, wgpu::Buffer, u32) {
    let n = n.max(2) as usize;
    let (w, h) = (n, n);

    // Domain: [-1.5, +1.5] in X and Z; we feed (x,z) into position.xy.
    let scale = 1.5f32;
    let step_x = (2.0 * scale) / (w as f32 - 1.0);
    let step_z = (2.0 * scale) / (h as f32 - 1.0);

    // Interleaved verts: [x, z, u, v]
    let mut verts = Vec::<f32>::with_capacity(w * h * 4);
    for j in 0..h {
        for i in 0..w {
            let x = -scale + i as f32 * step_x;
            let z = -scale + j as f32 * step_z;
            let u = i as f32 / (w as f32 - 1.0);
            let v = j as f32 / (h as f32 - 1.0);
            verts.extend_from_slice(&[x, z, u, v]);
        }
    }

    // Indexed triangles (CCW)
    let mut idx = Vec::<u32>::with_capacity((w - 1) * (h - 1) * 6);
    for j in 0..h - 1 {
        for i in 0..w - 1 {
            let a = (j * w + i) as u32;
            let b = (j * w + i + 1) as u32;
            let c = ((j + 1) * w + i) as u32;
            let d = ((j + 1) * w + i + 1) as u32;
            idx.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }

    use wgpu::util::DeviceExt;
    let v_usage = wgpu::BufferUsages::VERTEX;
    let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("terrain-xyuv-vbuf"),
        contents: bytemuck::cast_slice(&verts),
        usage: v_usage,
    });
    let i_usage = wgpu::BufferUsages::INDEX;
    let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("terrain-xyuv-ibuf"),
        contents: bytemuck::cast_slice(&idx),
        usage: i_usage,
    });

    // B15: Track buffer allocations (not host-visible)
    let tracker = global_tracker();
    let vbuf_size = (verts.len() * std::mem::size_of::<f32>()) as u64;
    let ibuf_size = (idx.len() * std::mem::size_of::<u32>()) as u64;
    tracker.track_buffer_allocation(vbuf_size, is_host_visible_usage(v_usage));
    tracker.track_buffer_allocation(ibuf_size, is_host_visible_usage(i_usage));
    (vbuf, ibuf, idx.len() as u32)
}
// T33-END:build-grid-xyuv

fn build_grid_mesh(device: &wgpu::Device, n: u32) -> (wgpu::Buffer, wgpu::Buffer, u32) {
    let n = n as usize;
    let w = n;
    let h = n;

    let scale = 1.5f32;
    let step_x = (2.0 * scale) / (w as f32 - 1.0);
    let step_z = (2.0 * scale) / (h as f32 - 1.0);

    let f = |x: f32, z: f32| -> f32 { (x * 1.3).sin() * 0.25 + (z * 1.1).cos() * 0.25 };

    // positions
    let mut pos = vec![0.0f32; w * h * 3];
    for j in 0..h {
        for i in 0..w {
            let x = -scale + i as f32 * step_x;
            let z = -scale + j as f32 * step_z;
            let y = f(x, z);
            let idx = (j * w + i) * 3;
            pos[idx + 0] = x;
            pos[idx + 1] = y;
            pos[idx + 2] = z;
        }
    }

    // normals via central differences
    let mut nrm = vec![0.0f32; w * h * 3];
    for j in 0..h {
        for i in 0..w {
            let i0 = if i > 0 { i - 1 } else { i };
            let i1 = if i + 1 < w { i + 1 } else { i };
            let j0 = if j > 0 { j - 1 } else { j };
            let j1 = if j + 1 < h { j + 1 } else { j };

            let p = |ii, jj| {
                let k = (jj * w + ii) * 3;
                glam::Vec3::new(pos[k], pos[k + 1], pos[k + 2])
            };
            let dx = p(i1, j) - p(i0, j);
            let dz = p(i, j1) - p(i, j0);
            let n = dz.cross(dx).normalize_or_zero();

            let k = (j * w + i) * 3;
            nrm[k] = n.x;
            nrm[k + 1] = n.y;
            nrm[k + 2] = n.z;
        }
    }

    // interleave pos + nrm
    let mut verts: Vec<f32> = Vec::with_capacity(w * h * 6);
    for k in 0..(w * h) {
        verts.extend_from_slice(&pos[k * 3..k * 3 + 3]);
        verts.extend_from_slice(&nrm[k * 3..k * 3 + 3]);
    }

    // indices
    let mut idx = Vec::<u32>::with_capacity((w - 1) * (h - 1) * 6);
    for j in 0..h - 1 {
        for i in 0..w - 1 {
            let a = (j * w + i) as u32;
            let b = (j * w + i + 1) as u32;
            let c = ((j + 1) * w + i) as u32;
            let d = ((j + 1) * w + i + 1) as u32;
            idx.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }

    let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("terrain-vbuf"),
        contents: bytemuck::cast_slice(&verts),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("terrain-ibuf"),
        contents: bytemuck::cast_slice(&idx),
        usage: wgpu::BufferUsages::INDEX,
    });
    (vbuf, ibuf, idx.len() as u32)
}

// MVP + light
fn build_view_matrices(width: u32, height: u32) -> (glam::Mat4, glam::Mat4, glam::Vec3) {
    let aspect = width as f32 / height as f32;
    let proj = crate::camera::perspective_wgpu(45f32.to_radians(), aspect, 0.1, 100.0);
    let view = glam::Mat4::look_at_rh(
        glam::Vec3::new(3.0, 2.0, 3.0),
        glam::Vec3::ZERO,
        glam::Vec3::Y,
    );
    let light = glam::Vec3::new(0.5, 1.0, 0.3).normalize();
    (view, proj, light)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};

    #[test]
    fn test_terrain_uniforms_layout() {
        // Verify TerrainUniforms struct is exactly 176 bytes as expected by WGSL shader
        assert_eq!(
            size_of::<TerrainUniforms>(),
            176,
            "TerrainUniforms size must be 176 bytes to match WGSL binding"
        );

        // Verify 16-byte alignment for std140 compatibility
        assert_eq!(
            align_of::<TerrainUniforms>(),
            16,
            "TerrainUniforms must be 16-byte aligned for std140 compatibility"
        );
    }

    #[test]
    fn test_default_proj_is_wgpu_clip() {
        // Verify that build_view_matrices uses WGPU clip space projection
        let (w, h) = (512, 384);
        let aspect = w as f32 / h as f32;
        let fovy_deg = 45.0_f32;
        let fovy_rad = fovy_deg.to_radians();
        let (znear, zfar) = (0.1, 100.0);

        let (_, proj, _) = build_view_matrices(w, h);
        let expected = crate::camera::perspective_wgpu(fovy_rad, aspect, znear, zfar);

        // Assert all 16 elements are approximately equal
        let proj_array = proj.to_cols_array();
        let expected_array = expected.to_cols_array();

        for (i, (&actual, &expected)) in proj_array.iter().zip(expected_array.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-6,
                "Element {} differs: actual={}, expected={}, diff={}",
                i,
                actual,
                expected,
                (actual - expected).abs()
            );
        }
    }
}

// A2-END:terrain-module
