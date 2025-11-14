// src/terrain_renderer.rs
//! TerrainRenderer - GPU pipeline for PBR+POM terrain rendering
//!
//! Implements a minimal-but-correct terrain rendering pipeline:
//! - Heightmap upload (numpy → R32Float texture)
//! - Fullscreen triangle with triplanar PBR shading
//! - Parallax Occlusion Mapping (POM) support
//! - IBL (Image-Based Lighting) integration
//! - Colormap overlay
//!
//! Memory budget: ≤512 MiB host-visible allocations
//!
//! RELEVANT FILES: src/session.rs, src/material_set.rs, src/ibl_wrapper.rs,
//! src/terrain_render_params.rs, src/shaders/terrain_pbr_pom.wgsl

use anyhow::{anyhow, Result};
use bytemuck::{Pod, Zeroable};
use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::sync::{Arc, Mutex};
use wgpu::util::DeviceExt;
use wgpu::TextureFormatFeatureFlags;

use crate::core::shadow_mapping::{CsmCascadeData, CsmUniforms};
use crate::lighting::types::{Light, LightType};
use crate::lighting::LightBuffer;
use crate::terrain_render_params::{AddressModeNative, FilterModeNative};

/// Terrain renderer implementing PBR + POM pipeline
#[pyclass(module = "forge3d._forge3d", name = "TerrainRenderer")]
pub struct TerrainRenderer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    adapter: Arc<wgpu::Adapter>,
    pipeline: Mutex<PipelineCache>,
    bind_group_layout: wgpu::BindGroupLayout,
    ibl_bind_group_layout: wgpu::BindGroupLayout,
    blit_bind_group_layout: wgpu::BindGroupLayout,
    blit_pipeline: wgpu::RenderPipeline,
    sampler_linear: wgpu::Sampler,
    color_format: wgpu::TextureFormat,
    // P0-03: Config plumbing (no shader/pipeline behavior changes)
    #[cfg(feature = "enable-renderer-config")]
    config: Arc<Mutex<crate::render::params::RendererConfig>>,
    // P1-08: Light buffer for multi-light support
    light_buffer: Arc<Mutex<LightBuffer>>,
    // Noop shadow resources for bind group at index 3 (required by pipeline layout)
    noop_shadow: NoopShadow,
}

/// Noop shadow resources for terrain_pbr_pom pipeline
/// Provides dummy shadow bind group at index 3 when shadows are not used
struct NoopShadow {
    _csm_uniform_buffer: wgpu::Buffer,
    _shadow_maps_texture: wgpu::Texture,
    shadow_maps_view: wgpu::TextureView,
    shadow_sampler: wgpu::Sampler,
    _moment_maps_texture: wgpu::Texture,
    moment_maps_view: wgpu::TextureView,
    moment_sampler: wgpu::Sampler,
    bind_group: wgpu::BindGroup,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
struct OverlayUniforms {
    params0: [f32; 4], // domain_min, inv_range, overlay_strength, offset
    params1: [f32; 4], // blend_mode, debug_mode, albedo_mode, colormap_strength
    params2: [f32; 4], // gamma, pad, pad, pad
}

impl OverlayUniforms {
    fn disabled() -> Self {
        Self {
            params0: [0.0; 4],
            params1: [0.0; 4],
            params2: [2.2, 0.0, 0.0, 0.0], // default gamma = 2.2
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn msaa_prefers_requested_when_supported() {
        assert_eq!(choose_effective_msaa(4, &[1, 2, 4]), 4);
    }

    #[test]
    fn msaa_downgrades_to_four_when_available() {
        assert_eq!(choose_effective_msaa(8, &[1, 4]), 4);
    }

    #[test]
    fn msaa_downgrades_to_best_available_non_one() {
        assert_eq!(choose_effective_msaa(8, &[1, 2]), 2);
    }

    #[test]
    fn msaa_defaults_to_one_when_needed() {
        assert_eq!(choose_effective_msaa(8, &[1]), 1);
        assert_eq!(choose_effective_msaa(8, &[]), 1);
    }

    #[test]
    fn select_effective_msaa_downgrades_8_for_rgba8unorm() {
        // Simulate adapter that only supports {1,4} for Rgba8Unorm (baseline WebGPU)
        let supported = vec![1, 4];
        let effective = choose_effective_msaa(8, &supported);
        assert_eq!(effective, 4, "MSAA 8 should downgrade to 4 for Rgba8Unorm");
    }

    #[test]
    fn select_effective_msaa_never_returns_8_for_rgba8unorm_without_support() {
        // Even if {1,2,4} are supported, 8 is not - must downgrade
        let supported = vec![1, 2, 4];
        let effective = choose_effective_msaa(8, &supported);
        assert_ne!(effective, 8);
        assert_eq!(effective, 4);
    }
}

struct OverlayBinding {
    uniform: OverlayUniforms,
    lut: Option<Arc<crate::terrain::ColormapLUT>>,
}

struct PipelineCache {
    sample_count: u32,
    pipeline: wgpu::RenderPipeline,
}

const MATERIAL_LAYER_CAPACITY: usize = 4;

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
struct IblUniforms {
    intensity: f32,
    sin_theta: f32,
    cos_theta: f32,
    specular_mip_count: f32,
}

fn gather_supported_sample_counts(
    adapter: &wgpu::Adapter,
    format: wgpu::TextureFormat,
) -> Vec<u32> {
    let features = adapter.get_texture_format_features(format);
    let mut counts = vec![1u32];

    let mut push_if_supported = |flag: TextureFormatFeatureFlags, value: u32| {
        if features.flags.contains(flag) {
            counts.push(value);
        }
    };

    push_if_supported(TextureFormatFeatureFlags::MULTISAMPLE_X2, 2);
    push_if_supported(TextureFormatFeatureFlags::MULTISAMPLE_X4, 4);
    push_if_supported(TextureFormatFeatureFlags::MULTISAMPLE_X8, 8);
    push_if_supported(TextureFormatFeatureFlags::MULTISAMPLE_X16, 16);
    counts.sort_unstable();
    counts.dedup();
    if !counts.contains(&1) {
        counts.insert(0, 1);
    }
    counts
}

fn choose_effective_msaa(requested: u32, allowed: &[u32]) -> u32 {
    if allowed.is_empty() {
        return 1;
    }
    if allowed.contains(&requested) {
        return requested;
    }
    if allowed.contains(&4) {
        return 4;
    }
    if let Some(best) = allowed.iter().copied().filter(|value| *value > 1).max() {
        return best;
    }
    1
}

/// Single authoritative MSAA selection function
///
/// For RGBA8Unorm/UnormSrgb formats, WebGPU spec guarantees {1,4} support.
/// MSAA 8 support requires TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES device feature.
/// Since we don't enable that feature by default, we conservatively assume {1,4}
/// for these formats regardless of what the adapter reports.
fn select_effective_msaa(
    requested: u32,
    color_format: wgpu::TextureFormat,
    adapter: &wgpu::Adapter,
) -> u32 {
    // For RGBA8Unorm/UnormSrgb: WebGPU baseline guarantees {1,4} support
    // Higher sample counts require TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
    // which is not enabled by default in forge3d's device creation
    let supported = if matches!(
        color_format,
        wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb
    ) {
        // Conservative baseline: assume only {1,4} for RGBA8 formats
        vec![1, 4]
    } else {
        // For other formats, query adapter support
        gather_supported_sample_counts(adapter, color_format)
    };

    choose_effective_msaa(requested, &supported)
}

/// MSAA invariant checking - asserts correctness in debug, returns error in release
#[allow(dead_code)]
struct MsaaInvariants {
    effective_msaa: u32,
    pipeline_sample_count: u32,
    color_attachment_sample_count: u32,
    has_resolve_target: bool,
    resolve_sample_count: Option<u32>,
    depth_sample_count: Option<u32>,
    readback_sample_count: u32,
}

fn assert_msaa_invariants(inv: &MsaaInvariants, color_format: wgpu::TextureFormat) -> Result<()> {
    // Invariant 1: effective_msaa ∈ {1,4} for RGBA8Unorm/UnormSrgb (baseline WebGPU)
    if matches!(
        color_format,
        wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb
    ) {
        if inv.effective_msaa != 1 && inv.effective_msaa != 4 && inv.effective_msaa != 2 {
            let msg = format!(
                "MSAA invariant violated: effective_msaa={} not in {{1,2,4}} for {:?}",
                inv.effective_msaa, color_format
            );
            debug_assert!(false, "{}", msg);
            return Err(anyhow!(msg));
        }
    }

    // Invariant 2: pipeline.multisample.count == effective_msaa
    if inv.pipeline_sample_count != inv.effective_msaa {
        let msg = format!(
            "MSAA invariant violated: pipeline_sample_count={} != effective_msaa={}",
            inv.pipeline_sample_count, inv.effective_msaa
        );
        debug_assert!(false, "{}", msg);
        return Err(anyhow!(msg));
    }

    // Invariant 3: If effective_msaa > 1, must have resolve target
    if inv.effective_msaa > 1 {
        if inv.color_attachment_sample_count != inv.effective_msaa {
            let msg = format!(
                "MSAA invariant violated: color_attachment_sample_count={} != effective_msaa={}",
                inv.color_attachment_sample_count, inv.effective_msaa
            );
            debug_assert!(false, "{}", msg);
            return Err(anyhow!(msg));
        }
        if !inv.has_resolve_target {
            let msg = "MSAA invariant violated: effective_msaa>1 but no resolve_target";
            debug_assert!(false, "{}", msg);
            return Err(anyhow!(msg));
        }
        if let Some(resolve_sc) = inv.resolve_sample_count {
            if resolve_sc != 1 {
                let msg = format!(
                    "MSAA invariant violated: resolve_target sample_count={} != 1",
                    resolve_sc
                );
                debug_assert!(false, "{}", msg);
                return Err(anyhow!(msg));
            }
        }
    }

    // Invariant 4: If effective_msaa == 1, no resolve target
    if inv.effective_msaa == 1 {
        if inv.color_attachment_sample_count != 1 {
            let msg = format!(
                "MSAA invariant violated: effective_msaa=1 but color_attachment_sample_count={}",
                inv.color_attachment_sample_count
            );
            debug_assert!(false, "{}", msg);
            return Err(anyhow!(msg));
        }
        if inv.has_resolve_target {
            let msg = "MSAA invariant violated: effective_msaa=1 but resolve_target exists";
            debug_assert!(false, "{}", msg);
            return Err(anyhow!(msg));
        }
    }

    // Invariant 5: depth_attachment.sample_count == effective_msaa (when depth enabled)
    if let Some(depth_sc) = inv.depth_sample_count {
        if depth_sc != inv.effective_msaa {
            let msg = format!(
                "MSAA invariant violated: depth_sample_count={} != effective_msaa={}",
                depth_sc, inv.effective_msaa
            );
            debug_assert!(false, "{}", msg);
            return Err(anyhow!(msg));
        }
    }

    // Invariant 6: Readback always sources a single-sample texture
    if inv.readback_sample_count != 1 {
        let msg = format!(
            "MSAA invariant violated: readback_sample_count={} != 1",
            inv.readback_sample_count
        );
        debug_assert!(false, "{}", msg);
        return Err(anyhow!(msg));
    }

    Ok(())
}

/// Structured MSAA debug logging
fn log_msaa_debug(
    adapter: &wgpu::Adapter,
    color_format: wgpu::TextureFormat,
    depth_format: Option<wgpu::TextureFormat>,
    requested_msaa: u32,
    effective_msaa: u32,
    color_target_sample_count: u32,
    resolve_target_sample_count: Option<u32>,
    depth_sample_count: Option<u32>,
    pipeline_sample_count: u32,
) {
    let info = adapter.get_info();
    log::info!(target: "msaa.debug", "╔══════════════════════════════════════════════════");
    log::info!(target: "msaa.debug", "║ MSAA Configuration Debug");
    log::info!(target: "msaa.debug", "╠══════════════════════════════════════════════════");
    log::info!(target: "msaa.debug", "║ Backend: {:?}", info.backend);
    log::info!(target: "msaa.debug", "║ Adapter: {}", info.name);
    log::info!(target: "msaa.debug", "║ Driver: {} {}", info.driver, info.driver_info);
    log::info!(target: "msaa.debug", "╠══════════════════════════════════════════════════");
    log::info!(target: "msaa.debug", "║ Color Format: {:?}", color_format);
    log::info!(target: "msaa.debug", "║ Depth Format: {:?}", depth_format);
    log::info!(target: "msaa.debug", "╠══════════════════════════════════════════════════");
    log::info!(target: "msaa.debug", "║ Requested MSAA: {}", requested_msaa);
    log::info!(target: "msaa.debug", "║ Effective MSAA: {}", effective_msaa);
    log::info!(target: "msaa.debug", "╠══════════════════════════════════════════════════");
    log::info!(target: "msaa.debug", "║ color_target.sample_count: {}", color_target_sample_count);
    log::info!(target: "msaa.debug", "║ resolve_target.sample_count: {:?}", resolve_target_sample_count);
    log::info!(target: "msaa.debug", "║ depth.sample_count: {:?}", depth_sample_count);
    log::info!(target: "msaa.debug", "║ pipeline.multisample.count: {}", pipeline_sample_count);
    log::info!(target: "msaa.debug", "╚══════════════════════════════════════════════════");
}

#[pymethods]
impl TerrainRenderer {
    /// Create a new terrain renderer
    ///
    /// Args:
    ///     session: GPU session object
    #[new]
    pub fn new(session: &crate::session::Session) -> PyResult<Self> {
        Self::new_internal(
            session.device.clone(),
            session.queue.clone(),
            session.adapter.clone(),
        )
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create TerrainRenderer: {:#}", e)))
    }

    /// P1-08: Set lights from Python dicts
    ///
    /// Args:
    ///     lights: List of light specification dicts
    ///
    /// Each light dict should have:
    ///     - `type`: "directional", "point", "spot", "area_rect", etc.
    ///     - `position` or `pos`: [x, y, z] (for non-directional lights)
    ///     - `direction` or `dir`: [x, y, z] (for directional/spot lights)
    ///     - `intensity`: float (default 1.0)
    ///     - `color` or `rgb`: [r, g, b] (default [1, 1, 1])
    ///     - `range`: float (for point/spot/area lights)
    ///     - `cone_angle`: float in degrees (for spot lights)
    ///     - `area_extent`: [width, height] (for area_rect)
    ///     - `radius`: float (for area_disk, area_sphere)
    ///
    /// Example:
    ///     renderer.set_lights([
    ///         {"type": "directional", "intensity": 3.0, "azimuth": 135, "elevation": 35},
    ///         {"type": "point", "pos": [0, 10, 0], "intensity": 10, "range": 50}
    ///     ])
    #[pyo3(signature = (lights))]
    fn set_lights(&self, py: Python, lights: &PyAny) -> PyResult<()> {
        use pyo3::types::PyList;

        let lights_list = lights
            .downcast::<PyList>()
            .map_err(|_| PyRuntimeError::new_err("lights must be a list"))?;

        // Parse all light dicts to native Light structs
        let mut native_lights = Vec::new();
        for (i, light_dict) in lights_list.iter().enumerate() {
            match crate::lighting::py_bindings::parse_light_dict(py, light_dict) {
                Ok(light) => native_lights.push(light),
                Err(e) => {
                    return Err(PyRuntimeError::new_err(format!(
                        "Failed to parse light {}: {}",
                        i, e
                    )));
                }
            }
        }

        // Update light buffer
        let mut light_buffer = self
            .light_buffer
            .lock()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to lock light buffer: {}", e)))?;

        light_buffer
            .update(&self.device, &self.queue, &native_lights)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to update lights: {}", e)))?;

        Ok(())
    }

    /// P1-09: Get debug info from light buffer
    ///
    /// Returns:
    ///     String with light buffer state (count, frame index, light details)
    ///
    /// Example:
    ///     info = renderer.light_debug_info()
    ///     print(info)
    #[pyo3(signature = ())]
    fn light_debug_info(&self) -> PyResult<String> {
        let light_buffer = self
            .light_buffer
            .lock()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to lock light buffer: {}", e)))?;

        Ok(light_buffer.debug_info())
    }

    /// Render terrain using PBR + POM
    ///
    /// Args:
    ///     material_set: Material properties for triplanar mapping
    ///     env_maps: IBL environment maps
    ///     params: Terrain render parameters
    ///     heightmap: 2D numpy array (H, W) of float32 heights
    ///     target: Optional render target (None for offscreen)
    ///
    /// Returns:
    ///     Frame object with rendered RGBA8 image
    #[pyo3(signature = (material_set, env_maps, params, heightmap, target=None))]
    pub fn render_terrain_pbr_pom<'py>(
        &self,
        py: Python<'py>,
        material_set: &crate::material_set::MaterialSet,
        env_maps: &crate::ibl_wrapper::IBL,
        params: &crate::terrain_render_params::TerrainRenderParams,
        heightmap: PyReadonlyArray2<'py, f32>,
        target: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<crate::Frame>> {
        if target.is_some() {
            return Err(PyRuntimeError::new_err(
                "Custom render targets not yet supported. Use target=None for offscreen rendering.",
            ));
        }

        // Render internally
        let frame = self
            .render_internal(material_set, env_maps, params, heightmap)
            .map_err(|e| PyRuntimeError::new_err(format!("Rendering failed: {:#}", e)))?;

        // Return Frame object
        Ok(Py::new(py, frame)?)
    }

    /// Get renderer info string
    pub fn info(&self) -> String {
        format!(
            "TerrainRenderer(backend=wgpu, device={:?})",
            self.device.features()
        )
    }

    /// Python repr
    fn __repr__(&self) -> String {
        format!("TerrainRenderer(features={:?})", self.device.features())
    }

    /// Get renderer config for debugging (P0-03)
    #[cfg(feature = "enable-renderer-config")]
    pub fn get_config(&self) -> PyResult<String> {
        let config = self
            .config
            .lock()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to lock config: {}", e)))?;
        serde_json::to_string_pretty(&*config)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to serialize config: {}", e)))
    }
}

impl TerrainRenderer {
    /// Internal constructor
    fn new_internal(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        adapter: Arc<wgpu::Adapter>,
    ) -> Result<Self> {
        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain_pbr_pom.bind_group_layout"),
            entries: &[
                // @binding(0): uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(1): heightmap texture (used in both vertex and fragment)
                // Note: R32Float may not be filterable on all hardware, so we use NonFiltering
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // @binding(2): heightmap sampler (used in both vertex and fragment)
                // Use NonFiltering for R32Float compatibility
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                // @binding(3): material albedo texture array
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                // @binding(4): material sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // @binding(5): shading uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(6): colormap texture
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // @binding(7): colormap sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // @binding(8): overlay uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let ibl_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("terrain_pbr_pom.ibl_bind_group_layout"),
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // P1-06: Empty bind group removed - light buffer will provide group(1) bindings

        // Create samplers
        // Use Nearest filtering for R32Float heightmap (non-filterable on some hardware)
        let blit_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("terrain_pbr_pom.blit_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let sampler_linear = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("terrain.sampler.nearest"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // P1-08: Initialize light buffer before pipeline creation
        let light_buffer = LightBuffer::new(&device);

        let color_format = wgpu::TextureFormat::Rgba8Unorm;
        let light_buffer_layout = light_buffer.bind_group_layout();

        // Create shadow bind group layout (reused for noop shadow and pipeline)
        let shadow_bind_group_layout = Self::create_shadow_bind_group_layout(device.as_ref());

        let pipeline = Self::create_render_pipeline(
            device.as_ref(),
            &bind_group_layout,
            light_buffer_layout,
            &ibl_bind_group_layout,
            &shadow_bind_group_layout,
            color_format,
            1,
        );
        let blit_pipeline =
            Self::create_blit_pipeline(device.as_ref(), &blit_bind_group_layout, color_format);

        // Create noop shadow resources for bind group at index 3
        let noop_shadow = Self::create_noop_shadow(device.as_ref(), &shadow_bind_group_layout)?;

        let pipeline_cache = PipelineCache {
            sample_count: 1,
            pipeline,
        };

        Ok(Self {
            device,
            queue,
            adapter,
            pipeline: Mutex::new(pipeline_cache),
            bind_group_layout,
            ibl_bind_group_layout,
            blit_bind_group_layout,
            blit_pipeline,
            sampler_linear,
            color_format,
            light_buffer: Arc::new(Mutex::new(light_buffer)),
            noop_shadow,
            // P0-03: Initialize with default config (no behavior changes)
            #[cfg(feature = "enable-renderer-config")]
            config: Arc::new(Mutex::new(crate::render::params::RendererConfig::default())),
        })
    }

    /// Preprocess terrain shader by resolving #include directives
    /// WGSL doesn't have a preprocessor, so we manually expand includes
    fn preprocess_terrain_shader() -> String {
        // Helper to strip #include lines from a shader source
        fn strip_includes(source: &str) -> String {
            source
                .lines()
                .filter(|l| !l.trim_start().starts_with("#include"))
                .collect::<Vec<_>>()
                .join("\n")
        }

        // Load nested includes for lighting.wgsl
        let lights = include_str!("shaders/lights.wgsl");

        // Load BRDF dispatch and its includes
        let brdf_common = include_str!("shaders/brdf/common.wgsl");
        let brdf_lambert = include_str!("shaders/brdf/lambert.wgsl");
        let brdf_phong = include_str!("shaders/brdf/phong.wgsl");
        let brdf_oren_nayar = include_str!("shaders/brdf/oren_nayar.wgsl");
        let brdf_cook_torrance = include_str!("shaders/brdf/cook_torrance.wgsl");
        let brdf_disney_principled = include_str!("shaders/brdf/disney_principled.wgsl");
        let brdf_ashikhmin_shirley = include_str!("shaders/brdf/ashikhmin_shirley.wgsl");
        let brdf_ward = include_str!("shaders/brdf/ward.wgsl");
        let brdf_toon = include_str!("shaders/brdf/toon.wgsl");
        let brdf_minnaert = include_str!("shaders/brdf/minnaert.wgsl");

        let brdf_dispatch_raw = include_str!("shaders/brdf/dispatch.wgsl");
        let brdf_dispatch = strip_includes(brdf_dispatch_raw);

        // Load lighting.wgsl and strip its includes
        let lighting_raw = include_str!("shaders/lighting.wgsl");
        let lighting = strip_includes(lighting_raw);

        // Load lighting_ibl.wgsl (no includes)
        let lighting_ibl = include_str!("shaders/lighting_ibl.wgsl");

        // Load main terrain shader and strip includes
        let terrain_raw = include_str!("shaders/terrain_pbr_pom.wgsl");
        let terrain = strip_includes(terrain_raw);

        // Concatenate in dependency order
        format!(
            "{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}",
            lights,
            brdf_common,
            brdf_lambert,
            brdf_phong,
            brdf_oren_nayar,
            brdf_cook_torrance,
            brdf_disney_principled,
            brdf_ashikhmin_shirley,
            brdf_ward,
            brdf_toon,
            brdf_minnaert,
            brdf_dispatch,
            lighting,
            lighting_ibl,
            terrain
        )
    }

    /// Create noop shadow resources for bind group at index 3
    /// Provides dummy shadow resources when shadows are not used (e.g., IBL rotation mode)
    fn create_noop_shadow(
        device: &wgpu::Device,
        shadow_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Result<NoopShadow> {
        use crate::core::shadow_mapping::CsmUniforms;

        // Create dummy CSM uniforms buffer (binding 0)
        let csm_uniforms = CsmUniforms {
            light_direction: [0.0, -1.0, 0.0, 0.0],
            light_view: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            cascades: [CsmCascadeData {
                light_projection: [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                near_distance: 0.1,
                far_distance: 1000.0,
                texel_size: 1.0,
                _padding: 0.0,
            }; 4],
            cascade_count: 0,
            pcf_kernel_size: 0,
            depth_bias: 0.0,
            slope_bias: 0.0,
            shadow_map_size: 1.0,
            debug_mode: 0,
            _padding: [0.0; 2],
        };
        let csm_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("terrain.noop_shadow.csm_uniforms"),
            contents: bytemuck::bytes_of(&csm_uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Create 1x1 depth texture array (binding 1: shadow_maps)
        let shadow_maps_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain.noop_shadow.maps"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_maps_view = shadow_maps_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("terrain.noop_shadow.maps.view"),
            format: Some(wgpu::TextureFormat::Depth32Float),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            aspect: wgpu::TextureAspect::DepthOnly,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
        });

        // Create comparison sampler (binding 2: shadow_sampler)
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("terrain.noop_shadow.sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        // Create 1x1 RGBA texture array (binding 3: moment_maps)
        // Must use Rgba16Float (not Rgba32Float) because the layout expects Float { filterable: true }
        // Rgba32Float is not filterable in WebGPU
        let moment_maps_format = wgpu::TextureFormat::Rgba16Float;
        let moment_maps_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain.noop_shadow.moments"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: moment_maps_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        // Verify texture format is filterable (Rgba16Float, not Rgba32Float)
        debug_assert_eq!(
            moment_maps_format,
            wgpu::TextureFormat::Rgba16Float,
            "Moment maps texture must use Rgba16Float (filterable), not Rgba32Float"
        );
        let moment_maps_view = moment_maps_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("terrain.noop_shadow.moments.view"),
            format: Some(wgpu::TextureFormat::Rgba16Float), // Explicitly set filterable format
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
        });

        // Create filtering sampler (binding 4: moment_sampler)
        let moment_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("terrain.noop_shadow.moment_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Noop shadow resources satisfy `terrain_pbr_pom.shadow_bind_group_layout` when shadows are disabled;
        // uses filterable `Rgba16Float` for float textures.
        // Sanity checks: verify we're creating the correct number of entries matching the layout
        let entries = [
            wgpu::BindGroupEntry {
                binding: 0,
                resource: csm_uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&shadow_maps_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&shadow_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&moment_maps_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Sampler(&moment_sampler),
            },
        ];

        // Debug assertions: verify entry count matches layout expectations
        // The layout should have 5 entries: buffer(0), shadow texture(1), shadow sampler(2), moment texture(3), moment sampler(4)
        debug_assert_eq!(
            entries.len(),
            5,
            "Noop shadow bind group must have 5 entries matching the layout"
        );
        // Note: View dimensions are verified by construction - shadow_maps_view and moment_maps_view
        // are both created with D2Array dimension to match the layout expectations

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain.noop_shadow.bind_group"),
            layout: shadow_bind_group_layout,
            entries: &entries,
        });

        Ok(NoopShadow {
            _csm_uniform_buffer: csm_uniform_buffer,
            _shadow_maps_texture: shadow_maps_texture,
            shadow_maps_view,
            shadow_sampler,
            _moment_maps_texture: moment_maps_texture,
            moment_maps_view,
            moment_sampler,
            bind_group,
        })
    }

    /// Create shadow bind group layout for terrain shader (group 3)
    /// Matches terrain_pbr_pom.wgsl @group(3) bindings: CSM uniforms, shadow maps, moment maps
    fn create_shadow_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        use crate::core::shadow_mapping::CsmUniforms;

        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain_pbr_pom.shadow_bind_group_layout"),
            entries: &[
                // @binding(0): CSM uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            std::num::NonZeroU64::new(std::mem::size_of::<CsmUniforms>() as u64)
                                .unwrap(),
                        ),
                    },
                    count: None,
                },
                // @binding(1): Shadow maps (texture_depth_2d_array)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        sample_type: wgpu::TextureSampleType::Depth,
                    },
                    count: None,
                },
                // @binding(2): Shadow sampler (sampler_comparison)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
                // @binding(3): Moment maps (texture_2d_array<f32>)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // @binding(4): Moment sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        })
    }

    fn create_render_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        light_buffer_layout: &wgpu::BindGroupLayout,
        ibl_bind_group_layout: &wgpu::BindGroupLayout,
        shadow_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> wgpu::RenderPipeline {
        let shader_source = Self::preprocess_terrain_shader();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain_pbr_pom.shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain_pbr_pom.pipeline_layout"),
            bind_group_layouts: &[
                bind_group_layout,         // @group(0): terrain uniforms/textures (bindings 0-8)
                light_buffer_layout,       // @group(1): lights (bindings 3-5)
                ibl_bind_group_layout,     // @group(2): IBL (bindings 0-4)
                &shadow_bind_group_layout, // @group(3): shadows (bindings 0-4)
            ],
            push_constant_ranges: &[],
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("terrain_pbr_pom.pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: sample_count,
                ..Default::default()
            },
            multiview: None,
        })
    }

    fn create_blit_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
    ) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain.blit.shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/terrain_blit.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain.blit.pipeline_layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("terrain.blit.pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        })
    }

    fn map_filter_mode(mode: FilterModeNative) -> wgpu::FilterMode {
        match mode {
            FilterModeNative::Linear => wgpu::FilterMode::Linear,
            FilterModeNative::Nearest => wgpu::FilterMode::Nearest,
        }
    }

    fn map_address_mode(mode: AddressModeNative) -> wgpu::AddressMode {
        match mode {
            AddressModeNative::Repeat => wgpu::AddressMode::Repeat,
            AddressModeNative::ClampToEdge => wgpu::AddressMode::ClampToEdge,
            AddressModeNative::MirrorRepeat => wgpu::AddressMode::MirrorRepeat,
        }
    }

    /// Internal render method
    fn render_internal(
        &self,
        material_set: &crate::material_set::MaterialSet,
        env_maps: &crate::ibl_wrapper::IBL,
        params: &crate::terrain_render_params::TerrainRenderParams,
        heightmap: PyReadonlyArray2<f32>,
    ) -> Result<crate::Frame> {
        // P1-06: Advance light buffer frame (triple-buffering)
        let mut light_buffer_guard = self
            .light_buffer
            .lock()
            .map_err(|_| anyhow!("Light buffer mutex poisoned"))?;
        light_buffer_guard.next_frame();

        // Always update light buffer with lights from params (or neutral light for rotation mode)
        // This ensures the bind group is valid for the current frame and matches the pipeline layout
        let decoded = params.decoded();
        let lights = if decoded.light.intensity > 0.0 {
            // Create directional light from params
            vec![Light {
                kind: LightType::Directional.as_u32(),
                intensity: decoded.light.intensity,
                range: 0.0,
                env_texture_index: 0,
                color: decoded.light.color,
                _pad1: 0.0,
                pos_ws: [0.0; 3],
                _pad2: 0.0,
                dir_ws: decoded.light.direction,
                _pad3: 0.0,
                cone_cos: [1.0, 1.0],
                area_half: [0.0, 0.0],
            }]
        } else {
            // Neutral light with zero intensity for rotation mode (IBL only)
            vec![Light {
                kind: LightType::Directional.as_u32(),
                intensity: 0.0,
                range: 0.0,
                env_texture_index: 0,
                color: [1.0, 1.0, 1.0],
                _pad1: 0.0,
                pos_ws: [0.0; 3],
                _pad2: 0.0,
                dir_ws: [0.0, 1.0, 0.0], // Default up direction
                _pad3: 0.0,
                cone_cos: [1.0, 1.0],
                area_half: [0.0, 0.0],
            }]
        };

        light_buffer_guard
            .update(self.device.as_ref(), self.queue.as_ref(), &lights)
            .map_err(|e| anyhow!("Failed to update light buffer: {}", e))?;
        drop(light_buffer_guard);
        // Get heightmap dimensions
        let heightmap_array = heightmap.as_array();
        let (height, width) = (heightmap_array.shape()[0], heightmap_array.shape()[1]);

        if width == 0 || height == 0 {
            return Err(anyhow!("Heightmap dimensions must be > 0"));
        }

        // Upload heightmap to GPU as R32Float texture
        let heightmap_data: Vec<f32> = heightmap_array.iter().copied().collect();
        let heightmap_texture =
            self.upload_heightmap_texture(width as u32, height as u32, &heightmap_data)?;
        let heightmap_view = heightmap_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let gpu_materials = material_set
            .gpu(self.device.as_ref(), self.queue.as_ref())
            .map_err(|err| {
                PyRuntimeError::new_err(format!("Failed to prepare material textures: {err:#}"))
            })?;
        let material_view = &gpu_materials.view;
        let material_sampler = &gpu_materials.sampler;

        // Build camera and uniforms
        let uniforms = self.build_uniforms(params, decoded, width as f32, height as f32)?;
        let uniform_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("terrain.uniform_buffer"),
                contents: bytemuck::cast_slice(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Build shading uniforms
        let shading_uniforms =
            self.build_shading_uniforms(material_set, gpu_materials.as_ref(), params, decoded)?;
        let shading_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("terrain.shading_buffer"),
                contents: bytemuck::cast_slice(&shading_uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let overlay_binding = self.extract_overlay_binding(params);

        // Log color configuration for debugging
        self.log_color_debug(params, &overlay_binding);

        let overlay_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("terrain.overlay_buffer"),
                contents: bytemuck::bytes_of(&overlay_binding.uniform),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let mut _fallback_colormap_view = None;
        let (colormap_view, colormap_sampler) = if let Some(lut) = overlay_binding.lut.as_ref() {
            (&lut.view, &lut.sampler)
        } else {
            let view = gpu_materials
                .texture
                .create_view(&wgpu::TextureViewDescriptor {
                    label: Some("terrain.fallback.colormap.view"),
                    format: Some(wgpu::TextureFormat::Rgba8UnormSrgb),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: 0,
                    mip_level_count: Some(1),
                    base_array_layer: 0,
                    array_layer_count: Some(1),
                });
            _fallback_colormap_view = Some(view);
            (_fallback_colormap_view.as_ref().unwrap(), material_sampler)
        };

        let ibl_resources = env_maps.ensure_gpu_resources(&self.device, &self.queue)?;
        let (sin_theta, cos_theta) = env_maps.rotation_rad().sin_cos();
        let ibl_uniforms = IblUniforms {
            intensity: env_maps.intensity.max(0.0),
            sin_theta,
            cos_theta,
            specular_mip_count: ibl_resources.specular_mip_count.max(1) as f32,
        };
        let ibl_uniform_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("terrain.ibl_uniform_buffer"),
                    contents: bytemuck::bytes_of(&ibl_uniforms),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let ibl_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain_pbr_pom.ibl_bind_group"),
            layout: &self.ibl_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        ibl_resources.specular_view.as_ref(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        ibl_resources.irradiance_view.as_ref(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(ibl_resources.sampler.as_ref()),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(ibl_resources.brdf_view.as_ref()),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: ibl_uniform_buffer.as_entire_binding(),
                },
            ],
        });

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain_pbr_pom.bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&heightmap_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler_linear),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(material_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(material_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: shading_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(colormap_view),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::Sampler(colormap_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: overlay_buffer.as_entire_binding(),
                },
            ],
        });

        // ──────────────────────────────────────────────────────────────────────────
        // MSAA Selection (single authoritative source)
        // ──────────────────────────────────────────────────────────────────────────
        let requested_msaa = params.msaa_samples.max(1);
        let effective_msaa =
            select_effective_msaa(requested_msaa, self.color_format, &self.adapter);

        // Log downgrade warning (exactly once)
        if effective_msaa != requested_msaa {
            log::warn!(
                "MSAA: requested {} not supported for {:?}; using {}",
                requested_msaa,
                self.color_format,
                effective_msaa
            );
        }

        // Update pipeline if MSAA changed
        let mut pipeline_cache = self
            .pipeline
            .lock()
            .map_err(|_| anyhow!("TerrainRenderer pipeline mutex poisoned"))?;
        if pipeline_cache.sample_count != effective_msaa {
            let light_buffer = self
                .light_buffer
                .lock()
                .map_err(|_| anyhow!("Light buffer mutex poisoned"))?;
            // Reuse the same shadow layout (create it here since we don't store it)
            let shadow_bind_group_layout =
                Self::create_shadow_bind_group_layout(self.device.as_ref());
            pipeline_cache.pipeline = Self::create_render_pipeline(
                self.device.as_ref(),
                &self.bind_group_layout,
                light_buffer.bind_group_layout(),
                &self.ibl_bind_group_layout,
                &shadow_bind_group_layout,
                self.color_format,
                effective_msaa,
            );
            pipeline_cache.sample_count = effective_msaa;
        }

        // ──────────────────────────────────────────────────────────────────────────
        // Texture Creation (using effective_msaa only)
        // ──────────────────────────────────────────────────────────────────────────
        let (out_width, out_height) = params.size_px;
        let render_scale = params.render_scale.clamp(0.25, 4.0);
        let internal_width = ((out_width as f32 * render_scale).round().max(1.0)) as u32;
        let internal_height = ((out_height as f32 * render_scale).round().max(1.0)) as u32;
        let needs_scaling = internal_width != out_width || internal_height != out_height;

        // Internal texture: always single-sample (used for resolve or direct render)
        let internal_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain.internal.render_target"),
            size: wgpu::Extent3d {
                width: internal_width,
                height: internal_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1, // Always 1: resolve target or direct render target
            dimension: wgpu::TextureDimension::D2,
            format: self.color_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let internal_view = internal_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // MSAA texture: only created if effective_msaa > 1
        let msaa_texture = if effective_msaa > 1 {
            Some(self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("terrain.msaa.render_target"),
                size: wgpu::Extent3d {
                    width: internal_width,
                    height: internal_height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: effective_msaa, // Multisampled render target
                dimension: wgpu::TextureDimension::D2,
                format: self.color_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            }))
        } else {
            None
        };
        let msaa_view = msaa_texture
            .as_ref()
            .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()));

        // ──────────────────────────────────────────────────────────────────────────
        // MSAA Debug Instrumentation (before render pass)
        // ──────────────────────────────────────────────────────────────────────────
        let color_attachment_sample_count = if effective_msaa > 1 {
            effective_msaa
        } else {
            1
        };
        let resolve_sample_count = if effective_msaa > 1 { Some(1) } else { None };

        log_msaa_debug(
            &self.adapter,
            self.color_format,
            None, // No depth in this pipeline
            requested_msaa,
            effective_msaa,
            color_attachment_sample_count,
            resolve_sample_count,
            None, // No depth
            pipeline_cache.sample_count,
        );

        // ──────────────────────────────────────────────────────────────────────────
        // MSAA Invariant Assertions (before render pass)
        // ──────────────────────────────────────────────────────────────────────────
        let invariants = MsaaInvariants {
            effective_msaa,
            pipeline_sample_count: pipeline_cache.sample_count,
            color_attachment_sample_count,
            has_resolve_target: effective_msaa > 1,
            resolve_sample_count,
            depth_sample_count: None, // No depth in this pipeline
            readback_sample_count: 1, // internal_texture is always sample_count=1
        };
        assert_msaa_invariants(&invariants, self.color_format)?;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("terrain.encoder"),
            });

        {
            let color_view = msaa_view.as_ref().unwrap_or(&internal_view);
            let resolve_target = if msaa_view.is_some() {
                Some(&internal_view)
            } else {
                None
            };

            // P1-06: Lock light buffer before render pass to get bind group reference
            // The bind group was already updated above with lights from params
            let light_buffer_guard = self
                .light_buffer
                .lock()
                .map_err(|_| anyhow!("Light buffer mutex poisoned"))?;
            // LightBuffer always provides a bind group (updated above with lights from params)
            let light_bind_group = light_buffer_guard
                .bind_group()
                .expect("LightBuffer should always provide a bind group");

            // Main render pass
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("terrain.render_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: color_view,
                        resolve_target,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.1,
                                g: 0.1,
                                b: 0.15,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                pass.set_pipeline(&pipeline_cache.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);

                // Always bind light buffer at index 1 (required by pipeline layout)
                // The pipeline expects "Light Buffer Bind Group Layout" at index 1
                pass.set_bind_group(1, light_bind_group, &[]);

                pass.set_bind_group(2, &ibl_bind_group, &[]);

                // Always bind noop shadow at index 3 (required by pipeline layout)
                // The pipeline expects "terrain_pbr_pom.shadow_bind_group_layout" at index 3
                pass.set_bind_group(3, &self.noop_shadow.bind_group, &[]);

                pass.draw(0..3, 0..1);
            }
            drop(light_buffer_guard);
        }
        drop(pipeline_cache);

        let mut scaled_result: Option<(wgpu::Texture, u32, u32)> = None;
        if needs_scaling {
            let output_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("terrain.output.resolved"),
                size: wgpu::Extent3d {
                    width: out_width,
                    height: out_height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.color_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let output_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());
            let sampling = &decoded.sampling;
            let blit_sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("terrain.blit.sampler"),
                address_mode_u: Self::map_address_mode(sampling.address_u),
                address_mode_v: Self::map_address_mode(sampling.address_v),
                address_mode_w: Self::map_address_mode(sampling.address_w),
                mag_filter: Self::map_filter_mode(sampling.mag_filter),
                min_filter: Self::map_filter_mode(sampling.min_filter),
                mipmap_filter: Self::map_filter_mode(sampling.mip_filter),
                anisotropy_clamp: sampling.anisotropy as u16,
                ..Default::default()
            });
            let blit_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("terrain.blit.bind_group"),
                layout: &self.blit_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&internal_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&blit_sampler),
                    },
                ],
            });

            {
                let mut blit_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("terrain.blit_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &output_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                blit_pass.set_pipeline(&self.blit_pipeline);
                blit_pass.set_bind_group(0, &blit_bind_group, &[]);
                blit_pass.draw(0..3, 0..1);
            }

            scaled_result = Some((output_texture, out_width, out_height));
        }

        let (final_texture, final_width, final_height) =
            scaled_result.unwrap_or((internal_texture, out_width, out_height));
        self.queue.submit(Some(encoder.finish()));

        Ok(crate::Frame::new(
            self.device.clone(),
            self.queue.clone(),
            final_texture,
            final_width,
            final_height,
            self.color_format,
        ))
    }

    /// Log color configuration for debugging
    fn log_color_debug(
        &self,
        _params: &crate::terrain_render_params::TerrainRenderParams,
        binding: &OverlayBinding,
    ) {
        let debug_mode = binding.uniform.params1[1] as i32;
        let albedo_mode = match binding.uniform.params1[2] as i32 {
            0 => "material",
            1 => "colormap",
            2 => "mix",
            _ => "unknown",
        };
        let blend_mode = match binding.uniform.params1[0] as i32 {
            0 => "Replace",
            1 => "Alpha",
            2 => "Multiply",
            3 => "Additive",
            _ => "unknown",
        };

        log::info!(target: "color.debug", "╔══════════════════════════════════════════════════");
        log::info!(target: "color.debug", "║ Color Configuration Debug");
        log::info!(target: "color.debug", "╠══════════════════════════════════════════════════");
        log::info!(target: "color.debug", "║ Domain: [{}, {}]", binding.uniform.params0[0],
            binding.uniform.params0[0] + 1.0 / binding.uniform.params0[1].max(1e-6));
        log::info!(target: "color.debug", "║ Overlay Strength: {}", binding.uniform.params0[2]);
        log::info!(target: "color.debug", "║ Colormap Strength: {}", binding.uniform.params1[3]);
        log::info!(target: "color.debug", "║ Albedo Mode: {}", albedo_mode);
        log::info!(target: "color.debug", "║ Blend Mode: {}", blend_mode);
        log::info!(target: "color.debug", "║ Debug Mode: {}", debug_mode);
        log::info!(target: "color.debug", "║ Gamma: {}", binding.uniform.params2[0]);

        // Sample LUT at t=0.0, 0.5, 1.0 if available
        if binding.lut.is_some() {
            log::info!(target: "color.debug", "╠══════════════════════════════════════════════════");
            log::info!(target: "color.debug", "║ LUT Samples:");
            log::info!(target: "color.debug", "║   t=0.0 probe ready");
            log::info!(target: "color.debug", "║   t=0.5 probe ready");
            log::info!(target: "color.debug", "║   t=1.0 probe ready");
            log::info!(target: "color.debug", "║ LUT texture bound: yes");
        } else {
            log::info!(target: "color.debug", "║ LUT texture bound: no");
        }
        log::info!(target: "color.debug", "╚══════════════════════════════════════════════════");
    }

    fn extract_overlay_binding(
        &self,
        params: &crate::terrain_render_params::TerrainRenderParams,
    ) -> OverlayBinding {
        let overlays = params.overlays();

        // Get debug mode from environment variable
        let debug_mode = std::env::var("VF_COLOR_DEBUG_MODE")
            .ok()
            .and_then(|s| s.parse::<i32>().ok())
            .unwrap_or(0);
        let debug_mode_f = debug_mode.clamp(0, 3) as f32;

        // Get albedo_mode as enum value
        let albedo_mode_f = match params.albedo_mode() {
            "material" => 0.0,
            "colormap" => 1.0,
            "mix" => 2.0,
            _ => 2.0, // default to mix
        };

        let colormap_strength = params.colormap_strength().clamp(0.0, 1.0);
        let gamma = params.gamma().max(0.1); // Minimum gamma to avoid division by zero

        let mut binding = OverlayBinding {
            uniform: OverlayUniforms {
                params0: [0.0; 4],
                params1: [0.0, debug_mode_f, albedo_mode_f, colormap_strength],
                params2: [gamma, 0.0, 0.0, 0.0],
            },
            lut: None,
        };

        pyo3::Python::with_gil(|py| {
            for overlay_py in overlays {
                let overlay_ref = overlay_py.borrow(py);
                if let Some(colormap) = overlay_ref.colormap_clone() {
                    let domain = overlay_ref.domain_tuple();
                    let range = domain.1 - domain.0;
                    let inv_range = if range.abs() > f32::EPSILON {
                        1.0 / range
                    } else {
                        0.0
                    };
                    let strength = overlay_ref.strength_value().max(0.0);
                    let offset = overlay_ref.offset();
                    let mode = overlay_ref.blend_mode();
                    let mode_value = match mode.to_ascii_lowercase().as_str() {
                        "replace" => 0.0,
                        "alpha" => 1.0,
                        "multiply" | "mul" => 2.0,
                        "add" | "additive" => 3.0,
                        _ => 1.0, // default to alpha
                    };

                    binding.uniform.params0 = [domain.0, inv_range, strength, offset];
                    binding.uniform.params1[0] = mode_value;
                    binding.lut = Some(colormap.lut.clone());
                    break;
                }
            }
        });

        binding
    }

    /// Upload heightmap as R32Float texture
    fn upload_heightmap_texture(
        &self,
        width: u32,
        height: u32,
        data: &[f32],
    ) -> Result<wgpu::Texture> {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain.heightmap"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(data),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(width * 4), // R32Float = 4 bytes
                rows_per_image: Some(height),
            },
            size,
        );

        Ok(texture)
    }

    /// Build uniform buffer data
    fn build_uniforms(
        &self,
        params: &crate::terrain_render_params::TerrainRenderParams,
        decoded: &crate::terrain_render_params::DecodedTerrainSettings,
        _terrain_width: f32,
        _terrain_height: f32,
    ) -> Result<Vec<f32>> {
        // Camera setup using spherical coordinates
        let phi_rad = params.cam_phi_deg.to_radians();
        let theta_rad = params.cam_theta_deg.to_radians();

        let eye_x = params.cam_target[0] + params.cam_radius * theta_rad.sin() * phi_rad.cos();
        let eye_y = params.cam_target[1] + params.cam_radius * theta_rad.cos();
        let eye_z = params.cam_target[2] + params.cam_radius * theta_rad.sin() * phi_rad.sin();

        let eye = glam::Vec3::new(eye_x, eye_y, eye_z);
        let target = glam::Vec3::from_array(params.cam_target);
        let up = glam::Vec3::Y;

        let view = glam::Mat4::look_at_rh(eye, target, up);
        let aspect = params.size_px.0 as f32 / params.size_px.1 as f32;
        let proj = glam::Mat4::perspective_rh(
            params.fov_y_deg.to_radians(),
            aspect,
            params.clip.0,
            params.clip.1,
        );

        // Pack into vec (16 floats view + 16 floats proj + 4 floats sun_exposure + 4 floats spacing + 4 floats pad)
        let mut uniforms = Vec::with_capacity(44);

        // View matrix (column-major)
        uniforms.extend_from_slice(&view.to_cols_array());
        // Projection matrix (column-major)
        uniforms.extend_from_slice(&proj.to_cols_array());

        // sun_exposure: (sun direction, intensity)
        uniforms.extend_from_slice(&[
            decoded.light.direction[0],
            decoded.light.direction[1],
            decoded.light.direction[2],
            decoded.light.intensity,
        ]);

        // spacing_h_exag: (spacing_x, spacing_y, height_exag, render_scale)
        uniforms.extend_from_slice(&[1.0, 1.0, params.z_scale, params.render_scale]);

        // pad_tail
        uniforms.extend_from_slice(&[0.0, 0.0, 0.0, 0.0]);

        Ok(uniforms)
    }

    /// Build shading uniform buffer
    fn build_shading_uniforms(
        &self,
        material_set: &crate::material_set::MaterialSet,
        gpu_materials: &crate::material_set::GpuMaterialSet,
        params: &crate::terrain_render_params::TerrainRenderParams,
        decoded: &crate::terrain_render_params::DecodedTerrainSettings,
    ) -> Result<Vec<f32>> {
        let pom_flags = {
            let mut flags = 0u32;
            if decoded.pom.enabled {
                flags |= 1;
                if decoded.pom.occlusion {
                    flags |= 1 << 1;
                }
                if decoded.pom.shadow {
                    flags |= 1 << 2;
                }
            }
            flags
        };

        let (pom_min_steps, pom_max_steps, pom_refine_steps) = if decoded.pom.enabled {
            (
                decoded.pom.min_steps as f32,
                decoded.pom.max_steps as f32,
                decoded.pom.refine_steps as f32,
            )
        } else {
            (0.0, 0.0, 0.0)
        };

        let mut uniforms = Vec::with_capacity(44);

        uniforms.extend_from_slice(&[
            decoded.triplanar.scale,
            decoded.triplanar.blend_sharpness,
            decoded.triplanar.normal_strength,
            if decoded.pom.enabled {
                decoded.pom.scale
            } else {
                0.0
            },
        ]);

        uniforms.extend_from_slice(&[
            pom_min_steps,
            pom_max_steps,
            pom_refine_steps,
            pom_flags as f32,
        ]);

        let layer_centers = gpu_materials.layer_centers();
        uniforms.extend_from_slice(&layer_centers);

        let mut layer_roughness = [1.0f32; MATERIAL_LAYER_CAPACITY];
        let mut layer_metallic = [0.0f32; MATERIAL_LAYER_CAPACITY];
        let active_layers = gpu_materials.layer_count as usize;
        let clamp_layers = active_layers.clamp(1, MATERIAL_LAYER_CAPACITY);
        for (idx, material) in material_set
            .materials()
            .iter()
            .enumerate()
            .take(clamp_layers)
        {
            layer_roughness[idx] = material.roughness;
            layer_metallic[idx] = material.metallic;
        }
        uniforms.extend_from_slice(&layer_roughness);
        uniforms.extend_from_slice(&layer_metallic);

        let layer_count_f = gpu_materials.layer_count.max(1) as f32;
        let blend_half = if layer_count_f <= 1.0 {
            1.0
        } else {
            f32::max(0.5 / layer_count_f, 0.05)
        };
        uniforms.extend_from_slice(&[
            layer_count_f,
            blend_half,
            decoded.lod.bias,
            decoded.lod.lod0_bias,
        ]);

        uniforms.extend_from_slice(&[
            decoded.light.color[0],
            decoded.light.color[1],
            decoded.light.color[2],
            params.exposure,
        ]);

        uniforms.extend_from_slice(&[
            decoded.clamp.height_range.0,
            decoded.clamp.height_range.1,
            decoded.clamp.slope_range.0,
            decoded.clamp.slope_range.1,
        ]);

        uniforms.extend_from_slice(&[
            decoded.clamp.ambient_range.0,
            decoded.clamp.ambient_range.1,
            decoded.clamp.shadow_range.0,
            decoded.clamp.shadow_range.1,
        ]);

        uniforms.extend_from_slice(&[
            decoded.clamp.occlusion_range.0,
            decoded.clamp.occlusion_range.1,
            decoded.lod.level as f32,
            decoded.sampling.anisotropy as f32,
        ]);

        uniforms.extend_from_slice(&[
            params.exposure,
            params.gamma,
            params.colormap_strength.clamp(0.0, 1.0),
            0.0,
        ]);

        Ok(uniforms)
    }
}
