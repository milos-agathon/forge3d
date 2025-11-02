// src/ibl_wrapper.rs
// PyO3 wrapper for image-based lighting resource setup
// Exists to bridge Python IBL configuration with GPU resource generation
// RELEVANT FILES: src/core/ibl.rs, src/terrain_renderer.rs, python/forge3d/terrain_params.py, src/shaders/terrain_pbr_pom.wgsl

use anyhow::{anyhow, Result};
use log::warn;
use pyo3::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use wgpu::{SamplerDescriptor, TextureAspect, TextureViewDescriptor, TextureViewDimension};

use crate::util::memory_budget;

/// Shared GPU handles consumed by the terrain renderer
/// Shared IBL GPU resources. Some fields are not yet read by all code paths; keep them for future integration.
#[allow(dead_code)]
pub(crate) struct IblGpuResources {
    pub irradiance_view: Arc<wgpu::TextureView>,
    pub specular_view: Arc<wgpu::TextureView>,
    pub brdf_view: Arc<wgpu::TextureView>,
    pub sampler: Arc<wgpu::Sampler>,
    pub specular_mip_count: u32,
}

/// Cached GPU state for IBL. Fields may be unused in some builds while plumbing lands.
#[allow(dead_code)]
pub(crate) struct IblGpuState {
    device: Arc<wgpu::Device>,
    quality: crate::core::ibl::IBLQuality,
    _renderer: crate::core::ibl::IBLRenderer,
    shared: Arc<IblGpuResources>,
}

/// Image-Based Lighting wrapper for Python
#[pyclass(module = "forge3d._forge3d", name = "IBL")]
pub struct IBL {
    pub(crate) environment_path: String,
    pub(crate) intensity: f32,
    pub(crate) rotation_deg: f32,
    pub(crate) hdr_image: Option<Arc<crate::formats::hdr::HdrImage>>,
    pub(crate) quality: crate::core::ibl::IBLQuality,
    /// Auto-select IBL quality by adapter; unused in some code paths until quality negotiation is wired globally.
    #[allow(dead_code)]
    pub(crate) use_auto_quality: bool,
    pub(crate) gpu_state: Arc<Mutex<Option<IblGpuState>>>,
    pub(crate) base_resolution: u32,
    pub(crate) cache_dir: Option<PathBuf>,
}

#[pymethods]
impl IBL {
    /// Create IBL from HDR environment map
    ///
    /// Args:
    ///     path: Path to HDR file (.hdr or .exr)
    ///     intensity: Light intensity multiplier (default: 1.0)
    ///     rotate_deg: Rotation in degrees (default: 0.0)
    ///     quality: Quality level: "low", "medium", "high", "ultra", or "auto" (default: "auto")
    ///
    /// Returns:
    ///     IBL object with prefiltered environment maps
    ///
    /// Note:
    ///     When quality="auto", the tier is selected based on GPU type (iGPU → Low, discrete → High)
    #[staticmethod]
    #[pyo3(signature = (path, intensity=1.0, rotate_deg=0.0, quality="auto"))]
    pub fn from_hdr(path: &str, intensity: f32, rotate_deg: f32, quality: &str) -> PyResult<Self> {
        // Validate inputs
        if intensity < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "intensity must be >= 0",
            ));
        }

        let quality_str = quality.to_lowercase();
        let use_auto_quality = quality_str == "auto";

        // Parse quality level (or use default for auto)
        let quality_level = match quality_str.as_str() {
            "auto" => crate::core::ibl::IBLQuality::Medium, // Placeholder, will be auto-selected
            "low" => crate::core::ibl::IBLQuality::Low,
            "medium" => crate::core::ibl::IBLQuality::Medium,
            "high" => crate::core::ibl::IBLQuality::High,
            "ultra" => crate::core::ibl::IBLQuality::Ultra,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid quality level: '{}'. Must be 'low', 'medium', 'high', 'ultra', or 'auto'",
                    quality
                )));
            }
        };

        // Load HDR image
        let hdr_image = crate::formats::hdr::load_hdr(path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!(
                "Failed to load HDR file '{}': {}",
                path, e
            ))
        })?;

        // Validate HDR dimensions
        if hdr_image.width == 0 || hdr_image.height == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "HDR image has invalid dimensions",
            ));
        }

        // Log HDR info
        log::info!(
            "Loaded HDR environment map: {} ({}x{}, {:.2} MB)",
            path,
            hdr_image.width,
            hdr_image.height,
            hdr_image.data_size_bytes() as f32 / (1024.0 * 1024.0)
        );

        Ok(Self {
            environment_path: path.to_string(),
            intensity,
            rotation_deg: rotate_deg,
            hdr_image: Some(Arc::new(hdr_image)),
            quality: quality_level,
            use_auto_quality,
            gpu_state: Arc::new(Mutex::new(None)),
            base_resolution: quality_level.base_environment_size(),
            cache_dir: None,
        })
    }

    /// Get environment map path
    #[getter]
    pub fn path(&self) -> String {
        self.environment_path.clone()
    }

    /// Get light intensity
    #[getter]
    pub fn intensity(&self) -> f32 {
        self.intensity
    }

    /// Set light intensity
    #[setter]
    pub fn set_intensity(&mut self, value: f32) -> PyResult<()> {
        if value < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "intensity must be >= 0",
            ));
        }
        self.intensity = value;
        Ok(())
    }

    /// Get rotation in degrees
    #[getter]
    pub fn rotation_deg(&self) -> f32 {
        self.rotation_deg
    }

    /// Set rotation in degrees
    #[setter]
    pub fn set_rotation_deg(&mut self, value: f32) {
        self.rotation_deg = value;
    }

    /// Get quality level as string
    #[getter]
    pub fn quality(&self) -> String {
        match self.quality {
            crate::core::ibl::IBLQuality::Low => "low".to_string(),
            crate::core::ibl::IBLQuality::Medium => "medium".to_string(),
            crate::core::ibl::IBLQuality::High => "high".to_string(),
            crate::core::ibl::IBLQuality::Ultra => "ultra".to_string(),
        }
    }

    /// Get HDR image dimensions
    #[getter]
    pub fn dimensions(&self) -> Option<(u32, u32)> {
        self.hdr_image.as_ref().map(|img| (img.width, img.height))
    }

    /// Python repr
    fn __repr__(&self) -> String {
        format!(
            "IBL(path='{}', intensity={:.1}, rotation_deg={:.1}, quality='{}')",
            self.environment_path,
            self.intensity,
            self.rotation_deg,
            self.quality()
        )
    }

    #[pyo3(text_signature = "($self, resolution)")]
    pub fn set_base_resolution(&mut self, resolution: u32) -> PyResult<()> {
        let clamped = resolution.max(16);
        self.base_resolution = clamped;
        if let Ok(mut guard) = self.gpu_state.lock() {
            *guard = None;
        }
        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn base_resolution(&self) -> u32 {
        self.base_resolution
    }

    #[pyo3(signature = (path=None))]
    #[pyo3(text_signature = "($self, path=None)")]
    pub fn set_cache_dir(&mut self, path: Option<&str>) -> PyResult<()> {
        self.cache_dir = path.map(PathBuf::from);
        if let Ok(mut guard) = self.gpu_state.lock() {
            *guard = None;
        }
        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn cache_dir(&self) -> Option<String> {
        self.cache_dir
            .as_ref()
            .map(|p| p.display().to_string())
    }
}

impl IBL {
    /// Get reference to HDR image
    pub fn hdr_image(&self) -> Option<&Arc<crate::formats::hdr::HdrImage>> {
        self.hdr_image.as_ref()
    }

    /// Get IBL quality level
    pub fn quality_level(&self) -> crate::core::ibl::IBLQuality {
        self.quality
    }

    /// Get rotation in radians
    pub fn rotation_rad(&self) -> f32 {
        self.rotation_deg.to_radians()
    }

    pub(crate) fn ensure_gpu_resources(
        &self,
        device: &Arc<wgpu::Device>,
        queue: &Arc<wgpu::Queue>,
    ) -> Result<Arc<IblGpuResources>> {
        let mut state_guard = self
            .gpu_state
            .lock()
            .map_err(|_| anyhow!("GPU state mutex poisoned"))?;

        let hdr_image = self
            .hdr_image
            .as_ref()
            .ok_or_else(|| anyhow!("HDR image data missing for IBL resource upload"))?;

        let adapter_info = crate::gpu::ctx().adapter.get_info();
        let mut quality_to_use = if self.use_auto_quality {
            memory_budget::auto_select_ibl_quality(&adapter_info)
        } else {
            self.quality
        };
        let requested_quality = quality_to_use;
        let mut usage_report = memory_budget::MemoryUsageReport::default();

        loop {
            let (irr_mem, spec_mem, brdf_mem) = memory_budget::estimate_ibl_memory(quality_to_use);
            let total = irr_mem + spec_mem + brdf_mem;

            if total <= memory_budget::MEMORY_BUDGET_CONSERVATIVE
                || quality_to_use == crate::core::ibl::IBLQuality::Low
            {
                usage_report.ibl_irradiance = irr_mem;
                usage_report.ibl_specular = spec_mem;
                usage_report.ibl_brdf = brdf_mem;
                break;
            }

            let downgraded = downgrade_ibl_quality(quality_to_use);
            if downgraded == quality_to_use {
                usage_report.ibl_irradiance = irr_mem;
                usage_report.ibl_specular = spec_mem;
                usage_report.ibl_brdf = brdf_mem;
                warn!(
                    "IBL quality {:?} still exceeds memory budget ({:.2} MiB); unable to downgrade further",
                    quality_to_use,
                    total as f32 / (1024.0 * 1024.0)
                );
                break;
            }

            warn!(
                "IBL quality {:?} estimates {:.2} MiB; downgrading to {:?}",
                quality_to_use,
                total as f32 / (1024.0 * 1024.0),
                downgraded
            );
            quality_to_use = downgraded;
        }

        usage_report.log_summary(&format!("IBL::{:?}", quality_to_use));

        if quality_to_use != requested_quality {
            warn!(
                "Adjusted IBL quality from {:?} to {:?} based on adapter '{}' budget",
                requested_quality, quality_to_use, adapter_info.name
            );
        }

        if let Some(state) = state_guard.as_ref() {
            if Arc::ptr_eq(&state.device, device) && state.quality == quality_to_use {
                return Ok(state.shared.clone());
            }
        }

        let mut renderer = crate::core::ibl::IBLRenderer::new(device.as_ref(), quality_to_use);
        renderer.set_base_resolution(self.base_resolution);
        if let Some(cache_dir) = self.cache_dir.as_ref() {
            renderer
                .configure_cache(cache_dir, Path::new(&self.environment_path))
                .map_err(|e| anyhow!("Failed to configure IBL cache: {}", e))?;
        }

        renderer
            .load_environment_map(
                device.as_ref(),
                queue.as_ref(),
                &hdr_image.data,
                hdr_image.width,
                hdr_image.height,
            )
            .map_err(|e| anyhow!("Failed to upload environment map: {}", e))?;

        renderer
            .initialize(device.as_ref(), queue.as_ref())
            .map_err(|e| anyhow!("Failed to build IBL resources: {}", e))?;

        let (irr_tex_opt, spec_tex_opt, brdf_tex_opt) = renderer.textures();
        let irr_tex = irr_tex_opt.ok_or_else(|| anyhow!("Irradiance texture missing"))?;
        let spec_tex = spec_tex_opt.ok_or_else(|| anyhow!("Specular texture missing"))?;
        let brdf_tex = brdf_tex_opt.ok_or_else(|| anyhow!("BRDF LUT texture missing"))?;

        let irradiance_view = Arc::new(irr_tex.create_view(&TextureViewDescriptor {
            label: Some("ibl.irradiance.shared"),
            format: Some(wgpu::TextureFormat::Rgba16Float),
            dimension: Some(TextureViewDimension::Cube),
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: Some(6),
        }));

        let specular_view = Arc::new(spec_tex.create_view(&TextureViewDescriptor {
            label: Some("ibl.specular.shared"),
            format: Some(wgpu::TextureFormat::Rgba16Float),
            dimension: Some(TextureViewDimension::Cube),
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: Some(6),
        }));

        let brdf_view = Arc::new(brdf_tex.create_view(&TextureViewDescriptor {
            label: Some("ibl.brdf_lut.shared"),
            ..TextureViewDescriptor::default()
        }));

        let sampler = Arc::new(device.create_sampler(&SamplerDescriptor {
            label: Some("ibl.env.sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }));

        let shared = Arc::new(IblGpuResources {
            irradiance_view,
            specular_view,
            brdf_view,
            sampler,
            specular_mip_count: quality_to_use.specular_mip_levels(),
        });

        *state_guard = Some(IblGpuState {
            device: device.clone(),
            quality: quality_to_use,
            _renderer: renderer,
            shared: shared.clone(),
        });

        Ok(shared)
    }
}

fn downgrade_ibl_quality(quality: crate::core::ibl::IBLQuality) -> crate::core::ibl::IBLQuality {
    use crate::core::ibl::IBLQuality::{High, Low, Medium, Ultra};
    match quality {
        Ultra => High,
        High => Medium,
        Medium => Low,
        Low => Low,
    }
}
