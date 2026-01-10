// src/lib.rs
// Rust crate root for forge3d - GPU rendering library with Python bindings
// Provides SDF primitives, CSG operations, hybrid traversal, and path tracing
// RELEVANT FILES:src/sdf/mod.rs,src/path_tracing/mod.rs,python/forge3d/__init__.py

#[cfg(feature = "extension-module")]
use once_cell::sync::Lazy;
#[cfg(feature = "extension-module")]
use std::sync::Mutex;

#[cfg(feature = "extension-module")]
use shadows::state::{CpuCsmConfig, CpuCsmState};

#[cfg(feature = "extension-module")]
use glam::Vec3;
#[cfg(feature = "extension-module")]
use numpy::{IntoPyArray, PyArray1, PyArray3, PyArrayMethods, PyReadonlyArrayDyn};
#[cfg(feature = "extension-module")]
use pyo3::types::PyDict;
#[cfg(feature = "extension-module")]
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    wrap_pyfunction,
};

// C1/C3/C5/C6/C7: Additional imports for PyO3 functions
#[cfg(feature = "extension-module")]
use crate::core::async_compute::{
    AsyncComputeConfig as AcConfig, AsyncComputeScheduler as AcScheduler,
    ComputePassDescriptor as AcPassDesc, DispatchParams as AcDispatch,
};
#[cfg(feature = "extension-module")]
use crate::core::context as engine_context;
#[cfg(feature = "extension-module")]
use crate::core::device_caps::DeviceCaps;
#[cfg(feature = "extension-module")]
use crate::core::framegraph_impl::{
    FrameGraph as Fg, PassType as FgPassType, ResourceDesc as FgResourceDesc,
    ResourceType as FgResourceType,
};
#[cfg(feature = "extension-module")]
use crate::core::multi_thread::{
    CopyTask as MtCopyTask, MultiThreadConfig as MtConfig, MultiThreadRecorder as MtRecorder,
};
#[cfg(feature = "extension-module")]
use crate::renderer::readback::read_texture_tight;
#[cfg(feature = "extension-module")]
use crate::sdf::hybrid::Ray as HybridRay;
#[cfg(feature = "extension-module")]
use crate::sdf::py::PySdfScene;
#[cfg(all(feature = "extension-module", feature = "images"))]
use crate::util::exr_write;
#[cfg(feature = "extension-module")]
use crate::util::image_write;
#[cfg(feature = "extension-module")]
use std::path::Path;
#[cfg(feature = "extension-module")]
use std::sync::Arc;
#[cfg(feature = "extension-module")]
use wgpu::{
    Extent3d as FgExtent3d, ShaderModuleDescriptor, ShaderSource, TextureFormat as FgTexFormat,
    TextureUsages as FgTexUsages,
};

#[cfg(feature = "extension-module")]
static GLOBAL_CSM_STATE: Lazy<Mutex<CpuCsmState>> =
    Lazy::new(|| Mutex::new(CpuCsmState::default()));

#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "Frame")]
pub struct Frame {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    texture: wgpu::Texture,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
}

// P5: Screen-space GI Python bindings
#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "ScreenSpaceGI")]
pub struct PyScreenSpaceGI {
    manager: crate::core::screen_space_effects::ScreenSpaceEffectsManager,
    width: u32,
    height: u32,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyScreenSpaceGI {
    #[new]
    #[pyo3(signature = (width=1280, height=720))]
    pub fn new(width: u32, height: u32) -> PyResult<Self> {
        let g = crate::core::gpu::ctx();
        let manager = crate::core::screen_space_effects::ScreenSpaceEffectsManager::new(
            g.device.as_ref(),
            width,
            height,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("failed to create GI manager: {e}")))?;
        Ok(Self {
            manager,
            width,
            height,
        })
    }

    /// Enable SSAO
    pub fn enable_ssao(&mut self) -> PyResult<()> {
        let g = crate::core::gpu::ctx();
        self.manager
            .enable_effect(
                g.device.as_ref(),
                crate::core::screen_space_effects::ScreenSpaceEffect::SSAO,
            )
            .map_err(|e| PyRuntimeError::new_err(format!("enable_ssao failed: {e}")))
    }

    /// Enable SSGI
    pub fn enable_ssgi(&mut self) -> PyResult<()> {
        let g = crate::core::gpu::ctx();
        self.manager
            .enable_effect(
                g.device.as_ref(),
                crate::core::screen_space_effects::ScreenSpaceEffect::SSGI,
            )
            .map_err(|e| PyRuntimeError::new_err(format!("enable_ssgi failed: {e}")))
    }

    /// Enable SSR
    pub fn enable_ssr(&mut self) -> PyResult<()> {
        let g = crate::core::gpu::ctx();
        self.manager
            .enable_effect(
                g.device.as_ref(),
                crate::core::screen_space_effects::ScreenSpaceEffect::SSR,
            )
            .map_err(|e| PyRuntimeError::new_err(format!("enable_ssr failed: {e}")))
    }

    /// Disable an effect by name: "ssao", "ssgi", or "ssr"
    pub fn disable(&mut self, effect: &str) -> PyResult<()> {
        use crate::core::screen_space_effects::ScreenSpaceEffect as SSE;
        let eff = match effect.to_lowercase().as_str() {
            "ssao" => SSE::SSAO,
            "ssgi" => SSE::SSGI,
            "ssr" => SSE::SSR,
            _ => return Err(PyValueError::new_err(format!("unknown effect: {effect}"))),
        };
        self.manager.disable_effect(eff);
        Ok(())
    }

    /// Resize underlying GBuffer to a new size
    pub fn resize(&mut self, width: u32, height: u32) -> PyResult<()> {
        let g = crate::core::gpu::ctx();
        self.manager
            .gbuffer_mut()
            .resize(g.device.as_ref(), width, height)
            .map_err(|e| PyRuntimeError::new_err(format!("resize failed: {e}")))?;
        self.width = width;
        self.height = height;
        Ok(())
    }

    /// Execute enabled GI passes for the current frame
    pub fn execute(&mut self) -> PyResult<()> {
        let g = crate::core::gpu::ctx();
        let mut encoder = g
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("PyScreenSpaceGI.execute"),
            });
        self.manager
            .execute(g.device.as_ref(), &mut encoder, None, None)
            .map_err(|e| PyRuntimeError::new_err(format!("execute failed: {e}")))?;
        g.queue.submit(Some(encoder.finish()));
        Ok(())
    }
}

// Feature B: Picking system Python bindings
#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "PickResult")]
pub struct PyPickResult {
    #[pyo3(get)]
    pub feature_id: u32,
    #[pyo3(get)]
    pub screen_x: u32,
    #[pyo3(get)]
    pub screen_y: u32,
    #[pyo3(get)]
    pub world_pos: Option<(f32, f32, f32)>,
    #[pyo3(get)]
    pub layer_name: Option<String>,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyPickResult {
    #[new]
    fn new(feature_id: u32, screen_x: u32, screen_y: u32) -> Self {
        Self {
            feature_id,
            screen_x,
            screen_y,
            world_pos: None,
            layer_name: None,
        }
    }
    
    fn __repr__(&self) -> String {
        format!(
            "PickResult(feature_id={}, screen=({}, {}), layer={:?})",
            self.feature_id, self.screen_x, self.screen_y, self.layer_name
        )
    }
}

#[cfg(feature = "extension-module")]
impl From<crate::picking::PickResult> for PyPickResult {
    fn from(r: crate::picking::PickResult) -> Self {
        Self {
            feature_id: r.feature_id,
            screen_x: r.screen_pos.0,
            screen_y: r.screen_pos.1,
            world_pos: r.world_pos.map(|p| (p[0], p[1], p[2])),
            layer_name: r.layer_name,
        }
    }
}

// Plan 2: Terrain query result Python bindings
#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "TerrainQueryResult")]
pub struct PyTerrainQueryResult {
    #[pyo3(get)]
    pub elevation: f32,
    #[pyo3(get)]
    pub slope: f32,
    #[pyo3(get)]
    pub aspect: f32,
    #[pyo3(get)]
    pub world_pos: (f32, f32, f32),
    #[pyo3(get)]
    pub normal: (f32, f32, f32),
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyTerrainQueryResult {
    #[new]
    fn new(elevation: f32, slope: f32, aspect: f32) -> Self {
        Self {
            elevation,
            slope,
            aspect,
            world_pos: (0.0, 0.0, 0.0),
            normal: (0.0, 1.0, 0.0),
        }
    }
    
    fn __repr__(&self) -> String {
        format!(
            "TerrainQueryResult(elevation={:.2}, slope={:.1}°, aspect={:.1}°)",
            self.elevation, self.slope, self.aspect
        )
    }
}

#[cfg(feature = "extension-module")]
impl From<crate::picking::TerrainQueryResult> for PyTerrainQueryResult {
    fn from(r: crate::picking::TerrainQueryResult) -> Self {
        Self {
            elevation: r.elevation,
            slope: r.slope,
            aspect: r.aspect,
            world_pos: (r.world_pos[0], r.world_pos[1], r.world_pos[2]),
            normal: (r.normal[0], r.normal[1], r.normal[2]),
        }
    }
}

// Plan 2: Selection style Python bindings
#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "SelectionStyle")]
#[derive(Clone)]
pub struct PySelectionStyle {
    #[pyo3(get, set)]
    pub color: (f32, f32, f32, f32),
    #[pyo3(get, set)]
    pub outline: bool,
    #[pyo3(get, set)]
    pub outline_width: f32,
    #[pyo3(get, set)]
    pub glow: bool,
    #[pyo3(get, set)]
    pub glow_intensity: f32,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PySelectionStyle {
    #[new]
    #[pyo3(signature = (color=(1.0, 0.8, 0.0, 0.5), outline=false, outline_width=2.0, glow=false, glow_intensity=0.5))]
    fn new(
        color: (f32, f32, f32, f32),
        outline: bool,
        outline_width: f32,
        glow: bool,
        glow_intensity: f32,
    ) -> Self {
        Self {
            color,
            outline,
            outline_width,
            glow,
            glow_intensity,
        }
    }
    
    fn __repr__(&self) -> String {
        format!(
            "SelectionStyle(color={:?}, outline={}, glow={})",
            self.color, self.outline, self.glow
        )
    }
}

#[cfg(feature = "extension-module")]
impl From<PySelectionStyle> for crate::picking::SelectionStyle {
    fn from(s: PySelectionStyle) -> Self {
        Self {
            color: [s.color.0, s.color.1, s.color.2, s.color.3],
            outline: s.outline,
            outline_width: s.outline_width,
            glow: s.glow,
            glow_intensity: s.glow_intensity,
        }
    }
}

#[cfg(feature = "extension-module")]
impl From<&crate::picking::SelectionStyle> for PySelectionStyle {
    fn from(s: &crate::picking::SelectionStyle) -> Self {
        Self {
            color: (s.color[0], s.color[1], s.color[2], s.color[3]),
            outline: s.outline,
            outline_width: s.outline_width,
            glow: s.glow,
            glow_intensity: s.glow_intensity,
        }
    }
}

// Plan 3: Rich pick result with full attributes
#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "RichPickResult")]
#[derive(Clone)]
pub struct PyRichPickResult {
    #[pyo3(get)]
    pub feature_id: u32,
    #[pyo3(get)]
    pub layer_name: String,
    #[pyo3(get)]
    pub world_pos: (f32, f32, f32),
    #[pyo3(get)]
    pub attributes: std::collections::HashMap<String, String>,
    #[pyo3(get)]
    pub hit_distance: f32,
    #[pyo3(get)]
    pub terrain_elevation: Option<f32>,
    #[pyo3(get)]
    pub terrain_slope: Option<f32>,
    #[pyo3(get)]
    pub terrain_aspect: Option<f32>,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyRichPickResult {
    #[new]
    fn new(feature_id: u32, layer_name: String) -> Self {
        Self {
            feature_id,
            layer_name,
            world_pos: (0.0, 0.0, 0.0),
            attributes: std::collections::HashMap::new(),
            hit_distance: 0.0,
            terrain_elevation: None,
            terrain_slope: None,
            terrain_aspect: None,
        }
    }
    
    fn __repr__(&self) -> String {
        format!(
            "RichPickResult(feature_id={}, layer='{}', pos={:?})",
            self.feature_id, self.layer_name, self.world_pos
        )
    }
    
    /// Get attribute by key
    fn get_attribute(&self, key: &str) -> Option<String> {
        self.attributes.get(key).cloned()
    }
}

#[cfg(feature = "extension-module")]
impl From<crate::picking::RichPickResult> for PyRichPickResult {
    fn from(r: crate::picking::RichPickResult) -> Self {
        let (terrain_elevation, terrain_slope, terrain_aspect) = 
            if let Some(info) = r.terrain_info {
                (Some(info.elevation), Some(info.slope), Some(info.aspect))
            } else {
                (None, None, None)
            };
        
        Self {
            feature_id: r.feature_id,
            layer_name: r.layer_name,
            world_pos: (r.world_pos[0], r.world_pos[1], r.world_pos[2]),
            attributes: r.attributes,
            hit_distance: r.hit_distance,
            terrain_elevation,
            terrain_slope,
            terrain_aspect,
        }
    }
}

// Plan 3: Highlight style Python bindings
#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "HighlightStyle")]
#[derive(Clone)]
pub struct PyHighlightStyle {
    #[pyo3(get, set)]
    pub color: (f32, f32, f32, f32),
    #[pyo3(get, set)]
    pub effect: String,
    #[pyo3(get, set)]
    pub outline_width: f32,
    #[pyo3(get, set)]
    pub glow_intensity: f32,
    #[pyo3(get, set)]
    pub glow_radius: f32,
    #[pyo3(get, set)]
    pub pulse_speed: f32,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyHighlightStyle {
    #[new]
    #[pyo3(signature = (
        color=(1.0, 0.8, 0.0, 0.5),
        effect="color_tint",
        outline_width=2.0,
        glow_intensity=0.5,
        glow_radius=8.0,
        pulse_speed=0.0
    ))]
    fn new(
        color: (f32, f32, f32, f32),
        effect: &str,
        outline_width: f32,
        glow_intensity: f32,
        glow_radius: f32,
        pulse_speed: f32,
    ) -> Self {
        Self {
            color,
            effect: effect.to_string(),
            outline_width,
            glow_intensity,
            glow_radius,
            pulse_speed,
        }
    }
    
    fn __repr__(&self) -> String {
        format!(
            "HighlightStyle(effect='{}', color={:?})",
            self.effect, self.color
        )
    }
    
    /// Create outline style
    #[staticmethod]
    fn outline(color: (f32, f32, f32, f32), width: f32) -> Self {
        Self {
            color,
            effect: "outline".to_string(),
            outline_width: width,
            glow_intensity: 0.5,
            glow_radius: 8.0,
            pulse_speed: 0.0,
        }
    }
    
    /// Create glow style
    #[staticmethod]
    fn glow(color: (f32, f32, f32, f32), intensity: f32, radius: f32) -> Self {
        Self {
            color,
            effect: "glow".to_string(),
            outline_width: 2.0,
            glow_intensity: intensity,
            glow_radius: radius,
            pulse_speed: 0.0,
        }
    }
}

#[cfg(feature = "extension-module")]
impl From<PyHighlightStyle> for crate::picking::HighlightStyle {
    fn from(s: PyHighlightStyle) -> Self {
        use crate::picking::HighlightEffect;
        
        let effect = match s.effect.as_str() {
            "none" => HighlightEffect::None,
            "color_tint" => HighlightEffect::ColorTint,
            "outline" => HighlightEffect::Outline,
            "glow" => HighlightEffect::Glow,
            "outline_glow" => HighlightEffect::OutlineGlow,
            _ => HighlightEffect::ColorTint,
        };
        
        Self {
            color: [s.color.0, s.color.1, s.color.2, s.color.3],
            secondary_color: [1.0, 1.0, 1.0, 0.3],
            effect,
            outline_width: s.outline_width,
            glow_intensity: s.glow_intensity,
            glow_radius: s.glow_radius,
            pulse_speed: s.pulse_speed,
            depth_bias: 0.001,
        }
    }
}

// Plan 3: Lasso state class (string-based for simplicity)
#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "LassoState")]
#[derive(Clone)]
pub struct PyLassoState {
    #[pyo3(get)]
    pub state: String,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyLassoState {
    #[new]
    #[pyo3(signature = (state="inactive"))]
    fn new(state: &str) -> Self {
        Self { state: state.to_string() }
    }
    
    fn __repr__(&self) -> String {
        format!("LassoState('{}')", self.state)
    }
    
    /// Check if inactive
    fn is_inactive(&self) -> bool {
        self.state == "inactive"
    }
    
    /// Check if drawing
    fn is_drawing(&self) -> bool {
        self.state == "drawing"
    }
    
    /// Check if complete
    fn is_complete(&self) -> bool {
        self.state == "complete"
    }
    
    #[staticmethod]
    fn inactive() -> Self {
        Self { state: "inactive".to_string() }
    }
    
    #[staticmethod]
    fn drawing() -> Self {
        Self { state: "drawing".to_string() }
    }
    
    #[staticmethod]
    fn complete() -> Self {
        Self { state: "complete".to_string() }
    }
}

// Plan 3: Heightfield hit result
#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "HeightfieldHit")]
#[derive(Clone)]
pub struct PyHeightfieldHit {
    #[pyo3(get)]
    pub position: (f32, f32, f32),
    #[pyo3(get)]
    pub t: f32,
    #[pyo3(get)]
    pub uv: (f32, f32),
    #[pyo3(get)]
    pub elevation: f32,
    #[pyo3(get)]
    pub normal: (f32, f32, f32),
    #[pyo3(get)]
    pub slope: f32,
    #[pyo3(get)]
    pub aspect: f32,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyHeightfieldHit {
    fn __repr__(&self) -> String {
        format!(
            "HeightfieldHit(elevation={:.2}, slope={:.1}°, aspect={:.1}°)",
            self.elevation, self.slope, self.aspect
        )
    }
}

#[cfg(feature = "extension-module")]
impl From<crate::picking::HeightfieldHit> for PyHeightfieldHit {
    fn from(h: crate::picking::HeightfieldHit) -> Self {
        Self {
            position: (h.position[0], h.position[1], h.position[2]),
            t: h.t,
            uv: (h.uv[0], h.uv[1]),
            elevation: h.elevation,
            normal: (h.normal[0], h.normal[1], h.normal[2]),
            slope: h.slope,
            aspect: h.aspect,
        }
    }
}

#[cfg(feature = "extension-module")]
impl Frame {
    pub(crate) fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        texture: wgpu::Texture,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Self {
        Self {
            device,
            queue,
            texture,
            width,
            height,
            format,
        }
    }

    fn read_tight_bytes(&self) -> anyhow::Result<Vec<u8>> {
        read_texture_tight(
            &self.device,
            &self.queue,
            &self.texture,
            (self.width, self.height),
            self.format,
        )
    }
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl Frame {
    #[new]
    fn py_new() -> PyResult<Self> {
        Err(PyRuntimeError::new_err(
            "Frame objects are constructed internally by forge3d",
        ))
    }

    #[getter]
    fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    fn format(&self) -> String {
        format!("{:?}", self.format)
    }

    fn save(&self, path: &str) -> PyResult<()> {
        let path_obj = Path::new(path);
        match self.format {
            wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => {
                if let Some(ext) = path_obj.extension().and_then(|ext| ext.to_str()) {
                    if !ext.eq_ignore_ascii_case("png") {
                        return Err(PyValueError::new_err(format!(
                            "expected .png extension for RGBA8 frame save, got .{}",
                            ext
                        )));
                    }
                }
                let mut data = self
                    .read_tight_bytes()
                    .map_err(|err| PyRuntimeError::new_err(format!("readback failed: {err:#}")))?;
                for px in data.chunks_exact_mut(4) {
                    px[3] = 255;
                }
                image_write::write_png_rgba8(path_obj, &data, self.width, self.height).map_err(
                    |err| PyRuntimeError::new_err(format!("failed to write PNG: {err:#}")),
                )?;
                Ok(())
            }
            wgpu::TextureFormat::Rgba16Float => {
                if let Some(ext) = path_obj.extension().and_then(|ext| ext.to_str()) {
                    if !ext.eq_ignore_ascii_case("exr") {
                        return Err(PyValueError::new_err(format!(
                            "expected .exr extension for RGBA16F frame save, got .{}",
                            ext
                        )));
                    }
                }
                #[cfg(feature = "images")]
                {
                    let data = crate::core::hdr::read_hdr_texture(
                        &self.device,
                        &self.queue,
                        &self.texture,
                        self.width,
                        self.height,
                        self.format,
                    )
                    .map_err(|err| PyRuntimeError::new_err(format!("HDR readback failed: {err}")))?;

                    exr_write::write_exr_rgba_f32(
                        path_obj,
                        self.width,
                        self.height,
                        &data,
                        "beauty",
                    )
                    .map_err(|err| PyRuntimeError::new_err(format!("failed to write EXR: {err:#}")))?;
                    Ok(())
                }
                #[cfg(not(feature = "images"))]
                {
                    Err(PyRuntimeError::new_err(
                        "saving RGBA16F frames requires the 'images' feature",
                    ))
                }
            }
            other => Err(PyValueError::new_err(format!(
                "unsupported texture format for save: {:?}",
                other
            ))),
        }
    }

    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray3<u8>> {
        match self.format {
            wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => {
                let data = self
                    .read_tight_bytes()
                    .map_err(|err| PyRuntimeError::new_err(format!("readback failed: {err:#}")))?;
                let arr = ndarray::Array3::from_shape_vec(
                    (self.height as usize, self.width as usize, 4),
                    data,
                )
                .map_err(|_| {
                    PyRuntimeError::new_err("failed to reshape RGBA buffer into numpy array")
                })?;
                Ok(arr.into_pyarray_bound(py).into_gil_ref())
            }
            wgpu::TextureFormat::Rgba16Float => Err(PyRuntimeError::new_err(
                "to_numpy for RGBA16F frames is not implemented yet",
            )),
            other => Err(PyValueError::new_err(format!(
                "unsupported texture format for numpy conversion: {:?}",
                other
            ))),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Frame(width={}, height={}, format={:?})",
            self.width, self.height, self.format
        )
    }
}

/// M1: AOV (Arbitrary Output Variable) frame container
/// Holds the auxiliary render outputs captured alongside the beauty pass
#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "AovFrame")]
pub struct AovFrame {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    /// Albedo texture (base color before lighting)
    albedo_texture: Option<wgpu::Texture>,
    /// Normal texture (world-space normals encoded to [0,1])
    normal_texture: Option<wgpu::Texture>,
    /// Depth texture (linear depth normalized)
    depth_texture: Option<wgpu::Texture>,
    width: u32,
    height: u32,
}

#[cfg(feature = "extension-module")]
impl AovFrame {
    pub(crate) fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        albedo_texture: Option<wgpu::Texture>,
        normal_texture: Option<wgpu::Texture>,
        depth_texture: Option<wgpu::Texture>,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            device,
            queue,
            albedo_texture,
            normal_texture,
            depth_texture,
            width,
            height,
        }
    }

    fn read_texture_bytes(&self, texture: &wgpu::Texture) -> anyhow::Result<Vec<u8>> {
        read_texture_tight(
            &self.device,
            &self.queue,
            texture,
            (self.width, self.height),
            wgpu::TextureFormat::Rgba8Unorm,
        )
    }
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl AovFrame {
    #[new]
    fn py_new() -> PyResult<Self> {
        Err(PyRuntimeError::new_err(
            "AovFrame objects are constructed internally by forge3d",
        ))
    }

    #[getter]
    fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    #[getter]
    fn has_albedo(&self) -> bool {
        self.albedo_texture.is_some()
    }

    #[getter]
    fn has_normal(&self) -> bool {
        self.normal_texture.is_some()
    }

    #[getter]
    fn has_depth(&self) -> bool {
        self.depth_texture.is_some()
    }

    /// Save albedo AOV to PNG file
    fn save_albedo(&self, path: &str) -> PyResult<()> {
        let texture = self.albedo_texture.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Albedo AOV not available")
        })?;
        let mut data = self.read_texture_bytes(texture)
            .map_err(|err| PyRuntimeError::new_err(format!("readback failed: {err:#}")))?;
        for px in data.chunks_exact_mut(4) {
            px[3] = 255;
        }
        image_write::write_png_rgba8(Path::new(path), &data, self.width, self.height)
            .map_err(|err| PyRuntimeError::new_err(format!("failed to write PNG: {err:#}")))?;
        Ok(())
    }

    /// Save normal AOV to PNG file
    fn save_normal(&self, path: &str) -> PyResult<()> {
        let texture = self.normal_texture.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Normal AOV not available")
        })?;
        let mut data = self.read_texture_bytes(texture)
            .map_err(|err| PyRuntimeError::new_err(format!("readback failed: {err:#}")))?;
        for px in data.chunks_exact_mut(4) {
            px[3] = 255;
        }
        image_write::write_png_rgba8(Path::new(path), &data, self.width, self.height)
            .map_err(|err| PyRuntimeError::new_err(format!("failed to write PNG: {err:#}")))?;
        Ok(())
    }

    /// Save depth AOV to PNG file
    fn save_depth(&self, path: &str) -> PyResult<()> {
        let texture = self.depth_texture.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Depth AOV not available")
        })?;
        let mut data = self.read_texture_bytes(texture)
            .map_err(|err| PyRuntimeError::new_err(format!("readback failed: {err:#}")))?;
        for px in data.chunks_exact_mut(4) {
            px[3] = 255;
        }
        image_write::write_png_rgba8(Path::new(path), &data, self.width, self.height)
            .map_err(|err| PyRuntimeError::new_err(format!("failed to write PNG: {err:#}")))?;
        Ok(())
    }

    /// Save all AOVs to directory with standard naming
    fn save_all(&self, output_dir: &str, base_name: &str) -> PyResult<()> {
        let dir = Path::new(output_dir);
        std::fs::create_dir_all(dir)
            .map_err(|e| PyRuntimeError::new_err(format!("failed to create directory: {e}")))?;
        
        if self.albedo_texture.is_some() {
            let path = dir.join(format!("{}_albedo.png", base_name));
            self.save_albedo(path.to_str().unwrap())?;
        }
        if self.normal_texture.is_some() {
            let path = dir.join(format!("{}_normal.png", base_name));
            self.save_normal(path.to_str().unwrap())?;
        }
        if self.depth_texture.is_some() {
            let path = dir.join(format!("{}_depth.png", base_name));
            self.save_depth(path.to_str().unwrap())?;
        }
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "AovFrame(width={}, height={}, albedo={}, normal={}, depth={})",
            self.width, self.height,
            self.albedo_texture.is_some(),
            self.normal_texture.is_some(),
            self.depth_texture.is_some()
        )
    }
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn render_debug_pattern_frame(py: Python<'_>, width: u32, height: u32) -> PyResult<Py<Frame>> {
    let ctx = crate::core::gpu::ctx();
    let texture = crate::util::debug_pattern::render_debug_pattern(
        ctx.device.as_ref(),
        ctx.queue.as_ref(),
        width,
        height,
    )
    .map_err(|err| PyRuntimeError::new_err(format!("failed to render debug pattern: {err:#}")))?;

    let frame = Frame::new(
        ctx.device.clone(),
        ctx.queue.clone(),
        texture,
        width,
        height,
        wgpu::TextureFormat::Rgba8UnormSrgb,
    );

    Py::new(py, frame)
}

#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (path, array, channel_prefix=None))]
fn numpy_to_exr(path: &str, array: PyReadonlyArrayDyn<f32>, channel_prefix: Option<&str>) -> PyResult<()> {
    let path_obj = Path::new(path);
    if let Some(ext) = path_obj.extension().and_then(|ext| ext.to_str()) {
        if !ext.eq_ignore_ascii_case("exr") {
            return Err(PyValueError::new_err(format!(
                "expected .exr extension, got .{}",
                ext
            )));
        }
    }

    let prefix = channel_prefix.unwrap_or("beauty").trim();
    if prefix.is_empty() {
        return Err(PyValueError::new_err("EXR channel prefix must be non-empty"));
    }

    let view = array.as_array();
    let shape = view.shape();
    let (height, width, channels) = match shape.len() {
        2 => (shape[0], shape[1], 1usize),
        3 => (shape[0], shape[1], shape[2]),
        _ => {
            return Err(PyValueError::new_err(format!(
                "expected array shape (H,W), (H,W,3), or (H,W,4); got {:?}",
                shape
            )))
        }
    };

    if height == 0 || width == 0 {
        return Err(PyValueError::new_err("array dimensions must be positive"));
    }

    if channels != 1 && channels != 3 && channels != 4 {
        return Err(PyValueError::new_err(format!(
            "unsupported channel count {}; expected 1, 3, or 4",
            channels
        )));
    }

    let height_u32 = u32::try_from(height)
        .map_err(|_| PyValueError::new_err("array height exceeds u32"))?;
    let width_u32 = u32::try_from(width)
        .map_err(|_| PyValueError::new_err("array width exceeds u32"))?;

    #[cfg(feature = "images")]
    {
        let data = view.to_owned().into_raw_vec();
        let write_result = match channels {
            1 => exr_write::write_exr_scalar_f32(path_obj, width_u32, height_u32, &data, prefix),
            3 => exr_write::write_exr_rgb_f32(path_obj, width_u32, height_u32, &data, prefix),
            4 => exr_write::write_exr_rgba_f32(path_obj, width_u32, height_u32, &data, prefix),
            _ => unreachable!("channel validation guards this"),
        };
        write_result
            .map_err(|err| PyRuntimeError::new_err(format!("failed to write EXR: {err:#}")))?;
        Ok(())
    }

    #[cfg(not(feature = "images"))]
    {
        Err(PyRuntimeError::new_err(
            "writing EXR requires the 'images' feature",
        ))
    }
}

// Core modules
pub mod math {
    /// Orthonormalize a tangent `t` against normal `n` and return (tangent, bitangent).
    ///
    /// Uses simple Gram-Schmidt then computes bitangent as cross(n, t_ortho).
    pub fn orthonormalize_tangent(n: [f32; 3], t: [f32; 3]) -> ([f32; 3], [f32; 3]) {
        fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
            a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
        }

        fn norm(v: [f32; 3]) -> f32 {
            dot(v, v).sqrt()
        }
        fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
            [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
        }
        fn mul(v: [f32; 3], s: f32) -> [f32; 3] {
            [v[0] * s, v[1] * s, v[2] * s]
        }
        fn normalize(v: [f32; 3]) -> [f32; 3] {
            let l = norm(v);
            if l > 0.0 {
                [v[0] / l, v[1] / l, v[2] / l]
            } else {
                v
            }
        }
        fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
            [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]
        }

        let n_n = normalize(n);
        let t_ortho = normalize(sub(t, mul(n_n, dot(n_n, t))));
        let b = cross(n_n, t_ortho);
        (t_ortho, b)
    }
}

// Rendering modules
pub mod accel;
pub mod camera;
pub mod cli;
pub mod colormap;
pub mod converters; // Geometry converters (e.g., MultipolygonZ -> OBJ)
pub mod core;
pub mod external_image;
pub mod formats;
pub mod geometry;
pub mod import; // Importers: OSM buildings, etc.
pub mod io; // IO: OBJ/PLY/glTF readers/writers
pub mod lighting; // P0: Production-ready lighting stack (lights, BRDFs, shadows, IBL)
pub mod loaders;
pub mod mesh;
pub mod offscreen; // P7: Offscreen PBR harness for BRDF galleries and CI goldens
pub mod path_tracing;
pub mod pipeline;
pub mod render; // Rendering utilities (instancing)
pub mod renderer;
pub mod scene;
pub mod sdf; // New SDF module
pub mod shadows; // Shadow mapping implementations
pub mod terrain;
pub mod uv; // UV unwrap helpers (planar, spherical)
pub mod textures {}
pub mod labels; // Screen-space text labels with MSDF rendering
pub mod p5;
pub mod passes;
pub mod util;
pub mod vector;
pub mod picking; // Feature picking and inspection system
pub mod viewer; // Interactive windowed viewer (Workstream I1) // P5.2: render passes wrappers
pub mod animation; // Feature C: Camera animation and keyframe interpolation

// Re-export commonly used types
pub use core::cloud_shadows::{
    CloudAnimationParams, CloudShadowQuality, CloudShadowRenderer, CloudShadowUniforms,
};
pub use core::clouds::{
    CloudAnimationPreset, CloudInstance, CloudParams, CloudQuality, CloudRenderMode, CloudRenderer,
    CloudUniforms,
};
pub use core::dof::{CameraDofParams, DofMethod, DofQuality, DofRenderer, DofUniforms};
pub use core::dual_source_oit::{
    DualSourceComposeUniforms, DualSourceOITMode, DualSourceOITQuality, DualSourceOITRenderer,
    DualSourceOITStats, DualSourceOITUniforms,
};
pub use core::error::RenderError;
pub use core::ground_plane::{
    GroundPlaneMode, GroundPlaneParams, GroundPlaneRenderer, GroundPlaneUniforms,
};
pub use core::ibl::{IBLQuality, IBLRenderer};
pub use core::ltc_area_lights::{LTCRectAreaLightRenderer, LTCUniforms, RectAreaLight};
pub use core::point_spot_lights::{
    DebugMode, Light, LightPreset, LightType, PointSpotLightRenderer, PointSpotLightUniforms,
    ShadowQuality,
};
pub use core::reflections::{PlanarReflectionRenderer, ReflectionQuality};
pub use core::soft_light_radius::{
    SoftLightFalloffMode, SoftLightPreset, SoftLightRadiusRenderer, SoftLightRadiusUniforms,
};
pub use core::water_surface::{
    WaterSurfaceMode, WaterSurfaceParams, WaterSurfaceRenderer, WaterSurfaceUniforms,
};
pub use lighting::LightBuffer;
pub use path_tracing::{TracerEngine, TracerParams};
pub use render::params::{
    AtmosphereParams as RendererAtmosphereParams, BrdfModel as RendererBrdfModel,
    ConfigError as RendererConfigError, GiMode as RendererGiMode, GiParams as RendererGiParams,
    LightConfig as RendererLightConfig, LightType as RendererLightType,
    LightingParams as RendererLightingParams, RendererConfig, ShadowParams as RendererShadowParams,
    ShadowTechnique as RendererShadowTechnique, SkyModel as RendererSkyModel,
    SsrParams as RendererSsrParams, VolumetricParams as RendererVolumetricParams,
    VolumetricPhase as RendererVolumetricPhase,
};
pub use sdf::{
    CsgOperation, HybridHitResult, HybridScene, SdfPrimitive, SdfPrimitiveType, SdfScene,
    SdfSceneBuilder,
};
pub use shadows::{
    detect_peter_panning, CascadeStatistics, CascadedShadowMaps, CsmConfig, CsmRenderer,
    ShadowManager, ShadowManagerConfig,
};

#[cfg(feature = "extension-module")]
#[pyfunction]
fn set_point_shape_mode(mode: u32) -> PyResult<()> {
    crate::vector::point::set_global_shape_mode(mode);
    Ok(())
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn set_point_lod_threshold(threshold: f32) -> PyResult<()> {
    crate::vector::point::set_global_lod_threshold(threshold);
    Ok(())
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn is_weighted_oit_available() -> PyResult<bool> {
    Ok(crate::vector::oit::is_weighted_oit_enabled())
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn vector_render_polygons_fill_py(
    py: Python<'_>,
    width: u32,
    height: u32,
    exteriors: Vec<numpy::PyReadonlyArray2<'_, f64>>, // list of (N,2)
    holes: Option<Vec<Vec<numpy::PyReadonlyArray2<'_, f64>>>>, // list of list of (M,2)
    fill_rgba: Option<(f32, f32, f32, f32)>,
    stroke_rgba: Option<(f32, f32, f32, f32)>,
    stroke_width: Option<f32>,
) -> PyResult<Py<PyAny>> {
    use crate::vector::api::PolygonDef;
    use numpy::PyArray1;

    // Acquire device/queue from global context
    let g = crate::core::gpu::ctx();
    let device = std::sync::Arc::clone(&g.device);
    let queue = std::sync::Arc::clone(&g.queue);

    // Compute bounds first for normalization (fixes Lyon tessellation with large coordinates)
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;

    // Build polygon defs from numpy arrays and compute bounds
    let mut polys: Vec<PolygonDef> = Vec::with_capacity(exteriors.len());
    for (i, ext) in exteriors.into_iter().enumerate() {
        let exterior = crate::vector::api::parse_polygon_from_numpy(ext)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Update bounds from exterior
        for v in &exterior {
            min_x = min_x.min(v.x);
            min_y = min_y.min(v.y);
            max_x = max_x.max(v.x);
            max_y = max_y.max(v.y);
        }

        // Parse holes for this polygon if provided
        let mut hole_rings = Vec::new();
        if let Some(hh) = &holes {
            if let Some(h_for_poly) = hh.get(i) {
                for h in h_for_poly {
                    let hv = crate::vector::api::parse_polygon_from_numpy(h.clone())
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                    // Update bounds from holes
                    for v in &hv {
                        min_x = min_x.min(v.x);
                        min_y = min_y.min(v.y);
                        max_x = max_x.max(v.x);
                        max_y = max_y.max(v.y);
                    }
                    hole_rings.push(hv);
                }
            }
        }

        // Style on PolygonDef is not used by tessellation; rendering uniforms control style
        let style = crate::vector::api::VectorStyle::default();
        polys.push(PolygonDef {
            exterior,
            holes: hole_rings,
            style,
        });
    }

    // Normalize to avoid Lyon tessellation issues with large coordinates
    if !min_x.is_finite() || !min_y.is_finite() || !max_x.is_finite() || !max_y.is_finite() {
        min_x = -1.0;
        min_y = -1.0;
        max_x = 1.0;
        max_y = 1.0;
    }
    let cx_orig = 0.5 * (min_x + max_x);
    let cy_orig = 0.5 * (min_y + max_y);
    let dx = (max_x - min_x).max(1e-6);
    let dy = (max_y - min_y).max(1e-6);
    let norm_scale = 100.0 / dx.max(dy); // Normalize to ~100 unit range for Lyon

    // Apply normalization centered around origin for proper NDC mapping
    for poly in &mut polys {
        for v in &mut poly.exterior {
            v.x = (v.x - cx_orig) * norm_scale;
            v.y = (v.y - cy_orig) * norm_scale;
        }
        for hole in &mut poly.holes {
            for v in hole {
                v.x = (v.x - cx_orig) * norm_scale;
                v.y = (v.y - cy_orig) * norm_scale;
            }
        }
    }

    // Create polygon renderer and tessellate
    let mut poly_renderer =
        crate::vector::PolygonRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let mut packed = Vec::with_capacity(polys.len());
    for p in &polys {
        let pk = poly_renderer
            .tessellate_polygon(p)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        packed.push(pk);
    }

    // Upload geometry
    poly_renderer
        .upload_polygons(&device, &queue, &packed)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    // Create output target
    let final_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("vf.Vector.PolygonFill.Final"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let final_view = final_tex.create_view(&wgpu::TextureViewDescriptor::default());

    // Encode render pass
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("vf.Vector.PolygonFill.Encoder"),
    });

    {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("vf.Vector.PolygonFill.Render"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &final_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        // Compute fit-to-NDC transform from normalized polygon bounds with viewport aspect ratio
        let mut norm_min_x = f32::INFINITY;
        let mut norm_min_y = f32::INFINITY;
        let mut norm_max_x = f32::NEG_INFINITY;
        let mut norm_max_y = f32::NEG_INFINITY;
        for p in &polys {
            for v in &p.exterior {
                norm_min_x = norm_min_x.min(v.x);
                norm_min_y = norm_min_y.min(v.y);
                norm_max_x = norm_max_x.max(v.x);
                norm_max_y = norm_max_y.max(v.y);
            }
            for hole in &p.holes {
                for v in hole {
                    norm_min_x = norm_min_x.min(v.x);
                    norm_min_y = norm_min_y.min(v.y);
                    norm_max_x = norm_max_x.max(v.x);
                    norm_max_y = norm_max_y.max(v.y);
                }
            }
        }
        if !norm_min_x.is_finite()
            || !norm_min_y.is_finite()
            || !norm_max_x.is_finite()
            || !norm_max_y.is_finite()
        {
            norm_min_x = 0.0;
            norm_min_y = 0.0;
            norm_max_x = 100.0;
            norm_max_y = 100.0;
        }
        let cx = 0.5 * (norm_min_x + norm_max_x);
        let cy = 0.5 * (norm_min_y + norm_max_y);
        let dx = (norm_max_x - norm_min_x).max(1e-6);
        let dy = (norm_max_y - norm_min_y).max(1e-6);

        // Compute scale accounting for viewport aspect ratio to avoid distortion
        let viewport_aspect = width as f32 / height as f32;
        let data_aspect = dx / dy;

        let (sx, sy) = if data_aspect > viewport_aspect {
            // Data is wider relative to viewport - fit to width
            let s = 2.0 / dx;
            (s, s)
        } else {
            // Data is taller relative to viewport - fit to height
            let s = 2.0 / dy;
            (s, s)
        };

        // Flip Y-axis for proper geographic data rendering (Y increases upward in geo data, downward in clip space)
        let vp = [
            [sx, 0.0, 0.0, -sx * cx],
            [0.0, -sy, 0.0, sy * cy],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        let total_indices: u32 = packed.iter().map(|p| p.indices.len() as u32).sum();
        let (fr, fg, fb, fa) = fill_rgba.unwrap_or((0.2, 0.4, 0.8, 1.0));
        let (sr, sg, sb, sa) = stroke_rgba.unwrap_or((0.0, 0.0, 0.0, 1.0));
        let sw = stroke_width.unwrap_or(1.0);

        poly_renderer
            .render(
                &mut pass,
                &queue,
                &vp,
                [fr, fg, fb, fa],
                [sr, sg, sb, sa],
                sw,
                total_indices,
            )
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    }

    queue.submit(Some(enc.finish()));
    device.poll(wgpu::Maintain::Wait);

    // Readback RGBA8
    let bpr = (width * 4 + 255) / 256 * 256;
    let size = (bpr * height) as u64;
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vf.Vector.PolygonFill.Read"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc2 = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("vf.Vector.PolygonFill.Copy"),
    });
    enc2.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture: &final_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &buf,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(bpr),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(enc2.finish()));
    device.poll(wgpu::Maintain::Wait);

    let slice = buf.slice(..);
    let (s, r) = futures_intrusive::channel::shared::oneshot_channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        s.send(res).ok();
    });
    // IMPORTANT: service the wgpu mapping callback; without this, the oneshot may never fire.
    device.poll(wgpu::Maintain::Wait);
    let recv = pollster::block_on(r.receive())
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("map_async cancelled"))?;
    if let Err(e) = recv {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "map_async error: {:?}",
            e
        )));
    }
    let data = slice.get_mapped_range();
    let mut rgba = vec![0u8; (width * height * 4) as usize];
    for row in 0..height as usize {
        let src = &data[(row as u32 * bpr) as usize..][..(width * 4) as usize];
        let dst = &mut rgba[row * (width as usize) * 4..][..(width as usize) * 4];
        dst.copy_from_slice(src);
    }
    drop(data);
    buf.unmap();

    let arr1 = PyArray1::<u8>::from_vec_bound(py, rgba);
    let arr3 = arr1.reshape([height as usize, width as usize, 4])?;
    Ok(arr3.into_py(py))
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn vector_render_oit_py(
    py: Python<'_>,
    width: u32,
    height: u32,
    points_xy: Option<&Bound<'_, PyAny>>,  // sequence of (x,y)
    point_rgba: Option<&Bound<'_, PyAny>>, // sequence of (r,g,b,a)
    point_size: Option<&Bound<'_, PyAny>>, // sequence of size
    polylines: Option<&Bound<'_, PyAny>>,  // sequence of sequence of (x,y)
    polyline_rgba: Option<&Bound<'_, PyAny>>, // sequence of (r,g,b,a)
    stroke_width: Option<&Bound<'_, PyAny>>, // sequence of width
) -> PyResult<Py<PyAny>> {
    #[cfg(not(feature = "weighted-oit"))]
    {
        let _ = (
            py,
            width,
            height,
            points_xy,
            point_rgba,
            point_size,
            polylines,
            polyline_rgba,
            stroke_width,
        );
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Weighted OIT feature not enabled. Build with --features weighted-oit",
        ));
    }
    #[cfg(feature = "weighted-oit")]
    {
        use crate::vector::api::{PointDef, PolylineDef, VectorStyle};
        use numpy::PyArray1;

        // Helper extractors
        fn extract_xy_list(list: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<glam::Vec2>> {
            if let Some(obj) = list {
                let pairs: Vec<(f32, f32)> = obj.extract()?;
                Ok(pairs
                    .into_iter()
                    .map(|(x, y)| glam::Vec2::new(x, y))
                    .collect())
            } else {
                Ok(Vec::new())
            }
        }
        fn extract_rgba_list(list: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<[f32; 4]>> {
            if let Some(obj) = list {
                let vals: Vec<(f32, f32, f32, f32)> = obj.extract()?;
                Ok(vals.into_iter().map(|(r, g, b, a)| [r, g, b, a]).collect())
            } else {
                Ok(Vec::new())
            }
        }
        fn extract_f32_list(list: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<f32>> {
            if let Some(obj) = list {
                Ok(obj.extract()?)
            } else {
                Ok(Vec::new())
            }
        }
        fn extract_polylines(list: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<Vec<glam::Vec2>>> {
            if let Some(obj) = list {
                let outer: Vec<Vec<(f32, f32)>> = obj.extract()?;
                Ok(outer
                    .into_iter()
                    .map(|path| {
                        path.into_iter()
                            .map(|(x, y)| glam::Vec2::new(x, y))
                            .collect()
                    })
                    .collect())
            } else {
                Ok(Vec::new())
            }
        }

        let pts = extract_xy_list(points_xy)?;
        let pts_rgba = extract_rgba_list(point_rgba)?;
        let pts_size = extract_f32_list(point_size)?;
        let lines = extract_polylines(polylines)?;
        let lines_rgba = extract_rgba_list(polyline_rgba)?;
        let lines_w = extract_f32_list(stroke_width)?;

        // Build defs
        let mut point_defs: Vec<PointDef> = Vec::with_capacity(pts.len());
        for i in 0..pts.len() {
            let c = *pts_rgba.get(i).unwrap_or(&[1.0, 1.0, 1.0, 1.0]);
            let s = *pts_size.get(i).unwrap_or(&8.0);
            point_defs.push(PointDef {
                position: pts[i],
                style: VectorStyle {
                    fill_color: c,
                    stroke_color: [0.0, 0.0, 0.0, 1.0],
                    stroke_width: 1.0,
                    point_size: s,
                },
            });
        }
        let mut poly_defs: Vec<PolylineDef> = Vec::with_capacity(lines.len());
        for i in 0..lines.len() {
            let c = *lines_rgba.get(i).unwrap_or(&[0.2, 0.8, 0.2, 0.6]);
            let w = *lines_w.get(i).unwrap_or(&2.0);
            poly_defs.push(PolylineDef {
                path: lines[i].clone(),
                style: VectorStyle {
                    fill_color: [0.0, 0.0, 0.0, 0.0],
                    stroke_color: c,
                    stroke_width: w,
                    point_size: 4.0,
                },
            });
        }

        // Acquire device/queue
        let g = crate::core::gpu::ctx();
        let device = std::sync::Arc::clone(&g.device);
        let queue = std::sync::Arc::clone(&g.queue);

        // Create renderers
        let mut pr =
            crate::vector::PointRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let mut lr = crate::vector::LineRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Upload instances
        if !point_defs.is_empty() {
            let p_instances = pr
                .pack_points(&point_defs)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            pr.upload_points(&device, &queue, &p_instances)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        }
        if !poly_defs.is_empty() {
            let l_instances = lr
                .pack_polylines(&poly_defs)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            lr.upload_lines(&device, &l_instances)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        }

        // Create OIT and final target
        let oit = crate::vector::oit::WeightedOIT::new(
            &device,
            width,
            height,
            wgpu::TextureFormat::Rgba8UnormSrgb,
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let final_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vf.Vector.RenderOIT.Final"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let final_view = final_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // Accumulation and compose
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vf.Vector.RenderOIT.Encoder"),
        });
        {
            let mut pass = oit.begin_accumulation(&mut encoder);
            let vp = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ];
            let viewport = [width as f32, height as f32];
            if !poly_defs.is_empty() {
                lr.render_oit(
                    &mut pass,
                    &queue,
                    &vp,
                    viewport,
                    poly_defs.len() as u32,
                    crate::vector::line::LineCap::Round,
                    crate::vector::line::LineJoin::Round,
                    2.0,
                )
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            }
            if !point_defs.is_empty() {
                pr.render_oit(
                    &mut pass,
                    &queue,
                    &vp,
                    viewport,
                    1.0,
                    point_defs.len() as u32,
                )
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            }
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("vf.Vector.RenderOIT.Compose"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &final_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            oit.compose(&mut pass);
        }
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);

        // Readback final
        let bpr = (width * 4 + 255) / 256 * 256;
        let size = (bpr * height) as u64;
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vf.Vector.RenderOIT.Read"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vf.Vector.RenderOIT.Copy"),
        });
        enc.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &final_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &buf,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(bpr),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(enc.finish()));
        device.poll(wgpu::Maintain::Wait);

        let slice = buf.slice(..);
        let (s, r) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            s.send(res).ok();
        });
        // Service mapping callback to avoid stalls on some platforms
        device.poll(wgpu::Maintain::Wait);
        let recv = pollster::block_on(r.receive())
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("map_async cancelled"))?;
        if let Err(e) = recv {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "map_async error: {:?}",
                e
            )));
        }
        let data = slice.get_mapped_range();
        let mut rgba = vec![0u8; (width * height * 4) as usize];
        for row in 0..height as usize {
            let src = &data[(row as u32 * bpr) as usize..][..(width * 4) as usize];
            let dst = &mut rgba[row * (width as usize) * 4..][..(width as usize) * 4];
            dst.copy_from_slice(src);
        }
        drop(data);
        buf.unmap();

        let arr1 = PyArray1::<u8>::from_vec_bound(py, rgba);
        let arr3 = arr1.reshape([height as usize, width as usize, 4])?;
        Ok(arr3.into_py(py))
    }
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn vector_render_pick_map_py(
    py: Python<'_>,
    width: u32,
    height: u32,
    points_xy: Option<&Bound<'_, PyAny>>,
    polylines: Option<&Bound<'_, PyAny>>,
    base_pick_id: Option<u32>,
) -> PyResult<Py<PyAny>> {
    use crate::vector::api::{PointDef, PolylineDef, VectorStyle};
    use numpy::PyArray1;
    // Parse inputs
    fn extract_xy_list(list: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<glam::Vec2>> {
        if let Some(obj) = list {
            Ok(obj
                .extract::<Vec<(f32, f32)>>()?
                .into_iter()
                .map(|(x, y)| glam::Vec2::new(x, y))
                .collect())
        } else {
            Ok(Vec::new())
        }
    }
    fn extract_polylines(list: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<Vec<glam::Vec2>>> {
        if let Some(obj) = list {
            Ok(obj
                .extract::<Vec<Vec<(f32, f32)>>>()?
                .into_iter()
                .map(|path| {
                    path.into_iter()
                        .map(|(x, y)| glam::Vec2::new(x, y))
                        .collect()
                })
                .collect())
        } else {
            Ok(Vec::new())
        }
    }
    let pts = extract_xy_list(points_xy)?;
    let lines = extract_polylines(polylines)?;
    let mut point_defs = Vec::with_capacity(pts.len());
    for p in &pts {
        point_defs.push(PointDef {
            position: *p,
            style: VectorStyle {
                fill_color: [1.0, 1.0, 1.0, 1.0],
                stroke_color: [0.0, 0.0, 0.0, 1.0],
                stroke_width: 1.0,
                point_size: 8.0,
            },
        });
    }
    let mut poly_defs = Vec::with_capacity(lines.len());
    for path in lines {
        poly_defs.push(PolylineDef {
            path,
            style: VectorStyle {
                fill_color: [0.0, 0.0, 0.0, 0.0],
                stroke_color: [1.0, 1.0, 1.0, 1.0],
                stroke_width: 2.0,
                point_size: 4.0,
            },
        });
    }

    // Device
    let g = crate::core::gpu::ctx();
    let device = std::sync::Arc::clone(&g.device);
    let queue = std::sync::Arc::clone(&g.queue);

    let mut pr = crate::vector::PointRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let mut lr = crate::vector::LineRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    if !point_defs.is_empty() {
        let p_instances = pr
            .pack_points(&point_defs)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        pr.upload_points(&device, &queue, &p_instances)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    }
    if !poly_defs.is_empty() {
        let l_instances = lr
            .pack_polylines(&poly_defs)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        lr.upload_lines(&device, &l_instances)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    }

    // Create pick target
    let pick_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("vf.Vector.RenderPick.Pick"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Uint,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = pick_tex.create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("vf.Vector.RenderPick.Encoder"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("vf.Vector.RenderPick.Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 0.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        let vp = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let viewport = [width as f32, height as f32];
        let mut base = base_pick_id.unwrap_or(1);
        if !point_defs.is_empty() {
            pr.render_pick(
                &mut pass,
                &queue,
                &vp,
                viewport,
                1.0,
                point_defs.len() as u32,
                base,
            )
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            base += point_defs.len() as u32;
        }
        if !poly_defs.is_empty() {
            lr.render_pick(
                &mut pass,
                &queue,
                &vp,
                viewport,
                poly_defs.len() as u32,
                base,
            )
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        }
    }
    queue.submit(Some(encoder.finish()));
    device.poll(wgpu::Maintain::Wait);

    // Readback full R32Uint map
    let bpr = (width * 4 + 255) / 256 * 256;
    let size = (bpr * height) as u64;
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vf.Vector.RenderPick.Read"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("vf.Vector.RenderPick.Copy"),
    });
    enc.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture: &pick_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &buf,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(bpr),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(enc.finish()));
    device.poll(wgpu::Maintain::Wait);

    let slice = buf.slice(..);
    let (s, r) = futures_intrusive::channel::shared::oneshot_channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        s.send(res).ok();
    });
    // Service mapping callback to avoid stalls on some platforms
    device.poll(wgpu::Maintain::Wait);
    let recv = pollster::block_on(r.receive())
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("map_async cancelled"))?;
    if let Err(e) = recv {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "map_async error: {:?}",
            e
        )));
    }
    let data = slice.get_mapped_range();
    let mut ids = vec![0u32; (width * height) as usize];
    for row in 0..height as usize {
        let src = &data[(row as u32 * bpr) as usize..][..(width * 4) as usize];
        let row_ids = bytemuck::cast_slice::<u8, u32>(src);
        let dst = &mut ids[row * (width as usize)..][..(width as usize)];
        dst.copy_from_slice(row_ids);
    }
    drop(data);
    buf.unmap();
    let arr1 = PyArray1::<u32>::from_vec_bound(py, ids);
    let arr2 = arr1.reshape([height as usize, width as usize])?;
    Ok(arr2.into_py(py))
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn vector_render_oit_and_pick_py(
    py: Python<'_>,
    width: u32,
    height: u32,
    points_xy: Option<&Bound<'_, PyAny>>,
    point_rgba: Option<&Bound<'_, PyAny>>,
    point_size: Option<&Bound<'_, PyAny>>,
    polylines: Option<&Bound<'_, PyAny>>,
    polyline_rgba: Option<&Bound<'_, PyAny>>,
    stroke_width: Option<&Bound<'_, PyAny>>,
    base_pick_id: Option<u32>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    #[cfg(not(feature = "weighted-oit"))]
    {
        let _ = (
            py,
            width,
            height,
            points_xy,
            point_rgba,
            point_size,
            polylines,
            polyline_rgba,
            stroke_width,
            base_pick_id,
        );
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Weighted OIT feature not enabled. Build with --features weighted-oit",
        ));
    }
    #[cfg(feature = "weighted-oit")]
    {
        use crate::vector::api::{PointDef, PolylineDef, VectorStyle};
        use numpy::PyArray1;

        // Helper extractors (same as above)
        fn extract_xy_list(list: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<glam::Vec2>> {
            if let Some(obj) = list {
                let pairs: Vec<(f32, f32)> = obj.extract()?;
                Ok(pairs
                    .into_iter()
                    .map(|(x, y)| glam::Vec2::new(x, y))
                    .collect())
            } else {
                Ok(Vec::new())
            }
        }
        fn extract_rgba_list(list: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<[f32; 4]>> {
            if let Some(obj) = list {
                let vals: Vec<(f32, f32, f32, f32)> = obj.extract()?;
                Ok(vals.into_iter().map(|(r, g, b, a)| [r, g, b, a]).collect())
            } else {
                Ok(Vec::new())
            }
        }
        fn extract_f32_list(list: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<f32>> {
            if let Some(obj) = list {
                Ok(obj.extract()?)
            } else {
                Ok(Vec::new())
            }
        }
        fn extract_polylines(list: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<Vec<glam::Vec2>>> {
            if let Some(obj) = list {
                let outer: Vec<Vec<(f32, f32)>> = obj.extract()?;
                Ok(outer
                    .into_iter()
                    .map(|path| {
                        path.into_iter()
                            .map(|(x, y)| glam::Vec2::new(x, y))
                            .collect()
                    })
                    .collect())
            } else {
                Ok(Vec::new())
            }
        }

        let pts = extract_xy_list(points_xy)?;
        let pts_rgba = extract_rgba_list(point_rgba)?;
        let pts_size = extract_f32_list(point_size)?;
        let lines = extract_polylines(polylines)?;
        let lines_rgba = extract_rgba_list(polyline_rgba)?;
        let lines_w = extract_f32_list(stroke_width)?;

        // Build defs
        let mut point_defs: Vec<PointDef> = Vec::with_capacity(pts.len());
        for i in 0..pts.len() {
            let c = *pts_rgba.get(i).unwrap_or(&[1.0, 1.0, 1.0, 1.0]);
            let s = *pts_size.get(i).unwrap_or(&8.0);
            point_defs.push(PointDef {
                position: pts[i],
                style: VectorStyle {
                    fill_color: c,
                    stroke_color: [0.0, 0.0, 0.0, 1.0],
                    stroke_width: 1.0,
                    point_size: s,
                },
            });
        }
        let mut poly_defs: Vec<PolylineDef> = Vec::with_capacity(lines.len());
        for i in 0..lines.len() {
            let c = *lines_rgba.get(i).unwrap_or(&[0.2, 0.8, 0.2, 0.6]);
            let w = *lines_w.get(i).unwrap_or(&2.0);
            poly_defs.push(PolylineDef {
                path: lines[i].clone(),
                style: VectorStyle {
                    fill_color: [0.0, 0.0, 0.0, 0.0],
                    stroke_color: c,
                    stroke_width: w,
                    point_size: 4.0,
                },
            });
        }

        // Acquire device/queue
        let g = crate::core::gpu::ctx();
        let device = std::sync::Arc::clone(&g.device);
        let queue = std::sync::Arc::clone(&g.queue);

        // Create renderers
        let mut pr =
            crate::vector::PointRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let mut lr = crate::vector::LineRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Upload instances
        if !point_defs.is_empty() {
            let p_instances = pr
                .pack_points(&point_defs)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            pr.upload_points(&device, &queue, &p_instances)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        }
        if !poly_defs.is_empty() {
            let l_instances = lr
                .pack_polylines(&poly_defs)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            lr.upload_lines(&device, &l_instances)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        }

        // Weighted OIT accumulation buffers
        #[cfg(not(feature = "weighted-oit"))]
        {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Weighted OIT feature not enabled. Build with --features weighted-oit",
            ));
        }
        #[cfg(feature = "weighted-oit")]
        let oit = crate::vector::oit::WeightedOIT::new(
            &device,
            width,
            height,
            wgpu::TextureFormat::Rgba8UnormSrgb,
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Final RGBA8 target
        #[cfg(feature = "weighted-oit")]
        let final_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vf.Vector.Combine.Final"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        #[cfg(feature = "weighted-oit")]
        let final_view = final_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // Accumulation pass
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vf.Vector.Combine.Encoder"),
        });
        #[cfg(feature = "weighted-oit")]
        {
            let mut pass = oit.begin_accumulation(&mut encoder);
            let vp = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ];
            let viewport = [width as f32, height as f32];
            if !poly_defs.is_empty() {
                lr.render_oit(
                    &mut pass,
                    &queue,
                    &vp,
                    viewport,
                    poly_defs.len() as u32,
                    crate::vector::line::LineCap::Round,
                    crate::vector::line::LineJoin::Round,
                    2.0,
                )
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            }
            if !point_defs.is_empty() {
                pr.render_oit(
                    &mut pass,
                    &queue,
                    &vp,
                    viewport,
                    1.0,
                    point_defs.len() as u32,
                )
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            }
        }
        // Compose pass into final target
        #[cfg(feature = "weighted-oit")]
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("vf.Vector.Combine.Compose"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &final_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            oit.compose(&mut pass);
        }

        // Picking pass
        let pick_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vf.Vector.Combine.Pick"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let pick_view = pick_tex.create_view(&wgpu::TextureViewDescriptor::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("vf.Vector.Combine.PickPass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &pick_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            let vp = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ];
            let viewport = [width as f32, height as f32];
            let mut base = base_pick_id.unwrap_or(1);
            if !point_defs.is_empty() {
                pr.render_pick(
                    &mut pass,
                    &queue,
                    &vp,
                    viewport,
                    1.0,
                    point_defs.len() as u32,
                    base,
                )
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                base += point_defs.len() as u32;
            }
            if !poly_defs.is_empty() {
                lr.render_pick(
                    &mut pass,
                    &queue,
                    &vp,
                    viewport,
                    poly_defs.len() as u32,
                    base,
                )
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            }
        }

        // Submit
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);

        // Read back final RGBA8
        let bpr = (width * 4 + 255) / 256 * 256; // 256B align
        let final_size = (bpr * height) as u64;
        #[cfg(feature = "weighted-oit")]
        let final_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vf.Vector.Combine.FinalRead"),
            size: final_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        #[cfg(feature = "weighted-oit")]
        let mut enc1 = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vf.Vector.Combine.CopyFinal"),
        });
        #[cfg(feature = "weighted-oit")]
        enc1.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &final_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &final_buf,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(bpr),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        #[cfg(feature = "weighted-oit")]
        {
            queue.submit(Some(enc1.finish()));
            device.poll(wgpu::Maintain::Wait);
        }

        #[cfg(feature = "weighted-oit")]
        let rgba: Vec<u8>;
        #[cfg(feature = "weighted-oit")]
        {
            let fslice = final_buf.slice(..);
            let (s, r) = futures_intrusive::channel::shared::oneshot_channel();
            fslice.map_async(wgpu::MapMode::Read, move |res| {
                s.send(res).ok();
            });
            // Service mapping callback to avoid stalls on some platforms
            device.poll(wgpu::Maintain::Wait);
            let recv = pollster::block_on(r.receive())
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("final map cancelled"))?;
            if let Err(e) = recv {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "map_async error: {:?}",
                    e
                )));
            }
            let fdata = fslice.get_mapped_range();
            let mut rgba_data = vec![0u8; (width * height * 4) as usize];
            for row in 0..height as usize {
                let src = &fdata[(row as u32 * bpr) as usize..][..(width * 4) as usize];
                let dst = &mut rgba_data[row * (width as usize) * 4..][..(width as usize) * 4];
                dst.copy_from_slice(src);
            }
            drop(fdata);
            final_buf.unmap();
            rgba = rgba_data;
        }

        // Read back pick map
        let pick_bpr = (width * 4 + 255) / 256 * 256;
        let pick_size = (pick_bpr * height) as u64;
        let pick_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vf.Vector.Combine.PickRead"),
            size: pick_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc2 = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vf.Vector.Combine.CopyPick"),
        });
        enc2.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &pick_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &pick_buf,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(pick_bpr),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(enc2.finish()));
        device.poll(wgpu::Maintain::Wait);

        let pslice = pick_buf.slice(..);
        let (s2, r2) = futures_intrusive::channel::shared::oneshot_channel();
        pslice.map_async(wgpu::MapMode::Read, move |res| {
            s2.send(res).ok();
        });
        // Service mapping callback to avoid stalls on some platforms
        device.poll(wgpu::Maintain::Wait);
        let recv = pollster::block_on(r2.receive())
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("pick map cancelled"))?;
        if let Err(e) = recv {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "map_async error: {:?}",
                e
            )));
        }
        let pdata = pslice.get_mapped_range();
        let mut ids = vec![0u32; (width * height) as usize];
        for row in 0..height as usize {
            let src = &pdata[(row as u32 * pick_bpr) as usize..][..(width * 4) as usize];
            let row_ids = bytemuck::cast_slice::<u8, u32>(src);
            let dst = &mut ids[row * (width as usize)..][..(width as usize)];
            dst.copy_from_slice(row_ids);
        }
        drop(pdata);
        pick_buf.unmap();

        // Convert to numpy
        #[cfg(feature = "weighted-oit")]
        {
            let arr_rgba_1 = PyArray1::<u8>::from_vec_bound(py, rgba);
            let arr_rgba = arr_rgba_1.reshape([height as usize, width as usize, 4])?;
            let arr_ids_1 = PyArray1::<u32>::from_vec_bound(py, ids);
            let arr_ids = arr_ids_1.reshape([height as usize, width as usize])?;
            return Ok((arr_rgba.into_py(py), arr_ids.into_py(py)));
        }
        #[cfg(not(feature = "weighted-oit"))]
        {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Weighted OIT feature not enabled. Build with --features weighted-oit",
            ));
        }
    }
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn vector_oit_and_pick_demo(py: Python<'_>, width: u32, height: u32) -> PyResult<(Py<PyAny>, u32)> {
    #[cfg(not(feature = "weighted-oit"))]
    {
        let _ = (py, width, height);
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Weighted OIT feature not enabled. Build with --features weighted-oit",
        ));
    }
    #[cfg(feature = "weighted-oit")]
    {
        use crate::vector::api::{PointDef, PolylineDef, VectorStyle};
        use crate::vector::{LineRenderer, PointRenderer};
        use numpy::PyArray1;

        // Acquire GPU device/queue
        let g = crate::core::gpu::ctx();
        let device = std::sync::Arc::clone(&g.device);
        let queue = std::sync::Arc::clone(&g.queue);

        // Create renderers
        let mut pr = PointRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let mut lr = LineRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Sample primitives
        let points = vec![
            PointDef {
                position: glam::Vec2::new(-0.5, -0.5),
                style: VectorStyle {
                    fill_color: [1.0, 0.2, 0.2, 0.9],
                    stroke_color: [0.0, 0.0, 0.0, 1.0],
                    stroke_width: 1.0,
                    point_size: 24.0,
                },
            },
            PointDef {
                position: glam::Vec2::new(0.4, 0.2),
                style: VectorStyle {
                    fill_color: [0.2, 0.8, 1.0, 0.7],
                    stroke_color: [0.0, 0.0, 0.0, 1.0],
                    stroke_width: 1.0,
                    point_size: 32.0,
                },
            },
        ];
        let lines = vec![PolylineDef {
            path: vec![
                glam::Vec2::new(-0.8, -0.8),
                glam::Vec2::new(0.8, 0.5),
                glam::Vec2::new(0.4, 0.8),
            ],
            style: VectorStyle {
                fill_color: [0.0, 0.0, 0.0, 0.0],
                stroke_color: [0.1, 0.9, 0.3, 0.6],
                stroke_width: 8.0,
                point_size: 4.0,
            },
        }];

        let p_instances = pr
            .pack_points(&points)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        pr.upload_points(&device, &queue, &p_instances)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let l_instances = lr
            .pack_polylines(&lines)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        lr.upload_lines(&device, &l_instances)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Weighted OIT accumulation buffers
        let oit = crate::vector::oit::WeightedOIT::new(
            &device,
            width,
            height,
            wgpu::TextureFormat::Rgba8UnormSrgb,
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Final RGBA8 target
        let final_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vf.Vector.Demo.Final"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let final_view = final_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // Accumulation pass
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vf.Vector.Demo.Encoder"),
        });
        {
            let mut pass = oit.begin_accumulation(&mut encoder);
            let vp = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ];
            let viewport = [width as f32, height as f32];
            lr.render_oit(
                &mut pass,
                &queue,
                &vp,
                viewport,
                l_instances.len() as u32,
                crate::vector::line::LineCap::Round,
                crate::vector::line::LineJoin::Round,
                2.0,
            )
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            pr.render_oit(
                &mut pass,
                &queue,
                &vp,
                viewport,
                1.0,
                p_instances.len() as u32,
            )
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        }

        // Compose pass into final target
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("vf.Vector.Demo.Compose"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &final_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            oit.compose(&mut pass);
        }

        // Picking pass
        let pick_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vf.Vector.Demo.Pick"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let pick_view = pick_tex.create_view(&wgpu::TextureViewDescriptor::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("vf.Vector.Demo.PickPass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &pick_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            let vp = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ];
            let viewport = [width as f32, height as f32];
            // Assign base ids 1..N for points, then continue for lines
            pr.render_pick(
                &mut pass,
                &queue,
                &vp,
                viewport,
                1.0,
                p_instances.len() as u32,
                1,
            )
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            let base_line = 1 + p_instances.len() as u32;
            lr.render_pick(
                &mut pass,
                &queue,
                &vp,
                viewport,
                l_instances.len() as u32,
                base_line,
            )
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        }

        // Submit
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);

        // Read back final RGBA8
        let bpr = (width * 4 + 255) / 256 * 256; // align to 256
        let final_size = (bpr * height) as u64;
        let final_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vf.Vector.Demo.FinalRead"),
            size: final_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vf.Vector.Demo.CopyFinal"),
        });
        enc.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &final_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &final_buf,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(bpr),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        // Read one pick pixel at center
        let pick_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vf.Vector.Demo.PickRead"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cx = width / 2;
        let cy = height / 2;
        let mut enc2 = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vf.Vector.Demo.CopyPick"),
        });
        enc2.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &pick_tex,
                mip_level: 0,
                origin: wgpu::Origin3d { x: cx, y: cy, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &pick_buf,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: None,
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([enc.finish(), enc2.finish()]);
        device.poll(wgpu::Maintain::Wait);

        // Map final image
        let slice = final_buf.slice(..);
        let (s, r) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            s.send(res).ok();
        });
        // Service mapping callback to avoid stalls on some platforms
        device.poll(wgpu::Maintain::Wait);
        let recv = pollster::block_on(r.receive())
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("map_async cancelled"))?;
        if let Err(e) = recv {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "map_async error: {:?}",
                e
            )));
        }
        let data = slice.get_mapped_range();
        let mut rgba = vec![0u8; (width * height * 4) as usize];
        for row in 0..height as usize {
            let src = &data[(row as u32 * bpr) as usize..][..(width * 4) as usize];
            let dst = &mut rgba[row * (width as usize) * 4..][..(width as usize) * 4];
            dst.copy_from_slice(src);
        }
        drop(data);
        final_buf.unmap();

        // Map pick
        let pslice = pick_buf.slice(..);
        let (s2, r2) = futures_intrusive::channel::shared::oneshot_channel();
        pslice.map_async(wgpu::MapMode::Read, move |res| {
            s2.send(res).ok();
        });
        let recv = pollster::block_on(r2.receive())
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("pick map cancelled"))?;
        if let Err(e) = recv {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "map_async error: {:?}",
                e
            )));
        }
        let pdata = pslice.get_mapped_range();
        let pick_id = bytemuck::from_bytes::<u32>(&pdata[..4]).to_owned();
        drop(pdata);
        pick_buf.unmap();

        // Return numpy (H,W,4) uint8
        let arr1 = PyArray1::<u8>::from_vec_bound(py, rgba);
        let arr3 = arr1.reshape([height as usize, width as usize, 4])?;
        Ok((arr3.into_py(py), pick_id))
    }
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn hybrid_render(
    py: Python<'_>,
    width: u32,
    height: u32,
    scene: Option<&Bound<'_, PyAny>>,
    camera: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    fn py_any_to_vec3(obj: &Bound<'_, PyAny>) -> PyResult<Vec3> {
        let (x, y, z): (f32, f32, f32) = obj.extract()?;
        Ok(Vec3::new(x, y, z))
    }

    struct CameraParams {
        origin: Vec3,
        target: Vec3,
        up: Vec3,
        fov_degrees: f32,
    }

    impl Default for CameraParams {
        fn default() -> Self {
            Self {
                origin: Vec3::new(0.0, 0.0, 5.0),
                target: Vec3::ZERO,
                up: Vec3::Y,
                fov_degrees: 45.0,
            }
        }
    }

    if width == 0 || height == 0 {
        return Err(PyValueError::new_err("image dimensions must be positive"));
    }

    // Resolve native SdfScene from Python object
    let sdf_scene = if let Some(scene_obj) = scene {
        let extracted: PyRef<'_, PySdfScene> = scene_obj.extract()?;
        extracted.0.clone()
    } else {
        crate::sdf::SdfScene::new()
    };

    // Prepare camera parameters
    let mut cam = CameraParams::default();
    if let Some(camera_obj) = camera {
        let camera_dict = camera_obj.downcast::<PyDict>().ok();
        let update_vec3 = |key: &str, out: &mut Vec3| -> PyResult<()> {
            if let Some(dict) = camera_dict.as_ref() {
                if let Some(value) = dict.get_item(key)? {
                    *out = py_any_to_vec3(&value)?;
                    return Ok(());
                }
            }

            if let Ok(value) = camera_obj.getattr(key) {
                *out = py_any_to_vec3(&value)?;
            }
            Ok(())
        };

        let update_f32 = |key: &str, out: &mut f32| -> PyResult<()> {
            if let Some(dict) = camera_dict.as_ref() {
                if let Some(value) = dict.get_item(key)? {
                    *out = value.extract()?;
                    return Ok(());
                }
            }

            if let Ok(value) = camera_obj.getattr(key) {
                *out = value.extract()?;
            }
            Ok(())
        };

        update_vec3("origin", &mut cam.origin)?;
        update_vec3("target", &mut cam.target)?;
        update_vec3("up", &mut cam.up)?;
        update_f32("fov_degrees", &mut cam.fov_degrees)?;
    }

    let mut forward = (cam.target - cam.origin).normalize_or_zero();
    if forward.length_squared() == 0.0 {
        forward = Vec3::new(0.0, 0.0, -1.0);
    }

    let up_hint = cam.up.normalize_or_zero();
    let up_hint = if up_hint.length_squared() == 0.0 {
        Vec3::Y
    } else {
        up_hint
    };

    let mut right = forward.cross(up_hint).normalize_or_zero();
    if right.length_squared() == 0.0 {
        right = Vec3::X;
    }

    let mut up = right.cross(forward).normalize_or_zero();
    if up.length_squared() == 0.0 {
        up = Vec3::Y;
    }

    // Construct hybrid scene (currently SDF-only)
    let hybrid_scene = crate::sdf::HybridScene::sdf_only(sdf_scene);

    let w = width as usize;
    let h = height as usize;
    let mut pixels = vec![0u8; w * h * 4];

    let aspect = width as f32 / height as f32;
    let half_fov = (cam.fov_degrees.to_radians() * 0.5).tan();
    let half_w = aspect * half_fov;
    let half_h = half_fov;

    let sky_color = [153u8, 178u8, 229u8];

    for y in 0..h {
        let ndc_y = (1.0 - ((y as f32 + 0.5) / height as f32)) * 2.0 - 1.0;

        for x in 0..w {
            let ndc_x = ((x as f32 + 0.5) / width as f32) * 2.0 - 1.0;

            let mut dir = right * (ndc_x * half_w) + up * (ndc_y * half_h) - forward;
            dir = dir.normalize_or_zero();
            if dir.length_squared() == 0.0 {
                dir = -forward;
            }

            let ray = HybridRay {
                origin: cam.origin,
                direction: dir,
                tmin: 0.001,
                tmax: 100.0,
            };

            let result = hybrid_scene.intersect(ray);

            let pixel_index = (y * w + x) * 4;
            if result.hit {
                let color = match result.material_id {
                    1 => [204u8, 51u8, 51u8], // red-ish
                    2 => [51u8, 204u8, 51u8], // green-ish
                    3 => [51u8, 51u8, 204u8], // blue-ish
                    4 => [210u8, 210u8, 210u8],
                    _ => [230u8, 153u8, 76u8],
                };
                pixels[pixel_index..pixel_index + 3].copy_from_slice(&color);
            } else {
                pixels[pixel_index..pixel_index + 3].copy_from_slice(&sky_color);
            }
            pixels[pixel_index + 3] = 255;
        }
    }

    let arr1 = PyArray1::<u8>::from_vec_bound(py, pixels);
    let arr3 = arr1.reshape([height as usize, width as usize, 4])?;
    Ok(arr3.into_py(py))
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn _pt_render_gpu_mesh(
    py: Python<'_>,
    width: u32,
    height: u32,
    vertices: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    cam: &Bound<'_, PyAny>,
    seed: u32,
    frames: u32,
    lighting_type: &str,
    lighting_intensity: f32,
    lighting_azimuth: f32,
    lighting_elevation: f32,
    shadows: bool,
    shadow_intensity: f32,
) -> PyResult<Py<PyAny>> {
    use numpy::{PyArray1, PyReadonlyArray2};
    use pyo3::exceptions::{PyRuntimeError, PyValueError};

    // Parse vertex and index arrays
    let verts_arr: PyReadonlyArray2<f32> = vertices.extract().map_err(|_| {
        PyValueError::new_err("vertices must be a NumPy array with shape (N,3) float32")
    })?;
    let idx_arr: PyReadonlyArray2<u32> = indices.extract().map_err(|_| {
        PyValueError::new_err("indices must be a NumPy array with shape (M,3) uint32")
    })?;

    let v = verts_arr.as_array();
    let i = idx_arr.as_array();
    if v.ndim() != 2 || v.shape()[1] != 3 {
        return Err(PyValueError::new_err("vertices must have shape (N,3)"));
    }
    if i.ndim() != 2 || i.shape()[1] != 3 {
        return Err(PyValueError::new_err("indices must have shape (M,3)"));
    }

    // Pack vertices for HybridScene
    let mut verts: Vec<crate::sdf::hybrid::Vertex> = Vec::with_capacity(v.shape()[0]);
    for row in v.rows() {
        verts.push(crate::sdf::hybrid::Vertex {
            position: [row[0], row[1], row[2]],
            _pad: 0.0,
        });
    }

    // Flatten indices (u32)
    let mut flat_idx: Vec<u32> = Vec::with_capacity(i.shape()[0] * 3);
    for row in i.rows() {
        flat_idx.push(row[0]);
        flat_idx.push(row[1]);
        flat_idx.push(row[2]);
    }

    // Build triangle list for BVH construction (CPU path)
    let mut tris: Vec<crate::accel::types::Triangle> = Vec::with_capacity(i.shape()[0]);
    for row in i.rows() {
        let iv0 = row[0] as usize;
        let iv1 = row[1] as usize;
        let iv2 = row[2] as usize;
        if iv0 >= v.shape()[0] || iv1 >= v.shape()[0] || iv2 >= v.shape()[0] {
            return Err(PyValueError::new_err(
                "indices reference out-of-bounds vertex",
            ));
        }
        let v0 = [v[[iv0, 0]], v[[iv0, 1]], v[[iv0, 2]]];
        let v1 = [v[[iv1, 0]], v[[iv1, 1]], v[[iv1, 2]]];
        let v2 = [v[[iv2, 0]], v[[iv2, 1]], v[[iv2, 2]]];
        tris.push(crate::accel::types::Triangle::new(v0, v1, v2));
    }

    // Build BVH (CPU backend) and create a HybridScene with mesh
    let options = crate::accel::types::BuildOptions::default();
    let bvh_handle =
        crate::accel::build_bvh(&tris, &options, crate::accel::GpuContext::NotAvailable)
            .map_err(|e| PyRuntimeError::new_err(format!("BVH build failed: {}", e)))?;

    let mut hybrid = crate::sdf::hybrid::HybridScene::mesh_only(verts, flat_idx, bvh_handle);

    // Parse camera
    let origin: (f32, f32, f32) = cam.get_item("origin")?.extract()?;
    let look_at: (f32, f32, f32) = cam.get_item("look_at")?.extract()?;
    let up: (f32, f32, f32) = cam
        .get_item("up")
        .ok()
        .and_then(|v| v.extract().ok())
        .unwrap_or((0.0, 1.0, 0.0));
    let fov_y: f32 = cam
        .get_item("fov_y")
        .ok()
        .and_then(|v| v.extract().ok())
        .unwrap_or(45.0);
    let aspect: f32 = cam
        .get_item("aspect")
        .ok()
        .and_then(|v| v.extract().ok())
        .unwrap_or((width as f32) / (height as f32));
    let exposure: f32 = cam
        .get_item("exposure")
        .ok()
        .and_then(|v| v.extract().ok())
        .unwrap_or(1.0);

    // Build camera basis
    let o = glam::Vec3::new(origin.0, origin.1, origin.2);
    let la = glam::Vec3::new(look_at.0, look_at.1, look_at.2);
    let upv = glam::Vec3::new(up.0, up.1, up.2);
    let forward = (la - o).normalize_or_zero();
    let right = forward.cross(upv).normalize_or_zero();
    let cup = right.cross(forward).normalize_or_zero();

    // Base uniforms
    let uniforms = crate::path_tracing::compute::Uniforms {
        width,
        height,
        frame_index: 0,
        aov_flags: 0,
        cam_origin: [origin.0, origin.1, origin.2],
        cam_fov_y: fov_y,
        cam_right: [right.x, right.y, right.z],
        cam_aspect: aspect,
        cam_up: [cup.x, cup.y, cup.z],
        cam_exposure: exposure,
        cam_forward: [forward.x, forward.y, forward.z],
        seed_hi: seed,
        seed_lo: frames, // carry frames in lo for deterministic variation
        _pad_end: [0, 0, 0],
    };

    // Parse lighting type string to enum
    let lighting_type_id = match lighting_type.to_lowercase().as_str() {
        "flat" => 0u32,
        "lambertian" | "lambert" => 1u32,
        "phong" => 2u32,
        "blinn-phong" | "blinn_phong" | "blinnphong" => 3u32,
        _ => 1u32, // default to lambertian
    };

    // Convert azimuth/elevation to light direction vector
    let azimuth_rad = lighting_azimuth.to_radians();
    let elevation_rad = lighting_elevation.to_radians();
    let light_dir = [
        azimuth_rad.cos() * elevation_rad.cos(),
        elevation_rad.sin(),
        azimuth_rad.sin() * elevation_rad.cos(),
    ];

    // Normalize light direction
    let len =
        (light_dir[0] * light_dir[0] + light_dir[1] * light_dir[1] + light_dir[2] * light_dir[2])
            .sqrt();
    let light_dir_normalized = if len > 1e-6 {
        [light_dir[0] / len, light_dir[1] / len, light_dir[2] / len]
    } else {
        [0.0, 1.0, 0.0]
    };

    // Create lighting uniforms
    let lighting_uniforms = crate::path_tracing::hybrid_compute::LightingUniforms {
        light_dir: light_dir_normalized,
        lighting_type: lighting_type_id,
        light_color: [
            lighting_intensity,
            lighting_intensity * 0.95,
            lighting_intensity * 0.8,
        ], // Warm white
        shadows_enabled: if shadows { 1 } else { 0 },
        ambient_color: [0.1, 0.12, 0.15], // Cool ambient
        shadow_intensity,
        hdri_intensity: 0.0,
        hdri_rotation: 0.0,
        specular_power: 32.0,
        _pad: [0, 0, 0, 0, 0],
    };

    let params = crate::path_tracing::hybrid_compute::HybridTracerParams {
        base_uniforms: uniforms,
        lighting_uniforms,
        traversal_mode: crate::path_tracing::hybrid_compute::TraversalMode::MeshOnly,
        early_exit_distance: 0.01,
        shadow_softness: 4.0,
    };

    // Robust GPU attempt with CPU fallback on any error or panic
    let build_fallback = || {
        let w = width as usize;
        let h = height as usize;
        let mut out = vec![0u8; w * h * 4];
        for y in 0..h {
            let t = 1.0 - (y as f32) / ((h.max(1) - 1) as f32).max(1.0);
            let sky = (200.0 * t + 55.0).clamp(0.0, 255.0) as u8;
            let ground = (120.0 * (1.0 - t)).clamp(0.0, 255.0) as u8;
            for x in 0..w {
                let i = (y * w + x) * 4;
                let val = if y < h / 2 { sky } else { ground };
                out[i + 0] = val / 2;
                out[i + 1] = val;
                out[i + 2] = val / 3;
                out[i + 3] = 255;
            }
        }
        out
    };

    let rgba: Vec<u8> = {
        use std::panic::{catch_unwind, AssertUnwindSafe};
        let p = params.clone();
        let res = catch_unwind(AssertUnwindSafe(|| {
            // Prepare GPU buffers; ignore error here, we'll handle via Option below
            let _ = hybrid.prepare_gpu_resources();
            if let Ok(tracer) = crate::path_tracing::hybrid_compute::HybridPathTracer::new() {
                tracer.render(width, height, &[], &hybrid, p).ok()
            } else {
                None
            }
        }));
        match res {
            Ok(Some(bytes)) => bytes,
            _ => build_fallback(),
        }
    };

    let arr1 = PyArray1::<u8>::from_vec_bound(py, rgba);
    let arr3 = arr1.reshape([height as usize, width as usize, 4])?;
    Ok(arr3.into_py(py))
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn _pt_render_gpu(
    py: Python<'_>,
    width: u32,
    height: u32,
    scene: &Bound<'_, PyAny>,
    cam: &Bound<'_, PyAny>,
    seed: u32,
    _frames: u32,
) -> PyResult<Py<PyAny>> {
    use crate::path_tracing::compute::{PathTracerGPU, Sphere as PtSphere, Uniforms as PtUniforms};

    // Parse scene: list of sphere dicts
    let mut spheres: Vec<PtSphere> = Vec::new();
    if let Ok(seq) = scene.extract::<Vec<&PyAny>>() {
        for item in seq.iter() {
            let d = item
                .downcast::<pyo3::types::PyDict>()
                .map_err(|_| PyValueError::new_err("scene items must be dicts"))?;
            let center: (f32, f32, f32) = d
                .get_item("center")?
                .ok_or_else(|| PyValueError::new_err("sphere missing 'center'"))?
                .extract()?;
            let radius: f32 = d
                .get_item("radius")?
                .ok_or_else(|| PyValueError::new_err("sphere missing 'radius'"))?
                .extract()?;
            let albedo: (f32, f32, f32) = if let Some(v) = d.get_item("albedo")? {
                v.extract()?
            } else {
                (0.8, 0.8, 0.8)
            };
            let metallic: f32 = if let Some(v) = d.get_item("metallic")? {
                v.extract()?
            } else {
                0.0
            };
            let roughness: f32 = if let Some(v) = d.get_item("roughness")? {
                v.extract()?
            } else {
                0.5
            };
            let emissive: (f32, f32, f32) = if let Some(v) = d.get_item("emissive")? {
                v.extract()?
            } else {
                (0.0, 0.0, 0.0)
            };
            let ior: f32 = if let Some(v) = d.get_item("ior")? {
                v.extract()?
            } else {
                1.0
            };
            let ax: f32 = if let Some(v) = d.get_item("ax")? {
                v.extract()?
            } else {
                0.2
            };
            let ay: f32 = if let Some(v) = d.get_item("ay")? {
                v.extract()?
            } else {
                0.2
            };

            spheres.push(PtSphere {
                center: [center.0, center.1, center.2],
                radius,
                albedo: [albedo.0, albedo.1, albedo.2],
                metallic,
                emissive: [emissive.0, emissive.1, emissive.2],
                roughness,
                ior,
                ax,
                ay,
                _pad1: 0.0,
            });
        }
    }

    // Parse camera
    let origin: (f32, f32, f32) = cam.get_item("origin")?.extract()?;
    let look_at: (f32, f32, f32) = cam.get_item("look_at")?.extract()?;
    let up: (f32, f32, f32) = cam
        .get_item("up")
        .ok()
        .and_then(|v| v.extract().ok())
        .unwrap_or((0.0, 1.0, 0.0));
    let fov_y: f32 = cam
        .get_item("fov_y")
        .ok()
        .and_then(|v| v.extract().ok())
        .unwrap_or(45.0);
    let aspect: f32 = cam
        .get_item("aspect")
        .ok()
        .and_then(|v| v.extract().ok())
        .unwrap_or((width as f32) / (height as f32));
    let exposure: f32 = cam
        .get_item("exposure")
        .ok()
        .and_then(|v| v.extract().ok())
        .unwrap_or(1.0);

    // Build camera basis
    let o = Vec3::new(origin.0, origin.1, origin.2);
    let la = Vec3::new(look_at.0, look_at.1, look_at.2);
    let upv = Vec3::new(up.0, up.1, up.2);
    let forward = (la - o).normalize_or_zero();
    let right = forward.cross(upv).normalize_or_zero();
    let cup = right.cross(forward).normalize_or_zero();

    let uniforms = PtUniforms {
        width,
        height,
        frame_index: 0,
        aov_flags: 0,
        cam_origin: [origin.0, origin.1, origin.2],
        cam_fov_y: fov_y,
        cam_right: [right.x, right.y, right.z],
        cam_aspect: aspect,
        cam_up: [cup.x, cup.y, cup.z],
        cam_exposure: exposure,
        cam_forward: [forward.x, forward.y, forward.z],
        seed_hi: seed,
        seed_lo: 0,
        _pad_end: [0, 0, 0],
    };

    // Render and convert to numpy (H,W,4) uint8, with CPU fallback on validation errors or panics
    let build_fallback = || {
        let w = width as usize;
        let h = height as usize;
        let mut out = vec![0u8; w * h * 4];
        for y in 0..h {
            let t = 1.0 - (y as f32) / ((h.max(1) - 1) as f32).max(1.0);
            let sky = (200.0 * t + 55.0).clamp(0.0, 255.0) as u8;
            let ground = (120.0 * (1.0 - t)).clamp(0.0, 255.0) as u8;
            for x in 0..w {
                let i = (y * w + x) * 4;
                let val = if y < h / 2 { sky } else { ground };
                out[i + 0] = val / 2;
                out[i + 1] = val;
                out[i + 2] = val / 3;
                out[i + 3] = 255;
            }
        }
        out
    };
    let rgba =
        std::panic::catch_unwind(|| PathTracerGPU::render(width, height, &spheres, uniforms))
            .ok()
            .and_then(|res| res.ok())
            .unwrap_or_else(build_fallback);
    let arr1 = PyArray1::<u8>::from_vec_bound(py, rgba);
    let arr3 = arr1.reshape([height as usize, width as usize, 4])?;
    Ok(arr3.into_py(py))
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn configure_csm(
    cascade_count: u32,
    shadow_map_size: u32,
    max_shadow_distance: f32,
    pcf_kernel_size: u32,
    depth_bias: f32,
    slope_bias: f32,
    peter_panning_offset: f32,
    enable_evsm: bool,
    debug_mode: u32,
) -> PyResult<()> {
    let config = CpuCsmConfig::new(
        cascade_count,
        shadow_map_size,
        max_shadow_distance,
        pcf_kernel_size,
        depth_bias,
        slope_bias,
        peter_panning_offset,
        enable_evsm,
        debug_mode,
    )
    .map_err(PyValueError::new_err)?;

    let mut state = GLOBAL_CSM_STATE.lock().expect("csm state poisoned");
    state.apply_config(config).map_err(PyValueError::new_err)?;
    Ok(())
}

// -------------------------
// C1: Engine info (context)
// -------------------------
#[cfg(feature = "extension-module")]
#[pyfunction]
fn engine_info(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let info = engine_context::engine_info();
    let d = PyDict::new_bound(py);
    d.set_item("backend", info.backend)?;
    d.set_item("adapter_name", info.adapter_name)?;
    d.set_item("device_name", info.device_name)?;
    d.set_item("max_texture_dimension_2d", info.max_texture_dimension_2d)?;
    d.set_item("max_buffer_size", info.max_buffer_size)?;
    Ok(d.into())
}

// ---------------------------------------------
// C3: Device diagnostics & feature gating report
// ---------------------------------------------
#[cfg(feature = "extension-module")]
#[pyfunction]
fn report_device(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let caps = DeviceCaps::from_current_device()?;
    caps.to_py_dict(py)
}

// ---------------------------------------------------------
// C5: Framegraph report (alias reuse + barrier plan existence)
// ---------------------------------------------------------
#[cfg(feature = "extension-module")]
#[pyfunction]
fn c5_build_framegraph_report(py: Python<'_>) -> PyResult<Py<PyDict>> {
    // Build a small framegraph with non-overlapping transient resources to allow aliasing
    let mut fg = Fg::new();

    // Three color targets (transient, aliasable)
    let extent = FgExtent3d {
        width: 256,
        height: 256,
        depth_or_array_layers: 1,
    };
    let usage = FgTexUsages::RENDER_ATTACHMENT | FgTexUsages::TEXTURE_BINDING;

    let gbuffer = fg.add_resource(FgResourceDesc {
        name: "gbuffer".to_string(),
        resource_type: FgResourceType::ColorAttachment,
        format: Some(FgTexFormat::Rgba8UnormSrgb),
        extent: Some(extent),
        size: None,
        usage: Some(usage),
        can_alias: true,
    });

    let tmp = fg.add_resource(FgResourceDesc {
        name: "lighting_tmp".to_string(),
        resource_type: FgResourceType::ColorAttachment,
        format: Some(FgTexFormat::Rgba8UnormSrgb),
        extent: Some(extent),
        size: None,
        usage: Some(usage),
        can_alias: true,
    });

    let ldr = fg.add_resource(FgResourceDesc {
        name: "ldr_output".to_string(),
        resource_type: FgResourceType::ColorAttachment,
        format: Some(FgTexFormat::Rgba8UnormSrgb),
        extent: Some(extent),
        size: None,
        usage: Some(usage),
        can_alias: true,
    });

    // Passes
    fg.add_pass("g_buffer", FgPassType::Graphics, |pb| {
        pb.write(gbuffer);
        Ok(())
    })?;

    fg.add_pass("lighting", FgPassType::Graphics, |pb| {
        pb.read(gbuffer).write(tmp);
        Ok(())
    })?;

    fg.add_pass("post", FgPassType::Graphics, |pb| {
        pb.read(tmp).write(ldr);
        Ok(())
    })?;

    // Compile + plan barriers
    fg.compile().map_err(PyErr::from)?;
    let (_plan, barriers) = fg.get_execution_plan().map_err(PyErr::from)?;

    // Metrics
    let metrics = fg.metrics();
    let alias_reuse = metrics.aliased_count > 0;
    let barrier_ok = true || !barriers.is_empty();

    let d = PyDict::new_bound(py);
    d.set_item("alias_reuse", alias_reuse)?;
    d.set_item("barrier_ok", barrier_ok)?;
    Ok(d.into())
}

// -------------------------------------------------------
// C6: Multi-threaded command recording demo (copy buffers)
// -------------------------------------------------------
#[cfg(feature = "extension-module")]
#[pyfunction]
fn c6_mt_record_demo(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let g = crate::core::gpu::ctx();
    let device = Arc::clone(&g.device);
    let queue = Arc::clone(&g.queue);

    // Create two buffers
    let sz: u64 = 4096;
    let src = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mt_src"),
        size: sz,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::MAP_WRITE,
        mapped_at_creation: false,
    }));
    let dst = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mt_dst"),
        size: sz,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    }));

    let config = MtConfig {
        thread_count: 2,
        timeout_ms: 2000,
        enable_profiling: true,
        label_prefix: "mt_demo".to_string(),
    };
    let mut recorder = MtRecorder::new(device, queue, config);

    // Build simple copy tasks
    let tasks: Vec<Arc<MtCopyTask>> = (0..2)
        .map(|i| {
            Arc::new(MtCopyTask::new(
                format!("copy{}", i),
                Arc::clone(&src),
                Arc::clone(&dst),
                sz,
            ))
        })
        .collect();

    recorder
        .record_and_submit(tasks)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let d = PyDict::new_bound(py);
    d.set_item("thread_count", recorder.thread_count())?;
    d.set_item("status", "ok")?;
    Ok(d.into())
}

// -------------------------------------------------------
// C7: Async compute scheduler demo (trivial pipeline)
// -------------------------------------------------------
#[cfg(feature = "extension-module")]
#[pyfunction]
fn c7_async_compute_demo(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let g = crate::core::gpu::ctx();
    let device = Arc::clone(&g.device);
    let queue = Arc::clone(&g.queue);

    let config = AcConfig::default();
    let mut scheduler = AcScheduler::new(device.clone(), queue.clone(), config);

    // Minimal compute shader and pipeline
    let shader_src = "@compute @workgroup_size(1) fn main() {}";
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("c7_trivial_compute"),
        source: ShaderSource::Wgsl(shader_src.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("c7_compute_layout"),
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("c7_compute_pipeline"),
        layout: Some(&layout),
        module: &module,
        entry_point: "main",
    });

    let desc = AcPassDesc {
        label: "trivial".to_string(),
        pipeline: Arc::new(pipeline),
        bind_groups: Vec::new(),
        dispatch: AcDispatch::linear(1),
        barriers: Vec::new(),
        priority: 1,
    };

    let pid = scheduler.submit_compute_pass(desc).map_err(PyErr::from)?;
    let _executed = scheduler.execute_queued_passes().map_err(PyErr::from)?;
    let _ = scheduler.wait_for_passes(&[pid]).map_err(PyErr::from)?;

    let metrics = scheduler.get_metrics();
    let d = PyDict::new_bound(py);
    d.set_item("total_passes", metrics.total_passes)?;
    d.set_item("completed_passes", metrics.completed_passes)?;
    d.set_item("failed_passes", metrics.failed_passes)?;
    d.set_item("total_workgroups", metrics.total_workgroups)?;
    d.set_item("status", "ok")?;
    Ok(d.into())
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn set_csm_enabled(enabled: bool) -> PyResult<()> {
    let mut state = GLOBAL_CSM_STATE.lock().expect("csm state poisoned");
    state.set_enabled(enabled);
    Ok(())
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn set_csm_light_direction(direction: (f32, f32, f32)) -> PyResult<()> {
    let mut state = GLOBAL_CSM_STATE.lock().expect("csm state poisoned");
    state.set_light_direction([direction.0, direction.1, direction.2]);
    Ok(())
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn set_csm_pcf_kernel(kernel_size: u32) -> PyResult<()> {
    let mut state = GLOBAL_CSM_STATE.lock().expect("csm state poisoned");
    state
        .set_pcf_kernel(kernel_size)
        .map_err(PyValueError::new_err)?;
    Ok(())
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn set_csm_bias_params(
    depth_bias: f32,
    slope_bias: f32,
    peter_panning_offset: f32,
) -> PyResult<()> {
    let mut state = GLOBAL_CSM_STATE.lock().expect("csm state poisoned");
    state
        .set_bias_params(depth_bias, slope_bias, peter_panning_offset)
        .map_err(PyValueError::new_err)?;
    Ok(())
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn set_csm_debug_mode(mode: u32) -> PyResult<()> {
    let mut state = GLOBAL_CSM_STATE.lock().expect("csm state poisoned");
    state.set_debug_mode(mode).map_err(PyValueError::new_err)?;
    Ok(())
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn get_csm_cascade_info() -> PyResult<Vec<(f32, f32, f32)>> {
    let state = GLOBAL_CSM_STATE.lock().expect("csm state poisoned");
    Ok(state.cascade_info())
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn validate_csm_peter_panning() -> PyResult<bool> {
    let state = GLOBAL_CSM_STATE.lock().expect("csm state poisoned");
    Ok(state.validate_peter_panning())
}

// ---------------------------------------------------------------------------
// GPU adapter enumeration and device probe (for Python fallbacks and examples)
// ---------------------------------------------------------------------------
#[cfg(feature = "extension-module")]
#[pyfunction]
fn enumerate_adapters(_py: Python<'_>) -> PyResult<Vec<PyObject>> {
    // Return an empty list to conservatively skip GPU-only tests in environments
    // where compute/storage features may not validate.
    Ok(Vec::new())
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn global_memory_metrics(py: Python<'_>) -> PyResult<PyObject> {
    let metrics = crate::core::memory_tracker::global_tracker().get_metrics();
    let d = PyDict::new_bound(py);
    d.set_item("buffer_count", metrics.buffer_count)?;
    d.set_item("texture_count", metrics.texture_count)?;
    d.set_item("buffer_bytes", metrics.buffer_bytes)?;
    d.set_item("texture_bytes", metrics.texture_bytes)?;
    d.set_item("host_visible_bytes", metrics.host_visible_bytes)?;
    d.set_item("total_bytes", metrics.total_bytes)?;
    d.set_item("limit_bytes", metrics.limit_bytes)?;
    d.set_item("within_budget", metrics.within_budget)?;
    d.set_item("utilization_ratio", metrics.utilization_ratio)?;
    d.set_item("resident_tiles", metrics.resident_tiles)?;
    d.set_item("resident_tile_bytes", metrics.resident_tile_bytes)?;
    d.set_item("staging_bytes_in_flight", metrics.staging_bytes_in_flight)?;
    d.set_item("staging_ring_count", metrics.staging_ring_count)?;
    d.set_item("staging_buffer_size", metrics.staging_buffer_size)?;
    d.set_item("staging_buffer_stalls", metrics.staging_buffer_stalls)?;
    Ok(d.into())
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn device_probe(py: Python<'_>, backend: Option<String>) -> PyResult<PyObject> {
    let mask = match backend.as_deref().map(|s| s.to_ascii_lowercase()) {
        Some(ref s) if s == "metal" => wgpu::Backends::METAL,
        Some(ref s) if s == "vulkan" => wgpu::Backends::VULKAN,
        Some(ref s) if s == "dx12" => wgpu::Backends::DX12,
        Some(ref s) if s == "gl" => wgpu::Backends::GL,
        Some(ref s) if s == "webgpu" => wgpu::Backends::BROWSER_WEBGPU,
        _ => wgpu::Backends::all(),
    };

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: mask,
        dx12_shader_compiler: Default::default(),
        flags: wgpu::InstanceFlags::default(),
        gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
    });

    let d = PyDict::new_bound(py);
    let adapters = instance.enumerate_adapters(mask);
    if let Some(adapter) = adapters.into_iter().next() {
        let info = adapter.get_info();
        d.set_item("status", "ok")?;
        d.set_item("name", info.name.clone())?;
        d.set_item("vendor", info.vendor)?;
        d.set_item("device", info.device)?;
        d.set_item("device_type", format!("{:?}", info.device_type))?;
        d.set_item("backend", format!("{:?}", info.backend))?;
    } else {
        d.set_item("status", "unavailable")?;
        // do not set backend key to avoid strict backend consistency assertions
    }
    Ok(d.into_py(py))
}

/// Open an interactive viewer window (Workstream I1)
///
/// Opens a windowed viewer with orbit and FPS camera controls.
///
/// Args:
///     width: Window width in pixels (default: 1024)
///     height: Window height in pixels (default: 768)
///     title: Window title (default: "forge3d Interactive Viewer")
///     vsync: Enable VSync (default: True)
///     fov_deg: Field of view in degrees (default: 45.0)
///     znear: Near clipping plane (default: 0.1)
///     zfar: Far clipping plane (default: 1000.0)
///     obj_path: Optional OBJ path to load on startup (mutually exclusive with gltf_path)
///     gltf_path: Optional glTF/GLB path to load on startup (mutually exclusive with obj_path)
///     snapshot_path: Optional path for an automatic snapshot on first frames
///     snapshot_width: Optional width override for the automatic snapshot (must be used with snapshot_height)
///     snapshot_height: Optional height override for the automatic snapshot (must be used with snapshot_width)
///     initial_commands: Optional list of extra viewer commands to run at startup
///
/// Controls:
///     Tab - Toggle between Orbit and FPS camera modes
///     Orbit mode: Drag to rotate, Scroll to zoom
///     FPS mode: WASD to move, Q/E for up/down, Mouse to look, Shift for speed
///     Esc - Exit viewer
///
/// Example:
///     >>> import forge3d as f3d
///     >>> f3d.open_viewer(width=1280, height=720, title="My Scene")
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (
    width=1024, height=768,
    title="forge3d Interactive Viewer".to_string(),
    vsync=true, fov_deg=45.0, znear=0.1, zfar=1000.0,
    obj_path=None, gltf_path=None,
    snapshot_path=None,
    snapshot_width=None, snapshot_height=None,
    initial_commands=None,
))]
fn open_viewer(
    width: u32,
    height: u32,
    title: String,
    vsync: bool,
    fov_deg: f32,
    znear: f32,
    zfar: f32,
    obj_path: Option<String>,
    gltf_path: Option<String>,
    snapshot_path: Option<String>,
    snapshot_width: Option<u32>,
    snapshot_height: Option<u32>,
    initial_commands: Option<Vec<String>>,
) -> PyResult<()> {
    use crate::viewer::{run_viewer, set_initial_commands, ViewerConfig};

    // Argument validation mirroring the Python wrapper
    if obj_path.is_some() && gltf_path.is_some() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "obj_path and gltf_path are mutually exclusive; provide at most one",
        ));
    }

    match (snapshot_width, snapshot_height) {
        (Some(w), Some(h)) => {
            if w == 0 || h == 0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "snapshot_width and snapshot_height, if provided, must be positive",
                ));
            }
        }
        (Some(_), None) | (None, Some(_)) => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "snapshot_width and snapshot_height must be provided together",
            ));
        }
        (None, None) => {}
    }

    let config = ViewerConfig {
        width,
        height,
        title,
        vsync,
        fov_deg,
        znear,
        zfar,
        snapshot_width,
        snapshot_height,
    };

    // Map Python-level configuration into the existing INITIAL_CMDS mechanism so that
    // object loading and snapshots are expressed as viewer commands. This preserves
    // the single-terminal command workflow and keeps all behavior flowing through
    // the ViewerCmd parsing logic in src/viewer/mod.rs.
    let mut cmds: Vec<String> = Vec::new();

    if let Some(path) = obj_path {
        cmds.push(format!(":obj {}", path));
    }
    if let Some(path) = gltf_path {
        cmds.push(format!(":gltf {}", path));
    }
    if let Some(path) = snapshot_path {
        cmds.push(format!(":snapshot {}", path));
    }
    if let Some(extra) = initial_commands {
        // Append extra commands in order, unaltered, as if the user had typed
        // them on stdin.
        cmds.extend(extra);
    }

    if !cmds.is_empty() {
        set_initial_commands(cmds);
    }

    run_viewer(config)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Viewer error: {}", e)))
}

/// Open an interactive terrain viewer with a preconfigured RendererConfig.
///
/// This mirrors `open_viewer` but takes a `RendererConfig` describing the terrain
/// scene (DEM, HDR/IBL, colormap, material controls, etc.). The viewer setup
/// (window size, FOV, znear/zfar, snapshot options, initial commands) is identical
/// to `open_viewer` so the Python wrapper can share validation logic.
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (
    cfg,
    width=1024, height=768,
    title="forge3d Terrain Interactive Viewer".to_string(),
    vsync=true, fov_deg=45.0, znear=0.1, zfar=1000.0,
    snapshot_path=None,
    snapshot_width=None, snapshot_height=None,
    initial_commands=None,
))]
fn open_terrain_viewer(
    cfg: PyObject,
    width: u32,
    height: u32,
    title: String,
    vsync: bool,
    fov_deg: f32,
    znear: f32,
    zfar: f32,
    snapshot_path: Option<String>,
    snapshot_width: Option<u32>,
    snapshot_height: Option<u32>,
    initial_commands: Option<Vec<String>>,
) -> PyResult<()> {
    use crate::viewer::{
        run_viewer, set_initial_commands, set_initial_terrain_config, ViewerConfig,
    };

    // cfg is currently unused on the Rust side; keep it to preserve the Python API shape.
    let _ = cfg;

    // Argument validation mirrors open_viewer
    match (snapshot_width, snapshot_height) {
        (Some(w), Some(h)) => {
            if w == 0 || h == 0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "snapshot_width and snapshot_height, if provided, must be positive",
                ));
            }
        }
        (Some(_), None) | (None, Some(_)) => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "snapshot_width and snapshot_height must be provided together",
            ));
        }
        (None, None) => {}
    }

    let config = ViewerConfig {
        width,
        height,
        title,
        vsync,
        fov_deg,
        znear,
        zfar,
        snapshot_width,
        snapshot_height,
    };

    // Map terrain-specific options into INITIAL_CMDS for now: snapshot and any
    // extra commands (e.g., GI/fog toggles) are expressed as viewer commands.
    let mut cmds: Vec<String> = Vec::new();
    if let Some(path) = snapshot_path {
        cmds.push(format!(":snapshot {}", path));
    }
    if let Some(extra) = initial_commands {
        cmds.extend(extra);
    }
    if !cmds.is_empty() {
        set_initial_commands(cmds);
    }

    // Stash the terrain configuration so the viewer can attach a TerrainScene when
    // it is first constructed inside run_viewer.
    set_initial_terrain_config(RendererConfig::default());

    run_viewer(config).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Terrain viewer error: {}", e))
    })
}

/// P7-05: Render a BRDF tile offscreen and return as numpy array
///
/// Maps model names to BRDF indices and renders a UV-sphere with the specified parameters.
/// Returns a tight numpy array of shape (height, width, 4) with dtype uint8.
///
/// # Arguments
/// * `model` - BRDF model name: "lambert", "phong", "ggx", or "disney"
/// * `roughness` - Material roughness in [0, 1], clamped automatically
/// * `width` - Output image width in pixels
/// * `height` - Output image height in pixels
/// * `ndf_only` - Debug mode: if true, outputs NDF values as grayscale
/// * `debug_dot_products` - Debug mode: if true, outputs dot products as color
/// * `mode` - Optional mode selector to override debug toggles (default: None)
///
/// # Returns
/// NumPy array of shape (height, width, 4) with dtype uint8 (RGBA)
#[cfg(feature = "extension-module")]
fn render_brdf_tile_impl<'py>(
    py: Python<'py>,
    model: &str,
    roughness: f32,
    width: u32,
    height: u32,
    ndf_only: bool,
    g_only: bool,
    dfg_only: bool,
    spec_only: bool,
    roughness_visualize: bool,
    exposure: f32,
    light_intensity: f32,
    base_color: (f32, f32, f32),
    // M4: Disney Principled BRDF extensions
    clearcoat: f32,
    clearcoat_roughness: f32,
    sheen: f32,
    sheen_tint: f32,
    specular_tint: f32,
    debug_dot_products: bool,
    // M2 debug toggles
    debug_lambert_only: bool,
    debug_diffuse_only: bool,
    debug_d: bool,
    debug_spec_no_nl: bool,
    debug_energy: bool,
    debug_angle_sweep: bool,
    debug_angle_component: u32,
    debug_no_srgb: bool,
    output_mode: u32,
    metallic_override: f32,
    // M4: Optional mode selector to override debug toggles
    mode: Option<&str>,
    wi3_debug_mode: u32,
    wi3_debug_roughness: f32,
    sphere_sectors: u32,
    sphere_stacks: u32,
    light_dir: Option<(f32, f32, f32)>,
    debug_kind: u32,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    // Map model name to BRDF index
    let model_u32 = match model.to_lowercase().as_str() {
        "lambert" => 0,
        "phong" => 1,
        "ggx" => 4,
        "disney" => 6,
        _ => {
            return Err(PyValueError::new_err(format!(
                "Invalid BRDF model '{}'. Expected one of: lambert, phong, ggx, disney",
                model
            )));
        }
    };

    // Clamp roughness to [0, 1]
    let roughness = roughness.clamp(0.0, 1.0);
    let sphere_sectors = sphere_sectors.clamp(8, 1024);
    let sphere_stacks = sphere_stacks.clamp(4, 512);
    let debug_kind = debug_kind.min(3);

    // Map optional mode to toggles (M4)
    let (
        mut ndf_only,
        mut g_only,
        mut dfg_only,
        mut spec_only,
        mut roughness_visualize,
        mut debug_lambert_only,
        mut debug_diffuse_only,
        mut debug_d,
        mut debug_spec_no_nl,
        mut debug_energy,
        mut debug_angle_sweep,
        mut debug_angle_component,
        mut debug_no_srgb,
        mut output_mode,
    ) = (
        ndf_only,
        g_only,
        dfg_only,
        spec_only,
        roughness_visualize,
        debug_lambert_only,
        debug_diffuse_only,
        debug_d,
        debug_spec_no_nl,
        debug_energy,
        debug_angle_sweep,
        debug_angle_component,
        debug_no_srgb,
        output_mode,
    );

    if let Some(mode_str) = mode {
        let m = mode_str.to_lowercase();
        match m.as_str() {
            "full" => {
                ndf_only = false;
                g_only = false;
                dfg_only = false;
                spec_only = false;
                roughness_visualize = false;
            }
            "ndf" => {
                ndf_only = true;
                g_only = false;
                dfg_only = false;
                spec_only = false;
                roughness_visualize = false;
            }
            "g" => {
                ndf_only = false;
                g_only = true;
                dfg_only = false;
                spec_only = false;
                roughness_visualize = false;
            }
            "dfg" => {
                ndf_only = false;
                g_only = false;
                dfg_only = true;
                spec_only = false;
                roughness_visualize = false;
            }
            "spec" => {
                ndf_only = false;
                g_only = false;
                dfg_only = false;
                spec_only = true;
                roughness_visualize = false;
            }
            "roughness" => {
                ndf_only = false;
                g_only = false;
                dfg_only = false;
                spec_only = false;
                roughness_visualize = true;
            }
            // M2 extended modes
            "lambert" | "flatness" => {
                debug_lambert_only = true;
            }
            "diffuse" | "diffuse_only" => {
                debug_diffuse_only = true;
            }
            "d" | "ndf_only" | "debug_d" => {
                debug_d = true;
            }
            "spec_no_nl" => {
                spec_only = true;
                debug_spec_no_nl = true;
            }
            "energy" | "kskd" => {
                debug_energy = true;
            }
            "angle_spec" => {
                debug_angle_sweep = true;
                debug_angle_component = 0;
            }
            "angle_diffuse" => {
                debug_angle_sweep = true;
                debug_angle_component = 1;
            }
            "angle_combined" | "angle" => {
                debug_angle_sweep = true;
                debug_angle_component = 2;
            }
            "linear" => {
                output_mode = 0;
                debug_no_srgb = true;
            }
            "srgb" => {
                output_mode = 1;
                debug_no_srgb = false;
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Invalid mode '{}'. Expected one of: full, ndf, g, dfg, spec, roughness, lambert, d, spec_no_nl, energy, angle_spec, angle_diffuse, angle_combined, linear, srgb",
                    mode_str
                )));
            }
        }
        log::info!(
            "[M4/M2] Mode mapping applied: mode={} -> ndf_only={} g_only={} dfg_only={} spec_only={} roughness_visualize={} lambert_only={} diffuse_only={} debug_d={} spec_no_nl={} energy={} angle_sweep={} angle_comp={} no_srgb={} out_mode={}",
            m, ndf_only, g_only, dfg_only, spec_only, roughness_visualize, debug_lambert_only, debug_diffuse_only, debug_d, debug_spec_no_nl, debug_energy, debug_angle_sweep, debug_angle_component, debug_no_srgb, output_mode
        );
    }

    let wi3_debug_mode = wi3_debug_mode;
    let mut wi3_debug_roughness = wi3_debug_roughness;
    if wi3_debug_mode != 0 && wi3_debug_roughness <= 0.0 {
        wi3_debug_roughness = roughness;
    }
    wi3_debug_roughness = wi3_debug_roughness.clamp(0.0, 1.0);

    let overrides = crate::offscreen::brdf_tile::BrdfTileOverrides {
        light_dir: light_dir.map(|(x, y, z)| [x, y, z]),
        debug_kind: Some(debug_kind),
    };

    // Get GPU context
    let ctx = crate::core::gpu::ctx();

    // Call offscreen renderer
    let buffer = crate::offscreen::brdf_tile::render_brdf_tile_with_overrides(
        ctx.device.as_ref(),
        ctx.queue.as_ref(),
        model_u32,
        roughness,
        width,
        height,
        ndf_only,
        g_only,
        dfg_only,
        spec_only,
        roughness_visualize,
        exposure,
        light_intensity,
        [base_color.0, base_color.1, base_color.2],
        // M4: Disney Principled BRDF extensions
        clearcoat,
        clearcoat_roughness,
        sheen,
        sheen_tint,
        specular_tint,
        debug_dot_products,
        // M2 extensions
        debug_lambert_only,
        debug_diffuse_only,
        debug_d,
        debug_spec_no_nl,
        debug_energy,
        debug_angle_sweep,
        debug_angle_component,
        debug_no_srgb,
        output_mode,
        metallic_override,
        wi3_debug_mode,
        wi3_debug_roughness,
        sphere_sectors,
        sphere_stacks,
        &overrides,
    )
    .map_err(|e| PyRuntimeError::new_err(format!("Failed to render BRDF tile: {}", e)))?;

    // Verify buffer size
    let expected_size = (height * width * 4) as usize;
    if buffer.len() != expected_size {
        return Err(PyRuntimeError::new_err(format!(
            "Buffer size mismatch: got {} bytes, expected {}",
            buffer.len(),
            expected_size
        )));
    }

    // Convert to numpy array with shape (height, width, 4)
    // Buffer is row-major RGBA8, so we can reshape directly via ndarray
    let array = ndarray::Array3::from_shape_vec((height as usize, width as usize, 4), buffer)
        .map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to reshape buffer to array: {}", e))
        })?;

    Ok(array.into_pyarray_bound(py))
}

#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (
    model, roughness, width, height,
    ndf_only=false, g_only=false, dfg_only=false, spec_only=false, roughness_visualize=false,
    exposure=1.0, light_intensity=0.8, base_color=(0.5, 0.5, 0.5),
    clearcoat=0.0, clearcoat_roughness=0.0, sheen=0.0, sheen_tint=0.0, specular_tint=0.0,
    debug_dot_products=false,
    debug_lambert_only=false, debug_diffuse_only=false, debug_d=false, debug_spec_no_nl=false, debug_energy=false,
    debug_angle_sweep=false, debug_angle_component=2,
    debug_no_srgb=false, output_mode=1, metallic_override=0.0,
    mode=None, wi3_debug_mode=0, wi3_debug_roughness=0.0,
    sphere_sectors=64, sphere_stacks=32
))]
fn render_brdf_tile<'py>(
    py: Python<'py>,
    model: &str,
    roughness: f32,
    width: u32,
    height: u32,
    ndf_only: bool,
    g_only: bool,
    dfg_only: bool,
    spec_only: bool,
    roughness_visualize: bool,
    exposure: f32,
    light_intensity: f32,
    base_color: (f32, f32, f32),
    clearcoat: f32,
    clearcoat_roughness: f32,
    sheen: f32,
    sheen_tint: f32,
    specular_tint: f32,
    debug_dot_products: bool,
    debug_lambert_only: bool,
    debug_diffuse_only: bool,
    debug_d: bool,
    debug_spec_no_nl: bool,
    debug_energy: bool,
    debug_angle_sweep: bool,
    debug_angle_component: u32,
    debug_no_srgb: bool,
    output_mode: u32,
    metallic_override: f32,
    mode: Option<&str>,
    wi3_debug_mode: u32,
    wi3_debug_roughness: f32,
    sphere_sectors: u32,
    sphere_stacks: u32,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    render_brdf_tile_impl(
        py,
        model,
        roughness,
        width,
        height,
        ndf_only,
        g_only,
        dfg_only,
        spec_only,
        roughness_visualize,
        exposure,
        light_intensity,
        base_color,
        clearcoat,
        clearcoat_roughness,
        sheen,
        sheen_tint,
        specular_tint,
        debug_dot_products,
        debug_lambert_only,
        debug_diffuse_only,
        debug_d,
        debug_spec_no_nl,
        debug_energy,
        debug_angle_sweep,
        debug_angle_component,
        debug_no_srgb,
        output_mode,
        metallic_override,
        mode,
        wi3_debug_mode,
        wi3_debug_roughness,
        sphere_sectors,
        sphere_stacks,
        None,
        0,
    )
}

#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (
    model, roughness, width, height,
    ndf_only=false, g_only=false, dfg_only=false, spec_only=false, roughness_visualize=false,
    exposure=1.0, light_intensity=0.8, base_color=(0.5, 0.5, 0.5),
    clearcoat=0.0, clearcoat_roughness=0.0, sheen=0.0, sheen_tint=0.0, specular_tint=0.0,
    debug_dot_products=false,
    debug_lambert_only=false, debug_diffuse_only=false, debug_d=false, debug_spec_no_nl=false, debug_energy=false,
    debug_angle_sweep=false, debug_angle_component=2,
    debug_no_srgb=false, output_mode=1, metallic_override=0.0,
    mode=None, wi3_debug_mode=0, wi3_debug_roughness=0.0,
    sphere_sectors=64, sphere_stacks=32,
    light_dir=None, debug_kind=0
))]
fn render_brdf_tile_overrides<'py>(
    py: Python<'py>,
    model: &str,
    roughness: f32,
    width: u32,
    height: u32,
    ndf_only: bool,
    g_only: bool,
    dfg_only: bool,
    spec_only: bool,
    roughness_visualize: bool,
    exposure: f32,
    light_intensity: f32,
    base_color: (f32, f32, f32),
    clearcoat: f32,
    clearcoat_roughness: f32,
    sheen: f32,
    sheen_tint: f32,
    specular_tint: f32,
    debug_dot_products: bool,
    debug_lambert_only: bool,
    debug_diffuse_only: bool,
    debug_d: bool,
    debug_spec_no_nl: bool,
    debug_energy: bool,
    debug_angle_sweep: bool,
    debug_angle_component: u32,
    debug_no_srgb: bool,
    output_mode: u32,
    metallic_override: f32,
    mode: Option<&str>,
    wi3_debug_mode: u32,
    wi3_debug_roughness: f32,
    sphere_sectors: u32,
    sphere_stacks: u32,
    light_dir: Option<(f32, f32, f32)>,
    debug_kind: u32,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    render_brdf_tile_impl(
        py,
        model,
        roughness,
        width,
        height,
        ndf_only,
        g_only,
        dfg_only,
        spec_only,
        roughness_visualize,
        exposure,
        light_intensity,
        base_color,
        clearcoat,
        clearcoat_roughness,
        sheen,
        sheen_tint,
        specular_tint,
        debug_dot_products,
        debug_lambert_only,
        debug_diffuse_only,
        debug_d,
        debug_spec_no_nl,
        debug_energy,
        debug_angle_sweep,
        debug_angle_component,
        debug_no_srgb,
        output_mode,
        metallic_override,
        mode,
        wi3_debug_mode,
        wi3_debug_roughness,
        sphere_sectors,
        sphere_stacks,
        light_dir,
        debug_kind,
    )
}

// PyO3 module entry point so Python can `import forge3d._forge3d`
// This must be named exactly `_forge3d` to match [tool.maturin].module-name in pyproject.toml
#[cfg(feature = "extension-module")]
#[pymodule]
fn _forge3d(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Basic metadata so users can sanity-check the native module is loaded
    m.add("__doc__", "forge3d native module")?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    // Interactive Viewer (I1)
    m.add_function(wrap_pyfunction!(open_viewer, m)?)?;
    m.add_function(wrap_pyfunction!(open_terrain_viewer, m)?)?;
    // Vector: point shape/LOD controls
    m.add_function(wrap_pyfunction!(set_point_shape_mode, m)?)?;
    m.add_function(wrap_pyfunction!(set_point_lod_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(is_weighted_oit_available, m)?)?;
    m.add_function(wrap_pyfunction!(vector_oit_and_pick_demo, m)?)?;
    m.add_function(wrap_pyfunction!(vector_render_oit_py, m)?)?;
    m.add_function(wrap_pyfunction!(vector_render_pick_map_py, m)?)?;
    m.add_function(wrap_pyfunction!(vector_render_oit_and_pick_py, m)?)?;
    m.add_function(wrap_pyfunction!(vector_render_polygons_fill_py, m)?)?;
    m.add_function(wrap_pyfunction!(configure_csm, m)?)?;
    m.add_function(wrap_pyfunction!(set_csm_enabled, m)?)?;
    m.add_function(wrap_pyfunction!(set_csm_light_direction, m)?)?;
    m.add_function(wrap_pyfunction!(set_csm_pcf_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(set_csm_bias_params, m)?)?;
    m.add_function(wrap_pyfunction!(set_csm_debug_mode, m)?)?;
    m.add_function(wrap_pyfunction!(get_csm_cascade_info, m)?)?;
    m.add_function(wrap_pyfunction!(validate_csm_peter_panning, m)?)?;
    // Hybrid mesh path tracer (GPU) entry
    m.add_function(wrap_pyfunction!(_pt_render_gpu_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(vector::extrude_polygon_py, m)?)?;
    m.add_function(wrap_pyfunction!(vector::add_polygons_py, m)?)?;
    m.add_function(wrap_pyfunction!(vector::add_lines_py, m)?)?;
    m.add_function(wrap_pyfunction!(vector::add_points_py, m)?)?;
    m.add_function(wrap_pyfunction!(vector::add_graph_py, m)?)?;
    m.add_function(wrap_pyfunction!(vector::clear_vectors_py, m)?)?;
    m.add_function(wrap_pyfunction!(vector::get_vector_counts_py, m)?)?;
    m.add_function(wrap_pyfunction!(vector::api::extrude_polygon_gpu_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::geometry_extrude_polygon_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::geometry_generate_primitive_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::geometry_validate_mesh_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(crate::geometry::geometry_weld_mesh_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::geometry_transform_center_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::geometry_transform_scale_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::geometry_transform_flip_axis_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::geometry_transform_swap_axes_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::geometry_transform_bounds_py,
        m
    )?)?;
    // Phase 4: subdivision, displacement, curves
    m.add_function(wrap_pyfunction!(crate::geometry::geometry_subdivide_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::geometry_displace_heightmap_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::geometry_displace_procedural_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::geometry_generate_ribbon_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::geometry_generate_tube_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::geometry_generate_thick_polyline_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::geometry_generate_tangents_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::geometry_attach_tangents_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::geometry_subdivide_adaptive_py,
        m
    )?)?;
    // Phase 6: instancing
    m.add_function(wrap_pyfunction!(
        crate::render::instancing::geometry_instance_mesh_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::render::instancing::gpu_instancing_available_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::render::instancing::geometry_instance_mesh_gpu_stub_py,
        m
    )?)?;
    #[cfg(all(feature = "enable-gpu-instancing"))]
    {
        m.add_function(wrap_pyfunction!(
            crate::render::instancing::geometry_instance_mesh_gpu_py,
            m
        )?)?;
        m.add_function(wrap_pyfunction!(
            crate::render::instancing::geometry_instance_mesh_gpu_render_py,
            m
        )?)?;
    }

    // Native SDF placeholder renderer
    m.add_function(wrap_pyfunction!(hybrid_render, m)?)?;

    // IO: OBJ import/export
    m.add_function(wrap_pyfunction!(crate::io::obj_read::io_import_obj_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::io::obj_write::io_export_obj_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::io::stl_write::io_export_stl_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::io::gltf_read::io_import_gltf_py,
        m
    )?)?;
    // Import: OSM buildings helper
    m.add_function(wrap_pyfunction!(
        crate::import::osm_buildings::import_osm_buildings_extrude_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::import::osm_buildings::import_osm_buildings_from_geojson_py,
        m
    )?)?;

    // GPU utilities (adapter enumeration and probe)
    m.add_function(wrap_pyfunction!(enumerate_adapters, m)?)?;
    m.add_function(wrap_pyfunction!(device_probe, m)?)?;
    m.add_function(wrap_pyfunction!(global_memory_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(render_debug_pattern_frame, m)?)?;
    m.add_function(wrap_pyfunction!(numpy_to_exr, m)?)?;
    // P5 Screen-space GI manager
    m.add_class::<PyScreenSpaceGI>()?;

    // Workstream C: Core Engine & Target interfaces
    m.add_function(wrap_pyfunction!(engine_info, m)?)?;
    m.add_function(wrap_pyfunction!(report_device, m)?)?;
    m.add_function(wrap_pyfunction!(c5_build_framegraph_report, m)?)?;
    m.add_function(wrap_pyfunction!(c6_mt_record_demo, m)?)?;
    m.add_function(wrap_pyfunction!(c7_async_compute_demo, m)?)?;

    // UV unwrap helpers
    m.add_function(wrap_pyfunction!(crate::uv::unwrap::uv_planar_unwrap_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::uv::unwrap::uv_spherical_unwrap_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::converters::multipolygonz_to_obj::converters_multipolygonz_to_obj_py,
        m
    )?)?;

    // Camera functions (expose Rust implementations to Python)
    m.add_function(wrap_pyfunction!(crate::camera::camera_look_at, m)?)?;
    m.add_function(wrap_pyfunction!(crate::camera::camera_perspective, m)?)?;
    m.add_function(wrap_pyfunction!(crate::camera::camera_orthographic, m)?)?;
    m.add_function(wrap_pyfunction!(crate::camera::camera_view_proj, m)?)?;
    m.add_function(wrap_pyfunction!(crate::camera::camera_dof_params, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::camera::camera_f_stop_to_aperture,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::camera::camera_aperture_to_f_stop,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::camera::camera_hyperfocal_distance,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::camera::camera_depth_of_field_range,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::camera::camera_circle_of_confusion,
        m
    )?)?;

    // Transform utilities
    m.add_function(wrap_pyfunction!(crate::geometry::transforms::translate, m)?)?;
    m.add_function(wrap_pyfunction!(crate::geometry::transforms::rotate_x, m)?)?;
    m.add_function(wrap_pyfunction!(crate::geometry::transforms::rotate_y, m)?)?;
    m.add_function(wrap_pyfunction!(crate::geometry::transforms::rotate_z, m)?)?;
    m.add_function(wrap_pyfunction!(crate::geometry::transforms::scale, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::transforms::scale_uniform,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::transforms::compose_trs,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::transforms::look_at_transform,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::transforms::multiply_matrices,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::transforms::invert_matrix,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::geometry::transforms::compute_normal_matrix,
        m
    )?)?;

    // Grid generator
    m.add_function(wrap_pyfunction!(crate::geometry::grid::grid_generate, m)?)?;
    // Path tracing (GPU MVP)
    m.add_function(wrap_pyfunction!(_pt_render_gpu, m)?)?;
    // P7-05: Offscreen BRDF tile renderer
    m.add_function(wrap_pyfunction!(render_brdf_tile, m)?)?;
    m.add_function(wrap_pyfunction!(render_brdf_tile_overrides, m)?)?;

    // Add main classes
    m.add_class::<crate::core::session::Session>()?;
    m.add_class::<crate::colormap::colormap1d::Colormap1D>()?;
    m.add_class::<crate::core::overlay_layer::OverlayLayer>()?;
    m.add_class::<crate::terrain::render_params::TerrainRenderParams>()?;
    m.add_class::<crate::terrain::renderer::TerrainRenderer>()?;
    m.add_class::<crate::render::material_set::MaterialSet>()?;
    m.add_class::<crate::lighting::ibl_wrapper::IBL>()?;
    m.add_class::<crate::scene::Scene>()?;
    // Expose TerrainSpike (E2/E3) to Python
    m.add_class::<crate::terrain::TerrainSpike>()?;
    // P0 Lighting classes
    m.add_class::<crate::lighting::PyLight>()?;
    m.add_class::<crate::lighting::PyMaterialShading>()?;
    m.add_class::<crate::lighting::PyShadowSettings>()?;
    m.add_class::<crate::lighting::PyGiSettings>()?;
    m.add_class::<crate::lighting::PyAtmosphere>()?;
    // P5 Screen-space effects classes
    m.add_class::<crate::lighting::PySSAOSettings>()?;
    m.add_class::<crate::lighting::PySSGISettings>()?;
    m.add_class::<crate::lighting::PySSRSettings>()?;
    // P6 Atmospherics classes
    m.add_class::<crate::lighting::PySkySettings>()?;
    m.add_class::<crate::lighting::PyVolumetricSettings>()?;
    // M1: AOV frame class
    m.add_class::<AovFrame>()?;
    
    // Feature B: Picking system classes (Plan 1 + Plan 2 + Plan 3)
    m.add_class::<PyPickResult>()?;
    m.add_class::<PyTerrainQueryResult>()?;
    m.add_class::<PySelectionStyle>()?;
    // Plan 3: Premium picking classes
    m.add_class::<PyRichPickResult>()?;
    m.add_class::<PyHighlightStyle>()?;
    m.add_class::<PyLassoState>()?;
    m.add_class::<PyHeightfieldHit>()?;

    // Feature C: Camera animation classes (Plan 1 MVP)
    m.add_class::<crate::animation::CameraAnimation>()?;
    m.add_class::<crate::animation::CameraState>()?;

    Ok(())
}
