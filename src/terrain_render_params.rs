// src/terrain_render_params.rs
// PyO3 terrain render parameter wrapper bridging Python configs to Rust
// Exists to store validated terrain settings in a native-friendly structure
// RELEVANT FILES: python/forge3d/terrain_params.py, src/overlay_layer.rs, src/terrain_renderer.rs, tests/test_terrain_render_params_native.py
#[cfg(feature = "extension-module")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "extension-module")]
use pyo3::prelude::*;
#[cfg(feature = "extension-module")]
use std::sync::Arc;
#[cfg(feature = "extension-module")]
fn tuple_to_u32_pair(value: &PyAny, name: &str) -> PyResult<(u32, u32)> {
    let pair: (i64, i64) = value.extract().map_err(|_| {
        PyValueError::new_err(format!(
            "{} must be a tuple of two integers, got {:?}",
            name, value
        ))
    })?;
    if pair.0 < 0 || pair.1 < 0 {
        return Err(PyValueError::new_err(format!(
            "{} components must be non-negative",
            name
        )));
    }
    Ok((pair.0 as u32, pair.1 as u32))
}

#[cfg(feature = "extension-module")]
fn tuple_to_f32_pair(value: &PyAny, name: &str) -> PyResult<(f32, f32)> {
    let pair: (f32, f32) = value.extract().map_err(|_| {
        PyValueError::new_err(format!(
            "{} must be a tuple of two floats, got {:?}",
            name, value
        ))
    })?;
    if !pair.0.is_finite() || !pair.1.is_finite() {
        return Err(PyValueError::new_err(format!(
            "{} values must be finite floats",
            name
        )));
    }
    Ok(pair)
}

#[cfg(feature = "extension-module")]
fn list_to_vec3(value: &PyAny, name: &str) -> PyResult<[f32; 3]> {
    let seq: Vec<f32> = value.extract().map_err(|_| {
        PyValueError::new_err(format!(
            "{} must be a list of three floats, got {:?}",
            name, value
        ))
    })?;
    if seq.len() != 3 {
        return Err(PyValueError::new_err(format!(
            "{} must contain exactly three floats",
            name
        )));
    }
    if seq.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err(format!(
            "{} entries must be finite floats",
            name
        )));
    }
    Ok([seq[0], seq[1], seq[2]])
}

#[cfg(feature = "extension-module")]
fn to_finite_f32(value: &PyAny, name: &str) -> PyResult<f32> {
    let v: f32 = value
        .extract()
        .map_err(|_| PyValueError::new_err(format!("{} must be a float value", name)))?;
    if !v.is_finite() {
        return Err(PyValueError::new_err(format!(
            "{} must be a finite float",
            name
        )));
    }
    Ok(v)
}

#[cfg(feature = "extension-module")]
fn extract_overlays(obj: &PyAny) -> PyResult<Vec<Py<crate::overlay_layer::OverlayLayer>>> {
    if obj.is_none() {
        return Ok(Vec::new());
    }
    obj.extract().map_err(|_| {
        PyValueError::new_err("overlays must be a sequence of forge3d.OverlayLayer objects")
    })
}

#[cfg(feature = "extension-module")]
#[derive(Clone, Copy)]
pub enum FilterModeNative {
    Linear,
    Nearest,
}

#[cfg(feature = "extension-module")]
#[derive(Clone, Copy)]
pub enum AddressModeNative {
    Repeat,
    ClampToEdge,
    MirrorRepeat,
}

#[cfg(feature = "extension-module")]
#[derive(Clone)]
pub struct LightSettingsNative {
    pub direction: [f32; 3],
    pub intensity: f32,
    pub color: [f32; 3],
}

#[cfg(feature = "extension-module")]
#[derive(Clone)]
pub struct TriplanarSettingsNative {
    pub scale: f32,
    pub blend_sharpness: f32,
    pub normal_strength: f32,
}

#[cfg(feature = "extension-module")]
#[derive(Clone)]
pub struct PomSettingsNative {
    pub enabled: bool,
    pub scale: f32,
    pub min_steps: u32,
    pub max_steps: u32,
    pub refine_steps: u32,
    pub shadow: bool,
    pub occlusion: bool,
}

#[cfg(feature = "extension-module")]
#[derive(Clone)]
pub struct LodSettingsNative {
    pub level: i32,
    pub bias: f32,
    pub lod0_bias: f32,
}

#[cfg(feature = "extension-module")]
#[derive(Clone)]
pub struct ClampSettingsNative {
    pub height_range: (f32, f32),
    pub slope_range: (f32, f32),
    pub ambient_range: (f32, f32),
    pub shadow_range: (f32, f32),
    pub occlusion_range: (f32, f32),
}

#[cfg(feature = "extension-module")]
#[derive(Clone)]
pub struct SamplingSettingsNative {
    pub mag_filter: FilterModeNative,
    pub min_filter: FilterModeNative,
    pub mip_filter: FilterModeNative,
    pub anisotropy: u32,
    pub address_u: AddressModeNative,
    pub address_v: AddressModeNative,
    pub address_w: AddressModeNative,
}

/// Shadow settings extracted from Python ShadowSettings dataclass
#[cfg(feature = "extension-module")]
#[derive(Clone)]
#[allow(dead_code)]
pub struct ShadowSettingsNative {
    pub enabled: bool,
    pub technique: String,
    pub resolution: u32,
    pub cascades: u32,
    pub max_distance: f32,
    pub softness: f32,
    pub pcss_light_radius: f32,
    pub intensity: f32,
    pub slope_scale_bias: f32,
    pub depth_bias: f32,
    pub normal_bias: f32,
}

#[cfg(feature = "extension-module")]
impl Default for ShadowSettingsNative {
    fn default() -> Self {
        Self {
            enabled: true,
            technique: "PCSS".to_string(),
            resolution: 2048,
            cascades: 4,
            max_distance: 3000.0,
            softness: 0.01,
            pcss_light_radius: 0.0,
            intensity: 1.0,
            slope_scale_bias: 0.001,
            depth_bias: 0.0005,
            normal_bias: 0.0002,
        }
    }
}

/// P2: Fog settings extracted from Python FogSettings or CLI args
/// When density = 0.0, fog is disabled (no-op for P1 compatibility)
#[cfg(feature = "extension-module")]
#[derive(Clone)]
pub struct FogSettingsNative {
    /// Fog density coefficient (0.0 = disabled)
    pub density: f32,
    /// Height falloff rate (higher = fog thins faster at altitude)
    pub height_falloff: f32,
    /// Base height for fog calculation (world-space Y)
    pub base_height: f32,
    /// Inscatter color (linear RGB, typically sky-tinted)
    pub inscatter: [f32; 3],
}

#[cfg(feature = "extension-module")]
impl Default for FogSettingsNative {
    fn default() -> Self {
        Self {
            density: 0.0,       // Disabled by default (P1 compatibility)
            height_falloff: 0.0,
            base_height: 0.0,
            inscatter: [1.0, 1.0, 1.0],
        }
    }
}

/// P4: Water reflection settings extracted from Python ReflectionSettings or CLI args
/// When enabled = false, reflections are disabled (no-op for P3 compatibility)
#[cfg(feature = "extension-module")]
#[derive(Clone)]
pub struct ReflectionSettingsNative {
    /// Enable water planar reflections
    pub enabled: bool,
    /// Reflection intensity (0.0-1.0, default 0.8)
    pub intensity: f32,
    /// Fresnel power for reflection falloff (default 5.0)
    pub fresnel_power: f32,
    /// Wave-based UV distortion strength (default 0.02)
    pub wave_strength: f32,
    /// Shore attenuation width - reduce reflections near land (default 0.3)
    pub shore_atten_width: f32,
    /// Water plane height in world space (default 0.0 = sea level)
    pub water_plane_height: f32,
}

#[cfg(feature = "extension-module")]
impl Default for ReflectionSettingsNative {
    fn default() -> Self {
        Self {
            enabled: false,     // Disabled by default (P3 compatibility)
            intensity: 0.8,
            fresnel_power: 5.0,
            wave_strength: 0.02,
            shore_atten_width: 0.3,
            water_plane_height: 0.0,
        }
    }
}

/// P6: Micro-detail settings for close-range surface enhancement
/// When enabled = false, micro-detail is disabled (no-op for P5 compatibility)
#[cfg(feature = "extension-module")]
#[derive(Clone)]
pub struct DetailSettingsNative {
    /// Enable micro-detail enhancement
    pub enabled: bool,
    /// World-space repeat interval for detail (default 2.0 meters)
    pub detail_scale: f32,
    /// Detail normal blending strength (0.0-1.0)
    pub normal_strength: f32,
    /// Albedo brightness noise amplitude (±percentage)
    pub albedo_noise: f32,
    /// Distance at which detail begins fading (world units)
    pub fade_start: f32,
    /// Distance at which detail is fully faded (world units)
    pub fade_end: f32,
}

#[cfg(feature = "extension-module")]
impl Default for DetailSettingsNative {
    fn default() -> Self {
        Self {
            enabled: false,     // Disabled by default (P5 compatibility)
            detail_scale: 2.0,
            normal_strength: 0.3,
            albedo_noise: 0.1,
            fade_start: 50.0,
            fade_end: 200.0,
        }
    }
}

#[cfg(feature = "extension-module")]
#[derive(Clone)]
pub struct DecodedTerrainSettings {
    pub light: LightSettingsNative,
    pub triplanar: TriplanarSettingsNative,
    pub pom: PomSettingsNative,
    pub lod: LodSettingsNative,
    pub clamp: ClampSettingsNative,
    pub sampling: SamplingSettingsNative,
    pub shadow: ShadowSettingsNative,
    pub fog: FogSettingsNative,
    pub reflection: ReflectionSettingsNative,
    pub detail: DetailSettingsNative,
}

#[cfg(feature = "extension-module")]
fn normalize_direction(x: f32, y: f32, z: f32) -> [f32; 3] {
    let len = (x * x + y * y + z * z).sqrt();
    if len <= 1e-6 {
        [0.0, 1.0, 0.0]
    } else {
        [x / len, y / len, z / len]
    }
}

#[cfg(feature = "extension-module")]
fn parse_filter_mode(value: &str, field: &str) -> PyResult<FilterModeNative> {
    match value {
        "Linear" | "linear" => Ok(FilterModeNative::Linear),
        "Nearest" | "nearest" => Ok(FilterModeNative::Nearest),
        other => Err(PyValueError::new_err(format!(
            "{} must be 'Linear' or 'Nearest', got {}",
            field, other
        ))),
    }
}

#[cfg(feature = "extension-module")]
fn parse_address_mode(value: &str, field: &str) -> PyResult<AddressModeNative> {
    match value {
        "Repeat" | "repeat" => Ok(AddressModeNative::Repeat),
        "ClampToEdge" | "clamp_to_edge" | "Clamp" | "clamp" => Ok(AddressModeNative::ClampToEdge),
        "MirrorRepeat" | "mirror_repeat" => Ok(AddressModeNative::MirrorRepeat),
        other => Err(PyValueError::new_err(format!(
            "{} must be 'Repeat', 'ClampToEdge', or 'MirrorRepeat', got {}",
            field, other
        ))),
    }
}

/// Terrain render parameter wrapper used by the native renderer.
#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "TerrainRenderParams")]
#[derive(Clone)]
pub struct TerrainRenderParams {
    pub size_px: (u32, u32),
    pub render_scale: f32,
    pub msaa_samples: u32,
    pub z_scale: f32,
    pub cam_target: [f32; 3],
    pub cam_radius: f32,
    pub cam_phi_deg: f32,
    pub cam_theta_deg: f32,
    pub cam_gamma_deg: f32,
    pub fov_y_deg: f32,
    pub clip: (f32, f32),
    pub exposure: f32,
    pub gamma: f32,
    pub albedo_mode: String,
    pub colormap_strength: f32,
    /// P5: AO weight/multiplier (0.0 = no AO effect, 1.0 = full AO). Default 0.0 for P4 compatibility.
    pub ao_weight: f32,
    pub height_curve_mode: String,
    pub height_curve_strength: f32,
    pub height_curve_power: f32,
    pub height_curve_lut: Option<Arc<Vec<f32>>>,
    pub overlays: Vec<Py<crate::overlay_layer::OverlayLayer>>,
    light: Py<PyAny>,
    ibl: Py<PyAny>,
    shadows: Py<PyAny>,
    triplanar: Py<PyAny>,
    pom: Py<PyAny>,
    lod: Py<PyAny>,
    sampling: Py<PyAny>,
    clamp: Py<PyAny>,
    python_object: Py<PyAny>,
    decoded: DecodedTerrainSettings,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl TerrainRenderParams {
    #[new]
    #[pyo3(signature = (params))]
    pub fn new(py: Python<'_>, params: Bound<'_, PyAny>) -> PyResult<Self> {
        let size_px = tuple_to_u32_pair(params.getattr("size_px")?.as_gil_ref(), "size_px")?;
        let render_scale =
            to_finite_f32(params.getattr("render_scale")?.as_gil_ref(), "render_scale")?;
        let msaa_samples: u32 = params
            .getattr("msaa_samples")?
            .extract::<u32>()
            .map_err(|_| PyValueError::new_err("msaa_samples must be an integer >= 1"))?;
        let z_scale = to_finite_f32(params.getattr("z_scale")?.as_gil_ref(), "z_scale")?;
        if !matches!(msaa_samples, 1 | 2 | 4 | 8) {
            return Err(PyValueError::new_err(
                "msaa_samples must be one of 1, 2, 4, or 8",
            ));
        }
        if render_scale <= 0.0 {
            return Err(PyValueError::new_err("render_scale must be positive"));
        }
        if z_scale <= 0.0 {
            return Err(PyValueError::new_err("z_scale must be positive"));
        }

        let cam_target = list_to_vec3(params.getattr("cam_target")?.as_gil_ref(), "cam_target")?;
        let cam_radius = to_finite_f32(params.getattr("cam_radius")?.as_gil_ref(), "cam_radius")?;
        let cam_phi_deg =
            to_finite_f32(params.getattr("cam_phi_deg")?.as_gil_ref(), "cam_phi_deg")?;
        let cam_theta_deg = to_finite_f32(
            params.getattr("cam_theta_deg")?.as_gil_ref(),
            "cam_theta_deg",
        )?;
        if cam_radius <= 0.0 {
            return Err(PyValueError::new_err("cam_radius must be positive"));
        }
        let cam_gamma_deg = params
            .getattr("cam_gamma_deg")?
            .extract::<f32>()
            .map_err(|_| PyValueError::new_err("cam_gamma_deg must be a float value"))?;
        let fov_y_deg = to_finite_f32(params.getattr("fov_y_deg")?.as_gil_ref(), "fov_y_deg")?;
        if !(0.0..=180.0).contains(&fov_y_deg) {
            return Err(PyValueError::new_err("fov_y_deg must be within [0, 180]"));
        }
        let clip = tuple_to_f32_pair(params.getattr("clip")?.as_gil_ref(), "clip")?;
        if clip.0 <= 0.0 || clip.0 >= clip.1 {
            return Err(PyValueError::new_err(
                "clip tuple must satisfy near > 0 and near < far",
            ));
        }

        let exposure = to_finite_f32(params.getattr("exposure")?.as_gil_ref(), "exposure")?;
        let gamma = to_finite_f32(params.getattr("gamma")?.as_gil_ref(), "gamma")?;
        if gamma <= 0.0 {
            return Err(PyValueError::new_err("gamma must be positive"));
        }
        let albedo_mode: String = params
            .getattr("albedo_mode")?
            .extract()
            .map_err(|_| PyValueError::new_err("albedo_mode must be a string"))?;
        let colormap_strength = to_finite_f32(
            params.getattr("colormap_strength")?.as_gil_ref(),
            "colormap_strength",
        )?;
        match albedo_mode.as_str() {
            "colormap" | "mix" | "material" => {}
            other => {
                return Err(PyValueError::new_err(format!(
                    "albedo_mode '{}' is not supported",
                    other
                )))
            }
        }
        if !(0.0..=1.0).contains(&colormap_strength) {
            return Err(PyValueError::new_err(
                "colormap_strength must be between 0 and 1",
            ));
        }

        // P5: AO weight (optional, default 0.0 for backward compatibility)
        let ao_weight = params
            .getattr("ao_weight")
            .ok()
            .and_then(|v| v.extract::<f32>().ok())
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);

        let height_curve_mode: String = params
            .getattr("height_curve_mode")?
            .extract()
            .map_err(|_| PyValueError::new_err("height_curve_mode must be a string"))?;
        let valid_modes = ["linear", "pow", "smoothstep", "lut"];
        if !valid_modes.contains(&height_curve_mode.as_str()) {
            return Err(PyValueError::new_err(format!(
                "height_curve_mode must be one of {:?}, got {}",
                valid_modes, height_curve_mode
            )));
        }
        let height_curve_strength = to_finite_f32(
            params.getattr("height_curve_strength")?.as_gil_ref(),
            "height_curve_strength",
        )?
        .clamp(0.0, 1.0);
        let height_curve_power = to_finite_f32(
            params.getattr("height_curve_power")?.as_gil_ref(),
            "height_curve_power",
        )?;
        if !(height_curve_power > 0.0) {
            return Err(PyValueError::new_err(
                "height_curve_power must be greater than zero",
            ));
        }

        let height_curve_lut: Option<Arc<Vec<f32>>> = if height_curve_mode == "lut" {
            let raw_lut = params.getattr("height_curve_lut")?;
            let lut_vec: Vec<f32> = raw_lut.extract().map_err(|_| {
                PyValueError::new_err("height_curve_lut must be convertible to a 1D float array")
            })?;
            if lut_vec.len() != 256 {
                return Err(PyValueError::new_err(
                    "height_curve_lut must have length 256 when height_curve_mode='lut'",
                ));
            }
            if lut_vec.iter().any(|v| !v.is_finite() || *v < 0.0 || *v > 1.0) {
                return Err(PyValueError::new_err(
                    "height_curve_lut values must be finite floats within [0, 1]",
                ));
            }
            Some(Arc::new(lut_vec))
        } else {
            None
        };

        let light = params.getattr("light")?;
        let ibl = params.getattr("ibl")?;
        let shadows = params.getattr("shadows")?;
        let triplanar = params.getattr("triplanar")?;
        let pom = params.getattr("pom")?;
        let lod = params.getattr("lod")?;
        let sampling = params.getattr("sampling")?;
        let clamp = params.getattr("clamp")?;

        let light_type: String = light.getattr("light_type")?.extract()?;
        let azimuth = to_finite_f32(
            light.getattr("azimuth_deg")?.as_gil_ref(),
            "light.azimuth_deg",
        )?;
        let elevation = to_finite_f32(
            light.getattr("elevation_deg")?.as_gil_ref(),
            "light.elevation_deg",
        )?;
        let light_intensity =
            to_finite_f32(light.getattr("intensity")?.as_gil_ref(), "light.intensity")?.max(0.0);
        let light_color: Vec<f32> = light
            .getattr("color")?
            .extract()
            .map_err(|_| PyValueError::new_err("light.color must be a sequence of three floats"))?;
        if light_color.len() != 3 {
            return Err(PyValueError::new_err(
                "light.color must contain exactly three components",
            ));
        }
        let azimuth_rad = azimuth.to_radians();
        let elevation_rad = elevation.to_radians();
        let cos_el = elevation_rad.cos();
        let sin_el = elevation_rad.sin();
        // Z-up coordinate system: X=East, Y=North, Z=Up
        // Azimuth 0° = East, 90° = South, 180° = West, 270° = North
        // Elevation 0° = horizon, 90° = zenith
        let direction = match light_type.as_str() {
            "Directional" | "directional" => normalize_direction(
                cos_el * azimuth_rad.cos(),   // X: horizontal component in azimuth direction
                cos_el * azimuth_rad.sin(),   // Y: horizontal component perpendicular
                sin_el,                       // Z: vertical component (up for Z-up system)
            ),
            _ => normalize_direction(
                cos_el * azimuth_rad.cos(),
                cos_el * azimuth_rad.sin(),
                sin_el,
            ),
        };

        let triplanar_native = TriplanarSettingsNative {
            scale: to_finite_f32(triplanar.getattr("scale")?.as_gil_ref(), "triplanar.scale")?,
            blend_sharpness: to_finite_f32(
                triplanar.getattr("blend_sharpness")?.as_gil_ref(),
                "triplanar.blend_sharpness",
            )?,
            normal_strength: to_finite_f32(
                triplanar.getattr("normal_strength")?.as_gil_ref(),
                "triplanar.normal_strength",
            )?,
        };

        let pom_native = PomSettingsNative {
            enabled: pom.getattr("enabled")?.extract()?,
            scale: to_finite_f32(pom.getattr("scale")?.as_gil_ref(), "pom.scale")?,
            min_steps: pom.getattr("min_steps")?.extract::<i64>()? as u32,
            max_steps: pom.getattr("max_steps")?.extract::<i64>()? as u32,
            refine_steps: pom.getattr("refine_steps")?.extract::<i64>()? as u32,
            shadow: pom.getattr("shadow")?.extract()?,
            occlusion: pom.getattr("occlusion")?.extract()?,
        };

        let lod_native = LodSettingsNative {
            level: lod.getattr("level")?.extract::<i64>()? as i32,
            bias: to_finite_f32(lod.getattr("bias")?.as_gil_ref(), "lod.bias")?,
            lod0_bias: to_finite_f32(lod.getattr("lod0_bias")?.as_gil_ref(), "lod.lod0_bias")?,
        };

        let clamp_native = ClampSettingsNative {
            height_range: tuple_to_f32_pair(
                clamp.getattr("height_range")?.as_gil_ref(),
                "clamp.height_range",
            )?,
            slope_range: tuple_to_f32_pair(
                clamp.getattr("slope_range")?.as_gil_ref(),
                "clamp.slope_range",
            )?,
            ambient_range: tuple_to_f32_pair(
                clamp.getattr("ambient_range")?.as_gil_ref(),
                "clamp.ambient_range",
            )?,
            shadow_range: tuple_to_f32_pair(
                clamp.getattr("shadow_range")?.as_gil_ref(),
                "clamp.shadow_range",
            )?,
            occlusion_range: tuple_to_f32_pair(
                clamp.getattr("occlusion_range")?.as_gil_ref(),
                "clamp.occlusion_range",
            )?,
        };

        let sampling_native = SamplingSettingsNative {
            mag_filter: parse_filter_mode(
                &sampling.getattr("mag_filter")?.extract::<String>()?,
                "sampling.mag_filter",
            )?,
            min_filter: parse_filter_mode(
                &sampling.getattr("min_filter")?.extract::<String>()?,
                "sampling.min_filter",
            )?,
            mip_filter: parse_filter_mode(
                &sampling.getattr("mip_filter")?.extract::<String>()?,
                "sampling.mip_filter",
            )?,
            anisotropy: sampling
                .getattr("anisotropy")?
                .extract::<i64>()?
                .clamp(1, 16) as u32,
            address_u: parse_address_mode(
                &sampling.getattr("address_u")?.extract::<String>()?,
                "sampling.address_u",
            )?,
            address_v: parse_address_mode(
                &sampling.getattr("address_v")?.extract::<String>()?,
                "sampling.address_v",
            )?,
            address_w: parse_address_mode(
                &sampling.getattr("address_w")?.extract::<String>()?,
                "sampling.address_w",
            )?,
        };

        // Parse shadow settings from Python ShadowSettings dataclass
        let softness = shadows.getattr("softness")?.extract().unwrap_or(0.01);
        // Optional PCSS light radius: default to 0.0 to preserve hard-shadow baseline when unset
        let pcss_light_radius = shadows
            .getattr("pcss_light_radius")
            .ok()
            .and_then(|value| value.extract().ok())
            .unwrap_or(0.0);
        let shadow_native = ShadowSettingsNative {
            enabled: shadows.getattr("enabled")?.extract().unwrap_or(true),
            technique: shadows
                .getattr("technique")?
                .extract::<String>()
                .unwrap_or_else(|_| "PCSS".to_string()),
            resolution: shadows
                .getattr("resolution")?
                .extract::<i64>()
                .unwrap_or(2048) as u32,
            cascades: shadows
                .getattr("cascades")?
                .extract::<i64>()
                .unwrap_or(1) as u32,
            max_distance: shadows
                .getattr("max_distance")?
                .extract()
                .unwrap_or(3000.0),
            softness,
            pcss_light_radius,
            intensity: shadows.getattr("intensity")?.extract().unwrap_or(1.0),
            slope_scale_bias: shadows
                .getattr("slope_scale_bias")?
                .extract()
                .unwrap_or(0.001),
            depth_bias: shadows.getattr("depth_bias")?.extract().unwrap_or(0.0005),
            normal_bias: shadows
                .getattr("normal_bias")?
                .extract()
                .unwrap_or(0.0002),
        };

        // P2: Extract fog settings (defaults to disabled for P1 compatibility)
        let fog_native = if let Ok(fog) = params.getattr("fog") {
            // Extract inscatter as Vec<f32> then convert to [f32; 3] (tuple extraction can fail)
            let inscatter_vec: Vec<f32> = fog
                .getattr("inscatter")
                .and_then(|v| v.extract())
                .unwrap_or_else(|_| vec![1.0, 1.0, 1.0]);
            let inscatter = [
                inscatter_vec.first().copied().unwrap_or(1.0),
                inscatter_vec.get(1).copied().unwrap_or(1.0),
                inscatter_vec.get(2).copied().unwrap_or(1.0),
            ];
            let density: f32 = fog.getattr("density").and_then(|v| v.extract()).unwrap_or(0.0);
            let height_falloff: f32 = fog.getattr("height_falloff").and_then(|v| v.extract()).unwrap_or(0.0);
            let base_height: f32 = fog.getattr("base_height").and_then(|v| v.extract()).unwrap_or(0.0);
            FogSettingsNative {
                density,
                height_falloff,
                base_height,
                inscatter,
            }
        } else {
            FogSettingsNative::default()
        };

        // P4: Extract reflection settings (defaults to disabled for P3 compatibility)
        let reflection_native = if let Ok(refl) = params.getattr("reflection") {
            let enabled: bool = refl.getattr("enabled").and_then(|v| v.extract()).unwrap_or(false);
            let intensity: f32 = refl.getattr("intensity").and_then(|v| v.extract()).unwrap_or(0.8);
            let fresnel_power: f32 = refl.getattr("fresnel_power").and_then(|v| v.extract()).unwrap_or(5.0);
            let wave_strength: f32 = refl.getattr("wave_strength").and_then(|v| v.extract()).unwrap_or(0.02);
            let shore_atten_width: f32 = refl.getattr("shore_atten_width").and_then(|v| v.extract()).unwrap_or(0.3);
            let water_plane_height: f32 = refl.getattr("water_plane_height").and_then(|v| v.extract()).unwrap_or(0.0);
            ReflectionSettingsNative {
                enabled,
                intensity,
                fresnel_power,
                wave_strength,
                shore_atten_width,
                water_plane_height,
            }
        } else {
            ReflectionSettingsNative::default()
        };

        // P6: Extract detail settings (defaults to disabled for P5 compatibility)
        let detail_native = if let Ok(det) = params.getattr("detail") {
            let enabled: bool = det.getattr("enabled").and_then(|v| v.extract()).unwrap_or(false);
            let detail_scale: f32 = det.getattr("detail_scale").and_then(|v| v.extract()).unwrap_or(2.0);
            let normal_strength: f32 = det.getattr("normal_strength").and_then(|v| v.extract()).unwrap_or(0.3);
            let albedo_noise: f32 = det.getattr("albedo_noise").and_then(|v| v.extract()).unwrap_or(0.1);
            let fade_start: f32 = det.getattr("fade_start").and_then(|v| v.extract()).unwrap_or(50.0);
            let fade_end: f32 = det.getattr("fade_end").and_then(|v| v.extract()).unwrap_or(200.0);
            DetailSettingsNative {
                enabled,
                detail_scale,
                normal_strength,
                albedo_noise,
                fade_start,
                fade_end,
            }
        } else {
            DetailSettingsNative::default()
        };

        let decoded = DecodedTerrainSettings {
            light: LightSettingsNative {
                direction,
                intensity: light_intensity,
                color: [light_color[0], light_color[1], light_color[2]],
            },
            triplanar: triplanar_native,
            pom: pom_native,
            lod: lod_native,
            clamp: clamp_native,
            sampling: sampling_native,
            shadow: shadow_native,
            fog: fog_native,
            reflection: reflection_native,
            detail: detail_native,
        };

        let overlays = extract_overlays(params.getattr("overlays")?.as_gil_ref())?;

        Ok(Self {
            size_px,
            render_scale,
            msaa_samples,
            z_scale,
            cam_target,
            cam_radius,
            cam_phi_deg,
            cam_theta_deg,
            cam_gamma_deg,
            fov_y_deg,
            clip,
            exposure,
            gamma,
            albedo_mode,
            colormap_strength,
            ao_weight,
            height_curve_mode,
            height_curve_strength,
            height_curve_power,
            height_curve_lut,
            overlays,
            light: light.unbind(),
            ibl: ibl.unbind(),
            shadows: shadows.unbind(),
            triplanar: triplanar.unbind(),
            pom: pom.unbind(),
            lod: lod.unbind(),
            sampling: sampling.unbind(),
            clamp: clamp.unbind(),
            python_object: params.into_py(py),
            decoded,
        })
    }

    #[getter]
    pub fn size_px(&self) -> (u32, u32) {
        self.size_px
    }

    #[getter]
    pub fn render_scale(&self) -> f32 {
        self.render_scale
    }

    #[getter]
    pub fn msaa_samples(&self) -> u32 {
        self.msaa_samples
    }

    #[getter]
    pub fn z_scale(&self) -> f32 {
        self.z_scale
    }

    #[getter]
    pub fn cam_target(&self) -> [f32; 3] {
        self.cam_target
    }

    #[getter]
    pub fn cam_radius(&self) -> f32 {
        self.cam_radius
    }

    #[getter]
    pub fn cam_phi_deg(&self) -> f32 {
        self.cam_phi_deg
    }

    #[getter]
    pub fn cam_theta_deg(&self) -> f32 {
        self.cam_theta_deg
    }

    #[getter]
    pub fn cam_gamma_deg(&self) -> f32 {
        self.cam_gamma_deg
    }

    #[getter]
    pub fn fov_y_deg(&self) -> f32 {
        self.fov_y_deg
    }

    #[getter]
    pub fn clip(&self) -> (f32, f32) {
        self.clip
    }

    #[getter]
    pub fn exposure(&self) -> f32 {
        self.exposure
    }

    #[getter]
    pub fn gamma(&self) -> f32 {
        self.gamma
    }

    #[getter]
    pub fn albedo_mode(&self) -> &str {
        &self.albedo_mode
    }

    #[getter]
    pub fn colormap_strength(&self) -> f32 {
        self.colormap_strength
    }

    /// P5: Get AO weight (0.0 = no AO, 1.0 = full AO)
    #[getter]
    pub fn ao_weight(&self) -> f32 {
        self.ao_weight
    }

    #[getter]
    pub fn height_curve_mode(&self) -> &str {
        &self.height_curve_mode
    }

    #[getter]
    pub fn height_curve_strength(&self) -> f32 {
        self.height_curve_strength
    }

    #[getter]
    pub fn height_curve_power(&self) -> f32 {
        self.height_curve_power
    }

    #[getter]
    pub fn height_curve_lut(&self) -> Option<Vec<f32>> {
        self.height_curve_lut.as_ref().map(|lut| lut.as_ref().clone())
    }

    #[getter]
    pub fn overlays(&self) -> Vec<Py<crate::overlay_layer::OverlayLayer>> {
        self.overlays.clone()
    }

    #[getter]
    pub fn light<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        self.light.clone_ref(py)
    }

    #[getter]
    pub fn ibl<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        self.ibl.clone_ref(py)
    }

    #[getter]
    pub fn shadows<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        self.shadows.clone_ref(py)
    }

    #[getter]
    pub fn triplanar<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        self.triplanar.clone_ref(py)
    }

    #[getter]
    pub fn pom<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        self.pom.clone_ref(py)
    }

    #[getter]
    pub fn lod<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        self.lod.clone_ref(py)
    }

    #[getter]
    pub fn sampling<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        self.sampling.clone_ref(py)
    }

    #[getter]
    pub fn clamp<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        self.clamp.clone_ref(py)
    }

    #[getter]
    pub fn python_object<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        self.python_object.clone_ref(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "TerrainRenderParams(size_px=({},{}) , overlays={}, msaa_samples={})",
            self.size_px.0,
            self.size_px.1,
            self.overlays.len(),
            self.msaa_samples
        )
    }
}

#[cfg(feature = "extension-module")]
impl TerrainRenderParams {
    pub(crate) fn decoded(&self) -> &DecodedTerrainSettings {
        &self.decoded
    }
}
