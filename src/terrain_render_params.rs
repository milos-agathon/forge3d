// src/terrain_render_params.rs
// PyO3 terrain render parameter wrapper bridging Python configs to Rust
// Exists to store validated terrain settings in a native-friendly structure
// RELEVANT FILES: python/forge3d/terrain_params.py, src/overlay_layer.rs, src/terrain_renderer.rs, tests/test_terrain_render_params_native.py
#[cfg(feature = "extension-module")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "extension-module")]
use pyo3::prelude::*;
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

#[cfg(feature = "extension-module")]
#[derive(Clone)]
pub struct DecodedTerrainSettings {
    pub light: LightSettingsNative,
    pub triplanar: TriplanarSettingsNative,
    pub pom: PomSettingsNative,
    pub lod: LodSettingsNative,
    pub clamp: ClampSettingsNative,
    pub sampling: SamplingSettingsNative,
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
        let direction = match light_type.as_str() {
            "Directional" | "directional" => normalize_direction(
                cos_el * azimuth_rad.cos(),
                elevation_rad.sin(),
                cos_el * azimuth_rad.sin(),
            ),
            _ => normalize_direction(
                cos_el * azimuth_rad.cos(),
                elevation_rad.sin(),
                cos_el * azimuth_rad.sin(),
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
