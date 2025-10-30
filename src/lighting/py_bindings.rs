// src/lighting/py_bindings.rs
// PyO3 bindings for P0 lighting types
// Exposes Rust lighting types to Python

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::exceptions::PyValueError;

use super::types::*;

// Helper function to extract f32 array from Python
fn extract_f32_array<const N: usize>(obj: &PyAny, name: &str) -> PyResult<[f32; N]> {
    let list: &PyList = obj.downcast()
        .map_err(|_| PyValueError::new_err(format!("{} must be a list", name)))?;

    if list.len() != N {
        return Err(PyValueError::new_err(format!("{} must have {} elements", name, N)));
    }

    let mut arr = [0.0f32; N];
    for (i, item) in list.iter().enumerate() {
        arr[i] = item.extract::<f32>()
            .map_err(|_| PyValueError::new_err(format!("{} elements must be floats", name)))?;
    }
    Ok(arr)
}

/// Python wrapper for Light configuration
#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "Light")]
#[derive(Clone)]
pub struct PyLight {
    #[pyo3(get, set)]
    pub light_type: String,
    #[pyo3(get, set)]
    pub intensity: f32,
    #[pyo3(get, set)]
    pub color: Vec<f32>,
    #[pyo3(get, set)]
    pub azimuth: f32,
    #[pyo3(get, set)]
    pub elevation: f32,
    #[pyo3(get, set)]
    pub position: Vec<f32>,
    #[pyo3(get, set)]
    pub direction: Vec<f32>,
    #[pyo3(get, set)]
    pub range: f32,
    #[pyo3(get, set)]
    pub spot_inner: f32,
    #[pyo3(get, set)]
    pub spot_outer: f32,
    #[pyo3(get, set)]
    pub env_texture_index: u32,
    #[pyo3(get, set)]
    pub area_width: f32,
    #[pyo3(get, set)]
    pub area_height: f32,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyLight {
    #[new]
    #[pyo3(signature = (light_type="Directional", intensity=3.0, color=None, azimuth=135.0, elevation=35.0, position=None, direction=None, range=2000.0, spot_inner=20.0, spot_outer=35.0, env_texture_index=0, area_width=1.0, area_height=1.0))]
    pub fn new(
        light_type: &str,
        intensity: f32,
        color: Option<Vec<f32>>,
        azimuth: f32,
        elevation: f32,
        position: Option<Vec<f32>>,
        direction: Option<Vec<f32>>,
        range: f32,
        spot_inner: f32,
        spot_outer: f32,
        env_texture_index: u32,
        area_width: f32,
        area_height: f32,
    ) -> PyResult<Self> {
        let color = color.unwrap_or(vec![1.0, 1.0, 1.0]);
        let position = position.unwrap_or(vec![0.0, 0.0, 0.0]);
        let direction = direction.unwrap_or(vec![0.0, -1.0, 0.0]);

        if color.len() != 3 {
            return Err(PyValueError::new_err("color must have 3 elements (RGB)"));
        }
        if position.len() != 3 {
            return Err(PyValueError::new_err("position must have 3 elements (XYZ)"));
        }
        if direction.len() != 3 {
            return Err(PyValueError::new_err("direction must have 3 elements (XYZ)"));
        }

        Ok(Self {
            light_type: light_type.to_string(),
            intensity,
            color,
            azimuth,
            elevation,
            position,
            direction,
            range,
            spot_inner,
            spot_outer,
            env_texture_index,
            area_width,
            area_height,
        })
    }
}

impl PyLight {
    /// Convert to native Rust type (not exposed to Python)
    pub fn to_native(&self) -> PyResult<Light> {
        let color = [self.color[0], self.color[1], self.color[2]];
        let position = [self.position[0], self.position[1], self.position[2]];
        let direction = [self.direction[0], self.direction[1], self.direction[2]];

        let light = match self.light_type.as_str() {
            "Directional" => Light::directional(self.azimuth, self.elevation, self.intensity, color),
            "Point" => Light::point(position, self.range, self.intensity, color),
            "Spot" => Light::spot(
                position,
                direction,
                self.range,
                self.spot_inner,
                self.spot_outer,
                self.intensity,
                color,
            ),
            "Environment" => Light::environment(self.intensity, self.env_texture_index),
            "AreaRect" => Light::area_rect(
                position,
                direction,
                self.area_width,
                self.area_height,
                self.intensity,
                color,
            ),
            "AreaDisk" => Light::area_disk(
                position,
                direction,
                self.area_width,  // Use width as radius
                self.intensity,
                color,
            ),
            "AreaSphere" => Light::area_sphere(
                position,
                self.area_width,  // Use width as radius
                self.intensity,
                color,
            ),
            _ => return Err(PyValueError::new_err(format!("Unknown light type: {}", self.light_type))),
        };

        Ok(light)
    }
}

/// Python wrapper for Material shading parameters (P2)
#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "MaterialShading")]
#[derive(Clone)]
pub struct PyMaterialShading {
    #[pyo3(get, set)]
    pub brdf: String,
    #[pyo3(get, set)]
    pub metallic: f32,
    #[pyo3(get, set)]
    pub roughness: f32,
    #[pyo3(get, set)]
    pub sheen: f32,
    #[pyo3(get, set)]
    pub clearcoat: f32,
    #[pyo3(get, set)]
    pub subsurface: f32,
    #[pyo3(get, set)]
    pub anisotropy: f32,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyMaterialShading {
    #[new]
    #[pyo3(signature = (brdf="CookTorranceGgx", roughness=0.5, metallic=0.0, sheen=0.0, clearcoat=0.0, subsurface=0.0, anisotropy=0.0))]
    pub fn new(
        brdf: &str,
        roughness: f32,
        metallic: f32,
        sheen: f32,
        clearcoat: f32,
        subsurface: f32,
        anisotropy: f32,
    ) -> PyResult<Self> {
        let mat = Self {
            brdf: brdf.to_string(),
            metallic,
            roughness,
            sheen,
            clearcoat,
            subsurface,
            anisotropy,
        };

        // Validate
        mat.to_native()?;

        Ok(mat)
    }

    /// Create a Lambert (pure diffuse) material
    #[staticmethod]
    pub fn lambert(roughness: f32) -> PyResult<Self> {
        Self::new("Lambert", roughness, 0.0, 0.0, 0.0, 0.0, 0.0)
    }

    /// Create a Phong material
    #[staticmethod]
    pub fn phong(roughness: f32, metallic: f32) -> PyResult<Self> {
        Self::new("Phong", roughness, metallic, 0.0, 0.0, 0.0, 0.0)
    }

    /// Create a Disney Principled material with full parameter control
    #[staticmethod]
    pub fn disney(
        roughness: f32,
        metallic: f32,
        sheen: f32,
        clearcoat: f32,
        subsurface: f32,
    ) -> PyResult<Self> {
        Self::new("DisneyPrincipled", roughness, metallic, sheen, clearcoat, subsurface, 0.0)
    }

    /// Create an anisotropic material
    #[staticmethod]
    pub fn anisotropic(brdf: &str, roughness: f32, anisotropy: f32) -> PyResult<Self> {
        Self::new(brdf, roughness, 0.0, 0.0, 0.0, 0.0, anisotropy)
    }
}

impl PyMaterialShading {
    /// Convert to native Rust type (not exposed to Python)
    pub fn to_native(&self) -> PyResult<MaterialShading> {
        let brdf_model = match self.brdf.as_str() {
            "Lambert" => BrdfModel::Lambert,
            "Phong" => BrdfModel::Phong,
            "BlinnPhong" => BrdfModel::BlinnPhong,
            "OrenNayar" => BrdfModel::OrenNayar,
            "CookTorranceGgx" => BrdfModel::CookTorranceGgx,
            "CookTorranceBeckmann" => BrdfModel::CookTorranceBeckmann,
            "DisneyPrincipled" => BrdfModel::DisneyPrincipled,
            "AshikhminShirley" => BrdfModel::AshikhminShirley,
            "Ward" => BrdfModel::Ward,
            "Toon" => BrdfModel::Toon,
            "Minnaert" => BrdfModel::Minnaert,
            "Subsurface" => BrdfModel::Subsurface,
            "Hair" => BrdfModel::Hair,
            _ => return Err(PyValueError::new_err(format!("Unknown BRDF model: {}", self.brdf))),
        };

        let mat = MaterialShading {
            brdf: brdf_model.as_u32(),
            metallic: self.metallic,
            roughness: self.roughness,
            sheen: self.sheen,
            clearcoat: self.clearcoat,
            subsurface: self.subsurface,
            anisotropy: self.anisotropy,
            _pad: 0.0,
        };

        mat.validate()
            .map_err(|e| PyValueError::new_err(e))?;

        Ok(mat)
    }
}

/// Python wrapper for Shadow settings
#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "ShadowSettings")]
#[derive(Clone)]
pub struct PyShadowSettings {
    #[pyo3(get, set)]
    pub technique: String,
    #[pyo3(get, set)]
    pub map_res: u32,
    #[pyo3(get, set)]
    pub bias: f32,
    #[pyo3(get, set)]
    pub normal_bias: f32,
    #[pyo3(get, set)]
    pub softness: f32,
    #[pyo3(get, set)]
    pub pcss_blocker_radius: f32,
    #[pyo3(get, set)]
    pub pcss_filter_radius: f32,
    #[pyo3(get, set)]
    pub light_size: f32,
    #[pyo3(get, set)]
    pub moment_bias: f32,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyShadowSettings {
    #[new]
    #[pyo3(signature = (
        technique="PCF",
        map_res=2048,
        bias=0.002,
        normal_bias=0.5,
        softness=1.25,
        pcss_blocker_radius=0.03,
        pcss_filter_radius=0.06,
        light_size=0.25,
        moment_bias=0.0005
    ))]
    pub fn new(
        technique: &str,
        map_res: u32,
        bias: f32,
        normal_bias: f32,
        softness: f32,
        pcss_blocker_radius: f32,
        pcss_filter_radius: f32,
        light_size: f32,
        moment_bias: f32,
    ) -> PyResult<Self> {
        let settings = Self {
            technique: technique.to_string(),
            map_res,
            bias,
            normal_bias,
            softness,
            pcss_blocker_radius,
            pcss_filter_radius,
            light_size,
            moment_bias,
        };

        // Validate
        settings.to_native()?;

        Ok(settings)
    }

    pub fn memory_mb(&self) -> PyResult<f64> {
        let settings = self.to_native()?;
        Ok(settings.memory_budget() as f64 / (1024.0 * 1024.0))
    }
}

impl PyShadowSettings {
    /// Convert to native Rust type (not exposed to Python)
    pub fn to_native(&self) -> PyResult<ShadowSettings> {
        let tech = match self.technique.as_str() {
            "Hard" => ShadowTechnique::Hard,
            "PCF" => ShadowTechnique::PCF,
            "PCSS" => ShadowTechnique::PCSS,
            "VSM" => ShadowTechnique::VSM,
            "EVSM" => ShadowTechnique::EVSM,
            "MSM" => ShadowTechnique::MSM,
            "CSM" => ShadowTechnique::CSM,
            _ => return Err(PyValueError::new_err(format!("Unknown shadow technique: {}", self.technique))),
        };

        let settings = ShadowSettings {
            tech: tech.as_u32(),
            map_res: self.map_res,
            bias: self.bias,
            normal_bias: self.normal_bias,
            softness: self.softness,
            pcss_blocker_radius: self.pcss_blocker_radius,
            pcss_filter_radius: self.pcss_filter_radius,
            light_size: self.light_size,
            moment_bias: self.moment_bias,
            _pad: [0.0; 3],
        };

        settings.validate()
            .map_err(|e| PyValueError::new_err(e))?;

        Ok(settings)
    }
}

/// Python wrapper for GI settings
#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "GiSettings")]
#[derive(Clone)]
pub struct PyGiSettings {
    #[pyo3(get, set)]
    pub technique: String,
    #[pyo3(get, set)]
    pub ibl_intensity: f32,
    #[pyo3(get, set)]
    pub ibl_rotation: f32,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyGiSettings {
    #[new]
    #[pyo3(signature = (technique="IBL", ibl_intensity=1.0, ibl_rotation=0.0))]
    pub fn new(technique: &str, ibl_intensity: f32, ibl_rotation: f32) -> PyResult<Self> {
        Ok(Self {
            technique: technique.to_string(),
            ibl_intensity,
            ibl_rotation,
        })
    }
}

impl PyGiSettings {
    /// Convert to native Rust type (not exposed to Python)
    pub fn to_native(&self) -> PyResult<GiSettings> {
        let tech = match self.technique.as_str() {
            "None" => GiTechnique::None,
            "IBL" => GiTechnique::Ibl,
            _ => return Err(PyValueError::new_err(format!("Unknown GI technique: {}", self.technique))),
        };

        Ok(GiSettings {
            tech: tech.as_u32(),
            ibl_intensity: self.ibl_intensity,
            ibl_rotation_deg: self.ibl_rotation,
            _pad: 0.0,
        })
    }
}

/// Python wrapper for Atmosphere settings
#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "Atmosphere")]
#[derive(Clone)]
pub struct PyAtmosphere {
    #[pyo3(get, set)]
    pub fog_density: f32,
    #[pyo3(get, set)]
    pub exposure: f32,
    #[pyo3(get, set)]
    pub sky_model: String,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyAtmosphere {
    #[new]
    #[pyo3(signature = (fog_density=0.0, exposure=1.0, sky_model="Off"))]
    pub fn new(fog_density: f32, exposure: f32, sky_model: &str) -> PyResult<Self> {
        Ok(Self {
            fog_density,
            exposure,
            sky_model: sky_model.to_string(),
        })
    }
}

impl PyAtmosphere {
    /// Convert to native Rust type (not exposed to Python)
    pub fn to_native(&self) -> PyResult<Atmosphere> {
        let sky = match self.sky_model.as_str() {
            "Off" => 0u32,
            "Preetham" => 1u32,
            _ => return Err(PyValueError::new_err(format!("Unknown sky model: {}", self.sky_model))),
        };

        Ok(Atmosphere {
            fog_density: self.fog_density,
            exposure: self.exposure,
            sky_model: sky,
            _pad: 0.0,
        })
    }
}

// ============================================================================
// P5: Screen-space effects Python bindings
// ============================================================================

/// Python wrapper for SSAO/GTAO settings (P5)
#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "SSAOSettings")]
#[derive(Clone)]
pub struct PySSAOSettings {
    #[pyo3(get, set)]
    pub radius: f32,
    #[pyo3(get, set)]
    pub intensity: f32,
    #[pyo3(get, set)]
    pub bias: f32,
    #[pyo3(get, set)]
    pub sample_count: u32,
    #[pyo3(get, set)]
    pub spiral_turns: f32,
    #[pyo3(get, set)]
    pub technique: String,  // "SSAO" or "GTAO"
    #[pyo3(get, set)]
    pub blur_radius: u32,
    #[pyo3(get, set)]
    pub temporal_alpha: f32,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PySSAOSettings {
    #[new]
    #[pyo3(signature = (radius=0.5, intensity=1.0, bias=0.025, sample_count=16, spiral_turns=7.0, technique="SSAO", blur_radius=2, temporal_alpha=0.0))]
    pub fn new(
        radius: f32,
        intensity: f32,
        bias: f32,
        sample_count: u32,
        spiral_turns: f32,
        technique: &str,
        blur_radius: u32,
        temporal_alpha: f32,
    ) -> PyResult<Self> {
        Ok(Self {
            radius,
            intensity,
            bias,
            sample_count,
            spiral_turns,
            technique: technique.to_string(),
            blur_radius,
            temporal_alpha,
        })
    }

    #[staticmethod]
    pub fn ssao(radius: f32, intensity: f32) -> PyResult<Self> {
        Ok(Self {
            radius,
            intensity,
            bias: 0.025,
            sample_count: 16,
            spiral_turns: 7.0,
            technique: "SSAO".to_string(),
            blur_radius: 2,
            temporal_alpha: 0.0,
        })
    }

    #[staticmethod]
    pub fn gtao(radius: f32, intensity: f32) -> PyResult<Self> {
        Ok(Self {
            radius,
            intensity,
            bias: 0.025,
            sample_count: 16,
            spiral_turns: 7.0,
            technique: "GTAO".to_string(),
            blur_radius: 2,
            temporal_alpha: 0.0,
        })
    }
}

impl PySSAOSettings {
    /// Convert to native Rust type (not exposed to Python)
    pub fn to_native(&self) -> PyResult<SSAOSettings> {
        let technique = match self.technique.as_str() {
            "SSAO" => 0u32,
            "GTAO" => 1u32,
            _ => return Err(PyValueError::new_err(format!("Unknown SSAO technique: {}", self.technique))),
        };

        let settings = SSAOSettings {
            radius: self.radius,
            intensity: self.intensity,
            bias: self.bias,
            sample_count: self.sample_count,
            spiral_turns: self.spiral_turns,
            technique,
            blur_radius: self.blur_radius,
            temporal_alpha: self.temporal_alpha,
        };

        settings.validate()
            .map_err(|e| PyValueError::new_err(e))?;

        Ok(settings)
    }
}

/// Python wrapper for SSGI settings (P5)
#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "SSGISettings")]
#[derive(Clone)]
pub struct PySSGISettings {
    #[pyo3(get, set)]
    pub ray_steps: u32,
    #[pyo3(get, set)]
    pub ray_radius: f32,
    #[pyo3(get, set)]
    pub ray_thickness: f32,
    #[pyo3(get, set)]
    pub intensity: f32,
    #[pyo3(get, set)]
    pub temporal_alpha: f32,
    #[pyo3(get, set)]
    pub use_half_res: bool,
    #[pyo3(get, set)]
    pub ibl_fallback: f32,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PySSGISettings {
    #[new]
    #[pyo3(signature = (ray_steps=24, ray_radius=5.0, ray_thickness=0.5, intensity=1.0, temporal_alpha=0.0, use_half_res=true, ibl_fallback=0.3))]
    pub fn new(
        ray_steps: u32,
        ray_radius: f32,
        ray_thickness: f32,
        intensity: f32,
        temporal_alpha: f32,
        use_half_res: bool,
        ibl_fallback: f32,
    ) -> PyResult<Self> {
        Ok(Self {
            ray_steps,
            ray_radius,
            ray_thickness,
            intensity,
            temporal_alpha,
            use_half_res,
            ibl_fallback,
        })
    }
}

impl PySSGISettings {
    /// Convert to native Rust type (not exposed to Python)
    pub fn to_native(&self) -> PyResult<SSGISettings> {
        let settings = SSGISettings {
            ray_steps: self.ray_steps,
            ray_radius: self.ray_radius,
            ray_thickness: self.ray_thickness,
            intensity: self.intensity,
            temporal_alpha: self.temporal_alpha,
            use_half_res: if self.use_half_res { 1u32 } else { 0u32 },
            ibl_fallback: self.ibl_fallback,
            _pad: 0.0,
        };

        settings.validate()
            .map_err(|e| PyValueError::new_err(e))?;

        Ok(settings)
    }
}

/// Python wrapper for SSR settings (P5)
#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "SSRSettings")]
#[derive(Clone)]
pub struct PySSRSettings {
    #[pyo3(get, set)]
    pub max_steps: u32,
    #[pyo3(get, set)]
    pub max_distance: f32,
    #[pyo3(get, set)]
    pub thickness: f32,
    #[pyo3(get, set)]
    pub stride: f32,
    #[pyo3(get, set)]
    pub intensity: f32,
    #[pyo3(get, set)]
    pub roughness_fade: f32,
    #[pyo3(get, set)]
    pub edge_fade: f32,
    #[pyo3(get, set)]
    pub temporal_alpha: f32,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PySSRSettings {
    #[new]
    #[pyo3(signature = (max_steps=48, max_distance=50.0, thickness=0.5, stride=1.0, intensity=1.0, roughness_fade=0.8, edge_fade=0.1, temporal_alpha=0.0))]
    pub fn new(
        max_steps: u32,
        max_distance: f32,
        thickness: f32,
        stride: f32,
        intensity: f32,
        roughness_fade: f32,
        edge_fade: f32,
        temporal_alpha: f32,
    ) -> PyResult<Self> {
        Ok(Self {
            max_steps,
            max_distance,
            thickness,
            stride,
            intensity,
            roughness_fade,
            edge_fade,
            temporal_alpha,
        })
    }
}

impl PySSRSettings {
    /// Convert to native Rust type (not exposed to Python)
    pub fn to_native(&self) -> PyResult<SSRSettings> {
        let settings = SSRSettings {
            max_steps: self.max_steps,
            max_distance: self.max_distance,
            thickness: self.thickness,
            stride: self.stride,
            intensity: self.intensity,
            roughness_fade: self.roughness_fade,
            edge_fade: self.edge_fade,
            temporal_alpha: self.temporal_alpha,
        };

        settings.validate()
            .map_err(|e| PyValueError::new_err(e))?;

        Ok(settings)
    }
}

// ============================================================================
// P6: Atmospherics & sky Python bindings
// ============================================================================

/// Python wrapper for Sky settings (P6)
#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "SkySettings")]
#[derive(Clone)]
pub struct PySkySettings {
    #[pyo3(get, set)]
    pub sun_direction: [f32; 3],
    #[pyo3(get, set)]
    pub turbidity: f32,
    #[pyo3(get, set)]
    pub ground_albedo: f32,
    #[pyo3(get, set)]
    pub model: String,  // "off", "preetham", or "hosek-wilkie"
    #[pyo3(get, set)]
    pub sun_intensity: f32,
    #[pyo3(get, set)]
    pub exposure: f32,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PySkySettings {
    #[new]
    #[pyo3(signature = (sun_direction=[0.3, 0.8, 0.5], turbidity=2.5, ground_albedo=0.2, model="hosek-wilkie", sun_intensity=20.0, exposure=1.0))]
    pub fn new(
        sun_direction: [f32; 3],
        turbidity: f32,
        ground_albedo: f32,
        model: &str,
        sun_intensity: f32,
        exposure: f32,
    ) -> PyResult<Self> {
        Ok(Self {
            sun_direction,
            turbidity,
            ground_albedo,
            model: model.to_string(),
            sun_intensity,
            exposure,
        })
    }

    #[staticmethod]
    pub fn preetham(turbidity: f32, ground_albedo: f32) -> PyResult<Self> {
        Ok(Self {
            sun_direction: [0.3, 0.8, 0.5],
            turbidity,
            ground_albedo,
            model: "preetham".to_string(),
            sun_intensity: 20.0,
            exposure: 1.0,
        })
    }

    #[staticmethod]
    pub fn hosek_wilkie(turbidity: f32, ground_albedo: f32) -> PyResult<Self> {
        Ok(Self {
            sun_direction: [0.3, 0.8, 0.5],
            turbidity,
            ground_albedo,
            model: "hosek-wilkie".to_string(),
            sun_intensity: 20.0,
            exposure: 1.0,
        })
    }

    pub fn with_sun_angles(&mut self, azimuth_deg: f32, elevation_deg: f32) {
        let az_rad = azimuth_deg.to_radians();
        let el_rad = elevation_deg.to_radians();

        self.sun_direction = [
            el_rad.cos() * az_rad.sin(),
            el_rad.sin(),
            el_rad.cos() * az_rad.cos(),
        ];
    }
}

impl PySkySettings {
    /// Convert to native Rust type (not exposed to Python)
    pub fn to_native(&self) -> PyResult<SkySettings> {
        let model = match self.model.to_lowercase().as_str() {
            "off" => 0u32,
            "preetham" => 1u32,
            "hosek-wilkie" | "hosek_wilkie" | "hosekwilkie" => 2u32,
            _ => return Err(PyValueError::new_err(format!("Unknown sky model: {}", self.model))),
        };

        let settings = SkySettings {
            sun_direction: self.sun_direction,
            turbidity: self.turbidity,
            ground_albedo: self.ground_albedo,
            model,
            sun_intensity: self.sun_intensity,
            exposure: self.exposure,
            _pad: [0.0; 4],
        };

        settings.validate()
            .map_err(|e| PyValueError::new_err(e))?;

        Ok(settings)
    }
}

/// Python wrapper for Volumetric settings (P6)
#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "VolumetricSettings")]
#[derive(Clone)]
pub struct PyVolumetricSettings {
    #[pyo3(get, set)]
    pub density: f32,
    #[pyo3(get, set)]
    pub height_falloff: f32,
    #[pyo3(get, set)]
    pub phase_g: f32,
    #[pyo3(get, set)]
    pub max_steps: u32,
    #[pyo3(get, set)]
    pub start_distance: f32,
    #[pyo3(get, set)]
    pub max_distance: f32,
    #[pyo3(get, set)]
    pub absorption: f32,
    #[pyo3(get, set)]
    pub sun_intensity: f32,
    #[pyo3(get, set)]
    pub scattering_color: [f32; 3],
    #[pyo3(get, set)]
    pub temporal_alpha: f32,
    #[pyo3(get, set)]
    pub ambient_color: [f32; 3],
    #[pyo3(get, set)]
    pub use_shadows: bool,
    #[pyo3(get, set)]
    pub jitter_strength: f32,
    #[pyo3(get, set)]
    pub phase_function: String,  // "isotropic" or "hg"
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyVolumetricSettings {
    #[new]
    #[pyo3(signature = (
        density=0.015,
        height_falloff=0.1,
        phase_g=0.7,
        max_steps=48,
        start_distance=0.1,
        max_distance=100.0,
        absorption=0.5,
        sun_intensity=1.0,
        scattering_color=[1.0, 1.0, 1.0],
        temporal_alpha=0.0,
        ambient_color=[0.3, 0.4, 0.5],
        use_shadows=true,
        jitter_strength=0.5,
        phase_function="hg"
    ))]
    pub fn new(
        density: f32,
        height_falloff: f32,
        phase_g: f32,
        max_steps: u32,
        start_distance: f32,
        max_distance: f32,
        absorption: f32,
        sun_intensity: f32,
        scattering_color: [f32; 3],
        temporal_alpha: f32,
        ambient_color: [f32; 3],
        use_shadows: bool,
        jitter_strength: f32,
        phase_function: &str,
    ) -> PyResult<Self> {
        Ok(Self {
            density,
            height_falloff,
            phase_g,
            max_steps,
            start_distance,
            max_distance,
            absorption,
            sun_intensity,
            scattering_color,
            temporal_alpha,
            ambient_color,
            use_shadows,
            jitter_strength,
            phase_function: phase_function.to_string(),
        })
    }

    #[staticmethod]
    pub fn with_god_rays(density: f32, phase_g: f32) -> PyResult<Self> {
        Ok(Self {
            density,
            phase_g,
            height_falloff: 0.1,
            max_steps: 48,
            start_distance: 0.1,
            max_distance: 100.0,
            absorption: 0.5,
            sun_intensity: 1.0,
            scattering_color: [1.0, 1.0, 1.0],
            temporal_alpha: 0.0,
            ambient_color: [0.3, 0.4, 0.5],
            use_shadows: true,
            jitter_strength: 0.5,
            phase_function: "hg".to_string(),
        })
    }

    #[staticmethod]
    pub fn uniform_fog(density: f32) -> PyResult<Self> {
        Ok(Self {
            density,
            phase_g: 0.0,
            height_falloff: 0.0,
            max_steps: 32,
            start_distance: 0.1,
            max_distance: 100.0,
            absorption: 0.5,
            sun_intensity: 1.0,
            scattering_color: [1.0, 1.0, 1.0],
            temporal_alpha: 0.0,
            ambient_color: [0.3, 0.4, 0.5],
            use_shadows: false,
            jitter_strength: 0.5,
            phase_function: "isotropic".to_string(),
        })
    }
}

impl PyVolumetricSettings {
    /// Convert to native Rust type (not exposed to Python)
    pub fn to_native(&self) -> PyResult<VolumetricSettings> {
        let phase_function = match self.phase_function.to_lowercase().as_str() {
            "isotropic" | "iso" => 0u32,
            "hg" | "henyey-greenstein" | "henyey_greenstein" => 1u32,
            _ => return Err(PyValueError::new_err(format!("Unknown phase function: {}", self.phase_function))),
        };

        let settings = VolumetricSettings {
            density: self.density,
            height_falloff: self.height_falloff,
            phase_g: self.phase_g,
            max_steps: self.max_steps,
            start_distance: self.start_distance,
            max_distance: self.max_distance,
            absorption: self.absorption,
            sun_intensity: self.sun_intensity,
            scattering_color: self.scattering_color,
            temporal_alpha: self.temporal_alpha,
            ambient_color: self.ambient_color,
            use_shadows: if self.use_shadows { 1u32 } else { 0u32 },
            jitter_strength: self.jitter_strength,
            phase_function,
            _pad: [0.0; 2],
        };

        settings.validate()
            .map_err(|e| PyValueError::new_err(e))?;

        Ok(settings)
    }
}
