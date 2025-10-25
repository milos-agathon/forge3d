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
    pub range: f32,
    #[pyo3(get, set)]
    pub spot_inner: f32,
    #[pyo3(get, set)]
    pub spot_outer: f32,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyLight {
    #[new]
    #[pyo3(signature = (light_type="Directional", intensity=3.0, color=None, azimuth=135.0, elevation=35.0, position=None, range=2000.0, spot_inner=20.0, spot_outer=35.0))]
    pub fn new(
        light_type: &str,
        intensity: f32,
        color: Option<Vec<f32>>,
        azimuth: f32,
        elevation: f32,
        position: Option<Vec<f32>>,
        range: f32,
        spot_inner: f32,
        spot_outer: f32,
    ) -> PyResult<Self> {
        let color = color.unwrap_or(vec![1.0, 1.0, 1.0]);
        let position = position.unwrap_or(vec![0.0, 0.0, 0.0]);

        if color.len() != 3 {
            return Err(PyValueError::new_err("color must have 3 elements (RGB)"));
        }
        if position.len() != 3 {
            return Err(PyValueError::new_err("position must have 3 elements (XYZ)"));
        }

        Ok(Self {
            light_type: light_type.to_string(),
            intensity,
            color,
            azimuth,
            elevation,
            position,
            range,
            spot_inner,
            spot_outer,
        })
    }
}

impl PyLight {
    /// Convert to native Rust type (not exposed to Python)
    pub fn to_native(&self) -> PyResult<Light> {
        let color = [self.color[0], self.color[1], self.color[2]];
        let position = [self.position[0], self.position[1], self.position[2]];

        let light = match self.light_type.as_str() {
            "Directional" => Light::directional(self.azimuth, self.elevation, self.intensity, color),
            "Point" => Light::point(position, self.range, self.intensity, color),
            "Spot" => Light::spot(
                position,
                [0.0, -1.0, 0.0], // Default downward direction
                self.range,
                self.spot_inner,
                self.spot_outer,
                self.intensity,
                color,
            ),
            "Environment" => Light::environment(self.intensity),
            _ => return Err(PyValueError::new_err(format!("Unknown light type: {}", self.light_type))),
        };

        Ok(light)
    }
}

/// Python wrapper for Material shading parameters
#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "MaterialShading")]
#[derive(Clone)]
pub struct PyMaterialShading {
    #[pyo3(get, set)]
    pub brdf: String,
    #[pyo3(get, set)]
    pub roughness: f32,
    #[pyo3(get, set)]
    pub metallic: f32,
    #[pyo3(get, set)]
    pub ior: f32,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyMaterialShading {
    #[new]
    #[pyo3(signature = (brdf="CookTorranceGgx", roughness=0.5, metallic=0.0, ior=1.5))]
    pub fn new(brdf: &str, roughness: f32, metallic: f32, ior: f32) -> PyResult<Self> {
        let mat = Self {
            brdf: brdf.to_string(),
            roughness,
            metallic,
            ior,
        };

        // Validate
        mat.to_native()?;

        Ok(mat)
    }
}

impl PyMaterialShading {
    /// Convert to native Rust type (not exposed to Python)
    pub fn to_native(&self) -> PyResult<MaterialShading> {
        let brdf_model = match self.brdf.as_str() {
            "Lambert" => BrdfModel::Lambert,
            "CookTorranceGgx" => BrdfModel::CookTorranceGgx,
            _ => return Err(PyValueError::new_err(format!("Unknown BRDF model: {}", self.brdf))),
        };

        let mat = MaterialShading {
            brdf: brdf_model.as_u32(),
            roughness: self.roughness,
            metallic: self.metallic,
            ior: self.ior,
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
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyShadowSettings {
    #[new]
    #[pyo3(signature = (technique="PCF", map_res=2048, bias=0.002, normal_bias=0.5, softness=1.25))]
    pub fn new(
        technique: &str,
        map_res: u32,
        bias: f32,
        normal_bias: f32,
        softness: f32,
    ) -> PyResult<Self> {
        let settings = Self {
            technique: technique.to_string(),
            map_res,
            bias,
            normal_bias,
            softness,
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
            _ => return Err(PyValueError::new_err(format!("Unknown shadow technique: {}", self.technique))),
        };

        let settings = ShadowSettings {
            tech: tech.as_u32(),
            map_res: self.map_res,
            bias: self.bias,
            normal_bias: self.normal_bias,
            softness: self.softness,
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
