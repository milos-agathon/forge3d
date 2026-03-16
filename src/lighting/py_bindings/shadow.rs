use crate::lighting::types::{ShadowSettings, ShadowTechnique};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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
        settings.to_native()?;
        Ok(settings)
    }

    pub fn memory_mb(&self) -> PyResult<f64> {
        let settings = self.to_native()?;
        Ok(settings.memory_budget() as f64 / (1024.0 * 1024.0))
    }
}

impl PyShadowSettings {
    pub fn to_native(&self) -> PyResult<ShadowSettings> {
        let tech = match self.technique.as_str() {
            "Hard" => ShadowTechnique::Hard,
            "PCF" => ShadowTechnique::PCF,
            "PCSS" => ShadowTechnique::PCSS,
            "VSM" => ShadowTechnique::VSM,
            "EVSM" => ShadowTechnique::EVSM,
            "MSM" => ShadowTechnique::MSM,
            "CSM" => ShadowTechnique::CSM,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown shadow technique: {}",
                    self.technique
                )))
            }
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

        settings.validate().map_err(PyValueError::new_err)?;
        Ok(settings)
    }
}
