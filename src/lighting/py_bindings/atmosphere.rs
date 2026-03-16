use crate::lighting::types::Atmosphere;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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
    pub fn to_native(&self) -> PyResult<Atmosphere> {
        let sky = match self.sky_model.as_str() {
            "Off" => 0u32,
            "Preetham" => 1u32,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown sky model: {}",
                    self.sky_model
                )))
            }
        };

        Ok(Atmosphere {
            fog_density: self.fog_density,
            exposure: self.exposure,
            sky_model: sky,
            _pad: 0.0,
        })
    }
}
