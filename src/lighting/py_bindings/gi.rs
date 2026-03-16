use crate::lighting::types::{GiSettings, GiTechnique};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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
    pub fn to_native(&self) -> PyResult<GiSettings> {
        let tech = match self.technique.as_str() {
            "None" => GiTechnique::None,
            "IBL" => GiTechnique::Ibl,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown GI technique: {}",
                    self.technique
                )))
            }
        };

        Ok(GiSettings {
            tech: tech.as_u32(),
            ibl_intensity: self.ibl_intensity,
            ibl_rotation_deg: self.ibl_rotation,
            _pad: 0.0,
        })
    }
}
