use crate::lighting::types::SkySettings;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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
    pub model: String,
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
    pub fn to_native(&self) -> PyResult<SkySettings> {
        let model = match self.model.to_lowercase().as_str() {
            "off" => 0u32,
            "preetham" => 1u32,
            "hosek-wilkie" | "hosek_wilkie" | "hosekwilkie" => 2u32,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown sky model: {}",
                    self.model
                )))
            }
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

        settings.validate().map_err(PyValueError::new_err)?;
        Ok(settings)
    }
}
