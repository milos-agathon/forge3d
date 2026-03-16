use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[cfg(feature = "extension-module")]
#[pyclass(module = "forge3d._forge3d", name = "SunPosition")]
#[derive(Clone)]
pub struct PySunPosition {
    #[pyo3(get)]
    pub azimuth: f64,
    #[pyo3(get)]
    pub elevation: f64,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PySunPosition {
    fn __repr__(&self) -> String {
        format!(
            "SunPosition(azimuth={:.2}°, elevation={:.2}°)",
            self.azimuth, self.elevation
        )
    }

    fn to_direction(&self) -> (f32, f32, f32) {
        let az_rad = self.azimuth.to_radians();
        let el_rad = self.elevation.to_radians();
        let cos_el = el_rad.cos();
        let x = -az_rad.sin() * cos_el;
        let y = el_rad.sin();
        let z = -az_rad.cos() * cos_el;
        (x as f32, y as f32, z as f32)
    }

    fn is_daytime(&self) -> bool {
        self.elevation > 0.0
    }
}

#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (latitude, longitude, datetime_utc))]
pub fn sun_position(latitude: f64, longitude: f64, datetime_utc: &str) -> PyResult<PySunPosition> {
    let pos = super::super::ephemeris::sun_position_from_iso(latitude, longitude, datetime_utc)
        .map_err(PyValueError::new_err)?;

    Ok(PySunPosition {
        azimuth: pos.azimuth,
        elevation: pos.elevation,
    })
}

#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (latitude, longitude, year, month, day, hour, minute, second=0))]
pub fn sun_position_utc(
    latitude: f64,
    longitude: f64,
    year: i32,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: u32,
) -> PySunPosition {
    let pos = super::super::ephemeris::sun_position(
        latitude, longitude, year, month, day, hour, minute, second,
    );

    PySunPosition {
        azimuth: pos.azimuth,
        elevation: pos.elevation,
    }
}
