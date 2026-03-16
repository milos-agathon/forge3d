use crate::lighting::types::VolumetricSettings;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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
    pub phase_function: String,
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
    pub fn to_native(&self) -> PyResult<VolumetricSettings> {
        let phase_function = match self.phase_function.to_lowercase().as_str() {
            "isotropic" | "iso" => 0u32,
            "hg" | "henyey-greenstein" | "henyey_greenstein" => 1u32,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown phase function: {}",
                    self.phase_function
                )))
            }
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

        settings.validate().map_err(PyValueError::new_err)?;
        Ok(settings)
    }
}
