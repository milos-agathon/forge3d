use crate::lighting::types::{SSAOSettings, SSGISettings, SSRSettings};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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
    pub technique: String,
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
    pub fn to_native(&self) -> PyResult<SSAOSettings> {
        let technique = match self.technique.as_str() {
            "SSAO" => 0u32,
            "GTAO" => 1u32,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown SSAO technique: {}",
                    self.technique
                )))
            }
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

        settings.validate().map_err(PyValueError::new_err)?;
        Ok(settings)
    }
}

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

        settings.validate().map_err(PyValueError::new_err)?;
        Ok(settings)
    }
}

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

        settings.validate().map_err(PyValueError::new_err)?;
        Ok(settings)
    }
}
