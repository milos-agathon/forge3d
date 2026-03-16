use super::*;

#[cfg(feature = "extension-module")]
#[pymethods]
impl Scene {
    pub fn get_light_count(&self) -> PyResult<usize> {
        if let Some(ref renderer) = self.point_spot_lights_renderer {
            Ok(renderer.light_count())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Point/spot lights not enabled. Call enable_point_spot_lights() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, light_id, x, y, z)")]
    pub fn check_light_affects_point(
        &self,
        light_id: u32,
        x: f32,
        y: f32,
        z: f32,
    ) -> PyResult<bool> {
        if let Some(ref renderer) = self.point_spot_lights_renderer {
            if let Some(light) = renderer.get_light(light_id) {
                Ok(light.affects_point([x, y, z]))
            } else {
                Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Light with ID {} not found",
                    light_id
                )))
            }
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Point/spot lights not enabled. Call enable_point_spot_lights() first.",
            ))
        }
    }
}
