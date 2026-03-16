use super::*;

#[cfg(feature = "extension-module")]
#[pymethods]
impl Scene {
    #[new]
    #[pyo3(text_signature = "(width, height, grid=128, colormap='viridis')")]
    pub fn new(
        width: u32,
        height: u32,
        grid: Option<u32>,
        colormap: Option<String>,
    ) -> PyResult<Self> {
        Self::new_impl(width, height, grid, colormap)
    }

    #[pyo3(text_signature = "($self, eye, target, up, fovy_deg, znear, zfar)")]
    pub fn set_camera_look_at(
        &mut self,
        eye: (f32, f32, f32),
        target: (f32, f32, f32),
        up: (f32, f32, f32),
        fovy_deg: f32,
        znear: f32,
        zfar: f32,
    ) -> PyResult<()> {
        use crate::camera;
        let aspect = self.width as f32 / self.height as f32;
        let eye_v = glam::Vec3::new(eye.0, eye.1, eye.2);
        let target_v = glam::Vec3::new(target.0, target.1, target.2);
        let up_v = glam::Vec3::new(up.0, up.1, up.2);
        camera::validate_camera_params(eye_v, target_v, up_v, fovy_deg, znear, zfar)?;
        self.scene.view = glam::Mat4::look_at_rh(eye_v, target_v, up_v);
        self.scene.proj = camera::perspective_wgpu(fovy_deg.to_radians(), aspect, znear, zfar);
        let uniforms = self
            .scene
            .globals
            .to_uniforms(self.scene.view, self.scene.proj);
        let g = crate::core::gpu::ctx();
        g.queue
            .write_buffer(&self.ubo, 0, bytemuck::bytes_of(&uniforms));
        self.last_uniforms = uniforms;
        // Update text3d renderer view/proj
        if let Some(ref mut tm) = self.text3d_renderer {
            tm.set_view_proj(self.scene.view, self.scene.proj);
            tm.upload_uniforms(&g.queue);
        }
        Ok(())
    }

    #[pyo3(text_signature = "($self, height_r32f)")]
    pub fn set_height_from_r32f(&mut self, height_r32f: &pyo3::types::PyAny) -> PyResult<()> {
        self.set_height_from_r32f_impl(height_r32f)
    }

    #[pyo3(text_signature = "()")]
    pub fn ssao_enabled(&self) -> bool {
        self.ssao_enabled
    }

    #[pyo3(text_signature = "(, enabled)")]
    pub fn set_ssao_enabled(&mut self, enabled: bool) -> PyResult<bool> {
        self.ssao_enabled = enabled;
        Ok(self.ssao_enabled)
    }

    #[pyo3(text_signature = "(, radius, intensity, bias=0.025)")]
    pub fn set_ssao_parameters(&mut self, radius: f32, intensity: f32, bias: f32) -> PyResult<()> {
        let g = crate::core::gpu::ctx();
        self.ssao.set_params(radius, intensity, bias, &g.queue);
        Ok(())
    }

    #[pyo3(text_signature = "()")]
    pub fn get_ssao_parameters(&self) -> (f32, f32, f32) {
        self.ssao.params()
    }
}
