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
        eye: (f64, f64, f64),
        target: (f64, f64, f64),
        up: (f64, f64, f64),
        fovy_deg: f32,
        znear: f32,
        zfar: f32,
    ) -> PyResult<()> {
        use crate::camera;
        let aspect = self.width as f32 / self.height as f32;
        // MENSURA: world positions cross the PyO3 boundary in f64 and are
        // narrowed only relative to the camera anchor. The scene's geometry
        // is authored relative to the world origin, so the anchor offset of
        // that origin is folded into the view transform; with the anchor at
        // the origin (all coordinates < 1 km) this is bit-identical to the
        // legacy path. Projection stays f32.
        let eye_d = glam::DVec3::new(eye.0, eye.1, eye.2);
        let target_d = glam::DVec3::new(target.0, target.1, target.2);
        let up_v = glam::DVec3::new(up.0, up.1, up.2).as_vec3();
        self.camera_anchor.rebase_if_needed(eye_d);
        let eye_v = self.camera_anchor.to_render_vec3(eye_d);
        let target_v = self.camera_anchor.to_render_vec3(target_d);
        camera::validate_camera_params(eye_v, target_v, up_v, fovy_deg, znear, zfar)?;
        let world_origin_offset = self.camera_anchor.model_offset(glam::DVec3::ZERO);
        self.scene.view = glam::Mat4::look_at_rh(eye_v, target_v, up_v)
            * glam::Mat4::from_translation(world_origin_offset);
        self.scene.proj = camera::perspective_wgpu(fovy_deg.to_radians(), aspect, znear, zfar);
        let uniforms = self
            .scene
            .globals
            .to_uniforms(self.scene.view, self.scene.proj);
        let g = crate::core::gpu::try_ctx()?;
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
        let g = crate::core::gpu::try_ctx()?;
        self.ssao.set_params(radius, intensity, bias, &g.queue);
        Ok(())
    }

    #[pyo3(text_signature = "()")]
    pub fn get_ssao_parameters(&self) -> (f32, f32, f32) {
        self.ssao.params()
    }
}
