use super::*;

#[cfg(feature = "extension-module")]
#[pymethods]
impl Scene {
    // B10: Ground Plane (Raster) API
    #[pyo3(text_signature = "($self)")]
    pub fn enable_ground_plane(&mut self) -> PyResult<()> {
        let g = crate::core::gpu::ctx();
        let renderer = crate::core::ground_plane::GroundPlaneRenderer::new(
            &g.device,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            Some(wgpu::TextureFormat::Depth32Float),
            1, // sample_count
        );

        self.ground_plane_renderer = Some(renderer);
        self.ground_plane_enabled = true;

        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn disable_ground_plane(&mut self) {
        self.ground_plane_enabled = false;
        self.ground_plane_renderer = None;
    }

    #[pyo3(text_signature = "($self)")]
    pub fn is_ground_plane_enabled(&self) -> bool {
        self.ground_plane_enabled && self.ground_plane_renderer.is_some()
    }

    #[pyo3(text_signature = "($self, mode)")]
    pub fn set_ground_plane_mode(&mut self, mode: &str) -> PyResult<()> {
        if let Some(ref mut renderer) = self.ground_plane_renderer {
            let mode_enum = match mode {
                "disabled" => crate::core::ground_plane::GroundPlaneMode::Disabled,
                "solid" => crate::core::ground_plane::GroundPlaneMode::Solid,
                "grid" => crate::core::ground_plane::GroundPlaneMode::Grid,
                "checkerboard" => crate::core::ground_plane::GroundPlaneMode::CheckerBoard,
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Mode must be one of: 'disabled', 'solid', 'grid', 'checkerboard'",
                    ))
                }
            };
            renderer.set_mode(mode_enum);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Ground plane not enabled. Call enable_ground_plane() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, height)")]
    pub fn set_ground_plane_height(&mut self, height: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.ground_plane_renderer {
            renderer.set_height(height);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Ground plane not enabled. Call enable_ground_plane() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, size)")]
    pub fn set_ground_plane_size(&mut self, size: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.ground_plane_renderer {
            renderer.set_size(size);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Ground plane not enabled. Call enable_ground_plane() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, major_spacing, minor_spacing)")]
    pub fn set_ground_plane_grid_spacing(
        &mut self,
        major_spacing: f32,
        minor_spacing: f32,
    ) -> PyResult<()> {
        if let Some(ref mut renderer) = self.ground_plane_renderer {
            renderer.set_grid_spacing(major_spacing, minor_spacing);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Ground plane not enabled. Call enable_ground_plane() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, major_width, minor_width)")]
    pub fn set_ground_plane_grid_width(
        &mut self,
        major_width: f32,
        minor_width: f32,
    ) -> PyResult<()> {
        if let Some(ref mut renderer) = self.ground_plane_renderer {
            renderer.set_grid_width(major_width, minor_width);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Ground plane not enabled. Call enable_ground_plane() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, r, g, b, alpha)")]
    pub fn set_ground_plane_color(&mut self, r: f32, g: f32, b: f32, alpha: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.ground_plane_renderer {
            renderer.set_albedo(glam::Vec3::new(r, g, b), alpha);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Ground plane not enabled. Call enable_ground_plane() first.",
            ))
        }
    }

    #[pyo3(
        text_signature = "($self, major_r, major_g, major_b, major_alpha, minor_r, minor_g, minor_b, minor_alpha)"
    )]
    pub fn set_ground_plane_grid_colors(
        &mut self,
        major_r: f32,
        major_g: f32,
        major_b: f32,
        major_alpha: f32,
        minor_r: f32,
        minor_g: f32,
        minor_b: f32,
        minor_alpha: f32,
    ) -> PyResult<()> {
        if let Some(ref mut renderer) = self.ground_plane_renderer {
            let major_color = glam::Vec3::new(major_r, major_g, major_b);
            let minor_color = glam::Vec3::new(minor_r, minor_g, minor_b);
            renderer.set_grid_colors(major_color, major_alpha, minor_color, minor_alpha);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Ground plane not enabled. Call enable_ground_plane() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, z_bias)")]
    pub fn set_ground_plane_z_bias(&mut self, z_bias: f32) -> PyResult<()> {
        if let Some(ref mut renderer) = self.ground_plane_renderer {
            renderer.set_z_bias(z_bias);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Ground plane not enabled. Call enable_ground_plane() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, preset)")]
    pub fn set_ground_plane_preset(&mut self, preset: &str) -> PyResult<()> {
        if let Some(ref mut renderer) = self.ground_plane_renderer {
            let params = match preset {
                "engineering" => crate::core::ground_plane::GroundPlaneParams::engineering_grid(),
                "architectural" => {
                    crate::core::ground_plane::GroundPlaneParams::architectural_grid()
                }
                "simple" => crate::core::ground_plane::GroundPlaneParams::simple_ground(),
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Preset must be one of: 'engineering', 'architectural', 'simple'",
                    ))
                }
            };
            renderer.update_params(params);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Ground plane not enabled. Call enable_ground_plane() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self)")]
    pub fn get_ground_plane_params(&self) -> PyResult<(f32, f32, f32, f32)> {
        if let Some(ref renderer) = self.ground_plane_renderer {
            Ok(renderer.get_params())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Ground plane not enabled. Call enable_ground_plane() first.",
            ))
        }
    }
}
