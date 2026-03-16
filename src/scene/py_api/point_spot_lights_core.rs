use super::*;

#[cfg(feature = "extension-module")]
#[pymethods]
impl Scene {
    // B13: Point & Spot Lights (Realtime) API
    #[pyo3(text_signature = "($self, max_lights=32)")]
    pub fn enable_point_spot_lights(&mut self, max_lights: Option<usize>) -> PyResult<()> {
        let g = crate::core::gpu::ctx();
        let max_lights = max_lights.unwrap_or(32);
        let renderer =
            crate::core::point_spot_lights::PointSpotLightRenderer::new(&g.device, max_lights);

        self.point_spot_lights_renderer = Some(renderer);
        self.point_spot_lights_enabled = true;

        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn disable_point_spot_lights(&mut self) {
        self.point_spot_lights_enabled = false;
        self.point_spot_lights_renderer = None;
    }

    #[pyo3(text_signature = "($self)")]
    pub fn is_point_spot_lights_enabled(&self) -> bool {
        self.point_spot_lights_enabled && self.point_spot_lights_renderer.is_some()
    }

    #[pyo3(text_signature = "($self, x, y, z, r, g, b, intensity, range)")]
    pub fn add_point_light(
        &mut self,
        x: f32,
        y: f32,
        z: f32,
        r: f32,
        g: f32,
        b: f32,
        intensity: f32,
        range: f32,
    ) -> PyResult<u32> {
        if let Some(ref mut renderer) = self.point_spot_lights_renderer {
            let light = crate::core::point_spot_lights::Light::point(
                [x, y, z],
                [r, g, b],
                intensity,
                range,
            );
            let light_id = renderer.add_light(light);

            if light_id == u32::MAX {
                Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "Maximum number of lights exceeded",
                ))
            } else {
                Ok(light_id)
            }
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Point/spot lights not enabled. Call enable_point_spot_lights() first.",
            ))
        }
    }

    #[pyo3(
        text_signature = "($self, x, y, z, dir_x, dir_y, dir_z, r, g, b, intensity, range, inner_cone_deg, outer_cone_deg, penumbra_softness)"
    )]
    pub fn add_spot_light(
        &mut self,
        x: f32,
        y: f32,
        z: f32,
        dir_x: f32,
        dir_y: f32,
        dir_z: f32,
        r: f32,
        g: f32,
        b: f32,
        intensity: f32,
        range: f32,
        inner_cone_deg: f32,
        outer_cone_deg: f32,
        penumbra_softness: f32,
    ) -> PyResult<u32> {
        if let Some(ref mut renderer) = self.point_spot_lights_renderer {
            let light = crate::core::point_spot_lights::Light::spot(
                [x, y, z],
                [dir_x, dir_y, dir_z],
                [r, g, b],
                intensity,
                range,
                inner_cone_deg,
                outer_cone_deg,
                penumbra_softness,
            );
            let light_id = renderer.add_light(light);

            if light_id == u32::MAX {
                Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "Maximum number of lights exceeded",
                ))
            } else {
                Ok(light_id)
            }
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Point/spot lights not enabled. Call enable_point_spot_lights() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, preset, x, y, z)")]
    pub fn add_light_preset(&mut self, preset: &str, x: f32, y: f32, z: f32) -> PyResult<u32> {
        if let Some(ref mut renderer) = self.point_spot_lights_renderer {
            let preset_enum = match preset {
                "room_light" => crate::core::point_spot_lights::LightPreset::RoomLight,
                "desk_lamp" => crate::core::point_spot_lights::LightPreset::DeskLamp,
                "street_light" => crate::core::point_spot_lights::LightPreset::StreetLight,
                "spotlight" => crate::core::point_spot_lights::LightPreset::Spotlight,
                "headlight" => crate::core::point_spot_lights::LightPreset::Headlight,
                "flashlight" => crate::core::point_spot_lights::LightPreset::Flashlight,
                "candle" => crate::core::point_spot_lights::LightPreset::Candle,
                "warm_lamp" => crate::core::point_spot_lights::LightPreset::WarmLamp,
                _ => return Err(pyo3::exceptions::PyValueError::new_err(
                    "Preset must be one of: 'room_light', 'desk_lamp', 'street_light', 'spotlight', 'headlight', 'flashlight', 'candle', 'warm_lamp'"
                )),
            };

            let light = preset_enum.to_light([x, y, z]);
            let light_id = renderer.add_light(light);

            if light_id == u32::MAX {
                Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "Maximum number of lights exceeded",
                ))
            } else {
                Ok(light_id)
            }
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Point/spot lights not enabled. Call enable_point_spot_lights() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self, light_id)")]
    pub fn remove_light(&mut self, light_id: u32) -> PyResult<bool> {
        if let Some(ref mut renderer) = self.point_spot_lights_renderer {
            Ok(renderer.remove_light(light_id))
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Point/spot lights not enabled. Call enable_point_spot_lights() first.",
            ))
        }
    }

    #[pyo3(text_signature = "($self)")]
    pub fn clear_all_lights(&mut self) -> PyResult<()> {
        if let Some(ref mut renderer) = self.point_spot_lights_renderer {
            renderer.clear_lights();
            Ok(())
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Point/spot lights not enabled. Call enable_point_spot_lights() first.",
            ))
        }
    }
}
