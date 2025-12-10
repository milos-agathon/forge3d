use super::structs::PointSpotLightRenderer;
use super::types::{Light, ShadowQuality, DebugMode};
use glam;

impl PointSpotLightRenderer {
    /// Add a light and return its ID
    pub fn add_light(&mut self, light: Light) -> u32 {
        if self.lights.len() >= self.max_lights {
            return u32::MAX; // No more space
        }

        let light_id = self.light_id_counter;
        self.light_id_counter += 1;

        let index = self.lights.len();
        self.lights.push(light);
        self.light_id_map.insert(light_id, index);

        self.uniforms.active_light_count = self.lights.len() as u32;

        light_id
    }

    /// Remove a light by ID
    pub fn remove_light(&mut self, light_id: u32) -> bool {
        if let Some(index) = self.light_id_map.remove(&light_id) {
            self.lights.remove(index);

            // Update indices in the map
            for (_, idx) in self.light_id_map.iter_mut() {
                if *idx > index {
                    *idx -= 1;
                }
            }

            self.uniforms.active_light_count = self.lights.len() as u32;
            true
        } else {
            false
        }
    }

    /// Get a mutable reference to a light by ID
    pub fn get_light_mut(&mut self, light_id: u32) -> Option<&mut Light> {
        if let Some(index) = self.light_id_map.get(&light_id) {
            self.lights.get_mut(*index)
        } else {
            None
        }
    }

    /// Get a reference to a light by ID
    pub fn get_light(&self, light_id: u32) -> Option<&Light> {
        if let Some(index) = self.light_id_map.get(&light_id) {
            self.lights.get(*index)
        } else {
            None
        }
    }

    /// Clear all lights
    pub fn clear_lights(&mut self) {
        self.lights.clear();
        self.light_id_map.clear();
        self.uniforms.active_light_count = 0;
    }

    /// Get number of active lights
    pub fn light_count(&self) -> usize {
        self.lights.len()
    }

    /// Set camera matrices
    pub fn set_camera(&mut self, view: glam::Mat4, proj: glam::Mat4) {
        self.uniforms.view_matrix = view.to_cols_array_2d();
        self.uniforms.proj_matrix = proj.to_cols_array_2d();
    }

    /// Set ambient lighting
    pub fn set_ambient(&mut self, color: [f32; 3], intensity: f32) {
        self.uniforms.ambient_color = color;
        self.uniforms.ambient_intensity = intensity;
    }

    /// Set shadow quality
    pub fn set_shadow_quality(&mut self, quality: ShadowQuality) {
        self.uniforms.shadow_quality = quality as u32;
    }

    /// Set debug mode
    pub fn set_debug_mode(&mut self, mode: DebugMode) {
        self.uniforms.debug_mode = mode as u32;
    }

    /// Set shadow parameters
    pub fn set_shadow_parameters(&mut self, bias: f32, normal_bias: f32, softness: f32) {
        self.uniforms.shadow_bias = bias;
        self.uniforms.shadow_normal_bias = normal_bias;
        self.uniforms.shadow_softness = softness;
    }

    /// Get current uniforms for inspection
    pub fn uniforms(&self) -> &super::types::PointSpotLightUniforms {
        &self.uniforms
    }

    /// Get all lights
    pub fn lights(&self) -> &[Light] {
        &self.lights
    }
}
