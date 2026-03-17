use super::*;

impl WaterSurfaceRenderer {
    pub(super) fn update_uniforms(&mut self) {
        self.uniforms.surface_params = [
            self.params.size,
            self.params.height,
            if self.enabled && self.params.mode != WaterSurfaceMode::Disabled {
                1.0
            } else {
                0.0
            },
            self.params.alpha,
        ];
        self.uniforms.color_params = [
            self.params.base_color.x,
            self.params.base_color.y,
            self.params.base_color.z,
            self.params.hue_shift,
        ];
        self.uniforms.wave_params[0] = self.params.wave_amplitude;
        self.uniforms.wave_params[1] = self.params.wave_frequency;
        self.uniforms.wave_params[2] = self.params.wave_speed;
        self.uniforms.tint_params = [
            self.params.tint_color.x,
            self.params.tint_color.y,
            self.params.tint_color.z,
            self.params.tint_strength,
        ];
        self.uniforms.lighting_params = [
            self.params.reflection_strength,
            self.params.refraction_strength,
            self.params.fresnel_power,
            self.params.roughness,
        ];
        self.uniforms.animation_params = [
            self.params.ripple_scale,
            self.params.ripple_speed,
            self.params.flow_direction.x,
            self.params.flow_direction.y,
        ];
        self.uniforms.foam_params = [
            self.params.foam_width_px,
            if self.params.foam_enabled {
                self.params.foam_intensity
            } else {
                0.0
            },
            self.params.foam_noise_scale,
            self.uniforms.foam_params[3],
        ];
        self.uniforms.debug_params[0] = self.params.debug_mode as f32;
        self.uniforms.world_transform =
            Mat4::from_translation(Vec3::new(0.0, self.params.height, 0.0)).to_cols_array_2d();
    }

    pub fn upload_uniforms(&self, queue: &Queue) {
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.uniforms]),
        );
    }

    pub fn get_params(&self) -> (f32, f32, f32, f32) {
        (
            self.params.height,
            self.params.alpha,
            self.params.hue_shift,
            self.params.tint_strength,
        )
    }

    pub fn create_ocean_water() -> WaterSurfaceParams {
        WaterSurfaceParams {
            mode: WaterSurfaceMode::Animated,
            base_color: Vec3::new(0.05, 0.2, 0.4),
            tint_color: Vec3::new(0.0, 0.5, 0.8),
            tint_strength: 0.3,
            wave_amplitude: 0.3,
            wave_frequency: 1.5,
            wave_speed: 1.2,
            reflection_strength: 1.0,
            alpha: 0.8,
            ..Default::default()
        }
    }

    pub fn create_lake_water() -> WaterSurfaceParams {
        WaterSurfaceParams {
            mode: WaterSurfaceMode::Reflective,
            base_color: Vec3::new(0.1, 0.3, 0.5),
            tint_color: Vec3::new(0.2, 0.6, 0.4),
            tint_strength: 0.2,
            wave_amplitude: 0.05,
            wave_frequency: 3.0,
            wave_speed: 0.5,
            reflection_strength: 0.6,
            alpha: 0.7,
            ..Default::default()
        }
    }

    pub fn create_river_water() -> WaterSurfaceParams {
        WaterSurfaceParams {
            mode: WaterSurfaceMode::Animated,
            base_color: Vec3::new(0.2, 0.4, 0.3),
            tint_color: Vec3::new(0.3, 0.5, 0.3),
            tint_strength: 0.4,
            wave_amplitude: 0.02,
            wave_frequency: 8.0,
            wave_speed: 2.0,
            flow_direction: Vec2::new(1.0, 0.0),
            reflection_strength: 0.3,
            alpha: 0.6,
            ..Default::default()
        }
    }
}
