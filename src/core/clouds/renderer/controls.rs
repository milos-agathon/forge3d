use super::*;

impl CloudRenderer {
    pub fn update_params(&mut self, params: CloudParams) {
        self.params = params;
        self.update_uniforms();
    }

    pub fn set_quality(&mut self, quality: CloudQuality) {
        self.params.quality = quality;
        self.noise_texture = None;
        self.noise_view = None;
        self.noise_resolution = 0;
        self.bind_group_textures = None;
        self.update_uniforms();
    }

    pub fn set_density(&mut self, density: f32) {
        self.params.density = density.clamp(0.0, 2.0);
        self.update_uniforms();
    }

    pub fn set_coverage(&mut self, coverage: f32) {
        self.params.coverage = coverage.clamp(0.0, 1.0);
        self.update_uniforms();
    }

    pub fn set_scale(&mut self, scale: f32) {
        self.params.scale = scale.max(10.0);
        self.update_uniforms();
    }

    pub fn set_wind(&mut self, direction: Vec2, strength: f32) {
        self.params.wind_direction = direction.normalize_or_zero();
        self.params.wind_strength = strength.clamp(0.0, 2.0);
        self.update_uniforms();
    }

    pub fn set_animation_preset(&mut self, preset: CloudAnimationPreset) {
        self.params.animation_preset = preset;
        self.params.wind_strength = preset.wind_strength();
        self.update_uniforms();
    }

    pub fn set_render_mode(&mut self, mode: CloudRenderMode) {
        self.params.render_mode = mode;
        self.update_uniforms();
    }

    pub fn update(&mut self, delta_time: f32) {
        self.time += delta_time * self.params.animation_preset.animation_speed();
        self.uniforms.camera_pos[3] = self.time;
    }

    pub(super) fn update_uniforms(&mut self) {
        self.uniforms.cloud_params = [
            self.params.coverage,
            self.params.scale,
            self.params.height,
            self.params.fade_distance,
        ];
        self.uniforms.wind_params = [
            self.params.wind_direction.x,
            self.params.wind_direction.y,
            self.params.wind_strength,
            self.params.animation_preset.animation_speed(),
        ];
        self.uniforms.scattering_params = [
            self.params.scatter_strength,
            self.params.absorption,
            self.params.phase_g,
            self.params.ambient_strength,
        ];
        let step_size = match self.params.quality {
            CloudQuality::Low => 12.0,
            CloudQuality::Medium => 8.0,
            CloudQuality::High => 5.0,
            CloudQuality::Ultra => 3.5,
        };
        let render_mode_flag = match self.params.render_mode {
            CloudRenderMode::Billboard => 0.0,
            CloudRenderMode::Volumetric => 1.0,
            CloudRenderMode::Hybrid => 2.0,
        };
        self.uniforms.render_params = [
            self.params.quality.max_ray_steps() as f32,
            step_size,
            self.params.quality.billboard_threshold(),
            render_mode_flag,
        ];
        self.uniforms.sun_direction[3] = self.params.density;
        self.uniforms.sky_params[3] = self.params.sun_intensity;
    }

    pub fn upload_uniforms(&self, queue: &Queue) {
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.uniforms]),
        );
    }

    pub fn set_camera(&mut self, view_proj: Mat4, camera_pos: Vec3) {
        self.uniforms.view_proj = view_proj.to_cols_array_2d();
        self.uniforms.camera_pos[0] = camera_pos.x;
        self.uniforms.camera_pos[1] = camera_pos.y;
        self.uniforms.camera_pos[2] = camera_pos.z;
    }

    pub fn set_sky_params(&mut self, sky_color: Vec3, sun_direction: Vec3, sun_intensity: f32) {
        self.uniforms.sky_params = [sky_color.x, sky_color.y, sky_color.z, sun_intensity];
        self.uniforms.sun_direction = [
            sun_direction.x,
            sun_direction.y,
            sun_direction.z,
            self.params.density,
        ];
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn get_params(&self) -> (f32, f32, f32, f32) {
        (
            self.params.density,
            self.params.coverage,
            self.params.scale,
            self.params.wind_strength,
        )
    }
}
