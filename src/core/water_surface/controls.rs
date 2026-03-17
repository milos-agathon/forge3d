use super::*;
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, Extent3d, ImageCopyTexture, ImageDataLayout, Origin3d,
    TextureDescriptor, TextureDimension, TextureUsages, TextureViewDescriptor,
};

impl WaterSurfaceRenderer {
    pub fn update_params(&mut self, params: WaterSurfaceParams) {
        self.params = params;
        self.update_uniforms();
    }

    pub fn set_mode(&mut self, mode: WaterSurfaceMode) {
        self.params.mode = mode;
        self.enabled = mode != WaterSurfaceMode::Disabled;
        self.update_uniforms();
    }

    pub fn set_height(&mut self, height: f32) {
        self.params.height = height;
        self.update_uniforms();
    }

    pub fn set_size(&mut self, size: f32) {
        self.params.size = size;
        self.update_uniforms();
    }

    pub fn set_base_color(&mut self, color: Vec3) {
        self.params.base_color = color;
        self.update_uniforms();
    }

    pub fn set_hue_shift(&mut self, hue_shift: f32) {
        self.params.hue_shift = hue_shift;
        self.update_uniforms();
    }

    pub fn set_tint(&mut self, tint_color: Vec3, tint_strength: f32) {
        self.params.tint_color = tint_color;
        self.params.tint_strength = tint_strength;
        self.update_uniforms();
    }

    pub fn set_alpha(&mut self, alpha: f32) {
        self.params.alpha = alpha.clamp(0.0, 1.0);
        self.update_uniforms();
    }

    pub fn set_wave_params(&mut self, amplitude: f32, frequency: f32, speed: f32) {
        self.params.wave_amplitude = amplitude;
        self.params.wave_frequency = frequency;
        self.params.wave_speed = speed;
        self.update_uniforms();
    }

    pub fn set_flow_direction(&mut self, direction: Vec2) {
        self.params.flow_direction = direction.normalize();
        self.update_uniforms();
    }

    pub fn set_lighting_params(
        &mut self,
        reflection_strength: f32,
        refraction_strength: f32,
        fresnel_power: f32,
        roughness: f32,
    ) {
        self.params.reflection_strength = reflection_strength;
        self.params.refraction_strength = refraction_strength;
        self.params.fresnel_power = fresnel_power;
        self.params.roughness = roughness;
        self.update_uniforms();
    }

    pub fn set_foam_params(&mut self, width_px: f32, intensity: f32, noise_scale: f32) {
        self.params.foam_width_px = width_px.max(0.0);
        self.params.foam_intensity = intensity.clamp(0.0, 1.0);
        self.params.foam_noise_scale = noise_scale.max(1.0);
        self.update_uniforms();
    }

    pub fn set_foam_enabled(&mut self, enabled: bool) {
        self.params.foam_enabled = enabled;
        self.update_uniforms();
    }

    pub fn set_debug_mode(&mut self, mode: u32) {
        self.params.debug_mode = mode;
        self.update_uniforms();
    }

    pub fn upload_water_mask(
        &mut self,
        device: &Device,
        queue: &Queue,
        data: &[u8],
        width: u32,
        height: u32,
    ) {
        assert_eq!(
            data.len() as u32,
            width * height,
            "mask data must be width*height bytes"
        );
        self.mask_size = (width, height);
        self.mask_texture = device.create_texture(&TextureDescriptor {
            label: Some("water_mask_texture"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.mask_view = self
            .mask_texture
            .create_view(&TextureViewDescriptor::default());
        self.mask_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("water_surface_mask_bind_group"),
            layout: &self.mask_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.mask_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.mask_sampler),
                },
            ],
        });
        queue.write_texture(
            ImageCopyTexture {
                texture: &self.mask_texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(width),
                rows_per_image: Some(height),
            },
            Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        self.uniforms.foam_params[3] = 1.0;
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if !enabled {
            self.params.mode = WaterSurfaceMode::Disabled;
        } else if self.params.mode == WaterSurfaceMode::Disabled {
            self.params.mode = WaterSurfaceMode::Transparent;
        }
        self.update_uniforms();
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled && self.params.mode != WaterSurfaceMode::Disabled
    }

    pub fn set_camera(&mut self, view_proj: Mat4) {
        self.uniforms.view_proj = view_proj.to_cols_array_2d();
    }

    pub fn update(&mut self, delta_time: f32) {
        self.animation_time += delta_time;
        self.uniforms.wave_params[3] = self.animation_time;
    }
}
