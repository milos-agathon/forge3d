use super::*;

fn settings_key(params: &crate::terrain::render_params::TerrainRenderParams) -> String {
    let s = &params.decoded().screen_space;
    let horizon = &params.decoded().height_ao;
    let base = format!(
        "{:?}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}",
        params.size_px,
        params.camera_mode,
        s.ssao_enabled,
        s.ssao_radius.to_bits(),
        s.ssao_intensity.to_bits(),
        s.ssgi_enabled,
        s.ssgi_intensity.to_bits(),
        s.ssr_enabled,
        s.ssr_intensity.to_bits(),
        s.taa_enabled,
        s.temporal_alpha.to_bits(),
        params.terrain_span.to_bits()
    );
    format!(
        "{base}|{}|{}|{}|{}|{}|{}|{:?}|{:?}",
        params.decoded().volumetrics.enabled,
        params.z_scale.to_bits(),
        params.fov_y_deg.to_bits(),
        params.height_curve_strength.to_bits(),
        params.height_curve_power.to_bits(),
        params.height_curve_mode,
        params
            .height_curve_lut
            .as_ref()
            .map(|lut| lut.iter().map(|v| v.to_bits()).collect::<Vec<_>>()),
        (
            horizon.enabled,
            horizon.directions,
            horizon.steps,
            horizon.max_distance.to_bits(),
            horizon.strength.to_bits(),
            horizon.resolution_scale.to_bits(),
            horizon.blur
        )
    )
}

impl ConcordiaFrame {
    pub(super) fn matches(
        &self,
        params: &crate::terrain::render_params::TerrainRenderParams,
    ) -> bool {
        self.settings_key == settings_key(params)
    }

    pub(super) fn configure_clouds(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        settings: Option<&Bound<'_, PyAny>>,
        time_seconds: f32,
    ) -> Result<bool> {
        let Some(settings) = settings.filter(|settings| !settings.is_none()) else {
            return Ok(false);
        };
        let enabled = settings.getattr("enabled")?.extract::<bool>()?
            && settings.getattr("shadows_enabled")?.extract::<bool>()?;
        if !enabled {
            return Ok(false);
        }
        let value = |name: &str| -> Result<f32> {
            let value = settings.getattr(name)?.extract::<f32>()?;
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(anyhow!("clouds.{name} must be finite and in [0,1]"));
            }
            Ok(value)
        };
        if !time_seconds.is_finite() {
            return Err(anyhow!("time_seconds must be finite"));
        }
        use crate::core::cloud_shadows::CloudShadowQuality;
        let quality = match settings.getattr("quality")?.extract::<String>()?.as_str() {
            "low" => CloudShadowQuality::Low,
            "medium" => CloudShadowQuality::Medium,
            "high" => CloudShadowQuality::High,
            "ultra" => CloudShadowQuality::Ultra,
            _ => return Err(anyhow!("unsupported cloud quality")),
        };
        if self.cloud.quality != quality {
            self.cloud.resize(device, quality)?;
            self.cloud.create_bind_group(device);
            self.frame_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("concordia.frame.bind_group"),
                layout: &self.frame_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.frame_uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(self.cloud.shadow_view()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(self.cloud.shadow_sampler()),
                    },
                ],
            });
        }
        self.cloud.set_cloud_coverage(value("coverage")?);
        self.cloud.set_cloud_density(value("density")?);
        self.cloud.set_shadow_intensity(value("shadow_strength")?);
        let offset = |name: &str| -> Result<f32> {
            let value = if settings.hasattr(name)? {
                settings.getattr(name)?.extract::<f32>()?
            } else {
                0.0
            };
            if !value.is_finite() {
                return Err(anyhow!("clouds.{name} must be finite"));
            }
            Ok(value)
        };
        self.cloud.uniforms._padding = [offset("shadow_offset_x")?, offset("shadow_offset_y")?];
        self.cloud.uniforms.time = time_seconds;
        self.cloud.upload_uniforms(queue);
        Ok(true)
    }
}

pub(super) fn key(params: &crate::terrain::render_params::TerrainRenderParams) -> String {
    settings_key(params)
}
