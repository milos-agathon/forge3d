use super::*;

pub(in crate::terrain::renderer) struct ShadowSetup {
    pub(in crate::terrain::renderer) eye: glam::Vec3,
    pub(in crate::terrain::renderer) view_matrix: glam::Mat4,
    pub(in crate::terrain::renderer) proj_matrix: glam::Mat4,
    pub(in crate::terrain::renderer) height_exag: f32,
    pub(in crate::terrain::renderer) height_min: f32,
    pub(in crate::terrain::renderer) shadow_bind_group: Option<wgpu::BindGroup>,
}

impl TerrainScene {
    pub(in crate::terrain::renderer) fn prepare_shadow_setup(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        params: &crate::terrain::render_params::TerrainRenderParams,
        decoded: &crate::terrain::render_params::DecodedTerrainSettings,
        heightmap_view: &wgpu::TextureView,
    ) -> Result<ShadowSetup> {
        let phi_rad = params.cam_phi_deg.to_radians();
        let theta_rad = params.cam_theta_deg.to_radians();
        let eye_x = params.cam_target[0] + params.cam_radius * theta_rad.sin() * phi_rad.cos();
        let eye_y = params.cam_target[1] + params.cam_radius * theta_rad.cos();
        let eye_z = params.cam_target[2] + params.cam_radius * theta_rad.sin() * phi_rad.sin();
        let eye = glam::Vec3::new(eye_x, eye_y, eye_z);
        let target = glam::Vec3::from_array(params.cam_target);
        let up = glam::Vec3::Y;
        let view_matrix = glam::Mat4::look_at_rh(eye, target, up);
        let aspect = params.size_px.0 as f32 / params.size_px.1 as f32;
        let proj_matrix = glam::Mat4::perspective_rh(
            params.fov_y_deg.to_radians(),
            aspect,
            params.clip.0,
            params.clip.1,
        );

        let sun_direction = glam::Vec3::new(
            -decoded.light.direction[0],
            -decoded.light.direction[1],
            -decoded.light.direction[2],
        );
        let terrain_spacing = params.terrain_span.max(1e-3);
        let height_exag = params.z_scale;
        let height_min = decoded.clamp.height_range.0;
        let height_max = decoded.clamp.height_range.1;

        let shadow_settings = &decoded.shadow;
        self.shadow_pcss_radius = shadow_settings.pcss_light_radius.max(0.0);
        let cascade_count = shadow_settings
            .cascades
            .max(1)
            .min(self.csm_renderer.shadow_map_views.len() as u32);
        let shadow_far = params
            .clip
            .1
            .min(shadow_settings.max_distance)
            .max(params.clip.0);

        let mut cascade_splits: Vec<f32> = Vec::with_capacity(cascade_count as usize + 1);
        cascade_splits.push(params.clip.0);
        for split in TERRAIN_DEFAULT_CASCADE_SPLITS
            .iter()
            .take(cascade_count.saturating_sub(1) as usize)
        {
            let clamped = (*split).min(shadow_far);
            if clamped > *cascade_splits.last().unwrap_or(&params.clip.0) {
                cascade_splits.push(clamped);
            }
        }
        while cascade_splits.len() < cascade_count as usize {
            let last = *cascade_splits.last().unwrap_or(&params.clip.0);
            let remaining = cascade_count as usize + 1 - cascade_splits.len();
            let step = (shadow_far - last) / remaining.max(1) as f32;
            cascade_splits.push((last + step).min(shadow_far));
        }
        if cascade_splits.len() == cascade_count as usize {
            cascade_splits.push(shadow_far);
        } else {
            *cascade_splits.last_mut().unwrap() = shadow_far;
        }

        self.csm_renderer.config.cascade_count = cascade_count;
        self.csm_renderer.config.cascade_splits = cascade_splits.clone();
        self.csm_renderer.config.shadow_map_size = shadow_settings.resolution;
        self.csm_renderer.config.max_shadow_distance = shadow_far;
        self.csm_renderer.config.depth_bias = shadow_settings.depth_bias;
        self.csm_renderer.config.slope_bias = shadow_settings.slope_scale_bias;
        self.csm_renderer.config.peter_panning_offset = shadow_settings.normal_bias;
        self.csm_renderer.config.pcf_kernel_size =
            if self.shadow_pcss_radius > 0.0 { 3 } else { 1 };

        use crate::lighting::types::ShadowTechnique;
        let technique_enum = match shadow_settings.technique.to_uppercase().as_str() {
            "HARD" => ShadowTechnique::Hard,
            "PCF" => ShadowTechnique::PCF,
            "PCSS" => ShadowTechnique::PCSS,
            "VSM" => ShadowTechnique::VSM,
            "EVSM" => ShadowTechnique::EVSM,
            "MSM" => ShadowTechnique::MSM,
            _ => {
                log::warn!(
                    target: "terrain.shadow",
                    "Unknown shadow technique '{}', defaulting to PCF",
                    shadow_settings.technique
                );
                ShadowTechnique::PCF
            }
        };
        self.csm_renderer.uniforms.technique = technique_enum.as_u32();
        self.shadow_technique = technique_enum.as_u32();

        let requires_moments = matches!(
            technique_enum,
            ShadowTechnique::VSM | ShadowTechnique::EVSM | ShadowTechnique::MSM
        );
        if requires_moments && self.moment_pass.is_none() {
            self.moment_pass = Some(crate::shadows::MomentGenerationPass::new(&self.device));
            log::info!(
                target: "terrain.shadow",
                "Created moment generation pass for technique: {:?}",
                technique_enum
            );
        } else if !requires_moments && self.moment_pass.is_some() {
            self.moment_pass = None;
            log::info!(target: "terrain.shadow", "Removed moment generation pass");
        }

        self.csm_renderer.config.pcf_kernel_size = match technique_enum {
            ShadowTechnique::Hard => 1,
            ShadowTechnique::PCSS => 5,
            _ => 3,
        };
        self.csm_renderer.uniforms.technique_params = [
            shadow_settings.softness * 10.0,
            shadow_settings.softness * 20.0,
            0.0005,
            shadow_settings.pcss_light_radius.max(0.5),
        ];

        log::info!(
            target: "terrain.shadow",
            "Shadow CLI params: enabled={}, technique={} (id={}), cascades={}, resolution={}, max_dist={:.0}, pcss_radius={:.4}",
            shadow_settings.enabled, shadow_settings.technique, technique_enum.as_u32(), shadow_settings.cascades,
            shadow_settings.resolution, shadow_settings.max_distance, self.shadow_pcss_radius
        );
        log::info!(
            target: "terrain.shadow",
            "Shadow bias: depth={:.6}, slope={:.6}, normal={:.6}, softness={:.4}, splits={:?}",
            shadow_settings.depth_bias, shadow_settings.slope_scale_bias,
            shadow_settings.normal_bias, shadow_settings.softness, cascade_splits
        );

        let height_curve = [
            match params.height_curve_mode.as_str() {
                "linear" => 0.0,
                "pow" => 1.0,
                "smoothstep" => 2.0,
                "lut" => 3.0,
                _ => 0.0,
            },
            params.height_curve_strength.clamp(0.0, 1.0),
            params.height_curve_power.max(0.01),
            0.0,
        ];

        let shadow_bind_group = if shadow_settings.enabled {
            let bind_group = self.render_shadow_depth_passes(
                encoder,
                heightmap_view,
                terrain_spacing,
                height_exag,
                height_min,
                height_max,
                view_matrix,
                proj_matrix,
                sun_direction,
                params.clip.0,
                shadow_far,
                height_curve,
            );

            if let Some(ref mut moment_pass) = self.moment_pass {
                if let Some(moment_texture) = &self.csm_renderer.evsm_maps {
                    let depth_view = self.csm_renderer.shadow_texture_view();
                    let moment_view = crate::shadows::create_moment_storage_view(
                        moment_texture,
                        self.csm_renderer.config.cascade_count,
                    );
                    moment_pass.prepare_bind_group(&self.device, &depth_view, &moment_view);
                    moment_pass.execute(
                        &self.queue,
                        encoder,
                        ShadowTechnique::from_u32(self.shadow_technique),
                        self.csm_renderer.config.cascade_count,
                        self.csm_renderer.config.shadow_map_size,
                        self.csm_renderer.uniforms.evsm_positive_exp,
                        self.csm_renderer.uniforms.evsm_negative_exp,
                    );
                    log::debug!(
                        target: "terrain.shadow",
                        "Executed moment generation pass for technique {} with {} cascades",
                        self.shadow_technique,
                        self.csm_renderer.config.cascade_count
                    );
                }
            }

            Some(bind_group)
        } else {
            None
        };

        Ok(ShadowSetup {
            eye,
            view_matrix,
            proj_matrix,
            height_exag,
            height_min,
            shadow_bind_group,
        })
    }
}
