use super::pipelines::{
    create_compute_pipeline, create_layouts, create_pipelines, sampled_texture_entry,
    sampler_entry, storage_texture_entry, uniform_entry, SsaoLayouts, SsaoPipelines,
};
use super::resources::{
    create_shared_buffers, create_textures, SsaoSharedBuffers, SsaoTextures, SsaoViewportTextures,
};
use super::*;
use crate::core::resource_tracker::{TrackedBuffer, TrackedTexture};

pub struct SsaoRenderer {
    pub(crate) settings: SsaoSettings,
    pub(super) settings_buffer: TrackedBuffer,
    pub(super) camera_buffer: TrackedBuffer,
    pub(super) ssao_pipeline: ComputePipeline,
    pub(super) gtao_pipeline: ComputePipeline,
    pub(super) concordia: bool,
    pub(super) contact_geometry: Option<(ComputePipeline, BindGroupLayout, TextureView)>,
    pub(super) ssao_bind_group_layout: BindGroupLayout,
    pub(super) blur_h_pipeline: ComputePipeline,
    pub(super) blur_v_pipeline: ComputePipeline,
    pub(super) blur_bind_group_layout: BindGroupLayout,
    pub(super) blur_settings: TrackedBuffer,
    pub(super) temporal_pipeline: ComputePipeline,
    pub(super) temporal_bind_group_layout: BindGroupLayout,
    pub(crate) temporal_params_buffer: TrackedBuffer,
    pub(super) composite_pipeline: ComputePipeline,
    pub(super) composite_bind_group_layout: BindGroupLayout,
    pub(super) comp_uniform: TrackedBuffer,
    pub(super) _noise_texture: TrackedTexture,
    pub(super) noise_view: TextureView,
    pub(super) noise_sampler: Sampler,
    pub(super) ssao_texture: TrackedTexture,
    pub(super) ssao_view: TextureView,
    pub(super) ssao_blurred: TrackedTexture,
    pub(super) ssao_blurred_view: TextureView,
    pub(super) ssao_history: TrackedTexture,
    pub(super) ssao_history_view: TextureView,
    pub(super) ssao_resolved: TrackedTexture,
    pub(super) ssao_resolved_view: TextureView,
    pub(super) _ssao_tmp: TrackedTexture,
    pub(super) ssao_tmp_view: TextureView,
    pub(super) ssao_composited: TrackedTexture,
    pub(super) ssao_composited_view: TextureView,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) frame_index: u32,
    pub(crate) temporal_enabled: bool,
    pub(crate) temporal_alpha: f32,
    pub(super) blur_enabled: bool,
    pub(super) history_valid: bool,
    pub(super) last_ao_ms: f32,
    pub(super) last_blur_ms: f32,
    pub(super) last_temporal_ms: f32,
}

impl SsaoRenderer {
    pub fn new(
        device: &Device,
        width: u32,
        height: u32,
        _material_format: TextureFormat,
    ) -> RenderResult<Self> {
        let mut settings = SsaoSettings::default();
        settings.inv_resolution = [1.0 / width as f32, 1.0 / height as f32];
        settings.proj_scale = 0.5 * height as f32;

        let SsaoSharedBuffers {
            settings_buffer,
            camera_buffer,
            blur_settings,
            temporal_params_buffer,
            comp_uniform,
        } = create_shared_buffers(device, &settings)?;
        let SsaoLayouts {
            ssao_bind_group_layout,
            blur_bind_group_layout,
            temporal_bind_group_layout,
            composite_bind_group_layout,
        } = create_layouts(device);
        let SsaoPipelines {
            ssao_pipeline,
            gtao_pipeline,
            blur_h_pipeline,
            blur_v_pipeline,
            temporal_pipeline,
            composite_pipeline,
        } = create_pipelines(
            device,
            &ssao_bind_group_layout,
            &blur_bind_group_layout,
            &temporal_bind_group_layout,
            &composite_bind_group_layout,
        );
        let SsaoTextures {
            noise_texture,
            noise_view,
            noise_sampler,
            viewport:
                SsaoViewportTextures {
                    ssao_texture,
                    ssao_view,
                    ssao_blurred,
                    ssao_blurred_view,
                    ssao_history,
                    ssao_history_view,
                    ssao_resolved,
                    ssao_resolved_view,
                    ssao_tmp,
                    ssao_tmp_view,
                    ssao_composited,
                    ssao_composited_view,
                },
        } = create_textures(device, width, height)?;

        Ok(Self {
            settings,
            contact_geometry: None,
            concordia: false,
            settings_buffer,
            camera_buffer,
            ssao_pipeline,
            gtao_pipeline,
            ssao_bind_group_layout,
            blur_h_pipeline,
            blur_v_pipeline,
            blur_bind_group_layout,
            blur_settings,
            temporal_pipeline,
            temporal_bind_group_layout,
            temporal_params_buffer,
            composite_pipeline,
            composite_bind_group_layout,
            comp_uniform,
            _noise_texture: noise_texture,
            noise_view,
            noise_sampler,
            ssao_texture,
            ssao_view,
            ssao_blurred,
            ssao_blurred_view,
            ssao_history,
            ssao_history_view,
            ssao_resolved,
            ssao_resolved_view,
            _ssao_tmp: ssao_tmp,
            ssao_tmp_view,
            ssao_composited,
            ssao_composited_view,
            width,
            height,
            frame_index: 0,
            temporal_enabled: true,
            temporal_alpha: 0.2,
            blur_enabled: true,
            history_valid: false,
            last_ao_ms: 0.0,
            last_blur_ms: 0.0,
            last_temporal_ms: 0.0,
        })
    }
}

impl SsaoRenderer {
    /// Restrict the existing GTAO kernel's terrain receivers to 3-D contacts.
    /// Height AO separately accounts for terrain, so the bounded resolve can
    /// combine disjoint geometry contributions without counting terrain twice.
    pub(crate) fn shared_contacts_source() -> RenderResult<String> {
        let mut source = Self::concordia_gtao_source()?;
        let before = "@group(0) @binding(3) var noise_texture: texture_2d<f32>;";
        if source.matches(before).count() != 1 {
            return Err(crate::core::error::RenderError::Render(
                "GTAO contact-kind declaration changed".into(),
            ));
        }
        source = source.replace(
            before,
            "@group(0) @binding(3) var occluder_kind: texture_2d<u32>;",
        );
        let before = "if (sample_depth <= 0.0) { continue; }";
        if source.matches(before).count() != 1 {
            return Err(crate::core::error::RenderError::Render(
                "GTAO contact sample anchor changed".into(),
            ));
        }
        Ok(source.replace(before, "if (sample_depth <= 0.0) { continue; }\nif (textureLoad(occluder_kind,pixel,0).r == 1u && textureLoad(occluder_kind,sample_pixel,0).r == 1u) { continue; }"))
    }

    pub(crate) fn configure_contact_geometry(
        &mut self,
        device: &Device,
        kind: TextureView,
    ) -> RenderResult<()> {
        let source = Self::shared_contacts_source()?;
        let mut entries = vec![
            sampled_texture_entry(0),
            sampled_texture_entry(1),
            sampled_texture_entry(2),
            sampled_texture_entry(3),
            sampler_entry(4),
            storage_texture_entry(5, TextureFormat::R32Float),
            uniform_entry(6),
            uniform_entry(7),
        ];
        entries[3].ty = wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Uint,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        };
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("concordia.gtao_contacts.layout"),
            entries: &entries,
        });
        let pipeline = crate::core::shader_registry::with_error_scope(
            device,
            "concordia.gtao_contacts.pipeline",
            || {
                let module = crate::core::shader_registry::create_labeled_shader_module(
                    device,
                    "concordia.gtao_contacts.shader",
                    &source,
                );
                let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
                    label: Some("concordia.gtao_contacts.pipeline_layout"),
                    bind_group_layouts: &[&layout],
                    push_constant_ranges: &[],
                });
                create_compute_pipeline(
                    device,
                    "concordia.gtao_contacts.pipeline",
                    &pipeline_layout,
                    &module,
                    "cs_gtao",
                )
            },
        );
        self.contact_geometry = Some((pipeline, layout, kind));
        self.invalidate_history();
        Ok(())
    }
}

impl SsaoRenderer {
    pub(crate) fn concordia_gtao_source() -> RenderResult<String> {
        let mut source = GTAO_SHADER_SRC.replace("\r\n", "\n");
        let before = r###"        let angle = (f32(d) / f32(direction_count)) * PI + angle_offset;
"###;
        if source.matches(before).count() != 1 {
            return Err(crate::core::error::RenderError::Render(
                "CONCORDIA GTAO specialization anchor changed".into(),
            ));
        }
        source = source.replacen(
            before,
            r###"        let angle = (f32(d) / f32(direction_count)) * TWO_PI + angle_offset;
"###,
            1,
        );
        let before = r###"        var horizon_cos = -1.0;
"###;
        if source.matches(before).count() != 1 {
            return Err(crate::core::error::RenderError::Render(
                "CONCORDIA GTAO specialization anchor changed".into(),
            ));
        }
        source = source.replacen(
            before,
            r###"        var blocked_fraction = 0.0;
"###,
            1,
        );
        let before = r###"            let step_px = r_screen * (f32(s) / f32(steps_per_direction));
"###;
        if source.matches(before).count() != 1 {
            return Err(crate::core::error::RenderError::Render(
                "CONCORDIA GTAO specialization anchor changed".into(),
            ));
        }
        source = source.replacen(
            before,
            r###"            let fraction = f32(s) / f32(steps_per_direction);
            let step_px = r_screen * fraction * fraction;
"###,
            1,
        );
        let before = r###"            let sample_pos = reconstruct_view_pos_linear(sample_uv, sample_depth);
"###;
        if source.matches(before).count() != 1 {
            return Err(crate::core::error::RenderError::Render(
                "CONCORDIA GTAO specialization anchor changed".into(),
            ));
        }
        source = source.replacen(
            before,
            r###"            if (sample_depth <= 0.0) { continue; }
            // Device depth belongs to the sampled texel centre. Reconstructing
            // at the continuous search UV invents a horizon on a flat plane.
            let sample_center = (vec2<f32>(sample_pixel) + 0.5) / vec2<f32>(dims);
            let sample_pos = reconstruct_view_pos_linear(sample_center, sample_depth);
"###,
            1,
        );
        let before = r###"            let h_cos = dot(horizon_vec / max(horizon_len, 1e-4), view_normal);
"###;
        if source.matches(before).count() != 1 {
            return Err(crate::core::error::RenderError::Render(
                "CONCORDIA GTAO specialization anchor changed".into(),
            ));
        }
        source = source.replacen(
            before,
            r###"            // Use the existing world-space bias for GTAO as for SSAO.
            // It removes reconstruction self-occlusion on a planar receiver.
            let normal_distance = max(dot(horizon_vec, view_normal) - settings.bias, 0.0);
            let h_cos = normal_distance / max(horizon_len, 1e-4);
"###,
            1,
        );
        let before = r###"            horizon_cos = max(horizon_cos, h_cos * attenuation);
"###;
        if source.matches(before).count() != 1 {
            return Err(crate::core::error::RenderError::Render(
                "CONCORDIA GTAO specialization anchor changed".into(),
            ));
        }
        source = source.replacen(before, r###"            let positive_horizon = clamp(h_cos, 0.0, 1.0);
            blocked_fraction = max(blocked_fraction, positive_horizon * positive_horizon * attenuation);
"###, 1);
        let before = r###"        let horizon_angle = acos(clamp(horizon_cos, -1.0, 1.0));
        visibility += sin(horizon_angle) - horizon_angle * cos(horizon_angle) + 0.5 * PI;
"###;
        if source.matches(before).count() != 1 {
            return Err(crate::core::error::RenderError::Render(
                "CONCORDIA GTAO specialization anchor changed".into(),
            ));
        }
        source = source.replacen(
            before,
            r###"        // h = dot(normal, horizon direction) = sin(elevation above the
        // tangent plane). The cosine-weighted blocked hemisphere fraction
        // in this azimuth is h^2. A tangent horizon therefore leaves exactly
        // unit visibility; a horizon at the normal blocks the whole slice.
        visibility += 1.0 - clamp(blocked_fraction, 0.0, 1.0);
"###,
            1,
        );
        let before = r###"    visibility = visibility / (f32(direction_count) * PI);
    let ao_raw = 1.0 - clamp(visibility * settings.intensity, 0.0, 1.0);
"###;
        if source.matches(before).count() != 1 {
            return Err(crate::core::error::RenderError::Render(
                "CONCORDIA GTAO specialization anchor changed".into(),
            ));
        }
        source = source.replacen(
            before,
            r###"    visibility = visibility / f32(direction_count);
    let ao_raw = 1.0 - clamp((1.0 - visibility) * settings.intensity, 0.0, 1.0);
"###,
            1,
        );
        source = super::super::concordia_reconstruction(
            &source,
            "camera",
            "reconstruct_view_pos_linear",
        )?;
        source = source.replace(
            "let z_lin = -view_pos.z;",
            "let z_lin = select(-view_pos.z, 1.0, camera.proj_matrix[3][3] != 0.0);",
        );
        Ok(source)
    }

    pub(crate) fn configure_concordia_gtao(&mut self, device: &Device) -> RenderResult<()> {
        let filter_source =
            include_str!("../../../shaders/filters/bilateral_separable.wgsl").replace("\r\n", "\n");
        let anchor = "let result = sum / weight_sum;";
        if filter_source.matches(anchor).count() != 2 {
            return Err(crate::core::error::RenderError::Render(
                "CONCORDIA bilateral anchors changed".into(),
            ));
        }
        let filter_source =
            filter_source.replace(anchor, "let result = clamp(sum / weight_sum, 0.0, 1.0);");
        let filter = crate::core::shader_registry::create_labeled_shader_module(
            device,
            "concordia.ao_filter.shader",
            &filter_source,
        );
        let filter_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("concordia.ao_filter.layout"),
            bind_group_layouts: &[&self.blur_bind_group_layout],
            push_constant_ranges: &[],
        });
        self.blur_h_pipeline = create_compute_pipeline(
            device,
            "concordia.ao_filter.h",
            &filter_layout,
            &filter,
            "cs_blur_h",
        );
        self.blur_v_pipeline = create_compute_pipeline(
            device,
            "concordia.ao_filter.v",
            &filter_layout,
            &filter,
            "cs_blur_v",
        );
        self.concordia = true;
        let source = Self::concordia_gtao_source()?;
        self.gtao_pipeline = crate::core::shader_registry::with_error_scope(
            device,
            "concordia.gtao.pipeline",
            || {
                let module = crate::core::shader_registry::create_labeled_shader_module(
                    device,
                    "concordia.gtao.shader",
                    &source,
                );
                let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("concordia.gtao.layout"),
                    bind_group_layouts: &[&self.ssao_bind_group_layout],
                    push_constant_ranges: &[],
                });
                create_compute_pipeline(
                    device,
                    "concordia.gtao.pipeline",
                    &layout,
                    &module,
                    "cs_gtao",
                )
            },
        );
        Ok(())
    }
}
