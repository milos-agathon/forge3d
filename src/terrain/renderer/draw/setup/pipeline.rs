use super::*;
use crate::terrain::renderer::core::TERRAIN_DEPTH_FORMAT;

pub(in crate::terrain::renderer) struct RenderTargets {
    pub(in crate::terrain::renderer) internal_texture: wgpu::Texture,
    pub(in crate::terrain::renderer) internal_view: wgpu::TextureView,
    #[allow(dead_code)]
    pub(in crate::terrain::renderer) msaa_texture: Option<wgpu::Texture>,
    pub(in crate::terrain::renderer) msaa_view: Option<wgpu::TextureView>,
    #[allow(dead_code)]
    pub(in crate::terrain::renderer) depth_texture: wgpu::Texture,
    pub(in crate::terrain::renderer) depth_view: wgpu::TextureView,
    pub(in crate::terrain::renderer) out_width: u32,
    pub(in crate::terrain::renderer) out_height: u32,
    pub(in crate::terrain::renderer) internal_width: u32,
    pub(in crate::terrain::renderer) internal_height: u32,
    pub(in crate::terrain::renderer) needs_scaling: bool,
    pub(in crate::terrain::renderer) sample_count: u32,
}

impl TerrainScene {
    pub(in crate::terrain::renderer) fn prepare_frame_lighting(
        &self,
        decoded: &crate::terrain::render_params::DecodedTerrainSettings,
    ) -> Result<()> {
        let mut light_buffer_guard = self
            .light_buffer
            .lock()
            .map_err(|_| anyhow!("Light buffer mutex poisoned"))?;
        light_buffer_guard.next_frame();

        let lights = if decoded.light.intensity > 0.0 {
            vec![Light {
                kind: LightType::Directional.as_u32(),
                intensity: decoded.light.intensity,
                range: 0.0,
                env_texture_index: 0,
                color: decoded.light.color,
                _pad1: 0.0,
                pos_ws: [0.0; 3],
                _pad2: 0.0,
                dir_ws: decoded.light.direction,
                _pad3: 0.0,
                cone_cos: [1.0, 1.0],
                area_half: [0.0, 0.0],
            }]
        } else {
            vec![Light {
                kind: LightType::Directional.as_u32(),
                intensity: 0.0,
                range: 0.0,
                env_texture_index: 0,
                color: [1.0, 1.0, 1.0],
                _pad1: 0.0,
                pos_ws: [0.0; 3],
                _pad2: 0.0,
                dir_ws: [0.0, 1.0, 0.0],
                _pad3: 0.0,
                cone_cos: [1.0, 1.0],
                area_half: [0.0, 0.0],
            }]
        };

        light_buffer_guard
            .update(self.device.as_ref(), self.queue.as_ref(), &lights)
            .map_err(|e| anyhow!("Failed to update light buffer: {}", e))?;
        Ok(())
    }

    pub(in crate::terrain::renderer) fn ensure_pipeline_sample_count(
        &self,
        effective_msaa: u32,
    ) -> Result<()> {
        let mut pipeline_cache = self
            .pipeline
            .lock()
            .map_err(|_| anyhow!("TerrainRenderer pipeline mutex poisoned"))?;
        if pipeline_cache.sample_count != effective_msaa {
            let light_buffer = self
                .light_buffer
                .lock()
                .map_err(|_| anyhow!("Light buffer mutex poisoned"))?;
            pipeline_cache.pipeline = Self::create_render_pipeline(
                self.device.as_ref(),
                &self.bind_group_layout,
                light_buffer.bind_group_layout(),
                &self.ibl_bind_group_layout,
                &self.shadow_bind_group_layout,
                &self.fog_bind_group_layout,
                &self.water_reflection_bind_group_layout,
                &self.material_layer_bind_group_layout,
                self.color_format,
                effective_msaa,
            );
            pipeline_cache.sample_count = effective_msaa;
        }
        Ok(())
    }

    pub(in crate::terrain::renderer) fn create_render_targets(
        &self,
        params: &crate::terrain::render_params::TerrainRenderParams,
        requested_msaa: u32,
        effective_msaa: u32,
    ) -> Result<RenderTargets> {
        let (out_width, out_height) = params.size_px;
        let render_scale = params.render_scale.clamp(0.25, 4.0);
        let internal_width = ((out_width as f32 * render_scale).round().max(1.0)) as u32;
        let internal_height = ((out_height as f32 * render_scale).round().max(1.0)) as u32;
        let needs_scaling = internal_width != out_width || internal_height != out_height;

        let internal_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain.internal.render_target"),
            size: wgpu::Extent3d {
                width: internal_width,
                height: internal_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.color_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let internal_view = internal_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let msaa_texture = if effective_msaa > 1 {
            Some(self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("terrain.msaa.render_target"),
                size: wgpu::Extent3d {
                    width: internal_width,
                    height: internal_height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: effective_msaa,
                dimension: wgpu::TextureDimension::D2,
                format: self.color_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            }))
        } else {
            None
        };
        let msaa_view = msaa_texture
            .as_ref()
            .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()));

        let depth_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain.depth.render_target"),
            size: wgpu::Extent3d {
                width: internal_width,
                height: internal_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: effective_msaa,
            dimension: wgpu::TextureDimension::D2,
            format: TERRAIN_DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let color_attachment_sample_count = if effective_msaa > 1 {
            effective_msaa
        } else {
            1
        };
        let resolve_sample_count = if effective_msaa > 1 { Some(1) } else { None };

        log_msaa_debug(
            &self.adapter,
            self.color_format,
            Some(TERRAIN_DEPTH_FORMAT),
            requested_msaa,
            effective_msaa,
            color_attachment_sample_count,
            resolve_sample_count,
            Some(effective_msaa),
            effective_msaa,
        );

        let invariants = MsaaInvariants {
            effective_msaa,
            pipeline_sample_count: effective_msaa,
            color_attachment_sample_count,
            has_resolve_target: effective_msaa > 1,
            resolve_sample_count,
            depth_sample_count: Some(effective_msaa),
            readback_sample_count: 1,
        };
        assert_msaa_invariants(&invariants, self.color_format)?;

        Ok(RenderTargets {
            internal_texture,
            internal_view,
            msaa_texture,
            msaa_view,
            depth_texture,
            depth_view,
            out_width,
            out_height,
            internal_width,
            internal_height,
            needs_scaling,
            sample_count: effective_msaa,
        })
    }
}
