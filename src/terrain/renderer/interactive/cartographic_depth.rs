//! Copy unjittered visibility from the production geometry path. The same
//! G-buffer is then cleared and populated by the jittered lighting frame.
use super::super::draw::{PreparedMaterials, RenderTargets, UploadedHeightInputs};
use super::*;

pub(super) struct CartographyDepth {
    pub(super) linear: TrackedTexture,
    pub(super) linear_view: wgpu::TextureView,
    surface_kind: TrackedTexture,
    pub(super) surface_kind_view: Arc<wgpu::TextureView>,
    terrain_uniform: Arc<TrackedBuffer>,
    frame_uniform: TrackedBuffer,
    frame_bind_group: wgpu::BindGroup,
    #[cfg(feature = "enable-gpu-instancing")]
    scatter: crate::render::mesh_instanced::MeshInstancedRenderer,
}

impl CartographyDepth {
    pub(super) fn new(
        scene: &TerrainScene,
        params: &crate::terrain::render_params::TerrainRenderParams,
        frame: &ConcordiaFrame,
    ) -> Result<Self> {
        let device = scene.device.as_ref();
        let (width, height) = params.size_px;
        let linear = tracked_create_texture(
            device,
            &wgpu::TextureDescriptor {
                label: Some("concordia.cartography.linear_depth"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        )?;
        let linear_view = linear.create_view(&Default::default());
        let surface_kind = tracked_create_texture(
            device,
            &wgpu::TextureDescriptor {
                label: Some("concordia.cartography.surface_kind"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Uint,
                usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        )?;
        let surface_kind_view = Arc::new(surface_kind.create_view(&Default::default()));
        let (_, view, projection) = super::camera_matrices(params);
        let values =
            TerrainScene::build_uniforms_with_matrices(params, params.decoded(), view, projection);
        let terrain_uniform = Arc::new(tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("concordia.cartography.terrain_uniform"),
                contents: bytemuck::cast_slice(&values),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        )?);
        let frame_uniform = tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("concordia.cartography.frame_uniform"),
                contents: bytemuck::bytes_of(&FrameUniforms::zeroed()),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        )?;
        let frame_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("concordia.cartography.frame_group"),
            layout: &frame.frame_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frame_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(frame.cloud.shadow_view()),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(frame.cloud.shadow_sampler()),
                },
            ],
        });
        Ok(Self {
            linear,
            linear_view,
            surface_kind,
            surface_kind_view,
            terrain_uniform,
            frame_uniform,
            frame_bind_group,
            #[cfg(feature = "enable-gpu-instancing")]
            scatter: crate::render::mesh_instanced::MeshInstancedRenderer::new_concordia(
                device,
                &scene.shadow_bind_group_layout,
                &frame.frame_layout,
            )?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode(
        &mut self,
        scene: &mut TerrainScene,
        encoder: &mut wgpu::CommandEncoder,
        params: &crate::terrain::render_params::TerrainRenderParams,
        heights: &UploadedHeightInputs,
        materials: &PreparedMaterials,
        ibl: &wgpu::BindGroup,
        media_environment_radiance: [f32; 3],
        media_environment: Option<&crate::formats::hdr::HdrImage>,
        media_environment_intensity: f32,
        curve: &wgpu::TextureView,
        targets: &mut RenderTargets,
        shadow: &super::super::shadows::ShadowSetup,
        time_seconds: f32,
        material_vt_ready: bool,
        sun_visibility: bool,
        timing: &mut Option<crate::core::gpu_timing::GpuTimingManager>,
    ) -> Result<u32> {
        let (_, view, projection) = super::camera_matrices(params);
        let mut values =
            TerrainScene::build_uniforms_with_matrices(params, params.decoded(), view, projection);
        if let Some(blend) = scene.height_detail_blend_override {
            values[39] = -(blend.clamp(0.0, 1.0) + 1.0);
        }
        scene
            .queue
            .write_buffer(&self.terrain_uniform, 0, bytemuck::cast_slice(&values));
        let frame = targets
            .concordia
            .as_mut()
            .ok_or_else(|| anyhow!("CONCORDIA frame state is unavailable"))?;
        let current = frame.current_uniforms;
        let mut base = current;
        base.proj = projection.to_cols_array_2d();
        base.inv_proj = projection.inverse().to_cols_array_2d();
        base.geometry[2] = 0.0;
        base.geometry[3] = 0.0;
        scene
            .queue
            .write_buffer(&self.frame_uniform, 0, bytemuck::bytes_of(&base));
        frame.current_uniforms = base;
        std::mem::swap(&mut frame.frame_bind_group, &mut self.frame_bind_group);
        clear_shared_geometry(encoder, targets)?;

        // Separate existing per-draw uniforms keep queue writes for the later
        // jittered pass from changing this already encoded geometry draw.
        #[cfg(feature = "enable-gpu-instancing")]
        let scatter_cache = (scene.scatter_renderer_sample_count, scene.scatter_concordia);
        #[cfg(feature = "enable-gpu-instancing")]
        {
            std::mem::swap(&mut scene.scatter_renderer, &mut self.scatter);
            scene.scatter_renderer_sample_count = 1;
            scene.scatter_concordia = true;
        }
        // Submit all resident clipmap geometry directly. No temporal culling
        // decisions or extra indirect-selection buffers enter this depth copy.
        let mut direct = params.clone();
        direct.culling = "none".to_owned();
        let result = scene.encode_forward_pass(
            encoder,
            &direct,
            direct.decoded(),
            heights,
            materials,
            &self.terrain_uniform,
            ibl,
            media_environment_radiance,
            media_environment,
            media_environment_intensity,
            curve,
            targets,
            shadow,
            material_vt_ready,
            false,
            sun_visibility,
            time_seconds,
            timing,
            &mut None,
        );
        #[cfg(feature = "enable-gpu-instancing")]
        {
            std::mem::swap(&mut scene.scatter_renderer, &mut self.scatter);
            scene.scatter_renderer_sample_count = scatter_cache.0;
            scene.scatter_concordia = scatter_cache.1;
        }
        let frame = targets
            .concordia
            .as_mut()
            .ok_or_else(|| anyhow!("CONCORDIA frame state is unavailable"))?;
        frame.current_uniforms = current;
        std::mem::swap(&mut frame.frame_bind_group, &mut self.frame_bind_group);
        result?;
        encoder.copy_texture_to_texture(
            frame.gbuffer().depth_texture.as_image_copy(),
            self.linear.as_image_copy(),
            wgpu::Extent3d {
                width: params.size_px.0,
                height: params.size_px.1,
                depth_or_array_layers: 1,
            },
        );
        encoder.copy_texture_to_texture(
            frame._surface_kind.as_image_copy(),
            self.surface_kind.as_image_copy(),
            wgpu::Extent3d {
                width: params.size_px.0,
                height: params.size_px.1,
                depth_or_array_layers: 1,
            },
        );
        Ok(1)
    }
}

pub(super) fn clear_shared_geometry(
    encoder: &mut wgpu::CommandEncoder,
    targets: &RenderTargets,
) -> Result<()> {
    let frame = targets
        .concordia
        .as_ref()
        .ok_or_else(|| anyhow!("CONCORDIA frame state is unavailable"))?;
    let mut attachments = frame.gbuffer().geometry_attachments(
        &targets.internal_view,
        wgpu::LoadOp::Load,
        false,
        &frame.surface_kind_view,
    );
    attachments[0] = None;
    let _clear = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("concordia.gbuffer.clear"),
        color_attachments: &attachments,
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    });
    Ok(())
}
