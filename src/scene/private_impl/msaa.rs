impl Scene {
    pub(super) fn rebuild_msaa_state(&mut self) -> Result<(), String> {
        let g = crate::core::gpu::ctx();
        let depth_format = if self.sample_count > 1 {
            Some(wgpu::TextureFormat::Depth32Float)
        } else {
            None
        };

        let (color, color_view) = create_color_texture(&g.device, self.width, self.height);
        let (normal, normal_view) = create_normal_texture(&g.device, self.width, self.height);
        let (msaa_color, msaa_view) =
            create_msaa_targets(&g.device, self.width, self.height, self.sample_count);
        let (msaa_normal, msaa_normal_view) =
            create_msaa_normal_targets(&g.device, self.width, self.height, self.sample_count);
        let (depth, depth_view) =
            create_depth_target(&g.device, self.width, self.height, self.sample_count);

        self.depth = depth;
        self.depth_view = depth_view;

        self.color = color;
        self.color_view = color_view;
        self.normal = normal;
        self.normal_view = normal_view;
        self.msaa_color = msaa_color;
        self.msaa_view = msaa_view;
        self.msaa_normal = msaa_normal;
        self.msaa_normal_view = msaa_normal_view;
        self.ssao
            .resize(
                &g.device,
                &g.queue,
                self.width,
                self.height,
                &self.color,
                &self.normal,
            )
            .map_err(|e| e)?;

        let height_filterable = g
            .device
            .features()
            .contains(wgpu::Features::FLOAT32_FILTERABLE);
        self.tp = crate::terrain::pipeline::TerrainPipeline::create(
            &g.device,
            TEXTURE_FORMAT,
            NORMAL_FORMAT,
            self.sample_count,
            depth_format,
            height_filterable,
        );

        self.bg0_globals = self.tp.make_bg_globals(&g.device, &self.ubo);
        if let (Some(ref view), Some(ref sampler)) = (&self.height_view, &self.height_sampler) {
            self.bg1_height = self.tp.make_bg_height(&g.device, view, sampler);
        }
        self.bg2_lut = self
            .tp
            .make_bg_lut(&g.device, &self.colormap.view, &self.colormap.sampler);

        self.ssao
            .resize(
                &g.device,
                &g.queue,
                self.width,
                self.height,
                &self.color,
                &self.normal,
            )
            .map_err(|e| e)?;

        if let Some(ref mut renderer) = self.reflection_renderer {
            renderer.create_bind_group(&g.device, &self.tp.bgl_reflection);
        }

        // Recreate native overlay bind group with current overlay/height views
        if let Some(ref mut ov) = self.overlay_renderer {
            ov.recreate_bind_group(&g.device, None, self.height_view.as_ref(), None, None);
            ov.upload_uniforms(&g.queue);
        }

        if let Some(ref mut renderer) = self.dof_renderer {
            renderer.resize(&g.device, self.width, self.height);
            if let Some(ref depth_view) = self.depth_view {
                let color_storage_view = self.color.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("scene-color-storage"),
                    format: Some(wgpu::TextureFormat::Rgba8Unorm),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: 0,
                    mip_level_count: None,
                    base_array_layer: 0,
                    array_layer_count: None,
                });
                renderer.create_bind_group(
                    &g.device,
                    &self.color_view,
                    depth_view,
                    Some(&color_storage_view),
                );
            }
        }

        // D11: Recreate 3D text renderer to match current depth format
        let mut text3d =
            crate::core::text_mesh::TextMeshRenderer::new(&g.device, TEXTURE_FORMAT, depth_format);
        text3d.set_view_proj(self.scene.view, self.scene.proj);
        text3d.upload_uniforms(&g.queue);
        self.text3d_renderer = Some(text3d);

        Ok(())
    }

    // B8: Render clouds
}

